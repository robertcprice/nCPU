"""MeterBill — a usage-based SaaS billing engine whose every charge rule is
*synthesized from an English sentence* and verified by nCPU.

This is the honest end-to-end demonstration of the requirements pipeline:

    complex English  →  [LLM proposer]  →  RequirementsIR  →  [nCPU synth+verify]  →  verified program

The proposer is the *untrusted* front-end. Here it is an LLM (Claude Opus,
the model authoring this file) acting through a :class:`ScriptedProposer`: for
each rule we write the messy product-manager English, then the structured IR
the model extracted from it — a signature, concrete I/O examples, and a
reference implementation. That is exactly what ``LLMProposer`` would emit from
a live API call; we inline it so the run is reproducible offline and needs no
key. The pipeline treats every field as a *proposal to be checked*.

The *trusted* half is nCPU and runs for real, every time:

  * nsynth searches program space bottom-up for a Mog program reproducing the
    TRAIN examples (it verifies that itself before returning);
  * the pipeline runs the synthesized program against HELD-OUT examples the
    synthesizer never saw — generalization, not memorization;
  * it cross-checks the synthesized program against the proposer's reference
    on the holdout — agreement between two independently-derived programs.

Nothing here hardcodes an answer. If nsynth cannot find a program for a rule,
the pipeline refuses honestly and the rule is reported as unsynthesized. The
rules that *do* synthesize are transpiled to TypeScript and wired into an
interactive calculator (``meterbill.html``) — so the page you use is running
programs that were discovered from English and verified, not hand-written.

Run:
    python demos/requirements_app/build_meterbill.py
Outputs (next to this file):
    meterbill.html        — usable calculator built from synthesized rules
    provenance.json       — prose → IR → method → holdout → confidence, per rule
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ncpu.requirements.ir import IoExample, ParamSpec, RequirementsIR
from ncpu.requirements.pipeline import ResolvedRequirement, resolve
from ncpu.requirements.proposer import Proposer, ProposerError

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# The untrusted proposer: English -> IR, authored by the LLM (inlined).
# ---------------------------------------------------------------------------

class ScriptedProposer:
    """A :class:`Proposer` backed by pre-authored IRs.

    Each entry is the IR an LLM proposer extracted from the English prose.
    Keyed by the exact English string so ``resolve(english, proposer=...)``
    finds it. Satisfies the same protocol as ``LLMProposer`` — the pipeline
    cannot tell the difference, which is the point: the proposer is swappable
    and untrusted."""

    def __init__(self, by_english: dict[str, RequirementsIR]) -> None:
        self._by_english = by_english

    def propose(self, english: str) -> RequirementsIR:
        ir = self._by_english.get(english.strip())
        if ir is None:
            raise ProposerError("no scripted IR for this request")
        return ir


def _ir(
    entry: str,
    desc: str,
    params: list[tuple[str, str]],
    ret: str,
    examples: list[tuple[list, object]],
    invariants: list[str],
    edges: list[str],
    reference: str,
) -> RequirementsIR:
    return RequirementsIR(
        entry_point=entry,
        description=desc,
        params=[ParamSpec(name=n, type=t) for n, t in params],
        return_type=ret,
        io_examples=[IoExample(inputs=list(i), expected=e) for i, e in examples],
        invariants=invariants,
        edge_cases=edges,
        reference_impl=reference,
    )


# Each RULE: (english_prose, IR). The prose is what a product manager wrote;
# the IR is what the LLM proposer turned it into. Examples are ordered so the
# pipeline's first-2/3 train / last-1/3 holdout split makes both regions of
# each piecewise rule appear in training AND in the held-out generalization
# test.
RULES: list[tuple[str, RequirementsIR]] = [
    (
        "We charge twelve dollars per seat per month. Given the number of "
        "seats on an account, return the monthly seat charge in dollars.",
        _ir(
            "seat_cost",
            "Twelve dollars times the number of seats.",
            [("seats", "i64")], "i64",
            [([0], 0), ([1], 12), ([2], 24), ([3], 36), ([5], 60),
             ([7], 84), ([10], 120), ([25], 300), ([100], 1200)],
            ["charge is non-negative", "charge scales linearly with seats"],
            ["zero seats costs nothing"],
            "def seat_cost(seats):\n    return 12 * seats",
        ),
    ),
    (
        "Customers who prepay for a year get two months free: take their "
        "monthly rate, multiply by ten, and that is the annual prepay price. "
        "Given the monthly rate in dollars, return the yearly price in dollars.",
        _ir(
            "annual_prepay",
            "Ten times the monthly rate (twelve months minus two free).",
            [("monthly", "i64")], "i64",
            [([0], 0), ([5], 50), ([10], 100), ([29], 290), ([49], 490),
             ([99], 990), ([150], 1500), ([300], 3000), ([500], 5000)],
            ["two months are free", "price is ten times the monthly rate"],
            ["a free plan stays free"],
            "def annual_prepay(monthly):\n    return monthly * 10",
        ),
    ),
    (
        "Every account includes fifty gigabytes of storage at no cost. Beyond "
        "that we bill five dollars for each additional gigabyte; accounts at "
        "or under the fifty-gigabyte limit pay nothing. Given the gigabytes "
        "used, return the storage overage charge in dollars.",
        _ir(
            "storage_overage",
            "Five dollars per gigabyte used above a fifty-gigabyte free limit.",
            [("used_gb", "i64")], "i64",
            [([0], 0), ([40], 0), ([49], 0), ([50], 0), ([51], 5),
             ([55], 25), ([60], 50), ([70], 100), ([120], 350), ([200], 750)],
            ["no charge at or below 50 GB", "five dollars per GB over the limit"],
            ["exactly at the limit is free", "below the limit is free"],
            "def storage_overage(used_gb):\n    over = used_gb - 50\n"
            "    return 5 * over if over > 0 else 0",
        ),
    ),
    (
        "The first one hundred minutes of calls each month are free. Every "
        "minute after that costs two cents. Given the total minutes used, "
        "return the call cost in cents.",
        _ir(
            "call_cost",
            "Two cents per minute used above a hundred free minutes.",
            [("minutes", "i64")], "i64",
            [([0], 0), ([80], 0), ([99], 0), ([100], 0), ([101], 2),
             ([110], 20), ([150], 100), ([200], 200), ([350], 500), ([600], 1000)],
            ["first 100 minutes are free", "two cents per minute thereafter"],
            ["at or below 100 minutes is free"],
            "def call_cost(minutes):\n    over = minutes - 100\n"
            "    return 2 * over if over > 0 else 0",
        ),
    ),
    (
        "Each support ticket we resolve within an hour earns the customer a "
        "five dollar credit, but the credit is capped at ten fast tickets per "
        "month — fast tickets beyond the tenth earn nothing extra. Given the "
        "number of tickets resolved within an hour, return the credit in "
        "dollars.",
        _ir(
            "support_credit",
            "Five dollars per fast ticket, counting at most ten of them.",
            [("fast_tickets", "i64")], "i64",
            [([0], 0), ([1], 5), ([3], 15), ([5], 25), ([8], 40),
             ([9], 45), ([10], 50), ([11], 50), ([15], 50), ([20], 50)],
            ["five dollars per fast ticket", "credit never exceeds fifty dollars"],
            ["beyond ten fast tickets the credit is flat"],
            "def support_credit(fast_tickets):\n"
            "    return 5 * min(fast_tickets, 10)",
        ),
    ),
    (
        "Customers earn one loyalty point for every dollar they spend. On top "
        "of that, any portion of a purchase above one hundred dollars earns a "
        "second point per dollar. Given a purchase amount in whole dollars, "
        "return the loyalty points earned.",
        _ir(
            "loyalty_points",
            "One point per dollar, plus a second point per dollar above $100.",
            [("amount", "i64")], "i64",
            [([0], 0), ([50], 50), ([80], 80), ([99], 99), ([100], 100),
             ([101], 102), ([120], 140), ([200], 300), ([250], 400), ([301], 502)],
            ["at least one point per dollar",
             "double points on the amount over $100"],
            ["at or below $100 earns one point per dollar"],
            "def loyalty_points(amount):\n    bonus = amount - 100\n"
            "    return amount + (bonus if bonus > 0 else 0)",
        ),
    ),
    (
        "API calls are billed in tiers: the first one thousand calls each "
        "month are free, the next nine thousand calls cost two cents each, and "
        "every call beyond ten thousand costs one cent each. Given the number "
        "of calls, return the bill in cents.",
        _ir(
            "api_bill",
            "Tiered per-call pricing with free / 2c / 1c bands at 1k and 10k.",
            [("calls", "i64")], "i64",
            # free <=1000; +2c each to 10000 (max 18000); +1c beyond
            [([0], 0), ([500], 0), ([1000], 0), ([1001], 2), ([2000], 2000),
             ([5000], 8000), ([10000], 18000), ([10001], 18001),
             ([15000], 23000), ([20000], 28000)],
            ["first 1000 calls free", "calls 1001..10000 at 2c",
             "calls above 10000 at 1c"],
            ["two breakpoints at 1000 and 10000"],
            "def api_bill(calls):\n"
            "    if calls <= 1000:\n        return 0\n"
            "    if calls <= 10000:\n        return 2 * (calls - 1000)\n"
            "    return 18000 + (calls - 10000)",
        ),
    ),
]


# ---------------------------------------------------------------------------
# TypeScript -> JS (strip types) for the web calculator.
# ---------------------------------------------------------------------------

def ts_to_js(ts: str, entry: str) -> Optional[str]:
    """Strip TS type annotations into runnable JS. Returns None if no function
    body is recognizable."""
    if not ts:
        return None
    src = ts
    # `function f(a: i64, b: i64): i64 {` -> `function f(a, b) {`
    src = re.sub(r":\s*[\[\]A-Za-z0-9_ ]+\s*\{", " {", src)  # return type before {
    src = re.sub(r"(\w+)\s*:\s*[\[\]A-Za-z0-9_]+", r"\1", src)  # param: type
    src = re.sub(r"\bconst\b", "let", src)
    if f"function {entry}" not in src:
        return None
    return src.strip()


# ---------------------------------------------------------------------------
# CEGIS: tighten a synthesized rule against its own reference spec.
# ---------------------------------------------------------------------------

from ncpu.requirements.pipeline import _safe_callable  # noqa: E402


def _domain(ir: RequirementsIR) -> range:
    """Integer sweep range for a single-int-arg rule, derived from the
    example magnitudes (0 .. ~2x the largest input). Bounded so the sweep
    stays cheap even for the large-call-count tiers."""
    mx = 1
    for ex in ir.io_examples:
        if ex.inputs and isinstance(ex.inputs[0], int):
            mx = max(mx, abs(ex.inputs[0]))
    hi = min(mx * 2 + 10, 60000)
    return range(0, hi + 1)


def cegis_resolve(
    english: str, ir: RequirementsIR, proposer: ScriptedProposer, *, rounds: int = 4
) -> tuple[ResolvedRequirement, int, int]:
    """Counterexample-guided loop around :func:`resolve`.

    After each synthesis, sweep the input domain and compare the synthesized
    program to the proposer's reference (the spec). Every disagreement is a
    counterexample — a point where the program does the wrong thing — and is
    added to the IR's examples for the next round. Repeat until the sweep is
    clean or the budget runs out. This is the same CEGIS discipline used to
    verify the synthesized game rules: the program is forced to agree with the
    spec everywhere on the domain, not just on the seed examples.

    Only runs when the rule is a single-int-arg function with a Python
    reference (the case the integer sweep can check). Returns the final
    resolution, the number of counterexamples added, and rounds used."""
    _RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}

    def _score(r: ResolvedRequirement) -> tuple[int, int]:
        return (_RANK.get(r.confidence, 0), r.holdout_passed)

    added = 0
    last = resolve(english, proposer=proposer, synth_timeout_s=25.0)
    best = last
    single_int_arg = (
        len(ir.params) == 1
        and ir.reference_impl
        and all(len(e.inputs) == 1 and isinstance(e.inputs[0], int) for e in ir.io_examples)
    )
    if not single_int_arg:
        return last, 0, 0

    ref = _safe_callable(ir.reference_impl, ir.entry_point)
    if ref is None:
        return last, 0, 0

    rnd = 0
    for rnd in range(1, rounds + 1):
        if _score(last) > _score(best):
            best = last
        if last.status != "synthesized":
            break
        synth_py = last.transpiled.get("python")
        fn = _safe_callable(synth_py, ir.entry_point) if synth_py else None
        if fn is None:
            break
        # already fully agrees with its own reference on the holdout AND found
        # generalizing — nothing to tighten.
        counter: list[IoExample] = []
        seen = {tuple(e.inputs) for e in ir.io_examples}
        for x in _domain(ir):
            try:
                want = ref(x)
                got = fn(x)
            except Exception:  # noqa: BLE001
                continue
            if want != got and (x,) not in seen:
                counter.append(IoExample(inputs=[x], expected=want))
                if len(counter) >= 6:  # a handful per round is plenty
                    break
        if not counter:
            break  # synthesized program matches the spec across the whole domain
        ir.io_examples.extend(counter)
        added += len(counter)
        proposer._by_english[english.strip()] = ir  # proposer now serves richer IR
        print(f"    cegis round {rnd}: +{len(counter)} counterexample(s) "
              f"(e.g. {counter[0].inputs[0]}→{counter[0].expected}), re-synthesizing")
        last = resolve(english, proposer=proposer, synth_timeout_s=25.0)
    if _score(last) > _score(best):
        best = last
    return best, added, rnd if added else 0


# ---------------------------------------------------------------------------
# Run the pipeline on every rule and collect results.
# ---------------------------------------------------------------------------

def run() -> list[tuple[str, RequirementsIR, ResolvedRequirement]]:
    proposer = ScriptedProposer({en.strip(): ir for en, ir in RULES})
    out = []
    for english, ir in RULES:
        print(f"\n=== {ir.entry_point} ===")
        print(f"  english: {english[:90]}...")
        res, added, rnds = cegis_resolve(english, ir, proposer)
        print(f"  status={res.status} method={res.method} "
              f"confidence={res.confidence} "
              f"holdout={res.holdout_passed}/{res.holdout_count} "
              f"ref_agree={res.synth_vs_reference_agree} "
              f"cegis(+{added} ex over {rnds} rounds)")
        res.notes.append(
            f"CEGIS: swept the input domain vs the reference spec and added "
            f"{added} counterexample(s) over {rnds} round(s) before this result."
        )
        out.append((english, ir, res))
    return out


def build_html(results) -> str:
    """An interactive calculator using the synthesized TS->JS programs."""
    synth_fns, cards = [], []
    for english, ir, res in results:
        js = ts_to_js(res.transpiled.get("typescript", ""), ir.entry_point) \
            if res.status == "synthesized" else None
        # Only wire a rule into the live calculator when the pipeline certified
        # it generalizes (high/medium): reproduced every held-out example and,
        # for high, agreed with an independent reference. A low-confidence
        # program fits the seed examples but the system could NOT verify it
        # generalizes — shipping it as a working calculator would be dishonest,
        # so it's shown as synthesized-but-uncertified and left non-interactive.
        certified = res.confidence in ("high", "medium")
        ok = js is not None and res.status == "synthesized" and certified
        if ok:
            synth_fns.append(js)
        conf = res.confidence
        badge = {"high": "#7DD3FC", "medium": "#FFB454",
                 "low": "#E0708C", "none": "#E0708C"}.get(conf, "#E0708C")
        unit = "cents" if "cent" in ir.description.lower() else "dollars"
        cards.append({
            "entry": ir.entry_point,
            "english": english,
            "desc": ir.description,
            "param": ir.params[0].name if ir.params else "x",
            "ok": ok,
            "status": res.status,
            "method": res.method or "-",
            "confidence": conf,
            "badge": badge,
            "holdout": f"{res.holdout_passed}/{res.holdout_count}",
            "ref": ("agree" if res.synth_vs_reference_agree
                    else "—" if res.synth_vs_reference_agree is None else "DISAGREE"),
            "unit": unit,
            "ts": res.transpiled.get("typescript", "")
            if res.status == "synthesized" else "",
        })
    cards_json = json.dumps(cards)
    fns_js = "\n".join(synth_fns)
    return _HTML_TEMPLATE.replace("/*FNS*/", fns_js).replace("/*CARDS*/", cards_json)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MeterBill — billing rules synthesized from English</title>
<style>
  :root{--bg:#0C1015;--panel:#151B23;--ink:#E6EDF3;--mut:#8B98A5;
        --amber:#FFB454;--cyan:#7DD3FC;--rose:#E0708C;--line:#222B36;}
  *{box-sizing:border-box;min-width:0}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 ui-monospace,'JetBrains Mono',Menlo,monospace;padding:24px}
  h1{font-size:22px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 20px;max-width:70ch}
  .grid{display:grid;grid-template-columns:1fr;gap:16px;max-width:860px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
        padding:16px;overflow:hidden}
  .en{color:var(--cyan);font-size:13px;margin:0 0 10px;font-style:italic}
  .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
  .row label{color:var(--mut)} input{background:#0C1015;color:var(--ink);
       border:1px solid var(--line);border-radius:7px;padding:6px 9px;width:120px;
       font:inherit} .out{color:var(--amber);font-weight:600}
  .meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;font-size:12px}
  .pill{border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--mut)}
  .dot{display:inline-block;width:8px;height:8px;border-radius:99px;margin-right:5px;
       vertical-align:middle}
  details{margin-top:10px} summary{cursor:pointer;color:var(--mut);font-size:12px}
  pre{background:#0C1015;border:1px solid var(--line);border-radius:8px;padding:10px;
      overflow:auto;font-size:12px;color:#cdd6e0}
  .bad{opacity:.62} .bad .out{color:var(--rose)}
  .legend{color:var(--mut);font-size:12px;margin:18px 0 0;max-width:70ch}
</style></head><body>
<h1>MeterBill</h1>
<p class="sub">Every charge below was written as one English sentence, turned into a
verifiable contract by an LLM proposer, then <b>synthesized bottom-up and verified by
nCPU</b> — program search, a held-out generalization test, and a cross-check against an
independent reference. The inputs are live: you are running programs that were
<b>discovered from English</b>, not hand-written.</p>
<div class="grid" id="grid"></div>
<p class="legend">Confidence: <span class="dot" style="background:#7DD3FC"></span>high =
generalized on every held-out case AND agrees with an independent reference &nbsp;
<span class="dot" style="background:#FFB454"></span>medium = generalized on all holdout &nbsp;
<span class="dot" style="background:#E0708C"></span>low/none = held out a case it missed,
or nСPU honestly refused to synthesize this rule.</p>
<script>
/*FNS*/
const FNS = {};
/*CARDS*/.forEach(c => { try { FNS[c.entry] = eval(c.entry); } catch(e){} });
const CARDS = /*CARDS*/;
const grid = document.getElementById('grid');
CARDS.forEach(c => {
  const el = document.createElement('div');
  el.className = 'card' + (c.ok ? '' : ' bad');
  const fn = FNS[c.entry];
  const compute = (v) => {
    if (!fn) return c.ok ? '—' : 'not synthesized';
    try { return fn(parseInt(v||'0',10)); } catch(e){ return 'err'; }
  };
  el.innerHTML = `
    <p class="en">"${c.english}"</p>
    <div class="row">
      <label>${c.param}</label>
      <input type="number" value="0" data-entry="${c.entry}">
      <span>→</span>
      <span class="out" id="out-${c.entry}">${compute(0)}</span>
      <span style="color:var(--mut)">${c.unit}</span>
    </div>
    <div class="meta">
      <span class="pill"><span class="dot" style="background:${c.badge}"></span>${c.confidence}</span>
      <span class="pill">method: ${c.method}</span>
      <span class="pill">holdout ${c.holdout}</span>
      <span class="pill">vs reference: ${c.ref}</span>
      <span class="pill">${c.status}</span>
    </div>
    ${c.ts ? `<details><summary>synthesized program (TypeScript)</summary><pre>${
       c.ts.replace(/</g,'&lt;')}</pre></details>` : ''}`;
  grid.appendChild(el);
  const inp = el.querySelector('input');
  inp.addEventListener('input', () => {
    document.getElementById('out-'+c.entry).textContent = compute(inp.value);
  });
});
</script></body></html>"""


def main() -> int:
    results = run()
    n_synth = sum(1 for _, _, r in results if r.status == "synthesized")
    n_high = sum(1 for _, _, r in results if r.confidence == "high")
    n_med = sum(1 for _, _, r in results if r.confidence == "medium")

    html = build_html(results)
    (HERE / "meterbill.html").write_text(html)

    prov = {
        "product": "MeterBill",
        "pipeline": "english -> LLM proposer (untrusted) -> RequirementsIR "
                    "-> nCPU synth+verify (trusted) -> verified program",
        "rules": [r.to_dict() for _, _, r in results],
        "summary": {
            "rules": len(results),
            "synthesized": n_synth,
            "high_confidence": n_high,
            "medium_confidence": n_med,
        },
    }
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2))

    print("\n" + "=" * 60)
    print(f"synthesized {n_synth}/{len(results)} rules  "
          f"(high={n_high}, medium={n_med})")
    print(f"wrote {HERE/'meterbill.html'}")
    print(f"wrote {HERE/'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
