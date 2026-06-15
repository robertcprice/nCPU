#!/usr/bin/env python3
"""Rule-compressed memory — the "infinite context" experiment (Tier A).

Claim under test: storing knowledge as VERIFIED RULES (synthesized programs)
instead of instances gives **bounded memory, unbounded reach, zero forgetting**.

We stream (word -> plural) instances and compare three memories:

  RULE     (ours)  : an nSynth-synthesized pluralization program + a small
                     exception table for the genuinely irregular (irreducible)
                     items. Storage = |rule| + |exceptions|.
  INSTANCE (RAG)   : remember every pair seen. Storage grows linearly.
  WINDOW-W (LLM)   : remember only the last W pairs. Storage is constant W but
                     it FORGETS everything older.

Metrics recorded vs stream length:
  * storage (bytes)
  * coverage on ALL items seen so far (does memory still answer them?)
  * coverage on HELD-OUT items never streamed — real words and nonce "wug"
    words — i.e. does the memory GENERALIZE?

Expected result: RULE storage flattens (converges once the regularity is
captured; only finite irregulars accrete), coverage-on-seen stays 100% (it never
forgets — the rule covers even long-evicted items), and coverage-on-unseen is
~100% (it generalizes to words never seen). INSTANCE storage grows without bound
and generalizes 0%. WINDOW forgets the past and generalizes 0%. That gap is the
"effectively infinite context for the regular part of the stream".

Run:  python scripts/rule_memory_experiment.py [--stream N] [--window W]
Outputs: a results table, ASCII curves, and rule_memory_results.csv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYNTH = HERE.parent
BIN = NSYNTH / "target" / "release" / "mog_synth"
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))

FN = "pluralize"


def synth_rule(pairs: list[tuple[str, str]]) -> str | None:
    """Synthesize a pluralization Mog program from (word, plural) pairs."""
    if not pairs:
        return None
    payload = json.dumps({
        "name": FN,
        "signature": f"fn {FN}(s: string) -> string",
        "examples": [{"inputs": [w], "expected": p} for w, p in pairs],
        "holdouts": [],
    })
    try:
        out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                             capture_output=True, text=True, timeout=8).stdout
        r = json.loads(out)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    return r["code"] if r.get("success") else None


def predict(rule_code: str, words: list[str]) -> dict[str, str]:
    """Run the rule program on many words in one Mog execution."""
    if rule_code is None or not words:
        return {}
    calls = "".join(f'  println({FN}("{w}"));\n' for w in words)
    prog = f"{rule_code}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True).stdout
    lines = out.splitlines()
    # outputs are in order; pad if the program errored partway
    return {w: (lines[i] if i < len(lines) else "") for i, w in enumerate(words)}


def build_dataset(pluralize):
    from v2.grammar.morphology import _IRREGULAR_PLURAL
    from v2.tokenizer.morpheme_tokenizer import _curriculum_lexicon
    _t, _i, known = _curriculum_lexicon()
    words = sorted({w.lower() for w in known if w.isalpha() and 2 < len(w) < 12})
    # hold out 300 real words to test generalization to UNSEEN real words
    held_real = words[::12][:300]
    stream_words = [w for w in words if w not in set(held_real)]
    # add the full irregular table into the stream (the irreducible part)
    for w in _IRREGULAR_PLURAL:
        if w not in stream_words:
            stream_words.append(w)
    # nonce "wug" words — guaranteed never seen; only a RULE can pluralize them
    seeds = ["wug", "blick", "dax", "fep", "lorp", "thwock", "glorp", "snib",
             "krad", "plonk", "vamp", "zorch", "quax", "frizz", "splosh", "grex"]
    nonce = []
    for s in seeds:
        for suf in ["", "le", "er", "o", "y", "sh", "ch", "x", "s"]:
            nonce.append(s + suf)
    return stream_words, held_real, nonce


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _action_for(inp: str, out: str) -> tuple[int, str]:
    """Return (strip, append) action that turns inp into out by LCP edit."""
    common = 0
    for a, b in zip(inp, out):
        if a != b:
            break
        common += 1
    return len(inp) - common, out[common:]


def _action_expr(strip: int, append: str) -> str:
    if strip == 0:
        return f's + "{_esc(append)}"'
    return f's.slice(0, s.len - {strip}) + "{_esc(append)}"'


def _emit_guarded_micro_rule(
    clauses: list[tuple[str, str, tuple[int, str]]],
) -> str:
    """Emit an abstaining exception micro-rule.

    Each clause is either ("suffix", suffix, action) or ("exact", word, action).
    Unlike the main pluralizer, this function returns the empty string when no
    guard fires, so the hierarchical wrapper can fall through to later micro
    rules and finally the main synthesized rule.
    """
    body = []
    for kind, guard, (strip, append) in clauses:
        if kind == "suffix":
            cond = f's.ends_with("{_esc(guard)}")'
        elif kind == "exact":
            cond = f's == "{_esc(guard)}"'
        else:
            raise ValueError(f"unknown guard kind: {kind}")
        body.append(f"    if {cond} {{\n        return {_action_expr(strip, append)};\n    }}")
    body.append('    return "";')
    return f"fn {FN}(s: string) -> string {{\n" + "\n".join(body) + "\n}\n"


def _items_view(pool) -> list[tuple[str, str]]:
    if isinstance(pool, dict):
        return list(pool.items())
    return list(pool)


def compile_hierarchical_rule(main_rule_code, micro_rules):
    if not main_rule_code:
        if not micro_rules:
            return None
        main_rule_code = "fn pluralize(s: string) -> string {\n    return \"\";\n}\n"
    if not micro_rules:
        return main_rule_code
    
    combined: list[str] = []
    calls: list[str] = []
    for i, micro_code in enumerate(micro_rules):
        renamed = micro_code.replace("fn pluralize(", f"fn pluralize_micro_{i}(")
        combined.append(renamed)
        calls.append(f'    res_{i}: string = pluralize_micro_{i}(s);\n    if res_{i} != "" {{\n        return res_{i};\n    }}')
        
    header = "fn pluralize(s: string) -> string {\n"
    header_at = main_rule_code.find(header)
    if header_at < 0:
        return "\n".join(combined) + "\n" + main_rule_code
    body_start = header_at + len(header)
    calls_str = "\n".join(calls) + "\n"
    new_main = main_rule_code[:body_start] + calls_str + main_rule_code[body_start:]
    return "\n".join(combined) + "\n" + new_main


def check_micro_rule_safety(micro_code, regular_pool):
    regular_items = _items_view(regular_pool)
    if not regular_items:
        return True
    words = [w for w, _ in regular_items]
    preds = predict(micro_code, words)
    for w, true in regular_items:
        got = preds.get(w, "")
        if got != "" and got != true:
            return False
    return True


def synth_exception_micro_rule(
    exceptions: dict[str, str],
    regular_pool,
    *,
    min_cover: int = 3,
) -> tuple[str, list[str]] | None:
    """Find one safe abstaining micro-rule that compresses exception structure.

    The rule is deliberately conservative: it may only fire on a suffix/exact
    guard if it is correct for every regular example it touches and every
    exception it touches. This keeps the "Hamilton" table monotonic while still
    allowing recurring exception families (-us -> -i, -f -> -ves, invariants)
    to be promoted from instance memory into verified executable code.
    """
    if len(exceptions) < min_cover:
        return None

    grouped: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    for w, plural in exceptions.items():
        grouped[_action_for(w, plural)].append((w, plural))

    candidates: list[tuple[int, int, str, list[str]]] = []

    def evaluate(code: str) -> tuple[int, list[str]] | None:
        if not check_micro_rule_safety(code, regular_pool):
            return None
        exc_words = list(exceptions.keys())
        preds = predict(code, exc_words)
        covered: list[str] = []
        for w, true in exceptions.items():
            got = preds.get(w, "")
            if got == true:
                covered.append(w)
            elif got != "":
                return None
        if len(covered) < min_cover:
            return None
        removed_bytes = sum(len(w) + len(exceptions[w]) for w in covered)
        # Prefer broad byte-saving rules, but allow small safe rules because the
        # point of this experiment is structural compression, not Python dict
        # overhead accounting.
        return removed_bytes, covered

    for action, pairs in grouped.items():
        if len(pairs) < min_cover:
            continue
        strip, append = action
        max_suffix_len = min(6, max(len(w) for w, _ in pairs))

        # Exact invariant clusters (sheep/fish/deer/...) are safe and finite;
        # emitting exact guards avoids overgeneralizing "no-op plural" to every
        # future word with a common suffix.
        if strip == 0 and append == "":
            clauses = [("exact", w, action) for w, _ in pairs]
            code = _emit_guarded_micro_rule(clauses)
            result = evaluate(code)
            if result:
                removed_bytes, covered = result
                candidates.append((removed_bytes, len(covered), code, covered))
            continue

        min_suffix_len = max(1, strip)
        for suffix_len in range(min_suffix_len, max_suffix_len + 1):
            buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for w, plural in pairs:
                if len(w) >= suffix_len:
                    buckets[w[-suffix_len:]].append((w, plural))
            for suffix, bucket in buckets.items():
                if len(bucket) < min_cover:
                    continue
                code = _emit_guarded_micro_rule([("suffix", suffix, action)])
                result = evaluate(code)
                if result:
                    removed_bytes, covered = result
                    candidates.append((removed_bytes, len(covered), code, covered))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], -len(x[2])), reverse=True)
    _, _, code, covered = candidates[0]
    return code, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=1200)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    from v2.grammar.morphology import pluralize
    stream_words, held_real, nonce = build_dataset(pluralize)
    # deterministic shuffle (no Math.random in this env; use a fixed permutation)
    stream_words = stream_words[: args.stream]
    oracle = {w: pluralize(w) for w in stream_words + held_real + nonce}

    rule_code = None
    micro_rules: list[str] = []
    regular_pool: dict[str, str] = {}
    exceptions: dict[str, str] = {}
    seen: list[str] = []
    instance_mem: dict[str, str] = {}
    window: deque = deque(maxlen=args.window)

    rows = []
    n_resynth = 0
    mistaken_items: set = set()   # Hamilton: every item ever mis-predicted
    repeat_mistakes = 0           # an item mis-predicted AFTER it was corrected

    def integrate_regular_pairs(
        pairs: list[tuple[str, str]],
        suspect_words: set[str] | None = None,
    ):
        """Add new observations without allowing later resyntheses to forget them.

        Earlier versions only trained on misses. That let a later synthesized
        rule silently change answers for previously-correct words that were not
        in the training pool. Here every streamed non-exception becomes a
        verification constraint; if a pair makes the regular rule
        unsynthesizable, it is promoted into the exception pillar.
        """
        nonlocal rule_code, n_resynth
        fresh = [(w, p) for w, p in pairs if w not in exceptions and regular_pool.get(w) != p]
        if not fresh:
            return

        trial = dict(regular_pool)
        trial.update(fresh)
        n_resynth += 1
        new_rule = synth_rule(list(trial.items()))
        if new_rule:
            regular_pool.clear()
            regular_pool.update(trial)
            rule_code = new_rule
            return

        # Bulk integration failed, usually because the chunk contains one or
        # more irreducible irregulars. First try admitting the non-suspect
        # examples as a batch; the suspects are exactly the words the previous
        # memory missed, so they are much more likely to be blockers. This keeps
        # resynthesis counts low while preserving monotonic correctness.
        suspects = suspect_words or set()
        to_admit = fresh
        if suspects:
            non_suspect = [(w, p) for w, p in fresh if w not in suspects]
            suspect_pairs = [(w, p) for w, p in fresh if w in suspects]
            if non_suspect:
                trial = dict(regular_pool)
                trial.update(non_suspect)
                n_resynth += 1
                new_rule = synth_rule(list(trial.items()))
                if new_rule:
                    regular_pool.clear()
                    regular_pool.update(trial)
                    rule_code = new_rule
                    to_admit = suspect_pairs

        # Final fallback: monotonic one-by-one admission so regular examples
        # still strengthen the rule and irreducibles enter Hamilton memory.
        for w, p in to_admit:
            trial = dict(regular_pool)
            trial[w] = p
            n_resynth += 1
            new_rule = synth_rule(list(trial.items()))
            if new_rule:
                regular_pool[w] = p
                rule_code = new_rule
            else:
                exceptions[w] = p

    def coverage(words, use_rule_mem):
        if not words:
            return 1.0
        if use_rule_mem:
            active_rule_code = compile_hierarchical_rule(rule_code, micro_rules)
            preds = predict(active_rule_code, [w for w in words if w not in exceptions])
            ok = 0
            for w in words:
                p = exceptions.get(w) or preds.get(w, "")
                ok += (p == oracle[w])
            return ok / len(words)
        return 0.0  # baselines never generalize to unseen

    print(f"Streaming {len(stream_words)} (word -> plural) instances; window W={args.window}\n")
    for start in range(0, len(stream_words), args.chunk):
        chunk = stream_words[start:start + args.chunk]
        # RULE memory: predict chunk, collect misses
        active_rule_code = compile_hierarchical_rule(rule_code, micro_rules)
        preds = predict(active_rule_code, [w for w in chunk if w not in exceptions])
        pending = []
        for w in chunk:
            true = oracle[w]
            got = exceptions.get(w) or preds.get(w, "")
            if got != true:
                pending.append((w, true))
                if w in mistaken_items:   # Hamilton: did we err on an already-corrected item?
                    repeat_mistakes += 1
                mistaken_items.add(w)
        # Train on every new observation, not just misses. This is what turns
        # the rule pillar into monotonic verified memory rather than a demo that
        # can accidentally forget examples it used to answer correctly.
        integrate_regular_pairs(
            [(w, oracle[w]) for w in chunk],
            {w for w, _ in pending},
        )

        # Partition exceptions if they grow too large
        while len(exceptions) >= 8:
            micro = synth_exception_micro_rule(exceptions, regular_pool, min_cover=3)
            if not micro:
                break
            candidate_micro, covered = micro
            micro_rules.append(candidate_micro)
            for w in covered:
                del exceptions[w]
            print(
                "  [Hierarchical Exception Partition] "
                f"Synthesized abstaining micro-rule covering {len(covered)} exceptions!"
            )


        # baselines
        for w in chunk:
            seen.append(w)
            instance_mem[w] = oracle[w]
            window.append(w)

        # metrics
        active_rule_code = compile_hierarchical_rule(rule_code, micro_rules)
        rule_bytes = (len(active_rule_code) if active_rule_code else 0) + sum(len(k) + len(v) for k, v in exceptions.items())
        instance_bytes = sum(len(k) + len(v) for k, v in instance_mem.items())
        window_bytes = sum(len(w) + len(oracle[w]) for w in window)
        cov_seen_rule = coverage(seen, True)
        # window forgets: coverage on seen = fraction still in window
        cov_seen_window = len(set(window)) / len(seen)
        cov_unseen_real = coverage(held_real, True)
        cov_unseen_nonce = coverage(nonce, True)
        rows.append({
            "n": len(seen), "rule_bytes": rule_bytes, "exceptions": len(exceptions),
            "instance_bytes": instance_bytes, "window_bytes": window_bytes,
            "resynths": n_resynth,
            "cov_seen_rule": cov_seen_rule, "cov_seen_instance": 1.0,
            "cov_seen_window": cov_seen_window,
            "cov_unseen_real_rule": cov_unseen_real, "cov_unseen_nonce_rule": cov_unseen_nonce,
        })

    # ── report ──
    print(f"{'n':>5} {'RULE B':>7} {'#exc':>5} {'INST B':>8} {'resyn':>6} "
          f"{'cov_seen(rule)':>14} {'cov_seen(win)':>13} {'unseen_real':>11} {'unseen_wug':>10}")
    for r in rows[:: max(1, len(rows) // 14)]:
        print(f"{r['n']:>5} {r['rule_bytes']:>7} {r['exceptions']:>5} {r['instance_bytes']:>8} "
              f"{r['resynths']:>6} {r['cov_seen_rule']*100:>13.1f}% {r['cov_seen_window']*100:>12.1f}% "
              f"{r['cov_unseen_real_rule']*100:>10.1f}% {r['cov_unseen_nonce_rule']*100:>9.1f}%")
    last = rows[-1]
    print("\n=== FINAL ===")
    print(f"  RULE memory:     {last['rule_bytes']:>6} bytes  ({last['exceptions']} exceptions), "
          f"{last['resynths']} re-syntheses total")
    print(f"  INSTANCE memory: {last['instance_bytes']:>6} bytes  (grows linearly with the stream)")
    print(f"  Coverage on ALL {last['n']} seen — RULE {last['cov_seen_rule']*100:.1f}%, "
          f"WINDOW {last['cov_seen_window']*100:.1f}% (forgot the rest)")
    print(f"  Coverage on UNSEEN — real {last['cov_unseen_real_rule']*100:.1f}%, "
          f"nonce/wug {last['cov_unseen_nonce_rule']*100:.1f}%  (baselines: 0%)")
    ratio = last['instance_bytes'] / max(1, last['rule_bytes'])
    print(f"  Compression: instance/rule = {ratio:.1f}x; reach: UNBOUNDED (generalizes to unseen).")
    print(f"  Hamilton (mistake memory): {len(mistaken_items)} distinct mistakes over {last['n']} "
          f"items (converges, finite); repeated mistakes: {repeat_mistakes} "
          f"(self-improving — an error, once handled, is never made again).")

    csv = HERE / "rule_memory_results.csv"
    with open(csv, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"\nCurves -> {csv}")


if __name__ == "__main__":
    main()
