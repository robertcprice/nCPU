# MeterBill — billing rules synthesized from English, verified by nCPU

MeterBill is a usage-based SaaS billing engine in which **every charge rule is
synthesized from a plain-English sentence and verified before it is allowed to
run**. It is the honest end-to-end demonstration of the nCPU *requirements
pipeline*:

```
complex English  →  [LLM proposer]  →  RequirementsIR  →  [nCPU synth + verify]  →  verified program
     (untrusted)                                              (trusted, runs for real)
```

The page you interact with (`meterbill.html`) is running programs that were
**discovered from English and checked for generalization** — not hand-written,
not pattern-matched from a template library.

---

## TL;DR

```bash
# from the repo root
PYTHONPATH=. python3 demos/requirements_app/build_meterbill.py
```

Outputs, written next to the script:

| File | What it is |
|------|------------|
| `meterbill.html` | Interactive calculator built only from the rules nCPU **certified** generalize |
| `provenance.json` | Full audit trail per rule: prose → IR → method → holdout → reference cross-check → confidence |

Current result: **7/7 rules synthesized and certified high-confidence** — including
the hard two-breakpoint tiered rule, which only certified after the pipeline
caught and rejected two overfits that fooled a sparse holdout but were wrong
across the full domain (see [the honesty showcase](#the-honesty-showcase-api_bill)).

---

## Why this is not a toy

Three properties separate this from a "LLM writes code" demo:

1. **The proposer is untrusted.** The LLM only *proposes* a structured
   `RequirementsIR` (a signature, I/O examples, a reference implementation). The
   pipeline treats every field as a claim to be checked, never as an answer.

2. **nCPU synthesizes bottom-up, then verifies generalization.** `nsynth`
   searches program space for a Mog program that reproduces the **training**
   examples, then the pipeline runs that program against **held-out** examples
   the synthesizer never saw, and **cross-checks** it against the proposer's
   reference implementation across the input domain. Memorization cannot pass;
   only generalization does.

3. **It refuses to certify what it can't prove.** A program is only wired into
   the live calculator once it survives a full-domain sweep against the
   reference; anything still wrong anywhere is downgraded to `low`-confidence and
   shown uncertified, never executed. The system never ships a billing rule it
   could not prove — and it caught two overfits doing exactly that on the hardest
   rule (see [`api_bill`](#the-honesty-showcase-api_bill) below).

Nothing in this demo hardcodes a numeric answer. The synthesizer's constant
vocabulary is *mined from each rule's own examples* (see
[the synthesizer notes](#how-the-rules-get-synthesized)).

---

## The seven rules

Each rule is messy product-manager English. The LLM proposer turns it into an
IR; nCPU does the rest. The synthesized program shown is what nsynth actually
returned (verify with a fresh run).

| Rule | English (abridged) | Synthesized program | Confidence |
|------|--------------------|--------------------|------------|
| `seat_cost` | "$12 per seat per month" | `12 * seats` | **high** |
| `annual_prepay` | "prepay a year → 2 months free (rate × 10)" | `10 * rate` | **high** |
| `storage_overage` | "first 50 GB free, then $5/GB" | `if 50 < x: (x - 50) * 5 else 0`  ≡ `max(0, x-50) * 5` | **high** |
| `call_cost` | "first 100 minutes free, then 2¢/min" | `if 100 < x: (x - 100) * 2 else 0`  ≡ `max(0, x-100) * 2` (cents) | **high** |
| `support_credit` | "$5 credit per ticket, capped at 10 tickets" | `if 10 < x: 50 else 5 * x`  ≡ `min(x,10) * 5` | **high** |
| `loyalty_points` | "1 pt/$ + bonus on the portion above $100" | `if 100 < x: (50 - x) * -2 else x`  ≡ `x + max(0, x-100)` | **high** |
| `api_bill` | "tiered: first 1k free, next 9k @ 2¢, rest @ 1¢" | `if 10000 < x: x + 8000; if 1000 < x: 2*(x-1000); else 0` | **high** |

The piecewise rules (`storage_overage`, `call_cost`, `support_credit`,
`loyalty_points`, `api_bill`) are the interesting ones: their breakpoints (50,
100, 10, 1000, 10000) are **not** in any hardcoded list — nsynth mined them from
the examples and the boundary-continuity scorer locked onto the threshold where
the branches actually meet.

### `api_bill`: solved at the engine level

`api_bill` is genuinely hard — a **two-breakpoint** tiered schedule, three affine
pieces. The engine now recovers it **directly**, from examples alone, with no
reference and zero CEGIS rounds:

```
fn api_bill(x) { if x <= 1000 { 0 } if x <= 10000 { 2*x - 2000 } else { x + 8000 } }
```

This is the `search_piecewise_affine` solver (in `nsynth`): it detects the affine
segments in the data (where the slope changes) and places each breakpoint at the
exact integer where the adjoining pieces intersect — so the breakpoints **1000**
and **10000** come straight out of the examples, and the program is correct
across the whole domain (`0` mismatches over `0..40000`), not just on the
samples. It is exact by construction, so it generalizes.

This was not always so, and the history is the honesty point. Before the
piecewise solver, branch search fit `api_bill` with programs that **passed the
sparse holdout but were overfits** — an integer-division flooring trick
`(x / 1001) * 2000`, and a bounded-range modulo `x % 10001` that wraps past the
largest training input. A system that trusts a 3-point holdout would have shipped
one. This one did not: the CEGIS sweep compared against the reference across the
whole domain and a **downgrade-only gate refused to certify any program still
wrong anywhere** — so the worst case was an honest *refusal*, never a wrong
shipped rule. The piecewise solver then closed the gap by solving it *correctly*,
turning the refusal into a clean certification.

The refusal path is still there for anything the synthesizer cannot solve
exactly. But the goal was never to refuse well — it was to **get it right**, and
now the engine does (see [the measured generalization gain](#measured-does-the-engine-actually-generalize)).

---

## How a rule flows through the pipeline

```
RULES[i] = (english, IR)
        │
        ▼
ScriptedProposer.propose(english) ──► RequirementsIR     # untrusted front-end
        │                                                  (LLM-authored, inlined
        │                                                   so the run is offline +
        │                                                   reproducible, no API key)
        ▼
resolve(english, proposer)            # ncpu/requirements/pipeline.py
        │
        ├─ _split(io_examples)        # STRIDED holdout: every 3rd example is held
        │                             #   out, so train + holdout both span the full
        │                             #   input domain (see note below)
        ├─ nsynth synthesize(train)   # bottom-up program search; self-verifies on train
        ├─ run program on holdout     # generalization check
        ├─ cross-check vs reference   # two independently-derived programs must agree
        └─ grade confidence           # none | low | medium | high
        │
        ▼
cegis_resolve(...)                    # sweep domain vs reference, feed disagreements
        │                             #   back as new examples, re-synthesize, keep best
        ▼
ResolvedRequirement  ──►  build_html()  # only high/medium rules become live widgets
                     └──►  provenance.json
```

### Why the holdout split is *strided*, not a tail slice

A tail slice (`examples[:n_train]`, `examples[n_train:]`) would put a whole
region of the input domain — e.g. all the large-`x` examples of a piecewise rule
— exclusively into the holdout set. Training then never sees that region, so it
cannot pin the rule there, and even the **correct** program looks like it fails
to generalize. A stride (`i % 3 == 2`) keeps both train and holdout spanning the
full spread. This is implemented in `ncpu/requirements/pipeline.py::_split`.

### CEGIS loop

`cegis_resolve` (in `build_meterbill.py`) is counterexample-guided: it evaluates
the synthesized program against the reference across `_domain(ir)` and collects
disagreements as new I/O examples. Two properties make it converge honestly:

- **Spread, not first-N** — counterexamples are sampled *evenly across the
  domain*, not taken as the first few. A tiered rule disagrees in contiguous runs
  (all of one tier near a wrong breakpoint); the first few would all land in one
  tier and never pin the *other* breakpoints. Spread evidence forces the true
  multi-tier program.
- **Clean-sweep gate** — a result is ranked above any other only if it survives a
  *full* domain sweep with zero disagreements, and a final gate **downgrades any
  best that is still wrong anywhere to `low`**. This can only lower a grade,
  never raise one, so it cannot manufacture confidence — it just refuses to ship
  a program a dense sweep proves wrong.

For the six single-threshold rules it converges in **+0 rounds**. `api_bill`
takes one round of spread counterexamples to move the search off its sparse-fit
overfits and onto the exact tiered rule, which then sweeps clean and certifies.

---

## How the rules get synthesized

The capability that makes the piecewise rules possible lives in the `nsynth`
Rust crate (`nsynth/src/solver/scalar_search.rs`,
`search_scalar_families.rs`, `post_enumerative.rs`):

- **Mined constants, breakpoints preserved** — `mine_scalar_constants()` derives
  each problem's candidate constant set from its own data: example
  inputs/outputs, intercepts (`t − x`), exact slopes (`t / x`), sorted
  step-diffs, and negations. This replaces a fixed `[-1, 0, 1, 2, 3, 10, 100]`
  pool that could never express a 50- or 100-unit breakpoint. Raw inputs and
  outputs are kept even when large, so a tier threshold like **10000** survives
  truncation (a by-magnitude cap would have dropped it, leaving tiered rules
  unexpressible).
- **Agreement ranking** — branch-expression candidates are ranked by how many
  examples they already satisfy *before* the 800-candidate cap, so a
  correct-but-deep expression like `(x − 50) × 5` survives truncation; survivors
  are then Occam-re-sorted (simplest first) for selection.
- **Boundary-continuity penalty** — for single-threshold rules, then/else pairs
  that jump discontinuously at the breakpoint are penalized, so search lands on
  the threshold where the branches meet (`50 < x`, not a lexically-tied
  `40 < x`).
- **No divide/mod by the input** — `/` and `%` are only allowed with a
  data-independent divisor (`x % 10` stays; `8000 % x` is banned). Dividing or
  modding *by* the input is almost never a real scalar rule — it is how branch
  search overfits a tiered function — so removing it makes genuinely multi-tier
  data fail single-branch search honestly and fall through to two-branch search.
- **Simplest representative, deterministic** — when several expressions produce
  identical outputs on the examples, the candidate pool keeps the *simplest* one
  (with a stable tiebreak) instead of whichever a hash map visited first. This
  makes synthesis reproducible and lets the clean affine `x + 8000` win over an
  equal-output modulo overfit `x % 10001 + 18001` every time.
- **Piecewise-affine recovery (any number of tiers)** — `search_piecewise_affine`
  detects the affine segments in the data (maximal runs of constant slope) and
  places each breakpoint at the exact integer where the adjoining pieces
  intersect. This recovers tiered/threshold/clamp rules of *any* number of
  segments **exactly**, so they generalize by construction rather than being fit.
  It commits only when the data is confidently piecewise-affine (2–6 segments,
  each backed by ≥2 colinear points), so curves and loops fall through to the
  other solvers instead of being faked with a per-point staircase.

---

## Measured: does the engine actually generalize?

A demo that passes seven hand-written rules proves little. The real question is
whether the *engine* — given only examples, no reference, no curation — produces
programs that are correct on inputs it never saw. `measure_generalization.py`
answers it: it generates random continuous piecewise-affine functions (the shape
of real tier schedules), feeds raw nsynth a modest training sample, and checks
the returned program against **dense unseen points**.

```bash
PYTHONPATH=. python3 demos/requirements_app/measure_generalization.py 40 7
```

A program counts as SOLVED only if it is exactly correct on every unseen point;
fitting the samples but diverging between them is an OVERFIT. Adding the
piecewise-affine solver moved the numbers (40 random rules, examples only):

| | before | after |
|---|--:|--:|
| **SOLVED** (correct on unseen) | 32% | **78%** |
| OVERFIT (wrong between samples) | 20% | 8% |
| FAILED (no program) | 48% | 15% |
| 2-tier | 3/13 | **13/13** |
| 3-tier | 0/9 | **5/9** |
| 4-tier | 0/8 | **4/8** |

The 105-problem solver benchmark stayed at 100% throughout and uses the piecewise
solver on *zero* of its problems — it never contained a multi-tier rule, which is
exactly why this capability gap was invisible until measured directly.

---

## Files

| File | Role |
|------|------|
| `build_meterbill.py` | Driver: rules, `ScriptedProposer`, `cegis_resolve`, `build_html`, HTML template |
| `measure_generalization.py` | The capability meter: random piecewise rules → raw nsynth → correctness on unseen points |
| `meterbill.html` | Generated calculator (commit artifact; regenerate by running the script) |
| `provenance.json` | Generated audit trail (commit artifact) |

Depends on `ncpu/requirements/{ir,proposer,pipeline}.py` and the `nsynth`
release binary (built automatically by the pipeline on first use).

## Reproducing

```bash
PYTHONPATH=. python3 demos/requirements_app/build_meterbill.py
open demos/requirements_app/meterbill.html      # macOS; or just open in a browser
```

Synthesis is deterministic for the certified rules. Set
`NSYNTH_RANDOM_RESTARTS=0` for the most reproducible run.
