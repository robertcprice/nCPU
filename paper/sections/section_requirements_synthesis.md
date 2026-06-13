# Requirements Synthesis: From Complex English to a Verified, Running Program

## 21. Requirements Synthesis

Section 20 demonstrated the honest-synthesis contract on programs specified by
input/output example pairs: a playable Pong whose every rule was discovered from
examples and verified by domain sweep. This section pushes the specification one
level closer to how software is actually requested — **a paragraph of
English** — while keeping the same contract: *every shipped program is
proof-carrying, and anything that cannot be proved is refused.*

The architecture is a deliberate split of trust:

```
complex English  →  [LLM proposer]  →  RequirementsIR  →  [nCPU synth + verify]  →  graded program
                       UNTRUSTED                              TRUSTED
```

The LLM is fluent at reading messy prose and is therefore the right tool to
*propose* a structured spec — but it is exactly the wrong tool to *trust*, since
it will confidently hallucinate a formula. So the LLM is confined to the
untrusted front-end: it emits a `RequirementsIR` (a signature, concrete I/O
examples, and a reference implementation), and the pipeline treats every field
as a claim to be checked. The trusted half is nCPU and runs for real on every
request: bottom-up program synthesis, held-out generalization testing, and a
cross-check between two independently-derived programs.

### 21.1 The Pipeline and Its Graded Verdict

`resolve(english, proposer)` (`ncpu/requirements/pipeline.py`) executes five
stages and emits one of four confidence grades:

1. **Propose.** The proposer turns English into a `RequirementsIR`. In a live
   deployment this is an API call; in the reproducible demo it is an inlined,
   pre-authored IR satisfying the same protocol — the pipeline cannot tell the
   difference, which is the point.
2. **Split.** The I/O examples are partitioned into train and holdout (§21.3).
3. **Synthesize.** `nsynth` searches program space for a Mog program reproducing
   the **training** examples, self-verifying before it returns. No solution →
   the request is refused (`unsynthesized`).
4. **Generalize.** The synthesized program is executed against the **held-out**
   examples it never saw. Memorization cannot pass this; only generalization
   does.
5. **Cross-check.** The synthesized program is compared against the proposer's
   reference implementation across the input domain — agreement between two
   programs derived by different means (search vs. the LLM's own code).

The verdict is graded, not binary: **high** (holdout-clean *and*
reference-agreeing), **medium** (holdout-clean, reference unconfirmed), **low**
(fits the seed examples but fails to generalize), **none** (no program found).
Only `high`/`medium` programs are certified for deployment; `low` is surfaced
honestly as synthesized-but-uncertified and is never wired into anything live.

### 21.2 Bottom-Up Piecewise Synthesis: Discovering Thresholds Nobody Typed

Real requirements are piecewise — "the first 100 minutes are free, then a dollar
a minute"; "a credit per ticket, capped at ten." A synthesizer that can only
reach for a hardcoded constant list cannot express a breakpoint at 50 or 100 if
those numbers were never on the list. Three changes to the scalar branch search
(`nsynth/src/solver/scalar_search.rs`, `search_scalar_families.rs`) make novel
threshold and tiered rules discoverable bottom-up, continuing the mined-vocabulary
principle of §20.2:

* **Mined constants, breakpoints preserved.** `mine_scalar_constants` derives
  each problem's candidate constant set from its own data — example inputs and
  outputs, intercepts (`t − x`), exact slopes (`t / x`), the gaps between sorted
  output steps, and negations of all of these — anchored on {−1, 0, 1}. The
  breakpoint 50 in "first 50 GB free" enters the vocabulary because the examples
  *imply* it, not because a human anticipated it. This replaces a fixed
  `[-1, 0, 1, 2, 3, 10, 100]` pool that could never have expressed it. Raw inputs
  and outputs are retained even when large, so a *tier* threshold such as 10000
  survives — a fixed-size by-magnitude cap would have dropped it and left tiered
  rules unexpressible.

* **No divide or mod by the input.** `/` and `%` are admitted only with a
  data-independent divisor: `x % 10` (a digit/parity operation) stays, but
  `8000 % x` and `expr / x` are forbidden. Dividing or modding *by* the input is
  almost never a genuine scalar rule — it is precisely how branch search overfits
  a piecewise function (faking the api_bill tiers with `x − (−8000 % x)`).
  Removing the class makes genuinely multi-tier data fail single-branch search
  *honestly* and fall through to two-branch search.

* **Simplest representative, hash-independent.** When several expressions produce
  identical outputs on the examples, the candidate pool keeps the structurally
  simplest one (with a stable canonical tiebreak) rather than whichever a hash
  map happened to visit first. This makes synthesis reproducible run to run, and
  it is what lets a clean affine `x + 8000` be selected over an equal-output
  bounded-range modulo `x % 10001 + 18001` deterministically — the two are
  indistinguishable on sparse samples, so without this the wrong one could be
  emitted depending on the hash seed.

* **Agreement-ranked candidate pool.** The branch-expression pool is capped (at
  800 candidates) for tractability. A correct-but-deep expression such as
  `(x − 50) × 5` used to be evicted by lexically-earlier noise before it was
  ever tried. Candidates are now ranked by *how many examples each already
  satisfies* before the cap is applied, so the program that explains the data
  survives truncation; survivors are then re-sorted by complexity (Occam) for
  final selection.

* **Boundary-continuity penalty.** A single-threshold rule should *connect*
  where its two branches meet. `boundary_continuity_penalty` scores then/else
  pairs by their discontinuity at the candidate breakpoint, so the search lands
  on the threshold where the pieces actually join (`50 < x`) rather than a
  numerically-tied alternative (`40 < x`) that happens to fit the sampled points
  but kinks in the wrong place. This is what makes the discovered breakpoint the
  *semantically* correct one, not merely an interpolating one.

* **Exact piecewise-affine recovery.** The branch searches above enumerate over a
  candidate vocabulary; a tiered rule with *k* breakpoints needs *k+1* pieces and
  *k* conditions, and beyond two breakpoints the enumeration cost and overfit
  risk both climb (Section 21.6). `search_piecewise_affine` solves the whole
  family in closed form instead. It sorts the examples by input, greedily splits
  them into maximal *exact-affine runs* (consecutive points sharing one integer
  slope), and places each breakpoint at the integer `x` where the two adjoining
  pieces intersect — the true threshold of a continuous tier schedule, read
  directly from the data rather than searched for. The recovered program is exact
  on every example *by construction of the segments*, so it generalizes to unseen
  inputs; it is emitted only when the data is confidently piecewise-affine (two to
  six segments, each supported by at least two colinear points), so a curve
  (quadratic, modulo, a loop) — which would fragment into many two-point
  "segments" — is rejected rather than reproduced as a per-point staircase.

* **Exact multi-argument linear recovery.** Everything above is single-input;
  real requirements are mostly multi-argument (`cost(base, units)`,
  `ship(weight, zone)`), and the engine solved *none* of them. Three solvers
  close the linear family for two and three arguments. `search_affine` recovers a
  global affine `c0 + Σ c_j·x_j` by solving the integer linear system the
  examples define (Gaussian elimination, rounded, then verified — a non-affine
  fit is rejected). `search_affine_threshold` and `search_affine_piecewise`
  recover a rule that is affine in all arguments within each tier of *one*
  threshold argument: sort by that argument, fit an affine to each tier, and
  place each breakpoint where the two adjoining pieces meet on the threshold axis
  — the multi-dimensional analogue of the 1-arg intersection, valid as a clean
  threshold exactly when the other arguments' slopes agree across the tiers so
  their terms cancel. These run *first* in the solver pipeline: a linear rule is
  recovered in microseconds, so it must short-circuit ahead of the search and
  gradient stages (which environment-specific initialisation can otherwise stall)
  rather than be reached by chance.

A verified single-branch, two-branch, or piecewise-affine search result also
pre-empts the native gradient-distillation stage
(`search_result_preempts_native_gradient`), since an exact, human-readable
program is preferable to a distilled approximation of one — and, for the
piecewise solver, returning the verified program directly is what keeps a
multi-tier rule from falling through to the slow gradient path and timing out.

### 21.3 Honest Holdout: Why the Split Is Strided

A naïve train/holdout split takes a tail slice. For a piecewise rule whose
examples are ordered by input magnitude, a tail slice exiles an entire region of
the input domain — say, every large-`x` example — exclusively into the holdout.
Training then never observes that region, cannot pin the rule there, and *even
the correct program* appears to fail generalization: the honest test has been
rigged into a guaranteed failure. The split is therefore **strided** — every
third example is held out (`i % 3 == 2`) — so train and holdout each span the
full input domain. The check becomes the honest one it was meant to be: learn
from a representative sample, verify on unseen points drawn from the same spread.
This is a one-line policy in `_split`, but it is the difference between a
generalization test that measures generalization and one that measures the
ordering of the example list.

### 21.4 Counterexample-Guided Tightening

When a synthesized program agrees with the reference on the holdout but might
still diverge elsewhere, the demo driver runs a CEGIS loop: it sweeps the input
domain comparing the synthesized program against the reference, collects
disagreements as new I/O examples, hands them back to the proposer, and
re-synthesizes. Two properties make the loop converge *honestly* rather than
merely terminate:

* **Counterexamples are spread, not first-N.** A piecewise rule disagrees in
  contiguous runs — all of one tier near a wrong breakpoint. Taking the first few
  disagreements would draw every counterexample from a single region and never
  pin the *other* breakpoints; the loop would keep proposing fits that are right
  where it has looked and wrong where it has not. Sampling the disagreements
  evenly across the domain supplies evidence from every region, which is what
  forces a true multi-tier program to emerge.

* **A clean-sweep gate, downgrade-only.** A candidate is ranked above another only
  if it survives a *full* domain sweep with zero disagreements, and a final gate
  reduces any surviving best that is still wrong anywhere to `low`. Because the
  gate can only ever lower a grade, it cannot manufacture confidence; it can only
  refuse to certify a program a dense sweep has proven wrong. This is what stops a
  program that fits the sparse holdout but is wrong between sampled points from
  being shipped — the failure mode §21.5 walks through in detail.

For rules that already generalize this converges in zero rounds; for the tiered
rule it is the mechanism that both *surfaces* the overfit and *drives the search*
to the correct program.

### 21.5 Case Study: MeterBill

`demos/requirements_app/build_meterbill.py` exercises the whole stack on a
usage-based SaaS billing engine. Seven charge rules are stated as
product-manager English; the pipeline synthesizes, verifies, and grades each,
transpiles the certified ones Mog → TypeScript, and assembles an interactive
calculator (`meterbill.html`, and a live page in the nCPU site) plus a
`provenance.json` audit trail recording prose → IR → method → holdout →
confidence for every rule.

**Result: 7/7 synthesized and certified high-confidence.** The five piecewise
rules are the substance — their breakpoints (50, 100, 10, 1000, 10000) were
mined, not given:

| Rule | English (abridged) | Discovered program | Grade |
|---|---|---|---|
| `seat_cost` | \$12 per seat | `12·x` | high |
| `annual_prepay` | year prepay = rate × 10 | `10·x` | high |
| `storage_overage` | first 50 GB free, then \$5/GB | `if 50<x: (x−50)·5 else 0` | high |
| `call_cost` | first 100 min free, then 2¢/min | `if 100<x: (x−100)·2 else 0` | high |
| `support_credit` | \$5/ticket, capped at 10 | `if 10<x: 50 else 5·x` | high |
| `loyalty_points` | 1 pt/\$ + bonus above \$100 | `if 100<x: x+(x−100) else x` | high |
| `api_bill` | tiered: free / 2¢ / 1¢ | `if 10000<x: x+8000; if 1000<x: 2(x−1000); else 0` | high |

The headline is the last row — not because it is refused, but because of what the
contract had to *reject* before certifying it. `api_bill` is a two-breakpoint
tiered schedule, strictly beyond a single-threshold program. The first fits the
search returned **passed the sparse holdout but were overfits**: an
integer-division flooring term `(x / 1001)·2000`, exact only at sampled points,
and a bounded-range modulo `x % 10001`, which mimics `x − 10001` up to the
largest training input and wraps past it. Each reproduced a three-point holdout
exactly. A pipeline that trusted the holdout would have certified a wrong billing
rule.

It does not. The CEGIS loop sweeps the *entire* input domain against the
reference, feeds back disagreements sampled **evenly across the domain** (so the
evidence pins every tier, not just the one nearest a wrong breakpoint), and a
final gate **refuses to certify any program still wrong anywhere on the sweep** —
an operation that can only lower a grade, never raise one. Only once the search
produced the exact tiered rule — zero disagreements across the domain and beyond
the training range — did `api_bill` certify. Two synthesizer properties made that
rule reachable at all: constant mining preserves large tier breakpoints (10000)
instead of truncating them by magnitude, and the candidate pool keeps the
*simplest* expression for each distinct behaviour, so the clean affine `x + 8000`
is selected over an equal-output modulo overfit deterministically rather than by
hash order. The honest-refusal path remains exactly as before for any rule the
sweep can never satisfy; `api_bill` simply earned its way off it.

### 21.6 Measured Generalization, Not a Curated Demo

A seven-rule demo that passes proves the rules were chosen to pass. The load-
bearing claim is about the *engine*: given only examples — no reference, no
counterexample loop, no curation — does it return programs correct on inputs it
never saw? We measure it directly. A harness generates random continuous
piecewise-affine functions (the shape of real tier schedules), feeds the raw
synthesizer a modest training sample, and scores the returned program against
*dense unseen points*. A program counts as solved only if it is exactly correct
on every unseen point; fitting the samples but diverging between them is an
overfit, scored as a failure of exactly the kind sparse holdouts miss.

One probe exists per rule shape; each was at, or near, a flat zero before the
exact solvers and is recovered exactly after — from examples alone, correct on
unseen inputs:

| Rule shape (random, examples only) | before | after |
|---|---:|---:|
| 1-arg piecewise / tiered | 32% | **78–80%** |
| 2-arg affine `c0+c1·a+c2·b` | 0% | **100%** |
| 3-arg affine | 0% | **100%** |
| 2-arg single-threshold | 0% | **~75%** |
| multi-arg tiered (affine in two args, tiered by one) | 0% | **83%** |

The result that matters is the shape, not any single percentage: the engine went
from solving *none* of three classes of rule (multi-tier, multi-argument affine,
multi-argument tiered) to solving most of each. The hardest, most realistic shape
— a tiered-pricing rule over several arguments — moved from 0% to 83%, and the
single change that carried it was placing each breakpoint at the *intersection*
of the adjoining affine pieces rather than at the last sampled point (overfit
71% → 4%): the breakpoint is recovered, not guessed.

Crucially, the 105-problem solver benchmark held at 100% throughout and invokes
these solvers on at most a handful of its problems — it contains no multi-tier or
multi-argument-tiered rule, so these gaps were invisible to it. Saturated
coverage on a fixed benchmark measures only what the benchmark contains; a
generated, held-out probe measures the capability itself.

### 21.7 What This Adds to the Stack

Section 20 showed verified synthesis from examples; this section closes the gap
to natural-language requirements without weakening the contract. The same
proof-or-refusal discipline now runs from a paragraph of English to a deployed,
typed function, with a graded confidence that is calibrated by construction:
*certified* means a program reproduced unseen examples and matched an
independently-derived reference, and *refused* means the system could not
establish that — and said so, instead of guessing.
