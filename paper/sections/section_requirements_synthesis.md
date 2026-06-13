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

* **Mined constants.** `mine_scalar_constants` derives each problem's candidate
  constant set from its own data — example inputs and outputs, intercepts
  (`t − x`), exact slopes (`t / x`), the gaps between sorted output steps, and
  negations of all of these — anchored on {−1, 0, 1}. The breakpoint 50 in
  "first 50 GB free" enters the vocabulary because the examples *imply* it, not
  because a human anticipated it. This replaces a fixed `[-1, 0, 1, 2, 3, 10,
  100]` pool that could never have expressed it.

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

A verified single- or two-branch search result also pre-empts the native
gradient-distillation stage (`search_result_preempts_native_gradient`), since an
exact, human-readable piecewise program is preferable to a distilled
approximation of one.

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
domain comparing the synthesized program against the reference, collects up to a
handful of disagreements per round as new I/O examples, hands them back to the
proposer, and re-synthesizes — keeping the best result by
`(confidence, holdout_passed)`. For rules that already generalize this converges
in zero rounds (the first synthesis matched the spec across the whole domain);
for rules that do not, it is the mechanism that *surfaces* the failure rather
than papering over it.

### 21.5 Case Study: MeterBill

`demos/requirements_app/build_meterbill.py` exercises the whole stack on a
usage-based SaaS billing engine. Seven charge rules are stated as
product-manager English; the pipeline synthesizes, verifies, and grades each,
transpiles the certified ones Mog → TypeScript, and assembles an interactive
calculator (`meterbill.html`, and a live page in the nCPU site) plus a
`provenance.json` audit trail recording prose → IR → method → holdout →
confidence for every rule.

**Result: 7/7 synthesized; 6 certified high-confidence; 1 held back.** The four
piecewise rules are the substance of the result — their breakpoints (50, 100,
10) were mined, not given:

| Rule | English (abridged) | Discovered program | Grade |
|---|---|---|---|
| `seat_cost` | \$12 per seat | `12·x` | high |
| `annual_prepay` | year prepay = rate × 10 | `10·x` | high |
| `storage_overage` | first 50 GB free, then \$5/GB | `if 50<x: (x−50)·5 else 0` | high |
| `call_cost` | first 100 min free, then 2¢/min | `if 100<x: (x−100)·2 else 0` | high |
| `support_credit` | \$5/ticket, capped at 10 | `if 10<x: 50 else 5·x` | high |
| `loyalty_points` | 1 pt/\$ + bonus above \$100 | `if 100<x: x+(x−100) else x` | high |
| `api_bill` | tiered: free / \$0.01 / \$0.005 | *(fit found, not certified)* | **low** |

The headline is the last row. `api_bill` is a two-breakpoint tiered schedule —
strictly beyond a single-threshold branch program. nsynth fits the seed
examples, but on held-out inputs the program drops to 2/3 and disagrees with the
reference; CEGIS adds counterexamples and it still does not certify. **The
pipeline refuses to ship it.** It is shown uncertified and is never executed in
the calculator. A billing system that confidently emits a *wrong* tiered price
is worse than one that says "not sure about this one" — and the value of the
whole construction is that it can tell the two situations apart and act
differently. The six certified rules carry a proof (holdout-clean plus an
agreeing independent reference); the seventh carries an honest refusal.

### 21.6 What This Adds to the Stack

Section 20 showed verified synthesis from examples; this section closes the gap
to natural-language requirements without weakening the contract. The same
proof-or-refusal discipline now runs from a paragraph of English to a deployed,
typed function, with a graded confidence that is calibrated by construction:
*certified* means a program reproduced unseen examples and matched an
independently-derived reference, and *refused* means the system could not
establish that — and said so, instead of guessing.
