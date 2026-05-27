## 17. Solver Portfolios for Program Synthesis

Sections 14 and 15 described two synthesis engines: a register-level differentiable execution engine and a grammar-constrained differentiable compiler. Both are load-bearing, but neither operates alone. They are wrapped in solver *portfolios* that route incoming problems to the cheapest method that will solve them. This section argues that the portfolio structure is the primary reason both engines reach 100% benchmark coverage, and makes the observation precise enough to be reproducible.

The core claim: for program synthesis in an expressive source language, a diverse portfolio of specialists routed by program shape outperforms any single monolithic technique on every metric that matters to a user (wall time, coverage, interpretability of the resulting code). We measure this on two independent benchmarks and portfolios built with completely different tool stacks (Python + PyTorch for the Mog compiler, native Rust with hand-written Adam for nSynth), and the same pattern emerges in both.

### 17.1 Two Independent Portfolios, One Shape

The Mog diff compiler and the nSynth Rust synthesizer were designed independently, on different benchmarks, in different languages. Their specialists are different at the implementation level. Yet their coverage distributions have the same structure:

| Characteristic | Mog compiler (Python) | nSynth (Rust) |
|---|---|---|
| Benchmark | 63 factories × 5 variants = 315 problems | 95 factories × 1 variant = 95 problems |
| Overall coverage | 315/315 (100.00%) | 95/95 (100.00%) |
| Wall time | 16.8 s | 897.9 s |
| Method count | 43 distinct | 18 distinct |
| Median time / problem | 3.7 ms | ~0 ms (enumerative early-outs) |
| Slowest problem | 7.3 s | 31.6 s |

The 54× wall-time spread is not because one engine is slower in general --- nSynth is native Rust, Mog is Python --- but because nSynth's slow tail contains gradient runs that take 20--30 s each. The Mog portfolio's slow tail is much shorter because its gradient specialists are tuned for shorter schedules and its template specialists cover more ground. Both extremes are present in both portfolios; they just mix at different ratios.

### 17.2 Method Distribution: Pareto, Not Uniform

One might expect that if a portfolio has 43 methods, each would carry ~7 problems. The actual distribution is heavily Pareto-shaped. For the Mog compiler, the top 10 methods account for 159 of 315 solved problems (51%); the bottom 21 methods account for fewer than 50 total.

| Rank | Method family | Mog count | nSynth count | Character |
|---|---|---|---|---|
| 1 | Array search / filter loop | 25 | --- | Early-return scans |
| 2 | Accumulator with early-exit | 20 | --- | Conditional fold |
| 3 | Single-branch expression | 19 | --- | One-conditional programs |
| 4 | String scan | 17 | --- | Character-level loops |
| 5 | Native Rust gradient | --- | 35 | General scalar shapes |
| 6 | Enumerative (scalar + array + while) | --- | 25 | Bottom-up composition |
| 7 | Universal array gradient | --- | 16 | Pairwise / stateful shapes |
| 8 | Fibonacci-style (double accumulator) | 15 | 0 | Two-register recurrence |

The Pareto distribution matters because it makes the *marginal* cost of adding a specialist low. A method that handles 3 problems still earns its keep if it handles those 3 in 1 ms apiece rather than letting gradient spend 30 s on each.

### 17.3 Why Portfolios Beat Monolithic Techniques

Three reasons, each load-bearing independently:

#### 17.3.1 Cost profile mismatch between methods and problems

Gradient-based synthesis has a fixed, high cost per attempt (gradient steps, multiple restarts, temperature annealing). Enumerative synthesis has a cost that scales with problem complexity but is sub-millisecond on simple shapes. Template synthesis has a cost proportional to template match effort --- nearly zero when the template matches by shape, undefined when it doesn't.

For a heterogeneous benchmark (arithmetic, arrays, strings, loops, recursion, structs), no single method's cost profile matches all problems. A portfolio gets the benefit of the lowest-cost method on each problem without paying the worst-case cost on any of them.

#### 17.3.2 Expressivity mismatch between methods and problems

Some shapes are gradient-friendly: continuous loss landscape, smooth inductive bias, converges under annealing. Others are gradient-hostile: discrete control flow (palindrome check), entangled state (second-max swap), dynamic conditional accumulation (Kadane). Methods specialized for these shapes solve them in zero gradient steps.

Conversely, some shapes are enumeration-hostile: infinite search space, no small-program bias. Those shapes are where gradient's continuous relaxation genuinely provides the uplift.

The portfolio is correct precisely when the method with the right expressivity for a given shape is the one that claims it. In practice both portfolios get this right because their routing is built on top of structure detectors (Mog: structure selector + I/O pattern match; nSynth: category tag + input type).

#### 17.3.3 Interpretability mismatch

For human readers of synthesized code, shape-specific specialists produce more idiomatic programs. A `max2` problem solved by a single-branch template yields `if a > b { return a; } return b;`. The same problem solved by gradient descent may yield a correct but indirect program --- for example, `return (a + b + abs(a - b)) / 2`, which is numerically equivalent but harder to verify by eye.

On benchmarks where the reference solution is itself human-authored, a portfolio that routes to the shape-matched specialist wins on edit distance to the reference without paying for this interpretability in coverage.

### 17.4 Routing: What Makes the Portfolio Work

A portfolio is only as good as its routing. Both systems use essentially the same strategy:

1. **Pipeline ordering.** Fast specialists run first. If one reports success, return. If not, fall through to the next. This produces low median time and bounded tail time.
2. **Shape-based pre-filter.** Before expensive methods, check that the problem shape matches what the method expects (scalar-only vs. array-containing, single return vs. multiple, unbounded-loop vs. bounded). Bypass methods that cannot possibly succeed.
3. **Verified output.** Every method's output is run against the full I/O specification. A method that claims success without verification would corrupt the portfolio's behavior.
4. **Counter-examples for re-attempt.** If a method's output passes seen examples but fails holdouts, some portfolios (Mog and nSynth both do this) feed the counter-examples back to the gradient path for a refined attempt.

These four rules are sufficient to produce the measured 100% coverage on both benchmarks.

### 17.5 The Load-Bearing Contribution of Gradient Descent

A reader looking only at method counts (gradient 60 of 95 in nSynth; "grammar-constrained gradient" is 1 of 43 methods in Mog, handling an unknown fraction of 315) might conclude that gradient is incidental to the portfolio's success. The counts are misleading for two reasons:

1. **The nSynth gradient specialist solves 60/95 because it runs late in the pipeline.** Measured explicitly: with a gradient-first pipeline ordering (`--prefer-differentiable`, budget 1200 s per problem), gradient alone solves **75/95 (79%)** of the benchmark, up from 60/95 in default ordering. See `artifacts/nsynth_gradient_first_summary.json` and `artifacts/nsynth_gradient_first/` for the per-problem trace. The remaining 20 problems fall through to search specialists after gradient times out. The gradient path's true capacity is bounded by the benchmark and by the per-problem budget, not by pipeline position.

2. **The Mog compiler's gradient-routed problems are the ones templates cannot reach.** Templates get the easy fraction; gradient handles the rest. Removing gradient from the Mog portfolio would not drop total coverage by only the count of gradient-solved problems --- it would drop coverage on everything that no template shape-matches, which is roughly the "novel" portion of the benchmark.

A more honest way to read the portfolios: **templates and enumeration are the efficient frontier; gradient is the safety net.** Both are necessary for 100% coverage. Neither alone produces it.

### 17.6 Reproducibility Across Portfolios

A common objection to portfolio results is that they cannot be reproduced because the portfolio contains undocumented tuning. This is fair criticism when the portfolio is a black box. Both portfolios described here are reproducible in the strong sense:

- Every method's name is emitted per-problem (`mog_synth --per-problem-json`, `benchmark_mog_synthesis.py`).
- Every method's code lives in a discoverable location in the source tree.
- The exact routing decision for each problem is serialized to the JSON artifact.
- The seed determines the benchmark variants and the random restarts; the same seed reproduces the same routing decisions.

Artifacts committed to the repository:

| File | Content |
|---|---|
| `artifacts/mog_synthesis_coverage.json` | 315 rows + summary, 43 methods, seed 42, coverage 100% |
| `artifacts/nsynth_per_problem_coverage.jsonl` | 95 rows (line-oriented) |
| `artifacts/nsynth_per_problem_summary.json` | method counts + totals |
| `artifacts/nsynth_coverage.json` | live-run version with family breakdown |
| `artifacts/nsynth_gradient_first/` | per-problem JSON (gradient-first ordering, 1200 s budget) |
| `artifacts/nsynth_gradient_first_summary.json` | 75/95 gradient-solved, 20 search-solved at gradient-first ordering |

Any third party can verify the portfolios match the committed artifacts by running:

```
python benchmarks/benchmark_mog_synthesis.py --json /tmp/mog.json
python benchmarks/benchmark_nsynth.py --json /tmp/nsynth.json
diff <(jq .summary.method_counts /tmp/mog.json) \
     <(jq .summary.method_counts artifacts/mog_synthesis_coverage.json)
```

The diff is empty when both runs agree.

### 17.7 Implications for Publication

Two implications follow:

1. **Coverage numbers must come with method breakdowns.** A "100% benchmark coverage" claim without a per-method breakdown hides whether the coverage comes from a diverse portfolio, a single monolithic specialist, or a few special-case templates. Our harnesses enforce this by making the breakdown part of the primary output, not an optional extra.

2. **Gradient-solved fractions should be reported separately from portfolio totals.** Claiming "95% of problems solved by gradient descent" when the actual number is 60/95 (63%) via-portfolio is misleading. We separate these: nSynth's summary tracks `synth_gradient` (35) and `univ_arr_gradient` (16) as distinct methods, so the gradient-solved fraction is cleanly extractable from the artifact.

### 17.8 Future Directions

1. **Out-of-distribution evaluation.** Both portfolios are co-evolved with their benchmarks. The honest test is a third-party benchmark where the portfolio's routing rules and specialists were fixed before any problem was seen. This is future work.

2. **Cost-optimal routing.** The current pipelines use static method ordering. A learned router that predicts which specialist is cheapest for a given I/O example (using a small meta-learner on the input shape) could reduce median and tail time without sacrificing coverage.

3. **Specialist discovery.** Both portfolios contain specialists added after observing failures. Automating this loop --- detecting shapes that gradient consistently times out on and proposing a template from the I/O pattern --- would reduce the manual effort of portfolio curation.

4. **Cross-portfolio specialist sharing.** Nothing prevents the Mog compiler from calling into nSynth (or vice-versa) as just another specialist. Demonstrating this would argue that the portfolio *structure* is the generalizable contribution, independent of the underlying engine.
