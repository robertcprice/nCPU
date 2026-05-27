## 18. On Sample Size in Coprocessor Real-World Evaluation

Section 11 of this paper describes the nCPU differentiable coprocessor: an arithmetic-specialist module injected into a transformer's forward pass and trained on synthetic and GSM8K arithmetic. The natural question is whether the coprocessor's gains on arithmetic transfer to real-world code generation, measured on HumanEval, or to chain-of-thought reasoning, measured on GSM8K.

The literature on AI coprocessors typically answers "yes" or "no" based on tiny problem samples --- often N=10, sometimes N=30 --- and reports deltas on that basis. This section argues that **no transfer conclusion is safe at N<100** and documents a protocol for honest reporting.

### 18.1 The Problem: Deltas at Small N are Indistinguishable from Noise

On a pass@1 benchmark with discrete outcomes (pass/fail), the variance of the estimated pass rate at sample size N is approximately p(1-p)/N, where p is the true rate. For a plausible baseline p=0.6 on HumanEval:

| N | Std error | 95% CI half-width | Minimum detectable delta (p=0.05) |
|---|---|---|---|
| 10 | 0.155 | ±0.30 | 0.43 |
| 30 | 0.089 | ±0.17 | 0.25 |
| 50 | 0.069 | ±0.14 | 0.19 |
| 100 | 0.049 | ±0.10 | 0.14 |
| 164 | 0.038 | ±0.07 | 0.11 |
| 500 | 0.022 | ±0.04 | 0.06 |

At N=10, you cannot reliably detect an improvement of less than 43 percentage points. At N=30, the minimum detectable effect is 25 points. Most real coprocessor effects --- at least, most that have appeared in published results on related work --- are single-digit to low-double-digit improvements. Measuring a 5-point effect at N=10 requires a sample approximately 75× larger before the confidence interval excludes zero.

This is not a novel observation in statistics. It is, however, routinely ignored in AI coprocessor papers, including in the nCPU coprocessor's own earlier real-world runs:

- `training_results/instruct_sweep/qwen3.5-4b/realworld_benchmark.json`: reports **0/10 baseline → 0/10 coprocessor** on a 10-problem coding suite. No signal either way; the baseline pass rate is below the resolution of the sample.
- `training_results/scaling_sweep_qwen35/qwen2.5-3b-instruct-humaneval-10.json`: a 10-problem HumanEval run reporting a delta of approximately -40%. At N=10, this is a change of four correct answers out of ten --- four coins flipping differently between baseline and coprocessor runs. The result is consistent with the coprocessor degrading 3B-instruct *or* with random variation around a true delta anywhere in [-0.7, +0.3].

We do not claim the qualitative finding is wrong. We claim it is not measured.

### 18.2 The Protocol: Commit to N Before Running

Two rules, applied in order:

1. **Choose N before seeing the baseline rate.** Set N based on the minimum effect size you would report as a finding. If you would report any positive integer pass@1 improvement, N must be at least 100. If you only care about effects of 10+ points, N can be as low as 50.

2. **Report baselines and deltas with confidence intervals, not point estimates.** "4B coprocessor: 42% ±5% vs baseline 39% ±5%" is honest. "4B coprocessor: 42% (baseline 39%)" hides the fact that the intervals overlap and the delta is not distinguishable from zero at the measured N.

A corollary: if your N is too small to distinguish a plausible positive effect from zero, *the correct published result is "no measurement"*, not "no effect" and not "small effect." This is the rule that failing coprocessor papers systematically violate.

### 18.3 Applying the Protocol to nCPU

The nCPU coprocessor's real-world evaluation is pinned at N=164 on HumanEval (the full set) and N=500 on GSM8K (a 500-problem slice). These sample sizes were chosen before measurement to support detection of a 10-point effect at p=0.6 with 95% confidence. The deploy script `scripts/gpu/deploy_coprocessor_realworld_vastai.sh` enforces these defaults; command-line flags exist to make the N explicit in any run.

The smaller N=10 runs committed in `training_results/` predate this protocol and should not be cited as transfer evidence in publication. They remain in the repository for traceability of the engineering history, not as scientific claims.

### 18.4 Why This Section Exists

Scientific progress in AI depends on a literature where claims of transfer are testable and distinguishable from noise. A field where single-digit pass@1 deltas are reported from 10-problem runs is a field where the replication crisis has not yet arrived only because nobody has tried to replicate. Writing out the sample-size requirement explicitly --- and enforcing it in our own deploy scripts --- is an attempt to not be part of that problem.

The nCPU coprocessor's actual transfer numbers, when measured at honest N, will appear in §11.9 of the main paper as they become available. If the coprocessor does not transfer at the promised N, the paper will say so. If it does, the delta will come with an interval. Either outcome is publishable; neither outcome is obscured by a too-small sample.

### 18.5 Related Practice

The GSM8K and HumanEval creators reported their benchmarks at their full N (1319 and 164 respectively). The broader open-model literature evaluates on these full N values and reports intervals; see the HELM and Open LLM Leaderboard conventions. Deviations from full N in published literature are typically motivated by cost but reported with their confidence implications.

Where the nCPU coprocessor's earlier runs deviated was not in choosing N=10 to save cost --- that is a legitimate engineering choice for smoke testing --- but in treating the resulting numbers as transfer evidence. This section separates those two use cases.

### 18.6 Tooling

`benchmarks/benchmark_coprocessor_realworld.py` accepts `--humaneval-count` and `--gsm8k-count` flags. The deploy script defaults to the full 164 and 500. Running with smaller N is supported and produces the same JSON output format, so a reader can replicate either regime. The regression test harness at `tests/test_coprocessor_sample_size.py` (forthcoming with the vast.ai results) will assert that every delta reported in `paper/` is accompanied by a confidence interval derived from an N≥100 sample.

The proposal is: don't let the literature ratchet downward on sample size just because the local benchmark is cheap to run small. Pick the N your finding requires, report the interval, and publish the null results honestly when the coprocessor does not transfer.
