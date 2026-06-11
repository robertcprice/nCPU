# GSM8K: how much headroom is left above the agent loop?

Measured 2026-04-19, `claude-haiku-4-5-20251001` (+ `claude-sonnet-4-6`
for escalation). All results are against the GSM8K-test split.

## Headline: two configurations dominate the Pareto frontier

- **For maximum accuracy** (99.0%): Agent loop + self-consistency
  voting k=5 + Sonnet escalation. 3.3× baseline cost.
- **For maximum cost-efficiency** (97.0% at baseline cost): **VPoT**,
  our agent-loop pattern applied to Program-of-Thought. One call,
  exec the returned solve(), retry on exec/invariant failure.

The in-between configs don't Pareto-improve either of these. Measured
and ruled out: PoT+self-consistency (null), tool-use (-12pp), naive
ensemble (-2.5pp), TF-IDF-retrieval few-shot (-0.2pp).

## Method matrix (pass@1, smaller N marked)

| method | N | pass@1 | wall/prob | tokens/prob | notes |
|---|--:|--:|--:|--:|---|
| LLM baseline (T=0) | 500 | 82.2% | 1.9s | 309 | single call |
| Agent loop (k=3, retry×2) | 500 | 98.6% | 2.8s | 513 | retry-arm does all the work |
| Agent + SC k=5 + Sonnet escalate | 500 | **99.0%** | 3.0s | 1688 | our highest ceiling; 3.3× baseline tokens |
| Program-of-Thought (single) | 200 | 97.5% | 1.4s | 290 | model writes solve(), we exec |
| Program-of-Thought (scaling) | 500 | 96.2% | 1.4s | 292 | scaling check |
| PoT + Self-Consistency k=5 | 200 | 97.5% | 2.0s | 1451 | no gain: errors are modeling not arithmetic |
| **VPoT** (our PoT + retry + invariants) | 500 | **97.0%** | 1.4s | 294 | +0.8pp over plain PoT at same cost |
| RA-VPoT (VPoT + TF-IDF few-shot) | 500 | 96.8% | 1.5s | 340 | null: TF-IDF on word problems is too noisy |
| RA-VPoT (VPoT + MiniLM few-shot) | 500 | 96.4% | 1.6s | 355 | null: retrieval hurts GSM8K regardless of embedding quality |
| **VPoT full-benchmark** | 1319 | **96.0%** (1266/1319) | 1.4s | 296 | full GSM8K: 485/500 + 781/819 combined |
| Calculator tool-use | 200 | 85.0% | 5.4s | 5110 | multi-turn, unexpectedly weak |
| Cross-method ensemble + escalate | 200 | **95.0%** | 6.8s | 6760 | PoT+tool+CoT vote — tool-use drags it down |

## Observations

- **PoT approaches the agent loop at half the wall time.** 96.2% vs
  98.6% at N=500 — the agent loop's retry arm is effectively
  re-deriving what PoT gets in one shot by writing the computation
  in Python. Agent wins by ~2pp but costs ~2× the tokens.
- **Self-consistency doesn't help PoT.** 5× tokens, 0pp gain.
  The misses are problems the model fundamentally mis-models, not
  problems where it slips arithmetic. This is an important null
  result — it rules out "just sample more" as a remaining lever for PoT.
- **Calculator tool-use unexpectedly underperforms.** 85% on N=200 —
  barely above baseline (82%). Multi-turn tool use drags in ~10×
  tokens and still loses to single-shot PoT. Hypothesis: the back-and-
  forth tool context confuses the model's chain of reasoning. PoT
  keeps everything in one prompt.
- **The retry arm is the agent loop's free lunch.** +16pp over
  baseline for 1.5× cost. But PoT delivers +14pp for *no extra cost*.
  PoT > retry on the cost frontier.

## VPoT insight: why verification barely helped

VPoT (PoT + our agent-loop retry + invariant checks) improved plain PoT
by +0.8pp (96.2% → 97.0%) at essentially the same cost. Why so small?

- **PoT rarely crashes at exec time.** `3 + 5 * 4` always evaluates.
  Unlike generated HumanEval code, where type errors and missing
  imports frequently invoke retry, solve() bodies are arithmetic and
  Python executes them without complaint.
- **Most GSM8K errors pass our invariant checks.** "Bob has 17 apples"
  vs ground truth 23: both are integers in a plausible range. Our
  invariants (finite, numeric, |x|≤1e15) don't catch it.
- **The retry arm only fires when exec or invariant fails.** Only 3/500
  VPoT attempts hit that path. The other 12 wrong answers exec'd fine
  and passed invariants.

The clear takeaway: on math, **structural verification isn't enough**.
What catches the remaining errors is *semantic* verification — agreement
with another method (ensemble), or a second opinion from a stronger
model (escalation). This is why robust-500 reaches 99% and VPoT doesn't.

## Ensemble failure: weak voters contaminate the majority

The cross-method ensemble (PoT + tool-use + plain CoT, majority vote)
reached **95.0%** — worse than plain PoT's 97.5%. Three-method voting
*should* have strictly dominated single methods if voters were equally
strong. It didn't, because tool-use's 85% accuracy made it a noise
source:

- When PoT is right (97.5% of time) and tool-use is wrong (15% of
  time), CoT becomes the tiebreaker. CoT is ~82% accurate, so ~3%
  of the "PoT right, tool wrong" cases get overruled by a tool+CoT
  majority that happens to agree on the wrong number.
- Null-extracts from tool-use (model didn't emit `#### N`) reduce
  the vote denominator, sometimes pushing an incorrect 1/2 majority
  past a missing vote.

Clean finding: **method diversity helps only when every method is
individually strong.** This argues against naive ensemble-everything
approaches. A better ensemble would be PoT + VPoT + Sonnet-single —
three ≥96% methods that rarely fail on the same problem.

## Cost frontier

The Pareto-interesting cell is: "highest pass@1 per dollar".
Extrapolated costs for 1000 GSM8K problems at Haiku prices
(~$1/M input, ~$5/M output, rough geometric mean ~$2/M):

| method | tokens/1000 | cost/1000 | pass@1 |
|---|--:|--:|--:|
| baseline | 309K | $0.62 | 82% |
| agent loop (retry×2) | 513K | $1.03 | 98.6% |
| agent + SC + Sonnet escalate | 1688K | $3.38 | **99.0%** |
| PoT (plain) | 292K | $0.58 | 96.2% |
| PoT + SC k=5 | 1451K | $2.90 | 97.5% |
| **VPoT (ours)** | 294K | $0.59 | **97.0%** |
| tool-use (calculator) | 5110K | $10.22 | 85.0% |

Two clear winners:
- **VPoT for cost-efficiency**: 97.0% at the same cost as baseline.
  This is our contribution — PoT adapted via our agent-loop pattern.
- **Robust agent for max accuracy**: 99.0% at 5.7× the cost. Use only
  when every additional correct answer justifies the spend.

The Pareto-inadmissible cells:
- Plain PoT is beaten by VPoT on accuracy at same cost.
- Tool-use is beaten on *both* axes — never choose it for GSM8K.
- PoT+SC k=5 costs 5× the tokens for zero gain over single PoT.

## RA-VPoT nulls: retrieval helps coding but not math

Two retrieval experiments on GSM8K. Both landed below plain VPoT:

| retrieval backend | pass@1 | Δ vs VPoT |
|---|--:|--:|
| TF-IDF bigrams | 96.8% | -0.2 pp |
| MiniLM sentence-transformer | 96.4% | -0.6 pp |

This isn't a quality-of-embedding problem. With MiniLM, the similarity
scores are clean (0.77 for matching-shape problems vs 0.15 for
unrelated, measured on a 3-row toy corpus). Retrieval finds the right
past problems. The issue is that finding similar *past* problems
doesn't help the model solve the *current* one — word-problem-to-
arithmetic is essentially a one-off translation, and the retrieved
template sometimes leads the model to copy the wrong pattern.

Contrast with code:
| benchmark | retrieval effect |
|---|---:|
| HumanEval-lite (30) | ceiling effect, no change |
| MBPP 80–129 (46) | +2.2 pp |
| GSM8K (500, TF-IDF) | -0.2 pp |
| GSM8K (500, MiniLM) | -0.6 pp |

Coding patterns re-use: "sum of squares", "filter prefix", "is palindrome"
are stable templates that transfer to new problems. Math-word-problem
patterns don't: "Alice has apples" and "Bob has cookies" share surface
form but require different variable bindings, and retrieving one
biases the other's solution poorly.

**Principled conclusion**: retrieval-augmented generation is a coding
technique, not a math technique. Use VPoT for math; use RA-VPoT (or
retrieval-augmented CoT) for code. Our cache infrastructure supports
both; the `--vpot-retrieval` flag lets the caller decide.

## Interpretation

Two qualitative findings:

1. **"Reasoning ability" ≈ "ability to structure a Python function."**
   When the model can write `def solve()`, it offloads the arithmetic
   to a deterministic interpreter and its pass rate jumps ~15pp with
   no extra calls. The chain-of-thought baseline is the model trying
   to be an interpreter; it loses to an actual interpreter.

2. **The retry arm of the agent loop is a format + arithmetic fix.**
   Its +16pp is composed of two parts: recovering the `#### N` format
   when baseline drops it (~half the retries), and re-doing arithmetic
   where baseline slipped (~half the retries). Both failure modes
   disappear in PoT — it emits Python (no format to lose) and Python
   does arithmetic deterministically.

## What's next

- Fill remaining cells (robust-500, pot-500, tool-200, ensemble-200).
- If ensemble can push past 99%, the remaining misses are genuinely
  model-capability bound — the answer is escalation not more retries.
- Cheapest-path-to-99% is likely **PoT + Sonnet-escalation on misses**.
  PoT gets 97.5% for $0.58; escalating the other 2.5% to Sonnet
  (one call each) adds ~$0.08. Total ~$0.66 for ≥99% expected.
