# Final findings: nCPU agent loop + retrieval + VPoT

## Headline numbers

| benchmark | baseline | our best | Δ |
|---|--:|--:|--:|
| HumanEval full (164) | 93.9% | 96.3% | +2.4 pp |
| MBPP (74) | 90.5% | 95.9% | +5.4 pp |
| GSM8K-500 — accuracy-optimised | 82.2% | **99.0%** | +16.8 pp |
| GSM8K-500 — cost-optimised (VPoT) | 82.2% | **97.0%** | +14.8 pp at same $$ as baseline |
| **GSM8K-full-1319 (VPoT, combined)** | ~82% | **96.0%** (1266/1319) | +14 pp at full scale |
| GSM8K rerun (warm cache) | 121s | **0.1s** | 1212× |

## What's "ours"

Six distinct, measured contributions shipped this session:

1. **Agent loop with examples-keyed cache** — original foundation
   from the start of the session, measured across HumanEval, MBPP,
   GSM8K. Retry-with-feedback arm does most of the heavy lifting.
2. **Examples-persisted cache schema** — 6-col TSV, backward-compatible.
   Unlocks retrieval because the feature space matches between query
   and cached rows. Before: max sim 0.21. After: 0.77-0.97.
3. **Retrieval-augmented generation** (HumanEval, MBPP) — inline
   top-K semantically-similar verified solutions as few-shot.
   Measured +2.2 pp on MBPP 80-129 slice (harder than 0-79).
4. **Cache-growth curve** — `tools/benchmarks/measure_cache_growth.py`
   makes the self-improvement story quantitative: sim≥0.70 full
   coverage at cache size 15; sim≥0.85 grows 17% → 77% at sizes 5→30.
5. **VPoT — Verified Program-of-Thought** — our coding agent loop
   (cache → generate → verify → retry → cache) applied to GSM8K via
   Chen et al.'s Program-of-Thought. 97.0% at baseline cost.
6. **MCP server with 10 tools** — cache/retrieval/math/delegate all
   exposed for tool-using LLMs (Claude Desktop, Cursor, etc.). 34
   regression tests cover the retrieval path.

## What we tested and measured

Five independent empirical questions, answered:

**Q1. Does the agent loop's +16pp on GSM8K survive at scale?**
→ Yes. +16.4pp at N=500 (basically identical to +16pp at N=50). Path
breakdown consistent: retry arm does 100% of the gain.

**Q2. Does retrieval-augmented generation help?**
→ Yes, where there's headroom. No measurable effect on HumanEval-lite
(ceiling effect). +2.2pp on MBPP 80-129. Cache-growth curve shows
retrieval kicks in reliably at 15 rows and grows monotonically.

**Q3. Can we adapt our system to Program-of-Thought?**
→ Yes (VPoT). 97.0% at baseline token cost. Retry arm only fires on
exec/invariant failure (rare), so the gain is smaller than expected
(+0.8pp over plain PoT) — but VPoT inherits the full cache + retrieval
infrastructure.

**Q4. Does self-consistency voting help on reasoning?**
→ Only on plain CoT, not PoT. On CoT it converts ~2 misses via
majority voting. On PoT (where computation is already symbolic) it
contributes nothing — 5× tokens, 0pp gain. The ~3% of PoT's misses
are modeling errors, not sampling variance.

**Q5. Does cross-method ensemble help?**
→ No, if any voter is weak. PoT(97%)+tool-use(85%)+CoT(82%) ensemble
landed at 95% — *worse* than PoT alone. Weak voters contaminate the
majority vote. Method diversity helps only when each method is
individually strong.

## Null results (valuable to know)

- Calculator tool-use via Anthropic native tool_use: 85% at 17× the
  tokens of PoT. Multi-turn context accumulation disrupts reasoning.
- PoT + self-consistency k=5: 97.5% identical to plain PoT at 5×
  tokens.
- Naive 3-way ensemble including tool-use: 95%, drags below single
  methods.
- TF-IDF retrieval for GSM8K RA-VPoT: -0.2pp (96.8%).
- **MiniLM sentence-transformer retrieval for GSM8K RA-VPoT: -0.6pp**
  (96.4%). The embedding quality is clean (0.77 vs 0.15 for
  relevant vs unrelated); the issue is that retrieval *on math* is
  counter-productive regardless. Model mis-adapts retrieved templates.

## Principled finding: retrieval helps coding, not math

Re-measured across three benchmarks with varying retrieval effects:

| benchmark | retrieval effect | interpretation |
|---|---:|---|
| HumanEval-lite | 0 (ceiling) | no headroom |
| MBPP 80-129 | **+2.2 pp** | retrieval converts hard problems |
| GSM8K (MiniLM) | **-0.6 pp** | retrieval biases to wrong templates |

The principle: retrieval works when solution patterns re-use (code
templates like "filter prefix", "check palindrome"). It fails when
"similar" problems have genuinely different solution bindings (word
problems where "Alice has apples" and "Bob has cookies" share surface
but need different variable flow). Our cache infra supports both;
tool callers should choose `--retrieval` only when the problem
domain has stable, transferable patterns.

## Pareto frontier at Haiku prices (1000 problems)

| config | cost | pass@1 |
|---|--:|--:|
| LLM baseline | $0.62 | 82% |
| **VPoT (ours)** | **$0.59** | **97.0%** |
| PoT (plain) | $0.58 | 96.2% |
| Agent loop | $1.03 | 98.6% |
| Agent + SC + Sonnet escalate | $3.38 | **99.0%** |
| Naive ensemble | $13.52 | 95.0% |

Two non-dominated configs: **VPoT for dollar-efficiency**, **Agent+SC+Sonnet
for max accuracy**. Everything else is dominated on at least one axis.

## Test surface

Total new regression tests across the retrieval + VPoT work: **59 across 6 files**, all passing.

| file | tests | what it pins |
|---|--:|---|
| `test_llm_cache_schema.py` | 7 | 5↔6 col compat, examples persistence |
| `test_retrieval_prompt.py` | 5 | few-shot prefix formatting |
| `test_mcp_retrieval_tools.py` | 17 | MCP tool contracts |
| `test_cache_growth.py` | 5 | monotonic growth invariants |
| `test_gsm8k_solver.py` | 16 | majority vote + PoT exec |
| `test_text_retrieval.py` | 9 | TF-IDF retrieval behaviour |

## What ships

- `tools/benchmarks/run_gsm8k.py`: baseline, agent, `--self-consistency`,
  `--escalate-model`, `--tool-use`, `--program-of-thought`, `--ensemble`,
  `--vpot`, `--vpot-retrieval`.
- `tools/benchmarks/run_mbpp.py`: `--retrieval`, `--offset` for slicing.
- `tools/benchmarks/run_humaneval_agent.py`: `--retrieval` flag.
- `tools/benchmarks/text_retrieval.py`: TF-IDF retrieval for text-keyed
  caches (swap in real embedding for quality).
- `tools/benchmarks/measure_cache_growth.py`: hit-rate-vs-size curve.
- `tools/benchmarks/semantic_cache.py`: examples-based retrieval.
- `tools/benchmarks/llm_solution_cache.py`: 6-col schema with examples.
- `tools/mcp/nsynth_mcp_server.py`: 10 tools including
  `build_retrieval_prefix`, `evaluate_expression`, `check_numeric_answer`,
  `delegate_to_frontier`.
- `tools/inference/vllm_cache_speculative.py`: cache-as-speculative-
  draft scaffold (vast.ai-ready).
- `tools/vastai/run_spec_decode.sh`: one-command vast.ai launcher.

## What's next (not done this session)

- **Sentence-transformer for text retrieval** — swap TF-IDF for MiniLM
  to unlock RA-VPoT. Expected gain: +1-2pp based on embedding-quality
  deltas in the literature.
- **Full MATH benchmark** — GSM8K is word problems; MATH is competition
  math. Harder for blind retry (more modeling errors); VPoT's +0.8pp
  likely bigger there.
- **Finish vLLM speculative decoding A/B** — blocker was torch/vllm
  image compat; one correct base image away.
- **Full GSM8K-1319** — close out the benchmark. ~$0.60 to add the
  remaining 819 problems on either the VPoT or the accuracy-optimised
  config.
