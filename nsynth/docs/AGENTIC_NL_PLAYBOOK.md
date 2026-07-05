# Agentic-NL Playbook — the powerful-coding-agent plan

The three-layer plan to make this a powerful *and trustworthy* coding agent, each
layer preserving the zero-false-positive guarantee (the LLM proposes; the verified
engine judges).

## Where we are (measured, 2026-07-05)

Agentic NL = feed the *literal prompt* to `handle_query` with NO examples handed
in, verify against the task's own hidden tests (`src/bin/mbpp_nl_one.rs`,
`nl_diag.rs`).

- HumanEval-NL: **16% solved** (25/154), **WRONG 21%** — after landing doctest
  spec-mining (was 6% / 44%).
- Root of "confidently wrong": *verified against an insufficient spec.* The
  guarantee ("reproduces the examples it was checked against") holds; 2-3 stated
  examples just don't determine the function, so an overfit passes honestly.

## Layer 1 — the consensus trust gate + confidence labeling (LANDED, no model)

`consensus_trust_gate` (`linguigenesis_bridge.rs`), wired in
`synthesize_from_requirement`: after a verified solve over an examples-only spec,
corroborate with an independent candidate:
- `Ambiguous` (divergence witness) → REFUSE (proven underdetermined).
- `Verified` → confident.
- `NoConsensus` (can't corroborate) → **TENTATIVE**: keep the candidate but tag
  `method` `:tentative`; callers present "matches your examples, not independently
  verified — confirm/add one" and never score it confident. Suite 82/82.

**MEASURED (HumanEval-NL, 154):** confidently-WRONG **33 → 21 (−36%)**;
`sum_product` (the canonical overfit) is now correctly TENTATIVE. Cost: 14 solves
labelled tentative (8 were actually right — the honesty tax; 6 were wrong = wrongs
avoided). SOLVED-confident 25 → 14.

**The remaining 21 confidently-wrong leak through UNGATED NL doors** (the gate is
per-door; it only covers `synthesize_from_requirement`):
- **5 — compositional door** (`try_compose_pipeline` → `into_solve_result` in
  `run_synthesis`) is not gated. FIX: run the same gate on the pipeline outcome
  (needs the composed Problem plumbed to the gate).
- **~15 — reference-backed leaks**: `is_examples_only` is FALSE when comprehension
  attaches the resolved op's reference, so the gate skips — but for NL that
  reference is COMPREHENSION-DERIVED (untrusted: if the wrong op was picked, its
  reference + holdouts are self-consistently wrong). FIX: in the NL path, treat a
  comprehension-derived reference as untrusted and run consensus regardless of
  `is_examples_only`.
- **1 — `ExplainCode`**: routed to explanation, not synthesis (routing artifact).

**To reach confidently-wrong ≈ 0:** (i) gate the compositional door; (ii) drop the
reference-trust assumption for the NL path; (iii) still-stronger corroborator for
the thin-spec `NoConsensus` tail. All three are scoped, suite-gated follow-ons.

## Layer 2 — a real planner in the gated LME lane (RUN THIS)

The LME lane already exists (`local_llm.rs` + `synthesize_via_local_llm`,
`docs/LLM_EMPOWERMENT_LANE.md`): the model proposes *what* (op / rephrase / I/O
examples / decomposition); the engine synthesizes + `verify_problem_code_strict`
gates every result. Fully inert unless the env vars are set → zero default
regression. Swap the model to a stronger lightweight planner:

```bash
# SmolLM3-3B (Apache-2.0, strong structured output, MLX-native). Alt: Phi-4-mini.
python3 -m mlx_lm server --model mlx-community/SmolLM3-3B-4bit --port 8765

export NSYNTH_LOCAL_LLM_URL=http://localhost:8765/v1/chat/completions
export NSYNTH_LOCAL_LLM_MODEL=mlx-community/SmolLM3-3B-4bit
export NSYNTH_LOCAL_LLM_EXAMPLES=1     # Mode B: model proposes I/O → engine verifies
export NSYNTH_LOCAL_LLM_AUTOSERVE=1    # optional: self-start the server
```

`synthesize_from_description` auto-falls-back to the lane on symbolic failure. The
model emits NO code that bypasses the verifier — the guarantee is preserved.
Improvement worth adding: schema-constrained decoding (Outlines / mlx grammar) so
the proposed JSON is always well-formed.

## Layer 3 — the verifier-guided planner flywheel (RUN THIS)

The verified engine is a free, correct teacher. Distill a tiny local planner on
data it certifies (PRISM / RSFT). Two data sources:

1. **Benchmark pairs** — every MBPP/HumanEval task IS an (NL prompt → I/O
   examples) pair. Direct SFT of the "propose examples" planner.
2. **Engine-verified pairs (RSFT)** — run any corpus through the engine; keep only
   proposals that LED TO a verified solve.

```bash
# Harvest verified (task -> Mog) pairs in mlx_lm.lora chat format:
NSYNTH_HARVEST=/tmp/train.jsonl bash scripts/run_mbpp_bench.sh /tmp/mbpp_bench.jsonl 8
# (for a PLANNER that proposes I/O, harvest (prompt -> examples) instead of
#  (prompt -> code) — a small addition to mbpp_nl_one on SOLVED.)

# RSFT fine-tune the planner on the verified pairs (Apple-Silicon-native):
python3 -m mlx_lm lora --model mlx-community/SmolLM3-3B-4bit --train \
    --data /tmp/train.jsonl --iters 500 --adapter-path /tmp/planner-adapter

# Point the lane at the fine-tuned adapter → the planner self-improves,
# taught entirely by the verifier, WITHOUT a bigger model.
export NSYNTH_LOCAL_LLM_MODEL=/tmp/planner-adapter
```

## Honest ceiling

Layers 1-3 fix **comprehension** (NL→intent) and **trust** (never confidently
wrong). They do NOT raise the engine's **synthesis ceiling** — arbitrary
DP/graph/parsing algorithms remain out of reach for the no-model synthesizer. This
makes a *powerful, trustworthy function/component synthesizer* ("write me this
function/module, verified") — not yet a "build me an app" agent. That last step
needs either the synthesis reach to grow or a bigger code-gen model in the same
gated-proposer slot (the architecture already supports it).
