# LLM-Empowerment Lane (LME) — untrusted tiny-LLM NL front door

**Status:** Mode A / Mode A′ / Mode B all built + e2e-verified (4/4 gated tests green).
**Default:** fully **inert** — zero effect on the LLM-free core unless env vars are set.

## Why this exists

The synthesis **engine** is broad (multi-fn, array/scalar, composition, strict
verification). The binding constraint on "universal coding agent" is **NL-comprehension
breadth**: the symbolic comprehension is phrasing-fragile (a long tail — `"sums a list"`
resolves, `"add up all the elements"` historically mis-resolved to scalar `add`).

LME closes the phrasing gap **without weakening the trust guarantee**. A tiny local LLM
(Gemma 4 E4B, MLX 8-bit) is used as an **untrusted translator** that proposes *what* to
synthesize. The LLM-free engine still *produces* the program and **`verify_problem_code_strict`
still gates every result.** The LLM never emits code and can never bypass the verifier.

## The trust property (invariant)

```
LLM proposes  →  {known op | canonical phrasing | I/O examples}
engine produces  →  program (pure LLM-free synthesis)
strict verify  →  examples + holdout/robustness floor  →  accept | reject
```

A wrong LLM proposal can only cause a **rejection** (Mode A/A′) or, in Mode B, a
program that fails the held-out generalization probe — never an unverified accept.
Modes are layered by risk; the riskiest (B) is behind a **separate** env gate.

## The three modes

| Mode | Trigger | LLM proposes | Engine path | Gate |
|------|---------|--------------|-------------|------|
| **A** — op | request maps to one known op | op name (validated vs **live registry** op set) | `synthesize_op_by_name` builds a `Problem` from the op's own `parse_example_cases` → `solve_problem` | `NSYNTH_LOCAL_LLM_URL` |
| **A′** — composition | request is a canonical rephrase of a filter/map/reduce | a canonical NL paraphrase | `synthesize_from_description` (filter/map/reduce composition), recursion-guarded | `NSYNTH_LOCAL_LLM_URL` |
| **B** — out-of-vocab | no known op fits (e.g. composite affine) | 6 I/O **examples** | examples → `Problem` w/ **held-out generalization probe** → `solve_problem` | `NSYNTH_LOCAL_LLM_URL` **and** `NSYNTH_LOCAL_LLM_EXAMPLES` |

`synthesize_via_local_llm` tries **A → A′ → B** in order, returning the first
strict-verified result.

### Mode B held-out guard (the extra safety for the riskiest tier)

The LLM's examples are split: the engine fits the **seed** examples and is then
strict-verified against the **reserved last 2** as `holdouts`. An *inconsistent*
LLM spec (examples that don't describe one function) yields a program that fits
the seed but **fails the holdouts → rejected**. So Mode B accepts only when the
LLM's own examples are self-consistent *and* the synthesized program generalizes
*and* it strict-verifies.

## Auto-fallback (the lane is actually used, not just callable)

`synthesize_from_description` (the agent's NL entry) tries the **symbolic** path
first and **auto-falls-back** to the LME lane on failure. Guarded by a
`thread_local IN_LLM_FALLBACK` cell to prevent recursion, and **inert without the
env** → zero default regression.

## Auto-serving

`ensure_server()` self-starts `python3 -m mlx_lm server` when
`NSYNTH_LOCAL_LLM_AUTOSERVE` is set, so the lane can come up cold.

## How to run

```bash
# 1. serve the model (Apple-Silicon-native MLX; NOT Ollama/GGUF)
python3 -m mlx_lm server --model lmstudio-community/gemma-4-E4B-it-MLX-8bit --port 8765

# 2. point nsynth at it (Mode A + A′)
export NSYNTH_LOCAL_LLM_URL=http://localhost:8765/v1/chat/completions
export NSYNTH_LOCAL_LLM_MODEL=lmstudio-community/gemma-4-E4B-it-MLX-8bit

# 3. (optional, riskier) enable Mode B out-of-vocab example synthesis
export NSYNTH_LOCAL_LLM_EXAMPLES=1

# run the gated e2e suite
cargo test --test local_llm_e2e -- --nocapture --test-threads=1
```

## Verified results (4/4 gated e2e, server up)

| Request | Symbolic alone | LME result (strict-verified) |
|---------|----------------|------------------------------|
| `"add up all the elements of an array"` | mis-resolves → scalar `add` | **Mode A** → `array_sum` fold (`enumerative-array`) |
| `"add up only the positive numbers in the list"` | mis-parses | **Mode A′** → filter+reduce `compose_add_filter_positive` |
| `"triple a number then add five"` | no single op | **Mode B** → `f(x) = (3*x)+5` (`search_polynomial_multi`) |
| auto-fallback on `"add up all the elements"` | fails | falls back → verified fold |

Model-currency note: Gemma 4 E4B is current-era (chosen over stale 2024 models).
It is a *reasoning* model with a `reasoning` field → example generation needs
`max_tokens ≥ 400` headroom (set to 512).

## Files

- `nsynth/src/local_llm.rs` — untrusted translator: `translate_op`, `canonical_rephrase`,
  `propose_examples`/`parse_examples` (Mode B), `ensure_server`, unit tests.
- `nsynth/src/linguigenesis_bridge.rs` — `known_op_names` (live registry menu),
  `synthesize_via_local_llm` (A→A′→B chain), `synthesize_op_by_name`,
  `synthesize_via_llm_examples` (Mode B), auto-fallback wiring.
- `nsynth/tests/local_llm_e2e.rs` — gated e2e (skips without `NSYNTH_LOCAL_LLM_URL`).
