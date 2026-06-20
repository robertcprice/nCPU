# nsynth — Synthesis Capability: Next Steps & The Big Idea

_Authored 2026-06-19, after closing the array→array and string→string gaps._

This doc has two halves:

1. **Near-term hardening** — finish the job: kill the recurring footguns, extend
   the new array engine, make the API surface match the CLI's real capability.
2. **The novel opportunity (the emphasis)** — a **Unified Typed Bottom-Up
   Structural Synthesizer (UTBUS)**: collapse nsynth's scattered exact
   synthesizers into one typed, compositional, proof-carrying engine. This is the
   actual path to "synthesize any kind of program," and it is publishable and
   productizable.

---

## Part 1 — Near-Term Hardening (finish the job)

Ordered by leverage. Items 1–2 are footguns that WILL recur; do them first.

### 1. `solved_cache` disk-fill + load-hang (CRITICAL, recurs)
**Symptom:** `~/.nsynth_solved_programs.json` grew to **13.4 GB** (twice — 6.7 GB,
then 13.4 GB), filled the disk to 97%, and the loader hangs UTF-8-validating the
multi-GB blob → 30 s solve hangs → exit 124. The file is a line format
(`key\tmethod\t0\t0\tcode`), not JSON.
**Root cause (obs 19228):** `solved_cache::max_entries()` defaults to `0` =
eviction disabled, **and** there is no per-entry size cap, **and** the loader has
no fast-fail on oversized/corrupt input.
**Fix:**
- Default `max_entries()` to a real bound (e.g. 50k) with LRU eviction.
- Per-entry size cap (skip caching programs/keys over N KB).
- Loader: if file > M MB or first bytes aren't the expected format, log + skip
  (don't block the solver on a corrupt cache).
- Add a `--cache-clear` / self-heal path.
**Why first:** every successful solve writes to it; left alone it refills the disk
and silently makes all timing/behavior unreliable (it already corrupted one full
audit into looking like solver failures).

### 2. Systemic array-output misfire in scalar-array teachers (latent crashes)
`search_partition_equal_sum` panicked (index OOB) on negative-sum arrays and
mis-fired on `-> [i64]` problems — fixed. But **~11 sibling teachers** in
`search_families.rs` share the identical prologue
(`if param_types != [ParamType::ArrayI64] { return None }`) with no output-type
guard. Any of them can mis-read an array `expected` as an int.
**Fix:** one guard at the common dispatch site (or a shared helper
`is_scalar_output(problem)`), instead of 11 copies. Audit each for `expected_int()`
calls on array-output problems.

### 3. Extend the array engine (`array_transform.rs`)
Currently single-array-input. Add:
- **Multi-array input:** `zipWith(a, b, λx,y. e)` — elementwise binary (add/sub/
  mul/min/max of two arrays).
- **Index-aware maps:** `out[i] = f(i, in[i])` (e.g. `in[i] + i`, `in[i] * i`).
- **Running min/max scans**, **dedup**, **take/drop(n)**, **rotate**.
- **Compositions:** map∘filter, filter∘map (handled naturally once Part 2 lands).

### 4. API surface parity (MCP + HTTP)
`nsynth_serve.rs` and the `ncpu-synth` MCP `synthesize_from_examples` hardcode
`expected: i64` (obs 19184/19218) — so the **string and array capabilities the CLI
now has are invisible to every API client.** Extend the wire schema to accept
float/bool/string/array `expected` (the CLI `--problem-json` parser already does;
reuse it). Without this, downstream users can't reach the new power.

### 5. Probabilistic false-pass (correctness)
`probabilistic.rs` can return `success:true` **without verification** on some array
problems (obs 19195: LIS outputs → Poisson fit → false pass). Gate every
probabilistic return behind `verify_problem_code_strict`. "Success" must always be
proof-carrying — this is the one place it currently isn't.

### 6. Regenerate the full capability audit (clean cache)
The original goal. With the cache hazard fixed (item 1), re-run the 16-domain
matrix with `NSYNTH_CACHE_PATH=` and publish the true boundary. Many prior
"failures" were the cache artifact; quantify what's actually left.

---

## Part 2 — The Novel Opportunity: UTBUS

### The observation

nsynth already contains **five-plus separate exact synthesizers**, each a siloed
enumerator with its own templates, constant discovery, and verify loop:

| Engine | Shape | Examples |
|---|---|---|
| `search_*` families | scalar → scalar | affine, polynomial, clamp, bitwise |
| `string_synth` | string → string | concat/upper/lower/trim/reverse/slice |
| `array_transform` (new) | array → array | map/sort/reverse/scan/filter |
| `enumerative-array` | array → scalar | folds |
| `structured_array` | array+k → scalar | kth-smallest, two-sum, binary-search |

They don't share code, they don't compose, and the gradient path is a sixth,
slower regime bolted alongside. **They are all the same thing:** typed bottom-up
enumeration with proof-carrying acceptance. The verifier
(`verify_problem_code_strict`, examples + holdouts) is already the universal
acceptance oracle.

### The idea

A single **Unified Typed Bottom-Up Structural Synthesizer**:

- **Typed grammar `G[τ]`** — productions indexed by the *output* type τ
  (Scalar i64/f64/bool, Str, Array(T), Pair, Struct, Tree).
- **Bottom-up enumeration** by increasing AST size, the classic BUS algorithm
  (Udupa/Albarghouthi; BUSTLE) — build programs of size *k* from verified
  sub-programs of size < *k*.
- **Observational-equivalence pruning** — key each candidate by its output vector
  on the examples; keep only the cheapest program per behavior. This is what makes
  depth-3+ enumeration tractable instead of exponential.
- **Recursive higher-order combinators** — `map`, `filter`, `fold`, `scan`,
  `zipWith` are first-class productions whose **function arguments are synthesized
  by the same engine on derived sub-examples**:
  - `map(arr, λx.e)` → synthesize `e` with `G[Int→Int]` on `(in[i], out[i])` pairs.
  - `filter(arr, λx.e)` → synthesize predicate with `G[Int→Bool]`.
  - `fold(arr, init, λacc,x.e)` → synthesize accumulator with `G[(Int,Int)→Int]`.
  - `zipWith(a,b, λx,y.e)` → `G[(Int,Int)→Int]`.
- **Proof-carrying acceptance** — the existing verifier. **No program is returned
  that isn't exact on examples + holdouts.** Zero false positives, by construction.

### Why this is the path to "any program"

The siloed engines top out at single-combinator programs. UTBUS gets
**composition for free**: "filter the evens, double them, then sum" is
`fold(map(filter(arr, isEven), ×2), +)` — three combinators, each function-arg
synthesized and verified independently, then composed. That compositional closure
— typed primitives + synthesized λ-bodies + OE pruning — is how you cover an open
ended space of programs with a *finite* grammar and still terminate in
milliseconds on the common cases.

### Architecture

```
              ┌──────────────────────────────────────────┐
   Problem ──▶│  Type lattice dispatch  (τ_in → τ_out)    │
              └──────────────────────────────────────────┘
                               │
                ┌──────────────┴───────────────┐
                ▼                               ▼
        G[Array(Int)]                      G[Int]            … G[Str], G[Bool]
   map/filter/scan/sort/                fold/index/len/
   reverse/concat/zip/take              arith over leaves
                │  λ-body                     │  leaves
                └──────────┬──────────────────┘
                           ▼
              ┌──────────────────────────────────────────┐
              │  Bottom-up enum + observational-equiv     │
              │  table (value-vector → cheapest program)  │
              └──────────────────────────────────────────┘
                           │  first size-bounded candidate
                           ▼
              ┌──────────────────────────────────────────┐
              │  verify_problem_code_strict (ex+holdouts) │  ← proof-carrying
              └──────────────────────────────────────────┘
```

### Implementation phases

- **A — Extract the core.** Pull out the shared engine: type lattice, OE table,
  size-bounded enumerator, verify hook. Re-implement `array_transform` + a scalar
  leaf-expr enumerator on top. **Gate:** parity with the existing siloed synths on
  the current benchmark (no regressions) — this is a safe refactor, not new power.
- **B — Higher-order combinators.** Add `map/filter/fold/scan/zipWith` with
  recursively synthesized λ-bodies. **This is where new programs appear** —
  compositions the siloed engines provably cannot express. Benchmark the delta.
- **C — Cross-type.** array→scalar via `fold` reusing the scalar engine; string↔array
  via `split`/`join`; struct/tree productions. One engine now spans every shape.
- **D — Learned cost model.** Order productions by prior solve success — wire into
  the existing `method_router` / meta-learner so the enumerator tries the
  historically-winning shapes first. Head-to-head vs the gradient path on speed +
  coverage.
- **E — Release.** The benchmark harness already exists (`--per-problem-json`,
  deterministic, paper-ready). Ship the numbers.

### Why it's publishable / valuable

- **Research:** "Typed Compositional Bottom-Up Synthesis with Proof-Carrying
  Acceptance." The combination — type-directed dispatch + recursive higher-order
  synthesis + observational-equivalence + holdout-verified (never-wrong) acceptance
  — as one real, benchmarked system, with an in-codebase comparison against the
  neural/gradient path. Venues: PLDI/OOPSLA/POPL; or NeurIPS/ICLR framed as
  neurosymbolic (exact engine vs gradient engine, same problems).
- **Open source:** a standalone Rust crate (`typed-bus-synth`) for example-driven
  exact synthesis. Differentiator vs FlashFill/most PBE: **typed + compositional +
  cannot return a wrong program.**
- **Product:** the MCP/HTTP server already exists. "Spec-to-code in milliseconds,
  verified" — B2B uses: test-cases→implementation, data-pipeline transforms,
  spreadsheet-formula synthesis (FlashFill-class but typed, compositional, general).
- **The hook for the LLM era:** *synthesis that cannot lie.* Every returned program
  is proof-carrying against held-out examples. That honesty guarantee is the
  selling point against hallucinating code generators.

### First concrete slice (when Part 2 starts)

Build Phase A behind a flag (`--utbus` or `NSYNTH_UTBUS=1`), leaving every legacy
path untouched, exactly as the agentic work was gated. Prove parity on the
benchmark, then turn on Phase B combinators and measure the new coverage. No legacy
regression risk until parity is demonstrated.
```
```
