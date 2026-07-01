# Mog Grammar-Constrained Decoding

Lightweight, training-free **decode-time** constraint that forces the local model to
emit valid Mog instead of drifting into Rust (`let mut …`) or Python (`def`, `elif`,
`print(...)`). It is an mlx `logits_processor` that masks the model's per-step logits
so illegal tokens can never be sampled — a decoder-level guarantee, no fine-tune.

- **Constraint logic (pure, model-free):** `scripts/mog_grammar.py`
- **Unit tests (no model, no GPU, no pytest):** `scripts/test_mog_grammar.py`
- **Drop-in OpenAI server:** `scripts/mog_constrained_server.py`

Authoritative Mog syntax source: the Rust lexer/parser in
`src/runtime/mod.rs` (keyword table `1815-1852`, ident charset `1808`).

---

## (a) Architecture — three constraint layers

The processor runs once per decode step. Each step it receives `(tokens, logits)`
where `logits` has shape `(1, V)`, and it returns modified logits. Illegal tokens are
set to `-inf`, which reliably excludes them under both argmax (`temp=0`) and
categorical sampling.

**Layer 1 — Static drift-ban (O(1)/step).**
A set of token ids whose decoded text *is / contains* a non-Mog keyword as a whole
word (`let`, `mut`, `def`, `println`, …) is precomputed **once** over the vocabulary
(`banned_token_ids`) and applied as a constant additive `-inf` mask every step. No
per-step work beyond one vectorized add.

**Layer 2 — Structural bracket-underflow mask (conservative).**
A live `MogStructuralState` tracks brace/paren/bracket depth (literal- and
comment-aware — brackets inside `"…"`, `'…'`, `//…` do **not** move depth). Each
step, a candidate token is masked if appending its text would drive any depth below
zero (a closing `}` / `)` / `]` with no matching opener). This uses precomputed
per-token running-minimum depth deltas plus three live integer counters — **no
per-step re-parse**. It is a proven **subset** of `MogStructuralState.would_break`
(the tests enforce `structural_masked(t) ⇒ would_break(decode(t))`), so it can never
mask a token that valid Mog would legitimately need. Tokens containing a quote or
comment-starter are conservatively marked non-simple and are **never** structurally
masked (we never over-mask).

**Layer 3 — Fence-keyed completion mask.**
Once the top-level `fn` has closed (all depths back to 0 after a body was seen) and,
when the output is fenced, the closing ```` ``` ```` has been emitted, the program is
`completed`. After that, every non-whitespace / non-EOS token is masked so generation
stops cleanly instead of appending a second stray program. Fence-keying (rather than
pure depth) lets a legitimate second top-level `fn` live inside one fenced block.

---

## (b) The ban list and the single boundary predicate

`BANNED_WORDS` is one flat, **CASE-SENSITIVE** list evaluated by one predicate
(`_pattern_hits`). Two entry shapes, one rule:

```
# word-form (alnum on both edges  -> word-boundary enforced on BOTH sides)
let  mut  def  elif  lambda  println  pub  public  class  use  impl  import  None
# symbol / phrase-form (a non-word edge char -> that edge matches as a plain substring)
print(   ::   #   .unwrap   .iter(   System.   console.
```

**Boundary predicate** (`_wc(c)` mirrors the Mog ident charset `mod.rs:1808`:
`c.isascii() and (c.isalnum() or c == "_")`): a pattern `P` hits text `T` at index
`i` iff the slice matches **and** each edge whose *pattern* char is a word char sits
on a word boundary in `T`. Symbol/phrase edges (`::`, `#`, `.`, `(`) have no word
boundary and match as plain substrings.

**Why case-sensitivity is load-bearing.** Lowercase `none`, `some`, `ok`, `err` are
*real Mog keywords* (`mod.rs:1834-1837`) and MUST survive; only capitalized `None`
(the Rust/Python sentinel) is banned. We therefore **never** `casefold()`.

**Why word boundaries are load-bearing.** Banning the bare word `let`/`mut`/… must
not kill valid identifiers that merely *contain* those letters. The predicate lets
all of these survive:

| survives (identifier / keyword / operator) | banned (drift) |
|---|---|
| `outlet` `mutex` `muted` `letter` `delete` | ` let` `let;` `mut` `def` |
| `important` `implementation` `publish` | `impl` `import` `pub` `public` |
| `NoneType` `none` `some` `ok` `err` `fn` | `None` `None(` `::` `println` |
| `->` (fn-signature arrow — never a substring of any pattern) | `.iter(` `.unwrap` |

Decode-based enumeration (`tokenizer.decode([id])`, not `convert_ids_to_tokens`) is
used so the real text — including the BPE leading-space marker (` let`) — is what the
predicate sees.

---

## (c) Accepted subset narrowing (honest scope)

This is a **lightweight** constraint, not a full CFG. Banning some tokens outright
narrows the accepted language to the **single-function value / string / array front
door** we actually generate today. Specifically:

- `impl` bans Mog `impl` blocks (`mod.rs:2060`).
- `import` bans Mog `import` declarations (`mod.rs:2137`).
- `::` bans the enum-construct form `Name::Variant`.
- `use` is banned as a bare word (it is *not* a Mog keyword; subsumes any `use std::…`
  drift at the cost of one rare identifier). We ban the **word** `use`, never the
  phrase `use std`.

These forms are unreachable through the current front door, so the narrowing is sound
here. The word-boundary rule still protects `implementation` / `important` /
`publish`, and `->` is never touched. If/when the front door grows to emit impl
blocks, imports, or enum constructors, those entries must be revisited (see the
upgrade path in §h).

---

## (d) The mlx contract (verified against source)

mlx_lm 0.31.2 at `/opt/homebrew/lib/python3.14/site-packages/mlx_lm`:

- `generate.py:307-322` — `generate_step(..., *, sampler, logits_processors:
  List[Callable[[mx.array, mx.array], mx.array]], ...)`. Signature is exactly
  `(tokens, logits) -> logits`.
- `generate.py:407` — `logits = logits[:, -1, :]` → **logits shape `(1, V)`**.
- `generate.py:408-416` — processors run **before** the sampler; setting entries to
  `-inf` reliably excludes them.
- `sample_utils.py:10` — `make_sampler(temp=0.0, …)`; `temp==0 → argmax`, else
  categorical. `sample_utils.py:111-112` confirms the 2-D `[:, idx]` additive-mask
  pattern used by mlx's own reference processor.

**CORRECTION A (first-call tokens).** The prefill loop feeds all-but-the-last prompt
token straight through the model and never through the `tokens` accumulator, so on
the **first** processor call `tokens.shape[0] == 1` (only the last prompt token) for
any multi-token prompt. We therefore **capture `prompt_len = tokens.shape[0]` on the
first call** and decode `tokens[prompt_len:]` thereafter — we **never** hardcode
`prompt_len = len(chat_template_ids)`. The structural state is advanced only by the
newly generated delta (`decoded[len(prev):]`), with a full rebuild fallback if a
BPE re-segmentation makes the decode non-monotonic.

The mask is applied additively (`candidate = logits + mask`, broadcasting
`(1,V)+(V,)`) rather than by in-place index assignment, which keeps the module free of
any mlx-specific mutation API and lets the unit tests inject `xp=numpy`.

---

## (e) The no-deadlock invariant

Generation must never hang. The processor uses a tiered fallback: if masking would
make **every** logit `-inf`, it first drops the structural mask (ban-only); if the
ban mask alone still leaves nothing finite, it returns the **ORIGINAL** unmasked
`logits`. Whitespace and EOS normally survive every step, so this is a guard, not a
hot path. A unit test drives a pathological all-close vocabulary and asserts a finite
survivor always remains, plus the extreme where even whitespace is banned returns the
original object unchanged.

---

## (f) How the Rust caller integrates — zero Rust changes

The Rust repair loop (`src/local_llm.rs`) POSTs the standard OpenAI body
`{model, messages:[system,user], temperature, max_tokens}` to `NSYNTH_LOCAL_LLM_URL`
and reads `choices[0].message.content` (falling back to `.reasoning`). Reachability is
checked by stripping `/chat/completions` from the URL and issuing `GET <base>/models`.

This server implements exactly that contract:

- `GET  /v1/models`           → `{"object":"list","data":[{"id":<model>,…}]}`
- `POST /v1/chat/completions` → `{choices:[{message:{role:"assistant",content:<text>}}], usage:{…}}`

`content` is always populated. So integration is a single env var — **the full
chat-completions URL**, matching how the Rust caller uses it verbatim:

```bash
export NSYNTH_LOCAL_LLM_URL="http://127.0.0.1:8765/v1/chat/completions"
export NSYNTH_LOCAL_LLM_MODEL="<model-path-or-name>"   # echoed back in responses
```

No Rust file is touched. The caller cannot tell the difference between this server and
`mlx_lm.server`, except that every returned program is now valid Mog.

---

## (g) Run and test commands

**Unit tests (no model, no GPU, no mlx, no pytest):**

```bash
python3 scripts/test_mog_grammar.py          # exit 0 on success
python3 -c "import sys; sys.path.insert(0,'scripts'); import mog_grammar"   # imports w/o mlx
```

**Wiring check (proves grammar↔server integration, still no model/GPU):**

```bash
python3 scripts/mog_constrained_server.py --check --model /dev/null
# -> OK: wiring valid (model NOT loaded)
```

**Serve (loads the model ONCE; needs the GPU/mlx — run only when the harvest is idle):**

```bash
python3 scripts/mog_constrained_server.py --model <mlx-model-path> --port 8765
# then, in the Rust process:
export NSYNTH_LOCAL_LLM_URL="http://127.0.0.1:8765/v1/chat/completions"
```

Sanity once live: `GET /v1/models` returns the model id; a Mog prompt returns fenced
Mog in `choices[0].message.content`; `let` / `mut` / `def` / `println` never appear;
`->`, `none`, `mutex`, and ordinary identifiers survive.

---

## (h) Measuring the lift — constrained vs unconstrained valid-Mog rate

The point of this layer is a higher **single-shot valid-Mog rate** (fraction of
first-try generations that parse as Mog, before any repair round). Measure it as an
A/B over the same prompts, changing only which server `NSYNTH_LOCAL_LLM_URL` points
at:

1. **Baseline (unconstrained):** point the env var at the plain `mlx_lm.server` for
   the same model. Run the MBPP driver
   (`scripts/run_mbpp_bench.sh <bench.jsonl> <timeout_s> <limit>`, built on
   `scripts/mbpp_prepare.py`) with the LLM lane enabled, and record two numbers:
   (i) the fraction of raw generations that **parse** as Mog on the first shot, and
   (ii) the end-to-end MBPP solve-rate.
2. **Constrained:** repoint the env var at
   `http://127.0.0.1:8765/v1/chat/completions` (this server), same model, same
   prompts, same seed/temperature. Re-run.
3. **Report the delta:** `Δ valid-Mog-rate = constrained − baseline` is the direct
   effect of the constraint; the solve-rate delta is the downstream effect (fewer
   repair rounds wasted on non-Mog output). Because the constraint is decoder-level,
   the baseline's non-Mog first-shots (Rust/Python drift) should collapse toward zero
   while the solve-rate is non-decreasing.

Keep temperature and `max_tokens` identical across arms; the only independent variable
is the server URL, so any difference is attributable to the constraint.

A cheaper offline proxy (no Rust): capture N raw completions from each arm and pipe
each through the Mog parser (`mbpp_solve_one` / the runtime's `lex`+`parse`) counting
parse-success. The constrained arm should show a strictly higher parse rate with no
loss of solved tasks.

---

## Upgrade path — full CFG / pushdown automaton (xgrammar-class)

This is deliberately a **drift-ban + structural guard**, not a complete grammar. It
does not enforce full Mog syntax (e.g. it will not reject a well-bracketed but
otherwise malformed statement, and a single token that *contains* a quote is left
unmasked structurally). The principled upgrade is a real grammar-constrained decoder:

- Compile the Mog grammar (the `mod.rs` lexer + parser productions) into a
  **pushdown automaton** and, at each step, mask every token that cannot extend a
  valid prefix — the xgrammar / llguidance / GBNF approach.
- That subsumes all three layers here: the drift-ban becomes "no production admits
  `let`", the bracket guard becomes the stack automaton, and completion becomes the
  accept state. It also handles quote-bearing and partially-illegal tokens that the
  conservative structural layer intentionally leaves alone.
- Until then, this lightweight layer removes the dominant, observed failure mode
  (whole-keyword drift to Rust/Python) at O(1)/step and zero training cost.
