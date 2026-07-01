"""Mog grammar-constrained decoding — pure constraint logic (no model, no GPU).

Three layers of decode-time constraint for the Mog language:

  1. Static drift-ban (O(1)/step): a precomputed set of token ids whose decoded
     text contains a non-Mog keyword (``let``/``mut``/``def``/``println`` ...) as a
     WHOLE WORD is masked to -inf every step. Word-boundary aware so valid Mog
     identifiers (``outlet``/``mutex``/``letter``/``important`` ...) survive.

  2. Structural bracket-underflow mask (conservative): a token whose text would
     drive brace/paren/bracket depth below zero (a close with no matching open)
     is masked. Uses precomputed per-token running-minimum depth deltas + three
     live integer counters — no per-step re-parse. Guaranteed subset of
     ``MogStructuralState.would_break``.

  3. Fence-keyed completion mask: once the top-level fn has closed (and, when the
     output is fenced, the closing ``` has been seen), every non-whitespace /
     non-EOS token is masked so generation stops cleanly.

NO-DEADLOCK INVARIANT: if masking would make every logit -inf, the structural
mask is dropped (ban-only); if still all -inf, the ORIGINAL logits are returned
unmasked. Generation must never hang.

Module hygiene: this file imports ONLY the standard library at import time. numpy
and mlx are imported lazily inside ``make_logits_processor`` so the unit tests run
on a box with neither the model nor mlx present.

Authoritative Mog syntax source: nsynth/src/runtime/mod.rs (lexer lex() ~1638,
keyword table 1815-1852, ident charset 1808).
"""

import types

# ---------------------------------------------------------------------------
# 1. Drift-ban word list + boundary predicate
# ---------------------------------------------------------------------------

# FINAL reconciled list. ONE flat list, ONE predicate, CASE-SENSITIVE.
#   * word-form entries (alnum on both edges) -> both-edge word-boundary enforced
#   * symbol/phrase-form entries (a non-word edge char) -> that edge is a plain
#     substring match (a symbol has no "word boundary").
# Case matters: lowercase `none`/`some`/`ok`/`err`/`mutex` are valid Mog keywords/
# identifiers and MUST survive, so we NEVER casefold. `None` (capitalised) is banned.
BANNED_WORDS = [
    # word-form (both edges are word chars -> word-boundary enforced on both sides)
    "let", "mut", "def", "elif", "lambda", "println", "pub", "public",
    "class", "use", "impl", "import", "None",
    # symbol / phrase-form (a non-word edge char -> that edge matches as substring)
    "print(", "::", "#", ".unwrap", ".iter(", "System.", "console.",
]


def _wc(c):
    """True iff ``c`` is a Mog identifier char (mirrors mod.rs:1808)."""
    return c.isascii() and (c.isalnum() or c == "_")


def _pattern_hits(text, pattern):
    """True iff ``pattern`` occurs in ``text`` with word-boundary respected on
    any edge whose pattern-edge char is itself a word char.

    Boundary is enforced ONLY on a word-char edge, so symbol/phrase patterns
    (``::``, ``#``, ``.iter(``, ``print(``) match as plain substrings, while word
    patterns (``let``) never fire inside a larger identifier (``outlet``).
    """
    n = len(text)
    m = len(pattern)
    if m == 0:
        return False
    left_is_word = _wc(pattern[0])
    right_is_word = _wc(pattern[-1])
    start = 0
    while True:
        i = text.find(pattern, start)
        if i < 0:
            return False
        j = i + m
        left_ok = (i == 0) or (not left_is_word) or (not _wc(text[i - 1]))
        right_ok = (j == n) or (not right_is_word) or (not _wc(text[j]))
        if left_ok and right_ok:
            return True
        start = i + 1


def _text_is_banned(text):
    """True iff any BANNED_WORD hits ``text`` under the boundary predicate."""
    for pattern in BANNED_WORDS:
        if _pattern_hits(text, pattern):
            return True
    return False


def banned_token_ids(decode_token, vocab_ids=None):
    """Return the set of vocab ids whose decoded text is/contains a banned word.

    Enumerate the whole vocab ONCE. Precomputed -> O(1)/step static mask.

    ``decode_token`` is a callable ``id -> str`` (use the tokenizer's *decode*, so
    the real text incl. the BPE leading-space marker ` let` is seen — never the raw
    subword form ``▁let``). ``vocab_ids`` is the iterable of ids to enumerate.

    For convenience ``decode_token`` may instead be a mapping ``{id: str}`` (then
    ``vocab_ids`` is not needed).
    """
    if isinstance(decode_token, dict) and vocab_ids is None:
        return {tid for tid, txt in decode_token.items() if _text_is_banned(txt)}
    if vocab_ids is None:
        raise ValueError(
            "vocab_ids (iterable of token ids) is required when decode_token is callable"
        )
    return {tid for tid in vocab_ids if _text_is_banned(decode_token(tid))}


# ---------------------------------------------------------------------------
# 2. Incremental structural tracker
# ---------------------------------------------------------------------------


def _run_scan(st, text, detect_underflow=False):
    """Advance the structural cursor ``st`` over ``text`` (single char scan).

    ``st`` is any object exposing the MogStructuralState scalar fields. Brackets,
    parens, braces and fence backticks are counted ONLY in code context (not inside
    a string/char literal or ``//`` comment) — the #1 soundness edge.

    When ``detect_underflow`` is True the scan returns True as soon as any depth
    goes below zero (an unmatched close); otherwise returns False.
    """
    for ch in text:
        # --- inside a line comment: consumes until newline ---
        if st.in_comment:
            if ch == "\n":
                st.in_comment = False
            st._prev_slash = False
            st._bt_run = 0
            continue
        # --- inside a string literal ---
        if st.in_string:
            if st.escape:
                st.escape = False
            elif ch == "\\":
                st.escape = True
            elif ch == '"':
                st.in_string = False
            continue
        # --- inside a char literal ---
        if st.in_char:
            if st.escape:
                st.escape = False
            elif ch == "\\":
                st.escape = True
            elif ch == "'":
                st.in_char = False
            continue
        # --- code context ---
        # fence backticks (never fed to depth counters)
        if ch == "`":
            st._prev_slash = False
            st._bt_run += 1
            if st._bt_run == 3:
                if not st.fence_open:
                    st.fence_open = True
                elif not st.fence_closed:
                    st.fence_closed = True
                st._bt_run = 0
            continue
        else:
            st._bt_run = 0
        # comment start needs two slashes
        if ch == "/":
            if st._prev_slash:
                st.in_comment = True
                st._prev_slash = False
            else:
                st._prev_slash = True
            continue
        else:
            st._prev_slash = False
        # string / char openers
        if ch == '"':
            st.in_string = True
            continue
        if ch == "'":
            st.in_char = True
            continue
        # depth chars
        if ch == "{":
            st.brace_depth += 1
            st.saw_body = True
        elif ch == "}":
            st.brace_depth -= 1
            if detect_underflow and st.brace_depth < 0:
                return True
        elif ch == "(":
            st.paren_depth += 1
        elif ch == ")":
            st.paren_depth -= 1
            if detect_underflow and st.paren_depth < 0:
                return True
        elif ch == "[":
            st.bracket_depth += 1
        elif ch == "]":
            st.bracket_depth -= 1
            if detect_underflow and st.bracket_depth < 0:
                return True
    return False


def _has_content_after_complete(text):
    """True iff ``text`` holds any char that is not whitespace and not a fence
    backtick (i.e. genuine program content emitted after the program is done)."""
    for ch in text:
        if (not ch.isspace()) and ch != "`":
            return True
    return False


class MogStructuralState:
    """Incremental structural tracker fed the decoded-so-far Mog text."""

    def __init__(self):
        self.brace_depth = 0
        self.paren_depth = 0
        self.bracket_depth = 0
        self.in_string = False
        self.in_char = False
        self.in_comment = False
        self.escape = False
        self.fence_open = False
        self.fence_closed = False
        self.saw_body = False  # True once brace_depth first reaches >= 1
        # internal cursor helpers (persist across feed() calls)
        self._prev_slash = False
        self._bt_run = 0

    def feed(self, text):
        """Advance the state by newly-decoded ``text`` (mutating)."""
        _run_scan(self, text, detect_underflow=False)

    def in_code_context(self):
        return not (self.in_string or self.in_char or self.in_comment)

    @property
    def completed(self):
        """The program is structurally complete.

        Depth-keyed at top level; fence-keyed when the output is fenced (the live
        harvest path) so a legitimate 2nd top-level fn inside the fence does not
        prematurely trip completion — completion waits for the closing ```.
        """
        if not (
            self.saw_body
            and self.brace_depth == 0
            and self.paren_depth == 0
            and self.bracket_depth == 0
        ):
            return False
        if not self.in_code_context():
            return False
        if self.fence_open:
            return self.fence_closed
        return True

    def _snapshot(self):
        return types.SimpleNamespace(
            brace_depth=self.brace_depth,
            paren_depth=self.paren_depth,
            bracket_depth=self.bracket_depth,
            in_string=self.in_string,
            in_char=self.in_char,
            in_comment=self.in_comment,
            escape=self.escape,
            fence_open=self.fence_open,
            fence_closed=self.fence_closed,
            saw_body=self.saw_body,
            _prev_slash=self._prev_slash,
            _bt_run=self._bt_run,
        )

    def would_break(self, next_text):
        """True iff appending ``next_text`` is structurally IMPOSSIBLE for valid
        Mog: it drives a depth below zero (unmatched close), OR the program is
        already complete and ``next_text`` carries real (non-ws/non-fence) content.

        Non-mutating: simulates on a snapshot.
        """
        if self.completed:
            return _has_content_after_complete(next_text)
        return _run_scan(self._snapshot(), next_text, detect_underflow=True)


# ---------------------------------------------------------------------------
# 3. Per-token structural statistics (precomputed once per vocab)
# ---------------------------------------------------------------------------


def _token_stats(text):
    """Return ``(min_brace, min_paren, min_bracket, simple, complete_ok)``.

    * ``simple`` — text contains no string/char literal delimiter or ``//`` (so it
      can be scanned as pure code in isolation); non-simple tokens are NEVER
      structurally masked (conservative: never over-mask).
    * ``min_*`` — the minimum running depth delta reached while scanning the token
      from a clean code context (only <= 0 values are interesting: a leading
      close).
    * ``complete_ok`` — text is empty or made solely of whitespace / fence
      backticks (allowed as trailing content once the program is complete).
    """
    simple = ('"' not in text and "'" not in text and "`" not in text and "//" not in text)
    mb = mp = mk = 0
    if simple:
        b = p = k = 0
        for ch in text:
            if ch == "{":
                b += 1
            elif ch == "}":
                b -= 1
                if b < mb:
                    mb = b
            elif ch == "(":
                p += 1
            elif ch == ")":
                p -= 1
                if p < mp:
                    mp = p
            elif ch == "[":
                k += 1
            elif ch == "]":
                k -= 1
                if k < mk:
                    mk = k
    complete_ok = (text == "") or all(ch.isspace() or ch == "`" for ch in text)
    return mb, mp, mk, simple, complete_ok


# ---------------------------------------------------------------------------
# 4. mlx logits processor
# ---------------------------------------------------------------------------


def make_logits_processor(tokenizer, banned, *, xp=None):
    """Build a fresh ``callable(tokens, logits) -> logits`` for mlx_lm.

    A new closure per call (the server builds one per request) holding the live
    ``MogStructuralState``, the captured ``prompt_len`` and the precomputed
    per-token arrays (built lazily on the first call from ``V = logits.shape[-1]``).

    ``xp`` is the array module: ``None`` -> lazily ``import mlx.core`` inside the
    processor (never at module top). Tests pass ``xp=numpy``. This kwarg is an
    additive, backward-compatible extension of the pinned 2-arg signature.
    """
    banned_set = set(int(b) for b in banned)
    neg = float("-inf")

    holder = {
        "np": None,          # numpy module (lazy)
        "xp": xp,            # array module for the final add (lazy mlx if None)
        "ready": False,      # precomputed arrays built?
        "prompt_len": None,  # captured on first call
        "prev": "",          # decoded-so-far text of the last call
        "state": MogStructuralState(),
        # precomputed arrays (numpy)
        "min_b": None, "min_p": None, "min_k": None,
        "simple": None, "complete_ok": None, "ban_mask": None,
    }

    def _get_np():
        if holder["np"] is None:
            import numpy as _np
            holder["np"] = _np
        return holder["np"]

    def _get_xp():
        if holder["xp"] is None:
            import mlx.core as _mx
            holder["xp"] = _mx
        return holder["xp"]

    def _build(V):
        np = _get_np()
        vocab = tokenizer.get_vocab()  # {token_str: id}
        # Defaults are the stats of the EMPTY string (_token_stats("") ==
        # (0, 0, 0, True, True)) so any vocab id not returned by get_vocab() (an
        # unused "hole" id that decodes to "") is treated as a harmless empty
        # token: simple, complete-ok, zero depth. Prevents over-masking holes.
        min_b = [0] * V
        min_p = [0] * V
        min_k = [0] * V
        simple = [True] * V
        complete_ok = [True] * V
        for _tok, tid in vocab.items():
            tid = int(tid)
            if tid < 0 or tid >= V:
                continue
            txt = tokenizer.decode([tid])
            mb, mp, mk, smp, cok = _token_stats(txt)
            min_b[tid] = mb
            min_p[tid] = mp
            min_k[tid] = mk
            simple[tid] = smp
            complete_ok[tid] = cok
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and 0 <= int(eos) < V:
            complete_ok[int(eos)] = True
        holder["min_b"] = np.array(min_b, dtype=np.int64)
        holder["min_p"] = np.array(min_p, dtype=np.int64)
        holder["min_k"] = np.array(min_k, dtype=np.int64)
        holder["simple"] = np.array(simple, dtype=bool)
        holder["complete_ok"] = np.array(complete_ok, dtype=bool)
        ban_mask = np.zeros(V, dtype=np.float32)
        for tid in banned_set:
            if 0 <= tid < V:
                ban_mask[tid] = neg
        holder["ban_mask"] = ban_mask
        holder["ready"] = True

    def _structural_bad(st):
        """Return a numpy bool array (len V) of structurally-masked ids for ``st``."""
        np = _get_np()
        V = holder["ban_mask"].shape[0]
        if st.completed:
            return ~holder["complete_ok"]
        if st.in_code_context() and not st._prev_slash and st._bt_run == 0:
            worst = np.minimum(
                np.minimum(holder["min_b"] + st.brace_depth, holder["min_p"] + st.paren_depth),
                holder["min_k"] + st.bracket_depth,
            )
            return holder["simple"] & (worst < 0)
        return np.zeros(V, dtype=bool)

    def structural_masked(st):
        """Public (test) helper: set of ids structurally masked for ``st``."""
        np = _get_np()
        return set(int(i) for i in np.where(_structural_bad(st))[0])

    def processor(tokens, logits):
        np = _get_np()
        xp = _get_xp()
        V = int(logits.shape[-1])
        if not holder["ready"]:
            _build(V)

        # --- advance structural state by the newly-decoded delta ---
        if holder["prompt_len"] is None:
            # first call: tokens == last prompt token only (prefill consumed the
            # rest); capture the boundary and decode nothing yet.
            holder["prompt_len"] = int(tokens.shape[0])
        else:
            gen_ids = tokens[holder["prompt_len"]:].tolist()
            decoded = tokenizer.decode(gen_ids) if gen_ids else ""
            prev = holder["prev"]
            st = holder["state"]
            if decoded.startswith(prev):
                st.feed(decoded[len(prev):])
            else:
                # non-monotonic decode (rare BPE re-segmentation): rebuild.
                st = MogStructuralState()
                st.feed(decoded)
                holder["state"] = st
            holder["prev"] = decoded

        st = holder["state"]
        processor.state = st  # expose for tests / introspection

        # --- build the additive mask (numpy) ---
        struct_bad = _structural_bad(st)
        mask_np = holder["ban_mask"] + np.where(struct_bad, neg, np.float32(0.0))

        mask_x = xp.array(mask_np)
        candidate = logits + mask_x

        # NO-DEADLOCK tiered fallback.
        if float(candidate.max()) == neg:
            ban_x = xp.array(holder["ban_mask"])
            candidate = logits + ban_x
            if float(candidate.max()) == neg:
                return logits  # ORIGINAL, unmasked — never hang
            return candidate
        return candidate

    # expose helpers for unit tests (no model required)
    processor.structural_masked = structural_masked
    processor.state = holder["state"]
    processor._holder = holder
    return processor
