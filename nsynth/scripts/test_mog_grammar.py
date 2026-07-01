#!/usr/bin/env python3
"""Unit tests for mog_grammar — NO model, NO GPU, NO mlx, NO pytest.

Run:  python3 scripts/test_mog_grammar.py      (exit 0 = all green, nonzero = fail)

Uses a FAKE tokenizer (a {word: id} dict + reverse) and injects numpy as the array
module so the constraint logic is exercised on plain token-id lists.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402  (test-only dependency; NOT imported by mog_grammar)
import mog_grammar as mg  # noqa: E402


# ---------------------------------------------------------------------------
# Fake tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """word -> id dict + reverse. decode joins the words; that is the only text
    model the constraint logic ever sees."""

    def __init__(self, words, eos="<eos>"):
        self.vocab = {}
        for i, w in enumerate(words):
            self.vocab[w] = i
        self.id2tok = {i: w for w, i in self.vocab.items()}
        self.eos_token_id = self.vocab.get(eos)

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids):
        return "".join(self.id2tok.get(int(i), "") for i in ids)

    def id(self, w):
        return self.vocab[w]


_checks = 0


def check(cond, msg):
    global _checks
    _checks += 1
    assert cond, "FAIL: " + msg


# ---------------------------------------------------------------------------
# (A) banned_token_ids — word-boundary + case sensitivity
# ---------------------------------------------------------------------------


def test_banned_token_ids():
    words = [
        # should be banned
        "let", "mut", "None", "::", "impl", "import", "use", " let", "def",
        "println", "print(", "#", ".unwrap", ".iter(", "System.", "console.",
        "lambda", "elif", "pub", "public", "class",
        # should SURVIVE (valid Mog idents / keywords / operators)
        "count", "mutex", "outlet", "delete", "muted", "letter", "fn", "->",
        "none", "some", "ok", "err", "important", "implementation", "publish",
        "NoneType", "used", "reuse", "user",
        "<eos>",
    ]
    tok = FakeTokenizer(words)

    # callable form (id -> str), enumerating the vocab ids
    banned_call = mg.banned_token_ids(lambda i: tok.id2tok[i], tok.id2tok.keys())
    # mapping form (dict id -> str)
    banned_map = mg.banned_token_ids(tok.id2tok)
    check(banned_call == banned_map, "callable and mapping forms must agree")
    banned = banned_call

    must_ban = ["let", "mut", "None", "::", "impl", "import", "use", " let",
                "def", "println", "print(", "#", ".unwrap", ".iter(",
                "System.", "console.", "lambda", "elif", "pub", "public", "class"]
    for w in must_ban:
        check(tok.id(w) in banned, "expected BANNED: %r" % w)

    must_survive = ["count", "mutex", "outlet", "delete", "muted", "letter",
                    "fn", "->", "none", "some", "ok", "err", "important",
                    "implementation", "publish", "NoneType", "used", "reuse",
                    "user", "<eos>"]
    for w in must_survive:
        check(tok.id(w) not in banned, "expected SURVIVE: %r" % w)

    # explicit boundary + case pairs called out in the plan
    check(mg._text_is_banned(" let") and not mg._text_is_banned("letter"),
          "' let' banned, 'letter' not")
    check(mg._text_is_banned("None") and not mg._text_is_banned("none"),
          "'None' banned, 'none' not (case-sensitive)")
    check(not mg._text_is_banned("mutex"), "'mutex' survives (Mog keyword)")
    check(not mg._text_is_banned("->"), "'->' never banned (fn signature op)")
    check(mg._text_is_banned("::") and mg._text_is_banned("a::b"),
          "'::' matches as plain substring")


# ---------------------------------------------------------------------------
# (B) MogStructuralState.feed / would_break
# ---------------------------------------------------------------------------


def test_structural_state():
    # fresh state: a bare close of any kind is impossible
    for close in ("}", ")", "]"):
        st = mg.MogStructuralState()
        check(st.would_break(close), "fresh %r must break" % close)

    # char literals never move depth
    st = mg.MogStructuralState()
    check(not st.would_break("''"), "empty char lit ok")
    check(not st.would_break("'x'"), "char lit ok")

    # matched open then close
    st = mg.MogStructuralState()
    st.feed("fn f() {")
    check(st.brace_depth == 1, "brace depth 1 after 'fn f() {'")
    check(not st.would_break("}"), "close of the open brace is fine")
    st.feed("return 1;")
    check(not st.would_break("}"), "still one open brace")
    st.feed("}")
    check(st.brace_depth == 0, "brace depth back to 0")
    check(st.would_break("}"), "an extra close now breaks")

    # a bracket inside a string literal must NOT move depth
    st = mg.MogStructuralState()
    st.feed('x = "a}b}c"')
    check(st.brace_depth == 0, "brace inside string ignored")
    check(st.in_code_context(), "back in code context after closed string")
    check(st.would_break("}"), "a real close now breaks (string ones did not count)")

    # a // comment swallows brackets until newline
    st = mg.MogStructuralState()
    st.feed("// } ) ]\n")
    check(st.brace_depth == 0 and st.paren_depth == 0 and st.bracket_depth == 0,
          "comment content moves nothing")
    check(not st.in_comment, "comment closed by newline")

    # unclosed string -> not in code context (so no structural masking there)
    st = mg.MogStructuralState()
    st.feed('x = "')
    check(not st.in_code_context(), "inside an open string literal")

    # escaped quote inside string closes correctly, depth unchanged
    st = mg.MogStructuralState()
    st.feed('"a\\"b"')
    check(st.in_code_context(), "string with escaped quote closed")
    check(st.brace_depth == 0, "depth unchanged across string")

    # nested parens/brackets underflow detection
    st = mg.MogStructuralState()
    st.feed("(a[0]")
    check(st.paren_depth == 1 and st.bracket_depth == 0, "paren open, bracket balanced")
    check(st.would_break("))"), "two closes overflow one open paren")
    check(not st.would_break(")"), "one close matches the open paren")


def test_completion_and_fence():
    # unfenced completion
    st = mg.MogStructuralState()
    st.feed("fn f() { return 1; }")
    check(st.completed, "unfenced program complete after fn closes")
    check(st.would_break("x"), "stray content after complete breaks")
    check(not st.would_break("   "), "trailing whitespace after complete is ok")
    check(not st.would_break("\n```"), "closing fence after complete is ok")

    # fenced: completion waits for the closing ```
    st = mg.MogStructuralState()
    st.feed("```mog\nfn f() { return 1; }\n")
    check(st.fence_open and not st.fence_closed, "fence open, not yet closed")
    check(not st.completed, "fenced program not complete until closing fence")
    st.feed("```")
    check(st.fence_closed, "closing fence seen")
    check(st.completed, "fenced program complete after closing fence")


# ---------------------------------------------------------------------------
# (C) logits processor over a fake vocab
# ---------------------------------------------------------------------------


def _proc_vocab():
    # a vocab rich enough for the structural + valid-stream + no-deadlock tests
    words = [
        # program tokens for `fn f() { return 1; }`
        "fn ", "f", "(", ")", " ", "{", "return ", "1", ";", "}",
        # closers / content / whitespace / fence / eos (no duplicate words)
        "}}", "]", "x", "\n", "```", "<eos>",
        # a couple of banned drift tokens
        "let", "mut",
        # a survivor identifier
        "count",
    ]
    return FakeTokenizer(words)


def _mask_of(proc, tok, tokens_ids):
    """Call the processor and return (candidate, set_of_masked_ids)."""
    logits = np.zeros((1, len(tok.vocab)), dtype=np.float32)
    cand = proc(np.array(tokens_ids, dtype=np.int64), logits)
    masked = {i for i in range(len(tok.vocab)) if float(cand[0, i]) == float("-inf")}
    return cand, masked


def test_processor_ban_always():
    tok = _proc_vocab()
    banned = mg.banned_token_ids(tok.id2tok)
    proc = mg.make_logits_processor(tok, banned, xp=np)
    # first call establishes prompt_len; fresh state
    _cand, masked = _mask_of(proc, tok, [tok.id("fn ")])
    check(tok.id("let") in masked, "'let' masked regardless of state")
    check(tok.id("mut") in masked, "'mut' masked regardless of state")
    check(tok.id("count") not in masked, "valid ident survives")
    check(tok.id("fn ") not in masked, "'fn ' survives")
    check(tok.id("{") not in masked, "'{' survives at fresh state")
    # a bare close at depth 0 is structurally masked
    check(tok.id("}") in masked, "'}' masked at depth 0")
    check(tok.id(")") in masked, "')' masked at depth 0")
    check(tok.id("]") in masked, "']' masked at depth 0")
    check(tok.id("}}") in masked, "'}}' masked at depth 0")
    check(tok.id("\n") not in masked, "whitespace survives")
    check(tok.id("<eos>") not in masked, "eos survives")


def test_processor_subset_property():
    """structural_masked(t) => state.would_break(decode(t)) across many states."""
    tok = _proc_vocab()
    banned = mg.banned_token_ids(tok.id2tok)
    proc = mg.make_logits_processor(tok, banned, xp=np)
    # init the precomputed arrays
    _mask_of(proc, tok, [tok.id("fn ")])

    fresh = mg.MogStructuralState()

    depth1 = mg.MogStructuralState()
    depth1.feed("fn f() {")

    in_string = mg.MogStructuralState()
    in_string.feed('x = "abc')  # open string

    complete = mg.MogStructuralState()
    complete.feed("fn f() { return 1; }")
    check(complete.completed, "sanity: complete state is completed")

    for label, st in (("fresh", fresh), ("depth1", depth1),
                      ("in_string", in_string), ("complete", complete)):
        for tid in proc.structural_masked(st):
            txt = tok.decode([tid])
            check(st.would_break(txt),
                  "subset violated in %s: id %d (%r) masked but not would_break"
                  % (label, tid, txt))
    # in_string context masks nothing structurally
    check(len(proc.structural_masked(in_string)) == 0,
          "no structural masking while inside a string literal")


def test_processor_valid_stream_never_blocked():
    tok = _proc_vocab()
    banned = mg.banned_token_ids(tok.id2tok)
    proc = mg.make_logits_processor(tok, banned, xp=np)

    stream = ["fn ", "f", "(", ")", " ", "{", " ", "return ", "1", ";", " ", "}"]
    # sanity: the stream reconstructs the program
    check("".join(stream) == "fn f() { return 1; }", "fake stream reconstructs program")
    stream_ids = [tok.id(w) for w in stream]

    prompt = [tok.id("fn ")]  # arbitrary single prompt token id (reused symbol ok)
    for k in range(len(stream_ids)):
        tokens_ids = prompt + stream_ids[:k]
        _cand, masked = _mask_of(proc, tok, tokens_ids)
        nxt = stream_ids[k]
        check(nxt not in masked,
              "valid-stream step %d: required token %r was masked" % (k, stream[k]))
    # the final '}' brings brace_depth 1 -> 0 and completes the program
    st = mg.MogStructuralState()
    st.feed("fn f() { return 1; }")
    check(st.completed, "program completes after the final '}'")


def test_processor_no_deadlock():
    # (1) pathological all-close vocab: every token underflows, but a survivor
    #     must remain (tier-2 drops the structural mask).
    tok = FakeTokenizer(["}", ")", "]", "}}"], eos=None)
    banned = mg.banned_token_ids(tok.id2tok)  # none banned
    proc = mg.make_logits_processor(tok, banned, xp=np)
    logits = np.zeros((1, len(tok.vocab)), dtype=np.float32)
    proc(np.array([0], dtype=np.int64), logits)  # first call sets prompt_len
    cand = proc(np.array([0, tok.id("}")], dtype=np.int64), logits)
    # after feeding a '}' at depth 0 the state underflows; every token would still
    # underflow -> tier-2 fallback keeps them finite.
    finite = [i for i in range(len(tok.vocab)) if float(cand[0, i]) > float("-inf")]
    check(len(finite) >= 1, "no-deadlock: at least one finite logit survives")

    # (2) extreme: EVERY id banned -> both tiers exhausted -> ORIGINAL logits back.
    tok2 = FakeTokenizer(["let", "mut", "impl"], eos=None)
    banned2 = mg.banned_token_ids(tok2.id2tok)
    check(banned2 == {0, 1, 2}, "sanity: all three ids banned")
    proc2 = mg.make_logits_processor(tok2, banned2, xp=np)
    logits2 = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    proc2(np.array([0], dtype=np.int64), logits2)
    out = proc2(np.array([0, 0], dtype=np.int64), logits2)
    check(out is logits2, "no-deadlock: all-banned returns the ORIGINAL logits object")
    check(np.array_equal(out, logits2), "original logits are unchanged")


def test_processor_prompt_isolation():
    tok = _proc_vocab()
    banned = mg.banned_token_ids(tok.id2tok)
    proc = mg.make_logits_processor(tok, banned, xp=np)
    logits = np.zeros((1, len(tok.vocab)), dtype=np.float32)
    # first call with a multi-token prompt that CONTAINS '{' and '(' ids
    prompt_ids = [tok.id("{"), tok.id("("), tok.id("fn ")]
    proc(np.array(prompt_ids, dtype=np.int64), logits)
    check(proc._holder["prompt_len"] == len(prompt_ids), "prompt_len captured")
    st = proc.state
    check(st.brace_depth == 0 and st.paren_depth == 0,
          "prompt tokens are NOT fed into structural state")
    check(not st.completed and not st.saw_body, "completion state empty after prompt")


def test_completed_masks_complement():
    """In the completed state the processor masks everything but ws/fence/eos."""
    tok = _proc_vocab()
    banned = mg.banned_token_ids(tok.id2tok)
    proc = mg.make_logits_processor(tok, banned, xp=np)
    # drive the processor's own state to completion via a program stream
    proc(np.array([tok.id("fn ")], dtype=np.int64), logits=np.zeros((1, len(tok.vocab)), np.float32))
    stream = ["fn ", "f", "(", ")", " ", "{", " ", "return ", "1", ";", " ", "}"]
    stream_ids = [tok.id(w) for w in stream]
    prompt = [tok.id("fn ")]
    cand = None
    for k in range(1, len(stream_ids) + 1):
        cand = proc(np.array(prompt + stream_ids[:k], dtype=np.int64),
                    np.zeros((1, len(tok.vocab)), np.float32))
    check(proc.state.completed, "processor state reached completion")
    masked = {i for i in range(len(tok.vocab)) if float(cand[0, i]) == float("-inf")}
    check(tok.id("x") in masked, "content token 'x' masked once complete")
    check(tok.id("count") in masked, "identifier masked once complete")
    check(tok.id("\n") not in masked, "whitespace survives when complete")
    check(tok.id("```") not in masked, "closing fence survives when complete")
    check(tok.id("<eos>") not in masked, "eos survives when complete")


# ---------------------------------------------------------------------------


def main():
    tests = [
        test_banned_token_ids,
        test_structural_state,
        test_completion_and_fence,
        test_processor_ban_always,
        test_processor_subset_property,
        test_processor_valid_stream_never_blocked,
        test_processor_no_deadlock,
        test_processor_prompt_isolation,
        test_completed_masks_complement,
    ]
    for t in tests:
        t()
        print("ok  %s" % t.__name__)
    print("\nALL PASSED — %d checks across %d tests" % (_checks, len(tests)))


if __name__ == "__main__":
    main()
