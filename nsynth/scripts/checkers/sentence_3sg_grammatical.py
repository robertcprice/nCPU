#!/usr/bin/env python3
"""sentence_3sg_grammatical — recover sentence-level 3sg-agreement grammaticality
as a verified Mog DNF program, over the LinguaGenesis morphology stream.

RULE (the oracle we must rediscover, never shown to the synthesizer):
    "The coach sails."   valid    (regular 3sg, -s)
    "The worker washes." valid    (sibilant 3sg, -es)
    "The clerk tidies."  valid     (consonant-y 3sg, -ies)
    "The coach sail."    invalid  (bare stem, no inflection)
    "The clerk studys."  invalid  (-s on a consonant-y stem; should be -ies)
    "The farmer displaies." invalid (-ies on a non-y stem)

Target DNF:  valid iff  <+es>  OR  <+ies>  OR  (<+s> AND NOT sibilant-stem)

This is the GENERAL 3sg rule for ALL verbs, not just sibilants. The <+s> token
(id 106) is ambiguous: it marks a correct regular 3sg ("sails") but ALSO appears
when the tokenizer over-splits a bare sibilant stem ("miss" -> mis + <+s>), and
a wrong "studys" carries <+s> too. So presence of a 3sg suffix alone cannot
separate grammatical from bare/wrong. We expose ONE structural feature — the verb
stem ends in a sibilant — and the DNF teacher learns the rule that accepts every
correct 3sg and rejects every error.

DATA: the curriculum's Stage-3 morphology generator emits positive ("The dog
walks.") and characteristic-error negative ("The dog walk.") 3sg sentences. Words
and labels come 100% from the curriculum; only the agreement-grammaticality label
(== `not is_negative`) drives the synthesis, and that label IS the rule we recover.

PERCEPTION (the small, clean feature array nsynth actually sees):
    100..108  the morpheme tokenizer's inflection-suffix tokens for the sentence
              (<+es>=104, <+ies>=101, <+s>=106, ... — the structural signal)
    901       stem-is-sibilant  (present iff the verb's BASE ends in s/sh/ch/x/z,
              computed from the surface verb, NOT from the label)

We deliberately do NOT dump the whole token stream: subject/determiner/period
tokens are pure noise for the agreement rule and make the DNF time out. Keeping
only the suffix band + one computed structural feature is the perception layer.

Run:
    python sentence_3sg_grammatical.py     # build + synth + holdout, prints JSON
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
MOG_SYNTH = Path("/Users/bobbyprice/projects/nCPU/nsynth/target/release/mog_synth")

STEM_SIBILANT = 901  # synthetic feature: verb base ends in a sibilant (s/sh/ch/x/z)
STEM_Y = 900         # synthetic feature: verb base ends in consonant + y
_SIBILANT_SUFFIXES = ("s", "sh", "ch", "x", "z")


def _cons_y(base: str) -> bool:
    return base.endswith("y") and not base.endswith(("ay", "ey", "oy", "uy", "iy"))


def find_verb_base(word: str, verb_bases: set[str]) -> str:
    word = word.lower()
    if word in verb_bases:
        return word
    if word.endswith("ied"):
        cand = word[:-3] + "y"
        if cand in verb_bases:
            return cand
    if word.endswith("ies"):
        cand = word[:-3] + "y"
        if cand in verb_bases:
            return cand
    if word.endswith("ed"):
        cand = word[:-2]
        if cand in verb_bases:
            return cand
    if word.endswith("es"):
        cand = word[:-2]
        if cand in verb_bases:
            return cand
    if word.endswith("s"):
        cand = word[:-1]
        if cand in verb_bases:
            return cand
    if word.endswith("ing"):
        cand = word[:-3]
        if cand in verb_bases:
            return cand
    for base in sorted(verb_bases, key=len, reverse=True):
        if word.startswith(base):
            return base
    return word


def _curriculum():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import morphology_productivity as morph  # type: ignore

    return morph


def _tokenizer():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer  # type: ignore

    return MorphemeTokenizer()


def _last_word(sentence: str) -> str:
    words = re.findall(r"[A-Za-z]+", sentence)
    return words[-1].lower() if words else ""


def _surface_to_base(morph):
    """Map every surface verb form -> its base, so we can compute the stem feature
    for any sentence (the perception layer's lexicon, animacy-style)."""
    to_base = {}
    for v in morph.REGULAR_VERBS:
        for form in (v.base, v.third_singular, v.past_regular, v.gerund):
            to_base[form.lower()] = v.base.lower()
    return to_base


def encode_sentence(sentence: str, tok, to_base) -> list[int]:
    """Perception: sentence -> small int feature array.

    Keep only inflection-suffix tokens (band 100..108) and add the sibilant-stem
    feature (901) and y-stem feature (900).
    """
    ids = tok.encode(sentence, add_bos=False, add_eos=False)
    feats = sorted(i for i in ids if 100 <= i <= 108)
    verb = _last_word(sentence)
    verb_bases = {b.lower() for b in to_base.values()}
    base = find_verb_base(verb, verb_bases)
    if base.endswith(_SIBILANT_SUFFIXES):
        feats = feats + [STEM_SIBILANT]
    if _cons_y(base):
        feats = feats + [STEM_Y]
    return feats


def build_rows(count: int = 600):
    """Stream Stage-3 3sg sentences -> (feature-array, label, sentence) rows.

    label = 1 if grammatical (not is_negative) else 0. Only the 3sg inflection
    family is kept (past/gerund are separate rules with their own checkers)."""
    morph = _curriculum()
    tok = _tokenizer()
    to_base = _surface_to_base(morph)
    gen = morph.Stage3MorphologyProductivityGenerator()
    generated = gen.generate(count=count, include_negative=True)

    rows, seen = [], set()
    for ex in generated:
        if ex.rule_ids != ["morphology.verb.3sg"]:
            continue
        sentence = ex.sentence.strip()
        if sentence in seen:
            continue
        seen.add(sentence)
        feats = encode_sentence(sentence, tok, to_base)
        label = 0 if ex.is_negative else 1
        rows.append((feats, label, sentence))

    # Inject adversarial negatives: y-stem verb + <+s> suffix (e.g. *replys)
    for v in morph.REGULAR_VERBS:
        base = v.base
        if _cons_y(base):
            wrong_sentence = f"The worker {base}s."
            feats = encode_sentence(wrong_sentence, tok, to_base)
            rows.append((feats, 0, f"adversarial: {wrong_sentence}"))

    rows.sort(key=lambda r: r[2])
    return rows


def stratified_split(rows, holdout_every: int = 4):
    """Split by (has-sibilant-feature, has-y-stem-feature, label) so every combination appears in TRAIN."""
    groups = defaultdict(list)
    for feats, label, sentence in rows:
        key = (STEM_SIBILANT in feats, STEM_Y in feats, label)
        groups[key].append((feats, label, sentence))
    examples, holdouts = [], []
    for _key, group in sorted(groups.items()):
        for i, (feats, label, _s) in enumerate(group):
            row = {"inputs": [list(feats)], "expected": label}
            (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)
    return examples, holdouts


def build_problem(count: int = 600):
    rows = build_rows(count)
    examples, holdouts = stratified_split(rows)
    problem = {
        "name": "sentence_3sg_ok",
        "signature": "fn sentence_3sg_ok(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }
    return problem, rows


def run_mog_synth(problem) -> dict:
    payload = json.dumps(problem)
    proc = subprocess.run(
        [str(MOG_SYNTH), "--problem-json", "-"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout.strip()
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"no JSON from mog_synth.\nstdout={out}\nstderr={proc.stderr}")


def _fmt_arr(arr) -> str:
    return "[" + ", ".join(str(x) for x in arr) + "]"


def measure_holdout(code: str, fn_name: str, holdouts) -> float:
    """Run the recovered program on each holdout array; return accuracy."""
    if not holdouts:
        return 0.0
    correct = 0
    for row in holdouts:
        arr = row["inputs"][0]
        expected = row["expected"]
        program = (
            code
            + "\nfn main() -> i64 {\n"
            + f"  println_i64({fn_name}({_fmt_arr(arr)}));\n"
            + "  return 0;\n}\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as fh:
            fh.write(program)
            tmp = fh.name
        try:
            proc = subprocess.run(
                [str(MOG_SYNTH), "--run-file", tmp],
                capture_output=True,
                text=True,
                timeout=30,
            )
            got = None
            for ln in proc.stdout.strip().splitlines():
                ln = ln.strip()
                if re.fullmatch(r"-?\d+", ln):
                    got = int(ln)
            if got == expected:
                correct += 1
        finally:
            Path(tmp).unlink(missing_ok=True)
    return correct / len(holdouts)


def main() -> None:
    problem, rows = build_problem()
    n_train = len(problem["examples"])
    n_hold = len(problem["holdouts"])
    pos = sum(e["expected"] for e in problem["examples"])
    combos = sorted({(STEM_SIBILANT in f, l) for f, l, _s in rows})
    sys.stderr.write(
        f"[sentence_3sg_grammatical] {n_train} train ({pos} pos / {n_train - pos} neg), "
        f"{n_hold} holdout; (sibilant?, label) combos: {len(combos)}\n"
    )

    result = run_mog_synth(problem)
    success = bool(result.get("success"))
    method = result.get("method", "")
    code = result.get("code", "")

    summary = {
        "success": success,
        "method": method,
        "train_n": n_train,
        "holdout_n": n_hold,
        "code": code,
    }
    if success and code:
        summary["holdout_accuracy"] = measure_holdout(
            code, "sentence_3sg_ok", problem["holdouts"]
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
