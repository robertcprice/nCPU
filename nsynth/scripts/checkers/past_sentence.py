#!/usr/bin/env python3
"""past_sentence — sentence-level PAST-tense grammaticality checker (nsynth).

Mirrors `task_sentence_past` from linguagenesis_bridge.py, but lifted to the
SENTENCE level: instead of synthesizing a bare suffix string, we build full
clause frames from the curriculum ("The student studied."), encode the whole
sentence with the morpheme tokenizer, and keep only the discriminating features.

RULE TO RECOVER (target DNF):
    valid iff <+ied>  OR  (<+ed> AND not y-stem)  OR  <+d>(for e-stems)

PERCEPTION (the feature array fed to nsynth):
    - which past-suffix token is present, from the morpheme tokenizer's
      inflection band: <+ied>=102, <+ed>=105, <+d>=107
    - one synthetic orthographic feature: the verb stem ends in consonant+y (900)
  Subject/determiner/period tokens are dropped as noise (they would make the DNF
  teacher time out). Labels = 1 if grammatical, 0 otherwise.

CURRICULUM GROUNDING:
    - Verb lexicon + correct past forms: morphology_productivity.REGULAR_VERBS
      (study->studied [<+ied>], walk->walked [<+ed>]).
    - Subject frames: real "The <noun>" subjects emitted by the Stage-5
      tense/aspect generator's grammar.tense.past examples.
    - Negatives are the over-regularization / bare errors:
        * consonant+y stem  -> *studyed (base+ed; should be -ied)  [<+ed>, wrong]
        * regular stem       -> *walkd   (base+d;  should be -ed)  [<+d>,  wrong]
        * any stem           -> *walk    (bare; no suffix)         [no suffix]

HONEST SCOPE NOTE: REGULAR_VERBS contains no e-stem verbs whose correct past
tokenizes to <+d>=107 ("liked" -> lik+<+ed>=105, not like+<+d>). So in
curriculum-grounded data the <+d> token only ever appears as the *wrong*
"walkd" error and is therefore a NEGATIVE feature here. The recoverable rule
reduces to:  valid iff <+ied>  OR  (<+ed> AND not y-stem).

Writes problem-json, calls mog_synth --problem-json, recovers the program, and
measures holdout accuracy by running the recovered Mog program on holdout arrays.
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

# Morpheme-tokenizer inflection band (verified): the past-suffix tokens.
TOK_IED = 102  # <+ied>
TOK_ED = 105   # <+ed>
TOK_D = 107    # <+d>
STEM_Y = 900   # synthetic feature: verb stem ends in consonant + y


def _curriculum():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import morphology_productivity as morph  # type: ignore

    return morph


def _tokenizer():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer  # type: ignore

    return MorphemeTokenizer()


def _subject_frames() -> list[str]:
    """Real 'The <noun>' subjects from the curriculum's past-tense generator."""
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import tense_aspect as ta  # type: ignore

    gen = ta.Stage5TenseAspectGenerator()
    generated = gen.generate(count=400, include_negative=True)
    subjects = set()
    for ex in generated:
        if ex.rule_ids == ["grammar.tense.past"] and not ex.is_negative:
            m = re.match(r"(The \w+) \w+\.", ex.sentence)
            if m:
                subjects.add(m.group(1))
    return sorted(subjects) or ["The dog"]


def _cons_y(base: str) -> bool:
    return base.endswith("y") and not base.endswith(("ay", "ey", "oy", "uy", "iy"))


def _suffix_band(tok, sentence: str) -> list[int]:
    ids = tok.encode(sentence, add_bos=False, add_eos=False)
    return [i for i in ids if 100 <= i <= 108]


def _encode(tok, sentence: str, cons_y: bool) -> list[int]:
    """Perception layer: sentence -> small feature array.

    Keep only past-suffix tokens (102/105/107) + the y-stem feature (900)."""
    band = sorted(i for i in _suffix_band(tok, sentence) if i in (TOK_IED, TOK_ED, TOK_D))
    feats = band + ([STEM_Y] if cons_y else [])
    return sorted(feats)


def build_problem(holdout_frac: float = 0.25) -> tuple[dict, dict]:
    """Build the feature-encoded problem-json. Returns (problem, meta)."""
    morph = _curriculum()
    tok = _tokenizer()
    frames = _subject_frames()

    # One (feature-array, label) row per (verb, error-type). We DO NOT collapse
    # to unique feature arrays: keeping per-verb rows gives the DNF teacher more
    # signal and lets us stratify a real holdout (every feature combo present in
    # train AND holdout). We track label consistency per feature array to fail
    # closed if the feature set turns out non-separable.
    rows: list[tuple[list[int], int]] = []
    label_of: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for vi, v in enumerate(morph.REGULAR_VERBS):
        base = v.base
        # Skip e-stems / doubling stems: their orthographic error lives in the
        # stem spelling, not the suffix token (out of scope; see module docstring).
        if base.endswith("e") or base != base.rstrip("e"):
            continue
        cons_y = _cons_y(base)
        frame = frames[vi % len(frames)]

        correct = v.past_regular  # studied / walked
        # over-regularized wrong past for this stem class
        wrong_suffix = base + "ed" if cons_y else base + "d"  # studyed / walkd
        bare = base  # walk / study

        pos_sentence = f"{frame} {correct}."
        wrong_sentence = f"{frame} {wrong_suffix}."
        bare_sentence = f"{frame} {bare}."

        for sentence, label in (
            (pos_sentence, 1),
            (wrong_sentence, 0),
            (bare_sentence, 0),
        ):
            arr = _encode(tok, sentence, cons_y)
            rows.append((arr, label))
            label_of[tuple(arr)].add(label)

    conflicts = sum(1 for labels in label_of.values() if len(labels) > 1)
    if conflicts:
        # Feature set is not separable — drop conflicting arrays and record it.
        bad = {k for k, labels in label_of.items() if len(labels) > 1}
        rows = [(arr, label) for arr, label in rows if tuple(arr) not in bad]

    # Stratify by (feature-combo, label) so every combination is in train; hold
    # out ~holdout_frac.
    groups: dict[tuple, list] = defaultdict(list)
    for arr, label in rows:
        groups[(tuple(arr), label)].append((arr, label))

    examples, holdouts = [], []
    hold_every = max(2, round(1 / holdout_frac))  # ~25% -> every 4th
    for _key, group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for i, (arr, label) in enumerate(group):
            row = {"inputs": [arr], "expected": label}
            (holdouts if i % hold_every == hold_every - 1 else examples).append(row)

    # Guarantee at least one holdout if any group had only one member.
    if not holdouts and examples:
        holdouts.append(examples.pop())

    problem = {
        "name": "valid_past_sentence",
        "signature": "fn valid_past_sentence(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }
    meta = {
        "conflicts": conflicts,
        "distinct_feature_combos": len({tuple(a) for a, _ in rows}),
        "n_rows": len(rows),
    }
    return problem, meta


def call_mog_synth(problem: dict) -> dict:
    proc = subprocess.run(
        [str(MOG_SYNTH), "--problem-json", "-"],
        input=json.dumps(problem),
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout.strip()
    # The CLI prints a JSON object; take the last JSON line if there's preamble.
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"no JSON from mog_synth.\nstdout={out}\nstderr={proc.stderr}")


def measure_holdout(code: str, fn_name: str, holdouts: list[dict]) -> tuple[int, int]:
    """Run the recovered program on each holdout array; return (correct, total)."""
    correct = 0
    for h in holdouts:
        arr = h["inputs"][0]
        expected = h["expected"]
        arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
        program = f"{code}\nfn main() -> i64 {{ println_i64({fn_name}({arr_lit})); return 0; }}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
            f.write(program)
            path = f.name
        try:
            proc = subprocess.run(
                [str(MOG_SYNTH), "--run-file", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            got = None
            for line in reversed(proc.stdout.strip().splitlines()):
                line = line.strip()
                if re.fullmatch(r"-?\d+", line):
                    got = int(line)
                    break
            if got == expected:
                correct += 1
        finally:
            Path(path).unlink(missing_ok=True)
    return correct, len(holdouts)


def main() -> None:
    problem, meta = build_problem()
    n_train = len(problem["examples"])
    n_hold = len(problem["holdouts"])
    pos = sum(e["expected"] for e in problem["examples"])
    sys.stderr.write(
        f"[past_sentence] {n_train} train ({pos} pos / {n_train - pos} neg), "
        f"{n_hold} holdout; {meta['distinct_feature_combos']} feature combos, "
        f"{meta['conflicts']} conflicts\n"
    )

    result = call_mog_synth(problem)
    success = bool(result.get("success"))
    method = result.get("method", "")
    code = result.get("code", "")

    report = {
        "success": success,
        "method": method,
        "train_n": n_train,
        "holdout_n": n_hold,
        "code": code,
    }

    if success and code:
        fn_name = "valid_past_sentence"
        correct, total = measure_holdout(code, fn_name, problem["holdouts"])
        report["holdout_correct"] = correct
        report["holdout_total"] = total
        report["holdout_accuracy"] = (correct / total) if total else 0.0

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
