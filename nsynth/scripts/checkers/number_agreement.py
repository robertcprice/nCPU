#!/usr/bin/env python3
"""number_agreement — recover subject-verb NUMBER agreement as a verified Mog program.

RULE (the oracle we must rediscover, never shown to the synthesizer):
    "The dog walks."   valid       (singular subject + 3sg verb)
    "The dogs walk."   valid       (plural   subject + base verb)
    "The dogs walks."  invalid     (plural   subject + 3sg  verb)
    "The dog walk."    invalid     (singular subject + base verb)

Target DNF:  valid iff (singular AND 3sg-suffix) OR (plural AND no-3sg-suffix)

DATA: every NounFrame in the LinguaGenesis curriculum (v2/curriculum/
determiners_number.py) gives a singular form, a plural form, a base verb, and the
curriculum's own computed 3sg verb form (NounFrame.verb_3sg via _make_3sg, the
real -s/-es/-ies inflection rule). We pair each subject (singular & plural) with
each verb form (3sg & base) -> 4 sentences/frame -> and label by AGREEMENT. The
words and the inflection come 100% from the curriculum; only the agreement label
is assigned here, and that label IS the rule we are asking nsynth to recover.

PERCEPTION (the small, clean feature array nsynth actually sees):
    902  subject-is-plural          (present iff the subject is the plural form)
    905  verb-carries-3sg-suffix    (present iff the verb is the curriculum's 3sg
                                      form: walks/teaches/tidies, computed from the
                                      sentence's last word, NOT from the label)

We deliberately do NOT dump the morpheme token stream: the plural noun "dogs"
also carries a <+s> suffix token (id 106), which is pure noise that collides with
the verb's own 3sg <+s>. Computing the two structural facts in Python is the
perception layer — it turns a noisy positional problem into a clean 2-feature DNF.

Run:
    python number_agreement.py            # build + synth + holdout, prints JSON summary
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

# Perception feature tokens (kept well clear of the tokenizer's own band 0..107).
SUBJ_PLURAL = 902  # subject is the plural noun form
VERB_3SG = 905     # verb carries the 3rd-person-singular suffix (-s/-es/-ies)


def _noun_frames():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum.determiners_number import NOUN_FRAMES  # type: ignore

    return NOUN_FRAMES


def _last_word(sentence: str) -> str:
    words = re.findall(r"[A-Za-z]+", sentence)
    return words[-1].lower() if words else ""


def _build_examples():
    """Build (feature-array, label, sentence) rows from the curriculum.

    For every NounFrame: pair {singular, plural} subject x {3sg, base} verb form.
    Label = 1 iff the pair AGREES (singular<->3sg, plural<->base).

    The verb-3sg perception feature is recomputed from the sentence's surface form
    (does the last word equal the curriculum's 3sg verb?) so the feature is a
    genuine perception of the sentence, independent of how we labelled it.
    """
    frames = _noun_frames()
    # surface 3sg verb forms, to recognise "is this verb the 3sg form?" from text.
    threesg_forms = {nf.verb_3sg.lower() for nf in frames}

    rows = []
    seen = set()
    for nf in frames:
        base = nf.verb_base.lower()
        threesg = nf.verb_3sg.lower()
        if base == threesg:
            # degenerate frame where 3sg == base (none expected); skip so the two
            # verb forms stay distinguishable.
            continue
        for subject, is_plural in ((nf.singular, False), (nf.plural, True)):
            for verb, is_3sg_form in ((threesg, True), (base, False)):
                sentence = f"The {subject} {verb}."
                if sentence in seen:
                    continue
                seen.add(sentence)
                # AGREEMENT oracle label (this is the rule nsynth must recover):
                label = 1 if (is_plural != is_3sg_form) else 0
                # PERCEPTION recomputed from the surface sentence:
                verb_seen = _last_word(sentence)
                feats = []
                if is_plural:
                    feats.append(SUBJ_PLURAL)
                if verb_seen in threesg_forms:  # perception of the 3sg suffix
                    feats.append(VERB_3SG)
                feats.sort()
                rows.append((feats, label, sentence))
    rows.sort(key=lambda r: r[2])
    return rows


def _stratified_split(rows, holdout_every: int = 4):
    """Split by (feature-combo, label) so every combination appears in TRAIN.

    With only two boolean features there are exactly four feature-combos and the
    rule maps each combo to a fixed label; stratifying guarantees the synthesizer
    sees all four (plural?, 3sg?) cells and isn't asked to extrapolate a cell it
    never trained on.
    """
    groups = defaultdict(list)
    for feats, label, sentence in rows:
        groups[(tuple(feats), label)].append((feats, label, sentence))
    examples, holdouts = [], []
    for _key, group in sorted(groups.items()):
        for i, (feats, label, _s) in enumerate(group):
            row = {"inputs": [list(feats)], "expected": label}
            (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)
    return examples, holdouts


def build_problem():
    rows = _build_examples()
    examples, holdouts = _stratified_split(rows)
    problem = {
        "name": "number_agreement_ok",
        "signature": "fn number_agreement_ok(arr: [i64]) -> i64",
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
    # The synthesizer prints a JSON object; grab the last JSON line if mixed.
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
    """Run the recovered Mog program on each holdout array; return accuracy."""
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
            got_line = proc.stdout.strip().splitlines()
            got = None
            for ln in got_line:
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
    combos = sorted({(tuple(f), l) for f, l, _s in rows})
    sys.stderr.write(
        f"[number_agreement] {n_train} train ({pos} pos / {n_train - pos} neg), "
        f"{n_hold} holdout; feature-combos seen: {len(combos)}\n"
    )
    for feats, label in combos:
        sys.stderr.write(f"    feats={list(feats)} -> label={label}\n")

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
        acc = measure_holdout(code, "number_agreement_ok", problem["holdouts"])
        summary["holdout_accuracy"] = acc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
