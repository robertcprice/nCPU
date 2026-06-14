#!/usr/bin/env python3
"""copular_agreement — a sentence-level grammaticality checker recovered as a
verified Mog program by nsynth's DNF teacher (search_array_dnf).

RULE TO RECOVER (copular BE agreement):
    "The dog is happy."   valid     (singular subject + 'is')
    "The dogs are happy." valid     (plural   subject + 'are')
    "The dogs is happy."  invalid   (plural subject + 'is')
    "The dog are happy."  invalid   (singular subject + 'are')

Target DNF over feature tokens:
    valid iff (singular AND is AND not are) OR (plural AND are AND not is)

PERCEPTION (the feature/encoding layer): every sentence is reduced to a SMALL
int feature array — NOT the whole token stream (noise makes the DNF time out).
We compute three structural feature tokens in Python:
    902  subject-is-plural   (else the subject is singular)
    903  contains the copula "is"
    904  contains the copula "are"
Each sentence becomes the sorted list of feature tokens that are TRUE for it,
so e.g. "The dogs is happy." -> [902, 903] and "The dog is happy." -> [903].
nsynth sees only these id arrays + labels (1=grammatical) and discovers the DNF
on its own — it is never shown the rule.

DATA SOURCE: words come from the LinguaGenesis curriculum only —
  * subjects: NOUN_FRAMES (singular/plural pairs) in determiners_number.py
  * adjectives: COPULAR_FRAMES in basic_clause_frames.py
We pair each subject (singular form and plural form) x {is, are} x adjective and
build "The <subject> <copula> <adjective>." The label is the copular-agreement
oracle, computed structurally — never hand-written per sentence.

Usage:
    python copular_agreement.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
MOG_SYNTH = Path("/Users/bobbyprice/projects/nCPU/nsynth/target/release/mog_synth")

# Feature tokens (the perception layer's vocabulary).
F_SUBJ_PLURAL = 902  # subject is plural (else singular)
F_HAS_IS = 903       # the copula "is" appears
F_HAS_ARE = 904      # the copula "are" appears


def _curriculum():
    sys.path.insert(0, str(LINGUAGENESIS))
    from v2.curriculum import basic_clause_frames as bcf  # type: ignore
    from v2.curriculum import determiners_number as dn  # type: ignore

    return bcf, dn


def _subject_pairs(dn) -> list:
    """(singular, plural) subject pairs from the curriculum NOUN_FRAMES."""
    pairs = []
    seen = set()
    for nf in dn.NOUN_FRAMES:
        if nf.singular in seen:
            continue
        seen.add(nf.singular)
        pairs.append((nf.singular, nf.plural))
    return pairs


def _adjectives(bcf) -> list:
    """Flat list of curriculum copular adjectives (deduped, deterministic)."""
    adjs = []
    seen = set()
    for _subj, adj_list in bcf.COPULAR_FRAMES:
        for a in adj_list:
            if a not in seen:
                seen.add(a)
                adjs.append(a)
    return adjs


def _oracle_label(is_plural_subject: bool, copula: str) -> int:
    """Copular agreement oracle. valid iff (singular & is) or (plural & are)."""
    if is_plural_subject:
        return 1 if copula == "are" else 0
    return 1 if copula == "is" else 0


def _features(is_plural_subject: bool, copula: str) -> list:
    """Encode the structural facts of one sentence to a sorted feature array."""
    feats = []
    if is_plural_subject:
        feats.append(F_SUBJ_PLURAL)
    if copula == "is":
        feats.append(F_HAS_IS)
    elif copula == "are":
        feats.append(F_HAS_ARE)
    return sorted(feats)


def build_rows(max_adjectives: int = 4) -> list:
    """All (feature-array, label, sentence) rows from the curriculum lexicon.

    The feature encoding collapses every sentence into one of only 4 distinct
    (feature-combo, label) classes, so thousands of identical encodings add no
    information and only slow the DNF teacher. We cap the adjective variety
    (`max_adjectives`) to keep a small, balanced train set while still drawing
    every subject (singular + plural) and both copulas from the curriculum.
    """
    bcf, dn = _curriculum()
    subjects = _subject_pairs(dn)
    adjectives = _adjectives(bcf)[:max_adjectives]

    rows = []
    seen_sentences = set()
    for singular, plural in subjects:
        for is_plural, subj in ((False, singular), (True, plural)):
            for copula in ("is", "are"):
                for adj in adjectives:
                    sentence = f"The {subj} {copula} {adj}."
                    if sentence in seen_sentences:
                        continue
                    seen_sentences.add(sentence)
                    feats = _features(is_plural, copula)
                    label = _oracle_label(is_plural, copula)
                    rows.append((feats, label, sentence))
    rows.sort(key=lambda r: r[2])
    return rows


def stratified_split(rows, holdout_every: int = 4):
    """Stratify by (feature-combo, label) so every combination appears in train;
    hold out ~25% (every 4th member of each group)."""
    groups = defaultdict(list)
    for feats, label, sentence in rows:
        groups[(tuple(feats), label)].append((feats, label, sentence))

    examples, holdouts = [], []
    for _key, group in sorted(groups.items()):
        for i, (feats, label, _s) in enumerate(group):
            row = {"inputs": [list(feats)], "expected": label}
            (holdouts if i % holdout_every == holdout_every - 1 else examples).append(row)
    return examples, holdouts


def build_problem() -> dict:
    rows = build_rows()
    examples, holdouts = stratified_split(rows)
    return {
        "name": "copular_agreement_ok",
        "signature": "fn copular_agreement_ok(arr: [i64]) -> i64",
        "examples": examples,
        "holdouts": holdouts,
    }


def call_mog_synth(problem: dict) -> dict:
    """Feed problem-json to mog_synth on stdin; parse the JSON result."""
    proc = subprocess.run(
        [str(MOG_SYNTH), "--problem-json", "-"],
        input=json.dumps(problem),
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"empty mog_synth output; stderr=\n{proc.stderr[-2000:]}")
    # The result JSON is the last JSON object printed.
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return json.loads(out)


def _eval_on_holdout(code: str, fn_name: str, holdouts: list) -> tuple:
    """Run the recovered Mog program on each holdout array; return (correct, n)."""
    correct = 0
    n = 0
    for row in holdouts:
        arr = row["inputs"][0]
        expected = row["expected"]
        arr_lit = "[" + ", ".join(str(x) for x in arr) + "]"
        program = (
            code
            + f"\nfn main() -> i64 {{ println_i64({fn_name}({arr_lit})); return 0; }}\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as tf:
            tf.write(program)
            tmp_path = tf.name
        try:
            proc = subprocess.run(
                [str(MOG_SYNTH), "--run-file", tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            got_lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
            got = None
            for l in reversed(got_lines):
                try:
                    got = int(l)
                    break
                except ValueError:
                    continue
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        n += 1
        if got is not None and got == expected:
            correct += 1
    return correct, n


def main() -> None:
    problem = build_problem()
    pos = sum(1 for e in problem["examples"] if e["expected"] == 1)
    neg = len(problem["examples"]) - pos
    sys.stderr.write(
        f"[copular_agreement] {len(problem['examples'])} train ({pos} pos / {neg} neg), "
        f"{len(problem['holdouts'])} holdout — words+labels from curriculum only\n"
    )

    result = call_mog_synth(problem)
    success = bool(result.get("success"))
    method = result.get("method", "")
    code = result.get("code", "")

    report = {
        "success": success,
        "method": method,
        "code": code,
        "train_n": len(problem["examples"]),
        "holdout_n": len(problem["holdouts"]),
    }

    if success and code:
        correct, n = _eval_on_holdout(code, "copular_agreement_ok", problem["holdouts"])
        report["holdout_correct"] = correct
        report["holdout_accuracy"] = (correct / n) if n else 0.0
        sys.stderr.write(
            f"[copular_agreement] solved by {method}; holdout {correct}/{n} "
            f"= {report['holdout_accuracy']:.3f}\n"
        )
    else:
        sys.stderr.write(
            f"[copular_agreement] NOT solved (method={method!r}). "
            f"Features may be wrong or rule not separable.\n"
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
