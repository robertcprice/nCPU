#!/usr/bin/env python3
"""KVRM constraint with a RECOVERED rule — verified-registry guarantee for language.

LinguaGenesis's validator-in-the-loop (v2/curriculum/validator_in_the_loop.py)
is draft → validate → revise: a model emits a candidate, a validator checks it,
and the loop re-rolls on violation so the final output is guaranteed to pass.

Here the *validator is a program nSynth recovered from the curriculum and
verified* (the 3sg grammaticality rule), not hand-written Python. We pair it with
a deliberately NOISY generator that sometimes drops the inflection, and show the
recovered-rule constraint catches every bad draft and only emits accepted output
— the program-synthesis analog of KVRM's verified-registry guarantee, applied to
language.

Run:  python scripts/kvrm_constrained_speak.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYNTH = HERE.parent
BIN = NSYNTH / "target" / "release" / "mog_synth"
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))


def synth(task: str) -> tuple[str, str]:
    payload = subprocess.run(
        [sys.executable, str(HERE / "linguagenesis_bridge.py"), "--task", task],
        capture_output=True, text=True, check=True,
    ).stdout
    out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                         capture_output=True, text=True).stdout
    r = json.loads(out)
    if not r["success"]:
        raise SystemExit(f"could not synthesize {task}: {r['error']}")
    fn = r["code"].split("fn ", 1)[1].split("(", 1)[0].strip()
    return fn, r["code"]


def _run(prog: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    return subprocess.run([str(BIN), "--run-file", path],
                          capture_output=True, text=True).stdout.strip()


def inflect(code: str, fn: str, base: str) -> str:
    return _run(f'{code}\nfn main() -> i64 {{\n  println({fn}("{base}"));\n  return 0;\n}}\n')


class RecoveredRuleValidator:
    """A validator whose decision IS a recovered, verified Mog program.

    Mirrors the shape of LinguaGenesis's tool.validate: `.ok(sentence)` returns
    whether the recovered 3sg grammaticality rule accepts the sentence.
    """

    def __init__(self):
        from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer
        self.fn, self.code = synth("sentence_3sg")
        self.tok = MorphemeTokenizer()

    def ok(self, sentence: str) -> bool:
        ids = self.tok.encode(sentence, add_bos=False, add_eos=False)
        lit = "[" + ", ".join(str(x) for x in ids) + "]"
        out = _run(f"{self.code}\nfn main() -> i64 {{\n  println_i64({self.fn}({lit}));\n  return 0;\n}}\n")
        return out.strip() == "1"


def main():
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")
    rng = random.Random(11)
    from v2.curriculum import morphology_productivity as m

    print("Recovered programs: 3sg inflection (realize) + 3sg rule (validate)\n")
    s3_fn, s3_code = synth("verb_3sg_form")
    validator = RecoveredRuleValidator()

    # Sibilant verbs — the slice the recovered 3sg rule covers (carry <+es>).
    sib = [v.base for v in m.REGULAR_VERBS
           if v.base.endswith(("ch", "sh", "ss", "x", "z"))]
    forms = {b: inflect(s3_code, s3_fn, b) for b in sib}
    subjects = [s for s in m.SINGULAR_SUBJECTS if s.split()[-1] not in {"child", "man", "woman", "mouse"}]

    def noisy_generate(subj: str, base: str, faulty: bool) -> str:
        # A faulty draft drops the 3sg inflection (bare stem); a good draft uses
        # the synthesized correct form.
        verb = base if faulty else forms[base]
        return f"{subj} {verb}."

    print("Draft → validate(recovered rule) → revise.  Faulty drafts are fenced:\n")
    accepted = 0
    refused = 0
    for _ in range(8):
        subj = rng.choice(subjects)
        base = rng.choice(sib)
        rounds = []
        final = None
        for r in range(3):
            faulty = (r == 0) and (rng.random() < 0.6)  # first draft sometimes bad
            cand = noisy_generate(subj, base, faulty)
            ok = validator.ok(cand)
            rounds.append((cand, ok))
            if ok:
                final = cand
                break
        tag = "accept" if final else "REFUSE"
        accepted += bool(final)
        refused += not final
        trace = "  ".join(f'[{"ok" if ok else "rejected"}] {c}' for c, ok in rounds)
        print(f"  {tag}: {trace}")

    print(f"\n{accepted} accepted, {refused} refused. Every accepted sentence passed a "
          "VERIFIED recovered program — the KVRM guarantee, applied to language.")


if __name__ == "__main__":
    main()
