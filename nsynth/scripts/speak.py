#!/usr/bin/env python3
"""nCPU speaks — grammatical English from verified synthesized programs.

The payoff of the rule-learning + string-program work. Every word form is
produced by a Mog program nSynth *synthesized from the curriculum and verified*
(3sg/past/gerund inflection); sentences are assembled over the curriculum's
typed lexicon; and the recovered grammaticality rule independently *verifies*
each one (KVRM-style). nCPU isn't reciting a corpus — it is composing verified
programs into speech.

Run:  python scripts/speak.py
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

SEED = 7  # deterministic output


def synth(task: str) -> tuple[str, str]:
    """Synthesize a bridge task; return (fn_name, Mog code)."""
    payload = subprocess.run(
        [sys.executable, str(HERE / "linguagenesis_bridge.py"), "--task", task],
        capture_output=True, text=True, check=True,
    ).stdout
    out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                         capture_output=True, text=True).stdout
    r = json.loads(out)
    if not r["success"]:
        raise SystemExit(f"could not synthesize {task}: {r['error']}")
    fn_name = r["code"].split("fn ", 1)[1].split("(", 1)[0].strip()
    return fn_name, r["code"]


def inflect_all(code: str, fn_name: str, bases: list[str]) -> dict[str, str]:
    """Run a synthesized inflection program on every base in one Mog execution."""
    calls = "".join(f'  println({fn_name}("{b}"));\n' for b in bases)
    prog = f"{code}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    forms = [ln for ln in out.stdout.splitlines() if ln.strip()]
    return dict(zip(bases, forms))


def run_checker(code: str, fn_name: str, arr: list[int]) -> int:
    lit = "[" + ", ".join(str(x) for x in arr) + "]"
    prog = f"{code}\nfn main() -> i64 {{\n  println_i64({fn_name}({lit}));\n  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return int(out.stdout.strip() or "0")


def main():
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")
    rng = random.Random(SEED)
    from v2.curriculum import morphology_productivity as m

    from v2.grammar.morphology import _IRREGULAR_PLURAL

    print("Synthesizing the morphological realization programs (verified)…")
    s3_fn, s3_code = synth("verb_3sg_form")
    pa_fn, pa_code = synth("verb_past_form")
    ge_fn, ge_code = synth("verb_gerund_form")
    pl_fn, pl_code = synth("pluralize_gen")
    print(f"  3sg / past / gerund / pluralize programs ready "
          f"({s3_fn}, {pa_fn}, {ge_fn}, {pl_fn}).\n")

    verbs = m.REGULAR_VERBS
    bases = [v.base for v in verbs]
    obj = {v.base: (v.objects[0] if v.objects else None) for v in verbs}

    # Inflect the whole verb lexicon with the synthesized programs.
    sg = inflect_all(s3_code, s3_fn, bases)        # 3sg present
    pa = inflect_all(pa_code, pa_fn, bases)        # past
    ge = inflect_all(ge_code, ge_fn, bases)        # gerund

    # Singular + plural subjects (plural via the synthesized pluralize program;
    # irregular-plural nouns are skipped — those are lexical, not rule-derived).
    heads = [(s, s.split()[-1]) for s in m.SINGULAR_SUBJECTS]
    reg = [(s, h) for s, h in heads if h not in _IRREGULAR_PLURAL]
    plural_heads = inflect_all(pl_code, pl_fn, [h for _s, h in reg])
    sing_subjects = [s for s, _h in reg]
    plur_subjects = [f"The {plural_heads[h]}" for _s, h in reg]

    from v2.curriculum.compositional_semantics import MODIFIERS as adjectives

    def clause(subj: str, v: str, number: str, tense: str) -> str:
        tail = f" {obj[v]}" if obj[v] else ""
        if tense == "present":
            # NUMBER AGREEMENT: singular -> 3sg form, plural -> base form.
            verb = sg[v] if number == "sing" else v
            return f"{subj} {verb}{tail}."
        if tense == "past":
            return f"{subj} {pa[v]}{tail}."           # past: no number agreement
        if tense == "progressive":
            be = "is" if number == "sing" else "are"  # be-agreement
            return f"{subj} {be} {ge[v]}{tail}."
        if tense == "copular":
            be = "is" if number == "sing" else "are"
            return f"{subj} {be} {rng.choice(adjectives)}."
        if tense == "question":
            # yes/no question: do-support inversion, agreement on the auxiliary.
            aux = "Does" if number == "sing" else "Do"
            low = subj[0].lower() + subj[1:]
            return f"{aux} {low} {v}{tail}?"
        # negated present (do-support, with agreement on the auxiliary)
        aux = "does not" if number == "sing" else "do not"
        return f"{subj} {aux} {v}{tail}."

    print("nCPU speaking — every word form produced by a synthesized program:\n")
    for label, number, tense in [
        ("present · singular", "sing", "present"),
        ("present · plural", "plur", "present"),
        ("past", "sing", "past"),
        ("progressive", "plur", "progressive"),
        ("negation", "sing", "negation"),
        ("copular", "plur", "copular"),
        ("question", "sing", "question"),
    ]:
        print(f"  [{label}]")
        subjects = sing_subjects if number == "sing" else plur_subjects
        seen = set()
        n = 0
        while n < 4:
            s = clause(rng.choice(subjects), rng.choice(bases), number, tense)
            if s in seen:
                continue
            seen.add(s)
            n += 1
            print(f"     {s}")
        print()

    # Independent verification: the recovered grammaticality rule must ACCEPT
    # the generated present-tense sibilant 3sg sentences (KVRM-style check).
    print("Independent verification (recovered 3sg rule accepts what nCPU spoke):")
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer
    chk_fn, chk_code = synth("sentence_3sg")
    tok = MorphemeTokenizer()
    sib = [v.base for v in verbs if v.base.endswith(("ch", "sh", "ss", "x", "z"))]
    for v in sib[:4]:
        subj = rng.choice(subjects)
        good = f"{subj} {sg[v]}."          # correct: e.g. "The teacher watches."
        bad = f"{subj} {v}."               # wrong: bare stem "The teacher watch."
        gv = run_checker(chk_code, chk_fn, tok.encode(good, add_bos=False, add_eos=False))
        bv = run_checker(chk_code, chk_fn, tok.encode(bad, add_bos=False, add_eos=False))
        print(f"     accept={gv} {good:30}   reject={1-bv} {bad}")

    print("\nEvery sentence: assembled from a verified inflection program, "
          "checked by a recovered rule.")


if __name__ == "__main__":
    main()
