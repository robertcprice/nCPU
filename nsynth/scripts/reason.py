#!/usr/bin/env python3
"""nCPU reasons — judges whether a logical argument is VALID, with the semantic
core of the parser synthesized rather than hand-written in Python.

The bridge's task_formal_logic learns the validity *rule* but leans on a Python
`classify()` that does the real understanding: detecting negation and deciding
whether a premise states the conditional's antecedent or its consequent. Here
those two decisions are themselves verified Mog programs:

  - prop_id(clause)      -> a symbol per proposition           (synthesized lexicon)
  - has_negation(clause) -> does the clause carry "not"         (synthesized rule)
  - same_prop(a, b)      -> prop_id(a) == prop_id(b)            (composition)
  - valid_argument(toks) -> is the argument form valid          (synthesized DNF)

So "is the premise the antecedent?" becomes `same_prop(premise, antecedent)` —
program-level proposition identity — and "is it negated?" becomes a synthesized
predicate. The only Python left is mechanical: splitting the sentence into its
grammatical slots (if-clause / premise / conclusion). Validity (modus ponens and
modus tollens valid; affirming-consequent and denying-antecedent invalid) is
decided entirely by synthesized programs over those parsed atoms.

Run:  python scripts/reason.py
"""

from __future__ import annotations

import json
import re
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
    return r["code"], r["method"]


SAME_PROP = """
fn same_prop(a: string, b: string) -> i64 {
    if prop_id(a) == prop_id(b) { return 1; }
    return 0;
}
"""


def run_int(program: str, call: str) -> int:
    prog = f"{program}\nfn main() -> i64 {{\n  println_i64({call});\n  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return int(out.stdout.strip() or "0")


CONNECTIVES = ("thus,", "therefore,", "so,", "hence,", "then,")


def bare(clause: str) -> str:
    """Mechanical lexing: strip connectives + negation wrappers to the bare
    proposition so prop_id can look it up. NOT a semantic decision."""
    c = clause.strip().rstrip(".").strip().lower()
    for conn in CONNECTIVES:
        if c.startswith(conn):
            c = c[len(conn):].strip()
    for wrapper in ("it is not the case that ", "it is not true that ",
                    "it is true that "):
        c = c.replace(wrapper, "")
    c = c.replace(" is not true", "").replace(" does not happen", "")
    return c.strip()


def segment(sentence: str):
    """Split a conditional argument into (antecedent, consequent, premise,
    conclusion). Pure syntax — no understanding of the propositions."""
    parts = [p.strip() for p in sentence.split(".") if p.strip()]
    if len(parts) != 3 or not parts[0].lower().startswith("if "):
        return None
    m = re.match(r"if (.*), then (.*)", parts[0], re.IGNORECASE)
    if not m:
        return None
    return m.group(1), m.group(2), parts[1], parts[2]


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    print("nCPU reasoning — proposition identity and negation are synthesized "
          "programs; validity is judged by them.\n")

    prop_code, prop_method = synth("prop_lexicon")
    neg_code, neg_method = synth("negation_detect")
    arg_code, arg_method = synth("formal_logic")
    print(f"  prop_id        : proposition lexicon via {prop_method} "
          f"({prop_code.count('if s ==')} propositions)")
    print(f"  has_negation   : negation cue via {neg_method} "
          f"-> {neg_code.splitlines()[1].strip()}")
    print(f"  valid_argument : validity rule via {arg_method}\n")

    brain = f"{prop_code}\n{neg_code}\n{arg_code}\n{SAME_PROP}"
    arg_fn = arg_code.split("fn ", 1)[1].split("(", 1)[0].strip()

    def judge(sentence: str) -> int:
        seg = segment(sentence)
        if seg is None:
            return -1
        a_str, _b_str, premise, conclusion = seg
        a_bare = bare(a_str)
        # premise tokens: assertA=1 / assertB=2, assertNeg=3 — all synthesized
        p_is_a = run_int(brain, f'same_prop("{bare(premise)}", "{a_bare}")')
        toks = [1 if p_is_a else 2]
        if run_int(brain, f'has_negation("{premise.lower()}")'):
            toks.append(3)
        # conclusion tokens: concludeA=4 / concludeB=5, concludeNeg=6
        c_is_a = run_int(brain, f'same_prop("{bare(conclusion)}", "{a_bare}")')
        toks.append(4 if c_is_a else 5)
        if run_int(brain, f'has_negation("{conclusion.lower()}")'):
            toks.append(6)
        lit = "[" + ", ".join(str(t) for t in toks) + "]"
        return run_int(brain, f"{arg_fn}({lit})")

    # Judge the curriculum's own conditional arguments.
    from v2.curriculum import formal_logic as fl  # type: ignore
    gen = fl.Stage8FormalLogicGenerator()
    generated = gen.generate(count=400, include_negative=True)
    rows = []
    seen = set()
    for e in generated:
        if not e.sentence.lower().startswith("if "):
            continue
        if e.sentence in seen:
            continue
        seen.add(e.sentence)
        rows.append((e.sentence, 0 if e.is_negative else 1))

    correct = 0
    samples = []
    for sent, gold in rows:
        got = judge(sent)
        if got == gold:
            correct += 1
        if len(samples) < 6:
            samples.append((sent, gold, got))
    print(f"  [curriculum arguments] judged {correct}/{len(rows)} validities correctly")
    for sent, gold, got in samples:
        mark = "OK " if got == gold else "ERR"
        verdict = "VALID  " if got == 1 else "invalid"
        print(f"     [{mark}] {verdict}  {sent}")
    print()

    # The four classic inference patterns, made explicit.
    p = fl.PROPOSITION_PAIRS[0]
    A, B = p.antecedent, p.consequent
    patterns = [
        (f"If {A}, then {B}. {A.capitalize()}. Therefore, {B}.", 1, "modus ponens"),
        (f"If {A}, then {B}. {B.capitalize()} is not true. "
         f"Therefore, {A} is not true.", 1, "modus tollens"),
        (f"If {A}, then {B}. {B.capitalize()}. Therefore, {A}.", 0,
         "affirming the consequent"),
        (f"If {A}, then {B}. {A.capitalize()} is not true. "
         f"Therefore, {B} is not true.", 0, "denying the antecedent"),
    ]
    print("  [the four inference patterns]")
    for sent, gold, name in patterns:
        got = judge(sent)
        mark = "OK " if got == gold else "ERR"
        verdict = "VALID  " if got == 1 else "invalid"
        print(f"     [{mark}] {verdict}  ({name})")

    print("\nValidity above was decided by synthesized programs: proposition "
          "identity (a learned lexicon), negation (a learned rule), and the "
          "validity DNF — no Python classifier in the reasoning path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
