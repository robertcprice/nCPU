#!/usr/bin/env python3
"""Recovered rules as deployable constraints (KVRM applied to language).

The bridge proves nSynth can *recover* a curriculum rule as a verified Mog
program. This demo shows the other half: those recovered programs are
**executable**, so they can be deployed as constraints/checkers/generators on
fresh inputs the synthesizer never saw — the program-synthesis analog of KVRM's
verified-registry guarantee, applied to language.

For each rule we (1) synthesize it from the curriculum via the bridge, then
(2) run the recovered program on held-out inputs:
  * pluralize        — GENERATE the plural form (incl. y→ies)
  * 3sg grammar      — ACCEPT/REJECT a sentence (a grammaticality constraint)
  * inference valid  — ACCEPT/REJECT a logical argument

Run:  python scripts/rule_constraint_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYNTH = HERE.parent
BIN = NSYNTH / "target" / "release" / "mog_synth"
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(NSYNTH / "scripts"))
sys.path.insert(0, str(LINGUAGENESIS))


def synthesize(task: str) -> str:
    """Run a bridge task through nSynth; return the recovered Mog program."""
    payload = subprocess.run(
        [sys.executable, str(HERE / "linguagenesis_bridge.py"), "--task", task],
        capture_output=True, text=True, check=True,
    ).stdout
    out = subprocess.run(
        [str(BIN), "--problem-json", "-"], input=payload, capture_output=True, text=True,
    ).stdout
    r = json.loads(out)
    if not r["success"]:
        raise SystemExit(f"synthesis failed for {task}: {r['error']}")
    return r["code"]


def run_main(program: str) -> list[str]:
    """Append no main — program already contains one — and execute it."""
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(program)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def demo_generation():
    print("\n=== 1. GENERATE — recovered `pluralize` on held-out nouns ===")
    code = synthesize("pluralize_gen")
    nouns = ["church", "lady", "fox", "dog", "bench", "story", "boy", "quiz"]
    calls = "".join(f'  println(pluralize("{n}"));\n' for n in nouns)
    prog = f"{code}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    outs = run_main(prog)
    for n, o in zip(nouns, outs):
        print(f"   {n:8} -> {o}")


def _run_array_rule(code: str, fn: str, arr: list[int]) -> int:
    lit = "[" + ", ".join(str(x) for x in arr) + "]"
    prog = f"{code}\nfn main() -> i64 {{\n  println_i64({fn}({lit}));\n  return 0;\n}}\n"
    return int(run_main(prog)[0])


def demo_grammar_check():
    print("\n=== 2. CHECK — recovered 3sg rule as a grammaticality constraint ===")
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer
    code = synthesize("sentence_3sg")
    tok = MorphemeTokenizer()
    # Held-out sentences: sibilant-verb 3sg, grammatical vs bare-stem.
    cases = [
        ("The fox watches.", 1), ("The fox watch.", 0),
        ("The boss misses.", 1), ("The boss miss.", 0),
        ("The judge pushes.", 1), ("The judge push.", 0),
    ]
    for sent, want in cases:
        arr = tok.encode(sent, add_bos=False, add_eos=False)
        got = _run_array_rule(code, "sentence_es_3sg", arr)
        mark = "ok" if got == want else "MISMATCH"
        print(f"   {'accept' if got else 'reject'}  {sent:24} [{mark}]")


def demo_logic_check():
    print("\n=== 3. CHECK — recovered inference rule as a validity constraint ===")
    code = synthesize("formal_logic")
    # feature tokens: assertA=1 assertB=2 assertNeg=3 concludeA=4 concludeB=5 concludeNeg=6
    cases = [
        ("modus ponens  (If A then B; A;  so B )",  [1, 5],       1),
        ("affirm conseq (If A then B; B;  so A )",  [2, 4],       0),
        ("modus tollens (If A then B; ~B; so ~A)",  [2, 3, 4, 6], 1),
        ("deny anteced  (If A then B; ~A; so ~B)",  [1, 3, 5, 6], 0),
    ]
    for name, arr, want in cases:
        got = _run_array_rule(code, "valid_argument", arr)
        mark = "ok" if got == want else "MISMATCH"
        print(f"   {'VALID ' if got else 'invalid'}  {name}  [{mark}]")


def main():
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")
    print("Recovered curriculum rules, deployed on held-out inputs")
    print("(each program was synthesized by nSynth and verified before this run)")
    demo_generation()
    demo_grammar_check()
    demo_logic_check()
    print("\nEvery rule is a small verified program — a deployable constraint, not weights.")


if __name__ == "__main__":
    main()
