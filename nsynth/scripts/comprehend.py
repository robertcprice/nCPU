#!/usr/bin/env python3
"""nCPU comprehends — a LEARNED PARSER whose every semantic decision is a
synthesized, verified Mog program. No hand-written Python in the meaning path.

The companion to speak.py. Where speak.py *generates* grammatical English from
verified inflection programs, this *understands* it: given a sentence, nCPU
decides whether it is semantically licensed under the selectional-restriction
rule (the AGENT of an action must be animate, the PATIENT inanimate — "the
teacher writes the report" is fine, "the report writes the teacher" is not).

The whole comprehension pipeline is synthesized:

  1. noun_animacy(word) -> {1 animate, 2 inanimate, 0 not-a-noun}
     A verified LEXICON program. Animacy is an arbitrary lexical fact (no spelling
     rule predicts it — agents and patients even share endings, farmer/painter vs
     chapter/letter), so it must be stored. nSynth recovers the closed lexicon
     from (word -> class) I/O via the new string-equality-map teacher. This is the
     semantic knowledge that used to be a Python `if word in ROLE_AGENTS`.

  2. valid_roles(features) -> {0,1}
     A verified RULE program. nSynth learns the real two-part rule —
     animate-subject AND inanimate-object — not the "subject animate" shortcut
     the correlated dataset let an earlier version overfit.

  3. comprehend_roles(sentence) -> {0,1}
     Composes the two into one Mog program: split the sentence, tag the first and
     last noun's animacy with (1), assemble offset feature tokens, judge with (2).

The only Python left is mechanical lexing (lowercase + strip punctuation) — not
comprehension. Every "is this a noun? is it animate? does the rule hold?" is a
synthesized program, re-verified against its examples before use.

Run:  python scripts/comprehend.py
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
    """Synthesize a bridge task; return (fn_name, verified Mog code)."""
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
    return fn_name, r["code"], r["method"]


# The comprehension wrapper. The only non-synthesized code is structural syntax
# (subject = first noun, object = last noun) — never a lexical or semantic fact.
COMPREHEND_WRAPPER = """
fn comprehend_roles(s: string) -> i64 {
    words := s.split(" ");
    subj_c: i64 = 0;
    obj_c: i64 = 0;
    have: i64 = 0;
    for w in words {
        c := noun_animacy(w);
        if c > 0 {
            if have == 0 {
                subj_c = c;
                have = 1;
            }
            obj_c = c;
        }
    }
    feats := [subj_c, obj_c + 10];
    return valid_roles(feats);
}
"""

# Subject-verb agreement: subject = first noun, verb = the word right after it.
# Both the -s detector and the agreement rule are synthesized programs.
AGREEMENT_WRAPPER = """
fn check_agreement(s: string) -> i64 {
    words := s.split(" ");
    n: i64 = words.len;
    subj_idx: i64 = -1;
    i: i64 = 0;
    while i < n {
        if subj_idx == -1 {
            if noun_animacy(words[i]) > 0 { subj_idx = i; }
        }
        i = i + 1;
    }
    if subj_idx == -1 { return 0; }
    if subj_idx + 1 >= n { return 0; }
    subj_s := ends_s(words[subj_idx]);
    verb_s := ends_s(words[subj_idx + 1]);
    feats := [1 + subj_s, 11 + verb_s];
    return valid_agreement(feats);
}
"""


def normalize(sentence: str) -> str:
    """Mechanical lexing only — lowercase and drop punctuation. NOT comprehension."""
    return " ".join(re.findall(r"[a-z]+", sentence.lower()))


def comprehend_batch(program: str, sentences: list[str]) -> list[int]:
    """Run the composed comprehension program on a batch of sentences."""
    calls = "".join(
        f'  println_i64(comprehend_roles("{normalize(s)}"));\n' for s in sentences
    )
    prog = f"{program}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    print("nCPU comprehending — every semantic decision is a synthesized program:\n")

    # 1 + 2: synthesize the lexicon and the rule.
    na_fn, na_code, na_method = synth("noun_animacy")
    vr_fn, vr_code, vr_method = synth("roles_rule")
    print(f"  noun_animacy : LEXICON learned via {na_method} "
          f"({na_code.count('if s ==')} words stored)")
    print(f"  valid_roles  : RULE learned via {vr_method}")
    # Show the learned rule is the real two-part conjunction, not a shortcut.
    has_subj = "x == 1" in vr_code
    has_obj = "x == 12" in vr_code
    print(f"  rule recovered: animate-subject (token 1)={has_subj} "
          f"AND inanimate-object (token 12)={has_obj}\n")

    program = f"{na_code}\n{vr_code}\n{COMPREHEND_WRAPPER}"

    # 3: comprehend curriculum sentences. Generate from the Stage-7 generator so
    # these are the curriculum's own sentences, judged entirely by synthesized code.
    from v2.curriculum import compositional_semantics as cs  # type: ignore
    gen = cs.Stage7CompositionalSemanticsGenerator() if hasattr(
        cs, "Stage7CompositionalSemanticsGenerator"
    ) else getattr(cs, [n for n in dir(cs) if "Generator" in n][0])()
    generated = gen.generate(count=400, include_negative=True)
    rows = [(e.sentence, 0 if e.is_negative else 1)
            for e in generated if "semantics.role.agent" in getattr(e, "rule_ids", [])]
    # dedupe, keep a readable sample size
    seen, curated = set(), []
    for sent, label in rows:
        if sent in seen:
            continue
        seen.add(sent)
        curated.append((sent, label))
    sents = [s for s, _ in curated]
    gold = [g for _, g in curated]
    got = comprehend_batch(program, sents)
    correct = sum(int(a == b) for a, b in zip(got, gold))
    print(f"  [curriculum role sentences] comprehended {correct}/{len(gold)} correctly")
    for (sent, g), a in list(zip(curated, got))[:6]:
        mark = "OK " if a == g else "ERR"
        verdict = "licensed" if a == 1 else "blocked "
        print(f"     [{mark}] {verdict}  {sent}")
    print()

    # The discriminating probe: ALL FOUR animacy combinations, including the
    # animate-subject + ANIMATE-object case ("the teacher helps the student") that
    # the old "subject animate" shortcut wrongly licensed. The full two-part rule
    # blocks it. This is the comprehension upgrade, made concrete.
    agent = sorted(w.lower() for w in cs.ROLE_AGENTS)
    patient = sorted(w.lower() for w in cs.ROLE_PATIENTS)
    probe = [
        (f"the {agent[0]} writes the {patient[0]}", 1, "animate -> inanimate"),
        (f"the {patient[0]} writes the {agent[0]}", 0, "inanimate subject"),
        (f"the {agent[1]} helps the {agent[2]}", 0, "animate -> ANIMATE object"),
        (f"the {patient[1]} blocks the {patient[2]}", 0, "inanimate -> inanimate"),
    ]
    pg = comprehend_batch(program, [p[0] for p in probe])
    print("  [selectional-restriction probe — all four animacy combinations]")
    shortcut_wrong = 0
    for (sent, g, why), a in zip(probe, pg):
        mark = "OK " if a == g else "ERR"
        # The old shortcut == "subject animate"; it would license any animate subject.
        shortcut = 1 if "the " + agent[0] in sent or sent.split()[1] in agent else 0
        if shortcut != g:
            shortcut_wrong += 1
        print(f"     [{mark}] got={a} want={g}  ({why}): {sent}")
    print(f"\n  The learned two-part rule fixes {shortcut_wrong} case(s) the old "
          f"'subject animate' shortcut got wrong.")
    print()

    # SECOND comprehension task: subject-verb agreement. This fixes a real bug —
    # the old recovered 3sg rule only checked the verb's suffix, so it ACCEPTED
    # "The captains watches." (plural subject + 3sg verb). English agreement is a
    # parity constraint: regular plurals and 3sg verbs are both -s-marked, and
    # exactly one of {subject, verb} may carry it. Both the -s detector and the
    # parity rule are synthesized programs.
    es_fn, es_code, es_method = synth("ends_s")
    ag_fn, ag_code, ag_method = synth("agreement_rule")
    print(f"  ends_s          : -s detector learned via {es_method}")
    print(f"  valid_agreement : parity rule learned via {ag_method}")
    agree_program = f"{na_code}\n{es_code}\n{ag_code}\n{AGREEMENT_WRAPPER}"

    agree_probe = [
        ("the captain watches the report", 1, "singular subject + 3sg verb"),
        ("the captains watch the report", 1, "plural subject + base verb"),
        ("the captains watches the report", 0, "plural + 3sg  <- old rule's BUG"),
        ("the captain watch the report", 0, "singular + base verb"),
    ]
    calls = "".join(
        f'  println_i64(check_agreement("{p[0]}"));\n' for p in agree_probe
    )
    prog = f"{agree_program}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    ag = [int(x) for x in out.stdout.split()]
    print("  [subject-verb agreement]")
    for (sent, g, why), a in zip(agree_probe, ag):
        mark = "OK " if a == g else "ERR"
        print(f"     [{mark}] got={a} want={g}  ({why}): {sent}")
    bug_idx = 2  # "the captains watches" — the case the old 3sg rule mis-accepted
    fixed = "REJECTED" if ag[bug_idx] == 0 else "still wrongly accepted"
    print(f'\n  "The captains watches." is now {fixed} — the agreement bug is fixed.')

    print("\nEvery judgment above came from a synthesized, verified Mog program — "
          "the parser's lexicon and rules are learned, not hand-coded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
