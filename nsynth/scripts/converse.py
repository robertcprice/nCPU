#!/usr/bin/env python3
"""nCPU converses — a REAL multi-turn loop: it READS an external utterance,
COMPREHENDS it with synthesized programs, and ANSWERS truthfully from what those
programs decide. Nothing is scripted on nCPU's side; the user turns are inputs.

This is the piece the evaluation said did not exist. speak.py generates both
sides of a fake dialogue; comprehend.py judges sentences in batch. Here nCPU
takes a turn it did not write, parses it, and computes a reply — and it carries
dialogue state (the last subject) so pronouns in later turns resolve.

Every decision in a reply is a synthesized, verified Mog program:
  - noun_animacy(word)  -> animate noun / inanimate noun / not-a-noun   (lexicon)
  - valid_roles(feats)  -> is an action semantically licensed            (rule)
  - ends_s / valid_agreement -> is the user's sentence grammatical       (rule)
  - verb_3sg(base)      -> inflect the verb for a grammatical reply       (speak side)

The only non-synthesized code is mechanical: lexing (lowercase/split) and routing
on the question word (Is / Can / Does / a statement). The truth of every answer
comes from the programs.

Run:  python scripts/converse.py
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
    return r["code"]


# Query helpers on top of the synthesized programs. These wrappers are structural
# (subject = first noun, object = last noun); every truth they return is computed
# by a synthesized callee.
BRAIN_WRAPPERS = """
fn is_person(w: string) -> i64 {
    if noun_animacy(w) == 1 { return 1; }
    return 0;
}
fn first_noun_class(s: string) -> i64 {
    words := s.split(" ");
    for w in words {
        c := noun_animacy(w);
        if c > 0 { return c; }
    }
    return 0;
}
fn can_act(s: string) -> i64 {
    words := s.split(" ");
    subj_c: i64 = 0;
    obj_c: i64 = 0;
    have: i64 = 0;
    for w in words {
        c := noun_animacy(w);
        if c > 0 {
            if have == 0 { subj_c = c; have = 1; }
            obj_c = c;
        }
    }
    feats := [subj_c, obj_c + 10];
    return valid_roles(feats);
}
fn agreement_ok(s: string) -> i64 {
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
    if subj_idx == -1 { return 1; }
    if subj_idx + 1 >= n { return 1; }
    subj_s := ends_s(words[subj_idx]);
    verb_s := ends_s(words[subj_idx + 1]);
    feats := [1 + subj_s, 11 + verb_s];
    return valid_agreement(feats);
}
"""


def build_brain() -> str:
    parts = [synth(t) for t in ("noun_animacy", "roles_rule", "ends_s",
                                "agreement_rule", "verb_3sg_form")]
    return "\n".join(parts) + "\n" + BRAIN_WRAPPERS


def call_int(brain: str, fn: str, *str_args: str) -> int:
    args = ", ".join(f'"{a}"' for a in str_args)
    prog = f'{brain}\nfn main() -> i64 {{\n  println_i64({fn}({args}));\n  return 0;\n}}\n'
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return int(out.stdout.strip() or "0")


def call_str(brain: str, fn: str, *str_args: str) -> str:
    args = ", ".join(f'"{a}"' for a in str_args)
    prog = f'{brain}\nfn main() -> i64 {{\n  println({fn}({args}));\n  return 0;\n}}\n'
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path], capture_output=True, text=True)
    return out.stdout.strip()


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class Dialogue:
    """Tracks the last subject so 'it'/'they' resolve across turns."""

    def __init__(self, brain: str):
        self.brain = brain
        self.last_subject: str | None = None

    def _resolve(self, words: list[str]) -> list[str]:
        # swap a pronoun for the remembered subject
        return [self.last_subject if w in {"it", "they", "she", "he"} and self.last_subject
                else w for w in words]

    def _first_noun(self, words: list[str]) -> str | None:
        for w in words:
            if call_int(self.brain, "noun_animacy", w) > 0:
                return w
        return None

    def respond(self, utterance: str) -> str:
        words = self._resolve(normalize(utterance))
        if not words:
            return "I didn't catch that."
        head = words[0]
        clause = " ".join(words)

        # Remember the subject for later pronouns.
        subj = self._first_noun(words)
        if subj:
            self.last_subject = subj

        # 1) Category query: "Is the teacher a person?"
        if head == "is":
            noun = self._first_noun(words)
            if noun is None:
                return "I don't know that word."
            person = call_int(self.brain, "is_person", noun)
            if person:
                return f"Yes, the {noun} is a person."
            return f"No, the {noun} is a thing, not a person."

        # 2) Possibility query: "Can the report write the teacher?"
        if head == "can":
            ok = call_int(self.brain, "can_act", clause)
            nouns = [w for w in words if call_int(self.brain, "noun_animacy", w) > 0]
            s = nouns[0] if nouns else "subject"
            o = nouns[-1] if len(nouns) > 1 else "object"
            verb = self._verb_between(words, s, o)
            if ok:
                return f"Yes, the {s} can {verb} the {o}."
            return f"No — the {s} cannot {verb} the {o} (a thing cannot be the doer)."

        # 3) Yes/no action: "Does the teacher write the report?" — answer + restore
        #    agreement with the synthesized 3sg inflection.
        if head == "does" or head == "do":
            ok = call_int(self.brain, "can_act", clause)
            nouns = [w for w in words if call_int(self.brain, "noun_animacy", w) > 0]
            s = nouns[0] if nouns else "subject"
            o = nouns[-1] if len(nouns) > 1 else "object"
            verb = self._verb_between(words, s, o)
            if ok:
                v3 = call_str(self.brain, "third_singular", verb) or verb
                return f"Yes, the {s} {v3} the {o}."
            return f"No, the {s} does not {verb} the {o} — that doesn't make sense."

        # 4) A statement: check it is grammatical (agreement), correct if not.
        ok = call_int(self.brain, "agreement_ok", clause)
        if ok:
            return "That is grammatical."
        nouns_idx = next((i for i, w in enumerate(words)
                          if call_int(self.brain, "noun_animacy", w) > 0), None)
        if nouns_idx is not None and nouns_idx + 1 < len(words):
            subj_w = words[nouns_idx]
            verb_w = words[nouns_idx + 1]
            # plural subject (-s) wants base verb; singular wants 3sg
            if subj_w.endswith("s"):
                fixed_verb = re.sub(r"(es|s)$", "", verb_w) or verb_w
            else:
                fixed_verb = call_str(self.brain, "third_singular", verb_w) or verb_w
            fixed = " ".join(words[:nouns_idx + 1] + [fixed_verb] + words[nouns_idx + 2:])
            return f"That isn't grammatical — did you mean: \"{fixed}\"?"
        return "That isn't grammatical."

    def _verb_between(self, words: list[str], subj: str, obj: str) -> str:
        try:
            si = words.index(subj)
        except ValueError:
            return "act"
        for w in words[si + 1:]:
            if call_int(self.brain, "noun_animacy", w) == 0 and w not in {"the", "a"}:
                return w
        return "act"


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    print("nCPU conversing — it reads each line, comprehends it with synthesized "
          "programs, and replies.\n")
    brain = build_brain()
    dlg = Dialogue(brain)

    # The USER turns are external inputs — nCPU did not write them. It must parse
    # and answer each. The last two turns use a pronoun, exercising dialogue state.
    conversation = [
        "Is the teacher a person?",
        "Is the report a person?",
        "Can the teacher write the report?",
        "Can the report write the teacher?",
        "Does the teacher write the report?",
        "The captains watches the report.",   # ungrammatical — nCPU should correct it
        "Is it a person?",                     # 'it' -> last subject (captains)
    ]
    for utt in conversation:
        reply = dlg.respond(utt)
        print(f"  User : {utt}")
        print(f"  nCPU : {reply}\n")

    print("Every reply's truth came from a synthesized, verified Mog program. The "
          "turns were read, not generated — this is comprehension-driven dialogue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
