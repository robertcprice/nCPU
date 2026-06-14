#!/usr/bin/env python3
"""Full-grammar validator-in-the-loop: all 4 recovered checkers fence faulty drafts.

Tier D of the rule-compressed memory roadmap. Every checker is a VERIFIED Mog
program nSynth recovered from the LinguaGenesis curriculum — the validator's
decision IS a synthesized program, not hand-written code. This demo exercises the
draft→validate→revise pattern across:

  sentence_3sg           (3sg verb agreement: "The dog walks." ok, "The dog walk." no)
  number_agreement       (subject-verb number: "The dogs walk." ok, "The dogs walks." no)
  copular_agreement      (BE agreement: "The dog is happy." ok, "The dog are happy." no)
  past_sentence          (past-tense well-formedness: "The dog walked." ok, "The dog walk." no)

For each checker, the demo:
  1. Generates a CORRECT sentence from the curriculum.
  2. Introduces an INTENTIONAL error the checker should catch.
  3. Simulates a draft→validate→revise loop: the "model" first drafts the
     incorrect version, the validator rejects it, and the model revises to
     the correct version.

Run:  python scripts/full_grammar_validator_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))

from v2.nsynth_validator import NsynthRuleValidator


def main():
    print("Loading all 4 recovered grammar checkers…")
    checkers = NsynthRuleValidator.all_checkers()
    for c in checkers:
        print(f"  ✓ {c.__class__.__name__:30s}  rule={c.rule_id:40s}  "
              f"{len(c.code):>4}B verified program")

    # ── test sentences (curriculum-generated, with intentional errors) ──
    tests = [
        # 3SG agreement
        {
            "checker": "for_3sg",
            "label": "3sg present tense",
            "correct": "The teacher watches the student.",
            "wrong":   "The teacher watch the student.",        # bare stem, no -es
            "revision": "The teacher watches the student.",
        },
        {
            "checker": "for_3sg",
            "label": "3sg y→ies",
            "correct": "The goat replies quickly.",
            "wrong":   "The goat replys quickly.",             # wrong y→s
            "revision": "The goat replies quickly.",
        },
        # Number agreement
        {
            "checker": "for_number_agreement",
            "label": "plural subject + 3sg verb",
            "correct": "The dogs walk.",
            "wrong":   "The dogs walks.",                      # plural subject, 3sg verb
            "revision": "The dogs walk.",
        },
        {
            "checker": "for_number_agreement",
            "label": "singular subject + bare verb",
            "correct": "The dog walks.",
            "wrong":   "The dog walk.",                        # singular subject, bare verb
            "revision": "The dog walks.",
        },
        # Copular agreement
        {
            "checker": "for_copular_agreement",
            "label": "plural + is",
            "correct": "The dogs are happy.",
            "wrong":   "The dogs is happy.",                   # plural subject, 'is'
            "revision": "The dogs are happy.",
        },
        {
            "checker": "for_copular_agreement",
            "label": "singular + are",
            "correct": "The dog is angry.",
            "wrong":   "The dog are angry.",                   # singular subject, 'are'
            "revision": "The dog is angry.",
        },
        # Past tense
        {
            "checker": "for_past_sentence",
            "label": "bare stem, no past suffix",
            "correct": "The dog walked.",
            "wrong":   "The dog walk.",                        # bare stem, past needed
            "revision": "The dog walked.",
        },
        {
            "checker": "for_past_sentence",
            "label": "y→ied past",
            "correct": "The dog carried the bone.",
            "wrong":   "The dog carryed the bone.",            # over-regularized
            "revision": "The dog carried the bone.",
        },
    ]

    # ── validator map by class name ──
    checker_map = {}
    for c in checkers:
        key = {
            "ThreeSgValidator": "for_3sg",
            "NumberAgreementValidator": "for_number_agreement",
            "CopularAgreementValidator": "for_copular_agreement",
            "PastSentenceValidator": "for_past_sentence",
        }.get(c.__class__.__name__, "")
        if key:
            checker_map[key] = c

    # ── run the draft→validate→revise loop ──
    print(f"\n{'='*80}")
    print(" draft → validate (recovered program) → revise")
    print(f"{'='*80}")

    passed, total = 0, 0
    for i, test in enumerate(tests):
        val = checker_map.get(test["checker"])
        if not val:
            print(f"  SKIP: no validator for {test['checker']}")
            continue
        total += 1

        print(f"\n── Test {i+1}: {test['label']} ({val.rule_id}) ──")

        # Round 1: the "model" drafts the WRONG version.
        draft = test["wrong"]
        result1 = val.validate(draft)
        status1 = "ACCEPT" if result1.ok else "REJECT"
        print(f"  Draft:   \"{draft}\"")
        print(f"  Verdict: {status1} "
              f"({'✓' if not result1.ok else '✗ should have rejected'})")

        if result1.ok:
            # The checker failed to catch the error — this would be a bug.
            print(f"  ❌ FAIL: checker should have rejected this!")
        else:
            # Round 2: the model revises to the CORRECT version.
            revision = test["revision"]
            result2 = val.validate(revision)
            status2 = "ACCEPT" if result2.ok else "REJECT"
            print(f"  Revise:  \"{revision}\"")
            print(f"  Verdict: {status2} "
                  f"{'✓' if result2.ok else '✗ should have accepted'}")
            if result2.ok:
                print(f"  ✅ pass: draft rejected, revision accepted")
                passed += 1
            else:
                print(f"  ❌ FAIL: checker rejected the revision!")

    print(f"\n{'='*80}")
    print(f" Result: {passed}/{total} test cases pass ({passed/total*100:.0f}%)")
    print(f" Every rejection = a recovered Mog program executing in a sandbox,")
    print(f" not a hand-written check — the decision IS synthesized, verified code.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()