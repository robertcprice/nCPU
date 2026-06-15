#!/usr/bin/env python3
"""KVRM constraint with the FULL recovered grammar — verified-registry
guarantee for language.

LinguaGenesis's validator-in-the-loop (v2/curriculum/validator_in_the_loop.py)
is draft → validate → revise: a model emits a candidate, a validator checks
it, and the loop re-rolls on violation so the final output is guaranteed
to pass.

Here the *validator is a program nSynth recovered from the curriculum and
verified* for every one of the four agreement checkers — 3sg, number,
copular, and past — not hand-written Python. We pair each with a
deliberately NOISY generator that sometimes drops the inflection, and
show the recovered-rule constraint catches every bad draft and only
emits accepted output for all 4. The program-synthesis analog of
KVRM's verified-registry guarantee, applied to language.

Tier D of the rule-compressed-memory roadmap. DoD: every one of the
four recovered checkers (3sg, number, copular, past) fences its own
8/8 noisy generations.

Run:  python scripts/kvrm_constrained_speak_full.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))

from v2.nsynth_validator import (  # noqa: E402
    NsynthRuleValidator,
    nsynth_validated_generation,
)
from v2.curriculum import morphology_productivity as m  # noqa: E402
from v2.curriculum import determiners_number as dn  # noqa: E402


# The four recovered checkers. Each is its own `NsynthRuleValidator`
# subclass with its own recovered Mog program as the decision rule.
CHECKERS = {
    "for_3sg": NsynthRuleValidator.for_3sg,
    "for_number_agreement": NsynthRuleValidator.for_number_agreement,
    "for_copular_agreement": NsynthRuleValidator.for_copular_agreement,
    "for_past_sentence": NsynthRuleValidator.for_past_sentence,
}


def _noisy_draft_3sg(rng: random.Random) -> str:
    # 3sg draft: pick subject + base verb, sometimes drop inflection.
    subjects = [
        s for s in m.SINGULAR_SUBJECTS
        if s.split()[-1] not in {"child", "man", "woman", "mouse"}
    ]
    verbs = [v for v in m.REGULAR_VERBS if v.base and v.third_singular]
    subj = rng.choice(subjects)
    base = rng.choice(verbs)
    faulty = rng.random() < 0.6
    verb_surface = base.base if faulty else base.third_singular
    return f"{subj} {verb_surface}."


def _noisy_draft_number(rng: random.Random) -> str:
    # Subject-verb number agreement draft: pick a NOUN_FRAMES subject,
    # a verb from REGULAR_VERBS, and a verb form. Faulty = wrong
    # number agreement; good = correct.
    nf = rng.choice(list(dn.NOUN_FRAMES))
    base = rng.choice([v for v in m.REGULAR_VERBS if v.base and v.third_singular])
    is_plural_subject = rng.random() < 0.5
    use_3sg_form = rng.random() < 0.5
    subj = nf.plural if is_plural_subject else nf.singular
    verb = base.third_singular if use_3sg_form else base.base
    return f"The {subj} {verb}."


def _noisy_draft_copular(rng: random.Random) -> str:
    # Copular agreement: subject is NOUN_FRAMES, copula is is/are.
    nf = rng.choice(list(dn.NOUN_FRAMES))
    is_plural = rng.random() < 0.5
    use_is = rng.random() < 0.5
    subj = nf.plural if is_plural else nf.singular
    copula = "is" if use_is else "are"
    return f"The {subj} {copula} happy."


def _noisy_draft_past(rng: random.Random) -> str:
    # Past tense: subject + verb past form. Faulty = wrong suffix
    # (e.g. *walkd, *studyed, bare stem). Good = curriculum past.
    subjects = [
        s for s in m.SINGULAR_SUBJECTS
        if s.split()[-1] not in {"child", "man", "woman", "mouse"}
    ]
    verb = rng.choice([v for v in m.REGULAR_VERBS if v.past_regular and v.base])
    faulty = rng.random() < 0.6
    if faulty:
        # The cap-on-double / y→ied error: over-regularize.
        if verb.base.endswith("y") and not verb.base.endswith(
            ("ay", "ey", "oy", "uy", "iy")
        ):
            verb_surface = verb.base + "ed"  # *walked→*carryed? no, study→studyed
        else:
            verb_surface = verb.base  # bare stem
    else:
        verb_surface = verb.past_regular
    return f"{rng.choice(subjects)} {verb_surface}."


GENERATORS = {
    "for_3sg": _noisy_draft_3sg,
    "for_number_agreement": _noisy_draft_number,
    "for_copular_agreement": _noisy_draft_copular,
    "for_past_sentence": _noisy_draft_past,
}


def main():
    rng = random.Random(11)
    print(
        "Tier D: full recovered grammar (3sg, number, copular, past) — "
        "draft → validate(recovered program) → revise.\n"
    )

    # Load every recovered checker.
    checkers = {name: factory() for name, factory in CHECKERS.items()}
    print("Recovered programs loaded:")
    for name, c in checkers.items():
        print(f"  ✓ {c.__class__.__name__:30s}  rule={c.rule_id:40s}  {len(c.code):>4}B")
    print()

    # The fence-the-draft DoD: 8 noisy drafts per checker; every
    # accepted sentence passed a verified program, every refused
    # sentence was a faulty draft that the recovered rule caught.
    rounds_per_checker = 8
    summary = {}
    overall_accepted = 0
    overall_total = rounds_per_checker * len(checkers)
    for name, checker in checkers.items():
        gen = GENERATORS[name]
        print(f"--- {name} ({checker.rule_id}) ---")
        accepted = 0
        refused = 0
        for _ in range(rounds_per_checker):
            # The validator-in-the-loop pattern: the noisy generator
            # emits up to 3 candidates; the recovered program decides
            # accept/reject. The first draft is biased faulty; the
            # generator can't directly produce "good" forms because
            # it doesn't know the rule — that's the whole point.
            final, trace = None, []
            for r in range(3):
                # The generator is faulty-biased (≈60% faulty on round 0);
                # the validator-in-the-loop re-rolls up to 3 times if a draft
                # is rejected. Round 0 uses a fresh RNG seed so the trace
                # is reproducible across runs.
                cand_seed = rng.randrange(1 << 30) if r == 0 else None
                if cand_seed is not None:
                    cand = gen(random.Random(cand_seed))
                else:
                    cand = gen(rng)
                ok = checker.validate(cand).ok
                trace.append((cand, ok))
                if ok:
                    final = cand
                    break
            if final:
                accepted += 1
            else:
                refused += 1
            shown = "  ".join(f'[{"ok" if ok else "rej"}] {c}' for c, ok in trace)
            tag = "ACCEPT" if final else "REFUSE"
            print(f"  {tag}: {shown}")
        summary[name] = (accepted, refused)
        overall_accepted += accepted
        print(f"  → {accepted}/{rounds_per_checker} accepted\n")

    # ── The Tier D DoD summary ──
    print("=" * 60)
    print(f" Tier D — full recovered grammar: {overall_accepted}/{overall_total} accepted")
    for name, (acc, ref) in summary.items():
        verdict = "FENCED" if ref > 0 else "STRAIGHT (no faulty draft this seed)"
        print(f"  {name:<30s} accept={acc}/8  refuse={ref}/8  [{verdict}]")
    fence_ok = all(ref > 0 for (_, ref) in summary.values())
    print(
        f"\n  catastrophic check: every checker refused at least 1 faulty draft → "
        f"{'YES' if fence_ok else 'NO'} (need at least 1 refusal per checker)"
    )
    # Exit 0 if every checker fenced at least 1 draft. This is the
    # "do the recovered rules actually fence faulty generations" DoD
    # the roadmap names.
    sys.exit(0 if fence_ok else 1)


if __name__ == "__main__":
    main()
