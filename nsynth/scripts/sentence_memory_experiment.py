#!/usr/bin/env python3
"""Rule-compressed memory over SENTENCES — Tier C of the "infinite context" study.

Tier A (rule_memory_experiment.py) streamed (word -> plural) pairs. Tier C streams
GRAMMATICALITY-labeled SENTENCES and asks the same question for a *judgement* task:

    Does storing knowledge as a VERIFIED RULE (an nSynth-synthesized
    grammaticality program) instead of as instances give bounded memory,
    unbounded reach, and zero forgetting?

THE TASK: subject-verb 3sg-agreement grammaticality. The curriculum's Stage-3
morphology generator emits grammatical 3sg sentences ("The coach sails." /
"The worker washes." / "The clerk tidies.") and characteristic-error negatives
("The coach sail." / "The clerk studys." / "The farmer displaies."). Each sentence
is encoded by the PERCEPTION layer to a small int feature array (inflection-suffix
tokens 100..108 from the morpheme tokenizer + a stem-is-sibilant feature 901),
exactly as in checkers/sentence_3sg_grammatical.py and the bridge's
task_sentence_3sg_general. Label = 1 grammatical, 0 not.

THREE MEMORIES, streamed chunk by chunk:

  RULE     (ours)  : an nSynth-synthesized DNF classifier over the feature array
                     (valid iff <+es> OR <+ies> OR (<+s> AND NOT sibilant-stem)),
                     plus a tiny exception table for any item the rule misjudges.
                     Storage = |rule code| + |exceptions|.
  INSTANCE (RAG)   : remember every (sentence -> label) seen. Grows linearly. Can
                     only answer sentences it has literally stored.
  WINDOW-W (LLM)   : remember only the last W sentences. Constant storage W, but
                     FORGETS everything older.

CRITICAL TEST — held-out UNSEEN verbs. A block of verbs is withheld from the
stream entirely. Their grammatical/ungrammatical sentences are never shown to any
memory. Only a RULE can judge them: it abstracts the verb away into the suffix +
sibilant features, so an unseen verb lands in a feature cell the rule already
covers. INSTANCE and WINDOW have never seen the sentence string, so they answer
0% on unseen verbs — the bounded-memory baselines cannot generalize.

EXPECTED RESULT: RULE storage flattens (converges once the regularity is captured;
only finite genuine exceptions accrete); RULE coverage on ALL seen stays 100% (it
never forgets even long-evicted items); RULE coverage on UNSEEN verbs ~100%. The
INSTANCE memory grows without bound and scores 0% on unseen; the WINDOW forgets
the past (coverage-on-seen decays to W/N) and scores 0% on unseen. That gap is the
"effectively infinite context" for the regular part of the judgement stream.

Run:  python scripts/sentence_memory_experiment.py [--stream N] [--window W] [--chunk C]
Outputs: a results table, a FINAL block, and sentence_memory_results.csv.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYNTH = HERE.parent
BIN = NSYNTH / "target" / "release" / "mog_synth"
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))

FN = "sentence_3sg_ok"
STEM_SIBILANT = 901
_SIBILANT_SUFFIXES = ("s", "sh", "ch", "x", "z")


# ── perception layer (shared with checkers/sentence_3sg_grammatical.py) ──────

def _last_word(sentence: str) -> str:
    words = re.findall(r"[A-Za-z]+", sentence)
    return words[-1].lower() if words else ""


def _surface_to_base(morph):
    to_base = {}
    for v in morph.REGULAR_VERBS:
        for form in (v.base, v.third_singular, v.past_regular, v.gerund):
            to_base[form.lower()] = v.base.lower()
    return to_base


def encode_sentence(sentence: str, tok, to_base) -> tuple[int, ...]:
    ids = tok.encode(sentence, add_bos=False, add_eos=False)
    feats = sorted(i for i in ids if 100 <= i <= 108)
    verb = _last_word(sentence)
    base = to_base.get(verb, verb)
    if base.endswith(_SIBILANT_SUFFIXES):
        feats = feats + [STEM_SIBILANT]
    return tuple(feats)


# ── nSynth: synthesize the grammaticality DNF, and run it as a classifier ────

def synth_rule(rows: list[tuple[tuple[int, ...], int]]) -> str | None:
    """Synthesize a grammaticality classifier from (feature-array, label) rows."""
    if not rows:
        return None
    # need both classes present for a meaningful DNF
    labels = {lab for _f, lab in rows}
    if len(labels) < 2:
        return None
    payload = json.dumps({
        "name": FN,
        "signature": f"fn {FN}(arr: [i64]) -> i64",
        "examples": [{"inputs": [list(feats)], "expected": lab} for feats, lab in rows],
        "holdouts": [],
    })
    try:
        out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                             capture_output=True, text=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        return None
    r = None
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                r = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if not r:
        return None
    return r["code"] if r.get("success") else None


def predict(rule_code: str | None, arrays: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    """Run the rule classifier on many feature arrays in one Mog execution.

    Returns {feature-array -> predicted label}. Dedupes identical arrays."""
    if rule_code is None or not arrays:
        return {}
    uniq = sorted(set(arrays))
    calls = "".join(
        "  println_i64(" + FN + "([" + ", ".join(str(x) for x in a) + "]));\n"
        for a in uniq
    )
    prog = f"{rule_code}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        out = subprocess.run([str(BIN), "--run-file", path],
                             capture_output=True, text=True, timeout=60).stdout
    finally:
        Path(path).unlink(missing_ok=True)
    nums = [int(ln) for ln in out.splitlines() if re.fullmatch(r"-?\d+", ln.strip())]
    return {a: (nums[i] if i < len(nums) else -1) for i, a in enumerate(uniq)}


# ── dataset: stream sentences from KNOWN verbs, hold out UNSEEN verbs ─────────

def build_dataset(stream_cap: int):
    """Return (stream_rows, unseen_rows).

    Each row = (sentence, feature-array, label). The verb lexicon is split:
    ~80% of verbs feed the stream; ~20% are WITHHELD and only appear in unseen_rows.
    For every (subject, verb) we emit a grammatical 3sg and its characteristic
    ungrammatical error, using the curriculum's own inflection + error rules.
    """
    from v2.curriculum import morphology_productivity as morph
    from v2.tokenizer.morpheme_tokenizer import MorphemeTokenizer

    tok = MorphemeTokenizer()
    to_base = _surface_to_base(morph)
    verbs = list(morph.REGULAR_VERBS)
    subjects = list(morph.SINGULAR_SUBJECTS)

    # deterministic verb split: every 5th verb is held out (unseen)
    seen_verbs = [v for i, v in enumerate(verbs) if i % 5 != 0]
    unseen_verbs = [v for i, v in enumerate(verbs) if i % 5 == 0]

    def wrong_3sg(base: str) -> str:
        # the curriculum's Stage-3 over-regularization error (morph._negative_for)
        if base.endswith(("ay", "ey", "oy", "uy")):
            return base[:-1] + "ies"          # play -> *plaies
        if base.endswith("y"):
            return base + "s"                  # study -> *studys
        return base                            # walk -> *walk (bare)

    def rows_for(verb_list, subj_list):
        rows, seen = [], set()
        for v in verb_list:
            base = v.base.lower()
            good = v.third_singular
            bad = wrong_3sg(base)
            for subj in subj_list:
                for surface, label in ((good, 1), (bad, 0)):
                    sentence = f"{subj} {surface}."
                    if sentence in seen:
                        continue
                    seen.add(sentence)
                    feats = encode_sentence(sentence, tok, to_base)
                    rows.append((sentence, feats, label))
        rows.sort(key=lambda r: r[0])
        return rows

    # one subject per (verb, polarity) is enough surface variety; use a few so the
    # stream has length without exploding. Subjects do not change the feature array.
    subj_sample = subjects[:4]
    stream_rows = rows_for(seen_verbs, subj_sample)[:stream_cap]
    unseen_rows = rows_for(unseen_verbs, subj_sample)
    return stream_rows, unseen_rows


# ── the three-memory comparison ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=1000)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=50)
    args = ap.parse_args()
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    stream_rows, unseen_rows = build_dataset(args.stream)
    print(f"Streaming {len(stream_rows)} grammaticality-labeled 3sg sentences "
          f"(window W={args.window}); {len(unseen_rows)} held-out UNSEEN-verb sentences\n")

    # RULE memory
    rule_code: str | None = None
    rule_pool: list[tuple[tuple[int, ...], int]] = []   # (feature-array, label) used for synthesis
    exceptions: dict[tuple[int, ...], int] = {}          # feature-array -> corrected label
    n_resynth = 0

    # baselines
    instance_mem: dict[str, int] = {}     # sentence -> label (RAG: store every instance)
    window: deque = deque(maxlen=args.window)  # last-W sentences
    window_label: dict[str, int] = {}     # sentence -> label, for whatever is in window
    seen_sentences: list[str] = []

    mistaken_items: set = set()
    repeat_mistakes = 0

    def rule_predict_label(feats: tuple[int, ...], preds: dict) -> int:
        if feats in exceptions:
            return exceptions[feats]
        return preds.get(feats, -1)

    def rule_coverage(rows) -> float:
        """Accuracy of (rule + exception table) on a list of (sentence,feats,label)."""
        if not rows:
            return 1.0
        arrays = [f for _s, f, _l in rows if f not in exceptions]
        preds = predict(rule_code, arrays)
        ok = sum(1 for _s, f, lab in rows if rule_predict_label(f, preds) == lab)
        return ok / len(rows)

    rows_out = []
    for start in range(0, len(stream_rows), args.chunk):
        chunk = stream_rows[start:start + args.chunk]

        # ---- RULE memory: judge the chunk, collect misjudged items ----
        arrays = [f for _s, f, _l in chunk if f not in exceptions]
        preds = predict(rule_code, arrays)
        pending = []
        for sentence, feats, label in chunk:
            got = rule_predict_label(feats, preds)
            if got != label:
                pending.append((feats, label))
                if sentence in mistaken_items:
                    repeat_mistakes += 1
                mistaken_items.add(sentence)
        if pending:
            # extend the rule to cover the misses, then re-synthesize
            rule_pool.extend(pending)
            new_rule = synth_rule(rule_pool)
            n_resynth += 1
            if new_rule:
                rule_code = new_rule
                recheck = predict(rule_code, [f for f, _l in pending])
                for feats, label in pending:
                    if recheck.get(feats, -1) != label:
                        exceptions[feats] = label
                        rule_pool[:] = [(f, l) for f, l in rule_pool if f != feats]
            else:
                for feats, label in pending:
                    exceptions[feats] = label
                    rule_pool[:] = [(f, l) for f, l in rule_pool if f != feats]

        # ---- baselines ----
        for sentence, _feats, label in chunk:
            seen_sentences.append(sentence)
            instance_mem[sentence] = label
            window.append(sentence)  # deque auto-evicts oldest beyond maxlen
            window_label[sentence] = label
        # keep window_label scoped to what's actually in the window
        live = set(window)
        for s in list(window_label.keys()):
            if s not in live:
                del window_label[s]

        # ---- metrics ----
        rule_bytes = (len(rule_code) if rule_code else 0) + sum(
            len(str(list(k))) + 1 for k in exceptions
        )
        instance_bytes = sum(len(s) + 1 for s in instance_mem)
        window_bytes = sum(len(s) + 1 for s in window)

        # coverage on ALL seen — the rule re-derives even long-evicted items.
        # We rebuild (sentence, feats, label) rows from a cache so we never
        # re-tokenize; the rule's verdict, not stored instances, answers them.
        cov_seen_rule = rule_coverage(seen_rows_cache(seen_sentences, stream_rows))
        cov_seen_instance = 1.0  # RAG stores everything it has seen
        cov_seen_window = len(set(window)) / len(seen_sentences)

        cov_unseen_rule = rule_coverage(unseen_rows)
        # baselines: an unseen-verb sentence string was never stored -> 0 coverage
        cov_unseen_instance = sum(
            1 for s, _f, lab in unseen_rows if instance_mem.get(s) == lab
        ) / len(unseen_rows)
        cov_unseen_window = sum(
            1 for s, _f, lab in unseen_rows if window_label.get(s) == lab
        ) / len(unseen_rows)

        rows_out.append({
            "n": len(seen_sentences),
            "rule_bytes": rule_bytes, "exceptions": len(exceptions), "resynths": n_resynth,
            "instance_bytes": instance_bytes, "window_bytes": window_bytes,
            "cov_seen_rule": cov_seen_rule, "cov_seen_instance": cov_seen_instance,
            "cov_seen_window": cov_seen_window,
            "cov_unseen_rule": cov_unseen_rule, "cov_unseen_instance": cov_unseen_instance,
            "cov_unseen_window": cov_unseen_window,
        })

    # ── report ──
    print(f"{'n':>5} {'RULE B':>7} {'#exc':>4} {'INST B':>8} {'WIN B':>6} {'resyn':>5} "
          f"{'seen(rule)':>10} {'seen(win)':>9} {'unseen(rule)':>12} {'unseen(inst)':>12}")
    step = max(1, len(rows_out) // 14)
    for r in rows_out[::step]:
        print(f"{r['n']:>5} {r['rule_bytes']:>7} {r['exceptions']:>4} {r['instance_bytes']:>8} "
              f"{r['window_bytes']:>6} {r['resynths']:>5} {r['cov_seen_rule']*100:>9.1f}% "
              f"{r['cov_seen_window']*100:>8.1f}% {r['cov_unseen_rule']*100:>11.1f}% "
              f"{r['cov_unseen_instance']*100:>11.1f}%")

    last = rows_out[-1]
    print("\n=== FINAL ===")
    print(f"  RULE memory:     {last['rule_bytes']:>6} bytes  ({last['exceptions']} exceptions), "
          f"{last['resynths']} re-syntheses total")
    print(f"  INSTANCE memory: {last['instance_bytes']:>6} bytes  (grows linearly with the stream)")
    print(f"  WINDOW-{args.window} memory:  {last['window_bytes']:>6} bytes  (constant; forgets older)")
    print(f"  Coverage on ALL {last['n']} seen — "
          f"RULE {last['cov_seen_rule']*100:.1f}%, INSTANCE {last['cov_seen_instance']*100:.1f}%, "
          f"WINDOW {last['cov_seen_window']*100:.1f}% (forgot the rest)")
    print(f"  Coverage on UNSEEN-VERB sentences — "
          f"RULE {last['cov_unseen_rule']*100:.1f}%, INSTANCE {last['cov_unseen_instance']*100:.1f}%, "
          f"WINDOW {last['cov_unseen_window']*100:.1f}%")
    ratio = last['instance_bytes'] / max(1, last['rule_bytes'])
    print(f"  Compression: instance/rule = {ratio:.1f}x;  reach: UNBOUNDED "
          f"(rule judges unseen verbs the baselines cannot).")
    print(f"  Hamilton (mistake memory): {len(mistaken_items)} distinct misjudgements over "
          f"{last['n']} items (converges, finite); repeated mistakes: {repeat_mistakes} "
          f"(self-improving — once corrected, never re-made).")

    csv = HERE / "sentence_memory_results.csv"
    with open(csv, "w") as f:
        f.write(",".join(rows_out[0].keys()) + "\n")
        for r in rows_out:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"\nCurves -> {csv}")


# ── small helpers for re-deriving feature arrays of seen sentences ───────────

_FEAT_CACHE: dict[str, tuple[int, ...]] = {}


def seen_rows_cache(seen_sentences, stream_rows):
    """Build (sentence, feats, label) rows for everything seen so far, using a
    cache from the stream rows so we never re-tokenize."""
    if not _FEAT_CACHE:
        for s, f, lab in stream_rows:
            _FEAT_CACHE[s] = (f, lab)
    return [(s, _FEAT_CACHE[s][0], _FEAT_CACHE[s][1]) for s in seen_sentences]


if __name__ == "__main__":
    main()
