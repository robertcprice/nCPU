#!/usr/bin/env python3
"""Tier B — lifelong multi-domain rule library with ZERO catastrophic forgetting.

Claim under test: a synthesized-program memory learns DOMAIN AFTER DOMAIN into ONE
persistent library, and because every entry is a VERIFIED DISCRETE PROGRAM keyed by
its own domain, adding domain N can NEVER degrade domains 1..N-1. This is the
structural guarantee that neural lifelong learners lack — there is no shared weight
matrix to overwrite, so forgetting is impossible by construction.

We stream three string-transduction domains IN SEQUENCE, each a curriculum-sourced
(input -> output) stream:

  D1  noun pluralization        cat -> cats,  fox -> foxes,  baby -> babies
  D2  string reverse            cat -> tac,   fox -> xof,    baby -> ybab
  D3  verb 3rd-person-singular  walk -> walks, watch -> watches, try -> tries

Each domain's rule is synthesized by nSynth as a Mog program
  signature  fn <domain>(s: string) -> string
and stored in a single on-disk library (lifelong_library.json). After EVERY new
domain is added we RE-TEST every earlier domain against the same held-out items it
was first measured on — demonstrating its accuracy never moves.

Metrics reported:
  * per-domain train + holdout accuracy at the moment it is learned
  * the FULL accuracy matrix: row = "library state after learning Dk",
    col = "accuracy on Dj's holdout" — the lower-left triangle must stay flat
  * total library size in bytes (sum of all stored program sources)

Run:  python scripts/lifelong_library_experiment.py
Outputs: the accuracy matrix, the forgetting summary, lifelong_library.json,
and lifelong_library_results.csv.
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
LIBRARY_PATH = HERE / "lifelong_library.json"
CSV_PATH = HERE / "lifelong_library_results.csv"
LINGUAGENESIS = Path("/Users/bobbyprice/projects/linguigenesis")
sys.path.insert(0, str(LINGUAGENESIS))


# ── nSynth interface (string -> string program synthesis) ──────────────────────

def synth_rule(fn_name: str, pairs: list[tuple[str, str]]) -> str | None:
    """Synthesize a string->string Mog program for `fn_name` from (in,out) pairs."""
    payload = json.dumps({
        "name": fn_name,
        "signature": f"fn {fn_name}(s: string) -> string",
        "examples": [{"inputs": [i], "expected": o} for i, o in pairs],
        "holdouts": [],
    })
    out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                         capture_output=True, text=True, timeout=180).stdout
    try:
        r = json.loads(out.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None
    return r.get("code") if r.get("success") else None


def predict(code: str, fn_name: str, words: list[str]) -> dict[str, str]:
    """Run one stored program over many inputs in a single Mog execution."""
    if not code or not words:
        return {}
    calls = "".join(f'  println({fn_name}("{w}"));\n' for w in words)
    prog = f"{code}\nfn main() -> i64 {{\n{calls}  return 0;\n}}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".mog", delete=False) as f:
        f.write(prog)
        path = f.name
    out = subprocess.run([str(BIN), "--run-file", path],
                         capture_output=True, text=True).stdout
    lines = out.splitlines()
    return {w: (lines[i] if i < len(lines) else "") for i, w in enumerate(words)}


def accuracy(code: str, fn_name: str, items: list[tuple[str, str]]) -> float:
    """Fraction of (input -> expected) pairs the stored program reproduces."""
    if not items:
        return 1.0
    preds = predict(code, fn_name, [i for i, _ in items])
    ok = sum(1 for i, o in items if preds.get(i, "") == o)
    return ok / len(items)


# ── persistent library (the "no shared weights" memory) ────────────────────────

def load_library() -> dict:
    if LIBRARY_PATH.exists():
        return json.loads(LIBRARY_PATH.read_text())
    return {}


def save_library(lib: dict) -> int:
    LIBRARY_PATH.write_text(json.dumps(lib, indent=2))
    # library "weight" = total bytes of all stored programs
    return sum(len(e["code"]) for e in lib.values())


# ── curriculum-sourced domain streams ──────────────────────────────────────────

def build_domains() -> dict[str, dict]:
    """Each domain: {fn, train:[(in,out)...], holdout:[(in,out)...]} stratified
    so the holdout exercises the same orthographic sub-rules as train."""
    from v2.grammar.morphology import pluralize, _IRREGULAR_PLURAL
    from v2.curriculum import morphology_productivity as morph

    # D1 — pluralization. Pick REGULAR nouns spanning every -s/-es/-ies/-ves class.
    # Curate from the curriculum lexicon by orthographic ending so each sub-rule
    # is represented; exclude the irregular table (irreducible, not the rule).
    from v2.tokenizer.morpheme_tokenizer import _curriculum_lexicon
    _t, _i, known = _curriculum_lexicon()
    lex = sorted({w.lower() for w in known if w.isalpha() and 2 < len(w) < 10})
    lex = [w for w in lex if w not in _IRREGULAR_PLURAL]

    def bucket(word: str) -> str:
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return "es"                      # bus->buses, fox->foxes, dish->dishes
        if word.endswith("y") and word[-2:-1] not in "aeiou":
            return "ies"                     # baby->babies
        return "s"                           # cat->cats

    by_bucket: dict[str, list[str]] = {"s": [], "es": [], "ies": []}
    for w in lex:
        by_bucket[bucket(w)].append(w)
    # take a balanced, deterministic slice from each bucket
    plural_words: list[str] = []
    for b, n in (("s", 40), ("es", 24), ("ies", 16)):
        plural_words += by_bucket[b][:n]
    plural_words = sorted(set(plural_words))
    d1_pairs = [(w, pluralize(w)) for w in plural_words]

    # D2 — reverse. Same word universe; pure orthographic transform.
    rev_words = sorted(set(plural_words))[:60]
    d2_pairs = [(w, w[::-1]) for w in rev_words]

    # D3 — verb 3sg. Curriculum REGULAR_VERBS, every sibilant/y/regular class.
    vb = sorted({lm.base for lm in morph.REGULAR_VERBS
                 if lm.base.isalpha() and len(lm.base) > 1})

    def vbucket(word: str) -> str:
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return "es"
        if word.endswith("y") and word[-2:-1] not in "aeiou":
            return "ies"
        return "s"

    vby: dict[str, list[str]] = {"s": [], "es": [], "ies": []}
    for w in vb:
        vby[vbucket(w)].append(w)
    verb_words: list[str] = []
    for b, n in (("s", 36), ("es", 16), ("ies", 12)):
        verb_words += vby[b][:n]
    verb_words = sorted(set(verb_words))
    d3_pairs = [(w, morph._correct_3sg_form(w)) for w in verb_words]

    def split(pairs: list[tuple[str, str]], fn: str, keyfn) -> dict:
        """Stratified 75/25 train/holdout: every sub-rule appears in both."""
        by: dict[str, list] = {}
        for p in pairs:
            by.setdefault(keyfn(p[0]), []).append(p)
        train, hold = [], []
        for _k, grp in sorted(by.items()):
            for idx, p in enumerate(grp):
                (hold if idx % 4 == 0 else train).append(p)
        return {"fn": fn, "train": sorted(set(train)), "holdout": sorted(set(hold))}

    return {
        "pluralize":  split(d1_pairs, "pluralize", bucket),
        "reverse":    split(d2_pairs, "reverse", lambda w: str(len(w) % 3)),
        "verb_3sg":   split(d3_pairs, "verb_3sg", vbucket),
    }


# ── experiment ─────────────────────────────────────────────────────────────────

def main():
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")

    # fresh library every run so the forgetting test is honest
    if LIBRARY_PATH.exists():
        LIBRARY_PATH.unlink()

    domains = build_domains()
    order = ["pluralize", "reverse", "verb_3sg"]
    print("Lifelong rule library — learning 3 domains in sequence into ONE memory.\n")
    for name in order:
        d = domains[name]
        print(f"  {name:<10} train={len(d['train']):>3}  holdout={len(d['holdout']):>3}  "
              f"e.g. {d['train'][0][0]} -> {d['train'][0][1]}")
    print()

    lib = load_library()
    # accuracy_matrix[k][j] = holdout acc on domain j AFTER learning domains 0..k
    matrix: dict[str, dict[str, float]] = {}
    learn_acc: dict[str, tuple[float, float]] = {}
    lib_bytes_after: dict[str, int] = {}

    for k, name in enumerate(order):
        d = domains[name]
        code = synth_rule(d["fn"], d["train"])
        if code is None:
            print(f"[FAIL] could not synthesize {name}")
            learn_acc[name] = (0.0, 0.0)
            continue
        # add the verified program to the persistent library
        lib[name] = {"fn": d["fn"], "code": code}
        lib_bytes = save_library(lib)
        lib_bytes_after[name] = lib_bytes

        tr = accuracy(code, d["fn"], d["train"])
        ho = accuracy(code, d["fn"], d["holdout"])
        learn_acc[name] = (tr, ho)
        print(f"[learn D{k+1}] {name:<10} method-verified program added "
              f"({len(code)} B)  train={tr*100:5.1f}%  holdout={ho*100:5.1f}%")

        # RE-TEST every domain learned so far (the forgetting probe)
        row: dict[str, float] = {}
        for j in range(k + 1):
            jn = order[j]
            jd = domains[jn]
            entry = lib[jn]
            row[jn] = accuracy(entry["code"], entry["fn"], jd["holdout"])
        matrix[name] = row

    # ── accuracy matrix ──
    print("\n=== ACCURACY MATRIX (holdout %) ===")
    print("rows = library state after learning Dk;  cols = accuracy on each domain's holdout\n")
    header = "after\\on  " + "".join(f"{n:>11}" for n in order)
    print(header)
    for k, name in enumerate(order):
        cells = []
        for jn in order:
            v = matrix.get(name, {}).get(jn)
            cells.append("     —" if v is None else f"{v*100:9.1f}%")
        cells = [f"{c:>11}" for c in (["—"] * 0 + [
            ("     —     " if matrix.get(name, {}).get(jn) is None
             else f"{matrix[name][jn]*100:9.1f}%") for jn in order])]
        print(f"D{k+1} {name:<7}" + "".join(cells))

    # ── forgetting analysis ──
    print("\n=== ZERO-FORGETTING CHECK ===")
    forgot = False
    for j, jn in enumerate(order):
        # the accuracy of domain j the moment it was learned ...
        baseline = learn_acc[jn][1]
        # ... versus its accuracy in the FINAL library (after all later domains)
        final = matrix[order[-1]].get(jn)
        if final is None:
            continue
        drift = (final - baseline) * 100
        status = "STABLE" if abs(drift) < 1e-9 else ("DEGRADED" if drift < 0 else "CHANGED")
        if drift < -1e-9:
            forgot = True
        print(f"  {jn:<10} learned@{baseline*100:5.1f}%  ->  final@{final*100:5.1f}%  "
              f"(drift {drift:+.1f}%)  [{status}]")

    total_bytes = sum(len(e["code"]) for e in lib.values())
    print("\n=== SUMMARY ===")
    print(f"  domains in library : {len(lib)}  ({', '.join(lib)})")
    print(f"  total library size : {total_bytes} bytes "
          f"({', '.join(f'{n}={len(lib[n][code_k])}B' for n in lib for code_k in ['code'])})")
    print(f"  catastrophic forgetting observed : {'YES' if forgot else 'NO'}")
    print(f"  early-domain holdout accuracy after ALL domains added : "
          f"pluralize={matrix[order[-1]]['pluralize']*100:.1f}%, "
          f"reverse={matrix[order[-1]]['reverse']*100:.1f}%")
    print(f"\n  Library persisted to {LIBRARY_PATH}")

    # ── csv ──
    with open(CSV_PATH, "w") as f:
        f.write("after_learning,domain,holdout_accuracy\n")
        for name in order:
            for jn, v in matrix.get(name, {}).items():
                f.write(f"{name},{jn},{v}\n")
    print(f"  Curves -> {CSV_PATH}")

    return {
        "learn_acc": learn_acc,
        "matrix": matrix,
        "total_bytes": total_bytes,
        "forgot": forgot,
    }


if __name__ == "__main__":
    main()
