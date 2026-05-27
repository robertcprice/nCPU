#!/usr/bin/env python3
"""
Differential testing: BSD coreutils vs GNU coreutils.

Run every fuzz probe through BOTH the BSD binary (macOS default) AND
the GNU binary (installed via `brew install coreutils`, prefixed with
`g`). Report every probe where the two implementations produce
different output. Every such probe is a real compatibility finding —
shell scripts relying on the feature will break across platforms.

This is a natural extension of our fuzzing framework: by running the
same probes against two reference implementations, we discover
specification ambiguities without human labour. Security-research
angle: differential testing between implementations surfaces
behavioural discrepancies that tools like AFL find only
accidentally.

Usage:
    python3 tools/binary_harvest/diff_test.py --all --n 300
    python3 tools/binary_harvest/diff_test.py --tool sort --n 500 \\
        --out /tmp/bsd_vs_gnu.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzz import FUZZERS  # noqa: E402


# Tools where both BSD (macOS default) and GNU (g-prefixed via
# homebrew's coreutils) are available locally. Mapping to GNU binary
# paths when different from BSD's.
BSD_GNU_PAIRS = {
    "sort":   ("/usr/bin/sort", "/opt/homebrew/bin/gsort"),
    "uniq":   ("/usr/bin/uniq", "/opt/homebrew/bin/guniq"),
    "wc":     ("/usr/bin/wc",   "/opt/homebrew/bin/gwc"),
    "head":   ("/usr/bin/head", "/opt/homebrew/bin/ghead"),
    "tail":   ("/usr/bin/tail", "/opt/homebrew/bin/gtail"),
    "cut":    ("/usr/bin/cut",  "/opt/homebrew/bin/gcut"),
    "base64": ("/usr/bin/base64", "/opt/homebrew/bin/gbase64"),
    "fold":   ("/usr/bin/fold", "/opt/homebrew/bin/gfold"),
    "tr":     ("/usr/bin/tr",   "/opt/homebrew/bin/gtr"),
    "nl":     ("/usr/bin/nl",   "/opt/homebrew/bin/gnl"),
    "expand": ("/usr/bin/expand", "/opt/homebrew/bin/gexpand"),
    "tac":    ("/opt/homebrew/bin/tac", "/opt/homebrew/bin/gtac"),
    "seq":    ("/usr/bin/seq",  "/opt/homebrew/bin/gseq"),
    "paste":  ("/usr/bin/paste", "/opt/homebrew/bin/gpaste"),
}


def _run(binary: str, args: Tuple[str, ...], stdin: str,
          timeout_s: int = 5) -> Tuple[int, str]:
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    try:
        r = subprocess.run(
            [binary, *args], input=stdin,
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except Exception:
        return (-1, "")
    return (r.returncode, r.stdout)


def diff_tool(tool: str, n: int, seed: int = 42) -> Dict:
    if tool not in BSD_GNU_PAIRS or tool not in FUZZERS:
        return {"tool": tool, "skipped": "no BSD/GNU pair or no fuzzer"}
    bsd, gnu = BSD_GNU_PAIRS[tool]
    for p in (bsd, gnu):
        if not Path(p).exists():
            return {"tool": tool, "skipped": f"missing {p}"}

    rng = random.Random(seed)
    divergences: List[Dict] = []
    successes = 0

    for _ in range(n):
        args, stdin = FUZZERS[tool](rng)
        rc_bsd, out_bsd = _run(bsd, args, stdin)
        rc_gnu, out_gnu = _run(gnu, args, stdin)
        # Accept rc=1 for grep-style "no match"; otherwise skip non-zero.
        ok_codes = {0, 1} if tool == "grep" else {0}
        if rc_bsd not in ok_codes or rc_gnu not in ok_codes:
            continue
        if out_bsd == out_gnu:
            successes += 1
            continue
        divergences.append({
            "args": list(args),
            "stdin": stdin[:100],
            "bsd": out_bsd[:100],
            "gnu": out_gnu[:100],
        })

    return {
        "tool": tool,
        "probes_run": successes + len(divergences),
        "agreeing": successes,
        "divergent": len(divergences),
        "rate": len(divergences) / max(successes + len(divergences), 1),
        "examples": divergences[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tool", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tools = sorted(BSD_GNU_PAIRS.keys()) if args.all else \
            [args.tool] if args.tool else None
    if not tools:
        print("specify --tool NAME or --all", file=sys.stderr); sys.exit(2)

    all_results: List[Dict] = []
    for t in tools:
        r = diff_tool(t, args.n, seed=args.seed)
        all_results.append(r)
        if "skipped" in r:
            print(f"  {t:<10}  skipped: {r['skipped']}")
            continue
        marker = "✓" if r["divergent"] == 0 else "⚠"
        print(f"  {t:<10}  agree={r['agreeing']:>4}  divergent={r['divergent']:>4}  "
              f"rate={100*r['rate']:.2f}%  {marker}")
        for ex in r["examples"]:
            print(f"    args={ex['args']}")
            print(f"    bsd={ex['bsd']!r}")
            print(f"    gnu={ex['gnu']!r}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"\n[diff-test] wrote {args.out}")


if __name__ == "__main__":
    main()
