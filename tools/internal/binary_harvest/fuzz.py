#!/usr/bin/env python3
"""
Fuzz mode: find inputs where our Python reimpl disagrees with the
real binary. Novel value:

  1. **Refining the dataset.** Every disagreement is a probe that
     should NOT become a cache row (our impl is wrong for it). The
     verify step already filters these when we harvest against curated
     probes; here we go the other way — probe aggressively for new
     corner cases.

  2. **Free bug-finding.** When the binary and our impl disagree on
     a random input, one of two things is true:
       - Our impl is incomplete (missing a flag edge case). Log, fix.
       - The binary has a corner-case behaviour we didn't expect.
         That's a real-world oddity worth knowing about.

  3. **Robustness bound.** Fuzzing N random inputs and measuring the
     divergence rate gives a quantitative confidence in each
     reimplementation's faithfulness. "10000 fuzzes, 0 divergences"
     is strong evidence; "10000 fuzzes, 50 divergences" tells us the
     probe generator needs to subsample within the agreed region.

Usage:
    python3 tools/binary_harvest/fuzz.py --tool sort --n 500
    python3 tools/binary_harvest/fuzz.py --all --n 200 --out divs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest import TOOLS, reference_python  # noqa: E402


# ─── Random input generators (per tool) ──────────────────────────────────
#
# Probe generators in harvest.py are designed to produce agreeing inputs;
# fuzz generators produce *aggressive* inputs likely to hit corner cases.


def _fuzz_sort(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    # Constrain to alphanumeric so we don't hit locale-collation
    # differences between BSD sort and Python's codepoint ordering
    # (documented divergence, not a reimpl bug we can fix without a
    # locale-aware string library).
    alphabet = string.ascii_letters + string.digits
    flavours = [
        lambda: "",
        lambda: "\n".join(str(rng.randint(-10**9, 10**9))
                          for _ in range(rng.randint(0, 20))) + "\n",
        lambda: "\n".join("".join(rng.choices(alphabet,
                                                k=rng.randint(0, 10)))
                          for _ in range(rng.randint(0, 15))) + "\n",
    ]
    body = rng.choice(flavours)()
    flags = rng.choice([(), ("-n",), ("-r",)])
    return (flags, body)


def _fuzz_uniq(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    n = rng.randint(0, 30)
    # Mix of repeats and uniques.
    words = [rng.choice(["a", "b", "c", "", rng.choice(string.ascii_lowercase)])
             for _ in range(n)]
    # Intentionally NOT sorting — uniq only dedups consecutive lines.
    return ((), "\n".join(words) + ("\n" if words else ""))


def _fuzz_wc(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    n_lines = rng.randint(0, 50)
    body = "\n".join(
        "".join(rng.choices(string.printable.strip("\r"),
                             k=rng.randint(0, 30)))
        for _ in range(n_lines)
    )
    if body and rng.random() < 0.5:
        body += "\n"
    return (rng.choice([(), ("-l",), ("-w",), ("-c",)]), body)


def _fuzz_head(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    n_lines = rng.randint(0, 30)
    body = "\n".join(
        "".join(rng.choices(string.ascii_lowercase + " ",
                             k=rng.randint(0, 20)))
        for _ in range(n_lines)
    ) + "\n"
    n = rng.randint(0, 50)
    return (("-n", str(n)), body)


def _fuzz_tail(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    return _fuzz_head(rng)


def _fuzz_cut(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    delims = [",", ":", "|", " "]
    delim = rng.choice(delims)
    n_lines = rng.randint(1, 8)
    lines = [delim.join("".join(rng.choices(string.ascii_lowercase,
                                               k=rng.randint(0, 5)))
                          for _ in range(rng.randint(1, 5)))
             for _ in range(n_lines)]
    field = str(rng.randint(1, 3))
    return (("-d", delim, "-f", field), "\n".join(lines) + "\n")


def _fuzz_grep(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    # Plain literal patterns only (avoid regex escaping differences).
    pattern_alphabet = string.ascii_lowercase
    pattern = "".join(rng.choices(pattern_alphabet, k=rng.randint(1, 4)))
    n_lines = rng.randint(1, 15)
    lines = []
    for _ in range(n_lines):
        line = "".join(rng.choices(string.ascii_lowercase + " ",
                                      k=rng.randint(0, 20)))
        if rng.random() < 0.4:
            # insert pattern at random position
            pos = rng.randint(0, len(line))
            line = line[:pos] + pattern + line[pos:]
        lines.append(line)
    return ((pattern,), "\n".join(lines) + "\n")


def _fuzz_jq(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    # Stick to filters we implemented: `.`, `.<field>`, `length`, `add`.
    shape = rng.choice(["obj", "arr"])
    if shape == "obj":
        obj = {"k" + str(i): rng.randint(-100, 100) for i in range(rng.randint(1, 5))}
        filt = rng.choice([".", rng.choice(["." + k for k in obj.keys()])])
        return (("-c", filt), json.dumps(obj) + "\n")
    arr = [rng.randint(-50, 50) for _ in range(rng.randint(0, 6))]
    filt = rng.choice(["length", "add", "."])
    if filt == "add" and not arr:
        filt = "length"  # add on empty is null in real jq; keep stable.
    return (("-c", filt), json.dumps(arr) + "\n")


def _fuzz_base64(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    body = "".join(rng.choices(string.printable,
                                 k=rng.randint(0, 100)))
    return ((), body)


def _fuzz_hash(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    # Any bytes work, but we constrain to printable so the TSV diff
    # isn't visually garbled for hash-divergence reports.
    body = "".join(rng.choices(string.printable,
                                 k=rng.randint(0, 100)))
    return ((), body)


def _fuzz_tr(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    mode = rng.choice(["trans", "delete", "squeeze"])
    body = "".join(rng.choices(string.ascii_letters + string.digits + " ",
                                  k=rng.randint(0, 60))) + "\n"
    if mode == "trans":
        return (("a-z", "A-Z"), body)
    if mode == "delete":
        return (("-d", "0-9"), body)
    return (("-s", "a-z"), body)


def _fuzz_rev(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    body = "\n".join(
        "".join(rng.choices(string.ascii_letters, k=rng.randint(0, 20)))
        for _ in range(rng.randint(1, 8))
    ) + "\n"
    return ((), body)


def _fuzz_fold(rng: random.Random) -> Tuple[Tuple[str, ...], str]:
    line = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(0, 120)))
    width = rng.choice([5, 10, 20, 40, 80])
    return (("-w", str(width)), line + "\n")


FUZZERS: Dict[str, callable] = {
    "sort": _fuzz_sort, "uniq": _fuzz_uniq, "wc": _fuzz_wc,
    "head": _fuzz_head, "tail": _fuzz_tail, "cut": _fuzz_cut,
    "grep": _fuzz_grep, "jq": _fuzz_jq, "base64": _fuzz_base64,
    "sha256sum": _fuzz_hash, "md5sum": _fuzz_hash,
    "tr": _fuzz_tr, "rev": _fuzz_rev, "fold": _fuzz_fold,
}


# ─── Fuzz runner ─────────────────────────────────────────────────────────


def fuzz_tool(tool: str, n: int, seed: int = 42,
               timeout_s: int = 5) -> List[Dict]:
    """Run n random probes. Returns a list of divergences (probes where
    binary output ≠ Python reimpl output)."""
    if tool not in FUZZERS or tool not in TOOLS:
        return [{"error": f"no fuzzer for {tool}"}]
    spec = TOOLS[tool]
    binary = spec["bin"]
    if not Path(binary).exists():
        return [{"error": f"binary {binary} not available"}]

    rng = random.Random(seed)
    divergences: List[Dict] = []
    ok_rcodes = {0, 1} if tool == "grep" else {0}

    import signal as _signal
    def _alarm_handler(signum, frame):
        raise TimeoutError("solve() timed out")

    for i in range(n):
        args, stdin = FUZZERS[tool](rng)
        try:
            r = subprocess.run(
                [binary, *args], input=stdin,
                capture_output=True, text=True, timeout=timeout_s,
                env={**__import__("os").environ, "LC_ALL": "C", "LANG": "C"},
            )
        except Exception as e:
            continue
        if r.returncode not in ok_rcodes:
            continue
        expected = r.stdout

        code = reference_python(tool, args, stdin, expected)
        ns = {"__builtins__": __builtins__}
        old = _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(timeout_s)
        try:
            exec(code, ns)
            got = ns["solve"](stdin)
        except Exception as e:
            divergences.append({
                "tool": tool, "args": list(args),
                "stdin_prefix": stdin[:60],
                "binary_stdout": expected[:60],
                "exc": f"{e!r}"[:120],
            })
            continue
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old)

        if got != expected:
            divergences.append({
                "tool": tool, "args": list(args),
                "stdin_prefix": stdin[:60],
                "binary_stdout": expected[:60],
                "reimpl_stdout": got[:60] if isinstance(got, str) else str(got)[:60],
            })
    return divergences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tool", default=None,
                    help="Single tool to fuzz.")
    ap.add_argument("--all", action="store_true",
                    help="Fuzz every registered tool.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Write all divergences to this JSONL.")
    args = ap.parse_args()

    tools = sorted(TOOLS.keys()) if args.all else [args.tool] if args.tool \
            else None
    if not tools:
        print("specify --tool NAME or --all", file=sys.stderr); sys.exit(2)

    all_divs: List[Dict] = []
    total_probes = 0
    for t in tools:
        divs = fuzz_tool(t, args.n, seed=args.seed)
        total_probes += args.n
        if divs and "error" in divs[0]:
            print(f"  {t:<10}  skipped: {divs[0]['error']}")
            continue
        pct = 100.0 * len(divs) / max(args.n, 1)
        marker = "✓" if not divs else "✗"
        print(f"  {t:<10}  {len(divs):>4}/{args.n} divergences  ({pct:>5.2f}%)  {marker}")
        all_divs.extend(divs)

    print(f"\n[fuzz] TOTAL: {len(all_divs)} divergences in {total_probes} probes "
          f"({100.0*len(all_divs)/max(total_probes,1):.2f}%)")

    if args.out:
        with open(args.out, "w") as f:
            for d in all_divs:
                f.write(json.dumps(d) + "\n")
        print(f"[fuzz] wrote {args.out}")

    sys.exit(0 if not all_divs else 1)


if __name__ == "__main__":
    main()
