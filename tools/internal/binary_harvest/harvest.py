#!/usr/bin/env python3
"""
Binary behavior → cache rows → distillation dataset.

Given a system binary (sort, uniq, wc, grep, …) or a small compiled
program, we:
  1. Synthesise diverse inputs using per-binary probes.
  2. Run the binary with each input, capture stdout (+ exit code).
  3. Write cache rows shaped identically to our LLM-solve cache:
       fingerprint = sha256(binary + args + stdin_kind + stdin)
       code        = a reference Python implementation (ground truth)
       examples    = list of {inputs, expected}
  4. Optionally emit a distillation dataset (JSONL of
     {prompt, completion}) that the weekly `auto_distill.sh` cron
     can feed straight into Qwen3.5-4B LoRA training.

Why: our LLM-solve cache is ~hundreds of rows of curated benchmarks.
Harvesting coreutils gives us *thousands* of rows of real, verified
Unix behaviour — the model trained on this learns to reimplement
standard utilities from spec + examples.

The Python reference implementations are short wrappers (e.g.
`def solve(lines): return sorted(lines)`) so the dataset trains the
model on "given this I/O spec, emit this Python function". Later
stages (distillation fine-tune, beam search during inference) use
the cache exactly the same way our existing HumanEval rows do.

Usage:
    python3 tools/binary_harvest/harvest.py --tool sort --n 40
    python3 tools/binary_harvest/harvest.py --tool wc --n 30 \\
        --cache ~/.nsynth_llm_solutions.tsv
    python3 tools/binary_harvest/harvest.py --all --n 25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from llm_solution_cache import _encode, _decode, _load_all  # noqa: E402


# ─── Per-binary probes ───────────────────────────────────────────────────
#
# Each tool entry declares:
#   - `probe(rng)` → list of (args_tuple, stdin_str) pairs to try
#   - `reference(args, stdin)` → Python that produces the expected output
#   - `label` → short descriptor used in cache fingerprints
#
# New tools drop in without touching the rest of the harness.


def _rand_ints(rng: random.Random, n: int, lo: int = -100, hi: int = 100) -> List[int]:
    return [rng.randint(lo, hi) for _ in range(n)]


def _rand_lines(rng: random.Random, n: int, w_lo: int = 1, w_hi: int = 8) -> List[str]:
    words = []
    for _ in range(n):
        wlen = rng.randint(w_lo, w_hi)
        words.append("".join(rng.choices(string.ascii_lowercase, k=wlen)))
    return words


# ─── sort ───────────────────────────────────────────────────────────────


def _probe_sort(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    # Numeric sort.
    for _ in range(3):
        xs = _rand_ints(rng, rng.randint(3, 15), -50, 50)
        out.append((("-n",), "\n".join(str(x) for x in xs) + "\n"))
    # Alpha sort.
    for _ in range(3):
        ws = _rand_lines(rng, rng.randint(3, 12))
        out.append(((), "\n".join(ws) + "\n"))
    # Reverse.
    for _ in range(2):
        ws = _rand_lines(rng, rng.randint(3, 10))
        out.append((("-r",), "\n".join(ws) + "\n"))
    return out


def _reference_sort(args: Tuple[str, ...], stdin: str) -> str:
    lines = stdin.splitlines()
    numeric = "-n" in args
    reverse = "-r" in args
    if numeric:
        keyed = sorted(lines, key=lambda s: int(s) if s.strip().lstrip("-").isdigit() else 0,
                        reverse=reverse)
    else:
        keyed = sorted(lines, reverse=reverse)
    return "\n".join(keyed) + ("\n" if lines else "")


# ─── uniq ───────────────────────────────────────────────────────────────


def _probe_uniq(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(5):
        ws = _rand_lines(rng, rng.randint(3, 8))
        # Inject adjacent dupes.
        doubled = []
        for w in ws:
            doubled.append(w)
            if rng.random() < 0.5:
                doubled.append(w)
        # Sort so uniq collapses adjacent duplicates meaningfully.
        doubled.sort()
        out.append(((), "\n".join(doubled) + "\n"))
        out.append((("-c",), "\n".join(doubled) + "\n"))
    return out


# ─── wc ─────────────────────────────────────────────────────────────────


def _probe_wc(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(5):
        n_lines = rng.randint(1, 20)
        body = "\n".join(
            " ".join(_rand_lines(rng, rng.randint(1, 6))) for _ in range(n_lines)
        ) + "\n"
        for flag in (("-l",), ("-w",), ("-c",), ()):
            out.append((flag, body))
    return out


# ─── head / tail ────────────────────────────────────────────────────────


def _probe_head(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(5):
        n = rng.randint(5, 25)
        body = "\n".join(_rand_lines(rng, n)) + "\n"
        for n_keep in (1, 3, 5, 10):
            out.append((("-n", str(n_keep)), body))
    return out


def _probe_tail(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    return _probe_head(rng)  # same probes, different semantics


# ─── cut ────────────────────────────────────────────────────────────────


def _probe_cut(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(4):
        n_lines = rng.randint(3, 8)
        lines = [",".join(_rand_lines(rng, 3, 2, 5)) for _ in range(n_lines)]
        body = "\n".join(lines) + "\n"
        for fn in ("1", "2", "1,3", "2,3"):
            out.append((("-d", ",", "-f", fn), body))
    return out


# ─── grep ───────────────────────────────────────────────────────────────


# ─── jq: JSON transformations ──────────────────────────────────────────


def _probe_jq(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    # Simple identity.
    for _ in range(2):
        obj = {"x": rng.randint(1, 100), "y": rng.randint(1, 100)}
        out.append((("-c", "."), json.dumps(obj) + "\n"))
    # Field projection.
    for _ in range(3):
        obj = {"name": rng.choice(["alice", "bob", "carol"]),
               "age": rng.randint(20, 80),
               "city": rng.choice(["nyc", "sf", "la"])}
        out.append((("-c", ".name"), json.dumps(obj) + "\n"))
    # Array length.
    for _ in range(3):
        arr = [rng.randint(1, 50) for _ in range(rng.randint(2, 6))]
        out.append((("-c", "length"), json.dumps(arr) + "\n"))
    # Sum of array.
    for _ in range(2):
        arr = [rng.randint(1, 20) for _ in range(rng.randint(2, 5))]
        out.append((("-c", "add"), json.dumps(arr) + "\n"))
    return out


def _impl_jq(args: Tuple[str, ...]) -> str:
    # Parse the filter (last non-flag arg).
    filt = [a for a in args if not a.startswith("-")][-1] if args else "."
    if filt == ".":
        return (
            "def solve(stdin: str) -> str:\n"
            "    import json\n"
            "    obj = json.loads(stdin)\n"
            "    return json.dumps(obj, separators=(',', ':')) + '\\n'\n"
        )
    if filt.startswith(".") and filt[1:].isidentifier():
        field = filt[1:]
        return (
            "def solve(stdin: str) -> str:\n"
            "    import json\n"
            "    obj = json.loads(stdin)\n"
            f"    v = obj.get({field!r})\n"
            "    return json.dumps(v, separators=(',', ':')) + '\\n'\n"
        )
    if filt == "length":
        return (
            "def solve(stdin: str) -> str:\n"
            "    import json\n"
            "    obj = json.loads(stdin)\n"
            "    return f'{len(obj)}\\n'\n"
        )
    if filt == "add":
        return (
            "def solve(stdin: str) -> str:\n"
            "    import json\n"
            "    arr = json.loads(stdin)\n"
            "    return f'{sum(arr)}\\n'\n"
        )
    # Fallback: unsupported filter, emit identity.
    return (
        "def solve(stdin: str) -> str:\n"
        "    import json\n"
        "    return json.dumps(json.loads(stdin), separators=(',', ':')) + '\\n'\n"
    )


# ─── base64: encode/decode ─────────────────────────────────────────────


def _probe_base64(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(6):
        body = "".join(rng.choices(string.ascii_letters + string.digits + " .,!?",
                                      k=rng.randint(10, 50)))
        out.append(((), body))
    return out


def _impl_base64(args: Tuple[str, ...]) -> str:
    return (
        "def solve(stdin: str) -> str:\n"
        "    import base64\n"
        "    return base64.b64encode(stdin.encode()).decode() + '\\n'\n"
    )


def _probe_grep(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    common_words = ["hello", "world", "apple", "pear", "fizz", "buzz", "test"]
    for _ in range(5):
        # Mix common + random words so grep sees both hits and misses.
        words = _rand_lines(rng, rng.randint(4, 10))
        for _ in range(rng.randint(1, 3)):
            words[rng.randint(0, len(words) - 1)] = rng.choice(common_words)
        body = "\n".join(words) + "\n"
        pattern = rng.choice(common_words)
        out.append(((pattern,), body))
    return out


# ─── sha256sum / md5sum: hash digests ──────────────────────────────────


def _probe_hash(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(8):
        body = "".join(rng.choices(string.printable, k=rng.randint(5, 80)))
        out.append(((), body))
    return out


def _impl_sha256sum(args: Tuple[str, ...]) -> str:
    return (
        "def solve(stdin: str) -> str:\n"
        "    import hashlib\n"
        "    h = hashlib.sha256(stdin.encode()).hexdigest()\n"
        "    return f'{h}  -\\n'\n"
    )


def _impl_md5sum(args: Tuple[str, ...]) -> str:
    return (
        "def solve(stdin: str) -> str:\n"
        "    import hashlib\n"
        "    h = hashlib.md5(stdin.encode()).hexdigest()\n"
        "    return f'{h}  -\\n'\n"
    )


# ─── tr: character replacement ─────────────────────────────────────────


def _probe_tr(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    common = "hello world the quick brown fox jumps over lazy dog"
    for _ in range(4):
        # lowercase → uppercase
        out.append((("a-z", "A-Z"),
                     common + "\n"))
    for _ in range(3):
        # delete a character class
        body = "".join(rng.choices(string.ascii_letters + string.digits + " ",
                                      k=rng.randint(20, 50))) + "\n"
        out.append((("-d", "0-9"), body))
    for _ in range(3):
        # squeeze runs
        body = "aaa" + "".join(rng.choices(string.ascii_lowercase, k=20)) + "bbb\n"
        out.append((("-s", "a-z"), body))
    return out


def _impl_tr(args: Tuple[str, ...]) -> str:
    # Supported modes: (FROM, TO) translate, ("-d", SET) delete, ("-s", SET) squeeze.
    if len(args) == 2 and not args[0].startswith("-"):
        return _impl_tr_translate(args[0], args[1])
    if args and args[0] == "-d" and len(args) >= 2:
        return _impl_tr_delete(args[1])
    if args and args[0] == "-s" and len(args) >= 2:
        return _impl_tr_squeeze(args[1])
    return "def solve(stdin: str) -> str:\n    return stdin\n"


def _expand_tr_set(s: str) -> str:
    """Expand things like 'a-z' into 'abcdefghijklmnopqrstuvwxyz'."""
    result = []
    i = 0
    while i < len(s):
        if i + 2 < len(s) and s[i + 1] == "-":
            start, end = ord(s[i]), ord(s[i + 2])
            result.extend(chr(c) for c in range(start, end + 1))
            i += 3
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _impl_tr_translate(src: str, dst: str) -> str:
    src_e = _expand_tr_set(src)
    dst_e = _expand_tr_set(dst)
    return (
        "def solve(stdin: str) -> str:\n"
        f"    return stdin.translate(str.maketrans({src_e!r}, {dst_e!r}))\n"
    )


def _impl_tr_delete(setspec: str) -> str:
    expanded = _expand_tr_set(setspec)
    return (
        "def solve(stdin: str) -> str:\n"
        f"    return stdin.translate(str.maketrans('', '', {expanded!r}))\n"
    )


def _impl_tr_squeeze(setspec: str) -> str:
    expanded = _expand_tr_set(setspec)
    return (
        "def solve(stdin: str) -> str:\n"
        f"    targets = set({expanded!r})\n"
        "    out = []\n"
        "    prev = None\n"
        "    for ch in stdin:\n"
        "        if ch in targets and ch == prev:\n"
        "            continue\n"
        "        out.append(ch)\n"
        "        prev = ch\n"
        "    return ''.join(out)\n"
    )


# ─── rev: reverse each line ────────────────────────────────────────────


def _probe_rev(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(8):
        lines = _rand_lines(rng, rng.randint(2, 6))
        out.append(((), "\n".join(lines) + "\n"))
    return out


def _impl_rev(args: Tuple[str, ...]) -> str:
    # Note: BSD rev (macOS) reverses each line including the trailing \n-bounded
    # chunk. Python needs to reverse each newline-separated line.
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.split('\\n')\n"
        "    reversed_lines = [line[::-1] for line in lines]\n"
        "    return '\\n'.join(reversed_lines)\n"
    )


# ─── fold: wrap lines at N columns ─────────────────────────────────────


def _probe_fold(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(8):
        long_line = "".join(rng.choices(string.ascii_lowercase,
                                           k=rng.randint(30, 100)))
        for width in (10, 20, 40):
            out.append((("-w", str(width)), long_line + "\n"))
    return out


def _impl_fold(args: Tuple[str, ...]) -> str:
    width = 80
    if "-w" in args:
        idx = args.index("-w")
        if idx + 1 < len(args):
            try: width = int(args[idx + 1])
            except ValueError: pass
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.split('\\n')\n"
        "    out = []\n"
        "    for line in lines:\n"
        f"        w = {width}\n"
        "        if not line:\n"
        "            out.append(line); continue\n"
        "        for i in range(0, len(line), w):\n"
        "            out.append(line[i:i+w])\n"
        "    return '\\n'.join(out)\n"
    )


# ─── xxd: hex dump ─────────────────────────────────────────────────────


def _probe_xxd(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(6):
        body = "".join(rng.choices(string.ascii_letters + string.digits + " \n.,",
                                      k=rng.randint(5, 60)))
        # -p plain postscript (hex only, no offsets) is the deterministic mode.
        out.append((("-p",), body))
    return out


def _impl_xxd(args: Tuple[str, ...]) -> str:
    plain = "-p" in args
    if plain:
        return (
            "def solve(stdin: str) -> str:\n"
            "    data = stdin.encode()\n"
            "    hex_str = data.hex()\n"
            "    lines = [hex_str[i:i+60] for i in range(0, len(hex_str), 60)]\n"
            "    return ('\\n'.join(lines) + '\\n') if lines else ''\n"
        )
    return "def solve(stdin: str) -> str:\n    return stdin\n"


# ─── base32 ────────────────────────────────────────────────────────────


def _probe_base32(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    return _probe_base64(rng)


def _impl_base32(args: Tuple[str, ...]) -> str:
    return (
        "def solve(stdin: str) -> str:\n"
        "    import base64\n"
        "    encoded = base64.b32encode(stdin.encode()).decode()\n"
        "    lines = [encoded[i:i+76] for i in range(0, len(encoded), 76)]\n"
        "    return '\\n'.join(lines) + '\\n'\n"
    )


# ─── gzip -9 | base64 pipeline: deterministic compression ───────────────


def _probe_gzip(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    # gzip output isn't text-safe so we pipe through base64 inside
    # run_binary. Handled specially below.
    out = []
    for _ in range(6):
        body = "".join(rng.choices(string.ascii_letters + " ",
                                     k=rng.randint(10, 100)))
        out.append((("-n", "-9"), body))
    return out


def _impl_gzip(args: Tuple[str, ...]) -> str:
    """gzip -n -9 is deterministic (no timestamp, max compression).
    We base64-encode the binary output to keep it text-safe in the cache."""
    return (
        "def solve(stdin: str) -> str:\n"
        "    import gzip, base64\n"
        "    compressed = gzip.compress(stdin.encode(), compresslevel=9, mtime=0)\n"
        "    return base64.b64encode(compressed).decode() + '\\n'\n"
    )


# ─── sed: simple substitution ──────────────────────────────────────────


def _probe_sed(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(5):
        body = "\n".join(_rand_lines(rng, rng.randint(3, 6))) + "\n"
        # Simple s/X/Y/g substitution with literal single chars.
        src = rng.choice(["a", "e", "i", "o"])
        dst = rng.choice(["X", "0", "!"])
        out.append((("-e", f"s/{src}/{dst}/g"), body))
    return out


def _impl_sed(args: Tuple[str, ...]) -> str:
    """Support only literal-char s/X/Y/g for determinism; fuzz stays
    in-scope. For anything beyond, emit identity (caller re-runs
    harvest with safer probes)."""
    import re as _re
    # Grab the expression after -e, if any.
    expr = None
    if "-e" in args:
        idx = args.index("-e")
        if idx + 1 < len(args):
            expr = args[idx + 1]
    if expr:
        m = _re.match(r"^s/([^/])/([^/]*)/g$", expr)
        if m:
            src, dst = m.group(1), m.group(2)
            return (
                "def solve(stdin: str) -> str:\n"
                f"    return stdin.replace({src!r}, {dst!r})\n"
            )
    return "def solve(stdin: str) -> str:\n    return stdin\n"


# ─── awk: print $N / sum column ────────────────────────────────────────


def _probe_awk(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    # Print column N.
    for _ in range(4):
        n_lines = rng.randint(3, 6)
        lines = [" ".join(_rand_lines(rng, 3)) for _ in range(n_lines)]
        body = "\n".join(lines) + "\n"
        field = rng.choice([1, 2, 3])
        out.append(((f"{{print ${field}}}",), body))
    # Sum numeric column 1.
    for _ in range(3):
        n_lines = rng.randint(3, 7)
        body = "\n".join(f"{rng.randint(1,100)} {rng.choice(['a','b'])}"
                          for _ in range(n_lines)) + "\n"
        out.append((("{sum+=$1} END {print sum}",), body))
    return out


def _impl_awk(args: Tuple[str, ...]) -> str:
    """Support only the two probe shapes: `{print $N}` and
    `{sum+=$1} END {print sum}`. Other programs emit identity."""
    import re as _re
    expr = args[0] if args else ""
    m = _re.match(r"^\{print \$(\d+)\}$", expr)
    if m:
        n = int(m.group(1))
        return (
            "def solve(stdin: str) -> str:\n"
            "    out = []\n"
            "    for line in stdin.splitlines():\n"
            "        parts = line.split()\n"
            f"        idx = {n - 1}\n"
            "        if idx < len(parts):\n"
            "            out.append(parts[idx])\n"
            "        else:\n"
            "            out.append('')\n"
            "    return ('\\n'.join(out) + '\\n') if out else ''\n"
        )
    if expr == "{sum+=$1} END {print sum}":
        return (
            "def solve(stdin: str) -> str:\n"
            "    total = 0\n"
            "    for line in stdin.splitlines():\n"
            "        parts = line.split()\n"
            "        if parts:\n"
            "            try: total += int(parts[0])\n"
            "            except ValueError: pass\n"
            "    # awk prints integer totals without decimals.\n"
            "    return f'{total}\\n'\n"
        )
    return "def solve(stdin: str) -> str:\n    return stdin\n"


# ─── cat: identity pass-through ────────────────────────────────────────


def _probe_cat(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(6):
        body = "\n".join(_rand_lines(rng, rng.randint(1, 8))) + "\n"
        out.append(((), body))
    return out


def _impl_cat(args: Tuple[str, ...]) -> str:
    return "def solve(stdin: str) -> str:\n    return stdin\n"


# ─── tac: reverse line order ───────────────────────────────────────────


def _probe_tac(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(6):
        n = rng.randint(2, 8)
        lines = _rand_lines(rng, n)
        out.append(((), "\n".join(lines) + "\n"))
    return out


def _impl_tac(args: Tuple[str, ...]) -> str:
    # GNU tac reverses the *order of lines*. Trailing newline handling
    # matches coreutils: if input ends with \n, output ends with \n.
    return (
        "def solve(stdin: str) -> str:\n"
        "    if not stdin: return ''\n"
        "    had_trailing = stdin.endswith('\\n')\n"
        "    body = stdin[:-1] if had_trailing else stdin\n"
        "    lines = body.split('\\n')\n"
        "    out = '\\n'.join(reversed(lines))\n"
        "    return out + ('\\n' if had_trailing else '')\n"
    )


# ─── seq: generate numeric sequence ────────────────────────────────────


def _probe_seq(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(6):
        start = rng.randint(1, 20)
        end = start + rng.randint(1, 20)
        out.append(((str(start), str(end)), ""))
    return out


def _impl_seq(args: Tuple[str, ...]) -> str:
    if len(args) == 2:
        try:
            start = int(args[0]); end = int(args[1])
            return (
                "def solve(stdin: str) -> str:\n"
                f"    return '\\n'.join(str(i) for i in range({start}, {end + 1})) + '\\n'\n"
            )
        except ValueError:
            pass
    return "def solve(stdin: str) -> str:\n    return ''\n"


# ─── paste: merge corresponding lines ─────────────────────────────────
# paste with single stdin merges lines into one tab-separated line.


def _probe_paste(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    # BSD paste needs an explicit file argument; `-` means stdin.
    out = []
    for _ in range(5):
        lines = _rand_lines(rng, rng.randint(2, 6))
        out.append((("-s", "-"), "\n".join(lines) + "\n"))
        out.append((("-s", "-d", ",", "-"), "\n".join(lines) + "\n"))
    return out


def _impl_paste(args: Tuple[str, ...]) -> str:
    delim = "\t"
    if "-d" in args:
        idx = args.index("-d")
        if idx + 1 < len(args):
            delim = args[idx + 1]
    # -s serialise: join all lines with delim.
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        f"    return {delim!r}.join(lines) + '\\n'\n"
    )


# ─── expand: tabs → spaces ─────────────────────────────────────────────


def _probe_expand(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(5):
        parts = _rand_lines(rng, 3, 3, 6)
        body = "\t".join(parts) + "\n"
        # default tabstop 8
        out.append(((), body))
        # smaller tabstop
        out.append((("-t", "4"), body))
    return out


def _impl_expand(args: Tuple[str, ...]) -> str:
    tabstop = 8
    if "-t" in args:
        idx = args.index("-t")
        if idx + 1 < len(args):
            try: tabstop = int(args[idx + 1])
            except ValueError: pass
    return (
        "def solve(stdin: str) -> str:\n"
        f"    return stdin.expandtabs({tabstop})\n"
    )


# ─── nl: number lines ──────────────────────────────────────────────────
# Default: right-justify number in 6 cols + \t, skip blanks, restart.
# Too many format options to match exactly across BSD/GNU; use -ba
# (number all lines) which is more portable.


def _probe_nl(rng: random.Random) -> List[Tuple[Tuple[str, ...], str]]:
    out = []
    for _ in range(4):
        n = rng.randint(2, 6)
        body = "\n".join(_rand_lines(rng, n)) + "\n"
        out.append((("-ba",), body))
    return out


def _impl_nl(args: Tuple[str, ...]) -> str:
    return (
        "def solve(stdin: str) -> str:\n"
        "    had_trailing = stdin.endswith('\\n')\n"
        "    body = stdin[:-1] if had_trailing else stdin\n"
        "    lines = body.split('\\n') if body else []\n"
        "    numbered = [f'{i:>6}\\t{line}' for i, line in enumerate(lines, 1)]\n"
        "    out = '\\n'.join(numbered)\n"
        "    return out + ('\\n' if had_trailing else '')\n"
    )


# ─── Tool registry ──────────────────────────────────────────────────────


TOOLS: Dict[str, dict] = {
    "sort":    {"probe": _probe_sort,    "bin": "/usr/bin/sort"},
    "uniq":    {"probe": _probe_uniq,    "bin": "/usr/bin/uniq"},
    "wc":      {"probe": _probe_wc,      "bin": "/usr/bin/wc"},
    "head":    {"probe": _probe_head,    "bin": "/usr/bin/head"},
    "tail":    {"probe": _probe_tail,    "bin": "/usr/bin/tail"},
    "cut":     {"probe": _probe_cut,     "bin": "/usr/bin/cut"},
    "grep":    {"probe": _probe_grep,    "bin": "/usr/bin/grep"},
    "jq":      {"probe": _probe_jq,      "bin": "/usr/bin/jq"},
    "base64":  {"probe": _probe_base64,  "bin": "/usr/bin/base64"},
    "sha256sum": {"probe": _probe_hash,  "bin": "/sbin/sha256sum"},
    "md5sum":  {"probe": _probe_hash,    "bin": "/sbin/md5sum"},
    "tr":      {"probe": _probe_tr,      "bin": "/usr/bin/tr"},
    "rev":     {"probe": _probe_rev,     "bin": "/usr/bin/rev"},
    "fold":    {"probe": _probe_fold,    "bin": "/usr/bin/fold"},
    "xxd":     {"probe": _probe_xxd,     "bin": "/usr/bin/xxd"},
    "base32":  {"probe": _probe_base32,  "bin": "/opt/homebrew/bin/base32"},
    # gzip header-byte compatibility across implementations is unreliable
    # (OS field, XFL bit, etc.) — drop rather than ship faithless rows.
    "sed":     {"probe": _probe_sed,     "bin": "/usr/bin/sed"},
    "awk":     {"probe": _probe_awk,     "bin": "/usr/bin/awk"},
    "cat":     {"probe": _probe_cat,     "bin": "/bin/cat"},
    "tac":     {"probe": _probe_tac,     "bin": "/opt/homebrew/bin/tac"},
    "seq":     {"probe": _probe_seq,     "bin": "/usr/bin/seq"},
    "paste":   {"probe": _probe_paste,   "bin": "/usr/bin/paste"},
    "expand":  {"probe": _probe_expand,  "bin": "/usr/bin/expand"},
    "nl":      {"probe": _probe_nl,      "bin": "/usr/bin/nl"},
}


# ─── Running + recording ────────────────────────────────────────────────


def run_binary(binary: str, args: Tuple[str, ...], stdin: str,
                timeout_s: int = 5) -> Tuple[int, str]:
    """Run binary with given args + stdin, return (exit_code, stdout).

    Forces LC_ALL=C so string-collation binaries (sort, uniq, tr) use
    byte ordering that matches Python's codepoint comparisons. Without
    this, BSD sort on macOS produces locale-aware orderings that our
    Python reimpls would falsely diverge from.

    For gzip specifically, stdout is binary — we b64-encode it so the
    cache stays text-safe. Our `_impl_gzip` emits the same encoding."""
    import base64 as _base64
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    tool_name = Path(binary).name
    is_binary_out = tool_name in ("gzip",)
    try:
        if is_binary_out:
            r = subprocess.run(
                [binary, *args],
                input=stdin.encode(),
                capture_output=True, timeout=timeout_s, env=env,
            )
            # Encode the binary stdout as base64 + trailing newline to
            # match _impl_gzip's output shape.
            encoded = _base64.b64encode(r.stdout).decode() + "\n"
            return (r.returncode, encoded)
        r = subprocess.run(
            [binary, *args],
            input=stdin,
            capture_output=True, text=True, timeout=timeout_s,
            env=env,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return (-1, f"[error: {e}]")
    return (r.returncode, r.stdout)


def fingerprint_row(tool: str, args: Tuple[str, ...], stdin: str) -> str:
    payload = f"{tool}|{'|'.join(args)}|{stdin}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def reference_python(tool: str, args: Tuple[str, ...], stdin: str,
                      stdout: str) -> str:
    """Emit a self-contained Python reimplementation of the utility for
    the given args. This is the gold-standard training target for the
    distillation cron — the model learns to emit the actual algorithm,
    not a subprocess wrapper."""
    return _IMPLS[tool](args)


def _impl_sort(args: Tuple[str, ...]) -> str:
    numeric = "-n" in args
    reverse = "-r" in args
    rev = "True" if reverse else "False"
    if numeric:
        # sort -n: BSD sort parses a leading numeric prefix and treats
        # the rest as 0 (e.g. "6b2JD" sorts as 6, not 0). Secondary key
        # is the line itself for tiebreaks. Both behaviours surfaced
        # by fuzzing — a strict "parse full string or 0" was wrong.
        key_block = (
            "    import re as _re\n"
            "    _NUMPFX = _re.compile(r'^\\s*([-+]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))')\n"
            "    def _k(s):\n"
            "        m = _NUMPFX.match(s)\n"
            "        try: return (float(m.group(1)) if m else 0.0, s)\n"
            "        except (ValueError, AttributeError): return (0.0, s)\n"
        )
        return (
            "def solve(stdin: str) -> str:\n"
            "    lines = stdin.splitlines()\n"
            + key_block +
            f"    result = sorted(lines, key=_k, reverse={rev})\n"
            "    return ('\\n'.join(result) + '\\n') if lines else ''\n"
        )
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        f"    result = sorted(lines, reverse={rev})\n"
        "    return ('\\n'.join(result) + '\\n') if lines else ''\n"
    )


def _impl_uniq(args: Tuple[str, ...]) -> str:
    count = "-c" in args
    if count:
        return (
            "def solve(stdin: str) -> str:\n"
            "    lines = stdin.splitlines()\n"
            "    out = []\n"
            "    for line in lines:\n"
            "        if out and out[-1][1] == line:\n"
            "            out[-1] = (out[-1][0] + 1, line)\n"
            "        else:\n"
            "            out.append((1, line))\n"
            "    formatted = [f'{n:>4} {s}' for n, s in out]\n"
            "    return ('\\n'.join(formatted) + '\\n') if formatted else ''\n"
        )
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        "    out = []\n"
        "    for line in lines:\n"
        "        if not out or out[-1] != line:\n"
        "            out.append(line)\n"
        "    return ('\\n'.join(out) + '\\n') if out else ''\n"
    )


def _impl_wc(args: Tuple[str, ...]) -> str:
    lines_flag = "-l" in args
    words_flag = "-w" in args
    bytes_flag = "-c" in args
    if lines_flag:
        return (
            "def solve(stdin: str) -> str:\n"
            "    return f'{stdin.count(chr(10)):>8}\\n'\n"
        )
    if words_flag:
        return (
            "def solve(stdin: str) -> str:\n"
            "    return f'{len(stdin.split()):>8}\\n'\n"
        )
    if bytes_flag:
        return (
            "def solve(stdin: str) -> str:\n"
            "    return f'{len(stdin.encode()):>8}\\n'\n"
        )
    # Default: lines, words, bytes.
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.count(chr(10))\n"
        "    words = len(stdin.split())\n"
        "    byts  = len(stdin.encode())\n"
        "    return f'{lines:>8}{words:>8}{byts:>8}\\n'\n"
    )


def _impl_head(args: Tuple[str, ...]) -> str:
    n = 10
    if "-n" in args:
        idx = args.index("-n")
        if idx + 1 < len(args):
            try: n = int(args[idx + 1])
            except ValueError: pass
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        f"    return ('\\n'.join(lines[:{n}]) + '\\n') if lines else ''\n"
    )


def _impl_tail(args: Tuple[str, ...]) -> str:
    n = 10
    if "-n" in args:
        idx = args.index("-n")
        if idx + 1 < len(args):
            try: n = int(args[idx + 1])
            except ValueError: pass
    # Guard against Python's `list[-0:]` == `list[:]` gotcha.
    slice_expr = "[:0]" if n == 0 else f"[-{n}:]"
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        f"    kept = lines{slice_expr}\n"
        "    return ('\\n'.join(kept) + '\\n') if kept else ''\n"
    )


def _impl_cut(args: Tuple[str, ...]) -> str:
    delim = ","
    fields_spec = "1"
    if "-d" in args:
        idx = args.index("-d")
        if idx + 1 < len(args):
            delim = args[idx + 1]
    if "-f" in args:
        idx = args.index("-f")
        if idx + 1 < len(args):
            fields_spec = args[idx + 1]
    # Parse fields_spec like "1,3" → [0, 2] (cut is 1-indexed).
    fields = [int(f) - 1 for f in fields_spec.split(",")]
    # BSD/GNU cut semantics: if the line contains no delimiter and -s
    # is not given, the whole line is emitted (as if it were field 1
    # standing alone). Surfaced by fuzzing.
    return (
        "def solve(stdin: str) -> str:\n"
        "    lines = stdin.splitlines()\n"
        "    out = []\n"
        f"    delim = {delim!r}\n"
        f"    fields = {fields}\n"
        "    for line in lines:\n"
        "        if delim not in line:\n"
        "            out.append(line)\n"
        "            continue\n"
        "        parts = line.split(delim)\n"
        "        picked = [parts[i] for i in fields if i < len(parts)]\n"
        "        out.append(delim.join(picked))\n"
        "    return ('\\n'.join(out) + '\\n') if out else ''\n"
    )


def _impl_grep(args: Tuple[str, ...]) -> str:
    pattern = args[0] if args else ""
    return (
        "def solve(stdin: str) -> str:\n"
        f"    pattern = {pattern!r}\n"
        "    matches = [line for line in stdin.splitlines() if pattern in line]\n"
        "    return ('\\n'.join(matches) + '\\n') if matches else ''\n"
    )


_IMPLS = {
    "sort": _impl_sort, "uniq": _impl_uniq, "wc": _impl_wc,
    "head": _impl_head, "tail": _impl_tail, "cut": _impl_cut,
    "grep": _impl_grep, "jq": _impl_jq, "base64": _impl_base64,
    "sha256sum": _impl_sha256sum, "md5sum": _impl_md5sum,
    "tr": _impl_tr, "rev": _impl_rev, "fold": _impl_fold,
    "xxd": _impl_xxd, "base32": _impl_base32, "gzip": _impl_gzip,
    "sed": _impl_sed, "awk": _impl_awk,
    "cat": _impl_cat, "tac": _impl_tac, "seq": _impl_seq,
    "paste": _impl_paste, "expand": _impl_expand, "nl": _impl_nl,
}


def record_row(cache_path: Path, fp: str, tool: str,
                args: Tuple[str, ...], stdin: str, stdout: str) -> None:
    """Append a 6-col cache row preserving existing rows. Uses the same
    TSV schema as our LLM-solve cache so downstream tools read it
    uniformly."""
    code = reference_python(tool, args, stdin, stdout)
    example = [{"inputs": [stdin], "expected": stdout}]
    import time as _time
    rows = _load_all()  # reads from NSYNTH_LLM_CACHE_PATH
    rows[fp] = {
        "model": f"binary:{tool}",
        "success_count": 1,
        "last_used_at": int(_time.time()),
        "code": code,
        "examples": example,
    }
    # Write back via the cache module's own _save_all.
    from llm_solution_cache import _save_all
    _save_all(rows)


def harvest_tool(tool: str, n: int, seed: int = 42,
                  verbose: bool = False) -> int:
    """Run n probes for the given tool, record results. Returns count
    of successfully recorded rows."""
    if tool not in TOOLS:
        print(f"[harvest] unknown tool: {tool}", file=sys.stderr)
        return 0
    rng = random.Random(seed)
    spec = TOOLS[tool]
    binary = spec["bin"]
    if not Path(binary).exists():
        # Some macOS locations differ — try /usr/local/bin, /opt/homebrew/bin.
        for alt in ("/usr/local/bin", "/opt/homebrew/bin", "/bin"):
            p = Path(alt) / tool
            if p.exists():
                binary = str(p); break
        else:
            print(f"[harvest] binary not found for {tool}", file=sys.stderr)
            return 0

    probes = spec["probe"](rng)
    # Sample up to n (or use all if fewer than n generated).
    if len(probes) > n:
        probes = rng.sample(probes, n)

    count = 0
    # grep returns rc=1 when pattern matches nothing — that's still a
    # legitimate I/O pair (empty stdout). Accept it as valid.
    ok_codes = {0, 1} if tool == "grep" else {0}
    for args, stdin in probes:
        rc, stdout = run_binary(binary, args, stdin)
        if rc not in ok_codes:
            if verbose:
                print(f"  [{tool} {args}] rc={rc} skip")
            continue
        fp = fingerprint_row(tool, args, stdin)
        record_row(Path(os.environ.get("NSYNTH_LLM_CACHE_PATH", "")),
                    fp, tool, args, stdin, stdout)
        count += 1
        if verbose:
            print(f"  [{tool} {args}] ok (out={len(stdout)}b fp={fp[:8]}…)")
    return count


def emit_distillation_jsonl(out_path: Path) -> int:
    """Scan the cache for rows with model prefix "binary:" and emit
    JSONL suitable for the weekly LoRA fine-tune."""
    rows = _load_all()
    n = 0
    with out_path.open("w") as f:
        for fp, r in rows.items():
            if not r.get("model", "").startswith("binary:"):
                continue
            tool = r["model"].split(":", 1)[1]
            ex = r.get("examples", [])
            if not ex:
                continue
            stdin = ex[0]["inputs"][0]
            expected = ex[0]["expected"]
            prompt = (
                f"Reimplement the Unix utility `{tool}` in Python. "
                f"Given the following stdin input, your `solve(stdin)` "
                f"function should return the stdout that `{tool}` "
                f"would produce.\n\n"
                f"Input:\n```\n{stdin[:600]}\n```\n\n"
                f"Expected output:\n```\n{expected[:600]}\n```\n\n"
                f"Write `def solve(stdin: str) -> str:` now."
            )
            f.write(json.dumps({"prompt": prompt,
                                "completion": r["code"]}) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tool", default=None,
                    help="Single tool to harvest (sort/uniq/wc/head/tail/cut/grep).")
    ap.add_argument("--all", action="store_true",
                    help="Harvest every tool in the registry.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of probes per tool (default 20).")
    ap.add_argument("--cache", default=None,
                    help="Cache path (sets NSYNTH_LLM_CACHE_PATH).")
    ap.add_argument("--emit-jsonl", default=None,
                    help="Write distillation JSONL after harvest.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.cache:
        os.environ["NSYNTH_LLM_CACHE_PATH"] = args.cache
    if not args.tool and not args.all:
        print("specify --tool NAME or --all", file=sys.stderr); sys.exit(2)

    tools = list(TOOLS.keys()) if args.all else [args.tool]
    total = 0
    for t in tools:
        got = harvest_tool(t, args.n, seed=args.seed, verbose=args.verbose)
        print(f"[harvest] {t}: +{got} rows")
        total += got
    print(f"[harvest] TOTAL {total} rows recorded to "
          f"{os.environ.get('NSYNTH_LLM_CACHE_PATH', 'default')}")

    if args.emit_jsonl:
        n = emit_distillation_jsonl(Path(args.emit_jsonl))
        print(f"[harvest] wrote {args.emit_jsonl} ({n} training pairs)")


if __name__ == "__main__":
    main()
