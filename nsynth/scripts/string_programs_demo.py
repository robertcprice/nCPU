#!/usr/bin/env python3
"""nCPU is fully capable of string-driven programs.

Each task below is given only input→output string examples; nSynth synthesizes a
verified Mog program (string in, string out). The fast morphology specialist
handles suffix transduction; the general enumerative synthesizer (string_synth)
handles reverse, case, slicing, field concatenation, separators, initials, and
their compositions. Run:  python scripts/string_programs_demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BIN = Path(__file__).resolve().parent.parent / "target" / "release" / "mog_synth"

TASKS = [
    ("reverse",      "fn f(s: string) -> string",            [("abc", "cba"), ("hello", "olleh"), ("x", "x")]),
    ("uppercase",    "fn f(s: string) -> string",            [("abc", "ABC"), ("heLLo", "HELLO")]),
    ("capitalize",   "fn f(s: string) -> string",            [("cat", "Cat"), ("dog", "Dog"), ("bird", "Bird")]),
    ("drop_last",    "fn f(s: string) -> string",            [("cats", "cat"), ("dogs", "dog"), ("birds", "bird")]),
    ("pluralize",    "fn f(s: string) -> string",            [("cat", "cats"), ("box", "boxes"), ("city", "cities"), ("bus", "buses"), ("toy", "toys")]),
    ("full_name",    "fn f(a: string, b: string) -> string", [(("john", "smith"), "john smith"), (("jane", "doe"), "jane doe")]),
    ("last_first",   "fn f(a: string, b: string) -> string", [(("john", "smith"), "smith, john"), (("jane", "doe"), "doe, jane"), (("amy", "lee"), "lee, amy")]),
    ("initials",     "fn f(a: string, b: string) -> string", [(("john", "smith"), "JS"), (("jane", "doe"), "JD"), (("amy", "lee"), "AL")]),
    ("email",        "fn f(a: string, b: string) -> string", [(("john", "acme"), "john@acme.com"), (("amy", "x"), "amy@x.com")]),
]


def synth(signature, examples):
    rows = []
    for inp, out in examples:
        ins = list(inp) if isinstance(inp, tuple) else [inp]
        rows.append({"inputs": ins, "expected": out})
    payload = json.dumps({"name": "f", "signature": signature, "examples": rows})
    out = subprocess.run([str(BIN), "--problem-json", "-"], input=payload,
                         capture_output=True, text=True, timeout=60).stdout
    return json.loads(out)


def main():
    if not BIN.exists():
        raise SystemExit(f"build nSynth first: {BIN} not found")
    print("nCPU string-driven programs — synthesized from examples, verified\n")
    ok = 0
    for name, sig, exs in TASKS:
        try:
            r = synth(sig, exs)
        except subprocess.TimeoutExpired:
            r = {"success": False, "method": "timeout", "code": None}
        status = "OK " if r["success"] else "—  "
        ok += r["success"]
        body = (r["code"].split("return", 1)[1].rsplit(";", 1)[0].strip()
                if r["success"] else "")
        print(f"  [{status}] {name:12} {r['method']:30} return {body}")
    print(f"\n{ok}/{len(TASKS)} string programs synthesized.")


if __name__ == "__main__":
    main()
