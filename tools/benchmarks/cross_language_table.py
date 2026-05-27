#!/usr/bin/env python3
"""
Cross-language correctness table.

For every problem in humaneval_lite.jsonl, generate Python + Rust +
TypeScript via nsynth_codegen, compile + run each, record pass/fail
per language. Emit artifacts/CROSS_LANGUAGE_CORRECTNESS.md.

Any ✗ in the table is a transpiler bug — the synthesized Mog is the
same, so differences can only come from how the transpiler expresses
the program in the target language.

Usage:
    python3 tools/benchmarks/cross_language_table.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional


def load_problems(path: Path) -> List[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def detect_runners() -> tuple[bool, bool, Optional[str]]:
    """Return (rustc_available, any_ts_runner, ts_runner_name)."""
    rust_ok = shutil.which("rustc") is not None
    ts_runner = None
    for cand in ("bun", "deno", "ts-node"):
        if shutil.which(cand):
            ts_runner = cand
            break
    return rust_ok, ts_runner is not None, ts_runner


def run_python(code: str, problem: dict) -> tuple[bool, str]:
    ns: dict = {}
    try:
        exec(code, ns)
    except Exception as e:
        return (False, f"exec: {e!r}"[:80])
    fn = ns.get(problem["name"])
    if fn is None:
        return (False, f"no fn {problem['name']}")
    for case in problem["test_cases"]:
        *args, expected = case
        try:
            got = fn(*args)
        except Exception as e:
            return (False, f"call: {e!r}"[:60])
        if got != expected:
            return (False, f"wrong {args}→{got} exp {expected}"[:60])
    return (True, "")


def run_rust(code: str, problem: dict) -> tuple[bool, str]:
    """Compile + execute the generated Rust. Requires `rustc` on PATH."""
    tmpdir = tempfile.mkdtemp(prefix="xlang_rs_")
    try:
        src = Path(tmpdir) / "main.rs"
        main_body = ["fn main() {", "    let mut p = 0i64;", "    let mut f = 0i64;"]
        for case in problem["test_cases"]:
            *args, expected = case
            args_str = ", ".join(f"{a}_i64" for a in args)
            main_body.append(f"    let got = {problem['name']}({args_str});")
            main_body.append(
                f"    if got == {expected}_i64 {{ p += 1; }} else {{ f += 1; eprintln!(\"wrong {args} got {{}} exp {expected}\", got); }}"
            )
        main_body.append("    if f == 0 { println!(\"OK\"); } else { println!(\"FAIL {}\", f); }")
        main_body.append("}")
        src.write_text(code + "\n" + "\n".join(main_body) + "\n")
        bin_path = Path(tmpdir) / "bin"
        compile_proc = subprocess.run(
            ["rustc", str(src), "-o", str(bin_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if compile_proc.returncode != 0:
            return (False, f"compile: {compile_proc.stderr.strip()[:60]}")
        run_proc = subprocess.run(
            [str(bin_path)], capture_output=True, text=True, timeout=10
        )
        if run_proc.stdout.strip() == "OK":
            return (True, "")
        return (False, run_proc.stdout.strip()[:60] or run_proc.stderr.strip()[:60])
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, f"err: {e!r}"[:60])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_typescript(code: str, problem: dict, runner: str) -> tuple[bool, str]:
    tmpdir = tempfile.mkdtemp(prefix="xlang_ts_")
    try:
        src = Path(tmpdir) / "fn.ts"
        main_body = ["let p = 0; let f = 0;"]
        for case in problem["test_cases"]:
            *args, expected = case
            args_str = ", ".join(str(a) for a in args)
            main_body.append(f"{{ const got = {problem['name']}({args_str});")
            main_body.append(
                f"if (got === {expected}) p++; else {{ f++; console.error(`wrong got ${{got}} exp {expected}`); }} }}"
            )
        main_body.append(
            'if (f === 0) console.log("OK"); else console.log(`FAIL ${f}`);'
        )
        src.write_text(code + "\n\n" + "\n".join(main_body) + "\n")
        if runner == "bun":
            cmd = ["bun", "run", str(src)]
        elif runner == "deno":
            cmd = ["deno", "run", "--allow-read", str(src)]
        elif runner == "ts-node":
            cmd = ["ts-node", "--esm", str(src)]
        else:
            return (False, "no runner")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.stdout.strip() == "OK":
            return (True, "")
        return (False, proc.stdout.strip()[:60] or proc.stderr.strip()[:60])
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, f"err: {e!r}"[:60])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def synthesize_for_lang(
    codegen_bin: Path, problem: dict, lang: str, timeout: int
) -> Optional[str]:
    spec = json.dumps({
        "name": problem["name"],
        "signature": problem["signature"],
        "examples": problem["examples"],
    })
    try:
        proc = subprocess.run(
            [str(codegen_bin), "--lang", lang, "--examples", spec],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default="tools/benchmarks/humaneval_lite.jsonl")
    ap.add_argument("--out", default="artifacts/CROSS_LANGUAGE_CORRECTNESS.md")
    ap.add_argument(
        "--codegen", default="nsynth/target/release/nsynth_codegen"
    )
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--skip-heavy", action="store_true",
                    help="Skip the Rust leg (rustc compile is slow at scale)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    problems = load_problems(Path(args.problems))
    codegen = Path(args.codegen)

    rust_ok, ts_ok, ts_runner = detect_runners()
    if args.skip_heavy:
        rust_ok = False
    print(
        f"[xlang] {len(problems)} problems; rustc={'yes' if rust_ok else 'no'}, "
        f"ts_runner={ts_runner or 'none'}"
    )

    rows: list[dict] = []
    for i, p in enumerate(problems, 1):
        name = p["name"]
        print(f"[{i}/{len(problems)}] {name} ...", end=" ", flush=True)
        row = {"name": name, "py": None, "rs": None, "ts": None,
               "py_note": "", "rs_note": "", "ts_note": ""}

        # Python
        code = synthesize_for_lang(codegen, p, "python", args.timeout)
        if code is None:
            row["py"] = False; row["py_note"] = "synthesis miss"
        else:
            ok, note = run_python(code, p)
            row["py"] = ok; row["py_note"] = note

        # Rust
        if rust_ok:
            code = synthesize_for_lang(codegen, p, "rust", args.timeout)
            if code is None:
                row["rs"] = False; row["rs_note"] = "synthesis miss"
            else:
                ok, note = run_rust(code, p)
                row["rs"] = ok; row["rs_note"] = note
        else:
            row["rs"] = None; row["rs_note"] = "skipped"

        # TypeScript
        if ts_ok:
            code = synthesize_for_lang(codegen, p, "typescript", args.timeout)
            if code is None:
                row["ts"] = False; row["ts_note"] = "synthesis miss"
            else:
                ok, note = run_typescript(code, p, ts_runner)
                row["ts"] = ok; row["ts_note"] = note
        else:
            row["ts"] = None; row["ts_note"] = "no runner"

        rows.append(row)
        mk = lambda v: "✓" if v else ("✗" if v is False else "-")
        print(f"py {mk(row['py'])}  rs {mk(row['rs'])}  ts {mk(row['ts'])}")

    # Emit the markdown table.
    lines: list[str] = []
    lines.append("# Cross-Language Correctness")
    lines.append("")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} — "
        f"{len(problems)} problems from {Path(args.problems).name}."
    )
    lines.append("")
    lines.append(
        "Every row is one HumanEval-lite problem synthesized three times "
        "(python/rust/typescript) and executed against its `test_cases`. "
        "A ✗ where another language has ✓ is a *transpiler* bug, not a "
        "synthesizer bug — the underlying Mog program is identical."
    )
    lines.append("")
    # Totals
    py_pass = sum(1 for r in rows if r["py"])
    rs_pass = sum(1 for r in rows if r["rs"])
    ts_pass = sum(1 for r in rows if r["ts"])
    lines.append(f"- Python: **{py_pass}/{len(rows)}** pass")
    if rust_ok:
        lines.append(f"- Rust: **{rs_pass}/{len(rows)}** pass")
    else:
        lines.append(f"- Rust: skipped (no rustc available)")
    if ts_ok:
        lines.append(f"- TypeScript: **{ts_pass}/{len(rows)}** pass")
    else:
        lines.append(f"- TypeScript: skipped (no runner: install bun / deno / ts-node)")
    lines.append("")
    lines.append("| # | problem | py | rs | ts | py_note | rs_note | ts_note |")
    lines.append("|--:|---------|:--:|:--:|:--:|---------|---------|---------|")
    for i, r in enumerate(rows, 1):
        mk = lambda v: "✓" if v else ("✗" if v is False else "-")
        lines.append(
            f"| {i} | {r['name']} | {mk(r['py'])} | {mk(r['rs'])} | {mk(r['ts'])} | "
            f"{r['py_note'][:30]} | {r['rs_note'][:30]} | {r['ts_note'][:30]} |"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[xlang] wrote {out}")


if __name__ == "__main__":
    main()
