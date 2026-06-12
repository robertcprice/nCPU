#!/usr/bin/env python3
"""Bridge Demo (ROADMAP Rung 8): a synthesized program runs on the GPU computer.

The fused artifact: "the program no human wrote, running on the computer
made of neural networks."

Pipeline (every stage real, no mocks):

  1. SYNTHESIZE   nsynth (Rust gradient synthesizer) is given only I/O
                  examples and produces a Mog program. No human writes code.
  2. TRANSPILE    Mog -> C via a small deterministic rewrite pass
                  (Mog is C-like; only `fn`/`: i64` differ).
  3. GPU COMPILE  The self-hosting C compiler (cc.c, paper section 16) runs
                  ON the rust_metal Metal GPU kernel via ncpu_metal.run_elf
                  and compiles the C source into ARM64 machine code inside
                  the GPU's virtual filesystem.
  4. GPU EXECUTE  The GPU-compiled binary is wrapped in a minimal ELF and
                  executed on the same rust_metal kernel. It prints its
                  results through real write() syscalls captured as stdout.
  5. VERIFY       GPU outputs are checked against (a) the original training
                  examples, (b) nsynth's holdout examples, (c) two inputs
                  the synthesizer NEVER saw, (d) a closed-form ground-truth
                  oracle, (e) a local clang build of the same C source, and
                  (f) the Mog->Python transpile executed locally.

Usage:
    python3 demos/bridge/synthesized_on_gpu.py             # fresh synthesis
    python3 demos/bridge/synthesized_on_gpu.py --quick     # allow nsynth cache

Artifacts:
    artifacts/bridge_demo_result.json
    artifacts/bridge_demo_transcript.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NSYNTH_BIN = PROJECT_ROOT / "nsynth" / "target" / "release" / "mog_synth"
GPU_SRC_DIR = PROJECT_ROOT / "ncpu" / "os" / "gpu" / "src"
CC_SOURCE = PROJECT_ROOT / "ncpu" / "os" / "gpu" / "programs" / "tools" / "cc.c"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CODE_BASE = 0x10000  # cc.c codegen target (CODE_BASE in cc.c)
DATA_BASE = 0x50000  # cc.c codegen data section (DATA_BASE in cc.c)

# ── The problem: sum of squares 1..n, specified ONLY as I/O examples ─────────
PROBLEM_NAME = "sum_of_squares"
TRAIN_EXAMPLES = [(1, 1), (2, 5), (3, 14), (4, 30), (5, 55), (6, 91)]
NSYNTH_HOLDOUTS = [(7, 140), (10, 385)]
# These two inputs are NEVER shown to the synthesizer in any form:
UNSEEN_INPUTS = [12, 20]


def oracle(n: int) -> int:
    """Independent ground truth: closed form n(n+1)(2n+1)/6."""
    return n * (n + 1) * (2 * n + 1) // 6


class Transcript:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


T = Transcript()


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — SYNTHESIZE: I/O examples -> Mog program (nsynth)
# ═════════════════════════════════════════════════════════════════════════════

def synthesize(allow_cache: bool) -> dict:
    problem = {
        "name": PROBLEM_NAME,
        "signature": f"fn {PROBLEM_NAME}(n: i64) -> i64",
        "examples": [{"inputs": [i], "expected": o} for i, o in TRAIN_EXAMPLES],
        "holdouts": [{"inputs": [i], "expected": o} for i, o in NSYNTH_HOLDOUTS],
    }
    env = dict(os.environ)
    if not allow_cache:
        # Force a genuinely fresh synthesis run (no solved-program memoization).
        env["NSYNTH_CACHE_PATH"] = ""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(NSYNTH_BIN), "--problem-json", "-"],
        input=json.dumps(problem),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"nsynth failed (rc={proc.returncode}): {proc.stderr[-500:]}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if not result.get("success") or not result.get("code"):
        raise RuntimeError(f"nsynth could not solve the problem: {result}")
    result["synthesis_seconds"] = round(elapsed, 2)
    return result


def transpile_mog_to_python(mog: str) -> str:
    """Use nsynth's own --transpile python for the local cross-check."""
    proc = subprocess.run(
        [str(NSYNTH_BIN), "--transpile", "python"],
        input=mog,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mog->python transpile failed: {proc.stderr[-300:]}")
    return proc.stdout


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRANSPILE: Mog -> C (deterministic rewrite pass)
# ═════════════════════════════════════════════════════════════════════════════

def mog_to_c(mog: str) -> str:
    """Convert a scalar i64 Mog function to C89.

    Mog is intentionally C-like; for the scalar i64 subset the only
    differences are the `fn name(a: i64) -> i64` signature and typed
    declarations `x: i64 = expr;`. Everything else (while, if, return,
    assignment, arithmetic, comparison) is already valid C.
    """
    out_lines: list[str] = []
    sig_re = re.compile(r"^\s*fn\s+(\w+)\s*\(([^)]*)\)\s*->\s*i64\s*\{\s*$")
    decl_re = re.compile(r"^(\s*)(\w+)\s*:\s*i64\s*=\s*(.+;)\s*$")
    # Mog control-flow conditions have no parentheses; C requires them.
    cond_re = re.compile(r"^(\s*)(\}\s*else\s+if|if|while)\s+(.+?)\s*\{\s*$")
    for line in mog.rstrip().splitlines():
        m = sig_re.match(line)
        if m:
            name, params = m.group(1), m.group(2)
            c_params = []
            for p in params.split(","):
                p = p.strip()
                if not p:
                    continue
                pname = p.split(":")[0].strip()
                c_params.append(f"long {pname}")
            out_lines.append(f"long {name}({', '.join(c_params) or 'void'}) {{")
            continue
        m = decl_re.match(line)
        if m:
            out_lines.append(f"{m.group(1)}long {m.group(2)} = {m.group(3)}")
            continue
        m = cond_re.match(line)
        if m:
            out_lines.append(f"{m.group(1)}{m.group(2)} ({m.group(3)}) {{")
            continue
        out_lines.append(line)
    c_func = "\n".join(out_lines) + "\n"
    # Fail closed: any Mog syntax remaining means the rewrite was incomplete.
    leftovers = [tok for tok in (": i64", "fn ", "->") if tok in c_func]
    if leftovers:
        raise RuntimeError(f"Mog->C transpile incomplete, leftover tokens: {leftovers}")
    return c_func


def build_driver_c(synth_func_c: str, func_name: str, inputs: list[int]) -> str:
    """Wrap the synthesized function in a driver that prints results.

    The driver is the only handwritten code in the binary and contains no
    knowledge of the algorithm — it just calls the synthesized function for
    each test input and prints the result via raw write() syscalls
    (syscall 64), the same protocol BusyBox uses on this kernel.
    """
    assigns = "\n".join(
        f"    inputs[{idx}] = {val};" for idx, val in enumerate(inputs)
    )
    return f"""\
/* AUTO-GENERATED bridge demo program.
 * The function `{func_name}` below was SYNTHESIZED by nsynth from I/O
 * examples (no human wrote it) and machine-transpiled from Mog to C.
 * The driver (print_num/main) is generic test scaffolding.
 */

{synth_func_c}
void print_num(long v) {{
    char buf[24];
    int pos = 23;
    buf[pos] = 10;
    pos = pos - 1;
    if (v == 0) {{ buf[pos] = 48; pos = pos - 1; }}
    while (v > 0) {{
        buf[pos] = 48 + v % 10;
        v = v / 10;
        pos = pos - 1;
    }}
    __syscall(64, 1, buf + pos + 1, 23 - pos);
}}

int main(void) {{
    long inputs[{len(inputs)}];
{assigns}
    int k = 0;
    while (k < {len(inputs)}) {{
        print_num({func_name}(inputs[k]));
        k = k + 1;
    }}
    return 0;
}}
"""


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — GPU COMPILE: self-hosting cc.c compiles the program ON the GPU
# ═════════════════════════════════════════════════════════════════════════════

def cross_compile_cc_elf(out_path: Path) -> None:
    """Host GCC compiles the compiler (Layer 1 of the meta-compilation stack).

    This produces the same ELF that `ncpu.os.gpu.runner.compile_c` builds,
    except we keep the ELF (rust_metal's loader consumes ELF directly)
    instead of objcopy'ing to a raw image.
    """
    cmd = [
        "aarch64-elf-gcc",
        "-nostdlib", "-ffreestanding", "-static", "-O2",
        "-march=armv8-a", "-mgeneral-regs-only",
        f"-T{GPU_SRC_DIR / 'arm64.ld'}",
        f"-I{GPU_SRC_DIR}",
        "-e", "_start",
        str(GPU_SRC_DIR / "arm64_start.S"),
        str(CC_SOURCE),
        "-o", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"host cross-compile of cc.c failed: {proc.stderr[-500:]}")


def gpu_compile(ncpu_metal, cc_elf: Path, c_source: str) -> tuple[bytes, dict]:
    """Run the self-hosting compiler on the rust_metal GPU kernel."""
    res = ncpu_metal.run_elf(
        elf_path=str(cc_elf),
        argv=["cc"],
        max_cycles=300_000_000,
        quiet=True,
        files=[
            ("/tmp/prog.c", c_source.encode()),
            ("/tmp/.cc_args", b"/tmp/prog.c\n/bin/prog\n"),
        ],
    )
    vfs = res.get("vfs_files", {})
    if "/bin/prog" not in vfs:
        raise RuntimeError(
            "GPU compiler did not produce /bin/prog.\n"
            f"cc stdout:\n{res.get('stdout', '')}\nstderr:\n{res.get('stderr', '')}"
        )
    stats = {
        "compile_cycles": res.get("total_cycles"),
        "compile_seconds": round(res.get("elapsed_secs", 0.0), 3),
        "cc_stdout": res.get("stdout", ""),
    }
    return bytes(vfs["/bin/prog"]), stats


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GPU EXECUTE: wrap raw binary in minimal ELF, run on rust_metal
# ═════════════════════════════════════════════════════════════════════════════

def wrap_raw_in_elf(raw: bytes) -> bytes:
    """Wrap cc.c's raw output (code [+ NCCD data section]) in a minimal ELF64.

    cc.c emits position-fixed ARM64 code for CODE_BASE with an optional data
    section tagged `NCCD<u32 size>` for DATA_BASE. rust_metal's loader
    consumes static ET_EXEC ELFs, so we emit one or two PT_LOAD segments.
    """
    nccd = raw.find(b"NCCD")
    if nccd > 0 and nccd + 8 <= len(raw):
        code = raw[:nccd]
        dsize = int.from_bytes(raw[nccd + 4:nccd + 8], "little")
        data = raw[nccd + 8:nccd + 8 + dsize]
    else:
        code, data = raw, b""

    phnum = 2 if data else 1
    ehsize, phentsize, align = 64, 56, 0x1000
    hdr_end = ehsize + phnum * phentsize
    code_off = (hdr_end + align - 1) & ~(align - 1)
    data_off = (code_off + len(code) + align - 1) & ~(align - 1)

    out = bytearray()
    out += b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8  # e_ident: ELF64 LE
    out += struct.pack(
        "<HHIQQQIHHHHHH",
        2,            # e_type   = ET_EXEC
        0xB7,         # e_machine = EM_AARCH64
        1,            # e_version
        CODE_BASE,    # e_entry (cc.c emits _start first)
        ehsize, 0,    # e_phoff, e_shoff
        0,            # e_flags
        ehsize, phentsize, phnum,
        0, 0, 0,      # e_shentsize, e_shnum, e_shstrndx
    )
    out += struct.pack(  # PT_LOAD: code, R+X
        "<IIQQQQQQ", 1, 5, code_off, CODE_BASE, CODE_BASE,
        len(code), len(code), align,
    )
    if data:
        out += struct.pack(  # PT_LOAD: data, R+W
            "<IIQQQQQQ", 1, 6, data_off, DATA_BASE, DATA_BASE,
            len(data), len(data), align,
        )
    out += b"\x00" * (code_off - len(out))
    out += code
    if data:
        out += b"\x00" * (data_off - len(out))
        out += data
    return bytes(out)


def gpu_execute(ncpu_metal, prog_elf: Path) -> tuple[list[int], dict]:
    res = ncpu_metal.run_elf(
        elf_path=str(prog_elf),
        argv=["prog"],
        max_cycles=50_000_000,
        quiet=True,
    )
    stdout = res.get("stdout", "")
    if res.get("stop_reason") not in ("EXIT", "SYSCALL"):
        raise RuntimeError(
            f"GPU execution did not exit cleanly: stop={res.get('stop_reason')} "
            f"exit={res.get('exit_code')} stdout={stdout!r}"
        )
    outputs = [int(line) for line in stdout.split() if line.strip()]
    stats = {
        "exec_cycles": res.get("total_cycles"),
        "exec_seconds": round(res.get("elapsed_secs", 0.0), 4),
        "exit_code": res.get("exit_code"),
        "stop_reason": res.get("stop_reason"),
        "raw_stdout": stdout,
    }
    return outputs, stats


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — VERIFY: local cross-checks (clang + Mog->Python transpile)
# ═════════════════════════════════════════════════════════════════════════════

def local_clang_run(c_source: str, tmpdir: Path) -> list[int]:
    """Sanity-compile the exact same C source with host clang and run it.

    `__syscall(64, fd, buf, len)` is shimmed to write(2) via a macro so the
    GPU source bytes stay untouched.
    """
    shim = (
        "#include <unistd.h>\n"
        "#define __syscall(nr, fd, buf, len) "
        "write((int)(fd), (const void *)(buf), (unsigned long)(len))\n\n"
    )
    src = tmpdir / "local_check.c"
    binp = tmpdir / "local_check"
    src.write_text(shim + c_source)
    proc = subprocess.run(
        ["clang", "-O1", "-o", str(binp), str(src)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"local clang sanity-compile failed: {proc.stderr[-500:]}")
    run = subprocess.run([str(binp)], capture_output=True, text=True, timeout=30)
    return [int(x) for x in run.stdout.split()]


def local_python_transpile_run(py_src: str, func_name: str, inputs: list[int]) -> list[int]:
    namespace: dict = {}
    exec(py_src, namespace)  # noqa: S102 — machine-generated transpile, local check
    fn = namespace[func_name]
    return [fn(i) for i in inputs]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true",
        help="allow nsynth's solved-program cache (skips fresh gradient synthesis)",
    )
    args = parser.parse_args()

    try:
        import ncpu_metal
    except ImportError:
        T.log("FATAL: ncpu_metal not importable. Build kernels/rust_metal with "
              "`maturin develop --release` (or pip install the wheel in "
              "kernels/rust_metal/dist/).")
        return 1

    if not NSYNTH_BIN.exists():
        T.log(f"FATAL: nsynth binary missing at {NSYNTH_BIN}. "
              "Build with `cargo build --release` in nsynth/.")
        return 1

    T.log("=" * 72)
    T.log("  BRIDGE DEMO — a program no human wrote, running on the GPU computer")
    T.log("=" * 72)

    all_inputs = (
        [i for i, _ in TRAIN_EXAMPLES]
        + [i for i, _ in NSYNTH_HOLDOUTS]
        + UNSEEN_INPUTS
    )
    expected = [oracle(i) for i in all_inputs]

    # ── Step 1: synthesize ──────────────────────────────────────────────────
    T.log("")
    T.log("[1] SYNTHESIZE — nsynth sees only I/O pairs "
          f"{TRAIN_EXAMPLES} (+ holdouts {NSYNTH_HOLDOUTS})")
    synth = synthesize(allow_cache=args.quick)
    mog = synth["code"]
    T.log(f"    method: {synth['method']}   time: {synth['synthesis_seconds']}s"
          f"{'   (cache allowed)' if args.quick else '   (cache disabled — fresh synthesis)'}")
    T.log("    synthesized Mog program:")
    for line in mog.rstrip().splitlines():
        T.log(f"      {line}")

    # ── Step 2: transpile ───────────────────────────────────────────────────
    T.log("")
    T.log("[2] TRANSPILE — Mog -> C (deterministic rewrite)")
    synth_c = mog_to_c(mog)
    c_source = build_driver_c(synth_c, PROBLEM_NAME, all_inputs)
    T.log("    synthesized function as C:")
    for line in synth_c.rstrip().splitlines():
        T.log(f"      {line}")

    # ── Step 3: compile ON the GPU ──────────────────────────────────────────
    T.log("")
    T.log("[3] GPU COMPILE — self-hosting cc.c runs on the rust_metal Metal kernel")
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        cc_elf = tmpdir / "cc.elf"
        cross_compile_cc_elf(cc_elf)
        T.log(f"    host gcc -> cc.elf ({cc_elf.stat().st_size:,} bytes) [Layer 1]")
        raw_bin, cstats = gpu_compile(ncpu_metal, cc_elf, c_source)
        T.log(f"    GPU compiled /tmp/prog.c -> /bin/prog ({len(raw_bin):,} bytes) "
              f"in {cstats['compile_cycles']:,} GPU cycles "
              f"({cstats['compile_seconds']}s) [Layer 2]")

        # ── Step 4: execute ON the GPU ──────────────────────────────────────
        T.log("")
        T.log("[4] GPU EXECUTE — GPU-compiled binary runs on the rust_metal kernel")
        prog_elf_bytes = wrap_raw_in_elf(raw_bin)
        prog_elf = tmpdir / "prog.elf"
        prog_elf.write_bytes(prog_elf_bytes)
        gpu_outputs, xstats = gpu_execute(ncpu_metal, prog_elf)
        T.log(f"    stop={xstats['stop_reason']} exit={xstats['exit_code']} "
              f"cycles={xstats['exec_cycles']:,} ({xstats['exec_seconds']}s) [Layer 3]")
        T.log(f"    GPU outputs: {gpu_outputs}")

        # ── Step 5: verify ──────────────────────────────────────────────────
        T.log("")
        T.log("[5] VERIFY")
        clang_outputs = local_clang_run(c_source, tmpdir)

    py_src = transpile_mog_to_python(mog)
    py_outputs = local_python_transpile_run(py_src, PROBLEM_NAME, all_inputs)

    checks = {
        "gpu_vs_oracle": gpu_outputs == expected,
        "gpu_vs_local_clang": gpu_outputs == clang_outputs,
        "gpu_vs_python_transpile": gpu_outputs == py_outputs,
    }
    n_train = len(TRAIN_EXAMPLES)
    n_hold = len(NSYNTH_HOLDOUTS)
    rows = []
    for idx, inp in enumerate(all_inputs):
        if idx < n_train:
            kind = "train"
        elif idx < n_train + n_hold:
            kind = "holdout"
        else:
            kind = "UNSEEN"
        ok = gpu_outputs[idx] == expected[idx] if idx < len(gpu_outputs) else False
        rows.append((inp, kind, gpu_outputs[idx] if idx < len(gpu_outputs) else None,
                     expected[idx], ok))
        T.log(f"    n={inp:<3} [{kind:^7}]  gpu={rows[-1][2]:<6} "
              f"expected={expected[idx]:<6} {'OK' if ok else 'MISMATCH'}")
    for name, ok in checks.items():
        T.log(f"    {name}: {'OK' if ok else 'MISMATCH'}")

    match = all(checks.values()) and all(r[4] for r in rows)
    T.log("")
    T.log("=" * 72)
    if match:
        T.log("  RESULT: MATCH — the synthesized program ran on the GPU computer")
        T.log("  and produced correct outputs on every input, including inputs")
        T.log("  the synthesizer never saw.")
    else:
        T.log("  RESULT: MISMATCH — see rows above. Artifact records the failure.")
    T.log("=" * 72)

    # ── Artifacts ────────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    result = {
        "demo": "bridge_rung8_synthesized_on_gpu",
        "problem": {
            "name": PROBLEM_NAME,
            "train_examples": TRAIN_EXAMPLES,
            "nsynth_holdouts": NSYNTH_HOLDOUTS,
            "unseen_inputs": UNSEEN_INPUTS,
        },
        "synthesis": {
            "method": synth["method"],
            "seconds": synth["synthesis_seconds"],
            "cache_allowed": bool(args.quick),
        },
        "mog": mog,
        "c_source": c_source,
        "gpu_compile": {
            "cycles": cstats["compile_cycles"],
            "seconds": cstats["compile_seconds"],
            "binary_bytes": len(raw_bin),
            "binary_sha256": hashlib.sha256(raw_bin).hexdigest(),
            "wrapped_elf_bytes": len(prog_elf_bytes),
        },
        "gpu_execute": {
            "cycles": xstats["exec_cycles"],
            "seconds": xstats["exec_seconds"],
            "exit_code": xstats["exit_code"],
            "stop_reason": xstats["stop_reason"],
        },
        "inputs": all_inputs,
        "gpu_outputs": gpu_outputs,
        "expected": expected,
        "local_clang_outputs": clang_outputs,
        "python_transpile_outputs": py_outputs,
        "checks": checks,
        "match": match,
        "runtime_path": (
            "nsynth(synth_gradient) -> Mog -> C -> host gcc(cc.c only) -> "
            "self-hosting cc ON rust_metal GPU (ncpu_metal.run_elf + VFS) -> "
            "minimal-ELF wrap -> execute ON rust_metal GPU (ncpu_metal.run_elf)"
        ),
    }
    result_path = ARTIFACTS_DIR / "bridge_demo_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    transcript_path = ARTIFACTS_DIR / "bridge_demo_transcript.txt"
    T.log(f"\nArtifacts: {result_path.relative_to(PROJECT_ROOT)}, "
          f"{transcript_path.relative_to(PROJECT_ROOT)}")
    T.save(transcript_path)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
