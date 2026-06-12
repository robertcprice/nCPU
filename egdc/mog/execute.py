"""Compile and execute Mog programs via the mogc compiler.

Provides helpers to compile Mog source code, run the resulting binary,
and compare stdout against expected output — modeled after the
_safe_execute() pattern in egdc/humaneval.py.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths to the Mog toolchain
# ---------------------------------------------------------------------------

_DEFAULT_MOG_ROOT = Path.home() / "projects" / "mog"
_SIBLING_MOG_ROOT = Path(__file__).resolve().parents[3] / "mog"


@lru_cache(maxsize=1)
def _mog_root_candidates() -> tuple[Path, ...]:
    roots: list[Path] = []
    env_root = os.environ.get("MOG_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.extend((_DEFAULT_MOG_ROOT, _SIBLING_MOG_ROOT))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return tuple(deduped)


def _resolve_toolchain_path(env_var: str, relative_parts: tuple[str, ...]) -> Path:
    env_path = os.environ.get(env_var)
    if env_path:
        return Path(env_path).expanduser()

    for root in _mog_root_candidates():
        candidate = root.joinpath(*relative_parts)
        if candidate.exists():
            return candidate

    return _DEFAULT_MOG_ROOT.joinpath(*relative_parts)


MOGC_BINARY = _resolve_toolchain_path("MOGC_BINARY", ("compiler", "target", "release", "mogc"))
MOG_RUNTIME = _resolve_toolchain_path("MOG_RUNTIME", ("runtime-rs", "target", "release", "libmog_runtime.a"))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompileResult:
    success: bool
    binary_path: Optional[str] = None
    stderr: str = ""
    returncode: int = -1


@dataclass
class ExecuteResult:
    compiled: bool = False
    compile_stderr: str = ""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    timed_out: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compile_mog(
    code: str,
    timeout: float = 10.0,
    *,
    mogc: Path | str = MOGC_BINARY,
    runtime: Path | str = MOG_RUNTIME,
    workdir: Optional[str] = None,
) -> CompileResult:
    """Write *code* to a temp .mog file, compile with mogc, return result.

    The caller is responsible for cleaning up CompileResult.binary_path when
    done (or use execute_mog which handles cleanup automatically).
    """
    mogc_path = Path(mogc).expanduser()
    runtime_path = Path(runtime).expanduser()

    if not mogc_path.is_file():
        return CompileResult(
            success=False,
            stderr=f"mogc not found at {mogc_path}",
            returncode=-1,
        )
    if not runtime_path.is_file():
        return CompileResult(
            success=False,
            stderr=f"mog runtime not found at {runtime_path}",
            returncode=-1,
        )

    mogc = str(mogc_path)
    runtime = str(runtime_path)

    td = workdir or tempfile.mkdtemp(prefix="mog_")
    src_path = os.path.join(td, "input.mog")
    bin_path = os.path.join(td, "output")

    with open(src_path, "w") as f:
        f.write(code)

    cmd = [mogc, src_path, "-o", bin_path, "--link", runtime]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            stderr="COMPILE_TIMEOUT",
            returncode=-1,
        )
    except FileNotFoundError:
        return CompileResult(
            success=False,
            stderr=f"mogc not found at {mogc}",
            returncode=-1,
        )

    if proc.returncode == 0 and os.path.isfile(bin_path):
        return CompileResult(
            success=True,
            binary_path=bin_path,
            stderr=proc.stderr,
            returncode=0,
        )
    else:
        return CompileResult(
            success=False,
            stderr=proc.stderr or proc.stdout,
            returncode=proc.returncode,
        )


def execute_mog(
    code: str,
    timeout: float = 5.0,
    compile_timeout: float = 10.0,
    *,
    mogc: Path | str = MOGC_BINARY,
    runtime: Path | str = MOG_RUNTIME,
) -> ExecuteResult:
    """Compile and run a Mog program.  Returns an ExecuteResult."""
    td = tempfile.mkdtemp(prefix="mog_exec_")

    cr = compile_mog(
        code,
        timeout=compile_timeout,
        mogc=mogc,
        runtime=runtime,
        workdir=td,
    )

    if not cr.success:
        return ExecuteResult(
            compiled=False,
            compile_stderr=cr.stderr,
            error=f"Compilation failed: {cr.stderr}",
        )

    assert cr.binary_path is not None
    bin_path = cr.binary_path

    # Make sure the binary is executable
    os.chmod(bin_path, 0o755)

    try:
        proc = subprocess.run(
            [bin_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", "/tmp"),
            },
        )
        return ExecuteResult(
            compiled=True,
            compile_stderr=cr.stderr,
            success=(proc.returncode == 0),
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return ExecuteResult(
            compiled=True,
            compile_stderr=cr.stderr,
            timed_out=True,
            error="RUNTIME_TIMEOUT",
        )


def check_mog_output(
    code: str,
    expected_output: str,
    timeout: float = 5.0,
    compile_timeout: float = 10.0,
    *,
    mogc: Path | str = MOGC_BINARY,
    runtime: Path | str = MOG_RUNTIME,
) -> bool:
    """Compile and run *code*, return True if stdout matches *expected_output*.

    Comparison strips trailing whitespace / newlines from both sides.
    """
    result = execute_mog(
        code,
        timeout=timeout,
        compile_timeout=compile_timeout,
        mogc=mogc,
        runtime=runtime,
    )
    if not result.success:
        return False
    return result.stdout.rstrip() == expected_output.rstrip()


def evaluate_mog_programs(
    programs: List[Tuple[str, str]],
    timeout: float = 5.0,
    compile_timeout: float = 10.0,
    *,
    mogc: Path | str = MOGC_BINARY,
    runtime: Path | str = MOG_RUNTIME,
) -> dict:
    """Evaluate a batch of (code, expected_output) pairs.

    Returns a dict with:
        total        – number of programs
        compiled     – number that compiled successfully
        passed       – number whose output matched expected
        compile_rate – compiled / total
        pass_rate    – passed / total
        results      – list of per-program ExecuteResult
    """
    total = len(programs)
    compiled = 0
    passed = 0
    results: List[ExecuteResult] = []

    for code, expected in programs:
        er = execute_mog(
            code,
            timeout=timeout,
            compile_timeout=compile_timeout,
            mogc=mogc,
            runtime=runtime,
        )
        results.append(er)
        if er.compiled:
            compiled += 1
        if er.success and er.stdout.rstrip() == expected.rstrip():
            passed += 1

    return {
        "total": total,
        "compiled": compiled,
        "passed": passed,
        "compile_rate": compiled / total if total else 0.0,
        "pass_rate": passed / total if total else 0.0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"mogc  : {MOGC_BINARY}  (exists={MOGC_BINARY.exists()})")
    print(f"runtime: {MOG_RUNTIME}  (exists={MOG_RUNTIME.exists()})")
    print()

    test_code = "fn main() -> int { println_i64(42); return 0; }"

    print("=== Compile test ===")
    cr = compile_mog(test_code)
    print(f"  success={cr.success}  binary={cr.binary_path}  rc={cr.returncode}")
    if cr.stderr:
        print(f"  stderr: {cr.stderr[:200]}")

    print()
    print("=== Execute test ===")
    er = execute_mog(test_code)
    print(f"  compiled={er.compiled}  success={er.success}  rc={er.returncode}")
    print(f"  stdout={er.stdout!r}")
    if er.stderr:
        print(f"  stderr: {er.stderr[:200]}")
    if er.error:
        print(f"  error: {er.error[:200]}")

    print()
    print("=== Output check ===")
    ok = check_mog_output(test_code, "42")
    print(f"  match='42' => {ok}")

    print()
    print("=== Batch evaluate ===")
    batch = [
        (test_code, "42"),
        ("fn main() -> int { println_i64(7); return 0; }", "7"),
        ("this is not valid mog code!!!", ""),
    ]
    stats = evaluate_mog_programs(batch)
    print(f"  total={stats['total']}  compiled={stats['compiled']}  passed={stats['passed']}")
    print(f"  compile_rate={stats['compile_rate']:.2%}  pass_rate={stats['pass_rate']:.2%}")

    print()
    if ok:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
