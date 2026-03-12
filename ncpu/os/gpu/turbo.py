#!/usr/bin/env python3
"""
Turbo Mode — Fully GPU-Native Execution for Cached Programs.

Programs using only GPU-handled syscalls run entirely on the Metal GPU
from load to exit. No Python syscall handling, no GPU<->Python round-trips.

GPU-handled syscalls:
  SYS_WRITE(stdout/stderr) — buffered on GPU (~64KB)
  SYS_BRK — heap management on GPU
  SYS_CLOSE(fd<=2) — no-op on GPU
  SYS_EXIT / SYS_EXIT_GROUP — halt on GPU

Usage:
    from ncpu.os.gpu.turbo import turbo_run, turbo_compile_and_run

    # Run a pre-compiled binary
    result = turbo_run("/tmp/hello.bin")
    print(result.output)
    print(f"Exit code: {result.exit_code}")
    print(f"GPU-native: {result.gpu_native_pct:.0f}% ({result.stats})")

    # Compile and run (uses cache)
    result = turbo_compile_and_run("hello.c")
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from kernels.mlx.gpu_cpu import GPUKernelCPU, StopReasonV2, BACKEND
from ncpu.os.gpu.compile_cache import CompileCache


@dataclass
class TurboResult:
    """Result of a turbo-mode GPU execution."""
    output: str                    # Captured stdout
    stderr_output: str             # Captured stderr
    exit_code: int                 # Program exit code (X0 at SYS_EXIT)
    cycles: int                    # Total GPU cycles
    elapsed_seconds: float         # Wall time
    ips: float                     # Instructions per second
    gpu_dispatches: int            # Number of GPU dispatches (1 = fully GPU-native)
    python_traps: int              # Number of traps to Python (0 = fully GPU-native)
    cache_hit: bool                # Whether compilation was cached
    backend: str                   # "rust" or "mlx"

    @property
    def gpu_native_pct(self) -> float:
        """Percentage of execution that was GPU-native (no Python involvement)."""
        if self.gpu_dispatches == 0:
            return 0.0
        return ((self.gpu_dispatches - self.python_traps) / self.gpu_dispatches) * 100.0

    @property
    def fully_gpu_native(self) -> bool:
        """True if the program ran entirely on GPU with zero Python traps."""
        return self.python_traps == 0

    @property
    def stats(self) -> str:
        """Human-readable stats string."""
        parts = [
            f"{self.cycles:,} cycles",
            f"{self.ips:,.0f} IPS",
            f"{self.gpu_dispatches} dispatch{'es' if self.gpu_dispatches != 1 else ''}",
        ]
        if self.python_traps > 0:
            parts.append(f"{self.python_traps} Python traps")
        if self.cache_hit:
            parts.append("cached")
        return ", ".join(parts)


# Module-level compilation cache (shared across calls)
_compile_cache = CompileCache()


def turbo_run(binary_path: str,
              max_cycles: int = 100_000_000,
              memory_size: int = 16 * 1024 * 1024,
              quiet: bool = True) -> TurboResult:
    """
    Run a pre-compiled ARM64 binary in turbo mode (fully GPU-native when possible).

    The binary is loaded at 0x10000, SP is set by the startup code.
    If the program only uses GPU-handled syscalls, it runs in a single
    GPU dispatch with zero Python involvement.

    If the program traps to Python (e.g., for file I/O), turbo mode
    falls back to a minimal syscall handler that counts traps.

    Args:
        binary_path: Path to the raw ARM64 binary file.
        max_cycles: Maximum GPU cycles before forced stop.
        memory_size: GPU memory size in bytes (default 16MB).
        quiet: Suppress diagnostic output.

    Returns:
        TurboResult with output, exit code, and execution statistics.

    Raises:
        FileNotFoundError: If binary_path does not exist.
    """
    path = Path(binary_path)
    if not path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    binary = path.read_bytes()
    return _turbo_execute(binary, max_cycles, memory_size, quiet, cache_hit=False)


def turbo_compile_and_run(c_file: str,
                          max_cycles: int = 100_000_000,
                          memory_size: int = 16 * 1024 * 1024,
                          extra_flags: Optional[list[str]] = None,
                          quiet: bool = True) -> TurboResult:
    """
    Compile a C file (with caching) and run in turbo mode.

    Uses the content-addressed compilation cache to avoid redundant
    recompilation. Cache keys include the source, compiler flags, startup
    assembly, linker script, and freestanding headers so ABI changes cannot
    return stale binaries.

    Args:
        c_file: Path to the C source file.
        max_cycles: Maximum GPU cycles before forced stop.
        memory_size: GPU memory size in bytes (default 16MB).
        extra_flags: Additional GCC flags for compilation.
        quiet: Suppress diagnostic output.

    Returns:
        TurboResult with output, exit code, and execution statistics.

    Raises:
        FileNotFoundError: If the source file does not exist.
        RuntimeError: If compilation fails.
    """
    import tempfile
    from ncpu.os.gpu.runner import compile_c, SRC_DIR, STARTUP_ASM, LINKER_SCRIPT

    c_path = Path(c_file)
    if not c_path.exists():
        raise FileNotFoundError(f"Source file not found: {c_file}")

    # Check compilation cache against the full toolchain inputs, not just the
    # top-level source file.
    source_bytes = c_path.read_bytes()
    flags_str = " ".join(extra_flags) if extra_flags else ""
    cache_material = bytearray(source_bytes)
    cache_material.extend(STARTUP_ASM.read_bytes())
    cache_material.extend(LINKER_SCRIPT.read_bytes())
    for header in sorted(SRC_DIR.glob("*.h")):
        cache_material.extend(header.name.encode("ascii"))
        cache_material.extend(header.read_bytes())
    cache_key = _compile_cache.cache_key(bytes(cache_material), flags_str)
    cached = _compile_cache.get(cache_key)

    if cached is not None:
        return _turbo_execute(cached, max_cycles, memory_size, quiet, cache_hit=True)

    # Cache miss — compile via GCC cross-compiler
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        if not compile_c(c_file, bin_path, extra_flags=extra_flags, quiet=quiet):
            raise RuntimeError(f"Compilation failed: {c_file}")

        binary = Path(bin_path).read_bytes()
    finally:
        Path(bin_path).unlink(missing_ok=True)

    # Store in cache for future runs
    _compile_cache.put(cache_key, binary)

    return _turbo_execute(binary, max_cycles, memory_size, quiet, cache_hit=False)


def _turbo_execute(binary: bytes,
                   max_cycles: int,
                   memory_size: int,
                   quiet: bool,
                   cache_hit: bool) -> TurboResult:
    """
    Core turbo execution loop.

    Loads the binary into GPU memory and runs it. Programs that only use
    GPU-handled syscalls complete in a single dispatch (gpu_dispatches=1,
    python_traps=0). Programs that trigger unhandled syscalls fall back
    to a minimal Python syscall handler.

    Args:
        binary: Raw ARM64 binary bytes.
        max_cycles: Maximum GPU cycles.
        memory_size: GPU memory size in bytes.
        quiet: Suppress diagnostic output.
        cache_hit: Whether the binary came from the compilation cache.

    Returns:
        TurboResult with full execution statistics.
    """
    cpu = GPUKernelCPU(memory_size=memory_size, quiet=True)
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    # Initialize GPU-side SVC write buffer for on-GPU stdout/stderr buffering
    if hasattr(cpu, 'init_svc_buffer'):
        cpu.init_svc_buffer()

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    total_cycles = 0
    gpu_dispatches = 0
    python_traps = 0
    exit_code = 0

    t0 = time.perf_counter()

    remaining = max_cycles
    while remaining > 0:
        result = cpu.execute(max_cycles=remaining)
        gpu_dispatches += 1
        total_cycles += result.cycles
        remaining -= result.cycles

        # Drain GPU-buffered output (SYS_WRITE handled entirely on GPU)
        if hasattr(cpu, 'drain_svc_buffer'):
            for fd, data in cpu.drain_svc_buffer():
                text = data.decode('utf-8', errors='replace') if isinstance(data, bytes) else str(data)
                if fd == 1:
                    stdout_parts.append(text)
                elif fd == 2:
                    stderr_parts.append(text)

        if result.stop_reason == StopReasonV2.HALT:
            # GPU handled SYS_EXIT — read exit code from X0
            exit_code = cpu.get_register(0)
            if exit_code < 0:
                exit_code = exit_code & 0xFF
            break

        elif result.stop_reason == StopReasonV2.SYSCALL:
            # Program used a syscall not handled on GPU — fall back to Python
            python_traps += 1
            syscall_num = cpu.get_register(8)

            if syscall_num in (93, 231):
                # SYS_EXIT (93) / SYS_EXIT_GROUP (231)
                exit_code = cpu.get_register(0)
                if exit_code < 0:
                    exit_code = exit_code & 0xFF
                break

            elif syscall_num == 64:
                # SYS_WRITE — handle at Python level as fallback
                fd = cpu.get_register(0)
                buf_addr = cpu.get_register(1)
                if buf_addr < 0:
                    buf_addr = buf_addr & 0xFFFFFFFF
                length = cpu.get_register(2)
                data = cpu.read_memory(buf_addr, length)
                text = data.decode('utf-8', errors='replace')
                if fd == 1:
                    stdout_parts.append(text)
                elif fd == 2:
                    stderr_parts.append(text)
                cpu.set_register(0, length)  # Return bytes written

            elif syscall_num == 214:
                # SYS_BRK — simple brk fallback
                requested = cpu.get_register(0)
                cpu.set_register(0, requested or 0x60000)

            else:
                # Unknown syscall — return -ENOSYS
                if not quiet:
                    print(f"[turbo] Unhandled syscall {syscall_num} "
                          f"at PC=0x{result.final_pc:X}")
                cpu.set_register(0, -1)

            # Advance PC past the SVC instruction
            cpu.set_pc(result.final_pc + 4)

        elif result.stop_reason == StopReasonV2.MAX_CYCLES:
            break

    elapsed = time.perf_counter() - t0
    ips = total_cycles / elapsed if elapsed > 0 else 0.0

    return TurboResult(
        output="".join(stdout_parts),
        stderr_output="".join(stderr_parts),
        exit_code=exit_code,
        cycles=total_cycles,
        elapsed_seconds=elapsed,
        ips=ips,
        gpu_dispatches=gpu_dispatches,
        python_traps=python_traps,
        cache_hit=cache_hit,
        backend=BACKEND,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Turbo mode -- fully GPU-native execution"
    )
    parser.add_argument("file", help="C source file or compiled .bin to run")
    parser.add_argument("--max-cycles", type=int, default=100_000_000,
                        help="Maximum GPU cycles (default: 100M)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress diagnostic output")
    args = parser.parse_args()

    path = Path(args.file)
    if path.suffix == ".bin":
        result = turbo_run(str(path), max_cycles=args.max_cycles,
                           quiet=args.quiet)
    elif path.suffix in (".c", ".h"):
        result = turbo_compile_and_run(str(path), max_cycles=args.max_cycles,
                                       quiet=args.quiet)
    else:
        print(f"Unknown file type: {path.suffix}")
        sys.exit(1)

    if result.output:
        print(result.output, end="")
    if result.stderr_output:
        print(result.stderr_output, end="", file=sys.stderr)

    native_str = ("FULLY GPU-NATIVE" if result.fully_gpu_native
                  else f"{result.python_traps} Python traps")
    print(f"\n--- Turbo Mode ---")
    print(f"Exit code:  {result.exit_code}")
    print(f"Cycles:     {result.cycles:,}")
    print(f"IPS:        {result.ips:,.0f}")
    print(f"Time:       {result.elapsed_seconds:.4f}s")
    print(f"Dispatches: {result.gpu_dispatches}")
    print(f"GPU-native: {result.gpu_native_pct:.0f}% ({native_str})")
    print(f"Cache:      {'HIT' if result.cache_hit else 'MISS'}")
    print(f"Backend:    {result.backend}")
