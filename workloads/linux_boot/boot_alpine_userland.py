#!/usr/bin/env python3
"""
Boot Alpine userland from initrd into the GPU ELF path.
This runs /sbin/init (or /init) using run_elf_gpu_only.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_kernel import NeuralARM64Kernel


def _pick_entry(kernel: NeuralARM64Kernel) -> list[str]:
    candidates = ["/sbin/init", "/init", "/bin/sh", "/bin/busybox"]
    for path in candidates:
        if path in kernel.files:
            if path == "/bin/busybox":
                return [path, "sh"]
            return [path]
    return ["/bin/busybox", "sh"]


def main() -> int:
    initrd_path = os.getenv("NEURAL_INITRD", "initrd.gz")
    if not Path(initrd_path).exists():
        print(f"ERROR: initrd not found at {initrd_path}")
        return 1

    kernel = NeuralARM64Kernel()
    kernel.load_initrd(initrd_path)

    argv = _pick_entry(kernel)
    if argv[0] not in kernel.files:
        print("ERROR: no init binary found in initrd")
        return 1

    elf_bytes = kernel._file_to_bytes(argv[0])
    print(f"Launching userland: {argv}")

    os.environ.setdefault("NEURAL_TRACE_PC", "0")
    os.environ.setdefault("NEURAL_TRACE_PC_LIMIT", "0")

    start = time.perf_counter()
    exit_code, elapsed = kernel.run_elf_gpu_only(
        elf_bytes,
        argv,
        max_instructions=10_000_000,
        batch_size=32768,
    )
    total = time.perf_counter() - start
    print(f"Exit code: {exit_code}  elapsed={elapsed:.2f}s total={total:.2f}s")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
