#!/usr/bin/env python3
"""
Benchmark GPU execution modes with superblock/speculation toggles.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_kernel import NeuralARM64Kernel


def _run_mode(label: str, env_overrides: dict, busybox_binary: bytes, argv: list[str], max_instructions: int) -> None:
    # Apply env overrides
    prev = {}
    for k, v in env_overrides.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = v

    kernel = NeuralARM64Kernel()
    start = time.perf_counter()
    exit_code, elapsed = kernel.run_elf_gpu_only(
        busybox_binary,
        argv,
        max_instructions=max_instructions,
        batch_size=32768,
    )
    total = time.perf_counter() - start
    ips = (kernel.total_instructions / elapsed) if elapsed > 0 else 0.0

    print(f"[{label}] exit={exit_code} inst={kernel.total_instructions:,} elapsed={elapsed:.2f}s total={total:.2f}s ips={ips:,.0f}")

    # Restore env
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark GPU toggles.")
    parser.add_argument("--max-instructions", type=int, default=2_000_000)
    parser.add_argument("--echo", nargs="?", const="Hello from Neural GPU!", help="Shortcut: busybox echo <msg>")
    parser.add_argument("--argv", dest="argv_override", nargs=argparse.REMAINDER, help="Explicit busybox argv (use after --argv)")
    parser.add_argument("argv", nargs="*", default=["echo", "Hello from Neural GPU!"])
    args = parser.parse_args()

    if args.echo is not None:
        argv = ["echo", args.echo]
    elif args.argv_override is not None and len(args.argv_override) > 0:
        argv = args.argv_override
    else:
        argv = args.argv

    binaries_dir = Path(__file__).parent / "binaries"
    busybox_path = binaries_dir / "busybox-static"
    if not busybox_path.exists():
        print(f"ERROR: {busybox_path} not found!")
        return 1

    with open(busybox_path, "rb") as f:
        busybox_binary = f.read()

    os.environ.setdefault("NEURAL_TRACE_PC", "0")
    os.environ.setdefault("NEURAL_TRACE_PC_LIMIT", "0")

    modes = [
        ("baseline", {"NEURAL_SUPERBLOCK": "1", "NEURAL_SPECULATE": "1"}),
        ("no_superblock", {"NEURAL_SUPERBLOCK": "0", "NEURAL_SPECULATE": "1"}),
        ("no_spec", {"NEURAL_SUPERBLOCK": "1", "NEURAL_SPECULATE": "0"}),
        ("no_both", {"NEURAL_SUPERBLOCK": "0", "NEURAL_SPECULATE": "0"}),
    ]

    for label, env in modes:
        _run_mode(label, env, busybox_binary, argv, args.max_instructions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
