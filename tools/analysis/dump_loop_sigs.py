#!/usr/bin/env python3
"""
Run a short busybox session and dump loop signature counts (GPU-side cache).
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_kernel import NeuralARM64Kernel


def _run_kernel(kernel: NeuralARM64Kernel, busybox_binary: bytes, argv: list[str]) -> None:
    try:
        kernel.run_elf_gpu_only(busybox_binary, argv, max_instructions=2_000_000, batch_size=32768)
    except Exception:
        # Keep loop sig dump robust even if execution errors out.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump GPU loop signature bins.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Run time before halting (default: 2.0s).")
    parser.add_argument("--trace", type=int, default=0, help="Dump last N GPU trace entries (default: 0).")
    parser.add_argument("--log-all", action="store_true", help="Log all backward branches (including vectorized).")
    parser.add_argument("--echo", nargs="?", const="Hello from Neural GPU!", help="Shortcut: busybox echo <msg>")
    parser.add_argument("--argv", dest="argv_override", nargs=argparse.REMAINDER, help="Explicit busybox argv (use after --argv)")
    parser.add_argument("argv", nargs="*", default=["echo", "Hello from Neural GPU!"], help="busybox argv")
    args = parser.parse_args()

    if args.echo is not None:
        argv = ["echo", args.echo]
    elif args.argv_override is not None and len(args.argv_override) > 0:
        argv = args.argv_override
    else:
        argv = args.argv

    kernel = NeuralARM64Kernel()
    if args.trace > 0:
        kernel.cpu._trace_enabled.fill_(1)
    if args.log_all:
        kernel.cpu._loop_log_all.fill_(1)

    binaries_dir = Path(__file__).parent / "binaries"
    busybox_path = binaries_dir / "busybox-static"
    if not busybox_path.exists():
        print(f"ERROR: {busybox_path} not found!")
        return 1

    with open(busybox_path, "rb") as f:
        busybox_binary = f.read()

    # Reduce trace spam unless explicitly enabled.
    os.environ.setdefault("NEURAL_TRACE_PC", "0")
    os.environ.setdefault("NEURAL_TRACE_PC_LIMIT", "0")

    print(f"Running busybox argv={argv} for ~{args.seconds:.2f}s...")
    t = threading.Thread(target=_run_kernel, args=(kernel, busybox_binary, argv), daemon=True)
    t.start()

    # Let it run briefly, then signal halt to force exit.
    time.sleep(max(0.01, args.seconds))
    kernel.cpu.halted = True
    t.join(timeout=1.0)

    # Dump non-zero loop signature bins (one CPU transfer)
    counts = kernel.cpu._loop_sig_counts.cpu()
    nonzero = counts.nonzero().flatten()
    sig_ptr = int(kernel.cpu._loop_sig_ptr[0].item())
    print(f"Loop sig writes: {sig_ptr}")
    print("Loop signature bins (first 20):")
    print(nonzero[:20].tolist())
    print("Counts (first 20):")
    print(counts[nonzero[:20]].tolist())

    if args.trace > 0:
        trace_ptr = int(kernel.cpu._trace_ptr[0].item())
        trace_start = max(0, trace_ptr - args.trace)
        trace_cpu = kernel.cpu._trace_buf[trace_start:trace_ptr].cpu()
        print(f"GPU trace entries (last {args.trace}):")
        for row in trace_cpu.tolist():
            print(f"  pc=0x{row[0]:X} inst=0x{row[1]:08X} op={row[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
