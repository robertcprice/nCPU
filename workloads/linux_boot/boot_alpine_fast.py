#!/usr/bin/env python3
"""
Fast Alpine Linux boot using Neural CPU with 32K batching (1.35M IPS!)
All execution through neural_cpu.py with pure tensor operations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from pathlib import Path

from neural_kernel import NeuralARM64Kernel


def _is_elf(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == b"\x7fELF"


def _pick_userland(kernel: NeuralARM64Kernel) -> list[str]:
    candidates = ["/sbin/init", "/init", "/bin/sh", "/bin/busybox"]
    for path in candidates:
        if path not in kernel.files:
            continue
        data = kernel._file_to_bytes(path)
        if _is_elf(data):
            if path == "/bin/busybox":
                return [path, "sh"]
            return [path]
    return ["/bin/busybox", "sh"]


def _fb_viewer(kernel: NeuralARM64Kernel, stop_event, interval: float):
    last = ""
    while not stop_event.is_set():
        fb = kernel.cpu.get_framebuffer_str()
        if fb and fb != last:
            print("\n=== GPU FRAMEBUFFER ===")
            print(fb)
            print("========================")
            last = fb
        time.sleep(interval)


def boot_fast():
    """Boot Alpine with fast parallel GPU execution"""
    print("=" * 70)
    print("  FAST ALPINE LINUX - Pure Tensor 32K Batch (1.35M IPS!)")
    print("=" * 70)
    print()
    print("  Loading neural kernel...")
    print()

    kernel = NeuralARM64Kernel()

    # Find busybox
    binaries_dir = Path(__file__).parent / "binaries"
    busybox_path = binaries_dir / "busybox-static"

    if not busybox_path.exists():
        print(f"ERROR: {busybox_path} not found!")
        return 1

    with open(busybox_path, 'rb') as f:
        busybox_binary = f.read()

    print(f"  Busybox binary: {len(busybox_binary):,} bytes")
    print()
    print("=" * 70)
    print("  Starting busybox echo (fast test)")
    print("=" * 70)
    print()

    try:
        initrd_path = os.getenv("NEURAL_INITRD", "initrd.gz")
        use_userland = os.getenv("NEURAL_USERLAND") == "1" and Path(initrd_path).exists()
        argv = ["echo", "Hello from Neural GPU!"]
        elf_bytes = busybox_binary
        if use_userland:
            print(f"  Loading initrd: {initrd_path}")
            kernel.load_initrd(initrd_path)
            argv = _pick_userland(kernel)
            elf_path = argv[0]
            elf_bytes = kernel._file_to_bytes(elf_path)
            print(f"  Userland entry: {argv}")
            os.environ.setdefault("NEURAL_FB_CONSOLE", "1")

        fb_view = os.getenv("NEURAL_FB_VIEW") == "1"
        fb_stop = None
        fb_thread = None
        if fb_view:
            interval = float(os.getenv("NEURAL_FB_REFRESH", "0.2"))
            import threading
            fb_stop = threading.Event()
            fb_thread = threading.Thread(
                target=_fb_viewer,
                args=(kernel, fb_stop, interval),
                daemon=True,
            )
            fb_thread.start()

        start = time.perf_counter()
        exit_code, elapsed = kernel.run_elf_gpu_only(
            elf_bytes,
            argv,
            max_instructions=5_000_000,
            batch_size=32768,
        )
        total = time.perf_counter() - start
        if fb_stop:
            fb_stop.set()
        if fb_thread:
            fb_thread.join(timeout=0.5)

        print()
        print("=" * 70)
        print(f"  Exit code: {exit_code}")
        print(f"  Time: {elapsed:.2f}s (total: {total:.2f}s)")
        print(f"  Instructions: {kernel.total_instructions:,}")
        if elapsed > 0:
            print(f"  IPS: {kernel.total_instructions/elapsed:,.0f}")
        print("=" * 70)

        if os.getenv("NEURAL_GPU_TRACE_DUMP") == "1":
            limit = int(os.getenv("NEURAL_GPU_TRACE_LIMIT", "50"))
            trace_ptr = int(kernel.cpu._trace_ptr[0].item())
            start_idx = max(0, trace_ptr - limit)
            trace_cpu = kernel.cpu._trace_buf[start_idx:trace_ptr].cpu()
            print("GPU trace (last entries):")
            for row in trace_cpu.tolist():
                print(f"  pc=0x{row[0]:X} inst=0x{row[1]:08X} op={row[2]}")
        return exit_code

    except KeyboardInterrupt:
        print("\n  Interrupted!")
        return 130
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(boot_fast())
