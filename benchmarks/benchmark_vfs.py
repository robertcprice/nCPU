#!/usr/bin/env python3
"""
Rust VFS vs Python VFS Performance Benchmark.

Compares filesystem operations (touch, rm, mv, mkdir, rmdir, symlink, readlink)
running on Rust Metal backend vs Python backend.

Usage:
    python3 benchmarks/benchmark_vfs.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.filesystem import GPUFilesystem
from ncpu.os.gpu.elf_loader import load_and_run_elf

BUSYBOX = "/Users/bobbyprice/projects/nCPU/demos/busybox.elf"
ITERATIONS = 10


def benchmark_operation(name, op_func, use_rust=True):
    """Benchmark a single operation."""
    times = []
    for _ in range(ITERATIONS):
        fs = GPUFilesystem()
        fs.mkdir('/tmp')

        start = time.perf_counter()
        result = op_func(fs, use_rust)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    return avg_time


def touch_op(fs, use_rust):
    return load_and_run_elf(BUSYBOX, ['touch', '/tmp/test'], filesystem=fs, use_rust=use_rust)


def rm_op(fs, use_rust):
    fs.files['/tmp/test'] = b'content'
    return load_and_run_elf(BUSYBOX, ['rm', '/tmp/test'], filesystem=fs, use_rust=use_rust)


def mv_op(fs, use_rust):
    fs.files['/tmp/old'] = b'content'
    return load_and_run_elf(BUSYBOX, ['mv', '/tmp/old', '/tmp/new'], filesystem=fs, use_rust=use_rust)


def mkdir_op(fs, use_rust):
    return load_and_run_elf(BUSYBOX, ['mkdir', '/tmp/newdir'], filesystem=fs, use_rust=use_rust)


def rmdir_op(fs, use_rust):
    fs.mkdir('/tmp/todel')
    return load_and_run_elf(BUSYBOX, ['rm', '-rf', '/tmp/todel'], filesystem=fs, use_rust=use_rust)


def symlink_op(fs, use_rust):
    fs.files['/tmp/target'] = b'content'
    return load_and_run_elf(BUSYBOX, ['ln', '-s', '/tmp/target', '/tmp/link'], filesystem=fs, use_rust=use_rust)


def readlink_op(fs, use_rust):
    fs.symlink('/tmp/target', '/tmp/link')
    fs.files['/tmp/target'] = b'content'
    return load_and_run_elf(BUSYBOX, ['readlink', '/tmp/link'], filesystem=fs, use_rust=use_rust)


def main():
    operations = [
        ("touch (create file)", touch_op),
        ("rm (delete file)", rm_op),
        ("mv (rename)", mv_op),
        ("mkdir", mkdir_op),
        ("rmdir (rm -rf)", rmdir_op),
        ("symlink (ln -s)", symlink_op),
        ("readlink", readlink_op),
    ]

    print(f"VFS Performance Benchmark ({ITERATIONS} iterations each)")
    print("=" * 70)
    print(f"{'Operation':<25} {'Rust (ms)':<15} {'Python (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    results = []
    for name, op_func in operations:
        # Rust
        rust_time = benchmark_operation(name, op_func, use_rust=True) * 1000

        # Python
        python_time = benchmark_operation(name, op_func, use_rust=False) * 1000

        speedup = python_time / rust_time if rust_time > 0 else 0
        print(f"{name:<25} {rust_time:<15.2f} {python_time:<15.2f} {speedup:<10.2f}x")

        results.append({
            'operation': name,
            'rust_ms': rust_time,
            'python_ms': python_time,
            'speedup': speedup
        })

    print("-" * 70)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}x")

    return results


if __name__ == '__main__':
    main()
