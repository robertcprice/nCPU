#!/usr/bin/env python3
"""
GPU HTTP Server Launcher — Boot an HTTP server running as compiled C on Metal GPU.

The server binary runs entirely on the GPU as ARM64 machine code.
Python mediates networking syscalls: the GPU program calls socket/bind/listen/accept/send/recv
via SVC traps, and Python opens real TCP sockets on its behalf.

Usage:
    python demos/net/serve.py
    # Then open http://localhost:8080 in your browser
"""

import sys
import os
import tempfile
import time
from pathlib import Path

NET_DIR = Path(__file__).parent
GPU_OS_DIR = NET_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


def main():
    banner = r"""
 ██████  ██████  ██    ██    ██   ██ ████████ ████████ ██████
██       ██   ██ ██    ██    ██   ██    ██       ██    ██   ██
██   ███ ██████  ██    ██    ███████    ██       ██    ██████
██    ██ ██      ██    ██    ██   ██    ██       ██    ██
 ██████  ██       ██████     ██   ██    ██       ██    ██

 GPU HTTP Server — Compiled C on Apple Silicon Metal
 ────────────────────────────────────────────────────
 ARM64 binary serving HTTP on a Metal compute shader
"""
    print(banner)

    # 1. Set up filesystem with web content
    print("[boot] Initializing filesystem...")
    fs = GPUFilesystem()

    fs.write_file("/index.html",
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>GPU HTTP Server</title></head>\n"
        "<body style=\"font-family: monospace; background: #1a1a2e; color: #0f0;\">\n"
        "<h1>Hello from the GPU!</h1>\n"
        "<p>This page is served by an HTTP server written in C,</p>\n"
        "<p>compiled to ARM64 machine code by GCC,</p>\n"
        "<p>running on an Apple Silicon Metal compute shader.</p>\n"
        "<hr>\n"
        "<p>The GPU executes ARM64 instructions natively.</p>\n"
        "<p>Python mediates networking via SVC syscall traps.</p>\n"
        "<p>The filesystem lives in GPU-accessible memory.</p>\n"
        "<hr>\n"
        "<p><em>nCPU &mdash; Neural CPU Project</em></p>\n"
        "</body>\n"
        "</html>\n"
    )

    fs.write_file("/about.html",
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>About</title></head>\n"
        "<body style=\"font-family: monospace; background: #1a1a2e; color: #0f0;\">\n"
        "<h1>About GPU HTTP Server</h1>\n"
        "<p>Architecture: ARM64 on Metal compute shader</p>\n"
        "<p>Pipeline: C source &rarr; GCC &rarr; raw binary &rarr; Metal GPU &rarr; Python I/O</p>\n"
        "<p><a href=\"/\">Back to index</a></p>\n"
        "</body>\n"
        "</html>\n"
    )

    fs.write_file("/api/status.json",
        '{"status":"running","engine":"Metal GPU","arch":"ARM64","mode":"compute"}\n'
    )

    print(f"[boot] {len(fs.files)} files ready")

    # 2. Compile the HTTP server
    c_file = NET_DIR / "httpd.c"
    if not c_file.exists():
        print(f"[boot] FATAL: {c_file} not found")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print(f"[boot] Compiling HTTP server...")
    if not compile_c(str(c_file), bin_path):
        print("[boot] FATAL: Compilation failed")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"[boot] Server binary: {len(binary):,} bytes")

    # 3. Load onto GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=0x10000)
    cpu.set_pc(0x10000)

    # 4. Create syscall handler with filesystem and networking
    socket_table = {}
    handler = make_syscall_handler(
        filesystem=fs,
        socket_table=socket_table,
    )

    # 5. Run
    print(f"[boot] Starting GPU HTTP server...")
    print(f"[boot] Open http://localhost:8080 in your browser")
    print("=" * 60)

    try:
        results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
    except KeyboardInterrupt:
        print("\n[shutdown] Server stopped by user")
    finally:
        # Close any open sockets
        for fd, sock in socket_table.items():
            try:
                sock.close()
            except Exception:
                pass
        if os.path.exists(bin_path):
            os.unlink(bin_path)

    print()
    print("=" * 60)
    print(f"Cycles: {results.get('total_cycles', 0):,}")
    print(f"Elapsed: {results.get('elapsed', 0):.3f}s")


if __name__ == "__main__":
    main()
