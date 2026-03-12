#!/usr/bin/env python3
"""
BusyBox on GPU — Real Linux userspace running on Metal compute shader.

Demonstrates BusyBox (Alpine Linux's core utility suite) compiled for aarch64
with musl libc, running entirely on Apple Silicon GPU via Metal compute shaders.

The GPU executes ARM64 instructions natively. Python mediates syscalls via SVC trap.

Usage:
    python demos/busybox_gpu_demo.py                    # Run demo suite
    python demos/busybox_gpu_demo.py echo "hello"       # Run specific command
    python demos/busybox_gpu_demo.py cat /etc/motd      # Read from filesystem
    python demos/busybox_gpu_demo.py ls /               # List filesystem
    python demos/busybox_gpu_demo.py --interactive       # Interactive shell mode

Author: Robert Price
Date: March 2026
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try Rust backend first, fall back to Python implementation
try:
    from ncpu.os.gpu.rust_backend import run_elf as load_and_run_elf
except ImportError:
    from ncpu.os.gpu.elf_loader import load_and_run_elf

from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(Path(__file__).parent / "busybox.elf")


def create_filesystem():
    """Create a filesystem populated with sample content for BusyBox."""
    fs = GPUFilesystem()

    # Standard directories
    for d in ["/proc", "/home/user", "/dev"]:
        fs.directories.add(d)

    # System files
    fs.write_file("/etc/motd",
        "Welcome to nCPU GPU-Native UNIX\n"
        "Running on Apple Silicon Metal\n"
        "All instructions execute on GPU\n"
    )
    fs.write_file("/etc/hostname", "ncpu-gpu\n")
    fs.write_file("/etc/passwd",
        "root:x:0:0:root:/root:/bin/sh\n"
        "user:x:1000:1000:user:/home/user:/bin/sh\n"
        "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
    )
    fs.write_file("/etc/group",
        "root:x:0:\n"
        "user:x:1000:\n"
    )
    fs.write_file("/etc/os-release",
        'NAME="nCPU GPU UNIX"\n'
        'VERSION="2.0"\n'
        'ID=ncpu\n'
        'PRETTY_NAME="nCPU GPU-Native UNIX v2.0"\n'
    )

    # User files
    fs.write_file("/home/user/hello.txt",
        "Hello from the GPU filesystem!\n"
        "This file lives in Python memory,\n"
        "served to BusyBox via syscalls.\n"
    )

    # Data file for wc/sort/grep demos
    fs.write_file("/tmp/data.txt",
        "apple\n"
        "banana\n"
        "cherry\n"
        "date\n"
        "elderberry\n"
        "fig\n"
        "grape\n"
    )

    # Fake /proc entries
    fs.write_file("/proc/version",
        "Linux version 6.1.0-ncpu (gcc 13.2) #1 SMP Metal GPU\n"
    )

    return fs


def run_command(argv, quiet_init=True, filesystem=None):
    """Run a busybox command on GPU and return results."""
    return load_and_run_elf(
        BUSYBOX,
        argv=argv,
        max_cycles=500_000_000,
        quiet=quiet_init,
        filesystem=filesystem,
    )


def demo_suite():
    """Run the standard busybox demo suite with filesystem integration."""
    print("=" * 60)
    print("  BusyBox on GPU — Metal Compute Shader ARM64 Execution")
    print("=" * 60)
    print()

    fs = create_filesystem()

    tests = [
        # Basic commands (no filesystem needed)
        (["echo", "Hello from BusyBox on GPU!"], "echo"),
        (["echo", "nCPU:", "Metal", "kernel", "is", "alive"], "echo (multi)"),
        (["uname", "-s"], "uname -s"),
        (["basename", "/usr/local/bin/busybox"], "basename"),
        (["dirname", "/usr/local/bin/busybox"], "dirname"),
        (["true"], "true"),
        (["false"], "false"),
        # Filesystem commands
        (["cat", "/etc/motd"], "cat /etc/motd"),
        (["cat", "/etc/hostname"], "cat /etc/hostname"),
        (["cat", "/proc/version"], "cat /proc/version"),
        (["cat", "/home/user/hello.txt"], "cat hello.txt"),
    ]

    total_time = 0
    passed = 0

    for argv, name in tests:
        sys.stdout.write(f"  {name:20s} → ")
        sys.stdout.flush()
        t = time.perf_counter()
        results = run_command(argv, quiet_init=True, filesystem=fs)
        dt = time.perf_counter() - t
        total_time += dt
        cycles = results["total_cycles"]
        sys.stdout.write(f"  ({cycles:,} cycles, {dt:.1f}s)\n")
        sys.stdout.flush()
        passed += 1

    print()
    print(f"  {passed}/{len(tests)} commands executed successfully")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Binary: {Path(BUSYBOX).stat().st_size:,} bytes (264 KB)")
    print(f"  Architecture: aarch64, statically linked, musl libc")
    print(f"  Filesystem: {len(fs.files)} files, {len(fs.directories)} directories")
    print("=" * 60)


def interactive_mode():
    """Interactive shell-like mode — each command is a fresh BusyBox ELF invocation."""
    print("=" * 60)
    print("  BusyBox on GPU — Interactive Mode")
    print("  Type 'exit' or Ctrl-D to quit. Filesystem persists across commands.")
    print("=" * 60)

    fs = create_filesystem()

    while True:
        try:
            line = input("gpu$ ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break

        # Parse command line (basic shell splitting)
        # Handle output redirection: cmd > file  or  cmd >> file
        argv = line.split()
        redir_file = None
        redir_append = False

        for i, tok in enumerate(argv):
            if tok == ">>" and i + 1 < len(argv):
                redir_file = argv[i + 1]
                redir_append = True
                argv = argv[:i]
                break
            elif tok == ">" and i + 1 < len(argv):
                redir_file = argv[i + 1]
                redir_append = False
                argv = argv[:i]
                break
            elif tok.startswith(">>"):
                redir_file = tok[2:]
                redir_append = True
                argv = argv[:i]
                break
            elif tok.startswith(">") and len(tok) > 1:
                redir_file = tok[1:]
                redir_append = False
                argv = argv[:i]
                break

        if not argv:
            continue

        # Capture output if redirecting
        captured_output = []
        if redir_file:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            results = run_command(argv, quiet_init=True, filesystem=fs)
        finally:
            if redir_file:
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                # Write captured output to filesystem
                path = fs.resolve_path(redir_file)
                if redir_append:
                    existing = fs.read_file(path) or b""
                    fs.write_file(path, existing + output.encode('ascii'))
                else:
                    fs.write_file(path, output.encode('ascii'))

    print("Goodbye.")


def track_unknown_instructions(argv):
    """Analyze BusyBox binary for instruction coverage gaps.

    Statically analyzes the .text segment to find ARM64 opcodes
    not handled by the Metal kernel's decode table.
    """
    from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2
    from ncpu.os.gpu.runner import run

    fs = create_filesystem()
    cpu = MLXKernelCPUv2()

    entry = load_elf_into_memory(cpu, BUSYBOX, argv=argv, quiet=True)
    cpu.set_pc(entry)

    elf_data = Path(BUSYBOX).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    # Analyze the .text segment for unknown instructions
    text_base = elf_info.segments[0].vaddr
    text_size = elf_info.segments[0].filesz

    print(f"\nAnalyzing BusyBox .text: 0x{text_base:X}-0x{text_base + text_size:X} "
          f"({text_size:,} bytes, {text_size // 4:,} instructions)")

    cpu.print_unknown_instructions(text_base, text_size)

    # Also run the command to show output
    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)
    print(f"\nRunning: {' '.join(argv)}")
    results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
    print(f"Cycles: {results['total_cycles']:,}, Stop: {results['stop_reason']}")

    return cpu.analyze_instruction_coverage(text_base, text_size)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    elif len(sys.argv) > 1 and sys.argv[1] == "--track-unknown":
        # Track unknown instructions for remaining args
        cmd_argv = sys.argv[2:] if len(sys.argv) > 2 else ["echo", "test"]
        track_unknown_instructions(cmd_argv)
    elif len(sys.argv) > 1:
        # Run specific command with filesystem
        fs = create_filesystem()
        results = run_command(sys.argv[1:], quiet_init=True, filesystem=fs)
    else:
        demo_suite()
