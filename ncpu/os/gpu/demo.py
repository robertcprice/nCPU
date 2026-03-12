#!/usr/bin/env python3
"""
GPU-Native Multi-Process UNIX OS Demo — A full UNIX shell running compiled C
on Apple Silicon Metal with fork/pipe/wait multi-process support.

Pipeline: C source → aarch64-elf-gcc -O2 → raw ARM64 binary → Metal GPU kernel → Python I/O

The shell binary runs entirely on the GPU as ARM64 machine code.
Python handles syscalls (filesystem, I/O, compilation, exec, fork, pipe, wait)
via SVC traps. Multi-process is implemented via memory swapping — each process
gets a 1MB backing store in GPU memory, and the ProcessManager context-switches
between them.

Features:
  - 24+ UNIX commands (ls, cd, pwd, cat, echo, mkdir, rm, rmdir, touch, wc, cp,
    head, cc, run, env, export, grep, sort, uniq, tee, ps, sha256, help, exit)
  - Pipes (ls | grep .c | sort)
  - Background jobs (cmd &)
  - Command chaining (cmd1 ; cmd2, cmd1 && cmd2, cmd1 || cmd2)
  - Output redirection (> and >>)
  - fork/wait/pipe/dup2 multi-process model
  - In-memory filesystem with standard UNIX directory structure
  - In-shell C compilation (cc) and execution (run)

Usage:
    python ncpu/os/gpu/demo.py              # Interactive shell (single-process, backward compat)
    python ncpu/os/gpu/demo.py --multiproc  # Multi-process mode with fork/pipe support
"""

import sys
import os
import time
import tempfile
from pathlib import Path

GPU_OS_DIR = Path(__file__).parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import (
    compile_c, run, make_syscall_handler, read_string_from_gpu,
    ProcessManager, run_multiprocess, HEAP_BASE,
)
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


# ═══════════════════════════════════════════════════════════════════════════════
# FILESYSTEM BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_filesystem() -> GPUFilesystem:
    """Create and populate the initial filesystem."""
    fs = GPUFilesystem()

    # Create standard directories
    for d in ["/home/user", "/var/log", "/usr/lib"]:
        fs.mkdir(d)

    # System files
    fs.write_file("/etc/motd",
        "Welcome to GPU-Native UNIX OS v2.0\n"
        "Running on Apple Silicon Metal - Multi-Process Edition\n"
    )
    fs.write_file("/etc/hostname", "gpu0\n")
    fs.write_file("/etc/os-release",
        "NAME=\"GPU-Native UNIX\"\n"
        "VERSION=\"2.0\"\n"
        "ARCH=\"ARM64 Metal\"\n"
        "FEATURES=\"fork pipe wait dup2 multiprocess\"\n"
    )

    # Example C program for in-shell compilation
    fs.write_file("/home/user/hello.c",
        '#include "arm64_libc.h"\n'
        '\n'
        'int main(void) {\n'
        '    printf("Hello from GPU-compiled C!\\n");\n'
        '    printf("Running on Metal silicon.\\n");\n'
        '    return 0;\n'
        '}\n'
    )

    # Fibonacci example
    fs.write_file("/home/user/fib.c",
        '#include "arm64_libc.h"\n'
        '\n'
        'int main(void) {\n'
        '    printf("Fibonacci sequence:\\n");\n'
        '    long a = 0, b = 1;\n'
        '    for (int i = 0; i < 20; i++) {\n'
        '        printf("  fib(%d) = %ld\\n", i, a);\n'
        '        long tmp = a + b;\n'
        '        a = b;\n'
        '        b = tmp;\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )

    # Fork test program
    fs.write_file("/home/user/fork_test.c",
        '#include "arm64_libc.h"\n'
        '\n'
        'int main(void) {\n'
        '    printf("Parent PID: %d\\n", getpid());\n'
        '    int pid = fork();\n'
        '    if (pid == 0) {\n'
        '        printf("Child process (PID %d, parent %d)\\n", getpid(), getppid());\n'
        '        exit(0);\n'
        '    } else if (pid > 0) {\n'
        '        printf("Forked child PID: %d\\n", pid);\n'
        '        int status;\n'
        '        waitpid(pid, &status, 0);\n'
        '        printf("Child exited, parent done\\n");\n'
        '    } else {\n'
        '        printf("Fork failed!\\n");\n'
        '    }\n'
        '    return 0;\n'
        '}\n'
    )

    # Pipe test program
    fs.write_file("/home/user/pipe_test.c",
        '#include "arm64_libc.h"\n'
        '\n'
        'int main(void) {\n'
        '    int pipefd[2];\n'
        '    if (pipe(pipefd) != 0) {\n'
        '        printf("pipe failed\\n");\n'
        '        return 1;\n'
        '    }\n'
        '    printf("Pipe created: read=%d write=%d\\n", pipefd[0], pipefd[1]);\n'
        '    int pid = fork();\n'
        '    if (pid == 0) {\n'
        '        close(pipefd[0]);\n'
        '        const char *msg = "hello from child via pipe";\n'
        '        write(pipefd[1], msg, strlen(msg));\n'
        '        close(pipefd[1]);\n'
        '        exit(0);\n'
        '    }\n'
        '    close(pipefd[1]);\n'
        '    char buf[128];\n'
        '    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);\n'
        '    if (n > 0) {\n'
        '        buf[n] = 0;\n'
        '        printf("Parent read: %s\\n", buf);\n'
        '    }\n'
        '    close(pipefd[0]);\n'
        '    wait(NULL);\n'
        '    printf("Pipe test passed\\n");\n'
        '    return 0;\n'
        '}\n'
    )

    # A README
    fs.write_file("/home/user/README.txt",
        "GPU-Native UNIX OS v2.0 - Multi-Process Edition\n"
        "================================================\n"
        "This shell is compiled C running on Apple Silicon Metal GPU.\n"
        "All commands execute as ARM64 machine code on the GPU.\n"
        "Python mediates I/O via syscall traps.\n"
        "\n"
        "New in v2.0:\n"
        "  Pipes:       ls | grep .c | sort\n"
        "  Background:  run /bin/fib &\n"
        "  Chaining:    echo hi ; echo bye\n"
        "  Append:      echo line >> /tmp/log.txt\n"
        "  New cmds:    grep, sort, uniq, tee, ps\n"
        "  Process:     fork, wait, pipe, dup2, getpid\n"
        "\n"
        "Try:\n"
        "  cat /etc/motd\n"
        "  echo Hello world > /tmp/test.txt\n"
        "  cat /tmp/test.txt\n"
        "  ls /home/user | grep .c\n"
        "  cc fork_test.c && run /bin/fork_test\n"
    )

    # Set initial cwd
    fs.chdir("/home/user")

    return fs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    multiproc = "--multiproc" in sys.argv or "-m" in sys.argv

    banner = r"""
 ██████  ██████  ██    ██      ██    ██ ███    ██ ██ ██   ██
██       ██   ██ ██    ██      ██    ██ ████   ██ ██  ██ ██
██   ███ ██████  ██    ██      ██    ██ ██ ██  ██ ██   ███
██    ██ ██      ██    ██      ██    ██ ██  ██ ██ ██  ██ ██
 ██████  ██       ██████        ██████  ██   ████ ██ ██   ██
"""
    if multiproc:
        subtitle = " GPU-Native UNIX OS v2.0 — Multi-Process Edition"
        features = " 24-command shell · fork/pipe/wait · compiled C on GPU"
    else:
        subtitle = " GPU-Native UNIX OS v2.0 — Running on Apple Silicon Metal"
        features = " 24-command shell · compiled C on GPU · in-memory filesystem"

    print(banner)
    print(subtitle)
    print(" " + "─" * (len(subtitle) - 1))
    print(features)
    print()

    # 1. Bootstrap filesystem
    print("[boot] Initializing filesystem...")
    fs = bootstrap_filesystem()
    entries = sorted(fs.files.keys())
    print(f"[boot] {len(entries)} files, {len(fs.directories)} directories")

    # 2. Compile the UNIX shell
    c_file = GPU_OS_DIR / "src" / "arm64_unix_shell.c"
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    print(f"[boot] Compiling shell: {c_file.name}")
    if not compile_c(str(c_file), bin_path):
        print("[boot] FATAL: Shell compilation failed")
        sys.exit(1)

    binary = Path(bin_path).read_bytes()
    print(f"[boot] Shell binary: {len(binary):,} bytes")

    # 3. Load onto Metal GPU
    cpu = MLXKernelCPUv2()

    # Track files created during session
    initial_file_count = len(fs.files)

    if multiproc:
        # ═══════════════════════════════════════════════════════════════════
        # MULTI-PROCESS MODE
        # ═══════════════════════════════════════════════════════════════════
        print("[boot] Multi-process mode: fork/pipe/wait enabled")

        proc_mgr = ProcessManager(cpu, fs)
        init_pid = proc_mgr.create_init_process(binary, fd_table={}, cwd="/home/user")
        print(f"[boot] Init process PID {init_pid}")

        # Base handler for non-process syscalls (filesystem, networking, etc.)
        def on_exec(bin_path_str: str) -> bool:
            resolved = fs.resolve_path(bin_path_str)
            binary_data = fs.read_file(resolved)
            if binary_data:
                cpu.load_program(binary_data, address=0x10000)
                cpu.set_pc(0x10000)
                print(f"[exec] Loaded {resolved} ({len(binary_data):,} bytes)")
                return True
            else:
                print(f"[exec] Binary not found: {resolved}")
                return False

        base_handler = make_syscall_handler(
            filesystem=fs,
            on_exec=on_exec,
        )

        print(f"[boot] Booting multi-process shell on Metal GPU...")
        print("=" * 60)

        start = time.perf_counter()
        results = run_multiprocess(
            proc_mgr, base_handler,
            max_total_cycles=500_000_000,
            time_slice=100_000,
            quiet=True,
        )
        elapsed = time.perf_counter() - start

        # Summary
        print()
        print("=" * 60)
        print("GPU-Native UNIX OS — Multi-Process Session Summary")
        print("=" * 60)
        print(f"  Total cycles:       {results['total_cycles']:,}")
        print(f"  Elapsed:            {elapsed:.3f}s")
        print(f"  IPS:                {results['ips']:,.0f}")
        print(f"  Processes created:  {results['processes_created']}")
        print(f"  Total forks:        {results['total_forks']}")
        print(f"  Context switches:   {results['total_context_switches']}")
        files_created = len(fs.files) - initial_file_count
        print(f"  Files created:      {files_created}")

    else:
        # ═══════════════════════════════════════════════════════════════════
        # SINGLE-PROCESS MODE (backward compatible)
        # ═══════════════════════════════════════════════════════════════════
        cpu.load_program(binary, address=0x10000)
        cpu.set_pc(0x10000)

        def on_exec(bin_path_str: str) -> bool:
            resolved = fs.resolve_path(bin_path_str)
            binary_data = fs.read_file(resolved)
            if binary_data:
                cpu.load_program(binary_data, address=0x10000)
                cpu.set_pc(0x10000)
                print(f"[exec] Loaded {resolved} ({len(binary_data):,} bytes)")
                return True
            else:
                print(f"[exec] Binary not found: {resolved}")
                return False

        handler = make_syscall_handler(
            filesystem=fs,
            on_exec=on_exec,
        )

        print(f"[boot] Booting shell on Metal GPU...")
        print("=" * 60)

        start = time.perf_counter()
        results = run(cpu, handler, max_cycles=500_000_000, quiet=True)
        elapsed = time.perf_counter() - start

        # Summary
        print()
        print("=" * 60)
        print("GPU-Native UNIX OS — Session Summary")
        print("=" * 60)
        print(f"  Total cycles:    {results['total_cycles']:,}")
        print(f"  Elapsed:         {elapsed:.3f}s")
        print(f"  IPS:             {results['ips']:,.0f}")
        print(f"  Syscalls:        {cpu.total_syscalls:,}")
        files_created = len(fs.files) - initial_file_count
        print(f"  Files created:   {files_created}")
        print(f"  Stop reason:     {results['stop_reason']}")

    # Cleanup
    if os.path.exists(bin_path):
        os.unlink(bin_path)


if __name__ == "__main__":
    main()
