#!/usr/bin/env python3
"""
Alpine Linux on GPU -- Full distro running on Metal compute shader.

Real Alpine Linux userspace (BusyBox + musl libc, aarch64) executing
entirely on Apple Silicon GPU via Metal compute shaders.

Each command spawns a fresh BusyBox ELF invocation on the GPU with a
shared Python-side filesystem that persists across commands -- exactly
like a real Linux system where /bin/busybox is the multi-call binary
behind every core utility.

Features:
    - Pipes: cat /etc/passwd | grep root | cut -d: -f1
    - Chaining: mkdir /tmp/d && echo ok ; ls /tmp
    - Redirection: echo hello > /tmp/file ; cat /tmp/file
    - 28+ working BusyBox applets on Metal GPU

Usage:
    python demos/alpine_gpu.py                # Interactive Alpine shell
    python demos/alpine_gpu.py --demo         # Automated demo suite

Author: Robert Price
Date: March 2026
"""

import argparse
import io
import shlex
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.elf_loader import load_and_run_elf
from ncpu.os.gpu.alpine import create_alpine_rootfs

BUSYBOX = str(Path(__file__).parent / "busybox.elf")

# ANSI color codes
GREEN = "\033[32m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RED = "\033[31m"


def run_command(argv, filesystem, quiet=True, max_cycles=200_000,
                stdin_data=None):
    """Run a BusyBox command on GPU with shared filesystem."""
    return load_and_run_elf(
        BUSYBOX,
        argv=argv,
        max_cycles=max_cycles,
        quiet=quiet,
        filesystem=filesystem,
        stdin_data=stdin_data,
    )


def run_and_capture(argv, filesystem, stdin_data=None):
    """Run command and capture stdout output, returning (output_str, results)."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        results = run_command(argv, filesystem, quiet=True,
                              stdin_data=stdin_data)
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return output, results


def run_pipeline(pipeline, filesystem):
    """Run a pipeline of commands, piping stdout of each to stdin of the next.

    Args:
        pipeline: List of argv lists, e.g. [["cat", "/etc/passwd"], ["grep", "root"]]
        filesystem: GPUFilesystem instance

    Returns:
        (final_output, total_cycles) tuple
    """
    stdin_data = None
    total_cycles = 0

    for i, argv in enumerate(pipeline):
        is_last = (i == len(pipeline) - 1)

        if is_last:
            # Last command: output goes to stdout
            results = run_command(argv, filesystem, quiet=True,
                                 stdin_data=stdin_data)
            total_cycles += results["total_cycles"]
            return None, total_cycles
        else:
            # Intermediate command: capture output for next stage
            output, results = run_and_capture(argv, filesystem,
                                              stdin_data=stdin_data)
            total_cycles += results["total_cycles"]
            stdin_data = output.encode("utf-8", errors="replace")

    return None, total_cycles


def run_pipeline_and_capture(pipeline, filesystem):
    """Run a pipeline and capture the final output as a string.

    Returns:
        (output_str, total_cycles) tuple
    """
    stdin_data = None
    total_cycles = 0

    for i, argv in enumerate(pipeline):
        output, results = run_and_capture(argv, filesystem,
                                          stdin_data=stdin_data)
        total_cycles += results["total_cycles"]
        stdin_data = output.encode("utf-8", errors="replace")

    return output, total_cycles


def split_pipeline(tokens):
    """Split a token list on '|' into a list of argv lists."""
    pipeline = []
    current = []
    for tok in tokens:
        if tok == "|":
            if current:
                pipeline.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        pipeline.append(current)
    return pipeline


def split_chains(line):
    """Split a command line on chain operators (;, &&, ||).

    Returns list of (argv_tokens, operator) tuples where operator is
    the connector AFTER this command (None for last command).
    """
    chains = []
    # Tokenize with shlex to handle quoting
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()

    current = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == ";":
            if current:
                chains.append((current, ";"))
            current = []
        elif tok == "&&":
            if current:
                chains.append((current, "&&"))
            current = []
        elif tok == "||":
            if current:
                chains.append((current, "||"))
            current = []
        else:
            current.append(tok)
        i += 1

    if current:
        chains.append((current, None))

    return chains


def extract_redirection(tokens):
    """Extract output redirection from token list.

    Returns (filtered_tokens, redir_file, redir_append) tuple.
    """
    redir_file = None
    redir_append = False
    filtered = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == ">>" and i + 1 < len(tokens):
            redir_file = tokens[i + 1]
            redir_append = True
            i += 2
            continue
        elif tok == ">" and i + 1 < len(tokens):
            redir_file = tokens[i + 1]
            redir_append = False
            i += 2
            continue
        elif tok.startswith(">>") and len(tok) > 2:
            redir_file = tok[2:]
            redir_append = True
            i += 1
            continue
        elif tok.startswith(">") and len(tok) > 1 and not tok.startswith(">>"):
            redir_file = tok[1:]
            redir_append = False
            i += 1
            continue
        filtered.append(tok)
        i += 1

    return filtered, redir_file, redir_append


def demo_suite():
    """Run a comprehensive showcase of Alpine Linux on GPU."""
    print("=" * 64)
    print(f"  {BOLD}Alpine Linux v3.20 on GPU -- Metal Compute Shader{RESET}")
    print(f"  Real BusyBox (musl libc, aarch64) on Apple Silicon GPU")
    print("=" * 64)
    print()

    fs = create_alpine_rootfs()

    # Create test data files for the demo
    fs.write_file("/tmp/fruits.txt",
                  "banana\napple\ncherry\nbanana\napple\ndate\n")
    fs.write_file("/tmp/nums.txt", "3\n1\n4\n1\n5\n9\n2\n6\n")
    fs.write_file("/tmp/message.txt", "Hello from Alpine Linux on GPU!\n")

    # Demo commands: either a list of strings (simple) or a single string (parsed)
    demo_sections = [
        ("System Identity", [
            ["cat", "/etc/alpine-release"],
            ["uname", "-a"],
            ["hostname"],
            ["id"],
        ]),
        ("Filesystem", [
            ["ls", "/"],
            ["cat", "/etc/os-release"],
            ["cat", "/proc/version"],
        ]),
        ("File Operations", [
            ["cat", "/etc/passwd"],
            ["head", "-n", "1", "/etc/passwd"],
            ["wc", "/etc/passwd"],
            ["cut", "-d:", "-f1", "/etc/passwd"],
            ["grep", "-F", "root", "/etc/passwd"],
        ]),
        ("Text Processing", [
            ["sort", "/tmp/fruits.txt"],
            ["sort", "-n", "/tmp/nums.txt"],
            ["uniq", "/tmp/fruits.txt"],
        ]),
        ("Pipes", [
            "cat /etc/passwd | grep -F root",
            "cat /etc/passwd | cut -d: -f1",
            "cat /etc/passwd | grep -F root | cut -d: -f1",
            "echo hello world | wc",
        ]),
        ("Utilities", [
            ["echo", "Hello from Alpine on GPU!"],
            ["expr", "2", "+", "3"],
            ["date"],
            ["env"],
        ]),
        ("File Management", [
            ["touch", "/tmp/created_on_gpu.txt"],
            ["mkdir", "/tmp/mydir"],
            ["cp", "/tmp/message.txt", "/tmp/mydir/copy.txt"],
            ["ls", "/tmp"],
        ]),
    ]

    total_time = 0
    total_cycles = 0
    passed = 0
    total = 0

    for section_name, commands in demo_sections:
        print(f"  {BOLD}{CYAN}--- {section_name} ---{RESET}")
        print()

        for cmd_spec in commands:
            total += 1

            # Parse command spec
            if isinstance(cmd_spec, str):
                # String with pipes/operators — parse it
                cmd_str = cmd_spec
                try:
                    tokens = shlex.split(cmd_spec)
                except ValueError:
                    tokens = cmd_spec.split()
                pipeline = split_pipeline(tokens)
            else:
                # Pre-split argv list (single command)
                cmd_str = " ".join(cmd_spec)
                pipeline = [cmd_spec]

            prompt_line = f"{GREEN}root@ncpu-gpu:/{RESET}# {cmd_str}"
            print(prompt_line)

            t = time.perf_counter()
            try:
                if len(pipeline) == 1:
                    output, results = run_and_capture(pipeline[0], fs)
                    cycles = results["total_cycles"]
                else:
                    output, cycles = run_pipeline_and_capture(pipeline, fs)

                dt = time.perf_counter() - t
                total_time += dt
                total_cycles += cycles

                if output.strip():
                    print(output, end="" if output.endswith("\n") else "\n")
                else:
                    print(f"{DIM}(no output){RESET}")

                passed += 1
            except Exception as e:
                dt = time.perf_counter() - t
                total_time += dt
                print(f"{RED}ERROR: {e}{RESET}")

        print()

    # Summary
    print("=" * 64)
    binary_size = Path(BUSYBOX).stat().st_size if Path(BUSYBOX).exists() else 0
    print(f"  {BOLD}Results{RESET}")
    print(f"  {passed}/{total} commands executed successfully on GPU")
    print(f"  Total wall time: {total_time:.1f}s")
    print(f"  Total GPU cycles: {total_cycles:,}")
    print(f"  Binary: {binary_size:,} bytes ({binary_size // 1024} KB)")
    print(f"  Architecture: aarch64, musl libc, statically linked")
    print(f"  Filesystem: {len(fs.files)} files, {len(fs.directories)} directories")
    print(f"  Distro: Alpine Linux v3.20")
    print("=" * 64)

    return passed, total


def interactive_mode():
    """Interactive Alpine Linux shell -- each command runs BusyBox on GPU."""
    fs = create_alpine_rootfs()

    # Show the Alpine motd
    motd = fs.read_file("/etc/motd")
    if motd:
        print(motd.decode("utf-8", errors="replace"), end="")

    print(f"Type '{BOLD}exit{RESET}' to quit. Each command executes BusyBox on GPU.")
    print(f"Supports pipes (|), chaining (;, &&, ||), and redirection (>, >>).")
    print()

    cwd = "/"

    while True:
        prompt = f"{GREEN}root@ncpu-gpu:{cwd}{RESET}# "

        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit", "logout"):
            break

        # Split on chain operators (; && ||)
        chains = split_chains(line)

        last_success = True
        for tokens, operator in chains:
            # Check chain operator from PREVIOUS command
            # (handled below after execution)

            if not tokens:
                continue

            # Handle cd locally
            if tokens[0] == "cd":
                target = tokens[1] if len(tokens) > 1 else "/"
                result = fs.chdir(target)
                if result == 0:
                    cwd = fs.getcwd()
                    last_success = True
                else:
                    print(f"ash: cd: can't cd to {target}: No such file or directory")
                    last_success = False

                # Handle chain operator
                if operator == "&&" and not last_success:
                    break
                if operator == "||" and last_success:
                    break
                continue

            # Extract redirection
            tokens, redir_file, redir_append = extract_redirection(tokens)
            if not tokens:
                continue

            # Split on pipes
            pipeline = split_pipeline(tokens)

            try:
                if redir_file:
                    # Capture output for redirection
                    if len(pipeline) == 1:
                        output, _ = run_and_capture(pipeline[0], fs)
                    else:
                        output, _ = run_pipeline_and_capture(pipeline, fs)

                    path = fs.resolve_path(redir_file)
                    if redir_append:
                        existing = fs.read_file(path) or b""
                        fs.write_file(path, existing + output.encode("utf-8", errors="replace"))
                    else:
                        fs.write_file(path, output.encode("utf-8", errors="replace"))
                elif len(pipeline) == 1:
                    # Single command — output directly to stdout
                    run_command(pipeline[0], fs, quiet=True)
                else:
                    # Pipeline — pipe through stages, last stage outputs to stdout
                    run_pipeline(pipeline, fs)

                last_success = True
            except Exception as e:
                print(f"Error: {e}")
                last_success = False

            # Handle chain operator
            if operator == "&&" and not last_success:
                break
            if operator == "||" and last_success:
                break

    print("Goodbye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alpine Linux v3.20 running on Apple Silicon Metal GPU"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated demo suite instead of interactive shell",
    )
    args = parser.parse_args()

    if not Path(BUSYBOX).exists():
        print(f"Error: BusyBox ELF not found at {BUSYBOX}")
        print("Cross-compile with: aarch64-linux-musl-gcc -static -mgeneral-regs-only")
        sys.exit(1)

    if args.demo:
        demo_suite()
    else:
        interactive_mode()
