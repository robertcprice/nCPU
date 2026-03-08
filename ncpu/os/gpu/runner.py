#!/usr/bin/env python3
"""
ARM64 Runner — Compile C programs with aarch64-elf-gcc and run on Metal GPU.

Pipeline: C source → GCC cross-compiler → raw binary → Metal GPU kernel → Python I/O

The GPU executes ARM64 instructions natively via Metal compute shaders.
Python mediates syscalls (read/write/exit/brk/filesystem/compile/exec) by
trapping SVC instructions.

Usage:
    from ncpu.os.gpu.runner import compile_c, load_and_run

    if compile_c("ncpu/os/gpu/src/arm64_game_of_life.c", "/tmp/game_of_life.bin"):
        load_and_run("/tmp/game_of_life.bin")
"""

import subprocess
import sys
import os
import struct
import time
import tempfile
import socket as socket_mod
import select
import copy
from enum import IntEnum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2


# ═══════════════════════════════════════════════════════════════════════════════
# COMPILATION
# ═══════════════════════════════════════════════════════════════════════════════

GPU_OS_DIR = Path(__file__).parent
SRC_DIR = GPU_OS_DIR / "src"
LINKER_SCRIPT = SRC_DIR / "arm64.ld"
STARTUP_ASM = SRC_DIR / "arm64_start.S"

GCC = "aarch64-elf-gcc"
OBJCOPY = "aarch64-elf-objcopy"

GCC_FLAGS = [
    "-nostdlib",
    "-ffreestanding",
    "-static",
    "-O2",
    "-march=armv8-a",
    "-mgeneral-regs-only",
    f"-T{LINKER_SCRIPT}",
    f"-I{SRC_DIR}",
    "-e", "_start",
]


def compile_c(c_file: str, bin_path: str, extra_flags: list[str] = None, quiet: bool = False) -> bool:
    """
    Compile a C file to a raw ARM64 binary for the Metal GPU kernel.

    Args:
        c_file: Path to the C source file
        bin_path: Output path for the raw binary
        extra_flags: Additional GCC flags
        quiet: Suppress output

    Returns:
        True on success, False on failure
    """
    c_file = Path(c_file)
    bin_path = Path(bin_path)

    if not c_file.exists():
        print(f"Error: source file not found: {c_file}")
        return False

    # Create temp dir for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        elf_path = Path(tmpdir) / "output.elf"

        # Compile: startup.S + source.c → ELF
        cmd = [GCC] + GCC_FLAGS
        if extra_flags:
            cmd += extra_flags
        cmd += [str(STARTUP_ASM), str(c_file), "-o", str(elf_path)]

        if not quiet:
            print(f"Compiling: {c_file.name}")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed:\n{result.stderr}")
            return False

        # Extract raw binary from ELF
        cmd = [OBJCOPY, "-O", "binary", str(elf_path), str(bin_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"objcopy failed:\n{result.stderr}")
            return False

        if not quiet:
            size = bin_path.stat().st_size
            print(f"Binary: {bin_path} ({size:,} bytes)")

    return True


def compile_c_from_string(source: str, bin_path: str, quiet: bool = False) -> bool:
    """Compile C source from a string."""
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False, dir=str(GPU_OS_DIR)) as f:
        f.write(source)
        c_path = f.name
    try:
        return compile_c(c_path, bin_path, quiet=quiet)
    finally:
        os.unlink(c_path)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU MEMORY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def read_string_from_gpu(cpu: MLXKernelCPUv2, addr: int, max_len: int = 4096) -> str:
    """Read a null-terminated string from GPU memory."""
    data = cpu.read_memory(addr, max_len)
    end = data.find(b'\x00')
    if end >= 0:
        data = data[:end]
    return data.decode('ascii', errors='replace')


def write_bytes_to_gpu(cpu: MLXKernelCPUv2, addr: int, data: bytes):
    """Write bytes to GPU memory."""
    cpu.write_memory(addr, data)


# ═══════════════════════════════════════════════════════════════════════════════
# SYSCALL HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

# Syscall numbers (Linux/ARM64)
SYS_GETCWD     = 17
SYS_MKDIRAT    = 34
SYS_UNLINKAT   = 35
SYS_CHDIR      = 49
SYS_OPENAT     = 56
SYS_CLOSE      = 57
SYS_GETDENTS64 = 61
SYS_LSEEK      = 62
SYS_READ       = 63
SYS_WRITE      = 64
SYS_FSTAT      = 80
SYS_EXIT       = 93
SYS_BRK        = 214

# Custom
SYS_COMPILE = 300
SYS_EXEC    = 301
SYS_GETCHAR = 302
SYS_CLOCK   = 303
SYS_SLEEP   = 304
SYS_SOCKET  = 305
SYS_BIND    = 306
SYS_LISTEN  = 307
SYS_ACCEPT  = 308
SYS_CONNECT = 309
SYS_SEND    = 310
SYS_RECV    = 311

# Process management syscalls
SYS_DUP3    = 24
SYS_PIPE2   = 59
SYS_GETPID  = 172
SYS_GETPPID = 173
SYS_FORK    = 220
SYS_WAIT4   = 260
SYS_PS       = 312
SYS_FLUSH_FB = 313
SYS_KILL     = 314
SYS_GETENV   = 315
SYS_SETENV   = 316

# Signal numbers
SIGTERM = 15
SIGKILL = 9

# Heap starts after .data/.bss segment (expanded layout: .data at 0x50000)
HEAP_BASE = 0x60000

# Monotonic clock baseline (for SYS_CLOCK)
_CLOCK_BASELINE = time.monotonic_ns()


def make_syscall_handler(
    on_write: Optional[Callable] = None,
    on_read: Optional[Callable] = None,
    filesystem=None,
    on_exec: Optional[Callable] = None,
    on_getchar: Optional[Callable] = None,
    raw_input_mode: bool = False,
    socket_table: Optional[dict] = None,
    on_framebuffer: Optional[Callable] = None,
) -> Callable:
    """
    Create a standard syscall handler for ARM64 programs on the Metal GPU.

    Args:
        on_write: Optional callback(fd, data) for custom write handling.
                  Return True to suppress default stdout output.
        on_read:  Optional callback(fd, max_len) -> bytes for custom read handling.
                  Return None to use default stdin.
        filesystem: Optional GPUFilesystem instance for file I/O.
        on_exec:  Optional callback(bin_path) for SYS_EXEC handling.
        on_getchar: Optional callback() -> int for custom getchar handling.
        raw_input_mode: If True, use raw terminal input for getchar.
        socket_table: Optional dict for tracking open sockets {fd: socket_obj}.

    Returns:
        Handler function: handler(cpu) -> bool (True=continue, False=stop)
    """
    heap_break = HEAP_BASE
    if socket_table is None:
        socket_table = {}

    def handler(cpu: MLXKernelCPUv2) -> bool:
        nonlocal heap_break

        syscall_num = cpu.get_register(8)
        x0 = cpu.get_register(0)
        x1 = cpu.get_register(1)
        x2 = cpu.get_register(2)
        x3 = cpu.get_register(3)

        if syscall_num == SYS_WRITE:
            fd = x0
            buf_addr = x1
            length = x2

            # Read buffer from GPU memory
            data = cpu.read_memory(buf_addr, length)

            # Custom handler gets first chance
            if on_write and on_write(fd, data):
                cpu.set_register(0, length)
                return True

            # Check if fd has been redirected via dup2 (e.g., stdout -> file)
            if filesystem and fd in filesystem.fd_table:
                entry = filesystem.fd_table[fd]
                fd_type = entry.get("type", "file")
                # Only redirect to filesystem if it's a real file/pipe, not virtual stdout/stderr
                if fd_type not in ("virtual",):
                    written = filesystem.write(fd, data)
                    cpu.set_register(0, written if written >= 0 else (-1))
                    return True

            # Filesystem fd > 2 (non-redirected)
            if filesystem and fd > 2:
                written = filesystem.write(fd, data)
                cpu.set_register(0, written if written >= 0 else (-1))
                return True

            # Default: write to stdout/stderr
            if fd in (1, 2):
                try:
                    sys.stdout.write(data.decode('ascii', errors='replace'))
                    sys.stdout.flush()
                except Exception:
                    pass

            cpu.set_register(0, length)
            return True

        elif syscall_num == SYS_READ:
            fd = x0
            buf_addr = x1
            max_len = x2

            # Check if fd has been redirected via dup2 (e.g., stdin -> pipe)
            if filesystem and fd in filesystem.fd_table:
                entry = filesystem.fd_table[fd]
                fd_type = entry.get("type", "file")
                if fd_type not in ("virtual",):
                    data = filesystem.read(fd, max_len)
                    if data is not None:
                        cpu.write_memory(buf_addr, data)
                        cpu.set_register(0, len(data))
                    else:
                        cpu.set_register(0, -1)
                    return True

            # Custom handler gets first chance
            if on_read:
                data = on_read(fd, max_len)
                if data is not None:
                    cpu.write_memory(buf_addr, data)
                    cpu.set_register(0, len(data))
                    return True

            # Filesystem fd > 2
            if filesystem and fd > 2:
                data = filesystem.read(fd, max_len)
                if data is not None:
                    cpu.write_memory(buf_addr, data)
                    cpu.set_register(0, len(data))
                else:
                    cpu.set_register(0, -1)
                return True

            # Default: read from stdin
            if fd == 0:
                try:
                    line = input()
                    data = (line + "\n").encode('ascii')[:max_len]
                    cpu.write_memory(buf_addr, data)
                    cpu.set_register(0, len(data))
                except (EOFError, OSError):
                    cpu.set_register(0, 0)  # EOF
            else:
                cpu.set_register(0, 0)

            return True

        elif syscall_num == SYS_EXIT:
            return False

        elif syscall_num == SYS_BRK:
            if x0 == 0:
                cpu.set_register(0, heap_break)
            elif x0 >= HEAP_BASE:
                heap_break = x0
                cpu.set_register(0, heap_break)
            else:
                cpu.set_register(0, heap_break)
            return True

        # ═══════════════════════════════════════════════════════════════
        # FILESYSTEM SYSCALLS
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_OPENAT and filesystem:
            # x0=dirfd (ignored, use cwd), x1=path, x2=flags, x3=mode
            path = read_string_from_gpu(cpu, x1)
            flags = x2
            fd = filesystem.open(path, flags)
            cpu.set_register(0, fd if fd >= 0 else (-1))
            return True

        elif syscall_num == SYS_CLOSE and filesystem:
            fd = x0
            result = filesystem.close(fd)
            cpu.set_register(0, result if result >= 0 else (-1))
            return True

        elif syscall_num == SYS_LSEEK and filesystem:
            fd = x0
            offset = x1
            # Handle signed offset
            if offset >= 0x8000000000000000:
                offset -= 0x10000000000000000
            whence = x2
            result = filesystem.lseek(fd, offset, whence)
            cpu.set_register(0, result if result >= 0 else (-1))
            return True

        elif syscall_num == SYS_FSTAT and filesystem:
            fd = x0
            buf_addr = x1
            info = filesystem.fstat(fd)
            if info:
                from ncpu.os.gpu.elf_loader import _pack_stat64
                stat_buf = _pack_stat64(info)
                cpu.write_memory(buf_addr, stat_buf)
                cpu.set_register(0, 0)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_MKDIRAT and filesystem:
            path = read_string_from_gpu(cpu, x1)
            result = filesystem.mkdir(path)
            cpu.set_register(0, result if result >= 0 else (-1))
            return True

        elif syscall_num == SYS_UNLINKAT and filesystem:
            path = read_string_from_gpu(cpu, x1)
            flags = x2
            if flags & 0x200:  # AT_REMOVEDIR
                result = filesystem.rmdir(path)
            else:
                result = filesystem.unlink(path)
            cpu.set_register(0, result if result >= 0 else (-1))
            return True

        elif syscall_num == SYS_GETCWD and filesystem:
            buf_addr = x0
            buf_size = x1
            cwd = filesystem.getcwd()
            cwd_bytes = cwd.encode('ascii') + b'\x00'
            if len(cwd_bytes) <= buf_size:
                cpu.write_memory(buf_addr, cwd_bytes)
                cpu.set_register(0, 0)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_CHDIR and filesystem:
            path = read_string_from_gpu(cpu, x0)
            result = filesystem.chdir(path)
            cpu.set_register(0, result if result >= 0 else (-1))
            return True

        elif syscall_num == SYS_GETDENTS64 and filesystem:
            # x0=fd, x1=buf, x2=bufsize
            # Real linux_dirent64: d_ino(8) d_off(8) d_reclen(2) d_type(1) d_name[]
            fd_num = x0
            buf_addr = x1
            buf_size = x2

            dirpath = None
            if fd_num in filesystem.fd_table:
                entry = filesystem.fd_table[fd_num]
                dirpath = entry.get("path")
                # Check if already consumed (second call returns 0)
                if entry.get("getdents_consumed"):
                    cpu.set_register(0, 0)
                    return True
            if not dirpath:
                dirpath = filesystem.getcwd()

            entries = filesystem.listdir(dirpath)
            if entries is None:
                cpu.set_register(0, -1)
                return True

            # Pack as real linux_dirent64 structs
            DT_DIR = 4
            DT_REG = 8
            buf = b""
            for idx, name in enumerate(entries):
                name_bytes = name.encode('ascii') + b'\x00'
                # Header: d_ino(8) + d_off(8) + d_reclen(2) + d_type(1) = 19 bytes
                raw_len = 19 + len(name_bytes)
                # Pad to 8-byte alignment
                d_reclen = (raw_len + 7) & ~7

                if len(buf) + d_reclen > buf_size:
                    break

                full_path = filesystem.resolve_path(
                    (dirpath + "/" if dirpath != "/" else "/") + name
                )
                is_dir = full_path in filesystem.directories
                d_ino = hash(full_path) & 0xFFFFFFFFFFFFFFFF
                d_off = idx + 1
                d_type = DT_DIR if is_dir else DT_REG

                dirent = struct.pack('<QQHB', d_ino, d_off, d_reclen, d_type)
                dirent += name_bytes
                # Pad to d_reclen
                dirent += b'\x00' * (d_reclen - len(dirent))
                buf += dirent

            cpu.write_memory(buf_addr, buf)
            cpu.set_register(0, len(buf))

            # Mark as consumed so next call returns 0
            if fd_num in filesystem.fd_table:
                filesystem.fd_table[fd_num]["getdents_consumed"] = True
            return True

        # ═══════════════════════════════════════════════════════════════
        # DUP3 / PIPE2 (needed for shell output redirection / pipes)
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_DUP3 and filesystem:
            old_fd = int(x0)
            new_fd = int(x1)
            result = filesystem.dup2(old_fd, new_fd)
            cpu.set_register(0, result)
            return True

        elif syscall_num == SYS_PIPE2 and filesystem:
            buf_addr = x0
            read_fd, write_fd = filesystem.create_pipe()
            if read_fd < 0:
                cpu.set_register(0, -1)
            else:
                pipe_data = struct.pack("<ii", read_fd, write_fd)
                cpu.write_memory(buf_addr, pipe_data)
                cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_GETPID:
            cpu.set_register(0, 1)  # Single-process is always PID 1
            return True

        elif syscall_num == SYS_GETPPID:
            cpu.set_register(0, 0)  # Init has no parent
            return True

        # ═══════════════════════════════════════════════════════════════
        # CUSTOM SYSCALLS
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_COMPILE and filesystem:
            src_path = read_string_from_gpu(cpu, x0)
            bin_path = read_string_from_gpu(cpu, x1)

            src_path = filesystem.resolve_path(src_path)
            bin_path = filesystem.resolve_path(bin_path)

            # Read source from filesystem
            source = filesystem.read_file(src_path)
            if source is None:
                print(f"[compile] Source not found: {src_path}")
                cpu.set_register(0, -1)
                return True

            # Write source to temp file, compile, read binary back
            with tempfile.NamedTemporaryFile(suffix=".c", mode="wb", delete=False, dir=str(GPU_OS_DIR)) as f:
                f.write(source)
                tmp_c = f.name

            tmp_bin = tmp_c.replace(".c", ".bin")
            try:
                ok = compile_c(tmp_c, tmp_bin, quiet=True)
                if ok:
                    binary = Path(tmp_bin).read_bytes()
                    filesystem.write_file(bin_path, binary)
                    print(f"[compile] {src_path} -> {bin_path} ({len(binary):,} bytes)")
                    cpu.set_register(0, 0)
                else:
                    print(f"[compile] Failed: {src_path}")
                    cpu.set_register(0, -1)
            finally:
                for f in [tmp_c, tmp_bin]:
                    if os.path.exists(f):
                        os.unlink(f)
            return True

        elif syscall_num == SYS_EXEC:
            bin_path_str = read_string_from_gpu(cpu, x0)
            if on_exec:
                result = on_exec(bin_path_str)
                if result:
                    cpu.set_register(0, 0)
                    return "exec"
                else:
                    cpu.set_register(0, -1)
                    return True
            else:
                # Default: load binary from filesystem, reset CPU
                if filesystem:
                    bin_path_resolved = filesystem.resolve_path(bin_path_str)
                    binary = filesystem.read_file(bin_path_resolved)
                    if binary:
                        cpu.load_program(binary, address=0x10000)
                        cpu.set_pc(0x10000)
                        print(f"[exec] Loaded {bin_path_resolved} ({len(binary):,} bytes)")
                        cpu.set_register(0, 0)
                        # Don't advance PC — we're resetting to the loaded program
                        return "exec"
                    else:
                        print(f"[exec] Binary not found: {bin_path_resolved}")
                        cpu.set_register(0, -1)
                else:
                    cpu.set_register(0, -1)
            return True

        # ═══════════════════════════════════════════════════════════════
        # INTERACTIVE I/O SYSCALLS
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_GETCHAR:
            if on_getchar:
                ch = on_getchar()
            elif raw_input_mode:
                import tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = ord(sys.stdin.read(1))
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
            else:
                try:
                    ch = ord(sys.stdin.read(1))
                except (EOFError, TypeError):
                    ch = -1
            cpu.set_register(0, ch)
            return True

        elif syscall_num == SYS_CLOCK:
            ms = (time.monotonic_ns() - _CLOCK_BASELINE) // 1_000_000
            cpu.set_register(0, ms)
            return True

        elif syscall_num == SYS_SLEEP:
            ms = x0
            if ms > 0:
                time.sleep(ms / 1000.0)
            cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════════
        # NETWORKING SYSCALLS
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_SOCKET:
            try:
                # x0=domain (AF_INET=2), x1=type (SOCK_STREAM=1), x2=protocol
                sock = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
                sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
                fd = sock.fileno()
                socket_table[fd] = sock
                cpu.set_register(0, fd)
            except OSError:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_BIND:
            fd = x0
            # x1 = addr (0 = INADDR_ANY), x2 = port
            port = x2
            sock = socket_table.get(fd)
            if sock:
                try:
                    sock.bind(("0.0.0.0", port))
                    cpu.set_register(0, 0)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_LISTEN:
            fd = x0
            backlog = x1 if x1 > 0 else 5
            sock = socket_table.get(fd)
            if sock:
                try:
                    sock.listen(backlog)
                    cpu.set_register(0, 0)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_ACCEPT:
            fd = x0
            sock = socket_table.get(fd)
            if sock:
                try:
                    conn, addr = sock.accept()
                    conn_fd = conn.fileno()
                    socket_table[conn_fd] = conn
                    cpu.set_register(0, conn_fd)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_CONNECT:
            fd = x0
            # x1 = addr (IP packed as 32-bit, or string pointer), x2 = port
            port = x2
            sock = socket_table.get(fd)
            if sock:
                try:
                    # Read IP string from GPU memory
                    addr_str = read_string_from_gpu(cpu, x1) if x1 > 255 else "127.0.0.1"
                    sock.connect((addr_str, port))
                    cpu.set_register(0, 0)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_SEND:
            fd = x0
            buf_addr = x1
            length = x2
            sock = socket_table.get(fd)
            if sock:
                try:
                    data = cpu.read_memory(buf_addr, length)
                    sent = sock.send(data)
                    cpu.set_register(0, sent)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        elif syscall_num == SYS_RECV:
            fd = x0
            buf_addr = x1
            max_len = x2
            sock = socket_table.get(fd)
            if sock:
                try:
                    data = sock.recv(max_len)
                    if data:
                        cpu.write_memory(buf_addr, data)
                        cpu.set_register(0, len(data))
                    else:
                        cpu.set_register(0, 0)
                except OSError:
                    cpu.set_register(0, -1)
            else:
                cpu.set_register(0, -1)
            return True

        # ═══════════════════════════════════════════════════════════════
        # FRAMEBUFFER SYSCALL
        # ═══════════════════════════════════════════════════════════════

        elif syscall_num == SYS_FLUSH_FB:
            # x0=width, x1=height, x2=framebuffer address in GPU memory
            width = x0
            height = x1
            fb_addr = x2
            if on_framebuffer and width > 0 and height > 0:
                # Read RGBA pixel data from GPU memory
                fb_size = width * height * 4  # 4 bytes per pixel (RGBA)
                fb_data = cpu.read_memory(fb_addr, fb_size)
                on_framebuffer(width, height, fb_data)
                cpu.set_register(0, 0)
            else:
                cpu.set_register(0, -1)
            return True

        else:
            print(f"[syscall] Unknown syscall {syscall_num} (x0={x0}, x1={x1}, x2={x2})")
            cpu.set_register(0, -1)
            return True

    return handler


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(
    cpu: MLXKernelCPUv2,
    handler: Callable,
    max_cycles: int = 50_000_000,
    batch_size: int = 100_000,
    quiet: bool = False,
) -> dict:
    """
    Execute ARM64 binary on the Metal GPU with Python-mediated syscalls.

    The GPU runs in batches of `batch_size` cycles. When the GPU hits an SVC
    instruction, control returns to Python for syscall handling, then the GPU
    resumes.

    Args:
        cpu: MLXKernelCPUv2 instance with program loaded
        handler: Syscall handler function (cpu -> bool)
        max_cycles: Maximum total cycles before forced stop
        batch_size: Cycles per GPU dispatch
        quiet: Suppress progress output

    Returns:
        Dict with total_cycles, elapsed, ips, stop_reason
    """
    import time
    import sys as _sys
    total_cycles = 0
    start = time.perf_counter()

    # Initialize GPU-side SVC write buffer (if memory large enough)
    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    def _drain_gpu_writes():
        """Drain GPU-buffered SYS_WRITE entries to stdout/stderr."""
        if cpu.memory_size <= cpu.SVC_BUF_BASE:
            return
        for fd, data in cpu.drain_svc_buffer():
            try:
                text = data.decode('ascii', errors='replace')
                _sys.stdout.write(text)
            except Exception:
                pass
        _sys.stdout.flush()

    while total_cycles < max_cycles:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        # Drain GPU-buffered writes after every dispatch
        _drain_gpu_writes()

        if result.stop_reason == StopReasonV2.HALT:
            break

        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            elif ret == "exec":
                # SYS_EXEC: CPU was reset, don't advance PC
                continue
            # Advance PC past the SVC instruction
            cpu.set_pc(cpu.pc + 4)

        elif result.stop_reason == StopReasonV2.MAX_CYCLES:
            # Batch exhausted, continue
            continue

    elapsed = time.perf_counter() - start
    ips = total_cycles / elapsed if elapsed > 0 else 0

    return {
        "total_cycles": total_cycles,
        "elapsed": elapsed,
        "ips": ips,
        "stop_reason": result.stop_reason.name_str if total_cycles > 0 else "NONE",
    }


def load_and_run(
    bin_path: str,
    handler: Optional[Callable] = None,
    max_cycles: int = 50_000_000,
    quiet: bool = False,
    load_addr: int = 0x10000,
) -> dict:
    """
    Load a binary and run it on the Metal GPU.

    Args:
        bin_path: Path to raw ARM64 binary
        handler: Syscall handler (default: standard handler)
        max_cycles: Maximum cycles
        quiet: Suppress output
        load_addr: Address to load binary at (default: 0x10000 per linker script)

    Returns:
        Execution results dict
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        print(f"Error: binary not found: {bin_path}")
        return {"total_cycles": 0, "elapsed": 0, "ips": 0, "stop_reason": "ERROR"}

    binary = bin_path.read_bytes()

    cpu = MLXKernelCPUv2()
    cpu.load_program(binary, address=load_addr)
    cpu.set_pc(load_addr)

    if handler is None:
        handler = make_syscall_handler()

    if not quiet:
        print(f"Loaded {len(binary):,} bytes at 0x{load_addr:X}")
        print(f"PC = 0x{load_addr:X}, SP will be set by _start")
        print("=" * 60)

    results = run(cpu, handler, max_cycles=max_cycles, quiet=quiet)

    if not quiet:
        print("=" * 60)
        print(f"Total cycles: {results['total_cycles']:,}")
        print(f"Elapsed: {results['elapsed']:.3f}s")
        print(f"IPS: {results['ips']:,.0f}")
        print(f"Stop: {results['stop_reason']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-PROCESS SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessState(IntEnum):
    FREE = 0
    READY = 1
    RUNNING = 2
    BLOCKED = 3
    ZOMBIE = 4


@dataclass
class Process:
    pid: int
    ppid: int
    state: ProcessState
    exit_code: int = 0
    registers: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.int64))
    pc: int = 0
    flags: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    heap_break: int = HEAP_BASE
    fd_table: dict = field(default_factory=dict)
    cwd: str = "/"
    wait_target: int = -1  # PID being waited on (-1 = any child)
    # Resource tracking
    fork_count: int = 0          # Number of forks this process has done
    total_cycles: int = 0        # Total cycles consumed
    pending_signal: int = 0      # Signal to deliver (0 = none)
    env: dict = field(default_factory=dict)  # Per-process environment variables


# Memory layout constants for multi-process
ACTIVE_BASE = 0x10000         # Active workspace starts here
ACTIVE_END = 0xFF000          # Active workspace ends here (stack top)
ACTIVE_SIZE = ACTIVE_END - ACTIVE_BASE
BACKING_STORE_BASE = 0x100000  # First backing store at 1MB
BACKING_STORE_SIZE = 0x100000  # 1MB per process
MAX_PROCESSES = 15             # PIDs 1-15
MAX_FORKS_PER_PROCESS = 32     # Fork bomb protection
MAX_FDS_PER_PROCESS = 64       # Per-process fd limit
MAX_CYCLES_PER_PROCESS = 100_000_000  # Per-process cycle limit


class ProcessManager:
    """Manages multiple processes on a single GPU using memory swapping."""

    def __init__(self, cpu: MLXKernelCPUv2, filesystem):
        self.cpu = cpu
        self.fs = filesystem
        self.processes: dict[int, Process] = {}
        self.next_pid: int = 1
        self.current_pid: int = -1
        self.total_forks: int = 0
        self.total_context_switches: int = 0

    def create_init_process(self, binary: bytes, fd_table: dict = None,
                            cwd: str = "/") -> int:
        """Create the initial process (PID 1) from a binary."""
        pid = self._alloc_pid()
        if pid < 0:
            return -1

        proc = Process(
            pid=pid, ppid=0, state=ProcessState.READY,
            pc=0x10000,
            heap_break=HEAP_BASE,
            fd_table=fd_table if fd_table else {},
            cwd=cwd,
        )
        # Set SP in registers
        proc.registers[31] = 0  # XZR — SP is set by _start code

        self.processes[pid] = proc

        # Load binary into active workspace
        self.cpu.load_program(binary, address=0x10000)
        self.cpu.set_pc(0x10000)

        # Save initial memory to backing store
        self._save_to_backing_store(pid)

        return pid

    def _alloc_pid(self) -> int:
        """Allocate the next available PID."""
        if self.next_pid > MAX_PROCESSES:
            # Try to find a free slot
            for i in range(1, MAX_PROCESSES + 1):
                if i not in self.processes:
                    return i
            return -1  # No slots
        pid = self.next_pid
        self.next_pid += 1
        return pid

    # ═══════════════════════════════════════════════════════════════════════
    # MEMORY SWAPPING
    # ═══════════════════════════════════════════════════════════════════════

    def _backing_addr(self, pid: int) -> int:
        """Get the backing store base address for a PID."""
        return BACKING_STORE_BASE + (pid - 1) * BACKING_STORE_SIZE

    def _save_to_backing_store(self, pid: int):
        """Save active workspace to a PID's backing store."""
        active_data = self.cpu.read_memory(ACTIVE_BASE, ACTIVE_SIZE)
        backing_addr = self._backing_addr(pid)
        self.cpu.write_memory(backing_addr, active_data)

    def _restore_from_backing_store(self, pid: int):
        """Restore a PID's backing store to active workspace."""
        backing_addr = self._backing_addr(pid)
        backing_data = self.cpu.read_memory(backing_addr, ACTIVE_SIZE)
        self.cpu.write_memory(ACTIVE_BASE, backing_data)

    def context_switch_out(self, pid: int):
        """Save the current process state from the GPU."""
        proc = self.processes[pid]
        proc.registers = self.cpu.get_registers_numpy()
        proc.pc = self.cpu.pc
        proc.flags = np.array([
            1.0 if f else 0.0 for f in self.cpu.get_flags()
        ], dtype=np.float32)
        # Save the current fd_table state back to the process
        proc.fd_table = dict(self.fs.fd_table)
        self._save_to_backing_store(pid)

    def context_switch_in(self, pid: int):
        """Load a process onto the GPU."""
        proc = self.processes[pid]
        self._restore_from_backing_store(pid)
        self.cpu.set_registers_numpy(proc.registers)
        self.cpu.set_pc(proc.pc)
        n, z, c, v = proc.flags
        self.cpu.set_flags(n > 0.5, z > 0.5, c > 0.5, v > 0.5)
        # Swap in this process's fd_table
        self.fs.fd_table = dict(proc.fd_table)
        self.fs.cwd = proc.cwd
        proc.state = ProcessState.RUNNING
        self.current_pid = pid
        self.total_context_switches += 1

    # ═══════════════════════════════════════════════════════════════════════
    # SCHEDULING
    # ═══════════════════════════════════════════════════════════════════════

    def schedule_next(self) -> Optional[int]:
        """Round-robin scheduler: pick next READY process."""
        ready = [p.pid for p in self.processes.values()
                 if p.state == ProcessState.READY]
        if not ready:
            return None
        # Simple round-robin: pick the first ready process after current
        ready.sort()
        for pid in ready:
            if pid > self.current_pid:
                return pid
        return ready[0]  # Wrap around

    # ═══════════════════════════════════════════════════════════════════════
    # PROCESS LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════

    def process_exit(self, pid: int, exit_code: int):
        """Mark a process as zombie, wake up waiting parent, reparent orphans."""
        proc = self.processes[pid]
        proc.state = ProcessState.ZOMBIE
        proc.exit_code = exit_code

        # Close all fds
        for fd in list(proc.fd_table.keys()):
            entry = proc.fd_table[fd]
            pipe_buf = entry.get("pipe_buffer")
            if pipe_buf:
                if entry.get("type") == "pipe_read":
                    pipe_buf.close_reader()
                elif entry.get("type") == "pipe_write":
                    pipe_buf.close_writer()
        proc.fd_table.clear()

        # Reparent orphaned children to init (PID 1)
        for child in self.processes.values():
            if child.ppid == pid and child.pid != pid:
                child.ppid = 1
                # If child is a zombie, wake init if it's waiting
                if child.state == ProcessState.ZOMBIE and 1 in self.processes:
                    init = self.processes[1]
                    if init.state == ProcessState.BLOCKED:
                        if init.wait_target == child.pid or init.wait_target == -1:
                            init.state = ProcessState.READY

        # Wake parent if it's blocked waiting for this child
        if proc.ppid in self.processes:
            parent = self.processes[proc.ppid]
            if parent.state == ProcessState.BLOCKED:
                if parent.wait_target == pid or parent.wait_target == -1:
                    parent.state = ProcessState.READY

    def kill_process(self, target_pid: int, signal: int, sender_pid: int) -> int:
        """Send a signal to a process. Returns 0 on success, -1 on error."""
        if target_pid not in self.processes:
            return -1  # ESRCH

        target = self.processes[target_pid]
        if target.state == ProcessState.ZOMBIE:
            return -1  # Already dead

        if signal == SIGKILL:
            # SIGKILL: immediate termination, cannot be caught
            self.process_exit(target_pid, 128 + SIGKILL)
            return 0
        elif signal == SIGTERM:
            # SIGTERM: mark for termination (process exits on next schedule)
            target.pending_signal = SIGTERM
            # If blocked, unblock so it can be killed
            if target.state == ProcessState.BLOCKED:
                target.state = ProcessState.READY
            return 0
        elif signal == 0:
            # Signal 0: check if process exists (for kill -0 pid)
            return 0
        else:
            return -1  # EINVAL — unsupported signal

    def reap_zombie(self, parent_pid: int, child_pid: int) -> Optional[Process]:
        """Reap a zombie child, return it, or None if no zombie found."""
        if child_pid == -1:
            # Any child
            for proc in self.processes.values():
                if proc.ppid == parent_pid and proc.state == ProcessState.ZOMBIE:
                    zombie = self.processes.pop(proc.pid)
                    return zombie
        else:
            if child_pid in self.processes:
                proc = self.processes[child_pid]
                if proc.ppid == parent_pid and proc.state == ProcessState.ZOMBIE:
                    return self.processes.pop(child_pid)
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # FORK
    # ═══════════════════════════════════════════════════════════════════════

    def fork(self, parent_pid: int) -> int:
        """Fork the parent process. Returns child PID or -1 on error."""
        parent = self.processes[parent_pid]

        # Fork bomb protection
        if parent.fork_count >= MAX_FORKS_PER_PROCESS:
            return -1  # EAGAIN — fork limit reached

        child_pid = self._alloc_pid()
        if child_pid < 0:
            return -1  # EAGAIN — process table full

        # Advance PC past the SVC instruction BEFORE saving state.
        # Both parent and child should resume at the instruction AFTER fork().
        self.cpu.set_pc(self.cpu.pc + 4)

        # Save parent's current GPU state to its backing store
        self.context_switch_out(parent_pid)

        # Create child process as a copy of parent.
        # Parent's fd_table was saved by context_switch_out above.
        # Clone it for the child with proper pipe refcounting.
        child_fd_table = self.fs.clone_fd_table()

        child = Process(
            pid=child_pid,
            ppid=parent_pid,
            state=ProcessState.READY,
            exit_code=0,
            registers=parent.registers.copy(),
            pc=parent.pc,  # Already advanced past SVC
            flags=parent.flags.copy(),
            heap_break=parent.heap_break,
            fd_table=child_fd_table,
            cwd=parent.cwd,
            wait_target=-1,
            env=dict(parent.env),  # Inherit environment
        )
        parent.fork_count += 1

        # Copy parent's backing store to child's backing store
        parent_backing = self._backing_addr(parent_pid)
        child_backing = self._backing_addr(child_pid)
        parent_mem = self.cpu.read_memory(parent_backing, ACTIVE_SIZE)
        self.cpu.write_memory(child_backing, parent_mem)

        # Set return values: child gets 0, parent gets child_pid
        child.registers[0] = 0
        parent.registers[0] = child_pid

        # Both processes are READY
        parent.state = ProcessState.READY
        child.state = ProcessState.READY

        self.processes[child_pid] = child
        self.total_forks += 1

        return child_pid

    # ═══════════════════════════════════════════════════════════════════════
    # SYSCALL HANDLING (multi-process)
    # ═══════════════════════════════════════════════════════════════════════

    def handle_syscall(self, pid: int, base_handler: Callable) -> str:
        """Handle syscalls for multi-process mode.

        Returns: "continue" (resume), "exit" (process exited), "fork" (forked),
                 "blocked" (process blocked), "exec" (exec'd new binary)
        """
        proc = self.processes[pid]
        syscall_num = self.cpu.get_register(8)
        x0 = self.cpu.get_register(0)
        x1 = self.cpu.get_register(1)

        if syscall_num == SYS_FORK:
            child_pid = self.fork(pid)
            if child_pid < 0:
                # Fork failed — set error, don't context-switch
                self.cpu.set_register(0, -1)
                return "continue"
            # fork() already advanced PC, saved both to backing stores,
            # set return values, and set both to READY.
            # Let the scheduler pick who runs next.
            return "fork"

        elif syscall_num == SYS_WAIT4:
            target_pid = x0
            # Handle signed: -1 means "any child"
            if target_pid >= 0x8000000000000000:
                target_pid = target_pid - 0x10000000000000000
            target_pid = int(target_pid)

            # Check for zombie children first
            zombie = self.reap_zombie(pid, target_pid)
            if zombie:
                self.cpu.set_register(0, zombie.pid)
                # Write exit status to memory if x1 != 0
                if x1 != 0:
                    status = (zombie.exit_code & 0xFF) << 8
                    self.cpu.write_memory(x1, struct.pack("<i", status))
                return "continue"

            # Check if there are any live children to wait for
            has_children = any(
                p.ppid == pid and p.state != ProcessState.ZOMBIE
                for p in self.processes.values()
            )
            if not has_children:
                self.cpu.set_register(0, -1)  # ECHILD
                return "continue"

            # Block the process
            proc.wait_target = target_pid
            proc.state = ProcessState.BLOCKED
            return "blocked"

        elif syscall_num == SYS_PIPE2:
            buf_addr = x0
            # Check fd limit
            if len(proc.fd_table) >= MAX_FDS_PER_PROCESS:
                self.cpu.set_register(0, -1)  # EMFILE
                return "continue"
            read_fd, write_fd = self.fs.create_pipe()
            if read_fd < 0:
                self.cpu.set_register(0, -1)
            else:
                # Write the two fds to GPU memory
                pipe_data = struct.pack("<ii", read_fd, write_fd)
                self.cpu.write_memory(buf_addr, pipe_data)
                # Also add to process fd table
                proc.fd_table[read_fd] = self.fs.fd_table[read_fd]
                proc.fd_table[write_fd] = self.fs.fd_table[write_fd]
                self.cpu.set_register(0, 0)
            return "continue"

        elif syscall_num == SYS_DUP3:
            old_fd = int(x0)
            new_fd = int(x1)
            result = self.fs.dup2(old_fd, new_fd)
            self.cpu.set_register(0, result)
            if result >= 0:
                proc.fd_table[new_fd] = self.fs.fd_table.get(new_fd, {})
            return "continue"

        elif syscall_num == SYS_GETPID:
            self.cpu.set_register(0, pid)
            return "continue"

        elif syscall_num == SYS_GETPPID:
            self.cpu.set_register(0, proc.ppid)
            return "continue"

        elif syscall_num == SYS_KILL:
            target_pid = int(x0)
            signal = int(x1)
            result = self.kill_process(target_pid, signal, pid)
            self.cpu.set_register(0, result)
            return "continue"

        elif syscall_num == SYS_GETENV:
            key_addr = int(x0)
            buf_addr = int(x1)
            buf_size = int(self.cpu.get_register(2))
            key = read_string_from_gpu(self.cpu, key_addr)
            value = proc.env.get(key, "")
            if value:
                val_bytes = value.encode('ascii') + b'\x00'
                if len(val_bytes) <= buf_size:
                    self.cpu.write_memory(buf_addr, val_bytes)
                    self.cpu.set_register(0, len(val_bytes) - 1)
                else:
                    self.cpu.set_register(0, -1)
            else:
                self.cpu.set_register(0, 0)  # Empty = not found
            return "continue"

        elif syscall_num == SYS_SETENV:
            key_addr = int(x0)
            val_addr = int(x1)
            key = read_string_from_gpu(self.cpu, key_addr)
            value = read_string_from_gpu(self.cpu, val_addr)
            proc.env[key] = value
            self.cpu.set_register(0, 0)
            return "continue"

        elif syscall_num == SYS_PS:
            # Write process list to stdout
            buf_addr = x0  # Buffer address to write process info
            info = "PID  PPID  STATE\n"
            for p in sorted(self.processes.values(), key=lambda p: p.pid):
                state_names = {0: "FREE", 1: "READY", 2: "RUN", 3: "BLOCK", 4: "ZOMBIE"}
                sname = state_names.get(p.state, "?")
                info += f"{p.pid:3d}  {p.ppid:4d}  {sname}\n"
            info_bytes = info.encode('ascii')
            if buf_addr != 0:
                self.cpu.write_memory(buf_addr, info_bytes + b'\x00')
                self.cpu.set_register(0, len(info_bytes))
            else:
                # Write to stdout directly
                sys.stdout.write(info)
                sys.stdout.flush()
                self.cpu.set_register(0, len(info_bytes))
            return "continue"

        elif syscall_num == SYS_EXIT:
            exit_code = int(x0)
            self.process_exit(pid, exit_code)
            return "exit"

        elif syscall_num == SYS_BRK:
            # Per-process brk
            if x0 == 0:
                self.cpu.set_register(0, proc.heap_break)
            elif x0 >= HEAP_BASE:
                proc.heap_break = x0
                self.cpu.set_register(0, proc.heap_break)
            else:
                self.cpu.set_register(0, proc.heap_break)
            return "continue"

        elif syscall_num == SYS_READ:
            # Intercept pipe reads to handle would-block
            fd = int(x0)
            if fd in self.fs.fd_table:
                entry = self.fs.fd_table[fd]
                if entry.get("type") == "pipe_read":
                    buf_addr = int(x1)
                    count = int(self.cpu.get_register(2))
                    pipe_buf = entry.get("pipe_buffer")
                    if pipe_buf:
                        data = pipe_buf.read(count)
                        if data is None:
                            # Would block — block process, retry later
                            # Don't advance PC so the read retries
                            proc.state = ProcessState.BLOCKED
                            proc.wait_target = -2  # Sentinel: blocked on pipe
                            return "blocked_pipe"
                        else:
                            # Got data (possibly empty = EOF)
                            if data:
                                self.cpu.write_memory(buf_addr, data)
                            self.cpu.set_register(0, len(data))
                            return "continue"
            # Not a pipe fd — delegate to base handler
            ret = base_handler(self.cpu)
            if ret == False:
                self.process_exit(pid, 0)
                return "exit"
            return "continue"

        elif syscall_num == SYS_WRITE:
            # Intercept pipe writes to unblock readers
            fd = int(x0)
            if fd in self.fs.fd_table:
                entry = self.fs.fd_table[fd]
                if entry.get("type") == "pipe_write":
                    buf_addr = int(x1)
                    length = int(self.cpu.get_register(2))
                    data = self.cpu.read_memory(buf_addr, length)
                    pipe_buf = entry.get("pipe_buffer")
                    if pipe_buf:
                        written = pipe_buf.write(data)
                        self.cpu.set_register(0, written if written >= 0 else -1)
                        # Wake up any blocked pipe readers
                        for p in self.processes.values():
                            if (p.state == ProcessState.BLOCKED and
                                p.wait_target == -2):
                                p.state = ProcessState.READY
                        return "continue"
            # Not a pipe fd — delegate to base handler
            ret = base_handler(self.cpu)
            if ret == False:
                self.process_exit(pid, 0)
                return "exit"
            return "continue"

        elif syscall_num == SYS_CLOSE:
            # Intercept close to handle pipe endpoint cleanup
            fd = int(x0)
            result = self.fs.close(fd)
            self.cpu.set_register(0, result if result >= 0 else -1)
            # After closing a write end, wake blocked pipe readers
            # (they'll get EOF on next read)
            for p in self.processes.values():
                if (p.state == ProcessState.BLOCKED and
                    p.wait_target == -2):
                    p.state = ProcessState.READY
            return "continue"

        else:
            # Delegate to base handler for filesystem, networking, etc.
            ret = base_handler(self.cpu)
            if ret == False:
                self.process_exit(pid, 0)
                return "exit"
            elif ret == "exec":
                return "exec"
            return "continue"


def run_multiprocess(
    proc_mgr: ProcessManager,
    base_handler: Callable,
    max_total_cycles: int = 50_000_000,
    time_slice: int = 50_000,
    quiet: bool = False,
) -> dict:
    """
    Multi-process execution loop with round-robin scheduling.

    Each process gets `time_slice` cycles before the scheduler considers
    switching to another READY process.
    """
    total_cycles = 0
    start = time.perf_counter()

    while total_cycles < max_total_cycles:
        pid = proc_mgr.schedule_next()
        if pid is None:
            # No runnable processes — check for deadlock
            has_blocked = any(
                p.state == ProcessState.BLOCKED
                for p in proc_mgr.processes.values()
            )
            if has_blocked:
                # Try to unblock pipe readers by draining writers
                # If all processes are blocked on pipes, we're deadlocked
                if not quiet:
                    print("[multiproc] Deadlock detected: all processes blocked")
            break

        # Context switch in
        proc_mgr.context_switch_in(pid)

        # Check for pending signals before execution
        proc = proc_mgr.processes[pid]
        if proc.pending_signal != 0:
            sig = proc.pending_signal
            proc.pending_signal = 0
            if sig in (SIGTERM, SIGKILL):
                proc_mgr.process_exit(pid, 128 + sig)
                continue

        # Check per-process cycle limit
        if proc.total_cycles >= MAX_CYCLES_PER_PROCESS:
            if not quiet:
                print(f"[multiproc] PID {pid} exceeded cycle limit, terminating")
            proc_mgr.context_switch_out(pid)
            proc_mgr.process_exit(pid, 137)  # Killed
            continue

        # Execute time slice
        result = proc_mgr.cpu.execute(max_cycles=time_slice)
        total_cycles += result.cycles
        proc.total_cycles += result.cycles

        if result.stop_reason == StopReasonV2.HALT:
            proc_mgr.context_switch_out(pid)
            proc_mgr.process_exit(pid, 0)

        elif result.stop_reason == StopReasonV2.SYSCALL:
            action = proc_mgr.handle_syscall(pid, base_handler)

            if action == "continue":
                # Advance PC past SVC
                proc_mgr.cpu.set_pc(proc_mgr.cpu.pc + 4)
                proc_mgr.context_switch_out(pid)
                proc = proc_mgr.processes.get(pid)
                if proc and proc.state == ProcessState.RUNNING:
                    proc.state = ProcessState.READY

            elif action == "fork":
                # PC already past SVC, parent was saved in fork()
                # Both parent and child are READY
                pass

            elif action == "blocked":
                # Blocked on wait — advance PC past SVC (result set when woken)
                proc_mgr.cpu.set_pc(proc_mgr.cpu.pc + 4)
                proc_mgr.context_switch_out(pid)

            elif action == "blocked_pipe":
                # Blocked on pipe read — do NOT advance PC (will retry read)
                proc_mgr.context_switch_out(pid)

            elif action == "exit":
                # Process already marked zombie, no save needed
                pass

            elif action == "exec":
                # CPU was reset to new binary, save new state
                proc_mgr.context_switch_out(pid)
                proc = proc_mgr.processes.get(pid)
                if proc:
                    proc.state = ProcessState.READY

        elif result.stop_reason == StopReasonV2.MAX_CYCLES:
            # Time slice exhausted, save and schedule next
            proc_mgr.context_switch_out(pid)
            proc = proc_mgr.processes.get(pid)
            if proc and proc.state == ProcessState.RUNNING:
                proc.state = ProcessState.READY

    elapsed = time.perf_counter() - start
    ips = total_cycles / elapsed if elapsed > 0 else 0

    return {
        "total_cycles": total_cycles,
        "elapsed": elapsed,
        "ips": ips,
        "total_forks": proc_mgr.total_forks,
        "total_context_switches": proc_mgr.total_context_switches,
        "processes_created": proc_mgr.next_pid - 1,
        "stop_reason": "COMPLETE",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile and run C on ARM64 Metal GPU")
    parser.add_argument("source", help="C source file to compile and run")
    parser.add_argument("--max-cycles", type=int, default=50_000_000)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        bin_path = f.name

    try:
        if not compile_c(args.source, bin_path, quiet=args.quiet):
            sys.exit(1)
        load_and_run(bin_path, max_cycles=args.max_cycles, quiet=args.quiet)
    finally:
        if os.path.exists(bin_path):
            os.unlink(bin_path)
