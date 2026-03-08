#!/usr/bin/env python3
"""
ELF Loader — Load standard aarch64 Linux ELF binaries into Metal GPU memory.

Handles:
- ELF64 parsing (aarch64 / EM_AARCH64 = 183)
- PT_LOAD segment loading
- Linux process stack setup (argc, argv, envp, auxv)
- Static binaries only (no dynamic linker)

The stack layout follows the Linux ABI for aarch64:
    SP → argc (8 bytes)
         argv[0] pointer
         argv[1] pointer
         ...
         NULL (8 bytes)
         envp[0] pointer
         envp[1] pointer
         ...
         NULL (8 bytes)
         auxv[0] (AT_type, AT_val — 16 bytes each)
         ...
         AT_NULL (16 bytes)
         [padding to 16-byte alignment]
         argv strings (null-terminated)
         envp strings (null-terminated)

Author: Robert Price
Date: 2026
"""

import struct
from pathlib import Path
from typing import Optional


# ELF constants
ELF_MAGIC = b'\x7fELF'
ELFCLASS64 = 2
ELFDATA2LSB = 1  # Little-endian
ET_EXEC = 2      # Executable
ET_DYN = 3       # Shared object (PIE)
EM_AARCH64 = 183

# Program header types
PT_NULL = 0
PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_NOTE = 4
PT_PHDR = 6

# Segment flags
PF_X = 1
PF_W = 2
PF_R = 4

# Auxiliary vector types (subset needed by musl)
AT_NULL = 0
AT_PHDR = 3       # Program headers address
AT_PHENT = 4      # Size of program header entry
AT_PHNUM = 5      # Number of program headers
AT_PAGESZ = 6     # Page size
AT_ENTRY = 9      # Entry point
AT_UID = 11
AT_EUID = 12
AT_GID = 13
AT_EGID = 14
AT_HWCAP = 16     # Hardware capabilities
AT_CLKTCK = 17    # Clock ticks per second
AT_SECURE = 23    # Secure mode (0 = normal, avoids ppoll/openat on init)
AT_RANDOM = 25    # Address of 16 random bytes
AT_HWCAP2 = 26


class ELFLoadError(Exception):
    pass


class ELFSegment:
    """Represents a loadable segment from an ELF binary."""
    __slots__ = ('vaddr', 'memsz', 'filesz', 'offset', 'flags', 'align')

    def __init__(self, vaddr, memsz, filesz, offset, flags, align):
        self.vaddr = vaddr
        self.memsz = memsz
        self.filesz = filesz
        self.offset = offset
        self.flags = flags
        self.align = align


class ELFInfo:
    """Parsed ELF binary information."""
    __slots__ = ('entry', 'segments', 'phdr_vaddr', 'phdr_size', 'phdr_count',
                 'elf_type', 'raw_data')

    def __init__(self):
        self.entry = 0
        self.segments = []
        self.phdr_vaddr = 0
        self.phdr_size = 0
        self.phdr_count = 0
        self.elf_type = ET_EXEC
        self.raw_data = b''


def parse_elf(data: bytes) -> ELFInfo:
    """Parse an ELF64 binary for aarch64.

    Args:
        data: Raw ELF file contents

    Returns:
        ELFInfo with entry point, loadable segments, etc.

    Raises:
        ELFLoadError: If the file is not a valid aarch64 ELF64
    """
    if len(data) < 64:
        raise ELFLoadError("File too small to be ELF")

    # Check magic
    if data[:4] != ELF_MAGIC:
        raise ELFLoadError(f"Bad ELF magic: {data[:4].hex()}")

    # Check class (64-bit)
    if data[4] != ELFCLASS64:
        raise ELFLoadError(f"Not ELF64 (class={data[4]})")

    # Check endianness (little)
    if data[5] != ELFDATA2LSB:
        raise ELFLoadError(f"Not little-endian (data={data[5]})")

    # Parse ELF header
    #   e_type(2) e_machine(2) e_version(4) e_entry(8) e_phoff(8) e_shoff(8)
    #   e_flags(4) e_ehsize(2) e_phentsize(2) e_phnum(2) ...
    (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff,
     e_flags, e_ehsize, e_phentsize, e_phnum) = struct.unpack_from(
        '<HHIQQQIHHH', data, 16
    )

    if e_machine != EM_AARCH64:
        raise ELFLoadError(f"Not aarch64 (machine={e_machine})")

    if e_type not in (ET_EXEC, ET_DYN):
        raise ELFLoadError(f"Not executable (type={e_type})")

    info = ELFInfo()
    info.entry = e_entry
    info.phdr_size = e_phentsize
    info.phdr_count = e_phnum
    info.elf_type = e_type
    info.raw_data = data

    # Parse program headers
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if off + e_phentsize > len(data):
            break

        (p_type, p_flags, p_offset, p_vaddr, p_paddr,
         p_filesz, p_memsz, p_align) = struct.unpack_from(
            '<IIQQQQQQ', data, off
        )

        if p_type == PT_LOAD:
            info.segments.append(ELFSegment(
                vaddr=p_vaddr, memsz=p_memsz, filesz=p_filesz,
                offset=p_offset, flags=p_flags, align=p_align
            ))
        elif p_type == PT_PHDR:
            info.phdr_vaddr = p_vaddr

    if not info.segments:
        raise ELFLoadError("No PT_LOAD segments found")

    return info


def load_elf_into_memory(
    cpu,
    elf_path: str,
    argv: Optional[list[str]] = None,
    envp: Optional[dict[str, str]] = None,
    stack_top: int = 0,       # 0 = auto-detect from segments
    memory_limit: int = 0,    # 0 = use CPU memory size
    quiet: bool = False,
) -> int:
    """Load an ELF binary into GPU memory and set up the process stack.

    Args:
        cpu: MLXKernelCPUv2 instance
        elf_path: Path to the ELF binary
        argv: Command-line arguments (default: [basename])
        envp: Environment variables (default: minimal set)
        stack_top: Top of stack address (grows down)
        memory_limit: Maximum memory address available
        quiet: Suppress output

    Returns:
        Entry point address (caller should set PC to this)
    """
    path = Path(elf_path)
    if not path.exists():
        raise ELFLoadError(f"File not found: {elf_path}")

    data = path.read_bytes()
    info = parse_elf(data)

    if argv is None:
        argv = [path.name]
    if envp is None:
        envp = {
            "PATH": "/bin:/usr/bin",
            "HOME": "/",
            "TERM": "dumb",
            "USER": "root",
        }

    # Auto-detect memory layout from segments
    if memory_limit == 0:
        memory_limit = cpu.memory_size
    if stack_top == 0:
        # Place stack near top of GPU memory, well above loaded segments
        max_seg_end = max(s.vaddr + s.memsz for s in info.segments)
        # Round up to next 64KB boundary, then add generous gap for heap
        heap_start = (max_seg_end + 0xFFFF) & ~0xFFFF
        # Stack at top of memory minus a small margin
        stack_top = (memory_limit - 0x1000) & ~0xF
        if not quiet:
            print(f"[elf] Auto layout: segments end 0x{max_seg_end:X}, "
                  f"heap 0x{heap_start:X}, stack 0x{stack_top:X}")

    # Load PT_LOAD segments into GPU memory
    for seg in info.segments:
        if seg.vaddr + seg.memsz > memory_limit:
            if not quiet:
                print(f"[elf] WARNING: segment at 0x{seg.vaddr:X} "
                      f"(size 0x{seg.memsz:X}) exceeds memory limit 0x{memory_limit:X}")
            # Truncate to fit
            avail = memory_limit - seg.vaddr
            if avail <= 0:
                continue
            filesz = min(seg.filesz, avail)
        else:
            filesz = seg.filesz

        # Load file data
        seg_data = data[seg.offset:seg.offset + filesz]
        cpu.write_memory(seg.vaddr, seg_data)

        # BSS: zero-fill memsz - filesz (already zero in fresh memory)

        if not quiet:
            flags_str = ""
            if seg.flags & PF_R: flags_str += "R"
            if seg.flags & PF_W: flags_str += "W"
            if seg.flags & PF_X: flags_str += "X"
            print(f"[elf] Loaded segment: 0x{seg.vaddr:08X}-"
                  f"0x{seg.vaddr + seg.memsz:08X} "
                  f"({seg.filesz:,} bytes file, {seg.memsz:,} bytes mem) [{flags_str}]")

    # Build the process stack (Linux aarch64 ABI)
    # Stack grows downward from stack_top
    sp = stack_top

    # First, write the string data at the bottom of the stack area
    # (higher addresses, since stack grows down)
    string_area = sp - 0x1000  # Reserve 4KB for strings

    # Write argv strings
    argv_addrs = []
    str_ptr = string_area
    for arg in argv:
        arg_bytes = arg.encode('utf-8') + b'\x00'
        cpu.write_memory(str_ptr, arg_bytes)
        argv_addrs.append(str_ptr)
        str_ptr += len(arg_bytes)

    # Write envp strings
    envp_strs = [f"{k}={v}" for k, v in envp.items()]
    envp_addrs = []
    for env_str in envp_strs:
        env_bytes = env_str.encode('utf-8') + b'\x00'
        cpu.write_memory(str_ptr, env_bytes)
        envp_addrs.append(str_ptr)
        str_ptr += len(env_bytes)

    # Write 16 random bytes for AT_RANDOM
    import os
    random_addr = str_ptr
    cpu.write_memory(random_addr, os.urandom(16))
    str_ptr += 16

    # Now build the stack frame above the string area
    # Layout (growing UP from a base address):
    #   argc
    #   argv[0..n-1] pointers
    #   NULL
    #   envp[0..m-1] pointers
    #   NULL
    #   auxv entries (pairs of uint64)
    #   AT_NULL

    # Build auxiliary vector
    auxv = []
    auxv.append((AT_PAGESZ, 4096))
    auxv.append((AT_ENTRY, info.entry))
    auxv.append((AT_UID, 0))
    auxv.append((AT_EUID, 0))
    auxv.append((AT_GID, 0))
    auxv.append((AT_EGID, 0))
    auxv.append((AT_HWCAP, 0))        # No special hardware caps
    auxv.append((AT_CLKTCK, 100))     # 100 Hz
    auxv.append((AT_SECURE, 0))       # Non-secure (avoids ppoll/openat in musl init)
    auxv.append((AT_RANDOM, random_addr))
    # Program headers — needed by musl for TLS init
    if info.phdr_vaddr:
        auxv.append((AT_PHDR, info.phdr_vaddr))
    elif info.segments:
        # Fallback: phdr is at first segment's vaddr + e_phoff
        e_phoff = struct.unpack_from('<Q', data, 32)[0]
        auxv.append((AT_PHDR, info.segments[0].vaddr + e_phoff))
    auxv.append((AT_PHENT, info.phdr_size))
    auxv.append((AT_PHNUM, info.phdr_count))
    auxv.append((AT_NULL, 0))

    # Calculate total stack frame size
    frame_size = 8                              # argc
    frame_size += len(argv) * 8 + 8             # argv pointers + NULL
    frame_size += len(envp_strs) * 8 + 8        # envp pointers + NULL
    frame_size += len(auxv) * 16                # auxv entries (type + value)

    # Align to 16 bytes
    frame_size = (frame_size + 15) & ~15

    # Stack pointer = string_area - frame_size (leave room)
    sp = (string_area - frame_size) & ~0xF  # 16-byte aligned

    # Write the stack frame
    offset = sp

    # argc
    cpu.write_memory(offset, struct.pack('<Q', len(argv)))
    offset += 8

    # argv pointers
    for addr in argv_addrs:
        cpu.write_memory(offset, struct.pack('<Q', addr))
        offset += 8
    cpu.write_memory(offset, struct.pack('<Q', 0))  # NULL terminator
    offset += 8

    # envp pointers
    for addr in envp_addrs:
        cpu.write_memory(offset, struct.pack('<Q', addr))
        offset += 8
    cpu.write_memory(offset, struct.pack('<Q', 0))  # NULL terminator
    offset += 8

    # Auxiliary vector
    for at_type, at_val in auxv:
        cpu.write_memory(offset, struct.pack('<QQ', at_type, at_val))
        offset += 16

    if not quiet:
        print(f"[elf] Entry point: 0x{info.entry:08X}")
        print(f"[elf] Stack pointer: 0x{sp:08X}")
        print(f"[elf] argc={len(argv)}, envp={len(envp_strs)} vars")
        print(f"[elf] {len(info.segments)} segments loaded, "
              f"binary size: {len(data):,} bytes")

    # Set SP register (x31 / SP)
    # The Metal kernel handles reg 31 as SP in load/store contexts
    # We need to write SP into the registers array directly
    import numpy as np
    regs = cpu.get_registers_numpy()
    regs[31] = sp
    cpu.set_registers_numpy(regs)

    return info.entry


def load_and_run_elf(
    elf_path: str,
    argv: Optional[list[str]] = None,
    envp: Optional[dict[str, str]] = None,
    handler=None,
    max_cycles: int = 500_000_000,
    quiet: bool = False,
    filesystem=None,
    stdin_data: Optional[bytes] = None,
) -> dict:
    """Load an ELF binary and run it on the Metal GPU.

    Args:
        elf_path: Path to the ELF binary
        argv: Command-line arguments
        envp: Environment variables
        handler: Syscall handler (default: standard with busybox syscalls)
        max_cycles: Maximum cycles
        quiet: Suppress output
        filesystem: GPUFilesystem instance
        stdin_data: Bytes to serve as stdin (for piped input)

    Returns:
        Execution results dict
    """
    from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2
    from ncpu.os.gpu.runner import make_syscall_handler, run

    cpu = MLXKernelCPUv2(quiet=quiet)

    entry = load_elf_into_memory(
        cpu, elf_path, argv=argv, envp=envp, quiet=quiet
    )
    cpu.set_pc(entry)

    # Determine heap base from ELF segments
    elf_data = Path(elf_path).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    if handler is None:
        handler = make_busybox_syscall_handler(
            filesystem=filesystem, heap_base=heap_base,
            stdin_data=stdin_data,
        )

    if not quiet:
        print("=" * 60)

    results = run(cpu, handler, max_cycles=max_cycles, quiet=quiet)

    if not quiet:
        print("=" * 60)
        print(f"Total cycles: {results['total_cycles']:,}")
        print(f"Elapsed: {results['elapsed']:.3f}s")
        print(f"IPS: {results['ips']:,.0f}")
        print(f"Stop: {results['stop_reason']}")

    # Attach CPU object for post-execution inspection (GPU superpowers)
    results["_cpu"] = cpu

    return results


def _pack_stat64(info: dict) -> bytes:
    """Pack a filesystem stat result into a Linux aarch64 stat64 struct (128 bytes).

    Linux aarch64 struct stat layout:
        0:  st_dev      (uint64)    8:  st_ino      (uint64)
        16: st_mode     (uint32)    20: st_nlink    (uint32)
        24: st_uid      (uint32)    28: st_gid      (uint32)
        32: st_rdev     (uint64)    40: __pad1      (int64)
        48: st_size     (int64)     56: st_blksize  (int32)
        60: __pad2      (int32)     64: st_blocks   (int64)
        72: st_atime    (int64)     80: st_atime_ns (int64)
        88: st_mtime    (int64)     96: st_mtime_ns (int64)
        104: st_ctime   (int64)     112: st_ctime_ns(int64)
        120: __unused4  (int32)     124: __unused5   (int32)
    """
    is_dir = info.get("type") == "dir"
    size = info.get("size", 0)
    path = info.get("path", "/")

    st_ino = hash(path) & 0xFFFFFFFFFFFFFFFF
    st_mode = 0o040755 if is_dir else 0o100644
    st_nlink = 2 if is_dir else 1
    st_size = 0 if is_dir else size
    st_blocks = (st_size + 511) // 512

    import time as _time
    now = int(_time.time())

    return struct.pack(
        '<QQIIIIQqqiiqqqqqqqii',
        1,                  # st_dev
        st_ino,             # st_ino
        st_mode,            # st_mode
        st_nlink,           # st_nlink
        0, 0,               # st_uid, st_gid
        0,                  # st_rdev
        0,                  # __pad1
        st_size,            # st_size
        4096,               # st_blksize
        0,                  # __pad2
        st_blocks,          # st_blocks
        now, 0,             # st_atime, st_atime_nsec
        now, 0,             # st_mtime, st_mtime_nsec
        now, 0,             # st_ctime, st_ctime_nsec
        0, 0,               # __unused4, __unused5
    )


def make_busybox_syscall_handler(filesystem=None, heap_base=None, stdin_data=None):
    """Create a syscall handler with additional syscalls needed by musl/busybox.

    This extends the standard handler with:
    - ioctl (terminal queries)
    - writev (vectored write)
    - mmap/mprotect/munmap (memory mapping stubs)
    - set_tid_address (TLS)
    - rt_sigaction/rt_sigprocmask (signal stubs)
    - clock_gettime (time)
    - fcntl (file control)
    - newfstatat (stat)
    """
    import time
    import struct
    from ncpu.os.gpu.runner import (
        make_syscall_handler, read_string_from_gpu,
        HEAP_BASE, SYS_EXIT, SYS_WRITE, SYS_READ, SYS_BRK,
        SYS_OPENAT, SYS_CLOSE, SYS_LSEEK, SYS_FSTAT,
        SYS_GETCWD, SYS_MKDIRAT, SYS_UNLINKAT, SYS_CHDIR,
        SYS_GETDENTS64, SYS_DUP3, SYS_PIPE2,
        SYS_GETPID, SYS_GETPPID,
    )

    # Build stdin reader if stdin_data is provided (for piped input)
    stdin_reader = None
    if stdin_data is not None:
        stdin_buf = bytearray(stdin_data)
        stdin_offset = [0]  # mutable ref

        def stdin_reader(fd, max_len):
            if fd != 0:
                return None
            remaining = len(stdin_buf) - stdin_offset[0]
            if remaining <= 0:
                return b""  # EOF
            chunk = bytes(stdin_buf[stdin_offset[0]:stdin_offset[0] + max_len])
            stdin_offset[0] += len(chunk)
            return chunk

    # Additional Linux/aarch64 syscall numbers
    SYS_EXIT_GROUP = 94
    SYS_IOCTL = 29
    SYS_FCNTL = 25
    SYS_WRITEV = 66
    SYS_CLOCK_GETTIME = 113
    SYS_RT_SIGACTION = 134
    SYS_RT_SIGPROCMASK = 135
    SYS_RT_SIGRETURN = 139
    SYS_MMAP = 222
    SYS_MPROTECT = 226
    SYS_MUNMAP = 215
    SYS_SET_TID_ADDRESS = 96
    SYS_SET_ROBUST_LIST = 99
    SYS_NEWFSTATAT = 79
    SYS_GETTID = 178
    SYS_GETUID = 174
    SYS_GETEUID = 175
    SYS_GETGID = 176
    SYS_GETEGID = 177
    SYS_UNAME = 160
    SYS_GETRANDOM = 278
    SYS_PPOLL = 73
    SYS_CLONE = 220
    SYS_READLINKAT = 78
    SYS_FACCESSAT = 48
    SYS_RENAMEAT = 38
    SYS_STATFS = 43
    SYS_FCHMOD = 52
    SYS_FCHOWN = 55
    SYS_PREAD64 = 67
    SYS_PWRITE64 = 68
    SYS_UTIMENSAT = 88
    SYS_SYSINFO = 179
    SYS_SYMLINKAT = 36
    SYS_FCHMODAT = 53
    SYS_NANOSLEEP = 101
    SYS_CLOCK_NANOSLEEP = 115
    SYS_SCHED_GETAFFINITY = 123
    SYS_TIMER_SETTIME = 110
    SYS_CLOCK_GETTIME = 113
    SYS_CLOCK_GETRES = 114
    SYS_GETCWD = 17

    _heap_base = heap_base if heap_base else HEAP_BASE
    heap_break = _heap_base
    mmap_next = _heap_base + 0x100000  # mmap region 1MB above heap start

    # Get the base syscall handler
    base_handler = make_syscall_handler(filesystem=filesystem, on_read=stdin_reader)

    def handler(cpu) -> bool:
        nonlocal heap_break, mmap_next

        syscall_num = cpu.get_register(8)
        x0 = cpu.get_register(0)
        x1 = cpu.get_register(1)
        x2 = cpu.get_register(2)
        x3 = cpu.get_register(3)

        # ═══════════════════════════════════════════════════════════
        # MUSL INITIALIZATION SYSCALLS
        # ═══════════════════════════════════════════════════════════

        if syscall_num == SYS_EXIT_GROUP:
            return False  # Exit

        elif syscall_num == SYS_SET_TID_ADDRESS:
            # musl calls this during init to set the TID pointer
            cpu.set_register(0, 1)  # Return TID = 1
            return True

        elif syscall_num == SYS_SET_ROBUST_LIST:
            # musl calls this for futex robust list
            cpu.set_register(0, 0)  # Success
            return True

        elif syscall_num == SYS_GETTID:
            cpu.set_register(0, 1)
            return True

        elif syscall_num == SYS_GETUID:
            cpu.set_register(0, 0)  # root
            return True

        elif syscall_num == SYS_GETEUID:
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_GETGID:
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_GETEGID:
            cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════
        # MEMORY MANAGEMENT (stubs for single-process GPU)
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_MMAP:
            # x0=addr, x1=length, x2=prot, x3=flags, x4=fd, x5=offset
            addr = x0
            length = x1
            if addr == 0:
                # Anonymous mapping — allocate from mmap region
                aligned = (mmap_next + 0xFFF) & ~0xFFF  # Page align
                mmap_next = aligned + length
                cpu.set_register(0, aligned)
            else:
                # Fixed mapping — just succeed
                cpu.set_register(0, addr)
            return True

        elif syscall_num == SYS_MPROTECT:
            # Just succeed — no memory protection on GPU
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_MUNMAP:
            # Just succeed
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_BRK:
            if x0 == 0:
                cpu.set_register(0, heap_break)
            elif x0 >= _heap_base:
                heap_break = x0
                cpu.set_register(0, heap_break)
            else:
                cpu.set_register(0, heap_break)
            return True

        # ═══════════════════════════════════════════════════════════
        # SIGNAL HANDLING (stubs)
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_RT_SIGACTION:
            cpu.set_register(0, 0)  # Success
            return True

        elif syscall_num == SYS_RT_SIGPROCMASK:
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_RT_SIGRETURN:
            cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════
        # I/O EXTENSIONS
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_IOCTL:
            fd = x0
            request = x1
            # TIOCGWINSZ = 0x5413 — terminal window size
            if request == 0x5413:
                # Return 24 rows, 80 columns
                import sys as _sys
                buf_addr = x2
                winsize = struct.pack('<HHHH', 24, 80, 0, 0)
                cpu.write_memory(buf_addr, winsize)
                cpu.set_register(0, 0)
            else:
                # Unknown ioctl — return ENOTTY
                cpu.set_register(0, -25)  # -ENOTTY
            return True

        elif syscall_num == SYS_WRITEV:
            fd = x0
            iov_addr = x1
            iovcnt = x2
            total = 0
            import sys as _sys
            for i in range(iovcnt):
                # struct iovec { void *iov_base; size_t iov_len; }
                base_addr = int.from_bytes(
                    cpu.read_memory(iov_addr + i * 16, 8), 'little')
                iov_len = int.from_bytes(
                    cpu.read_memory(iov_addr + i * 16 + 8, 8), 'little')
                if iov_len > 0 and base_addr > 0:
                    data = cpu.read_memory(base_addr, iov_len)
                    if fd in (1, 2):
                        _sys.stdout.write(data.decode('ascii', errors='replace'))
                        _sys.stdout.flush()
                    elif filesystem and fd > 2:
                        filesystem.write(fd, data)
                    total += iov_len
            cpu.set_register(0, total)
            return True

        elif syscall_num == SYS_FCNTL:
            fd = x0
            cmd = x1
            # F_GETFD=1, F_SETFD=2, F_GETFL=3, F_SETFL=4, F_DUPFD_CLOEXEC=1030
            if cmd in (1, 3):
                cpu.set_register(0, 0)  # Return 0 flags
            elif cmd in (2, 4):
                cpu.set_register(0, 0)  # Success
            elif cmd == 1030:  # F_DUPFD_CLOEXEC
                cpu.set_register(0, x2)  # Return the suggested fd
            else:
                cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════
        # TIME
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_CLOCK_GETTIME:
            clock_id = x0
            buf_addr = x1
            import time as _time
            t = _time.time()
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            cpu.write_memory(buf_addr, struct.pack('<qq', sec, nsec))
            cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════
        # SYSTEM INFO
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_UNAME:
            buf_addr = x0
            # struct utsname: 5 fields × 65 bytes each
            fields = [
                b'Linux\x00',           # sysname
                b'ncpu-gpu\x00',        # nodename
                b'6.1.0-ncpu\x00',      # release
                b'#1 SMP Metal GPU\x00',  # version
                b'aarch64\x00',         # machine
            ]
            offset = 0
            for field in fields:
                padded = field.ljust(65, b'\x00')
                cpu.write_memory(buf_addr + offset, padded)
                offset += 65
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_GETRANDOM:
            buf_addr = x0
            length = x1
            import os as _os
            cpu.write_memory(buf_addr, _os.urandom(min(length, 256)))
            cpu.set_register(0, min(length, 256))
            return True

        # ═══════════════════════════════════════════════════════════
        # FILE SYSTEM EXTENSIONS
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_NEWFSTATAT:
            # newfstatat(dirfd, path, statbuf, flags)
            dirfd = x0
            path_addr = x1
            buf_addr = x2
            flags = x3
            AT_EMPTY_PATH = 0x1000

            info = None
            if path_addr:
                path = read_string_from_gpu(cpu, path_addr)
            else:
                path = ""

            if filesystem:
                if path:
                    info = filesystem.stat(path)
                if info is None and (not path or (flags & AT_EMPTY_PATH)):
                    # Empty path with AT_EMPTY_PATH → fstat on dirfd
                    info = filesystem.fstat(dirfd) if dirfd >= 0 else None

            if info:
                stat_buf = _pack_stat64(info)
                cpu.write_memory(buf_addr, stat_buf)
                cpu.set_register(0, 0)
            else:
                cpu.set_register(0, -2)  # -ENOENT
            return True

        elif syscall_num == SYS_READLINKAT:
            cpu.set_register(0, -22)  # -EINVAL (no symlinks)
            return True

        elif syscall_num == SYS_FACCESSAT:
            # faccessat(dirfd, path, mode, flags)
            path = read_string_from_gpu(cpu, x1)
            if filesystem:
                resolved = filesystem.resolve_path(path)
                exists = (resolved in filesystem.files or
                         resolved in filesystem.directories)
                cpu.set_register(0, 0 if exists else -2)  # -ENOENT
            else:
                cpu.set_register(0, -2)
            return True

        elif syscall_num == SYS_PREAD64:
            # pread64(fd, buf, count, offset) — positional read
            fd = x0
            buf_addr = x1
            count = x2
            pos = x3
            if filesystem and fd in filesystem.fd_table:
                entry = filesystem.fd_table[fd]
                saved_offset = entry["offset"]
                entry["offset"] = pos
                data = filesystem.read(fd, count)
                entry["offset"] = saved_offset
                if data is not None:
                    cpu.write_memory(buf_addr, data)
                    cpu.set_register(0, len(data))
                else:
                    cpu.set_register(0, -9)  # -EBADF
            else:
                cpu.set_register(0, -9)  # -EBADF
            return True

        elif syscall_num == SYS_PWRITE64:
            # pwrite64(fd, buf, count, offset) — positional write
            fd = x0
            buf_addr = x1
            count = x2
            pos = x3
            if filesystem and fd in filesystem.fd_table:
                entry = filesystem.fd_table[fd]
                saved_offset = entry["offset"]
                entry["offset"] = pos
                data = cpu.read_memory(buf_addr, count)
                written = filesystem.write(fd, data)
                entry["offset"] = saved_offset
                cpu.set_register(0, written if written >= 0 else -9)
            else:
                cpu.set_register(0, -9)  # -EBADF
            return True

        elif syscall_num == SYS_RENAMEAT:
            # renameat(olddirfd, oldpath, newdirfd, newpath)
            old_path = read_string_from_gpu(cpu, x1)
            new_path = read_string_from_gpu(cpu, x3)
            if filesystem:
                result = filesystem.rename(old_path, new_path)
                cpu.set_register(0, result)
            else:
                cpu.set_register(0, -2)  # -ENOENT
            return True

        elif syscall_num == SYS_FCHMOD:
            # fchmod(fd, mode) — stub, no real permissions
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_FCHOWN:
            # fchown(fd, owner, group) — stub
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_UTIMENSAT:
            # utimensat(dirfd, path, times, flags)
            path_addr = x1
            if path_addr and filesystem:
                path = read_string_from_gpu(cpu, path_addr)
                resolved = filesystem.resolve_path(path)
                if resolved in filesystem.files or resolved in filesystem.directories:
                    cpu.set_register(0, 0)  # Success
                else:
                    cpu.set_register(0, -2)  # -ENOENT
            else:
                cpu.set_register(0, 0)  # No path = fstat-like, success
            return True

        elif syscall_num == SYS_STATFS:
            # statfs(path, buf) — return fake statfs struct
            buf_addr = x1
            # struct statfs: f_type(8), f_bsize(8), f_blocks(8), f_bfree(8),
            #   f_bavail(8), f_files(8), f_ffree(8), f_fsid(8), f_namelen(8),
            #   f_frsize(8), f_flags(8), f_spare[4](32) = 120 bytes
            statfs_buf = struct.pack('<qqqqqqqqqqq',
                0x4E435055,   # f_type "NCPU" magic
                4096,         # f_bsize
                1048576,      # f_blocks (4GB)
                524288,       # f_bfree
                524288,       # f_bavail
                65536,        # f_files
                32768,        # f_ffree
                0,            # f_fsid
                255,          # f_namelen
                4096,         # f_frsize
                0,            # f_flags
            )
            statfs_buf += b'\x00' * (120 - len(statfs_buf))
            cpu.write_memory(buf_addr, statfs_buf)
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_SYSINFO:
            # sysinfo(buf) — return fake sysinfo struct
            buf_addr = x0
            import time as _time
            uptime = int(_time.time()) % 86400
            # struct sysinfo (112 bytes on aarch64):
            #   uptime(8), loads[3](24), totalram(8), freeram(8),
            #   sharedram(8), bufferram(8), totalswap(8), freeswap(8),
            #   procs(2), pad(2), pad2(4), totalhigh(8), freehigh(8), mem_unit(4), pad3(4)
            sysinfo_buf = struct.pack('<q3qqqqqqqHHIqqII',
                uptime,            # uptime
                0, 0, 0,           # loads[3]
                256 * 1024 * 1024, # totalram (256MB)
                128 * 1024 * 1024, # freeram
                0,                 # sharedram
                0,                 # bufferram
                0,                 # totalswap
                0,                 # freeswap
                1,                 # procs
                0,                 # pad
                0,                 # pad2
                0,                 # totalhigh
                0,                 # freehigh
                1,                 # mem_unit
                0,                 # pad3
            )
            cpu.write_memory(buf_addr, sysinfo_buf)
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_PPOLL:
            # ppoll — just return immediately (no blocking)
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_CLONE:
            # Don't support clone/fork in busybox mode
            cpu.set_register(0, -38)  # -ENOSYS
            return True

        # ═══════════════════════════════════════════════════════════
        # ADDITIONAL POSIX SYSCALLS
        # ═══════════════════════════════════════════════════════════

        elif syscall_num == SYS_FCHMODAT:
            # fchmodat(dirfd, path, mode) — stub success
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_SYMLINKAT:
            # symlinkat(target, newdirfd, linkpath) — stub EPERM
            cpu.set_register(0, -1)  # -EPERM
            return True

        elif syscall_num in (SYS_NANOSLEEP, SYS_CLOCK_NANOSLEEP):
            # nanosleep/clock_nanosleep — return immediately
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_CLOCK_GETTIME:
            # clock_gettime(clockid, timespec*)
            # Write current time as struct timespec {tv_sec, tv_nsec}
            import time as _time
            ts_addr = x1
            now = _time.time()
            tv_sec = int(now)
            tv_nsec = int((now - tv_sec) * 1e9)
            cpu.write_memory(ts_addr, struct.pack('<qq', tv_sec, tv_nsec))
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_CLOCK_GETRES:
            # clock_getres(clockid, timespec*)
            ts_addr = x1
            if ts_addr != 0:
                cpu.write_memory(ts_addr, struct.pack('<qq', 0, 1000000))  # 1ms
            cpu.set_register(0, 0)
            return True

        elif syscall_num in (SYS_SCHED_GETAFFINITY, SYS_TIMER_SETTIME, 122, 158):
            # sched_getaffinity/sched_setaffinity/timer stubs — return success
            cpu.set_register(0, 0)
            return True

        elif syscall_num == SYS_GETCWD:
            # getcwd(buf, size)
            buf_addr = x0
            buf_size = x1
            cwd = b'/\x00'
            cpu.write_memory(buf_addr, cwd)
            cpu.set_register(0, buf_addr)
            return True

        # ═══════════════════════════════════════════════════════════
        # STUB SYSCALLS (return error codes instead of hanging)
        # ═══════════════════════════════════════════════════════════

        elif syscall_num in (56,):  # openat — return ENOENT if no filesystem
            if filesystem:
                return base_handler(cpu)
            cpu.set_register(0, -2)  # -ENOENT
            return True

        elif syscall_num in (57,):  # close
            if filesystem:
                return base_handler(cpu)
            cpu.set_register(0, 0)
            return True

        # ═══════════════════════════════════════════════════════════
        # FALL THROUGH TO BASE HANDLER
        # ═══════════════════════════════════════════════════════════

        else:
            return base_handler(cpu)

    return handler


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <elf-binary> [args...]")
        sys.exit(1)

    elf_path = sys.argv[1]
    argv = sys.argv[1:]  # argv[0] = program name

    results = load_and_run_elf(
        elf_path,
        argv=argv,
        max_cycles=500_000_000,
    )
