# C-to-GPU Compilation Pipeline

End-to-end flow from C source code to execution on the Metal GPU compute shader.

## Pipeline Overview

```
C source file (.c)
    │
    ▼
┌─────────────────────────────┐
│ aarch64-elf-gcc -O2         │  Cross-compiler (host)
│ + arm64_start.S (startup)   │  Startup code: SP init, call main, SVC EXIT
│ + arm64.ld (linker script)  │  Memory layout: .text=0x10000, .data=0x50000
└─────────────────────────────┘
    │
    ▼
ELF binary (.elf)
    │
    ▼
┌─────────────────────────────┐
│ aarch64-elf-objcopy -O bin  │  Extract raw binary
└─────────────────────────────┘
    │
    ▼
Raw ARM64 binary (.bin)
    │
    ▼
┌─────────────────────────────┐
│ load_program(binary, 0x10000)│  Load into GPU memory
│ set_pc(0x10000)              │  Set entry point
│ init_svc_buffer()            │  Initialize SVC write buffer
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Metal GPU Compute Shader     │  Fetch-decode-execute loop
│ 139 ARM64 instructions       │  Single-threaded, deterministic
│ SVC buffer for I/O           │  stdout/stderr buffered on GPU
└─────────────────────────────┘
    │
    ├── SYS_WRITE → GPU buffer (no Python trap)
    ├── SYS_BRK   → GPU heap management (no Python trap)
    ├── SYS_EXIT  → GPU halt (no Python trap)
    ├── Other SVC → Trap to Python syscall handler
    │
    ▼
Python syscall handler
    │
    ├── SYS_OPENAT/READ/WRITE → GPUFilesystem
    ├── SYS_EXEC → Load and run new binary
    ├── SYS_FORK/WAIT/PIPE → ProcessManager
    └── ... (50+ syscalls)
```

## Memory Layout

```
0x00000 ─────────────────────────────────────
        Reserved (interrupt vectors, etc.)
0x10000 ─────────────────────────────────────
        .text section (program code)
        Entry point: _start → main
0x50000 ─────────────────────────────────────
        .data section (initialized data)
        .rodata (string literals, constants)
        .bss (zero-initialized)
0x60000 ─────────────────────────────────────
        Heap (grows upward via SYS_BRK)
        malloc/free/calloc/realloc manage this
        ...
0xFF000 ─────────────────────────────────────
        Stack (grows downward)
        SP initialized to 0xFF000 by _start
0x100000 ────────────────────────────────────
        Process backing stores (multi-process mode)
        1MB per process, max 15 processes
        ...
0x3F0000 ────────────────────────────────────
        SVC write buffer (64KB)
        GPU-side buffered I/O
0x400000 ────────────────────────────────────
        End of typical program space
        ...
0x1000000 (16MB) ────────────────────────────
        End of GPU memory
```

## Cross-Compilation

### Toolchain

- **Compiler**: `aarch64-elf-gcc` (bare-metal ARM64 cross-compiler)
- **Flags**: `-nostdlib -ffreestanding -static -O2 -march=armv8-a -mgeneral-regs-only`
- **Startup**: `arm64_start.S` — sets SP, calls `main()`, issues SVC EXIT with return code
- **Linker script**: `arm64.ld` — places .text at 0x10000, .data at 0x50000

### Why `-mgeneral-regs-only`

The Metal shader does not implement floating-point or full NEON SIMD instructions. This flag forces GCC to use only general-purpose registers (X0-X30). SIMD load/store (LDR/STR Q-register) is supported for musl va_list save/restore only.

### Freestanding C Runtime

Programs include `selfhost.h` which provides:
- Memory: `malloc`, `free`, `calloc`, `realloc`
- I/O: `printf` (full format specifiers), `getchar`, `puts`
- Strings: `strlen`, `strcmp`, `strcpy`, `strncpy`, `strstr`, `memset`, `memcpy`, `memmove`
- Math: `strtol`, `strtoul`, `atoi`, `atol`, `itoa`
- Algorithms: `qsort`, `rand`, `srand`
- Process: `fork`, `wait`, `pipe`, `dup2`, `getpid`, `kill`
- System: `clock_ms`, `sleep_ms`, `open`, `read`, `write`, `close`

## Compilation Cache

The `CompileCache` avoids redundant recompilation:

```python
from ncpu.os.gpu.compile_cache import CompileCache

cache = CompileCache()  # ~/.ncpu/compile_cache/
key = cache.cache_key(source_bytes, flags_string)
binary = cache.get(key)  # SHA-256 content-addressed lookup
```

Cache hits skip GCC entirely, loading the binary directly from `~/.ncpu/compile_cache/`. This is especially beneficial for the 73 test programs in cc_demo, which would otherwise recompile on every run.

## ELF Loading (BusyBox Path)

For pre-compiled ELF binaries (like BusyBox), the pipeline uses `elf_loader.py` instead of GCC:

```
BusyBox ELF binary (264KB, aarch64-linux-musl-gcc -static)
    │
    ▼
┌─────────────────────────────┐
│ parse_elf(binary)            │  Parse ELF64 headers
│ load_elf_into_memory(cpu)    │  Load PT_LOAD segments
│ Setup Linux stack            │  argc, argv, envp, auxv
└─────────────────────────────┘
    │
    ▼
GPU execution with Linux syscall emulation
    │
    ├── SYS_OPENAT    → GPUFilesystem
    ├── SYS_GETDENTS64 → linux_dirent64 format
    ├── SYS_NEWFSTATAT → aarch64 stat64 struct (128 bytes)
    ├── SYS_MMAP      → Memory region allocation
    ├── SYS_UNAME     → Fake Linux 5.15.0 aarch64
    └── ... (50+ Linux syscalls)
```

## Self-Hosting Path

The most complex pipeline: GPU compiling C on GPU:

```
1. Host GCC compiles cc.c → Stage0 binary (compiler₀)
2. GPU loads and runs Stage0
   ├── Stage0 reads test.c from GPUFilesystem
   ├── Stage0 compiles test.c → ARM64 machine code
   ├── Stage0 writes binary via SYS_WRITE to filesystem
3. GPU loads and runs the GPU-compiled binary
4. Verify: exit code matches expected result

Self-compilation (meta-compilation):
1. Host GCC compiles cc.c → Stage0
2. GPU runs Stage0, which compiles cc.c → Stage1
3. GPU runs Stage1, which compiles test.c → test binary
4. GPU runs test binary → correct result
```

The self-hosting compiler uses NCCD compact binary format (not ELF) for data sections, since the GPU filesystem doesn't support `lseek` for ELF section patching.

## Syscall Handling

### GPU-Side (Zero Python Overhead)
| Syscall | Behavior |
|---------|----------|
| `SYS_WRITE(1/2)` | Buffer output at 0x3F0000 |
| `SYS_BRK` | Track heap break |
| `SYS_CLOSE(0/1/2)` | No-op, return 0 |
| `SYS_EXIT(93)` | Halt immediately |
| `SYS_EXIT_GROUP(231)` | Halt immediately |

### Python-Side (Full Emulation)
| Category | Syscalls |
|----------|----------|
| **Filesystem** | openat, read, write, close, fstat, newfstatat, getdents64, lseek, pread64, pwrite64 |
| **Memory** | brk, mmap, mprotect, munmap |
| **Process** | fork, wait4, pipe2, dup3, getpid, kill, exit_group |
| **System** | uname, clock_gettime, sched_getaffinity, sysinfo |
| **Network** | socket, bind, listen, accept, sendto, recvfrom |
