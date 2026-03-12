#!/usr/bin/env python3
"""
Self-Compilation Test — Can cc.c compile itself on the Metal GPU?

The ultimate test of a self-hosting compiler: cc.c running on the GPU
reads its own source code, includes the self-hosting header, and produces
an ARM64 binary. Then we verify the output by having the self-compiled
compiler compile a simple test program and checking the result.

Three layers:
  1. Host GCC compiles cc.c -> cc_stage0 binary
  2. GPU runs cc_stage0, which compiles cc.c -> cc_stage1 binary
  3. GPU runs cc_stage1, which compiles hello.c -> hello binary
  4. GPU runs hello binary -> check exit code
"""

import sys
import os
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
GPU_OS_DIR = TOOLS_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


def main():
    print("=" * 70)
    print("  SELF-COMPILATION TEST: cc.c compiles itself on Metal GPU")
    print("=" * 70)
    print()

    # ── Stage 0: Compile cc.c with host GCC ──────────────────────────────
    print("[Stage 0] Compiling cc.c with host GCC...")
    import tempfile
    cc_bin_path = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
    cc_src = str(TOOLS_DIR / "cc.c")

    if not compile_c(cc_src, cc_bin_path):
        print("FATAL: Cannot compile cc.c with host GCC")
        return 1

    cc_stage0 = Path(cc_bin_path).read_bytes()
    print(f"  Stage 0 binary: {len(cc_stage0):,} bytes")
    print()

    # ── Stage 1: Run cc_stage0 on GPU to compile cc.c -> cc_stage1 ───────
    print("[Stage 1] Running cc_stage0 on GPU to self-compile cc.c...")
    print("  (This compiles ~150KB of C source on the GPU — may take a while)")
    print()

    # Set up GPU filesystem with cc.c source and headers
    fs = GPUFilesystem()
    fs.mkdir("/tmp")
    fs.mkdir("/bin")
    fs.mkdir("/usr")
    fs.mkdir("/usr/include")

    # Read cc.c source
    cc_source = Path(cc_src).read_bytes()
    fs.write_file("/tmp/cc.c", cc_source)
    print(f"  Source: /tmp/cc.c ({len(cc_source):,} bytes)")

    # Read and install the self-hosting header
    selfhost_h = (GPU_OS_DIR / "src" / "arm64_selfhost.h").read_bytes()
    fs.write_file("/usr/include/arm64_selfhost.h", selfhost_h)
    print(f"  Header: /usr/include/arm64_selfhost.h ({len(selfhost_h):,} bytes)")

    # Write args file for the compiler
    args_content = "/tmp/cc.c\n/bin/cc\n"
    fs.write_file("/tmp/.cc_args", args_content.encode())

    # Run cc_stage0 on GPU
    cpu = MLXKernelCPUv2()
    cpu.load_program(cc_stage0, address=0x10000)
    cpu.set_pc(0x10000)
    base_handler = make_syscall_handler(filesystem=fs)

    # Wrap handler to debug exit and flush stdout
    def handler(cpu):
        syscall_num = cpu.get_register(8)
        if syscall_num == 93:  # SYS_EXIT
            code = cpu.get_register(0)
            print(f"\n  [DEBUG] SYS_EXIT called with code={code}", flush=True)
        ret = base_handler(cpu)
        if syscall_num == 93:
            print(f"  [DEBUG] base_handler returned: {ret!r}", flush=True)
        sys.stdout.flush()
        return ret

    start = time.perf_counter()
    # Self-compilation of 150KB source needs many cycles
    print("  [DEBUG] Starting run()...", flush=True)
    run_result = run(cpu, handler, max_cycles=2_000_000_000, quiet=False,
                     batch_size=5_000_000)
    elapsed = time.perf_counter() - start
    print(f"  [DEBUG] run() returned. stop_reason={run_result.get('stop_reason')}", flush=True)

    exit_code = cpu.get_register(0)
    cycles = run_result["total_cycles"]

    print(f"\n  Self-compilation result:", flush=True)
    print(f"    Exit code: {exit_code}", flush=True)
    print(f"    Cycles: {cycles:,}", flush=True)
    print(f"    Wall time: {elapsed:.1f}s", flush=True)

    if not fs.exists("/bin/cc"):
        print("\n  SELF-COMPILATION FAILED: /bin/cc not produced")
        # Error output was already printed to stdout via syscall handler
        os.unlink(cc_bin_path)
        return 1

    cc_stage1 = fs.read_file("/bin/cc")
    print(f"    Output: /bin/cc ({len(cc_stage1):,} bytes)")
    print()

    # ── Stage 2: Verify cc_stage1 by compiling a test program ────────────
    print("[Stage 2] Verifying self-compiled compiler with a test program...")

    test_source = """\
int main(void) {
    int a = 40;
    int b = 2;
    return a + b;
}
"""
    fs2 = GPUFilesystem()
    fs2.mkdir("/tmp")
    fs2.mkdir("/bin")
    fs2.write_file("/tmp/test.c", test_source.encode())
    fs2.write_file("/tmp/.cc_args", b"/tmp/test.c\n/bin/test\n")

    cpu2 = MLXKernelCPUv2()

    # Handle NCCD format for stage1 binary
    nccd_offset = cc_stage1.find(b'NCCD')
    if nccd_offset > 0 and nccd_offset + 8 <= len(cc_stage1):
        code_section = cc_stage1[:nccd_offset]
        data_size = int.from_bytes(cc_stage1[nccd_offset+4:nccd_offset+8], 'little')
        data_section = cc_stage1[nccd_offset+8:nccd_offset+8+data_size]
        cpu2.load_program(code_section, address=0x10000)
        if data_section:
            cpu2.write_memory(0x50000, data_section)
        print(f"  Stage 1 binary: {len(code_section):,} bytes code + {len(data_section):,} bytes data (NCCD)")
    else:
        cpu2.load_program(cc_stage1, address=0x10000)
        print(f"  Stage 1 binary: {len(cc_stage1):,} bytes (flat)")

    cpu2.set_pc(0x10000)
    base_handler2 = make_syscall_handler(filesystem=fs2)

    # Quiet handler: only log non-WRITE syscalls (putchar generates ~700 SVC traps)
    stage2_syscall_count = [0]
    stage2_stdout = []

    def handler2(cpu):
        syscall_num = cpu.get_register(8)
        stage2_syscall_count[0] += 1
        # Collect stdout output silently
        if syscall_num == 64:  # SYS_WRITE
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)
            x2 = cpu.get_register(2)
            if x0 == 1 and x2 == 1:  # putchar to stdout
                byte = cpu.read_memory(x1, 1)
                stage2_stdout.append(byte)
        elif syscall_num == 93:  # SYS_EXIT
            code = cpu.get_register(0)
            print(f"  [S2] EXIT code={code}", flush=True)
        elif syscall_num not in (214, 222, 226):  # Skip BRK/MMAP/MPROTECT noise
            x0 = cpu.get_register(0)
            x1 = cpu.get_register(1)
            x2 = cpu.get_register(2)
            print(f"  [S2 #{stage2_syscall_count[0]}] syscall={syscall_num} x0=0x{x0:x} x1=0x{x1:x} x2={x2}", flush=True)
        return base_handler2(cpu)

    start2 = time.perf_counter()
    run_result2 = run(cpu2, handler2, max_cycles=2_000_000_000, quiet=True,
                      batch_size=5_000_000)
    elapsed2 = time.perf_counter() - start2
    print(f"  [S2] Total syscalls: {stage2_syscall_count[0]}", flush=True)
    # Print collected stdout
    if stage2_stdout:
        stdout_text = b''.join(stage2_stdout).decode('ascii', errors='replace')
        print(f"  [S2] Compiler output:\n{stdout_text}", flush=True)

    exit_code2 = cpu2.get_register(0)
    cycles2 = run_result2["total_cycles"]

    print(f"  Stage 1 compilation: exit={exit_code2}, {cycles2:,} cycles, {elapsed2:.1f}s")

    if not fs2.exists("/bin/test"):
        print("  Stage 1 compiler FAILED to produce output")
        os.unlink(cc_bin_path)
        return 1

    test_bin = fs2.read_file("/bin/test")
    print(f"  Test binary: {len(test_bin):,} bytes")

    # ── Stage 3: Execute the test program ────────────────────────────────
    print("\n[Stage 3] Executing test program compiled by self-compiled compiler...")

    cpu3 = MLXKernelCPUv2()
    nccd_offset3 = test_bin.find(b'NCCD')
    if nccd_offset3 > 0 and nccd_offset3 + 8 <= len(test_bin):
        code3 = test_bin[:nccd_offset3]
        ds3 = int.from_bytes(test_bin[nccd_offset3+4:nccd_offset3+8], 'little')
        data3 = test_bin[nccd_offset3+8:nccd_offset3+8+ds3]
        cpu3.load_program(code3, address=0x10000)
        if data3:
            cpu3.write_memory(0x50000, data3)
    else:
        cpu3.load_program(test_bin, address=0x10000)

    cpu3.set_pc(0x10000)
    handler3 = make_syscall_handler()

    run_result3 = run(cpu3, handler3, max_cycles=10_000_000, quiet=True)
    exit_code3 = cpu3.get_register(0)
    expected = 42

    print(f"  Exit code: {exit_code3} (expected: {expected})")

    if exit_code3 == expected:
        print()
        print("=" * 70)
        print("  SELF-COMPILATION VERIFIED!")
        print()
        print("  cc.c (compiled by host GCC) ran on the Metal GPU,")
        print("  compiled its own source code into a new compiler binary,")
        print("  which then successfully compiled and ran a test program.")
        print()
        print(f"  Stage 0 (host GCC):    {len(cc_stage0):>10,} bytes")
        print(f"  Stage 1 (GPU self):    {len(cc_stage1):>10,} bytes")
        print(f"  Test binary:           {len(test_bin):>10,} bytes")
        print(f"  Self-compile cycles:   {cycles:>10,}")
        print(f"  Self-compile time:     {elapsed:>10.1f}s")
        print("=" * 70)
    else:
        print(f"\n  VERIFICATION FAILED: exit {exit_code3} != expected {expected}")

    os.unlink(cc_bin_path)
    return 0 if exit_code3 == expected else 1


if __name__ == "__main__":
    sys.exit(main())
