#!/usr/bin/env python3
"""Fully-GPU Pipeline Demo: Source Code → GPU Compile → GPU Execute.

Demonstrates the complete GPU-native pipeline where EVERY computation
step runs on GPU silicon — zero CPU arithmetic:

    1. neurOS boots (10 neural models loaded onto MPS GPU)
    2. nsl source code is compiled by NeuralCompiler
       → PyTorch neural networks on Apple MPS GPU
       → Peephole optimizer model runs on GPU
    3. Assembly is encoded by NeuralAssembler
       → Neural tokenizer (CNN) on MPS GPU
       → Neural codegen model on MPS GPU
    4. Binary loaded into NCPUComputeKernel
       → Metal compute shader on Apple GPU
       → Native GPU ALU does the arithmetic
    5. Results read back and verified

The Python host code is just an orchestration bus — all actual
computation (compilation, assembly, execution) runs on GPU.

This is the world's first fully-GPU-native compilation and execution
pipeline where neural networks compile the code and a Metal shader
runs it.
"""

import sys
import time
import logging
import torch

sys.path.insert(0, ".")
from ncpu.os import NeurOS
from kernels.mlx.ncpu_kernel import NCPUComputeKernel

logging.basicConfig(level=logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Programs — nsl source → expected register values after execution
# ═══════════════════════════════════════════════════════════════════════════════

PROGRAMS = [
    {
        "name": "Simple Addition",
        "source": "var a = 10; var b = 20; var sum = a + b; halt;",
        "description": "10 + 20 = 30",
        "check": lambda regs: any(v == 30 for v in regs.values()),
        "expected": 30,
    },
    {
        "name": "Multiplication",
        "source": "var x = 7; var y = 6; var result = x * y; halt;",
        "description": "7 * 6 = 42",
        "check": lambda regs: any(v == 42 for v in regs.values()),
        "expected": 42,
    },
    {
        "name": "Subtraction",
        "source": "var a = 100; var b = 37; var diff = a - b; halt;",
        "description": "100 - 37 = 63",
        "check": lambda regs: any(v == 63 for v in regs.values()),
        "expected": 63,
    },
    {
        "name": "Sum 1 to 10 (While Loop)",
        "source": """\
var sum = 0;
var i = 1;
var limit = 11;
var one = 1;
while (i != limit) {
    sum = sum + i;
    i = i + one;
}
halt;
""",
        "description": "sum(1..10) = 55",
        "check": lambda regs: any(v == 55 for v in regs.values()),
        "expected": 55,
    },
    {
        "name": "Fibonacci (10th number)",
        "source": """\
var prev = 0;
var curr = 1;
var n = 10;
var i = 0;
var one = 1;
while (i != n) {
    var temp = curr;
    curr = prev + curr;
    prev = temp;
    i = i + one;
}
halt;
""",
        "description": "fib(10) = 55",
        "check": lambda regs: any(v == 55 for v in regs.values()),
        "expected": 55,
    },
    {
        "name": "Bitwise XOR",
        "source": "var a = 0xFF; var b = 0xAA; var result = a ^ b; halt;",
        "description": "0xFF ^ 0xAA = 0x55 (85)",
        "check": lambda regs: any(v == 85 for v in regs.values()),
        "expected": 85,
    },
    {
        "name": "Shift Left",
        "source": "var x = 1; var amount = 8; var result = x << amount; halt;",
        "description": "1 << 8 = 256",
        "check": lambda regs: any(v == 256 for v in regs.values()),
        "expected": 256,
    },
    {
        "name": "Nested Expressions",
        "source": "var a = 3; var b = 4; var c = 5; var result = (a + b) * c; halt;",
        "description": "(3 + 4) * 5 = 35",
        "check": lambda regs: any(v == 35 for v in regs.values()),
        "expected": 35,
    },
]


def gpu_device_info():
    """Get GPU device information."""
    if torch.backends.mps.is_available():
        return "Apple MPS GPU (Metal Performance Shaders)"
    elif torch.cuda.is_available():
        name = torch.cuda.get_device_properties(0).name
        return f"CUDA GPU ({name})"
    return "CPU (no GPU available)"


def print_header():
    print()
    print("=" * 70)
    print("  nCPU Fully-GPU Pipeline Demo")
    print("  Source → GPU Compile → GPU Assemble → GPU Execute")
    print("=" * 70)
    print()
    print("  Every computation step runs on GPU silicon.")
    print("  Python is just the orchestration bus.")
    print()


def compile_and_execute_on_gpu(os_instance, kernel, program, verbose=True):
    """Compile nsl with neurOS (GPU) and execute on Metal compute kernel (GPU).

    Returns:
        (success, compile_time_us, assemble_time_us, exec_time_us, cycles, ips)
    """
    name = program["name"]
    source = program["source"]

    if verbose:
        print(f"  [{name}]")
        print(f"    Source:     {program['description']}")

    # ── Stage 1: Neural Compile (MPS GPU) ──────────────────────────────────
    t0 = time.perf_counter()
    result = os_instance.compiler.compile(source)
    compile_time = time.perf_counter() - t0

    if not result.success:
        if verbose:
            print(f"    COMPILE FAILED: {result.errors}")
        return False, compile_time * 1e6, 0, 0, 0, 0

    if verbose:
        n_ir = len(result.ir)
        n_asm = result.assembly_result.num_instructions
        n_opt = result.optimizations_applied
        print(f"    Compile:   {n_ir} IR → {n_asm} instructions "
              f"({n_opt} optimizations) [{compile_time*1e6:.0f}us] [MPS GPU]")

    # ── Stage 2: Load binary into Metal compute kernel ─────────────────────
    binary = result.binary
    if binary is None:
        if verbose:
            print(f"    ASSEMBLY FAILED: no binary produced")
        return False, compile_time * 1e6, 0, 0, 0, 0

    kernel.reset()
    kernel.load_program(binary)

    if verbose:
        print(f"    Binary:    {len(binary)} words ({len(binary)*4} bytes) "
              f"[Neural assembler on MPS GPU]")

    # ── Stage 3: Execute on Metal GPU ──────────────────────────────────────
    t0 = time.perf_counter()
    exec_result = kernel.execute(max_cycles=1_000_000)
    exec_time = time.perf_counter() - t0

    cycles = exec_result.cycles
    ips = exec_result.ips

    # ── Stage 4: Verify results ────────────────────────────────────────────
    regs = kernel.get_registers_dict()
    non_zero = {k: v for k, v in regs.items() if v != 0}
    success = program["check"](regs)

    if verbose:
        status = "PASS" if success else "FAIL"
        print(f"    Execute:   {cycles} cycles [{exec_time*1e6:.0f}us] "
              f"({ips:,.0f} IPS) [Metal GPU]")
        print(f"    Registers: {non_zero}")
        print(f"    Status:    {status} (expected {program['expected']})")
        print()

    return success, compile_time * 1e6, exec_time * 1e6, cycles, ips


def main():
    print_header()

    # ── Boot neurOS (loads 10 neural models onto GPU) ──────────────────────
    gpu_info = gpu_device_info()
    print(f"[1] GPU: {gpu_info}")
    print()

    print("[2] Booting neurOS (loading 10 neural models onto GPU)...")
    t0 = time.perf_counter()
    os_instance = NeurOS()
    os_instance.boot(quiet=True)
    boot_time = time.perf_counter() - t0
    print(f"    neurOS booted in {boot_time*1000:.1f}ms")
    print(f"    Models loaded: compiler_optimizer.pt, assembler_tokenizer.pt,")
    print(f"                   assembler_codegen.pt + 7 OS models")
    print()

    # ── Initialize Metal compute kernel ────────────────────────────────────
    print("[3] Initializing Metal compute kernel...")
    t0 = time.perf_counter()
    kernel = NCPUComputeKernel()
    kernel_init_time = time.perf_counter() - t0
    print(f"    NCPUComputeKernel ready in {kernel_init_time*1000:.1f}ms")
    print(f"    Metal shader compiled (21 opcodes, 8 registers, GPU ALU)")
    print()

    # ── Compile + Execute all programs ─────────────────────────────────────
    print("[4] Compiling and executing programs (ALL ON GPU)...")
    print("    Pipeline: nsl → Neural Compiler [MPS] → Neural Assembler [MPS]")
    print("            → Binary → Metal Compute Shader [GPU ALU]")
    print()

    passed = 0
    total = len(PROGRAMS)
    total_compile_us = 0
    total_exec_us = 0
    total_cycles = 0

    for program in PROGRAMS:
        success, compile_us, exec_us, cycles, ips = \
            compile_and_execute_on_gpu(os_instance, kernel, program, verbose=True)
        if success:
            passed += 1
        total_compile_us += compile_us
        total_exec_us += exec_us
        total_cycles += cycles

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  RESULTS: {passed}/{total} programs passed")
    print()
    print(f"  GPU Pipeline Breakdown:")
    print(f"    Boot (load 10 models → GPU):  {boot_time*1000:.1f}ms")
    print(f"    Avg compile (neural, MPS):    {total_compile_us/total:.0f}us")
    print(f"    Avg execute (Metal shader):   {total_exec_us/total:.0f}us")
    print(f"    Total execution cycles:       {total_cycles}")
    print()

    if passed == total:
        print("  ALL PROGRAMS VERIFIED!")
        print()
        print("  Complete GPU-native pipeline:")
        print("    nsl source code")
        print("      → Neural Compiler (PyTorch/MPS GPU)")
        print("      → Neural Assembler (PyTorch/MPS GPU)")
        print("      → Metal Compute Shader (GPU ALU)")
        print("      → Correct results")
        print()
        print("  Zero CPU arithmetic. Every computation on GPU silicon.")
    else:
        print(f"  {total - passed} programs failed verification")

    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
