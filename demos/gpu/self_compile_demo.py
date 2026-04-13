#!/usr/bin/env python3
"""Self-Compilation Demo: neurOS compiles and executes on nCPU.

Demonstrates the complete neural stack:
    1. neurOS boots (GPU-native neural OS)
    2. nsl source code is written
    3. Neural compiler: nsl → IR → optimized assembly
    4. Neural assembler: assembly → 32-bit binary
    5. Assembly loaded into nCPU (model CPU with neural ALU)
    6. nCPU executes with trained neural networks for every ALU op
    7. Results verified against expected values

This is the holy grail: a neural OS compiling a program that runs
on a neural CPU where every operation is a trained neural network.
"""

import sys
import time
import logging

sys.path.insert(0, ".")
from ncpu.os import NeurOS
from ncpu.model import CPU

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


def print_header():
    print()
    print("=" * 70)
    print("  nCPU Self-Compilation Demo")
    print("  Neural OS → Neural Compiler → Neural CPU")
    print("=" * 70)
    print()


def demo_compile_and_execute(os_instance, program, cpu, verbose=True):
    """Compile an nsl program with neurOS and execute on nCPU.

    Returns:
        (success: bool, compile_time_us: float, exec_cycles: int)
    """
    name = program["name"]
    source = program["source"]

    if verbose:
        print(f"  [{name}]")
        print(f"    Source:   {program['description']}")

    # Stage 1: Compile with neurOS compiler
    t0 = time.perf_counter()
    result = os_instance.compiler.compile(source)
    compile_time = time.perf_counter() - t0

    if not result.success:
        if verbose:
            print(f"    COMPILE FAILED: {result.errors}")
        return False, compile_time * 1e6, 0

    if verbose:
        print(f"    Compile:  {len(result.ir)} IR → "
              f"{result.assembly_result.num_instructions} asm "
              f"({result.optimizations_applied} optimizations) "
              f"[{compile_time*1e6:.0f}us]")

    # Stage 2: Load assembly into nCPU
    # The compiler produces nCPU assembly text which the model CPU understands
    assembly_text = result.assembly
    # Strip the compiler comment line
    asm_lines = [l for l in assembly_text.split("\n")
                 if l.strip() and not l.strip().startswith(";")]
    clean_asm = "\n".join(asm_lines)

    cpu.load_program(clean_asm)

    # Stage 3: Execute on neural CPU
    t0 = time.perf_counter()
    try:
        cpu.run(max_cycles=5000)
    except RuntimeError as e:
        if "Max cycles" not in str(e):
            if verbose:
                print(f"    EXEC ERROR: {e}")
            return False, compile_time * 1e6, 0
    exec_time = time.perf_counter() - t0
    cycles = cpu.get_cycle_count()

    # Stage 4: Verify results
    regs = cpu.dump_registers()
    non_zero = {k: v for k, v in regs.items() if v != 0}
    success = program["check"](regs)

    if verbose:
        status = "PASS" if success else "FAIL"
        print(f"    Execute:  {cycles} cycles [{exec_time*1e6:.0f}us]")
        print(f"    Result:   {non_zero}")
        print(f"    Status:   {status} (expected {program['expected']})")
        print()

    return success, compile_time * 1e6, cycles


def main():
    print_header()

    # Boot neurOS
    print("[1] Booting neurOS...")
    t0 = time.perf_counter()
    os_instance = NeurOS()
    os_instance.boot(quiet=True)
    boot_time = time.perf_counter() - t0
    print(f"    neurOS booted in {boot_time*1000:.1f}ms")
    print()

    # Create nCPU with neural ALU
    print("[2] Initializing nCPU with neural ALU...")
    t0 = time.perf_counter()
    cpu = CPU(
        mock_mode=True,
        neural_execution=True,
        models_dir="models",
        max_cycles=5000,
    )
    init_time = time.perf_counter() - t0
    print(f"    nCPU ready in {init_time*1000:.1f}ms (neural ALU loaded)")
    print()

    # Run all programs
    print("[3] Compiling and executing programs...")
    print("    neurOS compiler → nCPU assembly → neural ALU execution")
    print()

    passed = 0
    total = len(PROGRAMS)
    total_compile_us = 0
    total_cycles = 0

    for program in PROGRAMS:
        success, compile_us, cycles = demo_compile_and_execute(
            os_instance, program, cpu, verbose=True)
        if success:
            passed += 1
        total_compile_us += compile_us
        total_cycles += cycles

    # Summary
    print("=" * 70)
    print(f"  RESULTS: {passed}/{total} programs passed")
    print(f"  Average compile time: {total_compile_us/total:.0f}us")
    print(f"  Total execution cycles: {total_cycles}")
    print(f"  Boot time: {boot_time*1000:.1f}ms")
    print()

    if passed == total:
        print("  ALL PROGRAMS VERIFIED!")
        print("  Complete neural stack: nsl → compiler → assembler → neural CPU")
    else:
        print(f"  {total - passed} programs failed verification")

    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
