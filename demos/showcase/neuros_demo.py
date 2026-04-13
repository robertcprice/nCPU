#!/usr/bin/env python3
"""neurOS Interactive Demo.

Demonstrates the full neural operating system:
    1. Boot sequence
    2. Filesystem operations
    3. Process management
    4. Assembler: compile assembly → binary
    5. Compiler: compile nsl → assembly → binary
    6. End-to-end: write source file, compile, assemble, run
"""

import sys
import time
import logging
import torch

sys.path.insert(0, ".")
from ncpu.os import NeurOS

logging.basicConfig(level=logging.INFO, format="%(message)s")


def demo_boot():
    """Boot neurOS and show system status."""
    print("=" * 60)
    print("  neurOS Demo - GPU-Native Neural Operating System")
    print("=" * 60)
    print()

    os = NeurOS()
    stages = os.boot(quiet=False)
    print()
    return os


def demo_filesystem(os):
    """Demonstrate filesystem operations."""
    print("--- Filesystem Demo ---")
    output = os.shell.execute("ls /")
    for line in output:
        print(line)

    os.shell.execute("mkdir /home/demo")
    os.shell.execute("echo hello > /home/demo/test.txt")
    print()

    output = os.shell.execute("cat /etc/motd")
    for line in output:
        print(line)

    output = os.shell.execute("df")
    for line in output:
        print(line)
    print()


def demo_processes(os):
    """Demonstrate process management."""
    print("--- Process Management Demo ---")
    output = os.shell.execute("ps")
    for line in output:
        print(line)
    print()

    output = os.shell.execute("free")
    for line in output:
        print(line)
    print()


def demo_assembler(os):
    """Demonstrate the neural assembler."""
    print("--- Neural Assembler Demo ---")

    source = """\
    MOV R0, 0
    MOV R1, 1
    MOV R2, 10
    MOV R3, 0
    MOV R4, 1
loop:
    MOV R5, R1
    ADD R1, R0, R1
    MOV R0, R5
    ADD R3, R3, R4
    CMP R3, R2
    JNZ loop
    HALT
"""
    result = os.assembler.assemble(source)
    print(f"  Fibonacci program: {result.num_instructions} instructions assembled")
    print(f"  Labels: {result.labels}")
    print(f"  Binary:")
    for i, word in enumerate(result.binary):
        disasm = os.assembler.classical._format_instruction(
            os.assembler.classical.decode_word(word), i)
        print(f"    {i:4d}: 0x{word:08X}  {disasm}")
    print()

    # Disassemble
    print("  Disassembly:")
    print(os.assembler.disassemble(result.binary))
    print()


def demo_compiler(os):
    """Demonstrate the neural compiler."""
    print("--- Neural Compiler Demo ---")

    # Simple arithmetic
    source1 = "var x = 10; var y = 20; var sum = x + y; halt;"
    result1 = os.compiler.compile(source1)
    print(f"  Program 1: 'var x = 10; var y = 20; var sum = x + y; halt;'")
    print(f"    Variables: {result1.variables}")
    print(f"    IR instructions: {len(result1.ir)}")
    print(f"    Assembly instructions: {result1.assembly_result.num_instructions}")
    print(f"    Optimizations: {result1.optimizations_applied}")
    print(f"    Assembly:")
    for line in result1.assembly.split("\n"):
        print(f"      {line}")
    print()

    # Loop with conditional
    source2 = """\
var sum = 0;
var i = 1;
var limit = 11;
while (i != limit) {
    sum = sum + i;
    i = i + 1;
}
halt;
"""
    result2 = os.compiler.compile(source2)
    print(f"  Program 2: Sum 1 to 10")
    print(f"    Variables: {result2.variables}")
    print(f"    IR instructions: {len(result2.ir)}")
    print(f"    Assembly instructions: {result2.assembly_result.num_instructions}")
    print(f"    Assembly:")
    for line in result2.assembly.split("\n"):
        print(f"      {line}")
    print()

    # Fibonacci
    source3 = """\
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
"""
    result3 = os.compiler.compile(source3)
    print(f"  Program 3: Fibonacci")
    print(f"    Variables: {result3.variables}")
    print(f"    Assembly instructions: {result3.assembly_result.num_instructions}")
    print()


def demo_training(os):
    """Train the neural assembler on existing programs."""
    print("--- Neural Assembler Training Demo ---")

    # Read programs
    import os as stdlib_os
    programs = []
    prog_dir = "programs"
    if stdlib_os.path.exists(prog_dir):
        for fname in stdlib_os.listdir(prog_dir):
            if fname.endswith(".asm"):
                with open(f"{prog_dir}/{fname}") as f:
                    programs.append(f.read())

    if programs:
        stats = os.assembler.train_codegen(programs, epochs=200)
        print(f"  Training data: {stats['num_programs']} programs, "
              f"{stats['num_instructions']} instructions")
        print(f"  Bit accuracy: {stats['bit_accuracy']:.1%}")
        print(f"  Exact match: {stats['exact_match_rate']:.1%} "
              f"({stats['exact_matches']}/{stats['num_instructions']})")
    print()


def demo_system_status(os):
    """Show complete system status."""
    print("--- System Status ---")
    output = os.shell.execute("top")
    for line in output:
        print(line)
    print()
    output = os.shell.execute("uname")
    for line in output:
        print(line)
    print()


def main():
    os = demo_boot()
    demo_filesystem(os)
    demo_processes(os)
    demo_assembler(os)
    demo_compiler(os)
    demo_training(os)
    demo_system_status(os)

    print("=" * 60)
    status = os.status()
    print(f"  Boot time: {status['boot_time_ms']:.1f}ms")
    print(f"  Device: {status['device']}")
    print(f"  Components: {sum(1 for v in status.values() if v is not None) - 3}")
    print("=" * 60)


if __name__ == "__main__":
    main()
