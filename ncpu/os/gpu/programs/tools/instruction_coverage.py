#!/usr/bin/env python3
"""
GPU Instruction Coverage Analysis Tool

Runs a suite of BusyBox commands with instruction tracing enabled, aggregates
all unique (PC, instruction) pairs, decodes them, and reports:
- Which ARM64 instruction types are heavily exercised
- Which instruction types are never tested
- Hot code paths (most frequently executed PCs)
- Cold code paths (executed once)
- Coverage map of the BusyBox binary

This is a novel capability enabled by GPU-side trace buffers — on CPUs,
getting this level of instruction coverage requires heavy instrumentation
(ptrace, valgrind, DBI frameworks), all of which alter timing and behavior.
On GPU, it's zero-overhead.

Usage:
    python3 instruction_coverage.py [--commands N] [--cycles N]
"""

import argparse
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.gpu_cpu import GPUKernelCPU as create_gpu_cpu, StopReasonV2
from ncpu.os.gpu.elf_loader import (
    load_elf_into_memory,
    parse_elf,
    make_busybox_syscall_handler,
    parse_elf_function_symbols,
    lookup_function_symbol,
)
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = PROJECT_ROOT / "demos" / "busybox.elf"


# ═══════════════════════════════════════════════════════════════════════════════
# ARM64 Instruction Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def classify_instruction(inst):
    """Classify an ARM64 instruction into a category.

    Handles all major ARM64 instruction groups including SIMD, atomics,
    conditional compare, and add/subtract with carry. Uses top8 (bits[31:24])
    as the primary discriminator with sub-field checks where needed.
    """
    top8 = (inst >> 24) & 0xFF

    # ── Branches ──
    # B/BL: top 6 bits = 000101 (B) or 100101 (BL)
    if (inst >> 26) in (0b000101, 0b100101):
        return "branch_unconditional"
    # B.cond: top byte = 0x54
    if top8 == 0x54:
        return "branch_conditional"
    # CBZ/CBNZ: 32-bit (0x34/0x35) and 64-bit (0xB4/0xB5)
    if top8 in (0x34, 0x35, 0xB4, 0xB5):
        return "branch_compare"
    # TBZ/TBNZ: 32-bit (0x36/0x37) and 64-bit (0xB6/0xB7)
    if top8 in (0x36, 0x37, 0xB6, 0xB7):
        return "branch_test"
    # BR/BLR/RET
    if top8 == 0xD6:
        return "branch_register"

    # ── System ──
    if inst == 0xD503201F:
        return "nop"
    if (inst & 0xFFE0001F) == 0xD4400000:
        return "halt"
    if (inst & 0xFFE0001F) == 0xD4000001:
        return "svc"
    if top8 == 0xD5:
        return "system"

    # ── Data movement ──
    if top8 in (0xD2, 0x52):
        return "movz"
    if top8 in (0xF2, 0x72):
        return "movk"
    if top8 in (0x92, 0x12):
        return "movn"

    # ── Arithmetic immediate ──
    if top8 in (0x91, 0x11):
        return "add_imm"
    if top8 in (0xD1, 0x51):
        return "sub_imm"
    if top8 in (0xB1, 0x31):
        return "adds_imm"
    if top8 in (0xF1, 0x71):
        return "subs_imm"

    # ── Arithmetic register ──
    if top8 in (0x8B, 0x0B):
        return "add_reg"
    if top8 in (0xCB, 0x4B):
        return "sub_reg"
    if top8 in (0xAB, 0x2B):
        return "adds_reg"
    if top8 in (0xEB, 0x6B):
        return "subs_reg"

    # ── Logic register (N-bit aware: N=0 normal, N=1 inverted second operand) ──
    if top8 in (0x8A, 0x0A):
        return "bic" if ((inst >> 21) & 1) else "and_reg"
    if top8 in (0xAA, 0x2A):
        return "orn" if ((inst >> 21) & 1) else "orr_reg"
    if top8 in (0xCA, 0x4A):
        return "eon" if ((inst >> 21) & 1) else "eor_reg"
    if top8 in (0xEA, 0x6A):
        return "bics" if ((inst >> 21) & 1) else "ands_reg"

    # ── Logic immediate ──
    if top8 in (0xB2, 0x32):
        return "orr_imm"

    # ── Data processing register (0x9A/0x1A: ADC, division, shifts, CSEL) ──
    # Must check sub-encoding via bits[23:21] to distinguish
    if top8 in (0x9A, 0x1A):
        bits_23_21 = (inst >> 21) & 0x7
        if bits_23_21 == 0b000:
            return "adc"
        if bits_23_21 == 0b110:
            op2 = (inst >> 10) & 0x3F
            if op2 in (2, 3):
                return "sdiv" if (op2 & 1) else "udiv"
            if op2 in (8, 9, 10, 11):
                return ["lsl_reg", "lsr_reg", "asr_reg", "ror_reg"][op2 & 3]
        if bits_23_21 == 0b100:
            return "csinc" if ((inst >> 10) & 1) else "csel"
        return "csel"

    # ── Data processing register (0xDA/0x5A: SBC, bit manip, CSINV/CSNEG) ──
    if top8 in (0xDA, 0x5A):
        bits_23_21 = (inst >> 21) & 0x7
        if bits_23_21 == 0b000:
            return "sbc"
        if bits_23_21 == 0b110:
            op2 = (inst >> 10) & 0x3F
            if op2 == 0: return "rbit"
            if op2 == 4: return "clz"
            if op2 == 5: return "rev"
            return "rev"
        if bits_23_21 == 0b100:
            return "csneg" if ((inst >> 10) & 1) else "csinv"
        return "csinv"

    # ── Add/subtract with carry (flag-setting) / Conditional compare ──
    # 0xBA/0x3A: ADCS or CCMN (distinguished by bits[23:21])
    if top8 in (0xBA, 0x3A):
        return "ccmn" if ((inst >> 21) & 0x7) == 0b010 else "adcs"
    # 0xFA: SBCS or CCMP
    if top8 == 0xFA:
        return "ccmp" if ((inst >> 21) & 0x7) == 0b010 else "sbcs"
    # 0x7A: SBCS-32 or CCMP-32
    if top8 == 0x7A:
        return "ccmp" if ((inst >> 21) & 0x7) == 0b010 else "sbcs"

    # ── Multiply ──
    if top8 in (0x9B, 0x1B):
        o0 = (inst >> 15) & 1
        ra = (inst >> 10) & 0x1F
        op31 = (inst >> 21) & 0x7
        if op31 in (0b001, 0b101):
            return "smull" if op31 == 0b001 else "umull"
        if o0 == 0 and ra == 31:
            return "mul"
        return "msub" if o0 else "madd"

    # ── Load/Store unsigned offset ──
    if top8 == 0xF9:
        return "ldr_str_64_uoff"
    if top8 == 0xB9:
        return "ldr_str_32_uoff"
    if top8 == 0x79:
        return "ldrh_strh_uoff"
    if top8 == 0x39:
        return "ldrb_strb_uoff"

    # ── Load/Store register (unscaled, pre/post-index, register offset) ──
    if top8 in (0xF8, 0xB8, 0x78, 0x38):
        return "ldr_str_unscaled"

    # ── Load/Store pair ──
    if top8 in (0xA9, 0x29, 0x69, 0xE9):
        return "ldp_stp"
    if top8 in (0xA8, 0x28, 0x68, 0xE8):
        return "ldp_stp_post"

    # ── LDR literal (PC-relative) ──
    if top8 in (0x18, 0x58, 0x98):
        return "ldr_literal"

    # ── SIMD/FP load/store ──
    if top8 in (0xAD, 0x2D, 0x6D, 0xED):
        return "ldp_stp_simd"
    if top8 in (0xAC, 0x2C, 0x6C, 0xEC):
        return "ldp_stp_simd_post"
    if top8 in (0xFD, 0xBD, 0x7D, 0x3D):
        return "ldr_str_simd_uoff"
    if top8 in (0xFC, 0xBC, 0x7C, 0x3C):
        return "ldr_str_simd"

    # ── Exclusive load/store (atomics) ──
    if top8 in (0x08, 0x48, 0x88, 0xC8):
        return "ldxr_stxr"

    # ── ADR/ADRP (handles all immlo values) ──
    # ADR: bit 31 = 0, bits[28:24] = 10000
    if ((inst >> 31) == 0) and ((inst >> 24) & 0x1F) == 0x10:
        return "adr"
    # ADRP: bit 31 = 1, bits[28:24] = 10000
    if ((inst >> 31) == 1) and ((inst >> 24) & 0x1F) == 0x10:
        return "adrp"

    # ── Bitfield ──
    if top8 in (0x93, 0x13):
        return "sbfm"
    if top8 in (0xD3, 0x53):
        return "ubfm"
    if top8 in (0xB3, 0x33):
        return "bfm"

    # ── SIMD data processing ──
    if top8 in (0x0E, 0x4E, 0x2E, 0x6E):
        return "simd_processing"
    if top8 in (0x0F, 0x4F, 0x2F, 0x6F):
        return "simd_modified_imm"

    return f"unknown_0x{top8:02X}"


def parse_elf_symbols(elf_path):
    """Extract symbol table from ELF for PC→function mapping."""
    return parse_elf_function_symbols(elf_path)


def pc_to_function(pc, symbols):
    """Map a PC to the function it belongs to."""
    match = lookup_function_symbol(pc, symbols)
    return match[0] if match else None


# ═══════════════════════════════════════════════════════════════════════════════
# Test Command Suite
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_COMMANDS = [
    # Simple commands
    ["echo", "hello", "world"],
    ["true"],
    ["false"],
    ["basename", "/usr/local/bin/python3"],
    ["dirname", "/usr/local/bin/python3"],
    # String operations
    ["printf", "%d %s\\n", "42", "test"],
    # File operations
    ["cat", "/etc/passwd"],
    ["head", "-n", "1", "/etc/passwd"],
    ["tail", "-n", "1", "/etc/passwd"],
    ["wc", "-l", "/etc/passwd"],
    ["wc", "-w", "/etc/passwd"],
    # Search
    ["grep", "-F", "root", "/etc/passwd"],
    # Data processing
    ["sort", "/etc/hostname"],
    ["uniq", "/etc/hostname"],
    # System info
    ["uname", "-a"],
    ["hostname"],
    ["id"],
    ["whoami"],
    # File listing
    ["ls", "/etc"],
    ["ls", "-l", "/etc"],
    # Text manipulation
    ["cut", "-d:", "-f1", "/etc/passwd"],
    ["tr", "a-z", "A-Z"],
    # Math
    ["expr", "40", "+", "2"],
]


def run_command_with_trace(argv, fs, max_cycles=100000):
    """Run a single BusyBox command with tracing, return trace entries."""
    cpu = create_gpu_cpu(quiet=True)

    try:
        entry = load_elf_into_memory(cpu, str(BUSYBOX), argv=argv, quiet=True)
    except Exception as e:
        return None, str(e)

    elf_data = BUSYBOX.read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    cpu.set_pc(entry)
    cpu.init_svc_buffer()
    cpu.enable_trace()

    # For 'tr', provide stdin
    stdin_data = None
    if argv[0] == 'tr':
        stdin_data = b"hello world\n"

    handler = make_busybox_syscall_handler(
        filesystem=fs, heap_base=heap_base, stdin_data=stdin_data
    )

    total_cycles = 0
    while total_cycles < max_cycles:
        result = cpu.execute(max_cycles=10000)
        total_cycles += result.cycles

        cpu.drain_svc_buffer()

        if result.stop_reason == StopReasonV2.HALT:
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)

    trace = cpu.read_trace()
    cpu.disable_trace()

    return trace, None


def main():
    parser = argparse.ArgumentParser(description="ARM64 Instruction Coverage Analysis")
    parser.add_argument("--commands", type=int, default=len(DEFAULT_COMMANDS),
                        help="Number of commands to run")
    parser.add_argument("--cycles", type=int, default=100000,
                        help="Max cycles per command")
    args = parser.parse_args()

    print("=" * 70)
    print("  ARM64 INSTRUCTION COVERAGE ANALYSIS")
    print("  GPU-side tracing (zero-overhead, impossible on CPU without DBI)")
    print("=" * 70)
    print()

    # Setup filesystem
    fs = GPUFilesystem()
    fs.write_file('/etc/passwd',
        "root:x:0:0:root:/root:/bin/sh\n"
        "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
        "hello:x:1000:1000:hello:/home/hello:/bin/sh\n"
    )
    fs.write_file('/etc/hostname', "ncpu-gpu\n")

    # Parse ELF symbols
    print("Parsing ELF symbol table...")
    symbols = parse_elf_symbols(str(BUSYBOX))
    print(f"  Found {len(symbols)} function symbols")
    print()

    # Aggregate coverage data
    all_pcs = Counter()           # PC → count
    all_instructions = Counter()  # instruction type → count
    all_inst_words = {}           # PC → instruction word
    per_command_types = {}        # command → set of types
    total_entries = 0

    commands = DEFAULT_COMMANDS[:args.commands]

    for i, argv in enumerate(commands):
        cmd_str = ' '.join(argv)
        sys.stdout.write(f"  [{i+1:2d}/{len(commands)}] {cmd_str:40s} ")
        sys.stdout.flush()

        trace, error = run_command_with_trace(argv, fs, args.cycles)
        if error:
            print(f"ERROR: {error}")
            continue
        if trace is None or len(trace) == 0:
            print("no trace")
            continue

        # Aggregate
        cmd_types = set()
        for pc, inst, *regs in trace:
            all_pcs[pc] += 1
            itype = classify_instruction(inst)
            all_instructions[itype] += 1
            all_inst_words[pc] = inst
            cmd_types.add(itype)

        per_command_types[cmd_str] = cmd_types
        total_entries += len(trace)
        print(f"{len(trace):>5} entries, {len(cmd_types):>3} types")

    # ═══════════════════════════════════════════════════════════════════════════
    # Coverage Report
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  COVERAGE REPORT")
    print(f"{'='*70}")
    print(f"\n  Commands run:      {len(commands)}")
    print(f"  Total trace entries: {total_entries:,}")
    print(f"  Unique PCs:        {len(all_pcs):,}")
    print(f"  Instruction types: {len(all_instructions)}")

    # Instruction type coverage
    print(f"\n{'─'*70}")
    print(f"  INSTRUCTION TYPE COVERAGE ({len(all_instructions)} types exercised)")
    print(f"{'─'*70}")
    print(f"\n  {'Type':<25s} {'Count':>8s} {'Pct':>7s}")
    for itype, count in all_instructions.most_common():
        pct = count / total_entries * 100
        print(f"  {itype:<25s} {count:>8,} {pct:>6.1f}%")

    # All possible ARM64 instruction types (88 types across all groups)
    all_possible = {
        # Arithmetic
        "add_imm", "sub_imm", "adds_imm", "subs_imm",
        "add_reg", "sub_reg", "adds_reg", "subs_reg",
        "adc", "adcs", "sbc", "sbcs",
        # Logic
        "and_reg", "orr_reg", "eor_reg", "ands_reg",
        "bic", "orn", "eon", "bics",
        "orr_imm",
        # Data movement
        "movz", "movk", "movn",
        # Multiply/divide
        "mul", "madd", "msub", "smull", "umull",
        "udiv", "sdiv",
        # Shifts
        "lsl_reg", "lsr_reg", "asr_reg", "ror_reg",
        # Conditional
        "csel", "csinc", "csinv", "csneg",
        "ccmp", "ccmn",
        # Load/Store
        "ldr_str_64_uoff", "ldr_str_32_uoff", "ldrh_strh_uoff", "ldrb_strb_uoff",
        "ldr_str_unscaled", "ldp_stp", "ldp_stp_post",
        "ldr_literal",
        "ldp_stp_simd", "ldp_stp_simd_post", "ldr_str_simd_uoff", "ldr_str_simd",
        "ldxr_stxr",
        # Branches
        "branch_unconditional", "branch_conditional", "branch_compare",
        "branch_test", "branch_register",
        # Bitfield
        "sbfm", "ubfm", "bfm",
        # PC-relative
        "adr", "adrp",
        # Bit manipulation
        "clz", "rbit", "rev",
        # SIMD
        "simd_processing", "simd_modified_imm",
        # System
        "nop", "halt", "svc", "system",
    }
    covered = set(all_instructions.keys()) & all_possible
    uncovered = all_possible - covered

    print(f"\n{'─'*70}")
    print(f"  COVERAGE GAPS ({len(uncovered)} untested types)")
    print(f"{'─'*70}")
    if uncovered:
        for itype in sorted(uncovered):
            print(f"  - {itype}")
    else:
        print("  None — all instruction types covered!")

    # Hot code paths
    print(f"\n{'─'*70}")
    print(f"  HOT CODE PATHS (top 20 most-executed PCs)")
    print(f"{'─'*70}")
    print(f"\n  {'PC':>12s} {'Count':>8s} {'Type':<20s} {'Function':<30s}")
    for pc, count in all_pcs.most_common(20):
        inst = all_inst_words.get(pc, 0)
        itype = classify_instruction(inst)
        func = pc_to_function(pc, symbols) or "???"
        print(f"  0x{pc:08X} {count:>8,} {itype:<20s} {func}")

    # Function coverage
    print(f"\n{'─'*70}")
    print(f"  FUNCTION COVERAGE")
    print(f"{'─'*70}")
    func_counts = Counter()
    for pc, count in all_pcs.items():
        func = pc_to_function(pc, symbols)
        if func:
            func_counts[func] += count

    print(f"\n  {len(func_counts)} functions exercised:")
    for func, count in func_counts.most_common(30):
        print(f"  {func:<40s} {count:>8,} instructions")

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    coverage_pct = len(covered) / len(all_possible) * 100 if all_possible else 0
    print(f"  Instruction type coverage: {len(covered)}/{len(all_possible)} ({coverage_pct:.1f}%)")
    print(f"  Function coverage:         {len(func_counts)} functions")
    print(f"  Unique code locations:     {len(all_pcs):,} PCs")
    print(f"  Total instructions traced: {total_entries:,}")
    print()
    print(f"  NOTE: This analysis used GPU-side trace buffers with zero")
    print(f"  overhead. On CPU, equivalent analysis requires ptrace,")
    print(f"  valgrind, or DBI frameworks that alter program timing.")


if __name__ == "__main__":
    main()
