#!/usr/bin/env python3
"""
Trace-Based Grep Regex Hang Debugger

Uses the GPU instruction trace buffer to capture the last 4096 instructions
when grep enters an infinite loop in musl's TRE regex engine. Identifies
the exact loop body, register patterns, and memory state to find the
root-cause ARM64 instruction bug.

This debugging technique is IMPOSSIBLE on conventional CPUs where register
state and instruction history are destroyed after process exit.
"""

import struct
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.gpu_cpu import GPUKernelCPU as create_gpu_cpu, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = PROJECT_ROOT / "demos" / "busybox.elf"


# ═══════════════════════════════════════════════════════════════════════════════
# ARM64 Instruction Decoder (minimal, for trace analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def decode_arm64(inst):
    """Minimal ARM64 instruction decoder for trace analysis."""
    top8 = (inst >> 24) & 0xFF
    top11 = (inst >> 21) & 0x7FF

    # Branches
    if (inst >> 26) == 0b000101:  # B
        offset = inst & 0x3FFFFFF
        if offset & 0x2000000:
            offset -= 0x4000000
        return f"B #{offset*4:+d}"
    if (inst >> 25) == 0b0101010:  # B.cond
        cond = inst & 0xF
        cond_names = ['EQ','NE','CS','CC','MI','PL','VS','VC',
                      'HI','LS','GE','LT','GT','LE','AL','NV']
        offset = ((inst >> 5) & 0x7FFFF)
        if offset & 0x40000:
            offset -= 0x80000
        return f"B.{cond_names[cond]} #{offset*4:+d}"
    if top8 in (0x34, 0x35, 0xB4, 0xB5):  # CBZ/CBNZ (32/64-bit)
        op = (inst >> 24) & 1
        rt = inst & 0x1F
        sf = (inst >> 31) & 1
        reg = f"X{rt}" if sf else f"W{rt}"
        offset = ((inst >> 5) & 0x7FFFF)
        if offset & 0x40000:
            offset -= 0x80000
        return f"{'CBNZ' if op else 'CBZ'} {reg}, #{offset*4:+d}"
    if top8 in (0x36, 0x37, 0xB6, 0xB7):  # TBZ/TBNZ (32/64-bit)
        op = (inst >> 24) & 1
        rt = inst & 0x1F
        bit = ((inst >> 31) << 5) | ((inst >> 19) & 0x1F)
        offset = ((inst >> 5) & 0x3FFF)
        if offset & 0x2000:
            offset -= 0x4000
        return f"{'TBNZ' if op else 'TBZ'} X{rt}, #{bit}, #{offset*4:+d}"
    if top8 == 0xD6 and ((inst >> 21) & 0x7) in (0, 1, 2):
        rn = (inst >> 5) & 0x1F
        ops = {0b00: 'BR', 0b01: 'BLR', 0b10: 'RET'}
        op = (inst >> 21) & 0x3
        return f"{ops.get(op, '???')} X{rn}"

    # BL
    if (inst >> 26) == 0b100101:
        offset = inst & 0x3FFFFFF
        if offset & 0x2000000:
            offset -= 0x4000000
        return f"BL #{offset*4:+d}"

    # Data processing - immediate
    if top8 in (0xD2, 0xF2):
        rd = inst & 0x1F
        imm16 = (inst >> 5) & 0xFFFF
        hw = (inst >> 21) & 0x3
        op = "MOVZ" if top8 == 0xD2 else "MOVK"
        shift = hw * 16
        return f"{op} X{rd}, #0x{imm16:X}{f', LSL #{shift}' if shift else ''}"
    if top8 == 0x52:
        rd = inst & 0x1F
        imm16 = (inst >> 5) & 0xFFFF
        return f"MOVZ W{rd}, #0x{imm16:X}"

    # ADD/SUB immediate
    if top8 in (0x91, 0xB1):
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm12 = (inst >> 10) & 0xFFF
        op = "ADD" if top8 == 0x91 else "ADDS"
        return f"{op} X{rd}, X{rn}, #{imm12}"
    if top8 in (0xD1, 0xF1):
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm12 = (inst >> 10) & 0xFFF
        op = "SUB" if top8 == 0xD1 else "SUBS"
        return f"{op} X{rd}, X{rn}, #{imm12}"

    # LDR/STR immediate unsigned offset
    if top8 in (0xF9, 0xB9):
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        is_load = (inst >> 22) & 1
        imm12 = (inst >> 10) & 0xFFF
        scale = 3 if top8 == 0xF9 else 2
        offset = imm12 << scale
        op = "LDR" if is_load else "STR"
        sz = "X" if top8 == 0xF9 else "W"
        return f"{op} {sz}{rt}, [X{rn}, #{offset}]"

    # LDUR/STUR
    if (inst & 0xFFE00C00) in (0xF8400000, 0xF8000000, 0xB8400000, 0xB8000000):
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm9 = (inst >> 12) & 0x1FF
        if imm9 & 0x100:
            imm9 -= 0x200
        is_load = (inst >> 22) & 1
        sf = (inst >> 30) & 1
        op = "LDUR" if is_load else "STUR"
        sz = "X" if sf else "W"
        return f"{op} {sz}{rt}, [X{rn}, #{imm9}]"

    # LDRB/STRB
    if top8 == 0x39:
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm12 = (inst >> 10) & 0xFFF
        is_load = (inst >> 22) & 1
        op = "LDRB" if is_load else "STRB"
        return f"{op} W{rt}, [X{rn}, #{imm12}]"

    # Logic register
    if top8 in (0x8A, 0xAA, 0xCA, 0xEA, 0x0A, 0x2A, 0x4A, 0x6A):
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        rm = (inst >> 16) & 0x1F
        sf = (inst >> 31) & 1
        opc = (inst >> 29) & 0x3
        n = (inst >> 21) & 1
        ops = {(0,0): 'AND', (0,1): 'BIC', (1,0): 'ORR', (1,1): 'ORN',
               (2,0): 'EOR', (2,1): 'EON', (3,0): 'ANDS', (3,1): 'BICS'}
        op = ops.get((opc, n), '???')
        sz = "X" if sf else "W"
        return f"{op} {sz}{rd}, {sz}{rn}, {sz}{rm}"

    # SVC
    if (inst & 0xFFE0001F) == 0xD4000001:
        imm16 = (inst >> 5) & 0xFFFF
        return f"SVC #0x{imm16:X}"

    # NOP
    if inst == 0xD503201F:
        return "NOP"
    # HLT
    if (inst & 0xFFE0001F) == 0xD4400000:
        return "HLT"

    # CMP (alias for SUBS XZR)
    if top8 == 0xEB:
        rn = (inst >> 5) & 0x1F
        rm = (inst >> 16) & 0x1F
        rd = inst & 0x1F
        if rd == 31:
            return f"CMP X{rn}, X{rm}"
        return f"SUBS X{rd}, X{rn}, X{rm}"

    # MADD/MSUB
    if top8 == 0x9B:
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        ra = (inst >> 10) & 0x1F
        rm = (inst >> 16) & 0x1F
        o0 = (inst >> 15) & 1
        op = "MSUB" if o0 else "MADD"
        if not o0 and ra == 31:
            return f"MUL X{rd}, X{rn}, X{rm}"
        return f"{op} X{rd}, X{rn}, X{rm}, X{ra}"

    # CSEL/CSINC/CSINV/CSNEG
    if top8 in (0x9A, 0xDA):
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        cond = (inst >> 12) & 0xF
        rm = (inst >> 16) & 0x1F
        op2 = (inst >> 10) & 0x3
        cond_names = ['EQ','NE','CS','CC','MI','PL','VS','VC',
                      'HI','LS','GE','LT','GT','LE','AL','NV']
        ops = {0: 'CSEL', 1: 'CSINC', 2: 'CSINV', 3: 'CSNEG'}
        return f"{ops[op2]} X{rd}, X{rn}, X{rm}, {cond_names[cond]}"

    return f"??? 0x{inst:08X}"


def analyze_trace(trace, cpu, verbose=True):
    """Analyze a trace buffer to identify loops, patterns, and the root cause."""
    if not trace:
        print("No trace entries captured.")
        return

    print(f"\n{'='*70}")
    print(f"  TRACE ANALYSIS: {len(trace)} entries captured")
    print(f"{'='*70}\n")

    # 1. PC frequency analysis
    pc_counts = Counter(entry[0] for entry in trace)
    print("Top 15 most-executed PCs:")
    for pc, count in pc_counts.most_common(15):
        pct = count / len(trace) * 100
        # Find the instruction at this PC
        insts_at_pc = [e[1] for e in trace if e[0] == pc]
        inst = insts_at_pc[0] if insts_at_pc else 0
        decoded = decode_arm64(inst)
        print(f"  0x{pc:08X}: {count:>5} ({pct:5.1f}%)  {decoded}")

    # 2. Loop detection
    print(f"\n{'─'*70}")
    print("Loop Detection:")
    last_pcs = [e[0] for e in trace[-200:]]
    unique_last = len(set(last_pcs))
    print(f"  Last 200 instructions span {unique_last} unique PCs")

    if unique_last < 20:
        print(f"  ** TIGHT LOOP DETECTED ({unique_last} instructions) **")

        # Find the loop body
        loop_pcs = sorted(set(last_pcs))
        print(f"\n  Loop body ({len(loop_pcs)} instructions):")
        for pc in loop_pcs:
            insts_at_pc = [e[1] for e in trace if e[0] == pc]
            inst = insts_at_pc[0]
            decoded = decode_arm64(inst)
            print(f"    0x{pc:08X}: {decoded}")

    # 3. Instruction type distribution
    print(f"\n{'─'*70}")
    print("Instruction Type Distribution:")
    type_counts = Counter()
    for pc, inst, *regs in trace:
        decoded = decode_arm64(inst)
        itype = decoded.split()[0] if decoded else "???"
        type_counts[itype] += 1
    for itype, count in type_counts.most_common(15):
        pct = count / len(trace) * 100
        print(f"  {itype:10s}: {count:>5} ({pct:5.1f}%)")

    # 4. Register value patterns in the last 100 entries
    print(f"\n{'─'*70}")
    print("Register State in Last 50 Entries:")
    print(f"  {'PC':>12s} {'Instruction':>12s} {'x0':>18s} {'x1':>18s} {'x2':>18s} {'x3':>18s}")
    for pc, inst, x0, x1, x2, x3, *_rest in trace[-50:]:
        decoded = decode_arm64(inst)
        print(f"  0x{pc:08X}  {decoded:30s}  x0=0x{x0 & 0xFFFFFFFFFFFFFFFF:016X} x1=0x{x1 & 0xFFFFFFFFFFFFFFFF:016X} x2=0x{x2 & 0xFFFFFFFFFFFFFFFF:016X} x3=0x{x3 & 0xFFFFFFFFFFFFFFFF:016X}")

    # 5. Memory region analysis — what addresses are being accessed?
    print(f"\n{'─'*70}")
    print("Memory Access Analysis (from register values):")
    addr_ranges = Counter()
    for pc, inst, x0, x1, x2, x3, *_rest in trace:
        for reg_name, val in [('x0', x0), ('x1', x1), ('x2', x2), ('x3', x3)]:
            uval = val & 0xFFFFFFFFFFFFFFFF
            if 0x10000 <= uval <= 0x1000000:  # Plausible address range
                region = uval & 0xFFFF0000
                addr_ranges[f"{reg_name}→0x{region:08X}"] += 1
    if addr_ranges:
        print("  Register→Memory region frequency:")
        for key, count in addr_ranges.most_common(10):
            print(f"    {key}: {count} times")

    # 6. Check for reads from mmap'd regions
    print(f"\n{'─'*70}")
    print("Key Register Values at Final State:")
    for i in range(16):
        val = cpu.get_register(i)
        uval = val & 0xFFFFFFFFFFFFFFFF
        print(f"  X{i:2d} = 0x{uval:016X}  ({val})")

    # 7. Memory dump around hot addresses
    print(f"\n{'─'*70}")
    print("Memory Dump Around Hot Addresses:")
    hot_addrs = set()
    for pc, inst, x0, x1, x2, x3, *_rest in trace[-20:]:
        for val in [x0, x1, x2, x3]:
            uval = val & 0xFFFFFFFFFFFFFFFF
            if 0x10000 <= uval <= 0x800000:
                hot_addrs.add(uval & ~0xF)  # Align to 16

    for addr in sorted(hot_addrs)[:5]:  # Top 5 hot regions
        try:
            data = cpu.read_memory(addr, 64)
            print(f"\n  0x{addr:08X}:")
            for off in range(0, 64, 16):
                hex_str = ' '.join(f'{b:02X}' for b in data[off:off+16])
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[off:off+16])
                print(f"    +{off:02X}: {hex_str}  |{ascii_str}|")
        except Exception:
            pass

    return pc_counts, loop_pcs if unique_last < 20 else []


def main():
    print("=" * 70)
    print("  GREP REGEX HANG — TRACE-BASED ROOT CAUSE ANALYSIS")
    print("  Using GPU instruction trace buffer (impossible on CPU)")
    print("=" * 70)
    print()

    # Setup filesystem
    fs = GPUFilesystem()
    fs.write_file('/etc/passwd',
        "root:x:0:0:root:/root:/bin/sh\n"
        "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
        "hello:x:1000:1000:hello:/home/hello:/bin/sh\n"
    )

    # Create CPU
    cpu = create_gpu_cpu(quiet=True)

    # Load BusyBox with regex grep (NOT -F)
    argv = ['grep', 'hello', '/etc/passwd']
    print(f"Command: {' '.join(argv)}")
    print(f"Mode: REGEX (not fixed-string)")
    print()

    entry = load_elf_into_memory(cpu, str(BUSYBOX), argv=argv, quiet=True)
    elf_data = BUSYBOX.read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    cpu.set_pc(entry)
    cpu.init_svc_buffer()

    # Enable tracing
    cpu.enable_trace()
    print("Instruction tracing ENABLED")

    # Create syscall handler with mmap tracking
    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)
    all_mmaps = []

    # Execute with cycle limit — enough to enter the hang
    total_cycles = 0
    max_cycles = 200_000  # 200K cycles should be enough to see the loop
    batch_size = 10_000
    output_parts = []
    syscall_count = 0

    print(f"Executing up to {max_cycles:,} cycles...")
    print()

    while total_cycles < max_cycles:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        # Drain output
        for fd, data in cpu.drain_svc_buffer():
            output_parts.append(data.decode('ascii', errors='replace'))

        if result.stop_reason == StopReasonV2.HALT:
            print(f"Program HALTED at {total_cycles:,} cycles")
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            syscall_count += 1
            # Track mmap calls
            sysno = cpu.get_register(8)
            if sysno == 222:  # SYS_MMAP
                x0 = cpu.get_register(0)
                x1 = cpu.get_register(1)
                all_mmaps.append({
                    'addr': x0 & 0xFFFFFFFFFFFFFFFF,
                    'length': x1 & 0xFFFFFFFF,
                    'cycle': total_cycles,
                })

            ret = handler(cpu)
            if ret == False:
                print(f"Handler returned False at {total_cycles:,} cycles")
                break
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)

    program_output = ''.join(output_parts)
    halted = result.stop_reason == StopReasonV2.HALT if result else False

    print(f"Total cycles:  {total_cycles:,}")
    print(f"Total syscalls: {syscall_count}")
    print(f"Halted:        {'YES' if halted else 'NO (cycle limit or hang)'}")
    if program_output.strip():
        print(f"Output:        {program_output.strip()[:200]}")

    # Show mmap regions
    if all_mmaps:
        print(f"\nMemory mappings ({len(all_mmaps)} mmap calls):")
        for m in all_mmaps:
            print(f"  0x{m['addr']:08X} - 0x{m['addr'] + m['length']:08X} "
                  f"({m['length']:,} bytes) at cycle {m['cycle']:,}")

    # Read and analyze trace
    trace = cpu.read_trace()
    analyze_trace(trace, cpu)

    # Disable tracing
    cpu.disable_trace()


if __name__ == "__main__":
    main()
