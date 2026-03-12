#!/usr/bin/env python3
"""Trace the PARSING phase of regcomp to find where wchar_t values are lost.

Previous diagnostics captured only the NFA building phase (last 4096 of ~12K cycles).
This script captures the FIRST ~2000 cycles of regcomp, which includes:
- mbsrtowcs (byte->wchar_t conversion)
- early tre_compile/tre_parse setup

Static disassembly corrected the earlier call chain:
- 0x0041A374 is grep's regex compile call site
- 0x0041E4AC is BusyBox's "compile + report error" wrapper
- 0x00428C4C is the real musl regcomp entry (x0=regex_t*, x1=pattern, w2=cflags)
"""
import sys, struct
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU
from kernels.mlx.rust_runner import StopReasonV2
from pathlib import Path

ELF_PATH = "demos/busybox.elf"
GREP_COMPILE_CALL = 0x0041A374
REGCOMP_WRAPPER = 0x0041E4AC
REGCOMP_ENTRY = 0x00428C4C


def setup():
    cpu = GPUKernelCPU(quiet=True)
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n")
    entry = load_elf_into_memory(cpu, ELF_PATH, argv=["grep", "root", "/etc/passwd"], quiet=True)
    cpu.set_pc(entry)
    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF
    cpu.write_memory(0x3F0008, struct.pack('<Q', heap_base))
    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)
    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()
    return cpu, handler


def run_to_bp(cpu, handler, max_cycles=100_000):
    total = 0
    while total < max_cycles:
        result = cpu.execute(max_cycles=1000)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BP"
        elif result.stop_reason == StopReasonV2.WATCHPOINT:
            return total, "WP"
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return total, "EXIT"
            elif ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            return total, "HALT"
    return total, "MAX"


def run_batches(cpu, handler, target, batch=500):
    total = 0
    while total < target:
        result = cpu.execute(max_cycles=batch)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False: return total, "EXIT"
            elif ret != "exec": cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            return total, "HALT"
        elif result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BP"
    return total, "OK"


def decode(inst, pc):
    """Basic ARM64 instruction decode."""
    op = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rm = (inst >> 16) & 0x1F
    imm12 = (inst >> 10) & 0xFFF

    if (inst & 0xFC000000) == 0x94000000:  # BL
        imm26 = inst & 0x3FFFFFF
        if imm26 >= (1 << 25): imm26 -= (1 << 26)
        return f"BL 0x{(pc + imm26 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFC000000) == 0x14000000:  # B
        imm26 = inst & 0x3FFFFFF
        if imm26 >= (1 << 25): imm26 -= (1 << 26)
        return f"B 0x{(pc + imm26 * 4) & 0xFFFFFFFF:08X}"
    if inst == 0xD65F03C0: return "RET"
    if (inst & 0xFFC00000) == 0xA9800000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"STP X{rd}, X{rt2}, [X{rn}, #{imm7 * 8}]!"
    if (inst & 0xFFC00000) == 0xA9000000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"STP X{rd}, X{rt2}, [X{rn}, #{imm7 * 8}]"
    if (inst & 0xFFC00000) == 0xA9400000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"LDP X{rd}, X{rt2}, [X{rn}, #{imm7 * 8}]"
    if (inst & 0xFFC00000) == 0xA9C00000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"LDP X{rd}, X{rt2}, [X{rn}, #{imm7 * 8}]!"
    if op == 0xD1:
        sh = (inst >> 22) & 1
        v = imm12 << (12 if sh else 0)
        return f"SUB X{rd}, X{rn}, #{v}"
    if op == 0x91:
        sh = (inst >> 22) & 1
        v = imm12 << (12 if sh else 0)
        return f"ADD X{rd}, X{rn}, #{v}"
    if op == 0x51:
        return f"SUB W{rd}, W{rn}, #{imm12}"
    if op == 0x11:
        return f"ADD W{rd}, W{rn}, #{imm12}"
    if (inst & 0xFFC00000) == 0xF9000000:
        return f"STR X{rd}, [X{rn}, #{imm12 * 8}]"
    if (inst & 0xFFC00000) == 0xF9400000:
        return f"LDR X{rd}, [X{rn}, #{imm12 * 8}]"
    if (inst & 0xFFC00000) == 0xB9000000:
        return f"STR W{rd}, [X{rn}, #{imm12 * 4}]"
    if (inst & 0xFFC00000) == 0xB9400000:
        return f"LDR W{rd}, [X{rn}, #{imm12 * 4}]"
    if (inst & 0xFFC00000) == 0x39000000:
        return f"STRB W{rd}, [X{rn}, #{imm12}]"
    if (inst & 0xFFC00000) == 0x39400000:
        return f"LDRB W{rd}, [X{rn}, #{imm12}]"
    if (inst & 0xFFC00000) == 0x79000000:
        return f"STRH W{rd}, [X{rn}, #{imm12 * 2}]"
    if (inst & 0xFFC00000) == 0x79400000:
        return f"LDRH W{rd}, [X{rn}, #{imm12 * 2}]"
    if (inst & 0xFF000000) == 0x54000000:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        conds = ['EQ', 'NE', 'CS', 'CC', 'MI', 'PL', 'VS', 'VC',
                 'HI', 'LS', 'GE', 'LT', 'GT', 'LE', 'AL', 'NV']
        return f"B.{conds[inst & 0xF]} 0x{(pc + imm19 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFF000000) == 0x35000000:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        return f"CBNZ W{rd}, 0x{(pc + imm19 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFF000000) == 0x34000000:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        return f"CBZ W{rd}, 0x{(pc + imm19 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFF000000) == 0xB5000000:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        return f"CBNZ X{rd}, 0x{(pc + imm19 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFF000000) == 0xB4000000:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        return f"CBZ X{rd}, 0x{(pc + imm19 * 4) & 0xFFFFFFFF:08X}"
    if (inst & 0xFFE0001F) == 0xAA0003E0:
        return f"MOV X{rd}, X{rm}"
    if (inst & 0xFFE0001F) == 0x2A0003E0:
        return f"MOV W{rd}, W{rm}"
    if (inst & 0xFF800000) == 0xD2800000:
        hw = (inst >> 21) & 0x3
        imm16 = (inst >> 5) & 0xFFFF
        return f"MOVZ X{rd}, #0x{imm16:X}{f', LSL #{hw * 16}' if hw else ''}"
    if (inst & 0xFF800000) == 0x52800000:
        hw = (inst >> 21) & 0x3
        imm16 = (inst >> 5) & 0xFFFF
        return f"MOVZ W{rd}, #0x{imm16:X}{f', LSL #{hw * 16}' if hw else ''}"
    if (inst & 0xFF800000) == 0xF2800000:
        hw = (inst >> 21) & 0x3
        imm16 = (inst >> 5) & 0xFFFF
        return f"MOVK X{rd}, #0x{imm16:X}{f', LSL #{hw * 16}' if hw else ''}"
    # STP W pre-index
    if (inst & 0xFFC00000) == 0x29800000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"STP W{rd}, W{rt2}, [X{rn}, #{imm7 * 4}]!"
    # STP W offset
    if (inst & 0xFFC00000) == 0x29000000:
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 >= 64: imm7 -= 128
        return f"STP W{rd}, W{rt2}, [X{rn}, #{imm7 * 4}]"
    # STUR/LDUR W
    if (inst & 0xFFE00C00) == 0xB8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR W{rd}, [X{rn}, #{imm9}]"
    if (inst & 0xFFE00C00) == 0xB8400000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"LDUR W{rd}, [X{rn}, #{imm9}]"
    # STUR/LDUR X
    if (inst & 0xFFE00C00) == 0xF8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR X{rd}, [X{rn}, #{imm9}]"
    if (inst & 0xFFE00C00) == 0xF8400000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"LDUR X{rd}, [X{rn}, #{imm9}]"
    # CMP (SUBS XZR)
    if (inst & 0xFF200000) == 0xEB000000 and rd == 31:
        return f"CMP X{rn}, X{rm}"
    if (inst & 0xFF200000) == 0x6B000000 and rd == 31:
        return f"CMP W{rn}, W{rm}"
    if op == 0xF1 and rd == 31:
        return f"CMP X{rn}, #{imm12}"
    if op == 0x71 and rd == 31:
        return f"CMP W{rn}, #{imm12}"
    # TST (ANDS XZR)
    if (inst & 0xFF200000) == 0xEA000000 and rd == 31:
        return f"TST X{rn}, X{rm}"
    if (inst & 0xFF200000) == 0x6A000000 and rd == 31:
        return f"TST W{rn}, W{rm}"
    # ADRP
    if op & 0x9F == 0x90:
        immhi = (inst >> 5) & 0x7FFFF
        immlo = (inst >> 29) & 0x3
        imm = (immhi << 2) | immlo
        if imm >= (1 << 20): imm -= (1 << 21)
        page = ((pc >> 12) + imm) << 12
        return f"ADRP X{rd}, 0x{page & 0xFFFFFFFF:08X}"
    # LDR literal
    if op == 0x18:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        addr = pc + imm19 * 4
        return f"LDR W{rd}, [PC+{imm19*4:+d}] (0x{addr & 0xFFFFFFFF:08X})"
    if op == 0x58:
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= (1 << 18): imm19 -= (1 << 19)
        addr = pc + imm19 * 4
        return f"LDR X{rd}, [PC+{imm19*4:+d}] (0x{addr & 0xFFFFFFFF:08X})"

    return f"??? 0x{inst:08X} (op=0x{op:02X})"


def read_string(cpu, addr, max_len=64):
    data = cpu.read_memory(addr, max_len)
    null = data.find(0)
    if null >= 0: data = data[:null]
    return bytes(data)


if __name__ == "__main__":
    print("=" * 70)
    print("GREP PARSE PHASE TRACE — Capture mbsrtowcs + tre_parse_atom")
    print("=" * 70)

    # ========================================================
    # Phase 1: Verify grep's regex compile call arguments
    # ========================================================
    print("\n--- Phase 1: Verify grep regex compile args ---")
    cpu, handler = setup()
    cpu.set_breakpoint(0, GREP_COMPILE_CALL)
    cycles, stop = run_to_bp(cpu, handler)
    if stop == "BP":
        regs = {r: cpu.get_register(r) & 0xFFFFFFFFFFFFFFFF for r in range(4)}
        sp = cpu.get_register(31) & 0xFFFFFFFFFFFFFFFF
        print(f"  grep compile call (0x{GREP_COMPILE_CALL:08X}), cycle {cycles}")
        for r in range(4):
            v = regs[r]
            extra = ""
            if 0x400000 <= v <= 0x500000 or 0xFF0000 <= v <= 0x1000000:
                try: extra = f" → {read_string(cpu, v)}"
                except: pass
            print(f"  x{r} = 0x{v:016X}{extra}")
        print(f"  SP  = 0x{sp:X}")

    # ========================================================
    # Phase 2: Trace first 2000 cycles INSIDE the real regcomp entry
    # ========================================================
    print(f"\n--- Phase 2: First 2000 cycles inside 0x{REGCOMP_ENTRY:08X} ---")
    cpu2, handler2 = setup()
    cpu2.set_breakpoint(0, REGCOMP_ENTRY)
    cycles2, stop2 = run_to_bp(cpu2, handler2)
    print(f"  Reached 0x{REGCOMP_ENTRY:08X} at cycle {cycles2}: {stop2}")

    if stop2 == "BP":
        sp_entry = cpu2.get_register(31) & 0xFFFFFFFFFFFFFFFF
        regs_entry = {r: cpu2.get_register(r) & 0xFFFFFFFFFFFFFFFF for r in range(4)}
        print(f"  x0=0x{regs_entry[0]:X} x1=0x{regs_entry[1]:X} x2=0x{regs_entry[2]:X} x3=0x{regs_entry[3]:X}")
        print(f"  SP=0x{sp_entry:X}")
        for r in (0, 1, 2):
            v = regs_entry[r]
            if 0x400000 <= v <= 0x500000 or 0xFF0000 <= v <= 0x1000000:
                try: print(f"  *x{r} = {read_string(cpu2, v)}")
                except: pass

        # Enable trace, run 2000 cycles
        cpu2.clear_breakpoints()
        cpu2.enable_trace()
        cycles2b, stop2b = run_batches(cpu2, handler2, 2000)
        print(f"  Ran {cycles2b} more cycles ({stop2b}), PC=0x{cpu2.pc:X}")

        trace = cpu2.read_trace()
        print(f"  Trace: {len(trace)} entries\n")

        # Print ALL trace entries
        print("  FULL TRACE (first 2000 instructions):")
        bl_targets = {}
        for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
            d = decode(inst, pc)
            nzcv = f"{'N' if flags & 8 else '.'}{'Z' if flags & 4 else '.'}{'C' if flags & 2 else '.'}{'V' if flags & 1 else '.'}"
            print(f"  [{i:4d}] 0x{pc:08X}: {d:50s} x0={x0 & 0xFFFFFFFFFFFFFFFF:016X} x1={x1 & 0xFFFFFFFFFFFFFFFF:016X} x2={x2 & 0xFFFFFFFFFFFFFFFF:016X} x3={x3 & 0xFFFFFFFFFFFFFFFF:016X} {nzcv} SP={sp & 0xFFFFFFFFFFFFFFFF:X}")

            # Track BL targets
            if (inst & 0xFC000000) == 0x94000000:
                imm26 = inst & 0x3FFFFFF
                if imm26 >= (1 << 25): imm26 -= (1 << 26)
                target = (pc + imm26 * 4) & 0xFFFFFFFF
                bl_targets.setdefault(target, []).append(i)

        # Summary of BL targets
        print(f"\n  BL target summary ({len(bl_targets)} unique):")
        for target, indices in sorted(bl_targets.items()):
            print(f"    0x{target:08X}: called {len(indices)}x at entries {indices[:8]}")

        # Check for wchar_t values in stack
        sp_now = cpu2.get_register(31) & 0xFFFFFFFFFFFFFFFF
        print(f"\n  SP at entry: 0x{sp_entry:X}, SP now: 0x{sp_now:X}")
        print(f"  Stack frame size: {sp_entry - sp_now} bytes")

        # Scan stack for wchar_t 'r' (0x72) followed by 'o','o','t'
        print(f"\n  Scanning stack 0x{max(sp_now - 256, 0):X} - 0x{sp_entry + 256:X} for wchar_t 'root'...")
        scan_lo = max(sp_now - 256, 0)
        scan_hi = min(sp_entry + 256, 0x1000000)
        for addr in range(scan_lo, scan_hi, 4):
            try:
                val = struct.unpack('<I', cpu2.read_memory(addr, 4))[0]
                if val == 0x72:
                    vals = [struct.unpack('<I', cpu2.read_memory(addr + j * 4, 4))[0] for j in range(5)]
                    if vals[:4] == [0x72, 0x6F, 0x6F, 0x74]:
                        print(f"    *** FOUND wchar_t 'root' at 0x{addr:X}! vals={[hex(v) for v in vals]}")
                    else:
                        print(f"    Found 0x72 at 0x{addr:X}, next: {[hex(v) for v in vals]}")
            except:
                pass

    print("\n--- COMPLETE ---")
