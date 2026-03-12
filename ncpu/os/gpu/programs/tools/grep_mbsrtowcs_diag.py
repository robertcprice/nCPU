#!/usr/bin/env python3
"""Trace mbsrtowcs in grep to understand why wchar_t pattern is all zeros.

From previous diagnostics:
- NFA builder correctly copies code_min/code_max from AST literal nodes
- AST literal nodes have code_min=code_max=0 (parser stored zeros)
- No wchar_t 'root' found in heap memory
- regcomp calls mbsrtowcs to convert byte pattern to wchar_t
- mbsrtowcs either stores zeros or the VLA is misaligned

Static disassembly corrected the compile path:
- 0x0041A374 is grep's regex compile call site
- 0x0041E4AC is BusyBox's compile-and-report wrapper
- 0x00428C4C is the real musl regcomp entry

Strategy:
1. Break at the real regcomp entry and trace the first ~5K instructions inside it
2. Find LDRB instructions that load pattern bytes
3. Find STR W instructions that store wchar_t values
4. Find the transition from mbsrtowcs to tre_compile
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


def run_batches(cpu, handler, total_target, batch=1000):
    """Run in batches, handling syscalls."""
    total = 0
    while total < total_target:
        result = cpu.execute(max_cycles=batch)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return total, "EXIT"
            elif ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason in (StopReasonV2.HALT,):
            return total, "HALT"
        elif result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BREAKPOINT"
        elif result.stop_reason == StopReasonV2.WATCHPOINT:
            return total, "WATCHPOINT"
    return total, "MAX_CYCLES"


def run_to_bp(cpu, handler, max_cycles=100_000):
    """Run until a breakpoint/watchpoint/syscall/halt."""
    total = 0
    while total < max_cycles:
        result = cpu.execute(max_cycles=1000)
        total += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.BREAKPOINT:
            return total, "BP"
        if result.stop_reason == StopReasonV2.WATCHPOINT:
            return total, "WP"
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return total, "EXIT"
            if ret != "exec":
                cpu.set_pc(cpu.pc + 4)
        elif result.stop_reason == StopReasonV2.HALT:
            return total, "HALT"
    return total, "MAX"


if __name__ == "__main__":
    print("=" * 60)
    print("GREP MBSRTOWCS DIAGNOSTIC — Pattern Conversion Tracing")
    print("=" * 60)

    # === Phase 1: Break at the real regcomp entry, then trace from there ===
    print(f"\n--- Phase 1: Break at regcomp entry 0x{REGCOMP_ENTRY:08X} ---")
    cpu, handler = setup()
    cpu.set_breakpoint(0, REGCOMP_ENTRY)
    cycles, stop = run_to_bp(cpu, handler)
    print(f"  Reached regcomp: {stop} at cycle {cycles}, PC=0x{cpu.pc:X}")

    if stop != "BP":
        print("  Failed to reach the regcomp entry; aborting trace analysis.")
        raise SystemExit(1)

    regs = {r: cpu.get_register(r) & 0xFFFFFFFFFFFFFFFF for r in range(3)}
    print(f"  x0=0x{regs[0]:X} x1=0x{regs[1]:X} x2=0x{regs[2]:X}")
    try:
        print(f"  pattern = {cpu.read_memory(regs[1], 16).split(b'\\0', 1)[0]}")
    except Exception:
        pass

    cpu.clear_breakpoints()
    cpu.enable_trace()
    cycles2, stop2 = run_batches(cpu, handler, 5000)
    print(f"  Traced {cycles2} cycles from regcomp entry, stop={stop2}, PC=0x{cpu.pc:X}")

    trace = cpu.read_trace()
    print(f"  Trace entries: {len(trace)}")

    # Find the pattern "root" on the stack
    # argv[1] should point to the pattern string
    # Search stack region for "root"
    stack_top = 0xFFFF00  # Near top of 16MB memory
    for probe_addr in range(0xFFD000, 0x1000000, 0x100):
        try:
            d = cpu.read_memory(probe_addr, 256)
            idx = d.find(b'root')
            if idx >= 0 and idx < 252:
                # Check if it's the pattern (not part of /etc/passwd path)
                # Pattern "root" should be followed by \0
                if d[idx+4] == 0:
                    print(f"  Pattern 'root' found at 0x{probe_addr+idx:X}")
                    # Also show surrounding context
                    print(f"    Context: {d[max(0,idx-8):idx+16]}")
                    pattern_addr = probe_addr + idx
                    break
        except:
            pass
    else:
        print("  Pattern not found on stack, checking other regions...")
        # The ELF argv is set up by the loader
        # Check near the initial stack pointer
        for probe_addr in range(0xFF0000, 0x1000000, 0x100):
            d = cpu.read_memory(probe_addr, 256)
            idx = d.find(b'root')
            if idx >= 0 and idx < 252 and d[idx+4] == 0:
                print(f"  Pattern 'root' found at 0x{probe_addr+idx:X}")
                pattern_addr = probe_addr + idx
                break

    # Analyze the trace for key instruction patterns
    print("\n--- Phase 2: Analyze trace for mbsrtowcs pattern ---")

    # Look for LDRB instructions (byte loads) - these are how mbsrtowcs reads the pattern
    # LDRB encoding: 0x39400000 (unsigned offset) or 0x38 (complex)
    print("\n  LDRB instructions (byte loads):")
    ldrb_count = 0
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
        op = (inst >> 24) & 0xFF

        # LDRB unsigned offset: 0x39400000 mask
        if (inst & 0xFFC00000) == 0x39400000:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            rn_val = {0: x0, 1: x1, 2: x2, 3: x3}.get(rn, None)
            if rn_val is not None:
                addr = (rn_val & 0xFFFFFFFFFFFFFFFF) + imm12
                ldrb_count += 1
                if ldrb_count <= 50:
                    print(f"    [{i:4d}] PC=0x{pc:08X}: LDRB W{rd}, [X{rn}, #{imm12}] base=0x{rn_val&0xFFFFFFFFFFFFFFFF:X} → addr=0x{addr:X}")

        # LDRB post-index: 0x38 with bits 11-10 = 01
        if op == 0x38 and ((inst >> 10) & 0x3) == 0x1:
            opc = (inst >> 22) & 0x3
            if opc == 1:  # load
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                imm9 = (inst >> 12) & 0x1FF
                if imm9 >= 256: imm9 -= 512
                rn_val = {0: x0, 1: x1, 2: x2, 3: x3}.get(rn, None)
                if rn_val is not None:
                    addr = rn_val & 0xFFFFFFFFFFFFFFFF
                    ldrb_count += 1
                    if ldrb_count <= 50:
                        print(f"    [{i:4d}] PC=0x{pc:08X}: LDRB W{rd}, [X{rn}], #{imm9} base=0x{addr:X}")

    print(f"\n  Total LDRB instructions: {ldrb_count}")

    # Look for STR W post-index (the *ws++ = c pattern from mbsrtowcs)
    # STR W post-index: 0xB8 with bits 11-10 = 01, opc = 00
    print("\n  STR W post-index instructions (wchar_t store+increment):")
    str_post_count = 0
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
        op = (inst >> 24) & 0xFF
        if op == 0xB8 and ((inst >> 10) & 0x3) == 0x1:
            opc = (inst >> 22) & 0x3
            if opc == 0:  # store
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                imm9 = (inst >> 12) & 0x1FF
                if imm9 >= 256: imm9 -= 512
                rn_val = {0: x0, 1: x1, 2: x2, 3: x3}.get(rn, None)
                rd_val = {0: x0, 1: x1, 2: x2, 3: x3}.get(rd, None)
                str_post_count += 1
                base_str = f"base(x{rn})=0x{rn_val & 0xFFFFFFFFFFFFFFFF:X}" if rn_val is not None else f"base=x{rn}"
                val_str = f"val(w{rd})=0x{rd_val & 0xFFFFFFFF:X}" if rd_val is not None else f"val=w{rd}"
                print(f"    [{i:4d}] PC=0x{pc:08X}: STR W{rd}, [X{rn}], #{imm9} {base_str} {val_str}")

    print(f"\n  Total STR W post-index: {str_post_count}")

    # === Phase 3: Look for BL calls inside regcomp ===
    print("\n--- Phase 3: BL calls inside regcomp trace ---")
    bl_count = 0
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
        if (inst & 0xFC000000) == 0x94000000:  # BL
            imm26 = inst & 0x3FFFFFF
            if imm26 >= (1<<25): imm26 -= (1<<26)
            target = pc + imm26 * 4
            bl_count += 1
            print(f"    [{i:4d}] PC=0x{pc:08X}: BL 0x{target:08X}  x0=0x{x0 & 0xFFFFFFFFFFFFFFFF:X} x1=0x{x1 & 0xFFFFFFFFFFFFFFFF:X}")

    # === Phase 4: Find where wchar_t pattern SHOULD be stored ===
    # After mbsrtowcs, the wchar_t buffer is on the stack of regcomp
    # Let's track SP changes to find the VLA address
    print("\n--- Phase 4: SP tracking via SUB/ADD on X31 ---")
    sp_ops = 0
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace[:500]):
        op = (inst >> 24) & 0xFF
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F

        # SUB X31, X31, #imm (SP adjustment)
        if op == 0xD1 and rd == 31 and rn == 31:
            imm12 = (inst >> 10) & 0xFFF
            shift = (inst >> 22) & 1
            val = imm12 << (12 if shift else 0)
            print(f"    [{i:4d}] PC=0x{pc:08X}: SUB SP, SP, #{val}")
            sp_ops += 1
        # ADD X31, X31, #imm (SP restore)
        elif op == 0x91 and rd == 31 and rn == 31:
            imm12 = (inst >> 10) & 0xFFF
            shift = (inst >> 22) & 1
            val = imm12 << (12 if shift else 0)
            print(f"    [{i:4d}] PC=0x{pc:08X}: ADD SP, SP, #{val}")
            sp_ops += 1
        # STP with SP writeback (pre-index)
        elif (inst & 0xFFC00000) == 0xA9800000 and rn == 31:
            imm7 = (inst >> 15) & 0x7F
            if imm7 >= 64: imm7 -= 128
            print(f"    [{i:4d}] PC=0x{pc:08X}: STP X, X, [SP, #{imm7*8}]! (pre-index)")
            sp_ops += 1

    print(f"\n  SP operations in first 500 instructions: {sp_ops}")

    # === Phase 5: Nested function entries inside the regcomp trace ===
    print("\n--- Phase 5: Nested function entries in regex range ---")
    # Look for STP X29, X30, [SP, #-N]! (function prologue)
    for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace):
        if (inst & 0xFFE00000) == 0xA9800000:  # STP pre-index 64-bit
            rn = (inst >> 5) & 0x1F
            if rn == 31:  # SP
                rd = inst & 0x1F
                rt2 = (inst >> 10) & 0x1F
                if rd == 29 and rt2 == 30:  # x29, x30 (frame pointer, link register)
                    if 0x420000 <= pc < 0x440000:
                        print(f"    [{i:4d}] PC=0x{pc:08X}: function entry (STP X29, X30, [SP, #-N]!) x0=0x{x0&0xFFFFFFFFFFFFFFFF:X}")

    print("\n--- DIAGNOSTIC COMPLETE ---")
