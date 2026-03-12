#!/usr/bin/env python3
"""Diagnose regcomp NFA transition issue using direct cpu.execute() with watchpoints.

The key insight: run() doesn't handle WATCHPOINT stops, so we need to call
cpu.execute() directly and check stop_reason after each batch.
"""
import sys, struct
sys.path.insert(0, "/Users/bobbyprice/projects/nCPU")

from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU
from pathlib import Path

ELF_PATH = "demos/busybox.elf"


def make_fs():
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n")
    return fs


def decode_inst(inst):
    """Minimal ARM64 decode."""
    op = (inst >> 24) & 0xFF
    rd = inst & 0x1F
    rn = (inst >> 5) & 0x1F
    rt2 = (inst >> 10) & 0x1F
    rm = (inst >> 16) & 0x1F
    imm12 = (inst >> 10) & 0xFFF
    imm7 = (inst >> 15) & 0x7F

    # STP 32-bit signed
    if (inst & 0xFFC00000) == 0x29000000:
        s = imm7 if imm7 < 64 else imm7 - 128
        return f"STP W{rd}, W{rt2}, [X{rn}, #{s*4}]"
    if (inst & 0xFFC00000) == 0xA9000000:
        s = imm7 if imm7 < 64 else imm7 - 128
        return f"STP X{rd}, X{rt2}, [X{rn}, #{s*8}]"
    if (inst & 0xFFC00000) == 0xB9000000:
        return f"STR W{rd}, [X{rn}, #{imm12*4}]"
    if (inst & 0xFFC00000) == 0xF9000000:
        return f"STR X{rd}, [X{rn}, #{imm12*8}]"
    if (inst & 0xFFE00C00) == 0xB8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR W{rd}, [X{rn}, #{imm9}]"
    if (inst & 0xFFE00C00) == 0xF8000000:
        imm9 = (inst >> 12) & 0x1FF
        if imm9 >= 256: imm9 -= 512
        return f"STUR X{rd}, [X{rn}, #{imm9}]"
    if (inst & 0xFFC00000) == 0xB9400000:
        return f"LDR W{rd}, [X{rn}, #{imm12*4}]"
    if (inst & 0xFFC00000) == 0xF9400000:
        return f"LDR X{rd}, [X{rn}, #{imm12*8}]"
    if inst == 0xD65F03C0:
        return "RET"
    if (inst & 0xFC000000) == 0x94000000:
        imm26 = inst & 0x3FFFFFF
        if imm26 >= (1<<25): imm26 -= (1<<26)
        return f"BL #{imm26*4:+d}"
    if op == 0xD4:
        return "SVC #0"
    return f"0x{inst:08X} (op=0x{op:02X})"


def setup_cpu_for_grep():
    """Create and configure CPU for grep execution."""
    cpu = GPUKernelCPU(quiet=True)
    fs = make_fs()

    entry = load_elf_into_memory(
        cpu, ELF_PATH,
        argv=["grep", "root", "/etc/passwd"],
        quiet=True
    )
    cpu.set_pc(entry)

    elf_data = Path(ELF_PATH).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    SVC_BUF_BASE = 0x3F0000
    cpu.write_memory(SVC_BUF_BASE + 8, struct.pack('<Q', heap_base))

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)
    return cpu, handler, fs


def run_with_watchpoint_direct(wp_addr, label, max_cycles=50_000):
    """Run grep with watchpoint, using cpu.execute() directly."""
    cpu, handler, fs = setup_cpu_for_grep()

    # Init SVC buffer
    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    # Set watchpoint and enable trace
    cpu.enable_trace()
    cpu.set_watchpoint(0, wp_addr)

    print(f"\n  Watchpoint on 0x{wp_addr:X} ({label})")
    print(f"  Running with direct cpu.execute() in 1000-cycle batches...")

    total = 0
    batch = 1000
    from kernels.mlx.rust_runner import StopReasonV2

    while total < max_cycles:
        result = cpu.execute(max_cycles=batch)
        total += result.cycles

        # Drain GPU writes
        for fd, data in cpu.drain_svc_buffer():
            pass  # suppress output

        if result.stop_reason == StopReasonV2.WATCHPOINT:
            wp_info = cpu.read_watchpoint_info()
            print(f"\n  *** WATCHPOINT HIT at cycle {total} ***")
            if wp_info:
                idx, addr, old_val, new_val = wp_info
                print(f"  WP index: {idx}, addr: 0x{addr:X}")
                print(f"  Old: 0x{old_val:016X} ({old_val})")
                print(f"  New: 0x{new_val:016X} ({new_val})")
                lo = new_val & 0xFFFFFFFF
                hi = (new_val >> 32) & 0xFFFFFFFF
                print(f"  As 2x32: lo=0x{lo:08X} ({lo}), hi=0x{hi:08X} ({hi})")

            # Read trace
            trace = cpu.read_trace()
            print(f"\n  Trace: {len(trace)} entries. Last 30:")
            for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace[-30:]):
                d = decode_inst(inst)
                nzcv = f"{'N' if flags&8 else '.'}{'Z' if flags&4 else '.'}{'C' if flags&2 else '.'}{'V' if flags&1 else '.'}"
                print(f"    [{len(trace)-30+i:4d}] 0x{pc:08X}: {d:40s} x0=0x{x0&0xFFFFFFFFFFFFFFFF:X} x1=0x{x1&0xFFFFFFFFFFFFFFFF:X} x2=0x{x2&0xFFFFFFFFFFFFFFFF:X} x3=0x{x3&0xFFFFFFFFFFFFFFFF:X} {nzcv}")

            # Print registers around the interesting range
            print(f"\n  Key registers at watchpoint:")
            for r in range(0, 29):
                v = cpu.get_register(r)
                if v != 0:
                    print(f"    x{r} = 0x{v & 0xFFFFFFFFFFFFFFFF:016X} ({v})")

            return cpu, total, True

        elif result.stop_reason == StopReasonV2.SYSCALL:
            # Handle syscall
            ret = handler(cpu)
            if ret == False:
                print(f"\n  Program exited at cycle {total}. Watchpoint never fired.")
                return cpu, total, False
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

        elif result.stop_reason == StopReasonV2.HALT:
            print(f"\n  HALT at cycle {total}. Watchpoint never fired.")
            return cpu, total, False

        elif result.stop_reason == StopReasonV2.BREAKPOINT:
            print(f"\n  BREAKPOINT at cycle {total}, PC=0x{cpu.pc:X}")
            trace = cpu.read_trace()
            print(f"  Trace: {len(trace)} entries. Last 10:")
            for i, (pc, inst, x0, x1, x2, x3, flags, sp) in enumerate(trace[-10:]):
                d = decode_inst(inst)
                print(f"    [{len(trace)-10+i:4d}] 0x{pc:08X}: {d}")
            return cpu, total, True

    print(f"\n  Max cycles ({max_cycles}) reached. Watchpoint never fired.")
    return cpu, total, False


if __name__ == "__main__":
    print("=" * 60)
    print("GREP REGCOMP DIAGNOSTIC v2 — Direct cpu.execute()")
    print("=" * 60)

    # Test 1: Watchpoint on NFA state pointer (0x876068)
    # We know this is written correctly — let's find WHEN
    print("\n--- TEST 1: When is NFA state pointer at 0x876068 written? ---")
    cpu1, cycles1, hit1 = run_with_watchpoint_direct(0x876068, "state_ptr[0]")

    if hit1:
        # Read the transition data after the watchpoint
        data = cpu1.read_memory(0x876060, 48)
        cm = struct.unpack_from('<I', data, 0)[0]
        cx = struct.unpack_from('<I', data, 4)[0]
        sp = struct.unpack_from('<Q', data, 8)[0]
        print(f"\n  After watchpoint, transition[0]:")
        print(f"    code_min = 0x{cm:X}")
        print(f"    code_max = 0x{cx:X}")
        print(f"    state    = 0x{sp:X}")

    # Test 2: Watchpoint on code_min/code_max (0x876060)
    print("\n--- TEST 2: Is code_min/code_max at 0x876060 ever written? ---")
    cpu2, cycles2, hit2 = run_with_watchpoint_direct(0x876060, "code_min/max[0]")

    # Test 3: Also try watchpoint on the tags field (0x876070, offset 16)
    # Tags value = 1 (non-zero), so this should fire
    print("\n--- TEST 3: When is tags field at 0x876070 written? ---")
    cpu3, cycles3, hit3 = run_with_watchpoint_direct(0x876070, "tags[0]")

    # Test 4: Dump a wider range around the NFA after full completion
    print("\n--- TEST 4: Full NFA data after grep completion ---")
    cpu4, handler4, fs4 = setup_cpu_for_grep()
    if cpu4.memory_size > cpu4.SVC_BUF_BASE + 0x10000:
        cpu4.init_svc_buffer()

    from ncpu.os.gpu.runner import run
    result = run(cpu4, handler4, max_cycles=100_000, quiet=True)
    print(f"  Grep: {result['total_cycles']} cycles, exit={cpu4.get_register(0)}")

    # Check sizes around NFA
    # The TRE transition struct is 48 bytes on LP64:
    #   code_min(4) + code_max(4) + *state(8) + *tags(8) + assertions(4) + pad(4) + union(8) + *neg_classes(8) = 48
    print("\n  NFA chain (48-byte stride):")
    addrs = [0x876060, 0x8760D0, 0x876140, 0x8761B0, 0x876220]
    for i, addr in enumerate(addrs):
        d = cpu4.read_memory(addr, 48)
        cm = struct.unpack_from('<I', d, 0)[0]
        cx = struct.unpack_from('<I', d, 4)[0]
        sp = struct.unpack_from('<Q', d, 8)[0]
        tg = struct.unpack_from('<Q', d, 16)[0]
        ass = struct.unpack_from('<I', d, 24)[0]
        pad = struct.unpack_from('<I', d, 28)[0]
        u = struct.unpack_from('<Q', d, 32)[0]
        nc = struct.unpack_from('<Q', d, 40)[0]

        # Note: stride between entries is 0x70 = 112 bytes, not 48.
        # So there might be MULTIPLE transitions per state.
        cm_s = f"'{chr(cm)}'" if 32 <= cm < 127 else f"0x{cm:X}"
        cx_s = f"'{chr(cx)}'" if 32 <= cx < 127 else f"0x{cx:X}"

        print(f"  [{i}] 0x{addr:X}:")
        print(f"    code_min={cm_s:8s}  code_max={cx_s:8s}  state=0x{sp:X}")
        print(f"    tags=0x{tg:X}  assertions={ass}  pad={pad}  union=0x{u:X}  neg_classes=0x{nc:X}")

    # Check what's between the entries (0x876060+48 = 0x876090 to 0x8760D0)
    # This gap is 112 - 48 = 64 bytes. Could be a sentinel transition or state metadata.
    print("\n  Gap between transitions (sentinel/metadata):")
    for i in range(len(addrs) - 1):
        gap_start = addrs[i] + 48
        gap_end = addrs[i + 1]
        gap_size = gap_end - gap_start
        if gap_size > 0:
            d = cpu4.read_memory(gap_start, min(gap_size, 64))
            # Check if this looks like another transition (sentinel with code_min=-1)
            cm = struct.unpack_from('<i', d, 0)[0]  # signed!
            cx = struct.unpack_from('<i', d, 4)[0]
            sp = struct.unpack_from('<Q', d, 8)[0]
            print(f"  0x{gap_start:X}-0x{gap_end:X} ({gap_size}B): code_min={cm}, code_max={cx}, state=0x{sp:X}")
            # Raw hex of first 32 bytes
            print(f"    hex: {' '.join(f'{b:02X}' for b in d[:32])}")

    print("\n--- DIAGNOSTIC v2 COMPLETE ---")
