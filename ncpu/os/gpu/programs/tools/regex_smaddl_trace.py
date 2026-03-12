#!/usr/bin/env python3
"""
Trace the SMADDL instruction and NFA table address computations.

Key instructions:
  0x425FC0: smaddl x0, w0, w28, x26   -- compute entry address during inner build
  0x42613C: smaddl x19, w19, w28, x26  -- compute entry address during outer loop
  0x425F54: mov w1, #0xFFFFFFFF        -- terminal marker
  0x425F58: str w1, [x0]              -- write terminal to [x0]

We need to trace what x0 holds when str at 0x425F58 executes.
If x0 is wrong (offset by 8 bytes), that's the bug.

Strategy: Run with batch_size=1 in the critical region to see every instruction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")

# Key PCs to watch
SMADDL_1 = 0x425FC0  # smaddl x0, w0, w28, x26
SMADDL_2 = 0x42613C  # smaddl x19, w19, w28, x26
TERMINAL_MOV = 0x425F54  # mov w1, #0xFFFFFFFF
TERMINAL_STR = 0x425F58  # str w1, [x0]
FUNC_ENTRY = 0x425F20   # function entry (mov x5, x0)
STUCK_LOOP = 0x4261F4   # ldrsw x0, [x23] -- stuck loop start
STUCK_STR  = 0x426204   # str w1, [x3, x0] -- stuck store

# NFA builder region: 0x425F60 to 0x426268
BUILD_START = 0x425F60
BUILD_END = 0x426268

# The inner function that writes terminals: 0x425F20-0x425F5C
TERMINAL_FUNC_START = 0x425F20
TERMINAL_FUNC_END = 0x425F5C


def main():
    fs = GPUFilesystem()
    fs.write_file("/etc/passwd",
        "root:x:0:0:root:/root:/bin/sh\n"
        "hello_user:x:1000:1000:hello:/home/hello:/bin/sh\n"
    )

    cpu = MLXKernelCPUv2(quiet=True)
    argv = ["grep", "hello", "/etc/passwd"]
    entry = load_elf_into_memory(cpu, BUSYBOX, argv=argv, quiet=True)
    cpu.set_pc(entry)

    elf_data = Path(BUSYBOX).read_bytes()
    elf_info = parse_elf(elf_data)
    max_seg_end = max(s.vaddr + s.memsz for s in elf_info.segments)
    heap_base = (max_seg_end + 0xFFFF) & ~0xFFFF

    handler = make_busybox_syscall_handler(filesystem=fs, heap_base=heap_base)

    if cpu.memory_size > cpu.SVC_BUF_BASE + 0x10000:
        cpu.init_svc_buffer()

    # Phase 1: Run to cycle 14800 (before NFA build, but fast)
    total_cycles = 0
    while total_cycles < 14_800:
        result = cpu.execute(max_cycles=1000)
        total_cycles += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    print(f"Phase 1 complete: {total_cycles:,} cycles, PC=0x{cpu.pc:X}")

    # Phase 2: Run with batch_size=100, stop and analyze when we enter NFA builder
    # We need to find when PC enters the region 0x425F60-0x426268
    batch_size = 100
    in_builder = False
    builder_entry_cycle = 0

    while total_cycles < 20_000:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)
            continue

        pc = cpu.pc

        # Check if we're in the NFA builder region
        if BUILD_START <= pc <= BUILD_END and not in_builder:
            in_builder = True
            builder_entry_cycle = total_cycles
            print(f"\n  Entered NFA builder at cycle {total_cycles}, PC=0x{pc:X}")

            # Dump key registers
            x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
            x19 = cpu.get_register(19) & 0xFFFFFFFFFFFFFFFF
            x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
            x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            x27 = cpu.get_register(27) & 0xFFFFFFFFFFFFFFFF
            x28 = cpu.get_register(28) & 0xFFFFFFFFFFFFFFFF
            print(f"  x0=0x{x0:X} x19=0x{x19:X}")
            print(f"  x22=0x{x22:X} x23=0x{x23:X}")
            print(f"  x26=0x{x26:X} x27=0x{x27:X} x28=0x{x28:X}")
            break

        if pc == STUCK_LOOP or pc == STUCK_STR:
            print(f"  Hit stuck loop at cycle {total_cycles}, PC=0x{pc:X}")
            break

    if not in_builder:
        print("  Never entered NFA builder region!")
        return

    # Phase 3: Now run with batch_size=1 to trace every instruction in the builder
    # Focus on SMADDL results and terminal writes
    print(f"\n  Phase 3: Single-step tracing in NFA builder")
    print(f"  Watching for SMADDL (0x425FC0, 0x42613C) and terminal STR (0x425F58)")

    step_count = 0
    max_steps = 5000  # Don't go too far

    smaddl_results = []
    terminal_writes = []

    while step_count < max_steps and total_cycles < 25_000:
        # Take a snapshot of key registers BEFORE execution
        pc_before = cpu.pc

        result = cpu.execute(max_cycles=1)
        total_cycles += result.cycles
        step_count += 1

        for fd, data in cpu.drain_svc_buffer():
            pass

        if result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            ret = handler(cpu)
            if ret == False:
                return
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)
            continue

        pc_after = cpu.pc

        # Read instruction at pc_before
        inst_bytes = cpu.read_memory(pc_before, 4)
        inst = int.from_bytes(inst_bytes, 'little')

        # Track SMADDL at 0x425FC0
        if pc_before == SMADDL_1:
            x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            x28 = cpu.get_register(28) & 0xFFFFFFFFFFFFFFFF
            # After SMADDL: x0 = x26 + sign_extend(w0_before) * sign_extend(w28)
            # But we're reading AFTER, so x0 is the RESULT
            entry = {
                'cycle': total_cycles,
                'pc': pc_before,
                'result_x0': x0,
                'x26': x26,
                'x28_w28': x28 & 0xFFFFFFFF,
                'index': (x0 - x26) // 0x38 if x26 > 0 and x0 >= x26 else -1,
            }
            smaddl_results.append(entry)
            print(f"  SMADDL @0x425FC0 cycle {total_cycles}: "
                  f"x0=0x{x0:X} (x26=0x{x26:X} + index*0x{x28 & 0xFFFFFFFF:X})")
            if x26 > 0:
                offset = x0 - x26
                idx = offset // 0x38
                remainder = offset % 0x38
                print(f"    -> entry[{idx}] + 0x{remainder:X} (remainder should be 0)")

        # Track SMADDL at 0x42613C
        if pc_before == SMADDL_2:
            x19 = cpu.get_register(19) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            x28 = cpu.get_register(28) & 0xFFFFFFFFFFFFFFFF
            entry = {
                'cycle': total_cycles,
                'pc': pc_before,
                'result_x19': x19,
                'x26': x26,
                'x28_w28': x28 & 0xFFFFFFFF,
            }
            smaddl_results.append(entry)
            print(f"  SMADDL @0x42613C cycle {total_cycles}: "
                  f"x19=0x{x19:X} (x26=0x{x26:X} + index*0x{x28 & 0xFFFFFFFF:X})")
            if x26 > 0:
                offset = x19 - x26
                idx = offset // 0x38
                remainder = offset % 0x38
                print(f"    -> entry[{idx}] + 0x{remainder:X} (remainder should be 0)")

        # Track terminal write at 0x425F58
        if pc_before == TERMINAL_STR:
            x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
            x1 = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            tw = {
                'cycle': total_cycles,
                'x0_addr': x0,
                'x1_value': x1 & 0xFFFFFFFF,
                'x26': x26,
            }
            terminal_writes.append(tw)
            offset_from_x26 = x0 - x26 if x0 >= x26 else -(x26 - x0)
            entry_idx = offset_from_x26 // 0x38 if offset_from_x26 >= 0 else -1
            field_offset = offset_from_x26 % 0x38 if offset_from_x26 >= 0 else -1
            print(f"  *** TERMINAL WRITE @0x425F58 cycle {total_cycles}: "
                  f"str w1=0x{x1 & 0xFFFFFFFF:08X} to [x0]=0x{x0:X}")
            print(f"      offset from x26: 0x{offset_from_x26:X} = entry[{entry_idx}] + 0x{field_offset:X}")
            print(f"      (should be at offset 0x00 of some entry for code_min)")

            # Verify the actual write happened correctly
            data = cpu.read_memory(x0, 4)
            actual = int.from_bytes(data, 'little')
            print(f"      verify: mem[0x{x0:X}] = 0x{actual:08X} {'OK' if actual == 0xFFFFFFFF else 'MISMATCH!'}")

        # Track function entry at 0x425F20
        if pc_before == FUNC_ENTRY:
            x0 = cpu.get_register(0) & 0xFFFFFFFFFFFFFFFF
            x1 = cpu.get_register(1) & 0xFFFFFFFFFFFFFFFF
            x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            print(f"  FUNC ENTRY @0x425F20 cycle {total_cycles}: "
                  f"x0=0x{x0:X} x1=0x{x1:X} x2=0x{x2:X}")
            if x26 > 0:
                offset = x0 - x26 if x0 >= x26 else -(x26 - x0)
                print(f"    x0 offset from table base (x26=0x{x26:X}): 0x{offset:X}")

        # Track entry into stuck loop
        if pc_before == STUCK_LOOP:
            print(f"\n  ENTERED STUCK LOOP at cycle {total_cycles}")
            x2 = cpu.get_register(2) & 0xFFFFFFFFFFFFFFFF
            x3 = cpu.get_register(3) & 0xFFFFFFFFFFFFFFFF
            x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
            x23 = cpu.get_register(23) & 0xFFFFFFFFFFFFFFFF
            x26 = cpu.get_register(26) & 0xFFFFFFFFFFFFFFFF
            print(f"  x2=0x{x2:X} x3=0x{x3:X}")
            print(f"  x22=0x{x22:X} x23=0x{x23:X}")
            print(f"  x26=0x{x26:X}")
            break

        # If we've left the builder region AND the terminal function region, stop
        if step_count > 100 and not (BUILD_START <= pc_after <= BUILD_END) and \
           not (TERMINAL_FUNC_START <= pc_after <= TERMINAL_FUNC_END):
            if pc_after > 0x426300 or pc_after < 0x425E00:
                print(f"\n  Left builder region at cycle {total_cycles}, PC=0x{pc_after:X}")
                break

    # Summary
    print(f"\n  === SUMMARY ===")
    print(f"  Total SMADDL executions: {len(smaddl_results)}")
    print(f"  Total terminal writes: {len(terminal_writes)}")

    for i, tw in enumerate(terminal_writes):
        x26 = tw['x26']
        offset = tw['x0_addr'] - x26 if tw['x0_addr'] >= x26 else -(x26 - tw['x0_addr'])
        entry_idx = offset // 0x38 if offset >= 0 else -1
        field_offset = offset % 0x38 if offset >= 0 else -1
        print(f"  Terminal write {i}: addr=0x{tw['x0_addr']:X}, "
              f"entry[{entry_idx}]+0x{field_offset:X}, value=0x{tw['x1_value']:08X}")

    # Verify: check the NFA table entries
    if smaddl_results:
        x26 = smaddl_results[0].get('x26', 0)
        if x26 > 0:
            print(f"\n  NFA table verification (base=0x{x26:X}):")
            for i in range(10):
                addr = x26 + i * 0x38
                if addr + 4 < cpu.memory_size:
                    data = cpu.read_memory(addr, 4)
                    code_min = int.from_bytes(data, 'little', signed=True)
                    print(f"    entry[{i}] @ 0x{addr:X}: code_min = {code_min} (0x{code_min & 0xFFFFFFFF:08X})")


if __name__ == "__main__":
    main()
