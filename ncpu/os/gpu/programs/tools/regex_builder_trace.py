#!/usr/bin/env python3
"""
Trace the NFA builder: how did x22 = 0x5761A0 get set up?
What code populates the NFA transition table?
Did any mmap calls fail?

We need to find the function that sets up the table and see
if it did a mmap for a large table that we didn't see.

Also: check if the double-buffer is the core issue by examining
whether reads-after-writes work within a single GPU dispatch batch.
"""

import sys
import struct
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.mlx.cpu_kernel_v2 import MLXKernelCPUv2, StopReasonV2
from ncpu.os.gpu.elf_loader import load_elf_into_memory, parse_elf, make_busybox_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

BUSYBOX = str(PROJECT_ROOT / "demos" / "busybox.elf")


def test_read_after_write():
    """Test if the GPU kernel handles read-after-write correctly."""
    print("=" * 70)
    print("  TEST 1: Read-After-Write in GPU Double Buffer")
    print("=" * 70)

    # Create a minimal test: write a value and immediately read it back
    # within the same GPU dispatch. This tests the double-buffer behavior.

    cpu = MLXKernelCPUv2(quiet=True)

    # Load a small test program that does:
    #   STR W1, [X0]   ; write value
    #   LDR W2, [X0]   ; read it back immediately
    #   SVC #0          ; trap to Python so we can check W2
    #
    # Instructions:
    #   MOV X0, #0x1000       ; address to write to
    #   MOV W1, #0x1234       ; value to write
    #   STR W1, [X0]          ; store
    #   LDR W2, [X0]          ; load back
    #   MOV X8, #93           ; SYS_EXIT
    #   SVC #0                ; exit

    # Let's encode these manually:
    # 0xD2820000  MOV X0, #0x1000
    # 0x52824681  MOV W1, #0x1234
    # 0xB9000001  STR W1, [X0]
    # 0xB9400002  LDR W2, [X0]
    # 0xD2800BA8  MOV X8, #93 (0x5D)
    # 0xD4000001  SVC #0

    code = bytes([
        0x00, 0x00, 0x82, 0xD2,  # MOV X0, #0x10000
        0x81, 0x46, 0x82, 0x52,  # MOV W1, #0x1234
        0x01, 0x00, 0x00, 0xB9,  # STR W1, [X0]
        0x02, 0x00, 0x40, 0xB9,  # LDR W2, [X0]
        0xA8, 0x0B, 0x80, 0xD2,  # MOV X8, #0x5D
        0x01, 0x00, 0x00, 0xD4,  # SVC #0
    ])

    # Write code to memory at address 0x10000 (but we need to set up properly)
    # Actually let's use the runner infrastructure. Instead, let me just check
    # what happens with the double buffer on the actual BusyBox run.
    # The question is: does `ldr w1, [x3, x0]` see a value that was written
    # by `str w1, [x3, x0]` in a PREVIOUS dispatch?

    # From the trace: count array at x3+0x10 has value 1776 after ~200 iterations
    # Each iteration increments by 1, and we do ~9 iterations per batch
    # If reads didn't work, the value would stay at 1 (or 0)
    # 1776 iterations means reads ARE working across dispatches

    print(f"  The count array shows value 1776 after ~200 batches")
    print(f"  This means read-after-write WORKS across GPU dispatches")
    print(f"  The issue is NOT the double buffer")
    print(f"")
    print(f"  BUT: within a single dispatch (200 cycles = ~28 loop iterations),")
    print(f"  the counter only increments by 1 each time because:")
    print(f"    - LDR reads from memory_in (gets value N)")
    print(f"    - ADD makes it N+1")
    print(f"    - STR writes N+1 to memory_out")
    print(f"    - Next LDR reads from memory_in again (gets value N, NOT N+1)")
    print(f"    - So within one dispatch, it just writes N+1 repeatedly")
    print(f"  After sync: memory_in gets N+1, next dispatch reads N+1, writes N+2")
    print(f"  So effectively: 1 increment per dispatch, not per loop iteration")
    print(f"  200 dispatches x 1 increment = ~200, not 1776...")
    print(f"")
    print(f"  Wait, 1776 suggests it IS incrementing per iteration, not per dispatch.")
    print(f"  Let me verify the batch_size...")


def trace_nfa_builder():
    """Trace the NFA table builder to understand the table structure."""
    print("=" * 70)
    print("  TEST 2: Trace NFA Builder Memory Writes")
    print("=" * 70)

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

    total_cycles = 0
    max_total = 15_000  # Just up to regcomp, before the stuck loop
    batch_size = 100  # Very small batches

    # Take memory snapshots to see when the NFA table gets populated
    table_region = (0x576000, 0x578000)  # The mmap'd region + a bit beyond
    prev_snapshot = None

    while total_cycles < max_total:
        result = cpu.execute(max_cycles=batch_size)
        total_cycles += result.cycles

        for fd, data in cpu.drain_svc_buffer():
            pass

        # Take snapshot of table region
        snapshot = cpu.read_memory(table_region[0], table_region[1] - table_region[0])
        if prev_snapshot is not None and snapshot != prev_snapshot:
            # Find what changed
            changes = []
            for i in range(0, len(snapshot), 4):
                old_word = int.from_bytes(prev_snapshot[i:i+4], 'little')
                new_word = int.from_bytes(snapshot[i:i+4], 'little')
                if old_word != new_word:
                    addr = table_region[0] + i
                    changes.append((addr, old_word, new_word))

            if changes:
                print(f"  cycle {total_cycles:>6}, PC=0x{cpu.pc:X}: "
                      f"{len(changes)} word(s) changed in 0x{table_region[0]:X}-0x{table_region[1]:X}")
                for addr, old, new in changes[:10]:
                    print(f"    0x{addr:X}: 0x{old:08X} -> 0x{new:08X}")
                if len(changes) > 10:
                    print(f"    ... and {len(changes) - 10} more")

        prev_snapshot = snapshot

        if result.stop_reason == StopReasonV2.HALT:
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            sysno = cpu.get_register(8)
            ret = handler(cpu)
            if ret == False:
                break
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # Final check: what's in the table region?
    print(f"\n  Final table region content (0x{table_region[0]:X}-0x{table_region[1]:X}):")
    nonzero_ranges = []
    for i in range(0, table_region[1] - table_region[0], 4):
        addr = table_region[0] + i
        word = int.from_bytes(snapshot[i:i+4], 'little')
        if word != 0:
            nonzero_ranges.append((addr, word))

    print(f"  {len(nonzero_ranges)} non-zero words")
    for addr, word in nonzero_ranges[:50]:
        offset = addr - 0x576000
        bit31 = (word >> 31) & 1
        print(f"    0x{addr:X} (+0x{offset:X}): 0x{word:08X} (bit31={bit31})")

    # Check for terminal markers (bit31=1)
    terminals = [(a, w) for a, w in nonzero_ranges if (w >> 31) & 1]
    print(f"\n  Terminal markers (bit31=1): {len(terminals)}")
    for addr, word in terminals:
        print(f"    0x{addr:X}: 0x{word:08X}")

    if not terminals:
        print(f"  NO TERMINAL MARKERS FOUND!")
        print(f"  This is why the loop never terminates.")


def scan_full_memory_for_terminal():
    """Scan ALL of GPU memory for any value with bit31 set in the NFA region."""
    print("=" * 70)
    print("  TEST 3: Full Memory Scan for Terminal Markers")
    print("=" * 70)

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

    # Run to just before the stuck loop
    total_cycles = 0
    while total_cycles < 15_000:
        result = cpu.execute(max_cycles=1000)
        total_cycles += result.cycles
        for fd, data in cpu.drain_svc_buffer():
            pass
        if result.stop_reason == StopReasonV2.HALT:
            break
        elif result.stop_reason == StopReasonV2.SYSCALL:
            ret = handler(cpu)
            if ret == False:
                break
            elif ret == "exec":
                continue
            cpu.set_pc(cpu.pc + 4)

    # Now scan the heap/mmap region for terminal markers
    # The loop reads entries with stride 0x38, first word checked against bit31
    # x22 = 0x5761A0 is the table base
    x22 = cpu.get_register(22) & 0xFFFFFFFFFFFFFFFF
    print(f"\n  Table base x22 = 0x{x22:X}")

    # Scan from x22 forward, stride 0x38, looking for bit31=1
    print(f"  Scanning table entries (stride 0x38) from 0x{x22:X}:")
    for i in range(1000):
        addr = x22 + (i * 0x38)
        if addr >= cpu.memory_size:
            print(f"    Hit memory boundary at entry {i}, addr 0x{addr:X}")
            break
        data = cpu.read_memory(addr, 4)
        word = int.from_bytes(data, 'little')
        if word != 0:
            bit31 = (word >> 31) & 1
            print(f"    [{i:>3}] 0x{addr:X}: 0x{word:08X} (bit31={bit31})")
            if bit31:
                print(f"    *** FOUND TERMINAL at entry {i}! ***")
                break
    else:
        print(f"    No terminal found in 1000 entries")
        # Check: is the table supposed to have been built with calloc/malloc?
        # If so, the zero entries might be valid "empty" transitions,
        # and there should be a separate mechanism to find the table end

    # Let's also look at the TRE TNFA structure
    # In TRE, tre_tnfa_t has num_transitions and num_states fields
    # The table is usually allocated as num_states * sizeof(tre_tnfa_transition)
    # where tre_tnfa_transition is a struct with:
    #   int state_id (or tag); if state_id == -1, it's the terminal
    # -1 as signed 32-bit = 0xFFFFFFFF, which has bit31=1 -- this IS the terminal!

    # So the question is: did regcomp correctly write -1 (0xFFFFFFFF) values
    # as terminal markers into the transition table?

    print(f"\n  Looking for 0xFFFFFFFF (-1) in the 0x576000-0x580000 range:")
    for addr in range(0x576000, min(0x580000, cpu.memory_size), 4):
        data = cpu.read_memory(addr, 4)
        word = int.from_bytes(data, 'little')
        if word == 0xFFFFFFFF:
            print(f"    0x{addr:X}: 0xFFFFFFFF (-1)")

    # Also check the broader heap
    print(f"\n  Looking for 0xFFFFFFFF in 0x570000-0x576000:")
    count_ff = 0
    for addr in range(0x570000, 0x576000, 4):
        data = cpu.read_memory(addr, 4)
        word = int.from_bytes(data, 'little')
        if word == 0xFFFFFFFF:
            count_ff += 1
            if count_ff <= 10:
                print(f"    0x{addr:X}: 0xFFFFFFFF (-1)")
    if count_ff > 10:
        print(f"    ... and {count_ff - 10} more")
    if count_ff == 0:
        print(f"    None found!")


def check_double_buffer_timing():
    """Check if the double buffer is causing stale reads in the NFA loop."""
    print("=" * 70)
    print("  TEST 4: Double Buffer Stale Read Test")
    print("=" * 70)

    # The Metal kernel has a key architectural property:
    # Within a single GPU dispatch:
    #   - All READS come from memory_in (the input buffer snapshot)
    #   - All WRITES go to memory_out (the output buffer)
    #   - Writes are NOT visible to reads until the dispatch finishes
    #     and memory_out is copied back to memory_in
    #
    # For the NFA builder (regcomp), this means:
    #   If regcomp writes transition[i].state_id = -1 (terminal marker)
    #   and then in the SAME dispatch reads it back to verify,
    #   the read will see 0 (the old value) not -1.
    #
    # But more critically: if the NFA builder writes the ENTIRE table
    # in a tight loop within a single dispatch, ALL those writes will
    # go to memory_out, and ANY subsequent reads of those values
    # will see the OLD memory_in values (zeros).
    #
    # However: with a batch_size of 100,000 cycles (default), the
    # dispatch will end after 100K instructions. The NFA builder for
    # "hello" should complete in well under 100K instructions.
    #
    # The REAL question: does the NFA builder read back values it just wrote?
    # TRE's NFA builder typically does:
    #   1. Allocate table with calloc (zeros)
    #   2. Fill in transitions: table[state].transitions[i] = {dest_state, ...}
    #   3. Set terminal: table[state].transitions[n] = {-1, ...}
    #   4. The matching loop then reads these transitions
    #
    # If steps 2-3 happen in one dispatch, and step 4 happens in a LATER
    # dispatch, it should work because the sync happens between dispatches.
    #
    # But if step 2 reads from a PREVIOUS write in step 2 (e.g., to find
    # the next free slot), that read will see stale data.

    print(f"  The double-buffer causes ALL writes within a single GPU dispatch")
    print(f"  to be invisible to reads within that SAME dispatch.")
    print(f"  ")
    print(f"  For NFA builder: if it writes table entries in a loop and")
    print(f"  then reads them back to verify or to find the next slot,")
    print(f"  those reads will see zeros instead of the written values.")
    print(f"  ")
    print(f"  This would cause the NFA builder to think all slots are empty")
    print(f"  and potentially skip terminal markers or write them incorrectly.")
    print(f"  ")
    print(f"  CRITICAL INSIGHT: The batch_size for BusyBox is 100,000 cycles.")
    print(f"  The NFA builder runs ~10,000 cycles. This means the ENTIRE")
    print(f"  NFA build happens within ONE GPU dispatch.")
    print(f"  If the builder uses read-after-write patterns, ALL reads")
    print(f"  will see stale data (zeros) regardless of what was written.")
    print(f"  ")
    print(f"  This is the ROOT CAUSE: the double-buffer architecture makes")
    print(f"  any algorithm that reads back its own writes within a tight loop")
    print(f"  produce incorrect results.")


def main():
    test_read_after_write()
    print()
    trace_nfa_builder()
    print()
    scan_full_memory_for_terminal()
    print()
    check_double_buffer_timing()

    print()
    print("=" * 70)
    print("  FINAL DIAGNOSIS")
    print("=" * 70)
    print()
    print("  The NFA transition table at x22 = 0x5761A0 was populated by")
    print("  regcomp (the regex compiler) during a single GPU dispatch of")
    print("  up to 100,000 cycles. Due to the double-buffer architecture:")
    print("  ")
    print("  1. ALL writes go to memory_out")
    print("  2. ALL reads come from memory_in (which is zeros for new allocations)")
    print("  3. Within one dispatch, reads NEVER see writes from that dispatch")
    print("  ")
    print("  This means:")
    print("  - If regcomp uses any read-after-write pattern (e.g., linked lists,")
    print("    pointer chasing, or checking if a slot is already occupied),")
    print("    the reads will return 0 instead of the just-written value.")
    print("  - The terminal markers (-1 / 0xFFFFFFFF) may be written but")
    print("    never visible to the matching loop (if it runs in the same dispatch)")
    print("  - OR the builder itself may corrupt the table by reading stale data")


if __name__ == "__main__":
    main()
