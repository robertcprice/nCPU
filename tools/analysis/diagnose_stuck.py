#!/usr/bin/env python3
"""Diagnose where busybox execution gets stuck"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
from neural_kernel import NeuralARM64Kernel
import struct
import time

def diagnose():
    """Find where execution gets stuck"""
    print("Loading kernel...")
    kernel = NeuralARM64Kernel()
    cpu = kernel.cpu

    # Load busybox
    binaries_dir = Path(__file__).parent / "binaries"
    busybox_path = binaries_dir / "busybox-static"
    with open(busybox_path, 'rb') as f:
        busybox_binary = f.read()

    print(f"Busybox: {len(busybox_binary):,} bytes")

    # Parse ELF header
    e_entry = struct.unpack('<Q', busybox_binary[24:32])[0]
    e_phoff = struct.unpack('<Q', busybox_binary[32:40])[0]
    e_phentsize = struct.unpack('<H', busybox_binary[54:56])[0]
    e_phnum = struct.unpack('<H', busybox_binary[56:58])[0]

    print(f"Entry: 0x{e_entry:X}")

    PT_LOAD = 1

    # Load segments
    for i in range(e_phnum):
        ph_off = e_phoff + i * e_phentsize
        p_type = struct.unpack('<I', busybox_binary[ph_off:ph_off+4])[0]
        if p_type != PT_LOAD:
            continue
        p_offset = struct.unpack('<Q', busybox_binary[ph_off+8:ph_off+16])[0]
        p_vaddr = struct.unpack('<Q', busybox_binary[ph_off+16:ph_off+24])[0]
        p_filesz = struct.unpack('<Q', busybox_binary[ph_off+32:ph_off+40])[0]
        p_memsz = struct.unpack('<Q', busybox_binary[ph_off+40:ph_off+48])[0]

        segment_data = busybox_binary[p_offset:p_offset+p_filesz]
        if p_vaddr + len(segment_data) <= cpu.mem_size:
            cpu.memory[p_vaddr:p_vaddr+len(segment_data)] = torch.tensor(
                list(segment_data), dtype=torch.uint8, device=cpu.device
            )

        if p_memsz > p_filesz:
            bss_start = p_vaddr + p_filesz
            bss_end = min(p_vaddr + p_memsz, cpu.mem_size)
            cpu.memory[bss_start:bss_end] = 0

    # Setup stack with args
    args = ["echo", "Hello"]
    stack_top = 0xFFF30
    string_base = stack_top + 0x100
    string_off = 0
    arg_addrs = []

    for arg in args:
        arg_addrs.append(string_base + string_off)
        arg_bytes = arg.encode() + b'\x00'
        for b in arg_bytes:
            if string_base + string_off < cpu.mem_size:
                cpu.memory[string_base + string_off] = b
            string_off += 1

    ptr = stack_top
    cpu.memory[ptr:ptr+8] = torch.tensor([len(args), 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8, device=cpu.device)
    ptr += 8
    for addr in arg_addrs:
        addr_bytes = addr.to_bytes(8, 'little')
        cpu.memory[ptr:ptr+8] = torch.tensor(list(addr_bytes), dtype=torch.uint8, device=cpu.device)
        ptr += 8
    cpu.memory[ptr:ptr+8] = 0  # argv NULL
    ptr += 8
    cpu.memory[ptr:ptr+8] = 0  # envp NULL
    ptr += 8
    cpu.memory[ptr:ptr+8] = torch.tensor([6, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8, device=cpu.device)  # AT_PAGESZ
    ptr += 8
    cpu.memory[ptr:ptr+8] = torch.tensor([0, 0x10, 0, 0, 0, 0, 0, 0], dtype=torch.uint8, device=cpu.device)  # 4096
    ptr += 8
    cpu.memory[ptr:ptr+16] = 0  # AT_NULL

    cpu.regs[31] = stack_top
    cpu.pc = torch.tensor(e_entry, dtype=torch.int64, device=cpu.device)

    print(f"SP = 0x{stack_top:X}, PC = 0x{e_entry:X}")
    print()

    # Fast forward to get past startup
    print("Fast forwarding 5000 instructions...")
    executed_t, _ = cpu.run_parallel_gpu(5000)
    executed = int(executed_t.item())
    pc_after_startup = cpu.pc.item()
    print(f"After 5000 inst: PC=0x{pc_after_startup:X}, executed={executed}")
    print()

    # Now step slowly to find the stuck point
    print("Stepping slowly to find stuck point...")
    print("=" * 80)

    pc_history = []
    for step in range(200):
        pc = cpu.pc.item()
        inst_bytes = cpu.memory[pc:pc+4]
        inst = int(inst_bytes[0]) | (int(inst_bytes[1]) << 8) | (int(inst_bytes[2]) << 16) | (int(inst_bytes[3]) << 24)

        pc_history.append(pc)
        if len(pc_history) > 50:
            pc_history.pop(0)

        # Check for repeating pattern
        if len(pc_history) >= 20:
            # Look for cycles of length 5-15
            for cycle_len in range(5, 16):
                if len(pc_history) >= cycle_len * 2:
                    pattern1 = pc_history[-cycle_len:]
                    pattern2 = pc_history[-cycle_len*2:-cycle_len]
                    if pattern1 == pattern2:
                        print(f"\n*** DETECTED REPEATING CYCLE of length {cycle_len} at step {step} ***")
                        print(f"Cycle PCs: {[hex(p) for p in pattern1]}")
                        print()
                        print("Cycle instructions:")
                        for p in pattern1:
                            i_bytes = cpu.memory[p:p+4]
                            i = int(i_bytes[0]) | (int(i_bytes[1]) << 8) | (int(i_bytes[2]) << 16) | (int(i_bytes[3]) << 24)
                            print(f"  0x{p:X}: 0x{i:08X} -> {decode_inst(i)}")
                        print()
                        print("Register state:")
                        for r in range(0, 32, 4):
                            print(f"  X{r:2d}=0x{cpu.regs[r].item():X}  X{r+1:2d}=0x{cpu.regs[r+1].item():X}  X{r+2:2d}=0x{cpu.regs[r+2].item():X}  X{r+3:2d}=0x{cpu.regs[r+3].item():X}")
                        return

        if step < 20 or step % 10 == 0:
            print(f"Step {step:3d}: PC=0x{pc:X} inst=0x{inst:08X} -> {decode_inst(inst)}")

        # Execute one instruction
        cpu.run_parallel_gpu(1)

    print("\nNo repeating cycle detected in 200 steps")

def decode_inst(inst):
    """Simple ARM64 instruction decoder"""
    op = (inst >> 24) & 0xFF

    # Branches
    if (inst & 0xFC000000) == 0x14000000:
        imm26 = inst & 0x3FFFFFF
        if imm26 >= 0x2000000:
            imm26 -= 0x4000000
        return f"B #{imm26*4:+d}"
    if (inst & 0xFC000000) == 0x94000000:
        return "BL"
    if (inst & 0xFF000010) == 0x54000000:
        cond = inst & 0xF
        cond_names = ['EQ','NE','CS','CC','MI','PL','VS','VC','HI','LS','GE','LT','GT','LE','AL','NV']
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= 0x40000:
            imm19 -= 0x80000
        return f"B.{cond_names[cond]} #{imm19*4:+d}"
    if (inst & 0xFF000000) == 0xB4000000:
        rt = inst & 0x1F
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= 0x40000:
            imm19 -= 0x80000
        return f"CBZ X{rt}, #{imm19*4:+d}"
    if (inst & 0xFF000000) == 0xB5000000:
        rt = inst & 0x1F
        imm19 = (inst >> 5) & 0x7FFFF
        if imm19 >= 0x40000:
            imm19 -= 0x80000
        return f"CBNZ X{rt}, #{imm19*4:+d}"
    if (inst & 0xFFFFFC1F) == 0xD65F0000:
        return "RET"
    if (inst & 0xFFFFFFE0) == 0xD61F0000:
        rn = (inst >> 5) & 0x1F
        return f"BR X{rn}"
    if (inst & 0xFFFFFFE0) == 0xD63F0000:
        rn = (inst >> 5) & 0x1F
        return f"BLR X{rn}"

    # Loads
    if (inst & 0xFFC00000) == 0xF9400000:
        return "LDR (imm)"
    if (inst & 0xFFC00000) == 0x39400000:
        return "LDRB (imm)"
    if (inst & 0xFFE00C00) == 0xF8400000:
        return "LDR (reg)"
    if (inst & 0xFFC00000) == 0xA9400000:
        return "LDP"
    if (inst & 0xFFC00000) == 0xA8C00000:
        return "LDP (post)"

    # Stores
    if (inst & 0xFFC00000) == 0xF9000000:
        return "STR (imm)"
    if (inst & 0xFFC00000) == 0x39000000:
        return "STRB (imm)"
    if (inst & 0xFFE00C00) == 0xF8200000:
        return "STR (reg)"
    if (inst & 0xFFC00000) == 0xA9000000:
        return "STP"
    if (inst & 0xFFC00000) == 0xA8800000:
        return "STP (post)"
    if (inst & 0xFFC00000) == 0xA9800000:
        return "STP (pre)"

    # Arithmetic
    if (inst & 0xFF000000) == 0x91000000:
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm = (inst >> 10) & 0xFFF
        return f"ADD X{rd}, X{rn}, #{imm}"
    if (inst & 0xFF000000) == 0xD1000000:
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm = (inst >> 10) & 0xFFF
        return f"SUB X{rd}, X{rn}, #{imm}"
    if (inst & 0xFF200000) == 0xAB000000:
        return "ADDS (reg)"
    if (inst & 0xFF200000) == 0xEB000000:
        return "SUBS (reg)"
    if (inst & 0xFFE0FC00) == 0xEB00001F:
        return "CMP (reg)"
    if (inst & 0xFFC0001F) == 0xF100001F:
        return "CMP (imm)"

    # Moves
    if (inst & 0xFF800000) == 0xD2800000:
        return "MOVZ"
    if (inst & 0xFF800000) == 0xF2800000:
        return "MOVK"
    if (inst & 0xFFE0FFE0) == 0xAA0003E0:
        return "MOV (reg)"
    if (inst & 0x9F800000) == 0x90000000:
        return "ADRP"

    # SVC
    if (inst & 0xFFE0001F) == 0xD4000001:
        return "SVC"

    return f"op=0x{op:02X}"

if __name__ == "__main__":
    diagnose()
