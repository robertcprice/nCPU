#!/usr/bin/env python3
"""Quick test that tracks batches"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from pathlib import Path
from neural_kernel import NeuralARM64Kernel
import struct
import time

print("Quick test starting...")
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

# Setup stack
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
ptr += 16

cpu.regs[31] = stack_top
cpu.pc = torch.tensor(e_entry, dtype=torch.int64, device=cpu.device)

print(f"SP = 0x{stack_top:X}, PC = 0x{e_entry:X}")
print()

# Execute in small batches with progress
print("Running 10 batches of 1000 instructions each...")
total = 0
start = time.perf_counter()

for batch in range(10):
    pc_before = cpu.pc.item()
    executed_t, _ = cpu.run_parallel_gpu(1000)
    executed = int(executed_t.item())
    total += executed
    pc_after = cpu.pc.item()
    print(f"  Batch {batch}: PC 0x{pc_before:X}→0x{pc_after:X}, executed={executed}")

    if pc_before == pc_after:
        inst = int(cpu.memory[pc_after]) | (int(cpu.memory[pc_after+1]) << 8) | (int(cpu.memory[pc_after+2]) << 16) | (int(cpu.memory[pc_after+3]) << 24)
        print(f"    STUCK! inst=0x{inst:08X}, X8=0x{cpu.regs[8].item():X}")
        break

elapsed = time.perf_counter() - start
print(f"\nTotal: {total:,} instructions in {elapsed:.2f}s")
if elapsed > 0:
    print(f"IPS: {total/elapsed:,.0f}")
