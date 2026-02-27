#!/usr/bin/env python3
"""Simple test to debug syscall execution"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
print("Loading kernel...", flush=True)

from neural_kernel import NeuralARM64Kernel
kernel = NeuralARM64Kernel()
print("Kernel loaded!", flush=True)

with open('binaries/busybox-static', 'rb') as f:
    busybox_binary = f.read()
print(f"Binary: {len(busybox_binary):,} bytes", flush=True)

device = kernel.cpu.device
kernel.cpu.pc = torch.tensor(0, dtype=torch.int64, device=device)
kernel.cpu.regs.zero_()
load_result = kernel.load_elf(busybox_binary)
print(f"Entry: 0x{load_result['entry_point']:X}", flush=True)

kernel.cpu.pc = torch.tensor(load_result['entry_point'], dtype=torch.int64, device=device)

# Setup stack with proper AUXV
argv = ["echo", "test"]
env = {}
stack_base = 0x100000
string_area = 0x100100

# Write argument strings
arg_ptrs = []
str_ptr = string_area
for arg in argv:
    arg_bytes = arg.encode('utf-8') + b'\x00'
    kernel.cpu.memory[str_ptr:str_ptr + len(arg_bytes)] = torch.tensor(list(arg_bytes), dtype=torch.uint8, device=device)
    arg_ptrs.append(str_ptr)
    str_ptr += len(arg_bytes)

# Auxiliary vector - including AT_RANDOM properly
random_ptr = 0x100300
kernel.cpu.memory[random_ptr:random_ptr+16] = torch.randint(0, 256, (16,), dtype=torch.uint8, device=device)

auxv_entries = [
    (25, random_ptr),     # AT_RANDOM
    (6, 4096),            # AT_PAGESZ
    (9, load_result['entry_point']),  # AT_ENTRY
    (23, 0),              # AT_SECURE
    (0, 0),               # AT_NULL
]

frame_size = 8 + len(arg_ptrs) * 8 + 8 + 8 + len(auxv_entries) * 16
frame_size = (frame_size + 15) & ~0xF
sp = stack_base - frame_size

# Zero stack region
kernel.cpu.memory[0xFE000:stack_base] = torch.zeros(stack_base - 0xFE000, dtype=torch.uint8, device=device)

# Write stack frame
write_ptr = sp
kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(list(len(argv).to_bytes(8, 'little')), dtype=torch.uint8, device=device)
write_ptr += 8

for ptr in arg_ptrs:
    kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(list(ptr.to_bytes(8, 'little')), dtype=torch.uint8, device=device)
    write_ptr += 8

kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor([0]*8, dtype=torch.uint8, device=device)
write_ptr += 8
kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor([0]*8, dtype=torch.uint8, device=device)  # envp null
write_ptr += 8

for a_type, a_val in auxv_entries:
    kernel.cpu.memory[write_ptr:write_ptr+8] = torch.tensor(list(a_type.to_bytes(8, 'little')), dtype=torch.uint8, device=device)
    kernel.cpu.memory[write_ptr+8:write_ptr+16] = torch.tensor(list(a_val.to_bytes(8, 'little')), dtype=torch.uint8, device=device)
    write_ptr += 16

kernel.cpu.regs[31] = sp
print(f"SP: 0x{sp:X}", flush=True)

# Run in batches, checking for SVC
total = 0
for batch in range(2000):
    pc_before = int(kernel.cpu.pc.item())
    executed_t, _ = kernel.cpu.run_parallel_gpu(max_instructions=32768, batch_size=32768)
    executed = int(executed_t.item())
    total += executed

    if batch % 100 == 0:
        print(f"Batch {batch}: total={total:,}, PC=0x{int(kernel.cpu.pc.item()):X}", flush=True)

    if kernel.cpu._svc_t.item():
        pc = int(kernel.cpu.pc.item())
        x8 = int(kernel.cpu.regs[8].item())
        x0 = int(kernel.cpu.regs[0].item())
        x1 = int(kernel.cpu.regs[1].item())
        x2 = int(kernel.cpu.regs[2].item())
        print(f">>> SVC: syscall={x8}, x0=0x{x0:X}, x1=0x{x1:X}, x2={x2}", flush=True)
        break

    if executed == 0 or kernel.cpu.halted:
        print(f"Stopped: executed={executed}, halted={kernel.cpu.halted}", flush=True)
        break

print(f"Final: {total:,} instructions, PC=0x{int(kernel.cpu.pc.item()):X}", flush=True)
