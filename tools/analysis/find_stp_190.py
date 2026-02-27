#!/usr/bin/env python3
"""Find the STP instruction that stores to SP+0x190."""
import sys
sys.path.insert(0, '.')
from neural_kernel import NeuralARM64Kernel
from neural_cpu import device, OpType
import torch

kernel = NeuralARM64Kernel()

with open('binaries/busybox-static', 'rb') as f:
    busybox = f.read()

load_result = kernel.load_elf(busybox)

# Disassemble 0x4500-0x45D4 looking for STP to SP+0x190
print("=== Looking for STP/STR to SP+0x190 in startup code ===")

def decode_store(addr, inst):
    op = (inst >> 24) & 0xFF
    
    # STP signed offset (0xA9)
    if op == 0xA9:
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 & 0x40: imm7 -= 0x80
        load = (inst >> 22) & 1
        rn_name = 'SP' if rn == 31 else f'X{rn}'
        op_name = 'LDP' if load else 'STP'
        offset = imm7 * 8
        return f"{op_name} X{rt}, X{rt2}, [{rn_name}, #0x{offset:X}]", offset, rn, load
    # STP pre-index (0xA9, addr_mode=3)
    elif (inst & 0xFFC00000) == 0xA9800000:  # STP pre-index
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        rt2 = (inst >> 10) & 0x1F
        imm7 = (inst >> 15) & 0x7F
        if imm7 & 0x40: imm7 -= 0x80
        rn_name = 'SP' if rn == 31 else f'X{rn}'
        offset = imm7 * 8
        return f"STP X{rt}, X{rt2}, [{rn_name}, #0x{offset:X}]!", offset, rn, 0
    # STR unsigned offset
    elif op == 0xF9:
        rt = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        imm12 = (inst >> 10) & 0xFFF
        load = (inst >> 22) & 1
        rn_name = 'SP' if rn == 31 else f'X{rn}'
        op_name = 'LDR' if load else 'STR'
        offset = imm12 * 8
        return f"{op_name} X{rt}, [{rn_name}, #0x{offset:X}]", offset, rn, load
    return None, None, None, None

for addr in range(0x4500, 0x4700, 4):
    inst = kernel.cpu.read32(addr)
    result = decode_store(addr, inst)
    if result[0]:
        desc, offset, rn, is_load = result
        # Look for stores to SP with offset 0x190
        if rn == 31 and offset >= 0x180 and offset <= 0x1A0 and not is_load:
            print(f"  *** 0x{addr:05X}: 0x{inst:08X}  {desc} <-- FOUND!")
        elif rn == 31 and offset >= 0x180 and offset <= 0x1A0:
            print(f"      0x{addr:05X}: 0x{inst:08X}  {desc}")

# Let's also look at what instruction SHOULD store to SP+0x190
# Trace execution and look for stores
print("\n=== Tracing stores to SP+0x180-0x1A0 (first 200 instructions) ===")

# Set up CPU
argv = ['busybox', 'true']
string_area = 0x100000 - 0x1000
arg_ptrs = []
str_ptr = string_area
for arg in argv:
    arg_bytes = arg.encode('utf-8') + b'\x00'
    for i, b in enumerate(arg_bytes):
        kernel.cpu.memory[str_ptr + i] = b
    arg_ptrs.append(str_ptr)
    str_ptr += len(arg_bytes)

argc = len(argv)
auxv_entries = [(3, 0x40), (4, 56), (5, 7), (6, 4096), (7, 0), (9, load_result['entry_point']), (0, 0)]
total_size = 8 + (argc + 1) * 8 + 8 + len(auxv_entries) * 16
sp = (string_area - total_size) & ~0xF

write_pos = sp
argc_bytes = list(argc.to_bytes(8, 'little'))
kernel.cpu.memory[write_pos:write_pos+8] = torch.tensor(argc_bytes, dtype=torch.uint8, device=device)
write_pos += 8
for ptr in arg_ptrs:
    ptr_bytes = list(ptr.to_bytes(8, 'little'))
    kernel.cpu.memory[write_pos:write_pos+8] = torch.tensor(ptr_bytes, dtype=torch.uint8, device=device)
    write_pos += 8
kernel.cpu.memory[write_pos:write_pos+8] = torch.tensor([0]*8, dtype=torch.uint8, device=device)
write_pos += 8
kernel.cpu.memory[write_pos:write_pos+8] = torch.tensor([0]*8, dtype=torch.uint8, device=device)
write_pos += 8
for a_type, a_val in auxv_entries:
    type_bytes = list(a_type.to_bytes(8, 'little'))
    val_bytes = list(a_val.to_bytes(8, 'little'))
    kernel.cpu.memory[write_pos:write_pos+8] = torch.tensor(type_bytes, dtype=torch.uint8, device=device)
    kernel.cpu.memory[write_pos+8:write_pos+16] = torch.tensor(val_bytes, dtype=torch.uint8, device=device)
    write_pos += 16

kernel.cpu.pc = torch.tensor(load_result['entry_point'], dtype=torch.int64, device=device)
kernel.cpu.regs.zero_()
kernel.cpu.regs[31] = sp
kernel.cpu.halted = False

print(f"SP = 0x{sp:X}")
print(f"SP+0x190 = 0x{sp + 0x190:X}")

# Track memory writes to the region around SP+0x190
target_low = sp + 0x180
target_high = sp + 0x1A0

for i in range(200):
    pc = int(kernel.cpu.pc.item())
    inst = kernel.cpu.read32(pc)
    current_sp = int(kernel.cpu.regs[31].item())
    
    # Check for STP/STR instructions
    result = decode_store(pc, inst)
    desc, offset, rn, is_load = result
    
    if desc and not is_load:
        # It's a store instruction
        if rn == 31:  # Store relative to SP
            target_addr = current_sp + offset
            if target_low <= target_addr <= target_high:
                # Get the values being stored
                rt = inst & 0x1F
                rt2 = (inst >> 10) & 0x1F if (inst >> 24) & 0xFF in (0xA8, 0xA9) else None
                rt_val = int(kernel.cpu.regs[rt].item()) if rt != 31 else current_sp
                rt_val &= 0xFFFFFFFFFFFFFFFF
                
                if rt2 is not None:
                    rt2_val = int(kernel.cpu.regs[rt2].item()) if rt2 != 31 else current_sp
                    rt2_val &= 0xFFFFFFFFFFFFFFFF
                    print(f"[{i:3}] 0x{pc:05X}: {desc} -> addr 0x{target_addr:X} vals: X{rt}=0x{rt_val:X}, X{rt2}=0x{rt2_val:X}")
                else:
                    print(f"[{i:3}] 0x{pc:05X}: {desc} -> addr 0x{target_addr:X} val: X{rt}=0x{rt_val:X}")
    
    kernel.cpu.run(1)
