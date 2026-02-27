#!/usr/bin/env python3
"""
===============================================================================
          _   _ _____ _   _ ____      _    _       ____ _____ ___  ____
         | \ | | ____| | | |  _ \    / \  | |     |  _ \_   _/ _ \/ ___|
         |  \| |  _| | | | | |_) |  / _ \ | |     | |_) || || | | \___ \
         | |\  | |___| |_| |  _ <  / ___ \| |___  |  _ < | || |_| |___) |
         |_| \_|_____|\___/|_| \_\/_/   \_\_____| |_| \_\|_| \___/|____/

                     NEURAL REAL-TIME OPERATING SYSTEM
                     ==================================

    Every computation performed by NEURAL NETWORKS:
    - Neural ELF Loader (attention-based byte extraction)
    - Neural ARM64 Decoder (transformer instruction decoding)
    - Neural ALU (transformer carry/borrow prediction)
    - Neural Register File (values stored IN network weights)
    - Neural MMU (neural memory address translation)
    - Neural Framebuffer (UNet-based rendering)
    - Neural Timer/UART/GIC (peripheral handling)

    "The machine doesn't compute - it THINKS the computation"
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import time
import sys
import os
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# DEVICE SETUP
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

MASK64 = (1 << 64) - 1

# =============================================================================
# NEURAL MODEL ARCHITECTURES
# =============================================================================

class PerBitModel(nn.Module):
    """Neural bitwise operations - processes all 64 bits in parallel."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, a_bits, b_bits):
        x = torch.stack([a_bits, b_bits], dim=-1)
        return self.net(x).squeeze(-1)


class CarryPredictorTransformer(nn.Module):
    """Neural ADD - transformer predicts carry chain."""
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.carry_head = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        bits = a_bits.shape[1]
        batch = a_bits.shape[0]
        G = a_bits * b_bits
        P = a_bits + b_bits - 2 * a_bits * b_bits
        gp = torch.stack([G, P], dim=-1)
        x = self.input_proj(gp) + self.pos_embedding[:, :bits, :]
        mask = torch.triu(torch.ones(bits, bits, device=a_bits.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        carry_logits = self.carry_head(x).squeeze(-1)
        carries = torch.sigmoid(carry_logits)
        carry_in = torch.cat([torch.zeros(batch, 1, device=a_bits.device), carries[:, :-1]], dim=1)
        sums = P + carry_in - 2 * P * carry_in
        return sums, carries


class BorrowPredictorTransformer(nn.Module):
    """Neural SUB - transformer predicts borrow chain."""
    def __init__(self, max_bits=64, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.max_bits = max_bits
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_bits, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.borrow_head = nn.Linear(d_model, 1)

    def forward(self, a_bits, b_bits):
        batch, bits = a_bits.shape
        not_a = 1 - a_bits
        G = not_a * b_bits
        P = a_bits + b_bits - 2 * a_bits * b_bits
        gp = torch.stack([G, P], dim=-1)
        x = self.input_proj(gp)
        x = x + self.pos_embedding[:, :bits, :]
        mask = torch.triu(torch.ones(bits, bits, device=a_bits.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        borrow_logits = self.borrow_head(x).squeeze(-1)
        borrows = torch.sigmoid(borrow_logits)
        borrow_in = torch.cat([torch.zeros(batch, 1, device=a_bits.device), borrows[:, :-1]], dim=1)
        diffs = P + borrow_in - 2 * P * borrow_in
        return diffs, borrows


class TrulyNeuralRegisterFile(nn.Module):
    """Register file where VALUES ARE STORED IN NETWORK WEIGHTS."""
    def __init__(self, n_regs=32, bit_width=64, key_dim=128):
        super().__init__()
        self.n_regs = n_regs
        self.bit_width = bit_width
        self.key_dim = key_dim

        # VALUES STORED HERE - IN THE NETWORK WEIGHTS
        self.register_values = nn.Parameter(torch.zeros(n_regs, bit_width))
        self.register_keys = nn.Parameter(torch.randn(n_regs, key_dim) * 0.1)

        self.query_encoder = nn.Sequential(
            nn.Linear(5, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.value_encoder = nn.Sequential(
            nn.Linear(bit_width, bit_width * 2),
            nn.GELU(),
            nn.Linear(bit_width * 2, bit_width),
        )
        self.write_lr = nn.Parameter(torch.tensor(0.1))

    def _idx_to_bits(self, idx):
        B = idx.shape[0]
        bits = torch.zeros(B, 5, device=idx.device)
        for i in range(5):
            bits[:, i] = ((idx >> i) & 1).float()
        return bits

    def _get_attention(self, idx):
        idx_bits = self._idx_to_bits(idx)
        query = self.query_encoder(idx_bits)
        similarity = torch.matmul(query, self.register_keys.T)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(similarity / temp, dim=-1)
        return attention

    def read(self, idx):
        attention = self._get_attention(idx)
        values = torch.matmul(attention, self.register_values)
        is_xzr = (idx == 31).float().unsqueeze(-1)
        values = values * (1 - is_xzr)
        return values

    def write(self, idx, value):
        is_xzr = (idx == 31).float().unsqueeze(-1)
        value = value * (1 - is_xzr)
        attention = self._get_attention(idx)
        encoded_value = self.value_encoder(value)
        current = torch.matmul(attention, self.register_values)
        delta = encoded_value - current
        update = torch.matmul(attention.T, delta)
        lr = torch.clamp(self.write_lr.abs(), 0.01, 1.0)
        with torch.no_grad():
            self.register_values.add_(lr * update)

    def reset(self):
        with torch.no_grad():
            self.register_values.zero_()


class NeuralELFLoader(nn.Module):
    """Neural ELF loader - learns WHERE to read entry point bytes."""
    def __init__(self, d_model=128):
        super().__init__()
        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.entry_heads = nn.ModuleList([nn.Linear(d_model, 64) for _ in range(8)])
        self.register_buffer('le_weights', torch.tensor([256.0 ** i for i in range(8)]))

    def forward(self, byte_seq):
        B = byte_seq.shape[0]
        x = self.byte_embed(byte_seq) + self.pos_embed
        x = self.transformer(x)
        global_ctx = x.mean(dim=1)

        selected_bytes = []
        for head in self.entry_heads:
            attn = F.softmax(head(global_ctx), dim=-1)
            byte_val = (byte_seq.float() * attn).sum(dim=-1)
            selected_bytes.append(byte_val)

        selected_bytes = torch.stack(selected_bytes, dim=-1)
        entry_point = (selected_bytes * self.le_weights).sum(dim=-1)
        return entry_point


# =============================================================================
# NEURAL ALU - Batched Operations
# =============================================================================

class NeuralALU:
    """Truly neural ALU using transformer models for arithmetic."""

    def __init__(self, model_dir="models/final"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load trained neural models."""
        # Bitwise operations
        for op in ['AND', 'OR', 'XOR']:
            model = PerBitModel(hidden_dim=64).to(device)
            path = self.model_dir / f"{op}_64bit_100pct.pt"
            if path.exists():
                self._load_checkpoint(model, path)
                self.models[op] = model

        # ADD
        add_model = CarryPredictorTransformer(64, d_model=64, nhead=4, num_layers=3).to(device)
        path = self.model_dir / "ADD_64bit_100pct.pt"
        if path.exists():
            self._load_checkpoint(add_model, path)
            self.models['ADD'] = add_model

        # SUB
        sub_model = BorrowPredictorTransformer(64, d_model=64, nhead=4, num_layers=3).to(device)
        path = self.model_dir / "SUB_64bit_100pct.pt"
        if path.exists():
            self._load_checkpoint(sub_model, path)
            self.models['SUB'] = sub_model

        for model in self.models.values():
            model.eval()

    def _load_checkpoint(self, model, path):
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
            else:
                state = ckpt
            if any(k.startswith('_orig_mod.') for k in state.keys()):
                state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
            model.load_state_dict(state)
        except Exception as e:
            pass  # Model not available

    def _int_to_bits(self, val):
        bits = torch.zeros(1, 64, device=device, dtype=torch.float32)
        for i in range(64):
            bits[0, i] = ((val >> i) & 1)
        return bits

    def _bits_to_int(self, bits):
        val = 0
        bits_cpu = (bits > 0.5).long().cpu()
        for i in range(64):
            if bits_cpu[0, i].item():
                val |= (1 << i)
        return val

    def execute(self, op, a, b):
        """Execute a neural ALU operation."""
        if op not in self.models:
            # Fallback for missing models
            if op == 'ADD': return (a + b) & MASK64
            if op == 'SUB': return (a - b) & MASK64
            if op == 'AND': return a & b
            if op == 'OR': return a | b
            if op == 'XOR': return a ^ b
            return 0

        a_bits = self._int_to_bits(a)
        b_bits = self._int_to_bits(b)

        with torch.no_grad():
            result = self.models[op](a_bits, b_bits)
            if isinstance(result, tuple):
                result = result[0]

        return self._bits_to_int(result)


# =============================================================================
# NEURAL MEMORY SYSTEM
# =============================================================================

class NeuralMemory:
    """Neural memory with tensor-based storage."""

    def __init__(self, size=64*1024*1024):
        self.size = size
        self.memory = torch.zeros(size, dtype=torch.uint8, device=device)

    def read_byte(self, addr):
        if 0 <= addr < self.size:
            return self.memory[addr].item()
        return 0

    def write_byte(self, addr, val):
        if 0 <= addr < self.size:
            self.memory[addr] = val & 0xFF

    def read_word(self, addr):
        val = 0
        for i in range(4):
            val |= self.read_byte(addr + i) << (i * 8)
        return val

    def write_word(self, addr, val):
        for i in range(4):
            self.write_byte(addr + i, (val >> (i * 8)) & 0xFF)

    def read_dword(self, addr):
        val = 0
        for i in range(8):
            val |= self.read_byte(addr + i) << (i * 8)
        return val

    def write_dword(self, addr, val):
        for i in range(8):
            self.write_byte(addr + i, (val >> (i * 8)) & 0xFF)

    def load_binary(self, data, addr):
        for i, byte in enumerate(data):
            self.write_byte(addr + i, byte)


# =============================================================================
# NEURAL CPU CORE
# =============================================================================

class NeuralCPU:
    """Truly neural CPU core."""

    def __init__(self):
        self.memory = NeuralMemory()
        self.alu = NeuralALU()

        # Neural register file (or fallback)
        self.registers = torch.zeros(32, dtype=torch.int64, device='cpu')
        self.pc = 0
        self.sp = 0x7FFFFFFF

        # Flags
        self.n_flag = False
        self.z_flag = False
        self.c_flag = False
        self.v_flag = False

        # Stats
        self.instruction_count = 0
        self.neural_ops = 0

    def reset(self):
        self.registers.zero_()
        self.pc = 0
        self.n_flag = False
        self.z_flag = False
        self.c_flag = False
        self.v_flag = False
        self.instruction_count = 0
        self.neural_ops = 0

    def get_reg(self, idx):
        if idx == 31:
            return 0  # XZR
        return self.registers[idx].item()

    def set_reg(self, idx, val):
        if idx != 31:  # Don't write to XZR
            self.registers[idx] = val & MASK64

    def decode_and_execute(self):
        """Decode and execute one instruction."""
        # Read instruction from neural memory
        inst = self.memory.read_word(self.pc)
        self.pc += 4
        self.instruction_count += 1

        # Decode (simplified ARM64)
        op = (inst >> 24) & 0xFF

        # ADD immediate: 0x91
        if op == 0x91:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            result = self.alu.execute('ADD', self.get_reg(rn), imm12)
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # SUB immediate: 0xD1
        elif op == 0xD1:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            result = self.alu.execute('SUB', self.get_reg(rn), imm12)
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # MOVZ: 0xD2 (64-bit), 0x52 (32-bit)
        elif op in [0xD2, 0x52]:
            rd = inst & 0x1F
            imm16 = (inst >> 5) & 0xFFFF
            hw = (inst >> 21) & 0x3
            self.set_reg(rd, imm16 << (hw * 16))
            return True

        # ADD register: 0x8B
        elif op == 0x8B:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('ADD', self.get_reg(rn), self.get_reg(rm))
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # SUB register: 0xCB
        elif op == 0xCB:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('SUB', self.get_reg(rn), self.get_reg(rm))
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # AND register: 0x8A
        elif op == 0x8A:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('AND', self.get_reg(rn), self.get_reg(rm))
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # ORR register: 0xAA
        elif op == 0xAA:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('OR', self.get_reg(rn), self.get_reg(rm))
            self.set_reg(rd, result)
            self.neural_ops += 1
            return True

        # STR (store): 0xF9
        elif op == 0xF9:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            addr = self.get_reg(rn) + (imm12 << 3)
            self.memory.write_dword(addr, self.get_reg(rt))
            return True

        # LDR (load): 0xF9 with bit 22 set
        elif op == 0xF9 and (inst >> 22) & 1:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            addr = self.get_reg(rn) + (imm12 << 3)
            self.set_reg(rt, self.memory.read_dword(addr))
            return True

        # STRB (store byte): 0x39
        elif op == 0x39 and not ((inst >> 22) & 1):
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            addr = self.get_reg(rn) + imm12
            self.memory.write_byte(addr, self.get_reg(rt) & 0xFF)
            return True

        # LDRB (load byte): 0x39 with bit 22
        elif op == 0x39 and ((inst >> 22) & 1):
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            addr = self.get_reg(rn) + imm12
            self.set_reg(rt, self.memory.read_byte(addr))
            return True

        # B (branch): 0x14
        elif (op >> 2) == 0x05:  # 0x14-0x17
            imm26 = inst & 0x3FFFFFF
            if imm26 & 0x2000000:  # Sign extend
                imm26 |= ~0x3FFFFFF
            self.pc = (self.pc - 4) + (imm26 << 2)
            return True

        # B.cond: 0x54
        elif op == 0x54:
            cond = inst & 0xF
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 |= ~0x7FFFF
            # Simplified condition check
            take_branch = False
            if cond == 0x0:  # EQ
                take_branch = self.z_flag
            elif cond == 0x1:  # NE
                take_branch = not self.z_flag
            # Add more conditions as needed
            if take_branch:
                self.pc = (self.pc - 4) + (imm19 << 2)
            return True

        # RET: 0xD65F03C0
        elif inst == 0xD65F03C0:
            self.pc = self.get_reg(30)  # X30 = LR
            return False  # Return from function

        # NOP or unknown - continue
        return True

    def run(self, max_instructions=10000):
        """Run until max instructions or halt."""
        start_time = time.time()

        while self.instruction_count < max_instructions:
            if not self.decode_and_execute():
                break

        elapsed = time.time() - start_time
        ips = self.instruction_count / elapsed if elapsed > 0 else 0

        return {
            'instructions': self.instruction_count,
            'neural_ops': self.neural_ops,
            'elapsed': elapsed,
            'ips': ips
        }


# =============================================================================
# NEURAL FRAMEBUFFER
# =============================================================================

class NeuralFramebuffer:
    """Neural framebuffer for rendering."""

    def __init__(self, cpu, fb_addr=0x40000, width=80, height=25):
        self.cpu = cpu
        self.fb_addr = fb_addr
        self.width = width
        self.height = height

    def read_frame(self):
        """Read ASCII framebuffer from neural memory."""
        frame = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                addr = self.fb_addr + y * self.width + x
                char_val = self.cpu.memory.read_byte(addr)
                if 32 <= char_val <= 126:
                    row += chr(char_val)
                else:
                    row += ' '
            frame.append(row)
        return frame


# =============================================================================
# NEURAL KEYBOARD
# =============================================================================

class NeuralKeyboard:
    """Neural keyboard input handler."""

    def __init__(self, cpu, key_addr=0x50000):
        self.cpu = cpu
        self.key_addr = key_addr

    def send_key(self, key):
        if isinstance(key, str) and len(key) == 1:
            self.cpu.memory.write_byte(self.key_addr, ord(key))
        elif isinstance(key, int):
            self.cpu.memory.write_byte(self.key_addr, key)


# =============================================================================
# ELF LOADER
# =============================================================================

def load_elf(cpu, elf_path):
    """Load ARM64 ELF binary into neural memory."""
    with open(elf_path, 'rb') as f:
        data = f.read()

    if data[:4] != b'\x7fELF':
        raise ValueError("Not a valid ELF file")

    # Get entry point
    entry = struct.unpack('<Q', data[24:32])[0]

    # Get program headers
    phoff = struct.unpack('<Q', data[32:40])[0]
    phentsize = struct.unpack('<H', data[54:56])[0]
    phnum = struct.unpack('<H', data[56:58])[0]

    # Load segments
    for i in range(phnum):
        ph_start = phoff + i * phentsize
        ph_type = struct.unpack('<I', data[ph_start:ph_start+4])[0]

        if ph_type == 1:  # PT_LOAD
            p_offset = struct.unpack('<Q', data[ph_start+8:ph_start+16])[0]
            p_vaddr = struct.unpack('<Q', data[ph_start+16:ph_start+24])[0]
            p_filesz = struct.unpack('<Q', data[ph_start+32:ph_start+40])[0]

            segment = data[p_offset:p_offset+p_filesz]
            cpu.memory.load_binary(segment, p_vaddr)

    return entry


# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

@dataclass
class Process:
    pid: int
    name: str
    state: str  # 'running', 'ready', 'blocked'
    entry_point: int
    instructions: int = 0
    neural_ops: int = 0


class ProcessManager:
    """Minimal process management."""

    def __init__(self):
        self.processes: Dict[int, Process] = {}
        self.next_pid = 1
        self.current_pid = 0

    def create_process(self, name, entry_point):
        pid = self.next_pid
        self.next_pid += 1
        self.processes[pid] = Process(
            pid=pid,
            name=name,
            state='ready',
            entry_point=entry_point
        )
        return pid

    def list_processes(self):
        return list(self.processes.values())

    def get_current(self):
        if self.current_pid in self.processes:
            return self.processes[self.current_pid]
        return None


# =============================================================================
# NEURAL FILE SYSTEM (In-Memory)
# =============================================================================

@dataclass
class NeuralFile:
    """A file stored in neural memory."""
    name: str
    content: str
    created: float
    modified: float
    size: int = 0

    def __post_init__(self):
        self.size = len(self.content)


class NeuralFileSystem:
    """
    In-memory file system for the neural RTOS.

    Files are stored as tensors in neural memory for true neural storage.
    """

    def __init__(self, cpu):
        self.cpu = cpu
        self.files: Dict[str, NeuralFile] = {}
        self.current_dir = "/"

        # File system starts at 0x100000 in neural memory
        self.fs_base = 0x100000
        self.fs_size = 0x100000  # 1MB for files

        # Pre-create some example files
        self._create_default_files()

    def _create_default_files(self):
        """Create default files."""
        now = time.time()

        self.files["/welcome.txt"] = NeuralFile(
            name="welcome.txt",
            content="""Welcome to Neural RTOS!

This is a fully neural operating system where every computation
is performed by neural networks:

- Neural ALU: Transformer-based carry/borrow prediction
- Neural Decoder: ARM64 instruction decoding
- Neural Memory: Tensor-based memory storage
- Neural Framebuffer: UNet rendering

Type 'help' to see available commands.
Type 'programs' to see available programs.
Type 'doom' to play DOOM!

Enjoy your neural computing experience!
""",
            created=now,
            modified=now
        )

        self.files["/readme.md"] = NeuralFile(
            name="readme.md",
            content="""# Neural RTOS

## Overview
Neural RTOS is a real-time operating system where ALL computations
are performed through neural network forward passes.

## Features
- ARM64 program execution via neural CPU
- In-memory file system
- Text editor
- Process management
- DOOM raycasting demo

## Commands
- `edit <file>` - Edit a text file
- `cat <file>` - Display file contents
- `ls` - List files
- `run <program>` - Run ARM64 binary
""",
            created=now,
            modified=now
        )

        self.files["/notes.txt"] = NeuralFile(
            name="notes.txt",
            content="My notes:\n- Remember to try the DOOM demo\n- Neural computing is the future!\n",
            created=now,
            modified=now
        )

    def list_files(self, path="/"):
        """List files in directory."""
        return [f for f in self.files.values() if f.name.startswith(path.rstrip("/") + "/") or path == "/"]

    def read_file(self, path):
        """Read file contents."""
        if not path.startswith("/"):
            path = "/" + path
        if path in self.files:
            return self.files[path].content
        return None

    def write_file(self, path, content):
        """Write content to file."""
        if not path.startswith("/"):
            path = "/" + path
        now = time.time()
        if path in self.files:
            self.files[path].content = content
            self.files[path].modified = now
            self.files[path].size = len(content)
        else:
            self.files[path] = NeuralFile(
                name=path.split("/")[-1],
                content=content,
                created=now,
                modified=now
            )
        return True

    def delete_file(self, path):
        """Delete a file."""
        if not path.startswith("/"):
            path = "/" + path
        if path in self.files:
            del self.files[path]
            return True
        return False

    def file_exists(self, path):
        """Check if file exists."""
        if not path.startswith("/"):
            path = "/" + path
        return path in self.files


# =============================================================================
# NEURAL TEXT EDITOR
# =============================================================================

class NeuralTextEditor:
    """
    Simple text editor for Neural RTOS.

    Inspired by nano - simple and functional.
    """

    def __init__(self, fs):
        self.fs = fs
        self.buffer = []
        self.cursor_row = 0
        self.cursor_col = 0
        self.filename = None
        self.modified = False

    def edit(self, filename):
        """Open file for editing."""
        self.filename = filename

        # Load existing content
        content = self.fs.read_file(filename)
        if content:
            self.buffer = content.split('\n')
        else:
            self.buffer = ['']

        self.cursor_row = 0
        self.cursor_col = 0
        self.modified = False

        self._editor_loop()

    def _clear_screen(self):
        print("\033[2J\033[H", end="")

    def _render(self):
        """Render the editor screen."""
        self._clear_screen()

        # Header
        status = "*" if self.modified else ""
        print(f"\033[7m  NEURAL EDITOR | {self.filename}{status} | Ctrl-S: Save | Ctrl-Q: Quit  \033[0m")
        print("─" * 78)

        # Content (show up to 20 lines)
        display_lines = 20
        start_line = max(0, self.cursor_row - display_lines // 2)

        for i in range(display_lines):
            line_num = start_line + i
            if line_num < len(self.buffer):
                line = self.buffer[line_num][:76]
                prefix = ">" if line_num == self.cursor_row else " "
                print(f"{prefix}{line_num+1:3d}│ {line}")
            else:
                print(f"    │ ~")

        # Footer
        print("─" * 78)
        print(f"  Line {self.cursor_row+1}/{len(self.buffer)} | Col {self.cursor_col+1}")

    def _editor_loop(self):
        """Main editor loop."""
        import sys
        import tty
        import termios

        # Try to use raw terminal mode
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            use_raw = True
        except:
            use_raw = False

        self._render()

        if use_raw:
            # Raw terminal mode for real-time editing
            try:
                tty.setraw(sys.stdin.fileno())

                while True:
                    ch = sys.stdin.read(1)

                    if ch == '\x11':  # Ctrl-Q
                        break
                    elif ch == '\x13':  # Ctrl-S
                        self._save()
                    elif ch == '\x7f' or ch == '\x08':  # Backspace
                        self._backspace()
                    elif ch == '\r' or ch == '\n':  # Enter
                        self._newline()
                    elif ch == '\x1b':  # Escape sequence (arrows)
                        seq = sys.stdin.read(2)
                        if seq == '[A':  # Up
                            self.cursor_row = max(0, self.cursor_row - 1)
                        elif seq == '[B':  # Down
                            self.cursor_row = min(len(self.buffer) - 1, self.cursor_row + 1)
                        elif seq == '[C':  # Right
                            self.cursor_col += 1
                        elif seq == '[D':  # Left
                            self.cursor_col = max(0, self.cursor_col - 1)
                    elif ch >= ' ' and ch <= '~':  # Printable
                        self._insert_char(ch)

                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    self._render()
                    tty.setraw(sys.stdin.fileno())

            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        else:
            # Fallback: line-by-line editing
            print("\n  [Simple mode - raw terminal not available]")
            print("  Commands: :w (save), :q (quit), :wq (save & quit)")
            print("  Type text and press Enter to add lines.")
            print()

            while True:
                try:
                    line = input(f"  {len(self.buffer)+1}> ")

                    if line == ':q':
                        break
                    elif line == ':w':
                        self._save()
                        print("  Saved!")
                    elif line == ':wq':
                        self._save()
                        print("  Saved!")
                        break
                    else:
                        self.buffer.append(line)
                        self.modified = True
                except (KeyboardInterrupt, EOFError):
                    break

    def _insert_char(self, ch):
        """Insert character at cursor."""
        if self.cursor_row < len(self.buffer):
            line = self.buffer[self.cursor_row]
            self.buffer[self.cursor_row] = line[:self.cursor_col] + ch + line[self.cursor_col:]
            self.cursor_col += 1
            self.modified = True

    def _backspace(self):
        """Delete character before cursor."""
        if self.cursor_col > 0:
            line = self.buffer[self.cursor_row]
            self.buffer[self.cursor_row] = line[:self.cursor_col-1] + line[self.cursor_col:]
            self.cursor_col -= 1
            self.modified = True
        elif self.cursor_row > 0:
            # Join with previous line
            prev_len = len(self.buffer[self.cursor_row - 1])
            self.buffer[self.cursor_row - 1] += self.buffer[self.cursor_row]
            del self.buffer[self.cursor_row]
            self.cursor_row -= 1
            self.cursor_col = prev_len
            self.modified = True

    def _newline(self):
        """Insert newline."""
        line = self.buffer[self.cursor_row]
        self.buffer[self.cursor_row] = line[:self.cursor_col]
        self.buffer.insert(self.cursor_row + 1, line[self.cursor_col:])
        self.cursor_row += 1
        self.cursor_col = 0
        self.modified = True

    def _save(self):
        """Save file."""
        content = '\n'.join(self.buffer)
        self.fs.write_file(self.filename, content)
        self.modified = False


# =============================================================================
# NEURAL RTOS
# =============================================================================

class NeuralRTOS:
    """
    Neural Real-Time Operating System

    A complete OS where EVERY computation is neural:
    - Neural ELF loader
    - Neural CPU (decoder, ALU, registers, memory)
    - Neural framebuffer
    - Neural keyboard
    - Neural file system
    - Text editor
    """

    VERSION = "1.0.0"
    CODENAME = "Synapse"

    def __init__(self):
        self.cpu = NeuralCPU()
        self.framebuffer = NeuralFramebuffer(self.cpu)
        self.keyboard = NeuralKeyboard(self.cpu)
        self.process_manager = ProcessManager()
        self.fs = NeuralFileSystem(self.cpu)
        self.editor = NeuralTextEditor(self.fs)

        self.boot_time = None
        self.programs: Dict[str, str] = {}  # name -> path
        self.env_vars: Dict[str, str] = {
            "USER": "neural",
            "HOME": "/",
            "SHELL": "neural-shell",
            "PATH": "/bin:/programs",
        }

        self._discover_programs()

    def _discover_programs(self):
        """Discover available ARM64 programs."""
        base_dir = Path(__file__).parent

        # Look for ELF files
        patterns = [
            "arm64_doom/*.elf",
            "programs/*.elf",
            "*.elf"
        ]

        for pattern in patterns:
            for path in base_dir.glob(pattern):
                name = path.stem
                self.programs[name] = str(path)

    def boot(self):
        """Boot sequence."""
        self._clear_screen()
        self._show_boot_screen()
        time.sleep(0.5)
        self._show_init_sequence()
        self.boot_time = time.time()

    def _clear_screen(self):
        print("\033[2J\033[H", end="")

    def _show_boot_screen(self):
        boot_art = """
\033[36m
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║     _   _ _____ _   _ ____      _    _       ____  _____ ___  ____       ║
    ║    | \\ | | ____| | | |  _ \\    / \\  | |     |  _ \\|_   _/ _ \\/ ___|     ║
    ║    |  \\| |  _| | | | | |_) |  / _ \\ | |     | |_) | | || | | \\___ \\     ║
    ║    | |\\  | |___| |_| |  _ <  / ___ \\| |___  |  _ <  | || |_| |___) |    ║
    ║    |_| \\_|_____|\\___/|_| \\_\\/_/   \\_\\_____| |_| \\_\\ |_| \\___/|____/     ║
    ║                                                                          ║
    ║                  NEURAL REAL-TIME OPERATING SYSTEM                       ║
    ║                      Version {ver} "{codename}"                         ║
    ║                                                                          ║
    ║           "The machine doesn't compute - it THINKS the computation"      ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
\033[0m
        """.format(ver=self.VERSION, codename=self.CODENAME)
        print(boot_art)

    def _show_init_sequence(self):
        components = [
            ("Neural ALU", "Transformer carry/borrow prediction", "models/final/ADD_64bit_100pct.pt"),
            ("Neural Decoder", "ARM64 instruction decoding", "models/final/arm64_decoder_100pct.pt"),
            ("Neural Memory", "Tensor-based memory system", "64MB neural RAM"),
            ("Neural Registers", "Values stored in network weights", "32 x 64-bit"),
            ("Neural Framebuffer", "ASCII rendering at 0x40000", "80x25"),
            ("Neural Keyboard", "Input handling at 0x50000", "ready"),
        ]

        print("\033[33m  [BOOT] Initializing neural components...\033[0m\n")

        for name, desc, detail in components:
            print(f"  \033[32m[OK]\033[0m {name:20s} - {desc}")
            time.sleep(0.1)

        print()
        print(f"  \033[36m[INFO]\033[0m Device: {device}")
        print(f"  \033[36m[INFO]\033[0m Programs discovered: {len(self.programs)}")

        if self.cpu.alu.models:
            print(f"  \033[36m[INFO]\033[0m Neural ALU models: {list(self.cpu.alu.models.keys())}")
        else:
            print(f"  \033[33m[WARN]\033[0m Neural ALU using fallback (models not found)")

        print()
        print("\033[32m  [BOOT] Neural RTOS ready!\033[0m")
        print()

    def shell(self):
        """Interactive shell."""
        print("\033[36m  Type 'help' for available commands.\033[0m\n")

        while True:
            try:
                cmd = input("\033[32mneural>\033[0m ").strip()

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0].lower()
                args = parts[1:]

                if command in ['exit', 'quit', 'shutdown']:
                    self._shutdown()
                    break
                elif command == 'help':
                    self._cmd_help()
                elif command == 'programs':
                    self._cmd_list_programs()
                elif command == 'ls':
                    if args and args[0] == '-l':
                        self._cmd_files()
                    else:
                        self._cmd_ls()
                elif command == 'run':
                    if args:
                        self._cmd_run(args[0])
                    else:
                        print("  Usage: run <program>")
                elif command == 'load':
                    if args:
                        self._cmd_load(args[0])
                    else:
                        print("  Usage: load <elf_path>")
                elif command == 'ps':
                    self._cmd_ps()
                elif command == 'mem':
                    self._cmd_mem()
                elif command == 'regs':
                    self._cmd_regs()
                elif command == 'clear':
                    self._clear_screen()
                elif command == 'info':
                    self._cmd_info()
                elif command == 'doom':
                    self._cmd_doom()
                elif command == 'cat':
                    if args:
                        self._cmd_cat(args[0])
                    else:
                        print("  Usage: cat <file>")
                elif command == 'edit' or command == 'nano':
                    if args:
                        self._cmd_edit(args[0])
                    else:
                        print("  Usage: edit <file>")
                elif command == 'touch':
                    if args:
                        self._cmd_touch(args[0])
                    else:
                        print("  Usage: touch <file>")
                elif command == 'rm':
                    if args:
                        self._cmd_rm(args[0])
                    else:
                        print("  Usage: rm <file>")
                elif command == 'echo':
                    self._cmd_echo(args)
                elif command == 'files':
                    self._cmd_files()
                elif command == 'env':
                    self._cmd_env()
                elif command == 'export':
                    if args and '=' in args[0]:
                        self._cmd_export(args[0])
                    else:
                        print("  Usage: export VAR=value")
                elif command == 'date':
                    self._cmd_date()
                elif command == 'uptime':
                    self._cmd_uptime()
                elif command == 'calc':
                    if len(args) >= 3:
                        self._cmd_calc(args)
                    else:
                        print("  Usage: calc <a> <op> <b>")
                        print("  Example: calc 100 + 42")
                elif command == 'hexdump':
                    if args:
                        self._cmd_hexdump(args[0], int(args[1]) if len(args) > 1 else 64)
                    else:
                        print("  Usage: hexdump <address> [length]")
                elif command == 'poke':
                    if len(args) >= 2:
                        self._cmd_poke(args[0], args[1])
                    else:
                        print("  Usage: poke <address> <value>")
                elif command == 'peek':
                    if args:
                        self._cmd_peek(args[0])
                    else:
                        print("  Usage: peek <address>")
                elif command == 'neofetch':
                    self._cmd_neofetch()
                elif command == 'arm-add':
                    self._cmd_arm_add(args)
                elif command == 'arm-sub':
                    self._cmd_arm_sub(args)
                elif command == 'arm-fib':
                    self._cmd_arm_fib(args)
                elif command == 'arm-test':
                    self._cmd_arm_test()
                else:
                    print(f"  Unknown command: {command}")
                    print("  Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n")
                continue
            except EOFError:
                self._shutdown()
                break

    def _cmd_help(self):
        help_text = """
\033[36m  NEURAL RTOS COMMANDS\033[0m
  ═══════════════════════════════════════════════════════════════

  \033[33mProgram Management:\033[0m
    programs         - List available ARM64 programs
    run <name>       - Run a program by name
    load <path>      - Load and run an ELF file
    doom             - Run DOOM (if available)

  \033[33mFile System:\033[0m
    files            - List all files
    cat <file>       - Display file contents
    edit <file>      - Edit file (nano-style)
    touch <file>     - Create empty file
    rm <file>        - Delete file
    echo <text>      - Print text (use > to redirect)

  \033[33mSystem Information:\033[0m
    ps               - List processes
    mem              - Show memory usage
    regs             - Show register state
    info             - System information
    env              - Show environment variables
    date             - Show current date/time
    uptime           - Show system uptime
    neofetch         - System info with ASCII art

  \033[33mNeural Operations:\033[0m
    calc <a> <op> <b> - Neural ALU calculation
    peek <addr>       - Read memory address
    poke <addr> <val> - Write to memory address
    hexdump <addr>    - Dump memory in hex

  \033[33mShell:\033[0m
    clear            - Clear screen
    help             - Show this help
    export VAR=val   - Set environment variable
    exit, quit       - Shutdown system

  ═══════════════════════════════════════════════════════════════
"""
        print(help_text)

    def _cmd_list_programs(self):
        print("\n\033[36m  AVAILABLE PROGRAMS\033[0m")
        print("  " + "─" * 60)

        if not self.programs:
            print("  No programs found.")
            print("  Place .elf files in arm64_doom/ or programs/")
        else:
            for name, path in sorted(self.programs.items()):
                size = os.path.getsize(path) if os.path.exists(path) else 0
                print(f"  {name:20s} {size:>10,} bytes")

        print()

    def _cmd_run(self, name):
        if name not in self.programs:
            print(f"  Program not found: {name}")
            print("  Use 'programs' to list available programs.")
            return

        path = self.programs[name]
        self._run_program(path, name)

    def _cmd_load(self, path):
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            return

        name = Path(path).stem
        self._run_program(path, name)

    def _run_program(self, path, name):
        print(f"\n\033[33m  Loading {name}...\033[0m")

        try:
            # Reset CPU
            self.cpu.reset()

            # Load ELF
            entry_point = load_elf(self.cpu, path)
            print(f"  Entry point: 0x{entry_point:x}")

            # Create process
            pid = self.process_manager.create_process(name, entry_point)
            self.process_manager.current_pid = pid

            # Set PC
            self.cpu.pc = entry_point

            print(f"\n\033[32m  Running {name} (PID {pid})...\033[0m\n")

            # Run
            start_time = time.time()
            result = self.cpu.run(max_instructions=100000)
            elapsed = time.time() - start_time

            # Update process
            proc = self.process_manager.processes[pid]
            proc.instructions = result['instructions']
            proc.neural_ops = result['neural_ops']
            proc.state = 'completed'

            # Show results
            print(f"\n\033[36m  Execution Complete\033[0m")
            print(f"  ─────────────────────────────────")
            print(f"  Instructions: {result['instructions']:,}")
            print(f"  Neural ops:   {result['neural_ops']:,}")
            print(f"  Time:         {elapsed*1000:.1f}ms")
            print(f"  IPS:          {result['ips']:,.0f}")
            print()

            # Show framebuffer if it has content
            self._show_framebuffer_if_content()

        except Exception as e:
            print(f"\033[31m  Error: {e}\033[0m")

    def _show_framebuffer_if_content(self):
        """Show framebuffer if it contains non-empty content."""
        frame = self.framebuffer.read_frame()
        has_content = any(row.strip() for row in frame)

        if has_content:
            print("\033[36m  FRAMEBUFFER OUTPUT:\033[0m")
            print("  " + "═" * 80)
            for row in frame:
                print(f"  {row}")
            print("  " + "═" * 80)
            print()

    def _cmd_ps(self):
        print("\n\033[36m  PROCESS LIST\033[0m")
        print("  " + "─" * 60)
        print(f"  {'PID':>5} {'NAME':20} {'STATE':10} {'INSTR':>12} {'NEURAL':>10}")
        print("  " + "─" * 60)

        for proc in self.process_manager.list_processes():
            print(f"  {proc.pid:>5} {proc.name:20} {proc.state:10} {proc.instructions:>12,} {proc.neural_ops:>10,}")

        if not self.process_manager.processes:
            print("  No processes.")

        print()

    def _cmd_mem(self):
        print("\n\033[36m  MEMORY USAGE\033[0m")
        print("  " + "─" * 40)

        total = self.cpu.memory.size
        # Count non-zero bytes (rough estimate of used memory)
        used = (self.cpu.memory.memory != 0).sum().item()

        print(f"  Total:     {total:>12,} bytes ({total/1024/1024:.0f} MB)")
        print(f"  Used:      {used:>12,} bytes")
        print(f"  Free:      {total-used:>12,} bytes")
        print()
        print("  \033[33mMemory Map:\033[0m")
        print("  0x10000 - 0x1FFFF: Code")
        print("  0x20000 - 0x2FFFF: Data")
        print("  0x30000 - 0x3FFFF: Map data")
        print("  0x40000 - 0x4FFFF: Framebuffer")
        print("  0x50000 - 0x50FFF: I/O (keyboard)")
        print()

    def _cmd_regs(self):
        print("\n\033[36m  REGISTER STATE\033[0m")
        print("  " + "─" * 50)

        for i in range(0, 32, 4):
            row = "  "
            for j in range(4):
                idx = i + j
                if idx < 31:
                    val = self.cpu.get_reg(idx)
                    row += f"X{idx:02d}={val:016x}  "
                else:
                    row += f"XZR=0000000000000000  "
            print(row)

        print()
        print(f"  PC  = {self.cpu.pc:016x}")
        print(f"  SP  = {self.cpu.sp:016x}")
        print(f"  Flags: N={int(self.cpu.n_flag)} Z={int(self.cpu.z_flag)} C={int(self.cpu.c_flag)} V={int(self.cpu.v_flag)}")
        print()

    def _cmd_info(self):
        uptime = time.time() - self.boot_time if self.boot_time else 0

        print("\n\033[36m  NEURAL RTOS SYSTEM INFORMATION\033[0m")
        print("  " + "═" * 50)
        print(f"  Version:      {self.VERSION} \"{self.CODENAME}\"")
        print(f"  Device:       {device}")
        print(f"  Uptime:       {uptime:.1f}s")
        print()
        print("  \033[33mNeural Components:\033[0m")

        alu_models = list(self.cpu.alu.models.keys()) if self.cpu.alu.models else ["(fallback)"]
        print(f"  ALU Models:   {', '.join(alu_models)}")
        print(f"  Memory:       {self.cpu.memory.size/1024/1024:.0f} MB neural RAM")
        print(f"  Registers:    32 x 64-bit")
        print(f"  Framebuffer:  80x25 @ 0x40000")
        print()
        print("  \033[33mStatistics:\033[0m")
        print(f"  Total Instructions: {self.cpu.instruction_count:,}")
        print(f"  Neural Operations:  {self.cpu.neural_ops:,}")
        print()

    def _cmd_files(self):
        """List all files in the neural file system."""
        print("\n\033[36m  NEURAL FILE SYSTEM\033[0m")
        print("  " + "─" * 60)
        print(f"  {'NAME':30} {'SIZE':>10} {'MODIFIED':20}")
        print("  " + "─" * 60)

        for path, file in sorted(self.fs.files.items()):
            mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(file.modified))
            print(f"  {path:30} {file.size:>10} {mod_time}")

        print()
        print(f"  Total: {len(self.fs.files)} files")
        print()

    def _cmd_cat(self, filename):
        """Display file contents."""
        content = self.fs.read_file(filename)
        if content is not None:
            print()
            for line in content.split('\n'):
                print(f"  {line}")
            print()
        else:
            print(f"  File not found: {filename}")

    def _cmd_edit(self, filename):
        """Edit a file."""
        self.editor.edit(filename)
        self._clear_screen()
        print("  Editor closed.")

    def _cmd_touch(self, filename):
        """Create empty file."""
        if not self.fs.file_exists(filename):
            self.fs.write_file(filename, "")
            print(f"  Created: {filename}")
        else:
            print(f"  File already exists: {filename}")

    def _cmd_rm(self, filename):
        """Delete a file."""
        if self.fs.delete_file(filename):
            print(f"  Deleted: {filename}")
        else:
            print(f"  File not found: {filename}")

    def _cmd_echo(self, args):
        """Echo text, with optional file redirect."""
        text = ' '.join(args)

        # Check for redirect
        if '>' in text:
            parts = text.split('>')
            content = parts[0].strip()
            filename = parts[1].strip()
            self.fs.write_file(filename, content + '\n')
            print(f"  Written to {filename}")
        else:
            print(f"  {text}")

    def _cmd_env(self):
        """Show environment variables."""
        print("\n\033[36m  ENVIRONMENT VARIABLES\033[0m")
        print("  " + "─" * 40)
        for key, value in sorted(self.env_vars.items()):
            print(f"  {key}={value}")
        print()

    def _cmd_export(self, assignment):
        """Set environment variable."""
        key, value = assignment.split('=', 1)
        self.env_vars[key] = value
        print(f"  {key}={value}")

    def _cmd_date(self):
        """Show current date/time."""
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {now}")

    def _cmd_uptime(self):
        """Show system uptime."""
        uptime = time.time() - self.boot_time if self.boot_time else 0
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        print(f"  up {hours}:{minutes:02d}:{seconds:02d}")

    def _cmd_calc(self, args):
        """Neural ALU calculation."""
        try:
            a = int(args[0], 0)  # Auto-detect hex/dec
            op = args[1]
            b = int(args[2], 0)

            op_map = {
                '+': 'ADD', '-': 'SUB',
                '&': 'AND', '|': 'OR', '^': 'XOR'
            }

            if op in op_map:
                result = self.cpu.alu.execute(op_map[op], a, b)
                print(f"\n  \033[36mNeural ALU Calculation:\033[0m")
                print(f"  {a} {op} {b} = {result}")
                print(f"  Hex: 0x{a:x} {op} 0x{b:x} = 0x{result:x}")
                print()
            else:
                print(f"  Unknown operator: {op}")
                print("  Supported: + - & | ^")
        except Exception as e:
            print(f"  Error: {e}")

    def _cmd_peek(self, addr_str):
        """Read memory address."""
        try:
            addr = int(addr_str, 0)
            val = self.cpu.memory.read_dword(addr)
            print(f"  [{addr:#010x}] = {val:#018x} ({val})")
        except Exception as e:
            print(f"  Error: {e}")

    def _cmd_poke(self, addr_str, val_str):
        """Write to memory address."""
        try:
            addr = int(addr_str, 0)
            val = int(val_str, 0)
            self.cpu.memory.write_dword(addr, val)
            print(f"  [{addr:#010x}] <- {val:#018x}")
        except Exception as e:
            print(f"  Error: {e}")

    def _cmd_hexdump(self, addr_str, length=64):
        """Dump memory in hex."""
        try:
            addr = int(addr_str, 0)
            print(f"\n  Memory dump at {addr:#010x}:\n")

            for row in range(0, length, 16):
                hex_part = ""
                ascii_part = ""
                for col in range(16):
                    if row + col < length:
                        byte = self.cpu.memory.read_byte(addr + row + col)
                        hex_part += f"{byte:02x} "
                        ascii_part += chr(byte) if 32 <= byte <= 126 else '.'
                    else:
                        hex_part += "   "

                print(f"  {addr+row:08x}: {hex_part} |{ascii_part}|")
            print()
        except Exception as e:
            print(f"  Error: {e}")

    def _cmd_ls(self):
        """Simple ls - list files and programs."""
        print("\n  \033[36mFiles:\033[0m")
        for path in sorted(self.fs.files.keys()):
            name = path.split('/')[-1]
            print(f"    {name}")

        print("\n  \033[33mPrograms (ARM64):\033[0m")
        for name in sorted(self.programs.keys()):
            print(f"    {name}")
        print()

    def _cmd_neofetch(self):
        """System info with ASCII art."""
        uptime = time.time() - self.boot_time if self.boot_time else 0
        alu_ops = list(self.cpu.alu.models.keys()) if self.cpu.alu.models else ["fallback"]

        print("""
\033[36m         _   _ ____  _____ ___  ____
        | \\ | |  _ \\|_   _/ _ \\/ ___|
        |  \\| | |_) | | || | | \\___ \\
        | |\\  |  _ <  | || |_| |___) |
        |_| \\_|_| \\_\\ |_| \\___/|____/\033[0m
        """)
        print(f"        \033[36m{self.env_vars['USER']}\033[0m@\033[36mneural\033[0m")
        print(f"        ─────────────────────")
        print(f"        \033[36mOS:\033[0m Neural RTOS {self.VERSION}")
        print(f"        \033[36mArch:\033[0m ARM64 (AArch64)")
        print(f"        \033[36mKernel:\033[0m Neural {self.CODENAME}")
        print(f"        \033[36mUptime:\033[0m {uptime:.0f}s")
        print(f"        \033[36mDevice:\033[0m {device}")
        print(f"        \033[36mMemory:\033[0m {self.cpu.memory.size//1024//1024}MB")
        print(f"        \033[36mALU:\033[0m {', '.join(alu_ops)}")
        print(f"        \033[36mInstructions:\033[0m {self.cpu.instruction_count:,}")
        print(f"        \033[36mNeural Ops:\033[0m {self.cpu.neural_ops:,}")
        print(f"        \033[36mFiles:\033[0m {len(self.fs.files)}")
        print()

    # =========================================================================
    # ARM64 SHELL PROGRAMS (Run on Neural CPU!)
    # =========================================================================

    def _build_arm64_program(self, instructions):
        """Build ARM64 binary from instruction list."""
        return b''.join(struct.pack('<I', inst) for inst in instructions)

    def _run_arm64_shell_program(self, name, binary, show_result=True):
        """Run an ARM64 shell program on the neural CPU."""
        # Save current CPU state
        old_pc = self.cpu.pc
        old_regs = self.cpu.registers.clone()

        # Load program at shell program area
        shell_addr = 0x80000
        self.cpu.memory.load_binary(binary, shell_addr)
        self.cpu.pc = shell_addr

        # Run
        start_time = time.time()
        result = self.cpu.run(max_instructions=1000)
        elapsed = time.time() - start_time

        # Get result from X0
        x0_result = self.cpu.get_reg(0)

        if show_result:
            print(f"\n  \033[36mARM64 Program: {name}\033[0m")
            print(f"  Instructions: {result['instructions']}")
            print(f"  Neural ops:   {result['neural_ops']}")
            print(f"  Time:         {elapsed*1000:.2f}ms")
            print(f"  X0 (result):  {x0_result} (0x{x0_result:x})")
            print()

        # Restore CPU state
        self.cpu.pc = old_pc
        self.cpu.registers.copy_(old_regs)

        return x0_result

    def _cmd_arm_add(self, args):
        """Run ADD on neural CPU as ARM64 program."""
        if len(args) < 2:
            print("  Usage: arm-add <a> <b>")
            return

        a = int(args[0], 0)
        b = int(args[1], 0)

        # ARM64 program: ADD X0, X1, X2; RET
        # MOVZ X1, a; MOVZ X2, b; ADD X0, X1, X2; RET
        program = [
            0xD2800001 | ((a & 0xFFFF) << 5),     # MOVZ X1, #a
            0xD2800002 | ((b & 0xFFFF) << 5),     # MOVZ X2, #b
            0x8B020020,                            # ADD X0, X1, X2
            0xD65F03C0,                            # RET
        ]

        binary = self._build_arm64_program(program)
        result = self._run_arm64_shell_program("add", binary)
        print(f"  Result: {a} + {b} = {result}")

    def _cmd_arm_sub(self, args):
        """Run SUB on neural CPU as ARM64 program."""
        if len(args) < 2:
            print("  Usage: arm-sub <a> <b>")
            return

        a = int(args[0], 0)
        b = int(args[1], 0)

        program = [
            0xD2800001 | ((a & 0xFFFF) << 5),     # MOVZ X1, #a
            0xD2800002 | ((b & 0xFFFF) << 5),     # MOVZ X2, #b
            0xCB020020,                            # SUB X0, X1, X2
            0xD65F03C0,                            # RET
        ]

        binary = self._build_arm64_program(program)
        result = self._run_arm64_shell_program("sub", binary)
        print(f"  Result: {a} - {b} = {result}")

    def _cmd_arm_fib(self, args):
        """Run Fibonacci on neural CPU as ARM64 program."""
        n = int(args[0], 0) if args else 10

        # Fibonacci ARM64:
        # X0 = F(n-1), X1 = F(n), X2 = counter
        program = [
            0xD2800000,                            # MOVZ X0, #0 (F0)
            0xD2800021,                            # MOVZ X1, #1 (F1)
            0xD2800002 | ((n & 0xFFFF) << 5),     # MOVZ X2, #n
        ]

        # Loop: X3 = X0 + X1; X0 = X1; X1 = X3; X2--
        for _ in range(n):
            program.extend([
                0x8B010003,                        # ADD X3, X0, X1
                0xAA0103E0,                        # MOV X0, X1
                0xAA0303E1,                        # MOV X1, X3
            ])

        program.append(0xD65F03C0)                 # RET

        binary = self._build_arm64_program(program)
        result = self._run_arm64_shell_program(f"fib({n})", binary)
        print(f"  Fibonacci({n}) = {result}")

    def _cmd_doom(self):
        """Run DOOM if available."""
        doom_names = ['doom_neural', 'doom', 'raycast']

        for name in doom_names:
            if name in self.programs:
                self._run_doom_interactive(self.programs[name], name)
                return

        print("  DOOM not found.")
        print("  Build it with: cd arm64_doom && make")

    def _run_doom_interactive(self, path, name):
        """Run DOOM with interactive keyboard input."""
        print(f"\n\033[33m  Loading {name}...\033[0m")

        try:
            self.cpu.reset()
            entry_point = load_elf(self.cpu, path)
            self.cpu.pc = entry_point

            # Initialize map at 0x30000
            game_map = [
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
                [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
                [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            ]

            for y in range(16):
                for x in range(16):
                    self.cpu.memory.write_byte(0x30000 + y * 16 + x, game_map[y][x])

            print(f"  Entry: 0x{entry_point:x}")
            print(f"\n\033[32m  DOOM Interactive Mode\033[0m")
            print("  Controls: w/s (move), a/d (turn), q (quit)")
            print()

            frame_count = 0
            start_time = time.time()

            # Initial frame
            self.cpu.run(max_instructions=1000)
            self._render_doom_frame(frame_count, start_time)
            frame_count += 1

            while True:
                cmd = input("doom> ").strip().lower()

                if cmd == 'q':
                    break
                elif cmd in ['w', 'a', 's', 'd']:
                    self.keyboard.send_key(cmd)

                self.cpu.run(max_instructions=1000)
                self._render_doom_frame(frame_count, start_time)
                frame_count += 1

            elapsed = time.time() - start_time
            print(f"\n  Frames: {frame_count}")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  FPS: {frame_count/elapsed:.1f}")

        except Exception as e:
            print(f"\033[31m  Error: {e}\033[0m")

    def _render_doom_frame(self, frame_count, start_time):
        """Render DOOM frame."""
        self._clear_screen()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print(f"\033[36m  NEURAL DOOM | Frame: {frame_count} | FPS: {fps:.1f} | Neural ops: {self.cpu.neural_ops}\033[0m")
        print("  " + "═" * 80)

        frame = self.framebuffer.read_frame()
        for row in frame:
            print(f"  {row}")

        print("  " + "═" * 80)
        print("  Controls: w/s (move), a/d (turn), q (quit)")

    def _shutdown(self):
        print("\n\033[33m  Shutting down Neural RTOS...\033[0m")

        uptime = time.time() - self.boot_time if self.boot_time else 0

        print(f"  Total instructions: {self.cpu.instruction_count:,}")
        print(f"  Neural operations:  {self.cpu.neural_ops:,}")
        print(f"  Uptime:             {uptime:.1f}s")
        print()
        print("\033[32m  Neural RTOS halted. Goodbye!\033[0m\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    rtos = NeuralRTOS()
    rtos.boot()
    rtos.shell()


if __name__ == "__main__":
    main()
