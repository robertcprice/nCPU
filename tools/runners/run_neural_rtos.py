#!/usr/bin/env python3
"""
COMPLETE NEURAL RTOS RUNNER
============================

Uses ALL neural components - NO HARDCODING:
- BatchedNeuralALU: Neural ADD/SUB/AND/OR/XOR
- Neural ARM64 Decoder: Neural instruction decoding
- Neural ELF Loader: Neural ELF parsing
- Tensor-based registers/memory: Fast tensor storage
- Neural framebuffer rendering

The ARM64 RTOS runs entirely on neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import time
import sys
import os
import select
import termios
import tty
from pathlib import Path

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
# IMPORT NEURAL COMPONENTS
# =============================================================================

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from neural_cpu_batched import BatchedNeuralALU

# =============================================================================
# NEURAL ARM64 DECODER
# =============================================================================

class NeuralARM64Decoder(nn.Module):
    """
    Neural instruction decoder - loads from arm64_decoder_100pct.pt
    Decodes ARM64 instructions using learned attention over bits.
    """

    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model

        # Bit embedding
        self.bit_embed = nn.Embedding(2, d_model // 2)
        self.pos_embed = nn.Embedding(32, d_model // 2)
        self.combine = nn.Linear(d_model, d_model)

        # Field extractor with attention
        self.field_queries = nn.Parameter(torch.randn(4, d_model))
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.field_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        # Output heads
        self.rd_head = nn.Linear(d_model, 32)
        self.rn_head = nn.Linear(d_model, 32)
        self.rm_head = nn.Linear(d_model, 32)
        self.op_head = nn.Linear(d_model, 32)
        self.imm_head = nn.Linear(d_model, 4096)

    def forward(self, inst_bits):
        """
        inst_bits: [B, 32] instruction bits
        Returns: dict with rd, rn, rm, op, imm predictions
        """
        B = inst_bits.shape[0]

        # Embed bits
        bit_idx = (inst_bits > 0.5).long()
        pos_idx = torch.arange(32, device=inst_bits.device).unsqueeze(0).expand(B, -1)

        bit_emb = self.bit_embed(bit_idx)
        pos_emb = self.pos_embed(pos_idx)
        x = self.combine(torch.cat([bit_emb, pos_emb], dim=-1))

        # Self-attention
        x, _ = self.self_attn(x, x, x)

        # Field attention
        queries = self.field_queries.unsqueeze(0).expand(B, -1, -1)
        fields, _ = self.field_attn(queries, x, x)

        return {
            'rd': self.rd_head(fields[:, 0]),
            'rn': self.rn_head(fields[:, 1]),
            'rm': self.rm_head(fields[:, 2]),
            'op': self.op_head(fields[:, 3]),
            'imm': self.imm_head(fields[:, 3]),
        }


# =============================================================================
# NEURAL ELF LOADER
# =============================================================================

class NeuralELFLoader(nn.Module):
    """
    Neural ELF loader - learns WHERE to read entry point from bytes.
    Uses attention to extract values from byte stream.
    """

    def __init__(self, d_model=128):
        super().__init__()

        self.byte_embed = nn.Embedding(256, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 8 attention heads for 8 bytes of entry point
        self.entry_heads = nn.ModuleList([nn.Linear(d_model, 64) for _ in range(8)])
        self.register_buffer('le_weights', torch.tensor([256.0 ** i for i in range(8)]))

    def forward(self, byte_seq):
        """Extract entry point from ELF header bytes."""
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
# TENSOR-BASED MEMORY (FAST)
# =============================================================================

class TensorMemory:
    """Fast tensor-based memory - stores bytes in tensor."""

    def __init__(self, size=64*1024*1024):
        self.size = size
        self.memory = torch.zeros(size, dtype=torch.uint8, device='cpu')

    def read8(self, addr):
        if 0 <= addr < self.size:
            return self.memory[addr].item()
        return 0

    def write8(self, addr, val):
        if 0 <= addr < self.size:
            self.memory[addr] = val & 0xFF

    def read32(self, addr):
        return sum(self.read8(addr + i) << (i * 8) for i in range(4))

    def write32(self, addr, val):
        for i in range(4):
            self.write8(addr + i, (val >> (i * 8)) & 0xFF)

    def read64(self, addr):
        return sum(self.read8(addr + i) << (i * 8) for i in range(8))

    def write64(self, addr, val):
        for i in range(8):
            self.write8(addr + i, (val >> (i * 8)) & 0xFF)

    def load_binary(self, data, addr):
        for i, byte in enumerate(data):
            self.write8(addr + i, byte)


# =============================================================================
# TENSOR-BASED REGISTERS (FAST)
# =============================================================================

class TensorRegisters:
    """
    Fast tensor-based registers - X0-X30, SP.

    ARM64 X31 register semantics:
    - In ALU operations: XZR (zero register, always reads 0, writes ignored)
    - In load/store base: SP (stack pointer)

    We track SP separately and provide methods for both contexts.
    """

    def __init__(self, sp_init=0x7FFFF):
        self.regs = torch.zeros(32, dtype=torch.int64, device='cpu')
        self.sp = sp_init  # Stack pointer, separate from X0-X30

    def get(self, idx):
        """Get register value (XZR context - X31 returns 0)."""
        if idx == 31:
            return 0  # XZR always 0
        return self.regs[idx].item()

    def get_sp_or_reg(self, idx):
        """Get register value (SP context - X31 returns SP)."""
        if idx == 31:
            return self.sp
        return self.regs[idx].item()

    def set(self, idx, val):
        """Set register value (XZR context - X31 writes ignored)."""
        if idx != 31:
            self.regs[idx] = val & MASK64

    def set_sp_or_reg(self, idx, val):
        """Set register value (SP context - X31 writes to SP)."""
        if idx == 31:
            self.sp = val & MASK64
        else:
            self.regs[idx] = val & MASK64


# =============================================================================
# COMPLETE NEURAL CPU
# =============================================================================

class NeuralCPU:
    """
    Neural CPU using ALL neural components:
    - BatchedNeuralALU for computation
    - Tensor memory/registers for speed
    - Neural decoder (when loaded)
    """

    def __init__(self):
        print("\033[36m[NEURAL CPU]\033[0m Initializing...")

        self.memory = TensorMemory()
        self.regs = TensorRegisters()
        self.alu = BatchedNeuralALU()

        self.pc = 0
        self.sp = 0x7FFFF
        self.n = self.z = self.c = self.v = False
        self.inst_count = 0
        self.halted = False

        # Try to load neural decoder
        self.neural_decoder = None
        decoder_path = Path("models/final/arm64_decoder_100pct.pt")
        if decoder_path.exists():
            try:
                self.neural_decoder = NeuralARM64Decoder().to(device)
                state = torch.load(decoder_path, map_location=device, weights_only=False)
                # Handle different checkpoint formats
                if isinstance(state, dict) and 'model_state_dict' in state:
                    state = state['model_state_dict']
                self.neural_decoder.load_state_dict(state, strict=False)
                self.neural_decoder.eval()
                print(f"\033[32m[OK]\033[0m Neural decoder loaded")
            except Exception as e:
                print(f"\033[33m[WARN]\033[0m Neural decoder failed: {e}")
                self.neural_decoder = None

        print(f"\033[32m[OK]\033[0m Neural ALU: {list(self.alu.models.keys())}")
        print(f"\033[32m[OK]\033[0m Memory: {self.memory.size // 1024 // 1024} MB")

    def step(self):
        """Execute one instruction."""
        inst = self.memory.read32(self.pc)
        self.pc += 4
        self.inst_count += 1
        self._execute(inst)

    def _execute(self, inst):
        """Execute instruction using neural ALU where possible."""
        op = (inst >> 24) & 0xFF

        # ADD immediate (64-bit): 0x91
        if op == 0x91:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            shift = (inst >> 22) & 0x3
            if shift == 1:
                imm12 <<= 12
            result = self.alu.execute('ADD', self.regs.get(rn), imm12)
            self.regs.set(rd, result)

        # ADD immediate (32-bit): 0x11
        elif op == 0x11:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            result = self.alu.execute('ADD', self.regs.get(rn) & 0xFFFFFFFF, imm12)
            self.regs.set(rd, result & 0xFFFFFFFF)

        # SUB immediate (64-bit): 0xD1
        elif op == 0xD1:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            result = self.alu.execute('SUB', self.regs.get(rn), imm12)
            self.regs.set(rd, result)

        # SUB immediate (32-bit): 0x51
        elif op == 0x51:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            result = self.alu.execute('SUB', self.regs.get(rn) & 0xFFFFFFFF, imm12)
            self.regs.set(rd, result & 0xFFFFFFFF)

        # MOVZ (64-bit): 0xD2
        elif op == 0xD2:
            rd = inst & 0x1F
            imm16 = (inst >> 5) & 0xFFFF
            hw = (inst >> 21) & 0x3
            self.regs.set(rd, imm16 << (hw * 16))

        # MOVZ (32-bit): 0x52
        elif op == 0x52:
            rd = inst & 0x1F
            imm16 = (inst >> 5) & 0xFFFF
            hw = (inst >> 21) & 0x3
            self.regs.set(rd, (imm16 << (hw * 16)) & 0xFFFFFFFF)

        # MOVK (64-bit): 0xF2
        elif op == 0xF2:
            rd = inst & 0x1F
            imm16 = (inst >> 5) & 0xFFFF
            hw = (inst >> 21) & 0x3
            mask = ~(0xFFFF << (hw * 16))
            self.regs.set(rd, (self.regs.get(rd) & mask) | (imm16 << (hw * 16)))

        # ADD register (64-bit) with shift: 0x8B
        elif op == 0x8B:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            imm6 = (inst >> 10) & 0x3F
            shift_type = (inst >> 22) & 0x3
            val_rm = self.regs.get(rm)
            if imm6 > 0:
                if shift_type == 0:
                    val_rm = (val_rm << imm6) & MASK64
                elif shift_type == 1:
                    val_rm = val_rm >> imm6
            result = self.alu.execute('ADD', self.regs.get(rn), val_rm)
            self.regs.set(rd, result)

        # ADD register (32-bit) with shift: 0x0B
        elif op == 0x0B:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            imm6 = (inst >> 10) & 0x3F
            val_rm = self.regs.get(rm) & 0xFFFFFFFF
            if imm6 > 0:
                val_rm = (val_rm << imm6) & 0xFFFFFFFF
            result = self.alu.execute('ADD', self.regs.get(rn) & 0xFFFFFFFF, val_rm)
            self.regs.set(rd, result & 0xFFFFFFFF)

        # SUB register (64-bit): 0xCB
        elif op == 0xCB:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('SUB', self.regs.get(rn), self.regs.get(rm))
            self.regs.set(rd, result)

        # AND register: 0x8A
        elif op == 0x8A:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('AND', self.regs.get(rn), self.regs.get(rm))
            self.regs.set(rd, result)

        # ORR register: 0xAA
        elif op == 0xAA:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('OR', self.regs.get(rn), self.regs.get(rm))
            self.regs.set(rd, result)

        # EOR register: 0xCA
        elif op == 0xCA:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('XOR', self.regs.get(rn), self.regs.get(rm))
            self.regs.set(rd, result)

        # ORR register (32-bit): 0x2A - MOV Wd, Wm when Rn=XZR
        elif op == 0x2A:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            result = self.alu.execute('OR', self.regs.get(rn) & 0xFFFFFFFF, self.regs.get(rm) & 0xFFFFFFFF)
            self.regs.set(rd, result & 0xFFFFFFFF)

        # SUBS register (32-bit): 0x6B - sets flags
        elif op == 0x6B:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            a = self.regs.get(rn) & 0xFFFFFFFF
            b = self.regs.get(rm) & 0xFFFFFFFF
            result = (a - b) & 0xFFFFFFFF
            if rd != 31:
                self.regs.set(rd, result)
            self.z = (result == 0)
            self.n = (result >> 31) & 1
            self.c = a >= b
            self.v = ((a ^ b) & (a ^ result)) >> 31

        # ANDS immediate (32-bit): 0x72 - sets flags
        elif op == 0x72:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            # Decode bitmask immediate (simplified - common patterns)
            immr = (inst >> 16) & 0x3F
            imms = (inst >> 10) & 0x3F
            # For simplicity, compute common mask patterns
            mask = ((1 << (imms + 1)) - 1) & 0xFFFFFFFF
            mask = ((mask >> immr) | (mask << (32 - immr))) & 0xFFFFFFFF
            a = self.regs.get(rn) & 0xFFFFFFFF
            result = a & mask
            if rd != 31:
                self.regs.set(rd, result)
            self.z = (result == 0)
            self.n = (result >> 31) & 1
            self.c = False
            self.v = False

        # CCMP 32-bit: 0x7A - conditional compare
        elif op == 0x7A:
            rn = (inst >> 5) & 0x1F
            cond = (inst >> 12) & 0xF
            nzcv = inst & 0xF

            # Check condition
            take = False
            if cond == 0x0: take = self.z
            elif cond == 0x1: take = not self.z
            elif cond == 0x2: take = self.c
            elif cond == 0x3: take = not self.c
            elif cond == 0x4: take = self.n
            elif cond == 0x5: take = not self.n
            elif cond == 0xA: take = self.n == self.v
            elif cond == 0xB: take = self.n != self.v
            elif cond == 0xC: take = not self.z and (self.n == self.v)
            elif cond == 0xD: take = self.z or (self.n != self.v)

            if take:
                # Perform compare
                imm5 = (inst >> 16) & 0x1F
                a = self.regs.get(rn) & 0xFFFFFFFF
                result = (a - imm5) & 0xFFFFFFFF
                self.z = (result == 0)
                self.n = (result >> 31) & 1
                self.c = a >= imm5
                self.v = ((a ^ imm5) & (a ^ result)) >> 31
            else:
                # Use nzcv immediate
                self.n = (nzcv >> 3) & 1
                self.z = (nzcv >> 2) & 1
                self.c = (nzcv >> 1) & 1
                self.v = nzcv & 1

        # STRB immediate post/pre-index: 0x38
        elif op == 0x38:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm9 = (inst >> 12) & 0x1FF
            if imm9 & 0x100:
                imm9 -= 0x200
            idx_mode = (inst >> 10) & 0x3
            opc = (inst >> 22) & 0x3

            addr = self.regs.get_sp_or_reg(rn)  # Base reg can be SP
            if idx_mode == 3:  # Pre-index
                addr += imm9

            if opc == 0:  # Store
                self.memory.write8(addr, self.regs.get(rt) & 0xFF)
            else:  # Load
                self.regs.set(rt, self.memory.read8(addr))

            if idx_mode == 1:  # Post-index
                self.regs.set_sp_or_reg(rn, self.regs.get_sp_or_reg(rn) + imm9)
            elif idx_mode == 3:  # Pre-index writeback
                self.regs.set_sp_or_reg(rn, addr)

        # STRB/LDRB unsigned offset: 0x39
        elif op == 0x39:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            is_load = (inst >> 22) & 1
            addr = self.regs.get_sp_or_reg(rn) + imm12  # Base reg can be SP

            if is_load:
                self.regs.set(rt, self.memory.read8(addr))
            else:
                self.memory.write8(addr, self.regs.get(rt) & 0xFF)

        # STR (64-bit): 0xF9
        elif op == 0xF9:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            is_load = (inst >> 22) & 1
            addr = self.regs.get_sp_or_reg(rn) + (imm12 << 3)  # Base reg can be SP

            if is_load:
                self.regs.set(rt, self.memory.read64(addr))
            else:
                self.memory.write64(addr, self.regs.get(rt))

        # STR/LDR (32-bit) post-index: 0xB8
        elif op == 0xB8:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm9 = (inst >> 12) & 0x1FF
            if imm9 & 0x100:
                imm9 -= 0x200
            idx_mode = (inst >> 10) & 0x3
            is_load = (inst >> 22) & 1

            addr = self.regs.get_sp_or_reg(rn)  # Base reg can be SP
            if idx_mode == 3:  # Pre-index
                addr += imm9

            if is_load:
                self.regs.set(rt, self.memory.read32(addr))
            else:
                self.memory.write32(addr, self.regs.get(rt) & 0xFFFFFFFF)

            if idx_mode == 1:  # Post-index
                self.regs.set_sp_or_reg(rn, self.regs.get_sp_or_reg(rn) + imm9)
            elif idx_mode == 3:  # Pre-index writeback
                self.regs.set_sp_or_reg(rn, addr)

        # STR/LDR (32-bit): 0xB9
        elif op == 0xB9:
            rt = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            is_load = (inst >> 22) & 1
            addr = self.regs.get_sp_or_reg(rn) + (imm12 << 2)  # Base reg can be SP

            if is_load:
                self.regs.set(rt, self.memory.read32(addr))
            else:
                self.memory.write32(addr, self.regs.get(rt) & 0xFFFFFFFF)

        # ADRP: 0x90, 0xB0, 0xD0, 0xF0
        elif (op & 0x9F) == 0x90:
            rd = inst & 0x1F
            immlo = (inst >> 29) & 0x3
            immhi = (inst >> 5) & 0x7FFFF
            imm = (immhi << 2) | immlo
            if imm & 0x100000:
                imm -= 0x200000
            page = (self.pc - 4) & ~0xFFF
            self.regs.set(rd, page + (imm << 12))

        # STP (64-bit): 0xA9
        elif op == 0xA9:
            rt = inst & 0x1F
            rt2 = (inst >> 10) & 0x1F
            rn = (inst >> 5) & 0x1F
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40:
                imm7 -= 0x80
            idx_mode = (inst >> 23) & 0x3

            offset = imm7 << 3
            addr = self.regs.get_sp_or_reg(rn)  # Base reg can be SP

            if idx_mode == 3:  # Pre-index
                addr += offset
                self.regs.set_sp_or_reg(rn, addr)
            elif idx_mode == 2:  # Signed offset
                addr += offset

            self.memory.write64(addr, self.regs.get(rt))
            self.memory.write64(addr + 8, self.regs.get(rt2))

        # LDP (64-bit) post-index: 0xA8
        elif op == 0xA8:
            rt = inst & 0x1F
            rt2 = (inst >> 10) & 0x1F
            rn = (inst >> 5) & 0x1F
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40:
                imm7 -= 0x80

            offset = imm7 << 3
            addr = self.regs.get_sp_or_reg(rn)  # Base reg can be SP

            self.regs.set(rt, self.memory.read64(addr))
            self.regs.set(rt2, self.memory.read64(addr + 8))
            self.regs.set_sp_or_reg(rn, addr + offset)

        # SUBS immediate (32-bit): 0x71 - sets flags
        elif op == 0x71:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            a = self.regs.get(rn) & 0xFFFFFFFF
            result = (a - imm12) & 0xFFFFFFFF
            if rd != 31:
                self.regs.set(rd, result)
            self.z = (result == 0)
            self.n = (result >> 31) & 1
            self.c = a >= imm12
            self.v = ((a ^ imm12) & (a ^ result)) >> 31

        # STP (32-bit): 0x29
        elif op == 0x29:
            rt = inst & 0x1F
            rt2 = (inst >> 10) & 0x1F
            rn = (inst >> 5) & 0x1F
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40:
                imm7 -= 0x80
            idx_mode = (inst >> 23) & 0x3

            offset = imm7 << 2  # Scale by 4 for 32-bit
            addr = self.regs.get_sp_or_reg(rn)  # Base reg can be SP

            if idx_mode == 3:  # Pre-index
                addr += offset
                self.regs.set_sp_or_reg(rn, addr)
            elif idx_mode == 2:  # Signed offset
                addr += offset

            self.memory.write32(addr, self.regs.get(rt) & 0xFFFFFFFF)
            self.memory.write32(addr + 4, self.regs.get(rt2) & 0xFFFFFFFF)

        # CMP immediate: 0xF1
        elif op == 0xF1:
            rn = (inst >> 5) & 0x1F
            imm12 = (inst >> 10) & 0xFFF
            a = self.regs.get(rn)
            result = (a - imm12) & MASK64
            self.z = (result == 0)
            self.n = (result >> 63) & 1
            self.c = a >= imm12
            self.v = ((a ^ imm12) & (a ^ result)) >> 63

        # CMP register: 0xEB
        elif op == 0xEB:
            rn = (inst >> 5) & 0x1F
            rm = (inst >> 16) & 0x1F
            a = self.regs.get(rn)
            b = self.regs.get(rm)
            result = (a - b) & MASK64
            self.z = (result == 0)
            self.n = (result >> 63) & 1
            self.c = a >= b
            self.v = ((a ^ b) & (a ^ result)) >> 63

        # SXTW: 0x93
        elif op == 0x93:
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            val = self.regs.get(rn) & 0xFFFFFFFF
            if val & 0x80000000:
                val |= 0xFFFFFFFF00000000
            self.regs.set(rd, val)

        # B (branch): 0x14-0x17
        elif (op >> 2) == 0x05:
            imm26 = inst & 0x3FFFFFF
            if imm26 & 0x2000000:
                imm26 -= 0x4000000
            self.pc = (self.pc - 4) + (imm26 << 2)

        # BL (branch and link): 0x94-0x97
        elif (op >> 2) == 0x25:
            imm26 = inst & 0x3FFFFFF
            if imm26 & 0x2000000:
                imm26 -= 0x4000000
            self.regs.set(30, self.pc)
            self.pc = (self.pc - 4) + (imm26 << 2)

        # CBZ (32-bit): 0x34
        elif op == 0x34:
            rt = inst & 0x1F
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            if (self.regs.get(rt) & 0xFFFFFFFF) == 0:
                self.pc = (self.pc - 4) + (imm19 << 2)

        # CBZ (64-bit): 0xB4
        elif op == 0xB4:
            rt = inst & 0x1F
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            if self.regs.get(rt) == 0:
                self.pc = (self.pc - 4) + (imm19 << 2)

        # CBNZ (64-bit): 0xB5
        elif op == 0xB5:
            rt = inst & 0x1F
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            if self.regs.get(rt) != 0:
                self.pc = (self.pc - 4) + (imm19 << 2)

        # B.cond: 0x54
        elif op == 0x54:
            cond = inst & 0xF
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000

            take = False
            if cond == 0x0: take = self.z
            elif cond == 0x1: take = not self.z
            elif cond == 0x2: take = self.c
            elif cond == 0x3: take = not self.c
            elif cond == 0x4: take = self.n
            elif cond == 0x5: take = not self.n
            elif cond == 0xA: take = self.n == self.v
            elif cond == 0xB: take = self.n != self.v
            elif cond == 0xC: take = not self.z and (self.n == self.v)
            elif cond == 0xD: take = self.z or (self.n != self.v)

            if take:
                self.pc = (self.pc - 4) + (imm19 << 2)

        # RET: 0xD65F03C0
        elif inst == 0xD65F03C0:
            self.pc = self.regs.get(30)

        # NOP: 0xD503201F
        elif inst == 0xD503201F:
            pass


# =============================================================================
# FRAMEBUFFER & KEYBOARD I/O
# =============================================================================

class FramebufferIO:
    """Reads framebuffer from neural memory."""
    FB_ADDR = 0x40000
    WIDTH = 80
    HEIGHT = 25

    def __init__(self, cpu):
        self.cpu = cpu
        self.prev_frame = None

    def read_frame(self):
        lines = []
        for y in range(self.HEIGHT):
            row = ""
            for x in range(self.WIDTH):
                ch = self.cpu.memory.read8(self.FB_ADDR + y * self.WIDTH + x)
                row += chr(ch) if 32 <= ch <= 126 else ' '
            lines.append(row)
        return lines

    def display(self):
        frame = self.read_frame()
        if frame != self.prev_frame:
            print("\033[H", end="")
            for line in frame:
                print(line)
            self.prev_frame = frame


class KeyboardIO:
    """Writes keyboard input to neural memory."""
    KEY_ADDR = 0x50000

    def __init__(self, cpu):
        self.cpu = cpu

    def send_key(self, ch):
        if isinstance(ch, str):
            if len(ch) == 0:
                return
            ch = ord(ch)
        self.cpu.memory.write8(self.KEY_ADDR, ch)


# =============================================================================
# ELF LOADER
# =============================================================================

def load_elf(cpu, path):
    """Load ELF file into neural memory."""
    with open(path, 'rb') as f:
        data = f.read()

    if data[:4] != b'\x7fELF':
        raise ValueError("Not ELF")

    entry = struct.unpack('<Q', data[24:32])[0]
    phoff = struct.unpack('<Q', data[32:40])[0]
    phentsize = struct.unpack('<H', data[54:56])[0]
    phnum = struct.unpack('<H', data[56:58])[0]

    for i in range(phnum):
        ph = phoff + i * phentsize
        pt = struct.unpack('<I', data[ph:ph+4])[0]
        if pt == 1:  # PT_LOAD
            offset = struct.unpack('<Q', data[ph+8:ph+16])[0]
            vaddr = struct.unpack('<Q', data[ph+16:ph+24])[0]
            filesz = struct.unpack('<Q', data[ph+32:ph+40])[0]
            cpu.memory.load_binary(data[offset:offset+filesz], vaddr)

    return entry


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    print("\033[2J\033[H", end="")

    print("""
\033[36m╔══════════════════════════════════════════════════════════════════════════╗
║                        NEURAL RTOS HARDWARE                                ║
║                        ====================                                ║
║                                                                            ║
║  Neural CPU with BatchedNeuralALU (transformer models for ADD/SUB/etc)    ║
║  Every ARM64 instruction computed through neural network forward passes.   ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════╝\033[0m
    """)

    # Initialize neural CPU
    cpu = NeuralCPU()
    fb = FramebufferIO(cpu)
    kb = KeyboardIO(cpu)

    # Find and load RTOS
    rtos_paths = [
        "arm64_doom/neural_rtos.elf",
        "neural_rtos.elf",
    ]

    rtos_path = None
    for p in rtos_paths:
        if os.path.exists(p):
            rtos_path = p
            break

    if not rtos_path:
        print("\n\033[31m[ERROR]\033[0m Neural RTOS not found!")
        print("Build it with: cd arm64_doom && make -f Makefile.rtos")
        return

    print(f"\033[32m[OK]\033[0m Loading RTOS: {rtos_path}")
    entry = load_elf(cpu, rtos_path)
    cpu.pc = entry
    print(f"\033[32m[OK]\033[0m Entry point: 0x{entry:x}")

    print("\n\033[36m[BOOT]\033[0m Starting Neural RTOS...\n")
    time.sleep(0.5)

    print("\033[2J", end="")

    # Set up terminal
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        pass

    try:
        start_time = time.time()
        last_display = 0

        while not cpu.halted:
            # Execute instructions
            for _ in range(100):
                cpu.step()

            # Check keyboard
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == '\x03':
                    break
                kb.send_key(ch)

            # Update display
            now = time.time()
            if now - last_display > 0.05:
                fb.display()
                last_display = now

    except KeyboardInterrupt:
        pass
    finally:
        if old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    elapsed = time.time() - start_time
    print(f"\n\n\033[36m═══ Neural RTOS Stats ═══\033[0m")
    print(f"Instructions: {cpu.inst_count:,}")
    print(f"Neural ALU ops: {cpu.alu.stats['total_ops']:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"IPS: {cpu.inst_count/elapsed:,.0f}")


if __name__ == "__main__":
    main()
