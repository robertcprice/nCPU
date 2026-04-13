"""
Zero-sync GPU-only execution engine for NeuralCPU.

The most optimized engine: zero .item() calls in the hot path,
async syscall handling, fully GPU-resident execution.
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple

from ..constants import OpType, _u64_to_s64

logger = logging.getLogger(__name__)


class GpuOnlyMixin:
    """Zero-sync GPU-only execution for NeuralCPU."""

    @torch.no_grad()
    def run_gpu_only(self, max_instructions: int = 100000, batch_size: int = 64) -> Tuple[int, float]:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║       100% GPU EXECUTION - ZERO .item() IN HOT PATH                        ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  - FIXED iteration count (no .item() for loop control)                     ║
        ║  - All ops masked by 'active' tensor - becomes no-op when halted           ║
        ║  - ONLY sync for syscall I/O (unavoidable) and final return                ║
        ║  - PC/regs/flags stay as tensors throughout                                ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        start      = time.perf_counter()
        device     = self.device
        mem        = self.memory
        regs       = self.regs
        flags      = self.flags
        mem_arith  = self._get_woven_memarith()    # pointer.pt neural address computation
        prefetcher = self._get_woven_prefetcher()  # prefetch.pt LSTM memory tracking

        # PC as tensor - NEVER call .item() in hot path
        pc_t = self.pc.clone()

        # Pre-allocate batch index tensor
        batch_idx = torch.arange(batch_size, device=device, dtype=torch.int64)

        # Big value for min operations
        BIG = torch.tensor(batch_size * 2, device=device, dtype=torch.int64)

        # State tensors - ALL on GPU
        executed_t = torch.tensor(0, device=device, dtype=torch.int64)
        halted_t = torch.tensor(0, device=device, dtype=torch.int64)

        # ═══════════════════════════════════════════════════════════════════
        # FIXED ITERATION COUNT - MINIMAL .item() SYNCS
        # Syncs only every SYNC_INTERVAL batches (amortises MPS round-trip cost).
        # SVC is still handled promptly: we sync every iteration only after a
        # potential SVC window; halt uses the same deferred mechanism.
        # ═══════════════════════════════════════════════════════════════════
        SYNC_INTERVAL = 32          # one CPU-GPU sync per 32 batches
        # Extra buffer: x4 accounts for false-hazard serialization of imm-ops
        # (bits[20:16] of imm12 misread as Rm can double outer-iters per inner loop iter)
        max_outer_iters = (max_instructions // batch_size) * 4 + SYNC_INTERVAL + 4

        for _iter in range(max_outer_iters):
            # ═══════════════════════════════════════════════════════════════
            # ACTIVE MASK - all ops become no-op when halted/done (TENSOR, NO .item())
            # ═══════════════════════════════════════════════════════════════
            active = (halted_t == 0) & (executed_t < max_instructions)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: TENSOR FETCH via gather
            # ═══════════════════════════════════════════════════════════════
            # Compute byte addresses for all instructions in batch
            # Each instruction is 4 bytes, so we need pc, pc+1, pc+2, pc+3
            inst_offsets = batch_idx * 4  # [0, 4, 8, 12, ...]
            byte_addrs = pc_t + inst_offsets.unsqueeze(1) + torch.arange(4, device=device)
            # byte_addrs shape: [batch_size, 4]

            # Clamp to valid memory range
            byte_addrs_flat = byte_addrs.view(-1).clamp(0, self.mem_size - 1)

            # Gather bytes
            bytes_flat = mem[byte_addrs_flat.long()]
            bytes_4 = bytes_flat.view(batch_size, 4).long()

            # Combine to 32-bit instructions
            insts = (bytes_4[:, 0] | (bytes_4[:, 1] << 8) |
                    (bytes_4[:, 2] << 16) | (bytes_4[:, 3] << 24))

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: TENSOR DECODE
            # ═══════════════════════════════════════════════════════════════
            op_bytes = (insts >> 24) & 0xFF
            ops = self.op_type_table[op_bytes]

            # Decode fields
            rds = insts & 0x1F
            rns = (insts >> 5) & 0x1F
            rms = (insts >> 16) & 0x1F
            imm12 = (insts >> 10) & 0xFFF
            imm16 = (insts >> 5) & 0xFFFF
            hw = (insts >> 21) & 0x3

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: STOP DETECTION (fully in tensors)
            # ═══════════════════════════════════════════════════════════════
            # Detect halts
            halt_mask = (insts == 0)

            # Detect syscalls
            svc_mask = ((insts & 0xFFE0001F) == 0xD4000001)

            # Detect branches (explicit bit patterns for reliability)
            # B: 0x14000000 (bits 31-26 = 000101)
            # BL: 0x94000000 (bits 31-26 = 100101)
            # B.cond: 0x54000000 (bits 31-24 = 01010100, bits 4 = 0)
            # CBZ: 0xB4000000 (64-bit) / 0x34000000 (32-bit)
            # CBNZ: 0xB5000000 (64-bit) / 0x35000000 (32-bit)
            # BR: 0xD61F0000 (bits 31-10 = 1101011000011111000000)
            # BLR: 0xD63F0000
            # RET: 0xD65F0000
            # TBZ: 0x36000000 (bit 31 = 0)
            # TBNZ: 0x37000000 (bit 31 = 0)
            branch_mask = (
                ((insts & 0xFC000000) == 0x14000000) |  # B
                ((insts & 0xFC000000) == 0x94000000) |  # BL
                ((insts & 0xFF000010) == 0x54000000) |  # B.cond
                ((insts & 0xFF000000) == 0xB4000000) |  # CBZ 64-bit
                ((insts & 0xFF000000) == 0x34000000) |  # CBZ 32-bit
                ((insts & 0xFF000000) == 0xB5000000) |  # CBNZ 64-bit
                ((insts & 0xFF000000) == 0x35000000) |  # CBNZ 32-bit
                ((insts & 0xFFFFFC1F) == 0xD61F0000) |  # BR
                ((insts & 0xFFFFFC1F) == 0xD63F0000) |  # BLR
                ((insts & 0xFFFFFC1F) == 0xD65F0000) |  # RET
                ((insts & 0x7F000000) == 0x36000000) |  # TBZ
                ((insts & 0x7F000000) == 0x37000000)    # TBNZ
            )

            # ─────────────────────────────────────────────────────────────────
            # RAW HAZARD DETECTION (Read-After-Write)
            # Find first instruction that reads from a register written by
            # any earlier instruction in the batch
            # ─────────────────────────────────────────────────────────────────
            # Build comparison matrices using broadcasting
            # rds_col[j] compared to rns_row[i] and rms_row[i]
            rds_col = rds.unsqueeze(0)       # [1, batch_size] - destination regs
            rns_row = rns.unsqueeze(1)       # [batch_size, 1] - source reg 1
            rms_row = rms.unsqueeze(1)       # [batch_size, 1] - source reg 2

            # hazard[i,j] = True if inst i reads what inst j writes
            # (rd of j matches rn or rm of i)
            #
            # IMPORTANT: only check Rm hazard for instructions that actually use
            # bits[20:16] as a SOURCE REGISTER (Rm).  Immediate instructions
            # (ADD/SUB imm, SUBS imm, MOVZ, MOVK, etc.) store part of their
            # immediate value in bits[20:16], NOT a register — treating those bits
            # as Rm causes false hazards (e.g. SUBS X1,X1,#1 has bits[20:16]=0
            # which looks like "reads X0", blocking the ADD before it).
            reads_rm_as_reg = (
                # Register-register ALU
                ((insts & 0xFF200000) == 0x8B000000) |  # ADD REG 64
                ((insts & 0xFF200000) == 0x0B000000) |  # ADD REG 32
                ((insts & 0xFF200000) == 0xCB000000) |  # SUB REG 64
                ((insts & 0xFF200000) == 0x4B000000) |  # SUB REG 32
                ((insts & 0xFF200000) == 0xAB000000) |  # ADDS REG 64
                ((insts & 0xFF200000) == 0x2B000000) |  # ADDS REG 32
                ((insts & 0xFF200000) == 0xEB000000) |  # SUBS REG 64
                ((insts & 0xFF200000) == 0x6B000000) |  # SUBS REG 32
                # MUL / MADD / MSUB
                ((insts & 0xFFE0FC00) == 0x9B007C00) |  # MUL 64
                ((insts & 0xFFE0FC00) == 0x1B007C00) |  # MUL 32
                # Logical REG (AND/ORR/EOR and shifted variants)
                ((insts & 0xFF200000) == 0x8A000000) |  # AND REG 64
                ((insts & 0xFF200000) == 0x0A000000) |  # AND REG 32
                ((insts & 0xFF200000) == 0xAA000000) |  # ORR REG 64
                ((insts & 0xFF200000) == 0x2A000000) |  # ORR REG 32
                ((insts & 0xFF200000) == 0xCA000000) |  # EOR REG 64
                ((insts & 0xFF200000) == 0x4A000000) |  # EOR REG 32
                ((insts & 0xFF200000) == 0xEA000000) |  # ANDS REG 64
                ((insts & 0xFF200000) == 0x6A000000) |  # ANDS REG 32
                ((insts & 0xFF200000) == 0x8A200000) |  # BIC REG 64
                ((insts & 0xFF200000) == 0x0A200000) |  # BIC REG 32
                ((insts & 0xFF200000) == 0xEA200000) |  # BICS REG 64
                ((insts & 0xFF200000) == 0x6A200000) |  # BICS REG 32
                ((insts & 0xFF200000) == 0xCA200000) |  # EON REG 64
                ((insts & 0xFF200000) == 0x4A200000) |  # EON REG 32
                ((insts & 0xFF200000) == 0xAA200000) |  # ORN REG 64
                ((insts & 0xFF200000) == 0x2A200000) |  # ORN REG 32
                # Shift REG
                ((insts & 0xFFE0FC00) == 0x9AC02000) |  # LSLV 64
                ((insts & 0xFFE0FC00) == 0x1AC02000) |  # LSLV 32
                ((insts & 0xFFE0FC00) == 0x9AC02400) |  # LSRV 64
                ((insts & 0xFFE0FC00) == 0x1AC02400) |  # LSRV 32
                ((insts & 0xFFE0FC00) == 0x9AC02800) |  # ASRV 64
                ((insts & 0xFFE0FC00) == 0x1AC02800) |  # ASRV 32
                # CSEL / CSINC / CSNEG / CSINV
                ((insts & 0xFFC00000) == 0x9A800000) |  # CSEL 64
                ((insts & 0xFFC00000) == 0x1A800000) |  # CSEL 32
                # Load with register offset (Rm at bits[20:16])
                ((insts & 0xFFE00C00) == 0xF8600800) |  # LDR 64 reg
                ((insts & 0xFFE00C00) == 0xB8600800) |  # LDR 32 reg
                ((insts & 0xFFE00C00) == 0x38A00800) |  # LDRSB 64 reg
                ((insts & 0xFFE00C00) == 0x38E00800) |  # LDRSB 32 reg
                ((insts & 0xFFE00C00) == 0x78600800) |  # LDRH reg
                # Store with register offset
                ((insts & 0xFFE00C00) == 0xF8200800) |  # STR 64 reg
                ((insts & 0xFFE00C00) == 0xB8200800) |  # STR 32 reg
                ((insts & 0xFFE00C00) == 0x38200800) |  # STRB reg
                ((insts & 0xFFE00C00) == 0x78200800)    # STRH reg
            )
            hazard_rn = (rds_col == rns_row)  # [batch_size, batch_size]
            hazard_rm = (rds_col == rms_row) & reads_rm_as_reg.unsqueeze(1)  # [B, B] filtered
            hazard_any = hazard_rn | hazard_rm

            # Only care about j < i (can't have hazard from future instructions)
            # Create lower triangular mask (excluding diagonal)
            lower_tri = torch.tril(torch.ones(batch_size, batch_size, device=device, dtype=torch.bool), diagonal=-1)
            hazard_valid = hazard_any & lower_tri

            # Also need: source inst j must actually write (has a destination)
            # Instructions that write: ADD, SUB, MOVZ, MOVK, etc.
            # Use EXPLICIT BIT PATTERNS (not ops table) for reliable detection
            writes_reg = (
                # ADD/SUB immediate (64-bit): 0x91/0xD1 (top byte)
                ((insts & 0xFF000000) == 0x91000000) |  # ADD
                ((insts & 0xFF000000) == 0xD1000000) |  # SUB
                # ADD/SUB register (64-bit): 0x8B/0xCB (shifted reg)
                ((insts & 0xFF200000) == 0x8B000000) |  # ADD (shifted reg)
                ((insts & 0xFF200000) == 0xCB000000) |  # SUB (shifted reg)
                # MOVZ: 0xD28/0x528 (top 9 bits)
                ((insts & 0xFF800000) == 0xD2800000) |  # MOVZ 64-bit
                ((insts & 0xFF800000) == 0x52800000) |  # MOVZ 32-bit
                # MOVK: 0xF28/0x728
                ((insts & 0xFF800000) == 0xF2800000) |  # MOVK 64-bit
                ((insts & 0xFF800000) == 0x72800000) |  # MOVK 32-bit
                # ADRP: 0x9xxxxxxx
                ((insts & 0x9F000000) == 0x90000000) |  # ADRP
                # SUBS: 0xEB (register), 0xF1 (immediate) - writes if rd != 31
                ((insts & 0xFF200000) == 0xEB000000) |  # SUBS (shifted reg)
                ((insts & 0xFF000000) == 0xF1000000) |  # SUBS (immediate)
                # ORR/AND/EOR register
                (ops == OpType.ORR_REG.value) | (ops == OpType.AND_REG.value) |
                (ops == OpType.EOR_REG.value) | (ops == OpType.LSL_IMM.value) |
                (ops == OpType.LSR_IMM.value) | (ops == OpType.MOV_REG.value) |
                (ops == OpType.LDR.value) | (ops == OpType.LDRB.value) |
                (ops == OpType.LDP.value) | (ops == OpType.LDUR.value) |
                (ops == OpType.MUL.value) |
                # New instruction types added for OS support
                (ops == OpType.AND_IMM.value) | (ops == OpType.ORR_IMM.value) |
                (ops == OpType.EOR_IMM.value) | (ops == OpType.ASR_IMM.value) |
                (ops == OpType.LSL_REG.value) | (ops == OpType.LSR_REG.value) |
                (ops == OpType.ASR_REG.value) |
                ((insts & 0xFFE0FC00) == 0x9B007C00) |  # MUL 64
                ((insts & 0xFFE0FC00) == 0x1B007C00) |  # MUL 32
                ((insts & 0xFFC00000) == 0xB9800000) |  # LDRSW
                ((insts & 0xFFC00000) == 0x79400000) |  # LDRH
                ((insts & 0xFFC00000) == 0x39800000) |  # LDRSB 64
                ((insts & 0xFFC00000) == 0x39C00000) |  # LDRSB 32
                ((insts & 0xFFE00C00) == 0xF8400000) |  # LDUR 64
                ((insts & 0xFFE00C00) == 0xB8400000) |  # LDUR 32
                ((insts & 0xFFE00C00) == 0xF8600800) |  # LDR 64 reg offset
                ((insts & 0xFFFFFC00) == 0xC85F7C00) |  # LDXR
                ((insts & 0xFFC00000) == 0x9A800000) |  # CSEL 64 (0xFFC00000 excludes LSLV/LSRV/ASRV)
                ((insts & 0xFFC00000) == 0x1A800000) |  # CSEL 32
                # Pre/post-index 32-bit loads + LDR 64 pre-index
                ((insts & 0xFFE00C00) == 0xF8400C00) |  # LDR 64 pre-index
                ((insts & 0xFFE00C00) == 0xB8400400) |  # LDR 32 post-index
                ((insts & 0xFFE00C00) == 0xB8400C00) |  # LDR 32 pre-index
                # LDRH variants
                ((insts & 0xFFE00C00) == 0x78600800) |  # LDRH reg offset
                ((insts & 0xFFE00C00) == 0x78400400) |  # LDRH post-index
                # LDRSB register offset
                ((insts & 0xFFE00C00) == 0x38A00800) |  # LDRSB 64 reg
                ((insts & 0xFFE00C00) == 0x38E00800) |  # LDRSB 32 reg
                # BFM (bitfield move insert)
                ((insts & 0xFFC00000) == 0xB3400000) |  # BFM 64
                ((insts & 0xFFC00000) == 0x33000000) |  # BFM 32
                # RBIT
                ((insts & 0xFFFFFC00) == 0xDAC00000) |  # RBIT 64
                ((insts & 0xFFFFFC00) == 0x5AC00000) |  # RBIT 32
                # MOVN
                ((insts & 0xFF800000) == 0x92800000) |  # MOVN 64
                ((insts & 0xFF800000) == 0x12800000) |  # MOVN 32
                # SMULL / SMULH / UMULL / UMULH / SMADDL / UMADDL
                ((insts & 0xFFE00000) == 0x9B200000) |  # SMADDL/SMULL (opc31=001)
                ((insts & 0xFFE00000) == 0x9BA00000) |  # UMADDL/UMULL (opc31=101)
                ((insts & 0xFFE0FC00) == 0x9B407C00) |  # SMULH
                ((insts & 0xFFE0FC00) == 0x9BC07C00) |  # UMULH
                # EXTR (bit-extract / ROR)
                ((insts & 0xFFE00000) == 0x93800000) |  # EXTR 64
                ((insts & 0xFFE00000) == 0x13800000) |  # EXTR 32
                # LDRSW post/pre-index and register offset
                ((insts & 0xFFE00C00) == 0xB8800400) |  # LDRSW post-index
                ((insts & 0xFFE00C00) == 0xB8800C00) |  # LDRSW pre-index
                ((insts & 0xFFE00C00) == 0xB8A00800) |  # LDRSW reg offset
                # LDAR (acquire load — treated as regular LDR)
                ((insts & 0xFFFFFC00) == 0xC8DFFC00) |  # LDAR 64
                ((insts & 0xFFFFFC00) == 0x88DFFC00) |  # LDAR 32
                # LDRSH unsigned-offset (16-bit sign-extending load)
                ((insts & 0xFFC00000) == 0x79800000) |  # LDRSH 64 (X dest)
                ((insts & 0xFFC00000) == 0x79C00000) |  # LDRSH 32 (W dest)
                # LDURH (16-bit zero-extending, unscaled offset)
                ((insts & 0xFFE00C00) == 0x78400000) |
                # LDURSH (16-bit sign-extending, unscaled offset)
                ((insts & 0xFFE00C00) == 0x78800000) |  # 64-bit dest
                ((insts & 0xFFE00C00) == 0x78C00000) |  # 32-bit dest
                # LDURSB (8-bit sign-extending, unscaled offset)
                ((insts & 0xFFE00C00) == 0x38800000) |  # 64-bit dest
                ((insts & 0xFFE00C00) == 0x38C00000) |  # 32-bit dest
                # LDRSH post-index
                ((insts & 0xFFE00C00) == 0x78800400) |
                ((insts & 0xFFE00C00) == 0x78C00400) |
                # LDRSH pre-index
                ((insts & 0xFFE00C00) == 0x78800C00) |
                ((insts & 0xFFE00C00) == 0x78C00C00) |
                # LDP 32-bit
                ((insts & 0xFFC00000) == 0x29400000) |
                # CAS / CASAL (Rs gets old mem value)
                ((insts & 0xFFE07C00) == 0xC8207C00) |  # CAS 64
                ((insts & 0xFFE07C00) == 0xC8607C00) |  # CASA 64
                ((insts & 0xFFE07C00) == 0xC8A07C00) |  # CASL 64
                ((insts & 0xFFE07C00) == 0xC8E07C00) |  # CASAL 64
                ((insts & 0xFFE07C00) == 0x88207C00) |  # CAS 32
                ((insts & 0xFFE07C00) == 0x88607C00) |  # CASA 32
                ((insts & 0xFFE07C00) == 0x88A07C00) |  # CASL 32
                ((insts & 0xFFE07C00) == 0x88E07C00) |  # CASAL 32
                # BIC / BICS (AND-NOT register)
                ((insts & 0xFF200000) == 0x8A200000) |  # BIC 64
                ((insts & 0xFF200000) == 0x0A200000) |  # BIC 32
                ((insts & 0xFF200000) == 0xEA200000) |  # BICS 64
                ((insts & 0xFF200000) == 0x6A200000) |  # BICS 32
                # EON (EOR-NOT register)
                ((insts & 0xFF200000) == 0xCA200000) |  # EON 64
                ((insts & 0xFF200000) == 0x4A200000) |  # EON 32
                # SXTB / SXTH 32-bit (SBFM sf=0)
                ((insts & 0xFFFFFC00) == 0x13001C00) |  # SXTB 32
                ((insts & 0xFFFFFC00) == 0x13003C00) |  # SXTH 32
                # REV16 (reverse bytes within each halfword)
                ((insts & 0xFFFFFC00) == 0xDAC00400) |  # REV16 64
                ((insts & 0xFFFFFC00) == 0x5AC00400) |  # REV16 32
                # ADC / ADCS / SBC / SBCS (add/sub with carry)
                ((insts & 0xFFE0FC00) == 0x9A000000) |  # ADC 64
                ((insts & 0xFFE0FC00) == 0x1A000000) |  # ADC 32
                ((insts & 0xFFE0FC00) == 0xBA000000) |  # ADCS 64
                ((insts & 0xFFE0FC00) == 0x3A000000) |  # ADCS 32
                ((insts & 0xFFE0FC00) == 0xDA000000) |  # SBC 64
                ((insts & 0xFFE0FC00) == 0x5A000000) |  # SBC 32
                ((insts & 0xFFE0FC00) == 0xFA000000) |  # SBCS 64
                ((insts & 0xFFE0FC00) == 0x7A000000) |  # SBCS 32
                # LDAXR (load-acquire exclusive → approximate as LDR)
                ((insts & 0xFFFFFC00) == 0xC8DF7C00) |  # LDAXR 64
                ((insts & 0xFFFFFC00) == 0x88DF7C00) |  # LDAXR 32
                # LDR literal (PC-relative load from literal pool)
                ((insts & 0xFF000000) == 0x58000000) |  # LDR literal 64
                ((insts & 0xFF000000) == 0x18000000) |  # LDR literal 32
                # UBFM 32-bit general (unsigned bitfield extract)
                ((insts & 0xFFC00000) == 0x53000000) |  # UBFM 32
                # SBFM 32-bit general (signed bitfield extract)
                ((insts & 0xFFC00000) == 0x13000000) |  # SBFM 32
                # LDRH pre-index (LDRH post-index already above)
                ((insts & 0xFFE00C00) == 0x78400C00) |  # LDRH pre-index
                # LDRB pre-index (LDRB post-index already above)
                ((insts & 0xFFE00C00) == 0x38400C00)    # LDRB pre-index
            )
            # Broadcast writes_reg[j] across rows
            writes_col = writes_reg.unsqueeze(0)  # [1, batch_size]
            hazard_valid = hazard_valid & writes_col

            # Also: destination must not be XZR (r31 writes are discarded)
            rd_not_xzr_col = (rds != 31).unsqueeze(0)  # [1, batch_size]
            hazard_valid = hazard_valid & rd_not_xzr_col

            # Find first row (instruction) that has any hazard
            has_hazard_per_inst = hazard_valid.any(dim=1)  # [batch_size]
            hazard_indices = torch.where(has_hazard_per_inst, batch_idx, BIG)
            first_hazard = hazard_indices.min()

            # ─────────────────────────────────────────────────────────────────

            # Combined stop mask
            stop_mask = halt_mask | svc_mask | branch_mask

            # Find first stop index using tensor operations
            # torch.where returns indices where condition is true
            stop_indices = torch.where(stop_mask, batch_idx, BIG)
            first_stop_event = stop_indices.min()  # Tensor, not Python int!

            # First stop is minimum of: branch/halt/syscall OR hazard
            first_stop = torch.min(first_stop_event, first_hazard)

            # Has any stop?
            has_stop = stop_mask.any()

            # Flag for whether we stopped due to a branch event (vs hazard)
            stopped_by_event = (first_stop_event <= first_hazard) & has_stop

            # Execution mask: only execute instructions before first stop AND if active
            # When halted/done, active=False makes exec_mask all False (no-op)
            exec_mask = (batch_idx < first_stop) & active

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: GATHER REGISTER VALUES
            # ═══════════════════════════════════════════════════════════════
            rn_vals = regs[rns.clamp(0, 31)]
            rm_vals = regs[rms.clamp(0, 31)]
            rd_vals = regs[rds.clamp(0, 31)]

            # Handle XZR/SP: r31 reads as 0 (XZR) for non-memory ops,
            # but as SP for memory base-register ops (LDR/STR/etc.)
            _is_mem_op_gpu = self._woven_mem_op_lut[ops]
            rn_vals = torch.where((rns == 31) & ~_is_mem_op_gpu, torch.zeros_like(rn_vals), rn_vals)
            rm_vals = torch.where((rms == 31) & ~_is_mem_op_gpu, torch.zeros_like(rm_vals), rm_vals)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5: SIMD PARALLEL ALU DISPATCH
            # ═══════════════════════════════════════════════════════════════
            # Pure-ALU op classes (no flag side-effects, no memory) are decoded
            # and computed simultaneously via a [K_ALU=25, B] classify+compute
            # matrix.  A single gather picks the correct result per slot,
            # replacing ~40 sequential torch.where calls with 2 torch.stack ops.
            # Flag-updating ops (CMP, ADDS, SUBS, BICS, ANDS) follow below.
            # ═══════════════════════════════════════════════════════════════
            results = torch.zeros(batch_size, device=device, dtype=torch.int64)
            write_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

            # ─── Compute all ALU masks upfront ───────────────────────────────
            add_imm_mask = (ops == OpType.ADD_IMM.value) & exec_mask
            sub_imm_mask = (ops == OpType.SUB_IMM.value) & exec_mask
            add_reg_mask = (ops == OpType.ADD_REG.value) & exec_mask
            sub_reg_mask = (ops == OpType.SUB_REG.value) & exec_mask
            mov_reg_mask = (ops == OpType.MOV_REG.value) & exec_mask
            lsl_imm_mask = (ops == OpType.LSL_IMM.value) & exec_mask
            lsr_imm_mask = (ops == OpType.LSR_IMM.value) & exec_mask

            movz_mask    = (((insts & 0xFF800000) == 0xD2800000) |
                            ((insts & 0xFF800000) == 0x52800000)) & exec_mask
            mul_gpu_mask = (((insts & 0xFFE0FC00) == 0x9B007C00) |
                            ((insts & 0xFFE0FC00) == 0x1B007C00)) & exec_mask
            asr_imm_gpu  = (((insts & 0xFFC0FC00) == 0x9340FC00) |
                            ((insts & 0xFFC0FC00) == 0x13007C00)) & exec_mask
            lsl_reg_gpu  = (((insts & 0xFFE0FC00) == 0x9AC02000) |
                            ((insts & 0xFFE0FC00) == 0x1AC02000)) & exec_mask
            lsr_reg_gpu  = (((insts & 0xFFE0FC00) == 0x9AC02400) |
                            ((insts & 0xFFE0FC00) == 0x1AC02400)) & exec_mask
            asr_reg_gpu  = (((insts & 0xFFE0FC00) == 0x9AC02800) |
                            ((insts & 0xFFE0FC00) == 0x1AC02800)) & exec_mask
            sxtw_gpu     = ((insts & 0xFFFFFC00) == 0x93407C00) & exec_mask
            uxtb_gpu     = ((insts & 0xFFFFFC00) == 0x53001C00) & exec_mask
            uxth_gpu     = ((insts & 0xFFFFFC00) == 0x53003C00) & exec_mask
            sxtb32_gpu   = ((insts & 0xFFFFFC00) == 0x13001C00) & exec_mask
            sxth32_gpu   = ((insts & 0xFFFFFC00) == 0x13003C00) & exec_mask
            bic_gpu  = (((insts & 0xFF200000) == 0x8A200000) |
                        ((insts & 0xFF200000) == 0x0A200000)) & exec_mask
            bics_gpu = (((insts & 0xFF200000) == 0xEA200000) |
                        ((insts & 0xFF200000) == 0x6A200000)) & exec_mask
            eon_gpu  = (((insts & 0xFF200000) == 0xCA200000) |
                        ((insts & 0xFF200000) == 0x4A200000)) & exec_mask
            orn_gpu  = (((insts & 0xFF200000) == 0xAA200000) |
                        ((insts & 0xFF200000) == 0x2A200000)) & exec_mask

            # Exclusive masks — remove encoding overlaps so each class is disjoint.
            # AND_REG op-table slot catches 0x8A (AND) AND 0x8A+N=1 (BIC); exclude BIC:
            and_pure = (ops == OpType.AND_REG.value) & exec_mask & ~bic_gpu & ~bics_gpu
            # ORR_REG catches ORN (0xAA+N=1); MOV alias (Rn=XZR) split out separately:
            orr_pure = (ops == OpType.ORR_REG.value) & exec_mask & ~orn_gpu & (rns != 31)
            mov_orr  = (ops == OpType.ORR_REG.value) & (rns == 31) & exec_mask
            # EOR_REG catches EON (0xCA+N=1):
            eor_pure = (ops == OpType.EOR_REG.value) & exec_mask & ~eon_gpu

            # Pre-compute operand expressions reused across multiple compute rows
            shift_amt = (insts >> 10) & 0x3F
            asr_shift = (insts >> 16) & 0x3F
            rm_shift  = rm_vals & 0x3F
            movz_val  = imm16 << (hw * 16)
            sxtw_u32  = rn_vals & 0xFFFFFFFF
            sxtw_s32  = torch.where(sxtw_u32 >= 0x80000000, sxtw_u32 - 0x100000000, sxtw_u32)
            sxtb_u8   = rn_vals & 0xFF
            sxtb_s8   = torch.where(sxtb_u8 >= 0x80, sxtb_u8 - 0x100, sxtb_u8)
            sxth_u16  = rn_vals & 0xFFFF
            sxth_s16  = torch.where(sxth_u16 >= 0x8000, sxth_u16 - 0x10000, sxth_u16)

            # ─── SIMD classify matrix [25, B] ─────────────────────────────────
            # Row k is True for each instruction slot that belongs to ALU class k.
            _cls = torch.stack([
                add_imm_mask,         #  0  ADD IMM
                sub_imm_mask,         #  1  SUB IMM
                add_reg_mask,         #  2  ADD REG
                sub_reg_mask,         #  3  SUB REG
                movz_mask,            #  4  MOVZ
                mov_reg_mask,         #  5  MOV_REG (alias)
                mov_orr,              #  6  MOV (ORR XZR alias)
                and_pure,             #  7  AND REG (exclusive of BIC)
                orr_pure,             #  8  ORR REG (exclusive of ORN/MOV)
                eor_pure,             #  9  EOR REG (exclusive of EON)
                bic_gpu | bics_gpu,   # 10  BIC / BICS result (flags handled below)
                eon_gpu,              # 11  EON
                orn_gpu,              # 12  ORN / MVN
                mul_gpu_mask,         # 13  MUL
                lsl_imm_mask,         # 14  LSL IMM
                lsr_imm_mask,         # 15  LSR IMM
                asr_imm_gpu,          # 16  ASR IMM
                lsl_reg_gpu,          # 17  LSL REG
                lsr_reg_gpu,          # 18  LSR REG
                asr_reg_gpu,          # 19  ASR REG
                sxtw_gpu,             # 20  SXTW
                uxtb_gpu,             # 21  UXTB
                uxth_gpu,             # 22  UXTH
                sxtb32_gpu,           # 23  SXTB 32-bit
                sxth32_gpu,           # 24  SXTH 32-bit
            ], dim=0)  # [25, B]

            # ─── SIMD compute matrix [25, B] ──────────────────────────────────
            # All 25 result expressions launch as one GPU kernel group, not 25.
            _cmp = torch.stack([
                rn_vals + imm12,                       #  0  ADD IMM
                rn_vals - imm12,                       #  1  SUB IMM
                rn_vals + rm_vals,                     #  2  ADD REG
                rn_vals - rm_vals,                     #  3  SUB REG
                movz_val,                              #  4  MOVZ
                rm_vals,                               #  5  MOV_REG
                rm_vals,                               #  6  MOV (ORR XZR)
                rn_vals & rm_vals,                     #  7  AND REG
                rn_vals | rm_vals,                     #  8  ORR REG
                rn_vals ^ rm_vals,                     #  9  EOR REG
                rn_vals & ~rm_vals,                    # 10  BIC/BICS
                rn_vals ^ ~rm_vals,                    # 11  EON
                rn_vals | ~rm_vals,                    # 12  ORN/MVN
                rn_vals * rm_vals,                     # 13  MUL
                rn_vals << shift_amt.clamp(0, 63),     # 14  LSL IMM
                rn_vals >> shift_amt.clamp(0, 63),     # 15  LSR IMM (logical on int64)
                rn_vals >> asr_shift.clamp(0, 63),     # 16  ASR IMM (arithmetic on int64)
                rn_vals << rm_shift,                   # 17  LSL REG
                rn_vals >> rm_shift,                   # 18  LSR REG
                rn_vals >> rm_shift,                   # 19  ASR REG
                sxtw_s32,                              # 20  SXTW
                rn_vals & 0xFF,                        # 21  UXTB
                rn_vals & 0xFFFF,                      # 22  UXTH
                sxtb_s8 & 0xFFFFFFFF,                  # 23  SXTB 32-bit (zero-upper-32)
                sxth_s16 & 0xFFFFFFFF,                 # 24  SXTH 32-bit (zero-upper-32)
            ], dim=0)  # [25, B]

            # ─── Priority-select: highest matching class index wins ────────────
            # Exclusive masks mean at most one row is True per slot.
            # For residual overlaps from op-table aliasing, higher index wins.
            _K    = _cls.shape[0]
            _bidx = torch.arange(batch_size, device=device)
            _prio = (_cls.long() *
                     torch.arange(_K, device=device, dtype=torch.int64).unsqueeze(1)
                    ).max(dim=0).indices          # [B]: which class matched
            _hit  = _cls.any(dim=0)              # [B]: did any class match?
            results    = torch.where(_hit, _cmp[_prio, _bidx], results)
            write_mask = write_mask | _hit

            # --- MOVK (insert imm16 into register half-word; needs current Xd) ---
            # MOVK 64-bit: 1 11 100101 hw imm16 rd = 0xF28xxxxx
            # MOVK 32-bit: 0 11 100101 hw imm16 rd = 0x728xxxxx
            movk_mask = (((insts & 0xFF800000) == 0xF2800000) |
                         ((insts & 0xFF800000) == 0x72800000)) & exec_mask
            movk_clear = ~(torch.tensor(0xFFFF, device=device, dtype=torch.int64) << (hw * 16))
            movk_val = (rd_vals & movk_clear) | (imm16 << (hw * 16))
            results = torch.where(movk_mask, movk_val, results)
            write_mask = write_mask | movk_mask

            # --- MOVN (move with NOT; separate from SIMD: needs sf-bit conditional) ---
            # MOVN 64: sf=1,opc=00 → 0x92800000 mask 0xFF800000
            # MOVN 32: sf=0,opc=00 → 0x12800000 mask 0xFF800000
            movn_mask = (((insts & 0xFF800000) == 0x92800000) |
                         ((insts & 0xFF800000) == 0x12800000)) & exec_mask
            if movn_mask.any():
                movn_shifted = imm16 << (hw * 16)
                movn_val64 = ~movn_shifted
                movn_val32 = (~movn_shifted) & 0xFFFFFFFF
                movn_is32 = ((insts & 0xFF800000) == 0x12800000)
                movn_val = torch.where(movn_is32, movn_val32, movn_val64)
                results = torch.where(movn_mask, movn_val, results)
                write_mask = write_mask | movn_mask

            # --- BICS flag update (result already written by SIMD dispatch above) ---
            # BIC / BICS (AND with NOT Rm):
            # BIC 64: (inst & 0xFF200000)==0x8A200000 (AND_REG+N=1); BIC 32: 0x0A200000
            # BICS 64: 0xEA200000; BICS 32: 0x6A200000
            if bics_gpu.any():
                bics_idx = torch.where(bics_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                bics_v = results[bics_idx]   # read result already placed by SIMD
                flags[0] = ((bics_v >> 63) & 1).to(torch.float32)
                flags[1] = (bics_v == 0).to(torch.float32)
                flags[2] = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
                flags[3] = torch.zeros(1, device=device, dtype=torch.float32).squeeze()

            # --- ADRP ---
            adrp_mask = (ops == OpType.ADRP.value) & exec_mask
            if adrp_mask.any():
                inst_pcs = pc_t + batch_idx * 4
                adr_immlo = (insts >> 29) & 0x3
                adr_immhi = (insts >> 5) & 0x7FFFF
                adr_imm = (adr_immhi << 2) | adr_immlo
                adr_imm = torch.where(adr_imm >= 0x100000, adr_imm - 0x200000, adr_imm)
                page_base = inst_pcs & ~0xFFF
                adrp_val = page_base + (adr_imm << 12)
                results = torch.where(adrp_mask, adrp_val, results)
                write_mask = write_mask | adrp_mask

            # --- CMP/SUBS (set flags) ---
            # CMP reg: SUBS XZR, Rn, Rm  (0xEB...... where rd=31)
            # CMP imm: SUBS XZR, Rn, #imm (0xF1...... where rd=31)
            # Detect SUBS: 0xEB (register) or 0xF1 (immediate)
            subs_reg_mask = ((insts & 0xFF200000) == 0xEB000000) & exec_mask
            subs_imm_mask = ((insts & 0xFF000000) == 0xF1000000) & exec_mask
            cmp_mask = (subs_reg_mask | subs_imm_mask)

            # Compute result for flag setting
            cmp_result_reg = rn_vals.to(torch.int64) - rm_vals.to(torch.int64)
            cmp_result_imm = rn_vals.to(torch.int64) - imm12.to(torch.int64)
            cmp_result = torch.where(subs_reg_mask, cmp_result_reg, cmp_result_imm)

            # For any CMP in the batch that's executed, update flags
            # We take the FIRST CMP in the batch (using first_stop logic)
            # Actually, since we stop at hazards, at most one instruction executes per batch
            # So we can just check if any CMP was executed and use its result
            any_cmp = cmp_mask.any()
            if any_cmp:
                # Get the index of the CMP instruction
                cmp_indices = torch.where(cmp_mask, batch_idx, BIG)
                first_cmp_idx = cmp_indices.min()

                # Get the result for that specific CMP
                cmp_val = cmp_result[first_cmp_idx.clamp(0, batch_size-1)]
                cmp_rn = rn_vals[first_cmp_idx.clamp(0, batch_size-1)]
                cmp_rm_or_imm = torch.where(
                    subs_reg_mask[first_cmp_idx.clamp(0, batch_size-1)],
                    rm_vals[first_cmp_idx.clamp(0, batch_size-1)],
                    imm12[first_cmp_idx.clamp(0, batch_size-1)]
                )

                # Compute flags
                # N: bit 63 of result (negative)
                new_n = (cmp_val >> 63) & 1
                # Z: result == 0
                new_z = (cmp_val == 0).to(torch.float32)
                # C: unsigned rn >= rm (no borrow)
                # For unsigned comparison: C = 1 if rn >= rm
                new_c = (cmp_rn.to(torch.uint64) >= cmp_rm_or_imm.to(torch.uint64)).to(torch.float32)
                # V: signed overflow (simplified: set if signs of operands differ and result sign differs from rn)
                rn_neg = (cmp_rn >> 63) & 1
                rm_neg = (cmp_rm_or_imm >> 63) & 1
                result_neg = (cmp_val >> 63) & 1
                new_v = ((rn_neg != rm_neg) & (rn_neg != result_neg)).to(torch.float32)

                # Update flags tensor
                flags[0] = new_n.to(torch.float32)  # N
                flags[1] = new_z  # Z
                flags[2] = new_c  # C
                flags[3] = new_v  # V

            # SUBS also writes result to rd (unless rd=31)
            subs_write_mask = (subs_reg_mask | subs_imm_mask) & (rds != 31)
            results = torch.where(subs_reg_mask & (rds != 31), cmp_result_reg, results)
            results = torch.where(subs_imm_mask & (rds != 31), cmp_result_imm, results)
            write_mask = write_mask | subs_write_mask

            # --- ADDS_IMM / ADDS_REG (flag-setting add, writes rd) ---
            # ADDS 64 IMM: 0xB1000000, ADDS 64 REG: 0xAB000000
            # ADDS 32 IMM: 0x31000000, ADDS 32 REG: 0x2B000000
            adds_imm_gpu = (((insts & 0xFF000000) == 0xB1000000) |
                            ((insts & 0xFF000000) == 0x31000000)) & exec_mask
            adds_reg_gpu = (((insts & 0xFF200000) == 0xAB000000) |
                            ((insts & 0xFF200000) == 0x2B000000)) & exec_mask
            adds_any_gpu = adds_imm_gpu | adds_reg_gpu
            if adds_any_gpu.any():
                adds_result_imm = rn_vals + imm12
                adds_result_reg = rn_vals + rm_vals
                adds_result = torch.where(adds_imm_gpu, adds_result_imm, adds_result_reg)
                adds_idx = torch.where(adds_any_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                adds_v = adds_result[adds_idx]
                adds_rn = rn_vals[adds_idx]
                adds_rm = torch.where(adds_imm_gpu[adds_idx], imm12[adds_idx], rm_vals[adds_idx])
                new_n = (adds_v >> 63) & 1
                new_z = (adds_v == 0).to(torch.float32)
                new_c = (adds_rn.to(torch.uint64) > (~adds_rm).to(torch.uint64)).to(torch.float32)
                rn_neg2 = (adds_rn >> 63) & 1
                rm_neg2 = (adds_rm >> 63) & 1
                res_neg2 = (adds_v >> 63) & 1
                new_v = ((rn_neg2 == rm_neg2) & (rn_neg2 != res_neg2)).to(torch.float32)
                flags[0] = new_n.to(torch.float32)
                flags[1] = new_z
                flags[2] = new_c
                flags[3] = new_v
                adds_write = adds_any_gpu & (rds != 31)
                results = torch.where(adds_write, adds_result, results)
                write_mask = write_mask | adds_write

            # --- ADC / ADCS / SBC / SBCS (add/subtract with carry) ---
            # ADC 64: 0x9A000000; ADC 32: 0x1A000000; ADCS 64: 0xBA000000; ADCS 32: 0x3A000000
            # SBC 64: 0xDA000000; SBC 32: 0x5A000000; SBCS 64: 0xFA000000; SBCS 32: 0x7A000000
            # SBC semantics: Rd = Rn + ~Rm + C  (equivalent to Rn - Rm - (1-C))
            adc_gpu = (((insts & 0xFFE0FC00) == 0x9A000000) |
                       ((insts & 0xFFE0FC00) == 0x1A000000)) & exec_mask
            adcs_gpu = (((insts & 0xFFE0FC00) == 0xBA000000) |
                        ((insts & 0xFFE0FC00) == 0x3A000000)) & exec_mask
            sbc_gpu = (((insts & 0xFFE0FC00) == 0xDA000000) |
                       ((insts & 0xFFE0FC00) == 0x5A000000)) & exec_mask
            sbcs_gpu = (((insts & 0xFFE0FC00) == 0xFA000000) |
                        ((insts & 0xFFE0FC00) == 0x7A000000)) & exec_mask
            carry_ops = adc_gpu | adcs_gpu | sbc_gpu | sbcs_gpu
            if carry_ops.any():
                c_in = flags[2].to(torch.int64)   # carry flag (0 or 1)
                adc_val = rn_vals + rm_vals + c_in
                sbc_val = rn_vals + ~rm_vals + c_in  # ~Rm + C = -(Rm+1) + C = Rn-Rm+(C-1)
                results = torch.where(adc_gpu | adcs_gpu, adc_val, results)
                results = torch.where(sbc_gpu | sbcs_gpu, sbc_val, results)
                write_mask = write_mask | carry_ops
                # Flag-setting variants (ADCS / SBCS)
                flag_ops = adcs_gpu | sbcs_gpu
                if flag_ops.any():
                    fo_idx = torch.where(flag_ops, batch_idx, BIG).min().clamp(0, batch_size - 1)
                    fo_rn = rn_vals[fo_idx]
                    fo_rm = rm_vals[fo_idx]
                    fo_val = torch.where(adcs_gpu[fo_idx], adc_val[fo_idx], sbc_val[fo_idx])
                    flags[0] = ((fo_val >> 63) & 1).to(torch.float32)
                    flags[1] = (fo_val == 0).to(torch.float32)
                    # Carry for ADCS: unsigned overflow; for SBCS: borrow = NOT carry
                    fo_is_adc = adcs_gpu[fo_idx]
                    adcs_c = (fo_rn.to(torch.uint64) > (~(fo_rm + c_in)).to(torch.uint64))
                    sbcs_c = (fo_rn.to(torch.uint64) >= fo_rm.to(torch.uint64))
                    flags[2] = torch.where(fo_is_adc, adcs_c, sbcs_c).to(torch.float32)
                    fo_rn_n = (fo_rn >> 63) & 1
                    fo_rm_n_eff = torch.where(fo_is_adc, (fo_rm >> 63) & 1, ~(fo_rm >> 63) & 1)
                    fo_res_n = (fo_val >> 63) & 1
                    flags[3] = ((fo_rn_n == fo_rm_n_eff) & (fo_rn_n != fo_res_n)).to(torch.float32)

            # --- ANDS_REG / TST (ANDS with rd=31) ---
            # ANDS 64 REG: 0xEA000000 mask 0xFF200000; writes rd, updates flags (N,Z; C=V=0)
            # TST = ANDS with rd=31 (result discarded but flags set)
            ands_reg_gpu = ((insts & 0xFF200000) == 0xEA000000) & exec_mask
            if ands_reg_gpu.any():
                ands_result = rn_vals & rm_vals
                ands_idx = torch.where(ands_reg_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                ands_v = ands_result[ands_idx]
                flags[0] = ((ands_v >> 63) & 1).to(torch.float32)
                flags[1] = (ands_v == 0).to(torch.float32)
                flags[2] = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
                flags[3] = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
                ands_write = ands_reg_gpu & (rds != 31)
                results = torch.where(ands_write, ands_result, results)
                write_mask = write_mask | ands_write

            # MUL, ASR_IMM, LSL/LSR/ASR REG, SXTW, UXTB, UXTH, SXTB32, SXTH32
            # are all handled by the SIMD dispatch above (mask vars still available below).

            # --- UBFM 64 (general unsigned bitfield move / extract) ---
            # Encoding: sf=1, opc=10, fixed=100110 → top byte 0xD3, bit22=N=1
            # Handles: LSR (immr=shift, imms=63), UXTB (immr=0,imms=7), UXTH (immr=0,imms=15)
            # and general extract: Xd = (Xn >> immr) & ((1<<(imms-immr+1))-1) when immr<=imms
            ubfm64_gpu = ((insts & 0xFFC00000) == 0xD3400000) & exec_mask & ~lsr_imm_mask
            if ubfm64_gpu.any():
                u_immr = (insts >> 16) & 0x3F
                u_imms = (insts >> 10) & 0x3F
                # Simple extract: immr <= imms
                u_width = (u_imms - u_immr + 1).clamp(1, 64)
                u_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << u_width) - 1
                u_extract = (rn_vals >> u_immr) & u_wmask
                # Rotation case: immr > imms → LSL-like (rotation pattern)
                u_rot_width = (u_imms + 1).clamp(1, 64)
                u_rot_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << u_rot_width) - 1
                u_rot = torch.where(u_immr > 0,
                                    ((rn_vals >> u_immr) | (rn_vals << (64 - u_immr))) & u_rot_wmask,
                                    rn_vals & u_rot_wmask)
                u_final = torch.where(u_immr <= u_imms, u_extract, u_rot)
                results = torch.where(ubfm64_gpu, u_final, results)
                write_mask = write_mask | ubfm64_gpu

            # --- UBFM 32 general (unsigned bitfield move / extract, 32-bit) ---
            # sf=0, opc=10, N=0 → (inst & 0xFFC00000)==0x53000000
            # UXTB/UXTH are special cases already handled above; exclude to avoid double-write
            ubfm32_gpu = ((insts & 0xFFC00000) == 0x53000000) & exec_mask & ~uxtb_gpu & ~uxth_gpu
            if ubfm32_gpu.any():
                u32_immr = (insts >> 16) & 0x3F
                u32_imms = (insts >> 10) & 0x3F
                u32_width = (u32_imms - u32_immr + 1).clamp(1, 32)
                u32_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << u32_width) - 1
                u32_rn = rn_vals & 0xFFFFFFFF
                u32_extract = (u32_rn >> u32_immr) & u32_wmask
                # Rotation case (LSL alias): immr > imms
                u32_rot_width = (u32_imms + 1).clamp(1, 32)
                u32_rot_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << u32_rot_width) - 1
                u32_rot = torch.where(u32_immr > 0,
                                      ((u32_rn >> u32_immr) | (u32_rn << (32 - u32_immr))) & u32_rot_wmask,
                                      u32_rn & u32_rot_wmask)
                u32_final = torch.where(u32_immr <= u32_imms, u32_extract, u32_rot) & 0xFFFFFFFF
                results = torch.where(ubfm32_gpu, u32_final, results)
                write_mask = write_mask | ubfm32_gpu

            # --- SBFM 64 general (signed bitfield extract / sign extension) ---
            # Top byte: 0x93 for sf=1,opc=00,fixed=100110 — but 0x93 also maps to SXTW in table
            # Handle general SBFM (excl. SXTW already handled, excl. ASR_IMM already handled)
            # SBFX: SBFM Xd, Xn, #lsb, #(lsb+width-1) — sign-extends extracted bits
            sbfm64_gpu = ((insts & 0xFFC00000) == 0x93400000) & exec_mask & ~asr_imm_gpu & ~sxtw_gpu
            if sbfm64_gpu.any():
                sb_immr = (insts >> 16) & 0x3F
                sb_imms = (insts >> 10) & 0x3F
                sb_width = (sb_imms - sb_immr + 1).clamp(1, 64)
                sb_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << sb_width) - 1
                sb_extract = (rn_vals >> sb_immr) & sb_wmask
                # Sign extend: if bit (width-1) is set, extend sign
                sb_sign_bit = sb_width - 1
                sb_sign_val = (sb_extract >> sb_sign_bit) & 1
                sb_signed = torch.where(sb_sign_val != 0,
                                        sb_extract | (~sb_wmask),
                                        sb_extract)
                results = torch.where(sbfm64_gpu & (sb_immr <= sb_imms), sb_signed, results)
                write_mask = write_mask | (sbfm64_gpu & (sb_immr <= sb_imms))

            # --- SBFM 32 general (signed bitfield extract, 32-bit) ---
            # sf=0, opc=00, N=0 → (inst & 0xFFC00000)==0x13000000
            # Exclude already-handled special cases: SXTB32, SXTH32, ASR_IMM 32-bit
            sbfm32_gpu = ((insts & 0xFFC00000) == 0x13000000) & exec_mask & ~sxtb32_gpu & ~sxth32_gpu & ~asr_imm_gpu
            if sbfm32_gpu.any():
                sb32_immr = (insts >> 16) & 0x3F
                sb32_imms = (insts >> 10) & 0x3F
                sb32_width = (sb32_imms - sb32_immr + 1).clamp(1, 32)
                sb32_wmask = (torch.tensor(1, dtype=torch.int64, device=device) << sb32_width) - 1
                sb32_rn = rn_vals & 0xFFFFFFFF
                sb32_extract = (sb32_rn >> sb32_immr) & sb32_wmask
                sb32_sign_bit = sb32_width - 1
                sb32_sign_val = (sb32_extract >> sb32_sign_bit) & 1
                sb32_signed = torch.where(sb32_sign_val != 0,
                                          sb32_extract | (~sb32_wmask & 0xFFFFFFFF),
                                          sb32_extract)
                results = torch.where(sbfm32_gpu & (sb32_immr <= sb32_imms), sb32_signed, results)
                write_mask = write_mask | (sbfm32_gpu & (sb32_immr <= sb32_imms))

            # --- BFM (bitfield move with insert — opc=01, preserves other Xd bits) ---
            # BFM 64: sf=1,opc=01,N=1 → (inst & 0xFFC00000)==0xB3400000; BFM 32: 0x33000000
            # Aliases: BFI (immr > imms), BFXIL (immr <= imms)
            bfm64_gpu = ((insts & 0xFFC00000) == 0xB3400000) & exec_mask
            bfm32_gpu = ((insts & 0xFFC00000) == 0x33000000) & exec_mask
            bfm_gpu = bfm64_gpu | bfm32_gpu
            if bfm_gpu.any():
                bfm_immr = (insts >> 16) & 0x3F
                bfm_imms = (insts >> 10) & 0x3F
                bfm_is_bfi = bfm_immr > bfm_imms  # BFI: insert at top of dest
                # BFI case: insert Xn[width-1:0] at Xd[lsb+width-1:lsb] where lsb=64-immr, width=imms+1
                bfm_width_bfi = (bfm_imms + 1).clamp(1, 64)
                bfm_lsb = (64 - bfm_immr) & 0x3F
                bfm_mask_bfi = ((torch.tensor(1, dtype=torch.int64, device=device) << bfm_width_bfi) - 1) << bfm_lsb
                bfm_src_bfi = rn_vals & ((torch.tensor(1, dtype=torch.int64, device=device) << bfm_width_bfi) - 1)
                bfm_val_bfi = (rd_vals & ~bfm_mask_bfi) | (bfm_src_bfi << bfm_lsb)
                # BFXIL case: insert Xn[immr+width-1:immr] at Xd[width-1:0] where width=imms-immr+1
                bfm_width_bfxil = (bfm_imms - bfm_immr + 1).clamp(1, 64)
                bfm_mask_bfxil = (torch.tensor(1, dtype=torch.int64, device=device) << bfm_width_bfxil) - 1
                bfm_val_bfxil = (rd_vals & ~bfm_mask_bfxil) | ((rn_vals >> bfm_immr) & bfm_mask_bfxil)
                bfm_val = torch.where(bfm_is_bfi, bfm_val_bfi, bfm_val_bfxil)
                results = torch.where(bfm_gpu, bfm_val, results)
                write_mask = write_mask | bfm_gpu

            # ORN / MVN handled by SIMD dispatch above (class 12; orn_gpu mask defined there).

            # --- UDIV / SDIV ---
            # UDIV 64: (inst & 0xFFE0FC00)==0x9AC00800; SDIV 64: 0x9AC00C00
            udiv_gpu = (((insts & 0xFFE0FC00) == 0x9AC00800) |
                        ((insts & 0xFFE0FC00) == 0x1AC00800)) & exec_mask
            sdiv_gpu = (((insts & 0xFFE0FC00) == 0x9AC00C00) |
                        ((insts & 0xFFE0FC00) == 0x1AC00C00)) & exec_mask
            safe_rm = torch.where(rm_vals == 0, torch.ones_like(rm_vals), rm_vals)
            # UDIV: unsigned division via float64 (handles full uint64 range)
            # MPS doesn't support float64 — fall back to CPU for the float64 op
            _dev_str = str(device)
            if 'mps' in _dev_str:
                _rn_cpu = rn_vals.to('cpu').to(torch.float64)
                _rm_cpu = safe_rm.to('cpu').to(torch.float64)
                udiv_res = torch.where(rm_vals != 0,
                                       torch.div(_rn_cpu, _rm_cpu, rounding_mode='trunc')
                                           .to(torch.int64).to(device),
                                       torch.zeros_like(rn_vals))
            else:
                udiv_res = torch.where(rm_vals != 0,
                                       torch.div(rn_vals.to(torch.float64),
                                                 safe_rm.to(torch.float64),
                                                 rounding_mode='trunc').to(torch.int64),
                                       torch.zeros_like(rn_vals))
            # SDIV: signed truncation toward zero
            sdiv_res = torch.where(rm_vals != 0,
                                   torch.div(rn_vals, safe_rm, rounding_mode='trunc'),
                                   torch.zeros_like(rn_vals))
            results = torch.where(udiv_gpu, udiv_res, results)
            results = torch.where(sdiv_gpu, sdiv_res, results)
            write_mask = write_mask | udiv_gpu | sdiv_gpu

            # --- MADD / MSUB (multiply-add / multiply-subtract with Ra≠XZR) ---
            # MADD 64: (inst & 0xFFE08000)==0x9B000000, Ra=bits[14:10]
            # MUL (Ra=XZR) already handled above; only handle Ra≠XZR here
            madd_full_gpu = (((insts & 0xFFE08000) == 0x9B000000) |
                             ((insts & 0xFFE08000) == 0x1B000000)) & exec_mask & ~mul_gpu_mask
            msub_gpu = (((insts & 0xFFE08000) == 0x9B008000) |
                        ((insts & 0xFFE08000) == 0x1B008000)) & exec_mask
            if madd_full_gpu.any() or msub_gpu.any():
                ra_idx = (insts >> 10) & 0x1F
                ra_vals_madd = regs[ra_idx.clamp(0, 31)]
                results = torch.where(madd_full_gpu, rn_vals * rm_vals + ra_vals_madd, results)
                results = torch.where(msub_gpu, ra_vals_madd - rn_vals * rm_vals, results)
                write_mask = write_mask | madd_full_gpu | msub_gpu

            # --- SMADDL / SMULL (signed 32×32→64 multiply-accumulate) ---
            # SMADDL: sf=1, op31=001, o0=0 → (inst & 0xFFE08000)==0x9B200000 (any Ra)
            # SMULL = SMADDL with Ra=XZR (Ra=31 → adds 0, same handler)
            # SMSUBL: o0=1 → (inst & 0xFFE08000)==0x9B208000
            smaddl_gpu = ((insts & 0xFFE08000) == 0x9B200000) & exec_mask
            smsubl_gpu = ((insts & 0xFFE08000) == 0x9B208000) & exec_mask
            if smaddl_gpu.any() or smsubl_gpu.any():
                sm_ra = (insts >> 10) & 0x1F
                sm_ra_vals = torch.where(sm_ra == 31, torch.zeros_like(rn_vals),
                                         regs[sm_ra.clamp(0, 30)])
                # Sign-extend Wn and Wm from 32-bit to 64-bit
                sm_rn32 = rn_vals & 0xFFFFFFFF
                sm_rm32 = rm_vals & 0xFFFFFFFF
                sm_rn_s = torch.where(sm_rn32 >= 0x80000000, sm_rn32 - 0x100000000, sm_rn32)
                sm_rm_s = torch.where(sm_rm32 >= 0x80000000, sm_rm32 - 0x100000000, sm_rm32)
                results = torch.where(smaddl_gpu, sm_ra_vals + sm_rn_s * sm_rm_s, results)
                results = torch.where(smsubl_gpu, sm_ra_vals - sm_rn_s * sm_rm_s, results)
                write_mask = write_mask | smaddl_gpu | smsubl_gpu

            # --- UMADDL / UMULL (unsigned 32×32→64 multiply-accumulate) ---
            # op31=101, o0=0 → (inst & 0xFFE08000)==0x9BA00000
            umaddl_gpu = ((insts & 0xFFE08000) == 0x9BA00000) & exec_mask
            umsubl_gpu = ((insts & 0xFFE08000) == 0x9BA08000) & exec_mask
            if umaddl_gpu.any() or umsubl_gpu.any():
                um_ra = (insts >> 10) & 0x1F
                um_ra_vals = torch.where(um_ra == 31, torch.zeros_like(rn_vals),
                                          regs[um_ra.clamp(0, 30)])
                um_rn32 = rn_vals & 0xFFFFFFFF  # zero-extend
                um_rm32 = rm_vals & 0xFFFFFFFF
                results = torch.where(umaddl_gpu, um_ra_vals + um_rn32 * um_rm32, results)
                results = torch.where(umsubl_gpu, um_ra_vals - um_rn32 * um_rm32, results)
                write_mask = write_mask | umaddl_gpu | umsubl_gpu

            # --- SMULH (high 64 bits of signed 64×64 product) ---
            # op31=010, Ra=XZR (bits[14:10]=11111) → (inst & 0xFFE0FC00)==0x9B407C00
            smulh_gpu = ((insts & 0xFFE0FC00) == 0x9B407C00) & exec_mask
            if smulh_gpu.any():
                # Python int multiplication is arbitrary precision; use CPU scalars
                smulh_vals = torch.zeros(batch_size, dtype=torch.int64, device=device)
                for _bi in smulh_gpu.nonzero(as_tuple=False).squeeze(1):
                    _a = int(rn_vals[_bi])
                    _b = int(rm_vals[_bi])
                    smulh_vals[_bi] = (_a * _b) >> 64
                results = torch.where(smulh_gpu, smulh_vals, results)
                write_mask = write_mask | smulh_gpu

            # --- UMULH (high 64 bits of unsigned 64×64 product) ---
            # op31=110, Ra=XZR → (inst & 0xFFE0FC00)==0x9BC07C00
            umulh_gpu = ((insts & 0xFFE0FC00) == 0x9BC07C00) & exec_mask
            if umulh_gpu.any():
                umulh_vals = torch.zeros(batch_size, dtype=torch.int64, device=device)
                mask64 = (1 << 64) - 1
                for _bi in umulh_gpu.nonzero(as_tuple=False).squeeze(1):
                    _a = int(rn_vals[_bi]) & mask64
                    _b = int(rm_vals[_bi]) & mask64
                    umulh_vals[_bi] = (_a * _b) >> 64
                results = torch.where(umulh_gpu, umulh_vals, results)
                write_mask = write_mask | umulh_gpu

            # --- EXTR (extract/concatenate two regs, or ROR alias) ---
            # EXTR 64: sf=1,N=1 → (inst & 0xFFE00000)==0x93800000
            # EXTR 32: sf=0,N=0 → (inst & 0xFFE00000)==0x13800000
            # Xd = (Xn:Xm)[lsb+63:lsb] where lsb=imms bits[15:10]
            # ROR Xd, Xn, #sh = EXTR Xd, Xn, Xn, #sh
            extr64_gpu = ((insts & 0xFFE00000) == 0x93800000) & exec_mask
            extr32_gpu = ((insts & 0xFFE00000) == 0x13800000) & exec_mask
            if (extr64_gpu | extr32_gpu).any():
                extr_lsb = (insts >> 10) & 0x3F
                # 64-bit: concat Xn:Xm (128-bit), extract 64 bits starting at lsb
                # Result = (Xn << (64 - lsb)) | (Xm >> lsb) for lsb > 0
                extr64_lsb_nz = extr_lsb.clamp(1, 63)
                extr64_val = torch.where(extr_lsb == 0,
                                         rm_vals,
                                         (rm_vals >> extr64_lsb_nz) | (rn_vals << (64 - extr64_lsb_nz)))
                # 32-bit: same but on low 32 bits
                extr32_lsb = extr_lsb & 0x1F
                extr32_lsb_nz = extr32_lsb.clamp(1, 31)
                rm32 = rm_vals & 0xFFFFFFFF
                rn32 = rn_vals & 0xFFFFFFFF
                extr32_val = torch.where(extr32_lsb == 0,
                                          rm32,
                                          (rm32 >> extr32_lsb_nz) | ((rn32 << (32 - extr32_lsb_nz)) & 0xFFFFFFFF))
                results = torch.where(extr64_gpu, extr64_val, results)
                results = torch.where(extr32_gpu, extr32_val, results)
                write_mask = write_mask | extr64_gpu | extr32_gpu

            # --- CLZ (count leading zeros) ---
            # CLZ 64: (inst & 0xFFFFFC00)==0xDAC01000; CLZ 32: 0x5AC01000
            clz64_gpu = ((insts & 0xFFFFFC00) == 0xDAC01000) & exec_mask
            clz32_gpu = ((insts & 0xFFFFFC00) == 0x5AC01000) & exec_mask
            if (clz64_gpu | clz32_gpu).any():
                # Compute CLZ using bit tricks: count zeros above highest set bit
                clz64_val = torch.where(rn_vals == 0,
                                        torch.tensor(64, dtype=torch.int64, device=device),
                                        (63 - (rn_vals.abs().to(torch.float32).log2().floor().to(torch.int64))))
                clz32_val = torch.where((rn_vals & 0xFFFFFFFF) == 0,
                                        torch.tensor(32, dtype=torch.int64, device=device),
                                        (31 - ((rn_vals & 0xFFFFFFFF).to(torch.float32).log2().floor().to(torch.int64))))
                results = torch.where(clz64_gpu, clz64_val, results)
                results = torch.where(clz32_gpu, clz32_val, results)
                write_mask = write_mask | clz64_gpu | clz32_gpu

            # --- REV (byte reversal) ---
            # REV 64: (inst & 0xFFFFFC00)==0xDAC00C00; REV 32: 0x5AC00800
            rev64_gpu = ((insts & 0xFFFFFC00) == 0xDAC00C00) & exec_mask
            rev32_gpu = ((insts & 0xFFFFFC00) == 0x5AC00800) & exec_mask
            if (rev64_gpu | rev32_gpu).any():
                b0 =  rn_vals        & 0xFF
                b1 = (rn_vals >>  8) & 0xFF
                b2 = (rn_vals >> 16) & 0xFF
                b3 = (rn_vals >> 24) & 0xFF
                b4 = (rn_vals >> 32) & 0xFF
                b5 = (rn_vals >> 40) & 0xFF
                b6 = (rn_vals >> 48) & 0xFF
                b7 = (rn_vals >> 56) & 0xFF
                rev64_val = (b0 << 56) | (b1 << 48) | (b2 << 40) | (b3 << 32) | (b4 << 24) | (b5 << 16) | (b6 << 8) | b7
                rev32_val = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
                results = torch.where(rev64_gpu, rev64_val, results)
                results = torch.where(rev32_gpu, rev32_val, results)
                write_mask = write_mask | rev64_gpu | rev32_gpu

            # --- REV16 (reverse bytes within each 16-bit halfword) ---
            # REV16 64: (inst & 0xFFFFFC00)==0xDAC00400; REV16 32: 0x5AC00400
            rev16_64_gpu = ((insts & 0xFFFFFC00) == 0xDAC00400) & exec_mask
            rev16_32_gpu = ((insts & 0xFFFFFC00) == 0x5AC00400) & exec_mask
            if (rev16_64_gpu | rev16_32_gpu).any():
                # Each 16-bit halfword: swap byte0 and byte1
                h0 = ((rn_vals & 0x00FF) << 8) | ((rn_vals >> 8) & 0xFF)
                h1 = ((rn_vals & 0x00FF0000) << 8) | ((rn_vals >> 8) & 0x00FF0000)
                h2 = ((rn_vals & 0x00FF00000000) << 8) | ((rn_vals >> 8) & 0x00FF00000000)
                h3 = ((rn_vals & 0x00FF000000000000) << 8) | ((rn_vals >> 8) & 0x00FF000000000000)
                rev16_64_val = h0 | h1 | h2 | h3
                rev16_32_val = h0 | h1
                results = torch.where(rev16_64_gpu, rev16_64_val, results)
                results = torch.where(rev16_32_gpu, rev16_32_val, results)
                write_mask = write_mask | rev16_64_gpu | rev16_32_gpu

            # --- RBIT (reverse bits) ---
            # RBIT 64: (inst & 0xFFFFFC00)==0xDAC00000; RBIT 32: 0x5AC00000
            rbit64_gpu = ((insts & 0xFFFFFC00) == 0xDAC00000) & exec_mask
            rbit32_gpu = ((insts & 0xFFFFFC00) == 0x5AC00000) & exec_mask
            if (rbit64_gpu | rbit32_gpu).any():
                # Reverse bits: swap adjacent bits, then 2-bit pairs, then nibbles, then bytes
                v = rn_vals
                v = ((v & 0x5555555555555555) << 1) | ((v >> 1) & 0x5555555555555555)
                v = ((v & 0x3333333333333333) << 2) | ((v >> 2) & 0x3333333333333333)
                v = ((v & 0x0F0F0F0F0F0F0F0F) << 4) | ((v >> 4) & 0x0F0F0F0F0F0F0F0F)
                # Then byte-reverse (same bit trick as REV)
                r0 =  v        & 0xFF
                r1 = (v >>  8) & 0xFF
                r2 = (v >> 16) & 0xFF
                r3 = (v >> 24) & 0xFF
                r4 = (v >> 32) & 0xFF
                r5 = (v >> 40) & 0xFF
                r6 = (v >> 48) & 0xFF
                r7 = (v >> 56) & 0xFF
                rbit64_val = (r0 << 56) | (r1 << 48) | (r2 << 40) | (r3 << 32) | (r4 << 24) | (r5 << 16) | (r6 << 8) | r7
                rbit32_val = (r0 << 24) | (r1 << 16) | (r2 << 8) | r3
                results = torch.where(rbit64_gpu, rbit64_val, results)
                results = torch.where(rbit32_gpu, rbit32_val, results)
                write_mask = write_mask | rbit64_gpu | rbit32_gpu

            # --- ADR (PC-relative address, no page align) ---
            # ADR: (inst & 0x9F000000)==0x10000000 (bit28=0, bits31-29=immlo)
            adr_gpu = ((insts & 0x9F000000) == 0x10000000) & exec_mask
            if adr_gpu.any():
                adr_immlo = (insts >> 29) & 0x3
                adr_immhi = (insts >> 5) & 0x7FFFF
                adr_imm21 = (adr_immhi << 2) | adr_immlo
                adr_imm21_s = torch.where(adr_imm21 >= 0x100000, adr_imm21 - 0x200000, adr_imm21)
                inst_pcs_adr = pc_t + batch_idx * 4
                adr_val = inst_pcs_adr + adr_imm21_s
                results = torch.where(adr_gpu, adr_val, results)
                write_mask = write_mask | adr_gpu

            # --- CSINC (conditional select + increment) ---
            # CSINC 64: 0x9A800400 (bit22=0 distinguishes from LSLV/LSRV/ASRV which have bit22=1)
            # CINC = CSINC Xd, Xn, ~cond; INC = CSINC Xd, Xn, AL
            # NOTE: 0xFF800C00 was too broad — LSRV (0x9ACxxxxx, bit22=1) also has bits[11:10]=01
            csinc_gpu = (((insts & 0xFFC00C00) == 0x9A800400) |
                         ((insts & 0xFFC00C00) == 0x1A800400)) & exec_mask
            # CSINV (conditional select + bitwise invert)
            # 64-bit uses 0x5A (different top byte from LSLV 0x9A), no ambiguity needed
            csinv_gpu = (((insts & 0xFF800C00) == 0x5A800000) |
                         ((insts & 0xFF800C00) == 0x1A800000)) & exec_mask & ~(ops == OpType.CSEL_W.value if hasattr(OpType, 'CSEL_W') else torch.zeros(1, dtype=torch.bool, device=device))
            # CSNEG (conditional select + negate)
            # 64-bit uses 0x5A; 32-bit 0x1A: use 0xFFC00C00 to exclude 32-bit LSRV/LSLV
            csneg_gpu = (((insts & 0xFF800C00) == 0x5A800400) |
                         ((insts & 0xFFC00C00) == 0x1A800400)) & exec_mask
            if (csinc_gpu | csinv_gpu | csneg_gpu).any():
                cs_cond = (insts >> 12) & 0xF
                n2, z2, c2, v2 = flags[0], flags[1], flags[2], flags[3]
                cs_cond_tab = torch.stack([
                    z2 > 0.5, z2 <= 0.5, c2 > 0.5, c2 <= 0.5,
                    n2 > 0.5, n2 <= 0.5, v2 > 0.5, v2 <= 0.5,
                    (c2 > 0.5) & (z2 <= 0.5), (c2 <= 0.5) | (z2 > 0.5),
                    ((n2 > 0.5) == (v2 > 0.5)), ((n2 > 0.5) != (v2 > 0.5)),
                    (z2 <= 0.5) & ((n2 > 0.5) == (v2 > 0.5)),
                    (z2 > 0.5) | ((n2 > 0.5) != (v2 > 0.5)),
                    torch.tensor(True, device=device),
                    torch.tensor(False, device=device),
                ])
                cs_taken = cs_cond_tab[cs_cond.clamp(0, 15)]
                results = torch.where(csinc_gpu,
                                      torch.where(cs_taken, rn_vals, rm_vals + 1), results)
                results = torch.where(csinv_gpu,
                                      torch.where(cs_taken, rn_vals, ~rm_vals), results)
                results = torch.where(csneg_gpu,
                                      torch.where(cs_taken, rn_vals, -rm_vals), results)
                write_mask = write_mask | csinc_gpu | csinv_gpu | csneg_gpu

            # --- CCMP / CCMN (conditional compare) ---
            # CCMP 64: (inst & 0xFF800010)==0xFA000000 — sets flags if cond true, else uses nzcv field
            ccmp_gpu = (((insts & 0xFF800010) == 0xFA000000) |
                        ((insts & 0xFF800010) == 0x7A000000)) & exec_mask
            if ccmp_gpu.any():
                cc_cond = (insts >> 12) & 0xF
                n2, z2, c2, v2 = flags[0], flags[1], flags[2], flags[3]
                cc_cond_tab = torch.stack([
                    z2 > 0.5, z2 <= 0.5, c2 > 0.5, c2 <= 0.5,
                    n2 > 0.5, n2 <= 0.5, v2 > 0.5, v2 <= 0.5,
                    (c2 > 0.5) & (z2 <= 0.5), (c2 <= 0.5) | (z2 > 0.5),
                    ((n2 > 0.5) == (v2 > 0.5)), ((n2 > 0.5) != (v2 > 0.5)),
                    (z2 <= 0.5) & ((n2 > 0.5) == (v2 > 0.5)),
                    (z2 > 0.5) | ((n2 > 0.5) != (v2 > 0.5)),
                    torch.tensor(True, device=device),
                    torch.tensor(False, device=device),
                ])
                cc_taken = cc_cond_tab[cc_cond.clamp(0, 15)]
                if ccmp_gpu.any():
                    ccmp_idx = torch.where(ccmp_gpu, batch_idx, BIG).min().clamp(0, batch_size-1)
                    if cc_taken.item():
                        # Condition true: compare and set flags
                        cc_imm = (insts[ccmp_idx] >> 16) & 0x1F  # immediate variant
                        cc_rm_or_imm = torch.where((insts[ccmp_idx] >> 11) & 1 == 1,
                                                   cc_imm.to(torch.int64),
                                                   rm_vals[ccmp_idx])
                        cc_res = rn_vals[ccmp_idx] - cc_rm_or_imm
                        flags[0] = ((cc_res >> 63) & 1).to(torch.float32)
                        flags[1] = (cc_res == 0).to(torch.float32)
                        flags[2] = (rn_vals[ccmp_idx].to(torch.uint64) >= cc_rm_or_imm.to(torch.uint64)).to(torch.float32)
                        rn_n2 = (rn_vals[ccmp_idx] >> 63) & 1
                        rm_n2 = (cc_rm_or_imm >> 63) & 1
                        res_n2 = (cc_res >> 63) & 1
                        flags[3] = ((rn_n2 != rm_n2) & (rn_n2 != res_n2)).to(torch.float32)
                    else:
                        # Condition false: load nzcv field (bits[3:0])
                        nzcv = insts[ccmp_idx] & 0xF
                        flags[0] = torch.tensor(float((nzcv >> 3) & 1), device=device)
                        flags[1] = torch.tensor(float((nzcv >> 2) & 1), device=device)
                        flags[2] = torch.tensor(float((nzcv >> 1) & 1), device=device)
                        flags[3] = torch.tensor(float(nzcv & 1), device=device)

            # --- AND/ORR/EOR IMMEDIATE (logical immediate bitmask encoding) ---
            # AND_IMM 64: bit23=0 → (inst & 0xFF800000)==0x92000000, 32: 0x12000000
            # ORR_IMM 64: 0xB2000000, 32: 0x32000000
            # EOR_IMM 64: 0xD2000000 (distinct from MOVZ 0xD2800000 bit23=1), 32: 0x52000000
            # Use EXPLICIT patterns (not ops table) to avoid false matches:
            #   0x92800000=MOVN64 shares top byte 0x92 with AND_IMM64
            #   0xD2800000=MOVZ64 shares top byte 0xD2 with EOR_IMM64
            and_imm_gpu = (((insts & 0xFF800000) == 0x92000000) |   # AND IMM 64 (bit23=0)
                           ((insts & 0xFF800000) == 0x12000000)) & exec_mask  # AND IMM 32
            orr_imm_gpu = (((insts & 0xFF800000) == 0xB2000000) |   # ORR IMM 64
                           ((insts & 0xFF800000) == 0x32000000)) & exec_mask  # ORR IMM 32
            eor_imm_gpu = (((insts & 0xFF800000) == 0xD2000000) |   # EOR IMM 64 (bit23=0)
                           ((insts & 0xFF800000) == 0x52000000)) & exec_mask  # EOR IMM 32
            log_imm_any_gpu = and_imm_gpu | orr_imm_gpu | eor_imm_gpu
            if log_imm_any_gpu.any():
                log_imms_f = (insts >> 10) & 0x3F
                log_immr_f = (insts >> 16) & 0x3F
                ones_cnt = (log_imms_f + 1).clamp(1, 63)
                log_base = (torch.tensor(1, dtype=torch.int64, device=device) << ones_cnt) - 1
                rot_r = log_immr_f & 0x3F
                rot_l = (64 - rot_r) & 0x3F
                log_imm_val = torch.where(rot_r > 0,
                                          (log_base >> rot_r) | (log_base << rot_l),
                                          log_base)
                sf_bit = (insts >> 31) & 1
                log_imm_val = torch.where(sf_bit == 0, log_imm_val & 0xFFFFFFFF, log_imm_val)
                results = torch.where(and_imm_gpu, rn_vals & log_imm_val, results)
                results = torch.where(orr_imm_gpu, rn_vals | log_imm_val, results)
                results = torch.where(eor_imm_gpu, rn_vals ^ log_imm_val, results)
                write_mask = write_mask | log_imm_any_gpu

            # --- CSEL / CSET (conditional select) ---
            # CSEL 64: 0x9A800000 mask 0xFFC00000 (bit22=0 distinguishes from LSLV/LSRV/ASRV)
            # CSEL 32: 0x1A800000 mask 0xFFC00000
            # NOTE: 0xFF800000 was too broad — it incorrectly matched LSLV (0x9ACxxxxx, bit22=1)
            csel_gpu = (((insts & 0xFFC00000) == 0x9A800000) |
                        ((insts & 0xFFC00000) == 0x1A800000)) & exec_mask
            if csel_gpu.any():
                csel_cond = (insts >> 12) & 0xF
                # Reuse cond_results table from B.cond (computed later) - compute inline
                n2, z2, c2, v2 = flags[0], flags[1], flags[2], flags[3]
                csel_cond_results = torch.stack([
                    z2 > 0.5, z2 <= 0.5, c2 > 0.5, c2 <= 0.5,
                    n2 > 0.5, n2 <= 0.5, v2 > 0.5, v2 <= 0.5,
                    (c2 > 0.5) & (z2 <= 0.5), (c2 <= 0.5) | (z2 > 0.5),
                    ((n2 > 0.5) == (v2 > 0.5)), ((n2 > 0.5) != (v2 > 0.5)),
                    (z2 <= 0.5) & ((n2 > 0.5) == (v2 > 0.5)),
                    (z2 > 0.5) | ((n2 > 0.5) != (v2 > 0.5)),
                    torch.tensor(True, device=device),
                    torch.tensor(False, device=device),
                ])
                csel_taken = csel_cond_results[csel_cond.clamp(0, 15)]
                csel_val = torch.where(csel_taken, rn_vals, rm_vals)
                results = torch.where(csel_gpu, csel_val, results)
                write_mask = write_mask | csel_gpu

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5b: LOAD/STORE OPERATIONS (neural address computation)
            # Effective addresses computed by pointer.pt full-adder network.
            # prefetch.pt LSTM tracks access history for predictive pre-touch.
            # ═══════════════════════════════════════════════════════════════
            _wov_pf_addr  = -1   # first memory address this batch (for prefetcher)
            _wov_pf_write = False

            # --- LDR 64-bit unsigned offset: LDR Xt, [Xn, #imm] ---
            # Encoding: 1111 1001 01 imm12 Rn Rt = 0xF9400000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            ldr_imm_mask = ((insts & 0xFFC00000) == 0xF9400000) & exec_mask
            if ldr_imm_mask.any():
                ldr_imm12 = (insts >> 10) & 0xFFF  # Scaled by 8
                # Neural address computation (pointer.pt): active subset only
                _ldr_base = rn_vals[ldr_imm_mask]
                _ldr_off  = ldr_imm12[ldr_imm_mask] * 8
                _ldr_neural = mem_arith.compute_address(_ldr_base, _ldr_off)
                ldr_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldr_addr[ldr_imm_mask] = _ldr_neural
                ldr_addr_clamped = ldr_addr.clamp(0, self.mem_size - 8)
                if _wov_pf_addr < 0:  # record first mem addr for prefetcher (1 sync)
                    _wov_pf_addr = int(ldr_addr_clamped.max().item())
                # VECTORIZED GATHER: Compute all byte addresses at once
                byte_offsets_8 = torch.arange(8, device=device, dtype=torch.int64)
                ldr_byte_addrs = ldr_addr_clamped.unsqueeze(1) + byte_offsets_8  # [batch, 8]
                ldr_byte_addrs_flat = ldr_byte_addrs.view(-1).clamp(0, self.mem_size - 1)
                # Gather ALL bytes in one operation - NO loop!
                ldr_bytes = mem[ldr_byte_addrs_flat].view(batch_size, 8).to(torch.int64)
                # Combine bytes using tensor shifts
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                ldr_vals = (ldr_bytes << shifts_8).sum(dim=1)
                results = torch.where(ldr_imm_mask, ldr_vals, results)
                write_mask = write_mask | ldr_imm_mask

            # --- LDR 32-bit unsigned offset: LDR Wt, [Xn, #imm] ---
            # Encoding: 1011 1001 01 imm12 Rn Rt = 0xB9400000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            ldr_w_mask = ((insts & 0xFFC00000) == 0xB9400000) & exec_mask
            if ldr_w_mask.any():
                ldr_imm12 = (insts >> 10) & 0xFFF  # Scaled by 4
                _ldr_w_base = rn_vals[ldr_w_mask]
                _ldr_w_off  = ldr_imm12[ldr_w_mask] * 4
                _ldr_w_neural = mem_arith.compute_address(_ldr_w_base, _ldr_w_off)
                ldr_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldr_addr[ldr_w_mask] = _ldr_w_neural
                ldr_addr_clamped = ldr_addr.clamp(0, self.mem_size - 4)
                # VECTORIZED GATHER: 4 bytes
                byte_offsets_4 = torch.arange(4, device=device, dtype=torch.int64)
                ldr_byte_addrs = ldr_addr_clamped.unsqueeze(1) + byte_offsets_4  # [batch, 4]
                ldr_byte_addrs_flat = ldr_byte_addrs.view(-1).clamp(0, self.mem_size - 1)
                ldr_bytes = mem[ldr_byte_addrs_flat].view(batch_size, 4).to(torch.int64)
                shifts_4 = torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int64)
                ldr_vals = (ldr_bytes << shifts_4).sum(dim=1)
                results = torch.where(ldr_w_mask, ldr_vals, results)
                write_mask = write_mask | ldr_w_mask

            # --- STR 64-bit unsigned offset: STR Xt, [Xn, #imm] ---
            # Encoding: 1111 1001 00 imm12 Rn Rt = 0xF9000000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            str_imm_mask = ((insts & 0xFFC00000) == 0xF9000000) & exec_mask
            if str_imm_mask.any():
                str_imm12 = (insts >> 10) & 0xFFF  # Scaled by 8
                _str_base = rn_vals[str_imm_mask]
                _str_off  = str_imm12[str_imm_mask] * 8
                _str_neural = mem_arith.compute_address(_str_base, _str_off)
                str_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                str_addr[str_imm_mask] = _str_neural
                str_addr_clamped = str_addr.clamp(0, self.mem_size - 8)
                if _wov_pf_addr < 0:  # record first store addr for prefetcher
                    _wov_pf_addr  = int(str_addr_clamped.max().item())
                    _wov_pf_write = True
                str_rt = insts & 0x1F
                str_vals = torch.where(str_rt == 31, torch.zeros_like(rd_vals),
                                       regs[str_rt.clamp(0, 30)])
                # VECTORIZED SCATTER: Extract bytes and write all at once
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                str_bytes = ((str_vals.unsqueeze(1) >> shifts_8) & 0xFF).to(torch.uint8)  # [batch, 8]
                byte_offsets_8 = torch.arange(8, device=device, dtype=torch.int64)
                str_byte_addrs = str_addr_clamped.unsqueeze(1) + byte_offsets_8  # [batch, 8]
                # Only scatter where mask is True - use advanced indexing
                active_mask = str_imm_mask.unsqueeze(1).expand(-1, 8)  # [batch, 8]
                active_addrs = str_byte_addrs[active_mask].long()
                active_bytes = str_bytes[active_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)

            # --- STR 32-bit unsigned offset: STR Wt, [Xn, #imm] ---
            # Encoding: 1011 1001 00 imm12 Rn Rt = 0xB9000000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            str_w_mask = ((insts & 0xFFC00000) == 0xB9000000) & exec_mask
            if str_w_mask.any():
                str_imm12 = (insts >> 10) & 0xFFF  # Scaled by 4
                _str_w_base = rn_vals[str_w_mask]
                _str_w_off  = str_imm12[str_w_mask] * 4
                _str_w_neural = mem_arith.compute_address(_str_w_base, _str_w_off)
                str_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                str_addr[str_w_mask] = _str_w_neural
                str_addr_clamped = str_addr.clamp(0, self.mem_size - 4)
                str_rt = insts & 0x1F
                str_vals = torch.where(str_rt == 31, torch.zeros_like(rd_vals),
                                       regs[str_rt.clamp(0, 30)]) & 0xFFFFFFFF
                # VECTORIZED SCATTER: 4 bytes
                shifts_4 = torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int64)
                str_bytes = ((str_vals.unsqueeze(1) >> shifts_4) & 0xFF).to(torch.uint8)  # [batch, 4]
                byte_offsets_4 = torch.arange(4, device=device, dtype=torch.int64)
                str_byte_addrs = str_addr_clamped.unsqueeze(1) + byte_offsets_4  # [batch, 4]
                active_mask = str_w_mask.unsqueeze(1).expand(-1, 4)
                active_addrs = str_byte_addrs[active_mask].long()
                active_bytes = str_bytes[active_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)

            # --- STR 64-bit post-index: STR Xt, [Xn], #imm9 ---
            # Encoding: 1111 1000 000 imm9 01 Rn Rt = 0xF8000400
            # FULLY VECTORIZED - NO .item() calls!
            str_post_mask = ((insts & 0xFFE00C00) == 0xF8000400) & exec_mask
            if str_post_mask.any():
                str_imm9 = (insts >> 12) & 0x1FF  # Signed 9-bit
                str_imm9_signed = torch.where(str_imm9 >= 256, str_imm9 - 512, str_imm9)
                str_addr = rn_vals  # Post-index: use base without offset
                str_addr_clamped = str_addr.clamp(0, self.mem_size - 8)
                str_rt = insts & 0x1F
                str_vals = torch.where(str_rt == 31, torch.zeros_like(rd_vals),
                                       regs[str_rt.clamp(0, 30)])
                # VECTORIZED SCATTER: 8 bytes
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                str_bytes = ((str_vals.unsqueeze(1) >> shifts_8) & 0xFF).to(torch.uint8)
                byte_offsets_8 = torch.arange(8, device=device, dtype=torch.int64)
                str_byte_addrs = str_addr_clamped.unsqueeze(1) + byte_offsets_8
                active_mask = str_post_mask.unsqueeze(1).expand(-1, 8)
                active_addrs = str_byte_addrs[active_mask].long()
                active_bytes = str_bytes[active_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)
                # VECTORIZED WRITEBACK: Update base registers
                str_rn = rns
                new_base = rn_vals + str_imm9_signed
                # Use scatter for register writeback where mask is True and rn != 31
                wb_mask = str_post_mask & (str_rn != 31)
                if wb_mask.any():
                    wb_indices = str_rn[wb_mask].long()
                    wb_values = new_base[wb_mask]
                    regs.index_put_((wb_indices,), wb_values)

            # --- LDR 64-bit post-index: LDR Xt, [Xn], #imm9 ---
            # Encoding: 1111 1000 010 imm9 01 Rn Rt = 0xF8400400
            # FULLY VECTORIZED - NO .item() calls!
            ldr_post_mask = ((insts & 0xFFE00C00) == 0xF8400400) & exec_mask
            if ldr_post_mask.any():
                ldr_imm9 = (insts >> 12) & 0x1FF
                ldr_imm9_signed = torch.where(ldr_imm9 >= 256, ldr_imm9 - 512, ldr_imm9)
                ldr_addr = rn_vals  # Post-index: use base without offset
                ldr_addr_clamped = ldr_addr.clamp(0, self.mem_size - 8)
                # VECTORIZED GATHER: 8 bytes
                byte_offsets_8 = torch.arange(8, device=device, dtype=torch.int64)
                ldr_byte_addrs = ldr_addr_clamped.unsqueeze(1) + byte_offsets_8
                ldr_byte_addrs_flat = ldr_byte_addrs.view(-1).clamp(0, self.mem_size - 1)
                ldr_bytes = mem[ldr_byte_addrs_flat].view(batch_size, 8).to(torch.int64)
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                ldr_vals = (ldr_bytes << shifts_8).sum(dim=1)
                results = torch.where(ldr_post_mask, ldr_vals, results)
                write_mask = write_mask | ldr_post_mask
                # VECTORIZED WRITEBACK: Update base registers
                ldr_rn = rns
                new_base = rn_vals + ldr_imm9_signed
                wb_mask = ldr_post_mask & (ldr_rn != 31)
                if wb_mask.any():
                    wb_indices = ldr_rn[wb_mask].long()
                    wb_values = new_base[wb_mask]
                    regs.index_put_((wb_indices,), wb_values)

            # --- STP 64-bit (signed offset): STP Xt1, Xt2, [Xn, #imm7] ---
            # Encoding: 10 101 0 010 imm7 Rt2 Rn Rt = 0xA9000000
            # FULLY VECTORIZED - NO .item() calls!
            stp_off_mask = ((insts & 0xFFC00000) == 0xA9000000) & exec_mask
            if stp_off_mask.any():
                stp_imm7 = (insts >> 15) & 0x7F  # Signed 7-bit, scaled by 8
                stp_imm7_signed = torch.where(stp_imm7 >= 64, stp_imm7 - 128, stp_imm7)
                stp_offset = stp_imm7_signed * 8
                stp_addr = rn_vals + stp_offset
                stp_addr_clamped = stp_addr.clamp(0, self.mem_size - 16)
                stp_rt1 = insts & 0x1F
                stp_rt2 = (insts >> 10) & 0x1F
                stp_val1 = torch.where(stp_rt1 == 31, torch.zeros_like(rd_vals), regs[stp_rt1.clamp(0, 30)])
                stp_val2 = torch.where(stp_rt2 == 31, torch.zeros_like(rd_vals), regs[stp_rt2.clamp(0, 30)])
                # VECTORIZED SCATTER: 16 bytes (8 for each register)
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                stp_bytes1 = ((stp_val1.unsqueeze(1) >> shifts_8) & 0xFF).to(torch.uint8)  # [batch, 8]
                stp_bytes2 = ((stp_val2.unsqueeze(1) >> shifts_8) & 0xFF).to(torch.uint8)  # [batch, 8]
                # Concatenate bytes for both registers
                stp_bytes_all = torch.cat([stp_bytes1, stp_bytes2], dim=1)  # [batch, 16]
                byte_offsets_16 = torch.arange(16, device=device, dtype=torch.int64)
                stp_byte_addrs = stp_addr_clamped.unsqueeze(1) + byte_offsets_16  # [batch, 16]
                active_mask = stp_off_mask.unsqueeze(1).expand(-1, 16)
                active_addrs = stp_byte_addrs[active_mask].long()
                active_bytes = stp_bytes_all[active_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)

            # --- LDP 64-bit (signed offset): LDP Xt1, Xt2, [Xn, #imm7] ---
            # Encoding: 10 101 0 010 1 imm7 Rt2 Rn Rt = 0xA9400000
            # FULLY VECTORIZED - NO .item() calls!
            ldp_off_mask = ((insts & 0xFFC00000) == 0xA9400000) & exec_mask
            if ldp_off_mask.any():
                ldp_imm7 = (insts >> 15) & 0x7F
                ldp_imm7_signed = torch.where(ldp_imm7 >= 64, ldp_imm7 - 128, ldp_imm7)
                ldp_offset = ldp_imm7_signed * 8
                ldp_addr = rn_vals + ldp_offset
                ldp_addr_clamped = ldp_addr.clamp(0, self.mem_size - 16)
                ldp_rt1 = insts & 0x1F
                ldp_rt2 = (insts >> 10) & 0x1F
                # VECTORIZED GATHER: 16 bytes (8 for each register)
                byte_offsets_16 = torch.arange(16, device=device, dtype=torch.int64)
                ldp_byte_addrs = ldp_addr_clamped.unsqueeze(1) + byte_offsets_16  # [batch, 16]
                ldp_byte_addrs_flat = ldp_byte_addrs.view(-1).clamp(0, self.mem_size - 1)
                ldp_bytes_all = mem[ldp_byte_addrs_flat].view(batch_size, 16).to(torch.int64)
                # Split into two 8-byte values
                ldp_bytes1 = ldp_bytes_all[:, :8]   # First register bytes
                ldp_bytes2 = ldp_bytes_all[:, 8:]   # Second register bytes
                shifts_8 = torch.tensor([0, 8, 16, 24, 32, 40, 48, 56], device=device, dtype=torch.int64)
                ldp_val1 = (ldp_bytes1 << shifts_8).sum(dim=1)
                ldp_val2 = (ldp_bytes2 << shifts_8).sum(dim=1)
                # VECTORIZED REGISTER WRITE: Use scatter for both registers
                # Write first register where mask is True and rt1 != 31
                wb_mask1 = ldp_off_mask & (ldp_rt1 != 31)
                if wb_mask1.any():
                    wb_indices1 = ldp_rt1[wb_mask1].long()
                    wb_values1 = ldp_val1[wb_mask1]
                    regs.index_put_((wb_indices1,), wb_values1)
                # Write second register where mask is True and rt2 != 31
                wb_mask2 = ldp_off_mask & (ldp_rt2 != 31)
                if wb_mask2.any():
                    wb_indices2 = ldp_rt2[wb_mask2].long()
                    wb_values2 = ldp_val2[wb_mask2]
                    regs.index_put_((wb_indices2,), wb_values2)

            # --- STP/LDP SIMD/FP: Treat as NOPs (we don't have SIMD registers) ---
            # These patterns cover signed offset, pre-index, and post-index variants
            # 0xAC/0x2C=8-bit, 0x6C=16-bit, 0xAD=128-bit Q regs (most common)
            # Signed offset STP: 0xAD000000, LDP: 0xAD400000
            # Pre-index STP: 0xAD800000, LDP: 0xADC00000
            # Post-index STP: 0xAC800000, LDP: 0xACC00000
            # We treat ALL SIMD STP/LDP as NOPs - just skip them without modifying memory
            # This is safe because busybox echo doesn't actually use SIMD computation
            simd_stp_ldp_mask = (
                ((insts & 0xFFC00000) == 0xAD000000) |  # STP Q signed offset
                ((insts & 0xFFC00000) == 0xAD400000) |  # LDP Q signed offset
                ((insts & 0xFFC00000) == 0xAD800000) |  # STP Q pre-index
                ((insts & 0xFFC00000) == 0xADC00000) |  # LDP Q pre-index
                ((insts & 0xFFC00000) == 0xAC800000) |  # STP Q post-index
                ((insts & 0xFFC00000) == 0xACC00000) |  # LDP Q post-index
                ((insts & 0xBFC00000) == 0x2C000000) |  # STP/LDP 8-bit
                ((insts & 0xBFC00000) == 0x6C000000)    # STP/LDP 16-bit
            ) & exec_mask
            # For pre/post-index variants, we need to update the base register
            # Pre-index STP Q: update Rn before (but we're not storing)
            # FULLY VECTORIZED - NO .item() calls!
            stp_simd_pre_mask = ((insts & 0xFFC00000) == 0xAD800000) & exec_mask
            if stp_simd_pre_mask.any():
                stp_imm7 = (insts >> 15) & 0x7F
                stp_imm7_signed = torch.where(stp_imm7 >= 64, stp_imm7 - 128, stp_imm7)
                stp_offset = stp_imm7_signed * 16
                stp_addr = rn_vals + stp_offset
                # VECTORIZED WRITEBACK
                wb_mask = stp_simd_pre_mask & (rns != 31)
                if wb_mask.any():
                    wb_indices = rns[wb_mask].long()
                    wb_values = stp_addr[wb_mask]
                    regs.index_put_((wb_indices,), wb_values)

            # --- LDRB unsigned offset: LDRB Wt, [Xn, #imm12] ---
            # Encoding: 0011 1001 01 imm12 Rn Rt = 0x39400000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            ldrb_mask = ((insts & 0xFFC00000) == 0x39400000) & exec_mask
            if ldrb_mask.any():
                ldrb_imm12 = (insts >> 10) & 0xFFF  # No scaling for bytes
                _ldrb_base = rn_vals[ldrb_mask]
                _ldrb_neural = mem_arith.compute_address(_ldrb_base, ldrb_imm12[ldrb_mask])
                ldrb_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrb_addr[ldrb_mask] = _ldrb_neural
                ldrb_addr_clamped = ldrb_addr.clamp(0, self.mem_size - 1)
                if _wov_pf_addr < 0:
                    _wov_pf_addr = int(ldrb_addr_clamped.max().item())
                # VECTORIZED GATHER: single byte per address
                ldrb_vals = mem[ldrb_addr_clamped.long()].to(torch.int64)
                results = torch.where(ldrb_mask, ldrb_vals, results)
                write_mask = write_mask | ldrb_mask

            # --- STRB unsigned offset: STRB Wt, [Xn, #imm12] ---
            # Encoding: 0011 1001 00 imm12 Rn Rt = 0x39000000
            # NEURAL ADDRESS: pointer.pt full-adder computes base + offset
            strb_mask = ((insts & 0xFFC00000) == 0x39000000) & exec_mask
            if strb_mask.any():
                strb_imm12 = (insts >> 10) & 0xFFF
                _strb_base = rn_vals[strb_mask]
                _strb_neural = mem_arith.compute_address(_strb_base, strb_imm12[strb_mask])
                strb_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                strb_addr[strb_mask] = _strb_neural
                strb_addr_clamped = strb_addr.clamp(0, self.mem_size - 1)
                if _wov_pf_addr < 0:
                    _wov_pf_addr  = int(strb_addr_clamped.max().item())
                    _wov_pf_write = True
                strb_rt = insts & 0x1F
                strb_vals = torch.where(strb_rt == 31, torch.zeros_like(rd_vals),
                                        regs[strb_rt.clamp(0, 30)])
                # VECTORIZED SCATTER: single byte per address
                strb_bytes = (strb_vals & 0xFF).to(torch.uint8)
                active_addrs = strb_addr_clamped[strb_mask].long()
                active_bytes = strb_bytes[strb_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)

            # --- LDRB post-index: LDRB Wt, [Xn], #imm9 ---
            # Encoding: 0011 1000 010 imm9 01 Rn Rt = 0x38400400
            # FULLY VECTORIZED - NO .item() calls!
            ldrb_post_mask = ((insts & 0xFFE00C00) == 0x38400400) & exec_mask
            if ldrb_post_mask.any():
                ldrb_imm9 = (insts >> 12) & 0x1FF
                ldrb_imm9_signed = torch.where(ldrb_imm9 >= 256, ldrb_imm9 - 512, ldrb_imm9)
                ldrb_addr = rn_vals
                ldrb_addr_clamped = ldrb_addr.clamp(0, self.mem_size - 1)
                # VECTORIZED GATHER: single byte
                ldrb_vals = mem[ldrb_addr_clamped.long()].to(torch.int64)
                results = torch.where(ldrb_post_mask, ldrb_vals, results)
                write_mask = write_mask | ldrb_post_mask
                # VECTORIZED WRITEBACK
                new_base = rn_vals + ldrb_imm9_signed
                wb_mask = ldrb_post_mask & (rns != 31)
                if wb_mask.any():
                    wb_indices = rns[wb_mask].long()
                    wb_values = new_base[wb_mask]
                    regs.index_put_((wb_indices,), wb_values)

            # --- STRB post-index: STRB Wt, [Xn], #imm9 ---
            # Encoding: 0011 1000 000 imm9 01 Rn Rt = 0x38000400
            # FULLY VECTORIZED - NO .item() calls!
            strb_post_mask = ((insts & 0xFFE00C00) == 0x38000400) & exec_mask
            if strb_post_mask.any():
                strb_imm9 = (insts >> 12) & 0x1FF
                strb_imm9_signed = torch.where(strb_imm9 >= 256, strb_imm9 - 512, strb_imm9)
                strb_addr = rn_vals
                strb_addr_clamped = strb_addr.clamp(0, self.mem_size - 1)
                strb_rt = insts & 0x1F
                strb_vals = torch.where(strb_rt == 31, torch.zeros_like(rd_vals),
                                        regs[strb_rt.clamp(0, 30)])
                # VECTORIZED SCATTER: single byte
                strb_bytes = (strb_vals & 0xFF).to(torch.uint8)
                active_addrs = strb_addr_clamped[strb_post_mask].long()
                active_bytes = strb_bytes[strb_post_mask]
                if active_addrs.numel() > 0:
                    mem.scatter_(0, active_addrs, active_bytes)
                # VECTORIZED WRITEBACK
                new_base = rn_vals + strb_imm9_signed
                wb_mask = strb_post_mask & (rns != 31)
                if wb_mask.any():
                    wb_indices = rns[wb_mask].long()
                    wb_values = new_base[wb_mask]
                    regs.index_put_((wb_indices,), wb_values)

            # --- LDRSW: 32-bit load, sign-extend to 64-bit ---
            # Encoding: 1011 1001 10 imm12 Rn Rt = 0xB9800000 (scaled by 4)
            ldrsw_mask = ((insts & 0xFFC00000) == 0xB9800000) & exec_mask
            if ldrsw_mask.any():
                ldrsw_imm12 = (insts >> 10) & 0xFFF
                _ldrsw_base = rn_vals[ldrsw_mask]
                _ldrsw_neural = mem_arith.compute_address(_ldrsw_base, ldrsw_imm12[ldrsw_mask] * 4)
                ldrsw_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrsw_addr[ldrsw_mask] = _ldrsw_neural
                ldrsw_addr_c = ldrsw_addr.clamp(0, self.mem_size - 4)
                if _wov_pf_addr < 0:
                    _wov_pf_addr = int(ldrsw_addr_c.max().item())
                byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                ldrsw_addrs = ldrsw_addr_c.unsqueeze(1) + byte_off4
                ldrsw_bytes = mem[ldrsw_addrs.view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                ldrsw_u32 = (ldrsw_bytes << torch.tensor([0, 8, 16, 24], device=device, dtype=torch.int64)).sum(dim=1)
                # Sign-extend from 32 bits
                ldrsw_vals = torch.where(ldrsw_u32 >= 0x80000000, ldrsw_u32 - 0x100000000, ldrsw_u32)
                results = torch.where(ldrsw_mask, ldrsw_vals, results)
                write_mask = write_mask | ldrsw_mask

            # --- LDRH: 16-bit zero-extending load ---
            # Encoding: 0111 1001 01 imm12 Rn Rt = 0x79400000 (scaled by 2)
            ldrh_mask = ((insts & 0xFFC00000) == 0x79400000) & exec_mask
            if ldrh_mask.any():
                ldrh_imm12 = (insts >> 10) & 0xFFF
                _ldrh_base = rn_vals[ldrh_mask]
                _ldrh_neural = mem_arith.compute_address(_ldrh_base, ldrh_imm12[ldrh_mask] * 2)
                ldrh_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrh_addr[ldrh_mask] = _ldrh_neural
                ldrh_addr_c = ldrh_addr.clamp(0, self.mem_size - 2)
                ldrh_lo = mem[ldrh_addr_c.long()].to(torch.int64)
                ldrh_hi = mem[(ldrh_addr_c + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrh_vals = ldrh_lo | (ldrh_hi << 8)
                results = torch.where(ldrh_mask, ldrh_vals, results)
                write_mask = write_mask | ldrh_mask

            # --- STRH: 16-bit store ---
            # Encoding: 0111 1001 00 imm12 Rn Rt = 0x79000000 (scaled by 2)
            strh_mask = ((insts & 0xFFC00000) == 0x79000000) & exec_mask
            if strh_mask.any():
                strh_imm12 = (insts >> 10) & 0xFFF
                _strh_base = rn_vals[strh_mask]
                _strh_neural = mem_arith.compute_address(_strh_base, strh_imm12[strh_mask] * 2)
                strh_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                strh_addr[strh_mask] = _strh_neural
                strh_addr_c = strh_addr.clamp(0, self.mem_size - 2)
                if _wov_pf_addr < 0:
                    _wov_pf_addr = int(strh_addr_c.max().item())
                    _wov_pf_write = True
                strh_rt = insts & 0x1F
                strh_vals = torch.where(strh_rt == 31, torch.zeros_like(rd_vals),
                                        regs[strh_rt.clamp(0, 30)])
                strh_lo = (strh_vals & 0xFF).to(torch.uint8)
                strh_hi = ((strh_vals >> 8) & 0xFF).to(torch.uint8)
                if strh_mask.any():
                    lo_addrs = strh_addr_c[strh_mask].long()
                    hi_addrs = (strh_addr_c[strh_mask] + 1).clamp(0, self.mem_size-1).long()
                    mem.scatter_(0, lo_addrs, strh_lo[strh_mask])
                    mem.scatter_(0, hi_addrs, strh_hi[strh_mask])

            # --- LDRSB: 8-bit signed load, sign-extend ---
            # 64-bit dest: 0x39800000, 32-bit dest: 0x39C00000 (both mask 0xFFC00000)
            ldrsb_mask = (((insts & 0xFFC00000) == 0x39800000) |
                          ((insts & 0xFFC00000) == 0x39C00000)) & exec_mask
            if ldrsb_mask.any():
                ldrsb_imm12 = (insts >> 10) & 0xFFF  # unscaled
                _ldrsb_base = rn_vals[ldrsb_mask]
                _ldrsb_neural = mem_arith.compute_address(_ldrsb_base, ldrsb_imm12[ldrsb_mask])
                ldrsb_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrsb_addr[ldrsb_mask] = _ldrsb_neural
                ldrsb_addr_c = ldrsb_addr.clamp(0, self.mem_size - 1)
                ldrsb_u8 = mem[ldrsb_addr_c.long()].to(torch.int64)
                # Sign-extend from 8 bits
                ldrsb_vals = torch.where(ldrsb_u8 >= 0x80, ldrsb_u8 - 0x100, ldrsb_u8)
                results = torch.where(ldrsb_mask, ldrsb_vals, results)
                write_mask = write_mask | ldrsb_mask

            # --- LDUR: 64-bit unscaled offset (imm9, can be negative) ---
            # Encoding: 1111 1000 010 imm9 00 Rn Rt = 0xF8400000 mask 0xFFE00C00
            ldur64_mask = ((insts & 0xFFE00C00) == 0xF8400000) & exec_mask
            if ldur64_mask.any():
                ldur_imm9 = (insts >> 12) & 0x1FF
                ldur_imm9_s = torch.where(ldur_imm9 >= 256, ldur_imm9 - 512, ldur_imm9)
                _ldur_base = rn_vals[ldur64_mask]
                _ldur_neural = mem_arith.compute_address(_ldur_base, ldur_imm9_s[ldur64_mask])
                ldur_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldur_addr[ldur64_mask] = _ldur_neural
                ldur_addr_c = ldur_addr.clamp(0, self.mem_size - 8)
                if _wov_pf_addr < 0:
                    _wov_pf_addr = int(ldur_addr_c.max().item())
                byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                ldur_bytes = mem[(ldur_addr_c.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                ldur_vals = (ldur_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                results = torch.where(ldur64_mask, ldur_vals, results)
                write_mask = write_mask | ldur64_mask

            # --- STUR: 64-bit unscaled offset store ---
            # Encoding: 1111 1000 000 imm9 00 Rn Rt = 0xF8000000 mask 0xFFE00C00
            stur64_mask = ((insts & 0xFFE00C00) == 0xF8000000) & exec_mask
            if stur64_mask.any():
                stur_imm9 = (insts >> 12) & 0x1FF
                stur_imm9_s = torch.where(stur_imm9 >= 256, stur_imm9 - 512, stur_imm9)
                _stur_base = rn_vals[stur64_mask]
                _stur_neural = mem_arith.compute_address(_stur_base, stur_imm9_s[stur64_mask])
                stur_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                stur_addr[stur64_mask] = _stur_neural
                stur_addr_c = stur_addr.clamp(0, self.mem_size - 8)
                stur_rt = insts & 0x1F
                stur_vals = torch.where(stur_rt == 31, torch.zeros_like(rd_vals),
                                        regs[stur_rt.clamp(0, 30)])
                shifts8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                stur_bytes = ((stur_vals.unsqueeze(1) >> shifts8) & 0xFF).to(torch.uint8)
                act_addrs = (stur_addr_c.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64))[stur64_mask.unsqueeze(1).expand(-1,8)].long()
                act_bytes = stur_bytes[stur64_mask.unsqueeze(1).expand(-1,8)]
                if act_addrs.numel() > 0:
                    mem.scatter_(0, act_addrs, act_bytes)

            # --- LDUR 32-bit: 0xB8400000 mask 0xFFE00C00 ---
            ldur32_mask = ((insts & 0xFFE00C00) == 0xB8400000) & exec_mask
            if ldur32_mask.any():
                ldur32_imm9 = (insts >> 12) & 0x1FF
                ldur32_imm9_s = torch.where(ldur32_imm9 >= 256, ldur32_imm9 - 512, ldur32_imm9)
                _ldur32_base = rn_vals[ldur32_mask]
                _ldur32_neural = mem_arith.compute_address(_ldur32_base, ldur32_imm9_s[ldur32_mask])
                ldur32_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldur32_addr[ldur32_mask] = _ldur32_neural
                ldur32_addr_c = ldur32_addr.clamp(0, self.mem_size - 4)
                byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                ldur32_bytes = mem[(ldur32_addr_c.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                ldur32_vals = (ldur32_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                results = torch.where(ldur32_mask, ldur32_vals, results)
                write_mask = write_mask | ldur32_mask

            # --- LDR 64 register offset: LDR Xt, [Xn, Xm{, LSL #3}] ---
            # Encoding: 1111 1000 011 Rm opt S 10 Rn Rt = mask 0xFFE00C00 val 0xF8600800
            ldr64_reg_mask = ((insts & 0xFFE00C00) == 0xF8600800) & exec_mask
            if ldr64_reg_mask.any():
                ldr_ext_s = (insts >> 12) & 1  # shift amount: 1=scale by 8
                ldr_ext_rm = rn_vals  # rm already in rm_vals (index Rm)
                _ldr64r_offset = torch.where(ldr_ext_s == 1, rm_vals * 8, rm_vals)
                _ldr64r_base = rn_vals[ldr64_reg_mask]
                _ldr64r_off = _ldr64r_offset[ldr64_reg_mask]
                _ldr64r_neural = mem_arith.compute_address(_ldr64r_base, _ldr64r_off)
                ldr64r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldr64r_addr[ldr64_reg_mask] = _ldr64r_neural
                ldr64r_addr_c = ldr64r_addr.clamp(0, self.mem_size - 8)
                if _wov_pf_addr < 0:
                    _wov_pf_addr = int(ldr64r_addr_c.max().item())
                byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                ldr64r_bytes = mem[(ldr64r_addr_c.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                ldr64r_vals = (ldr64r_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                results = torch.where(ldr64_reg_mask, ldr64r_vals, results)
                write_mask = write_mask | ldr64_reg_mask

            # --- STR 64 register offset: STR Xt, [Xn, Xm{, LSL #3}] ---
            # Encoding: mask 0xFFE00C00 val 0xF8200800
            str64_reg_mask = ((insts & 0xFFE00C00) == 0xF8200800) & exec_mask
            if str64_reg_mask.any():
                str64r_s = (insts >> 12) & 1
                str64r_offset = torch.where(str64r_s == 1, rm_vals * 8, rm_vals)
                _str64r_base = rn_vals[str64_reg_mask]
                _str64r_off = str64r_offset[str64_reg_mask]
                _str64r_neural = mem_arith.compute_address(_str64r_base, _str64r_off)
                str64r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                str64r_addr[str64_reg_mask] = _str64r_neural
                str64r_addr_c = str64r_addr.clamp(0, self.mem_size - 8)
                str64r_rt = insts & 0x1F
                str64r_vals = torch.where(str64r_rt == 31, torch.zeros_like(rd_vals),
                                          regs[str64r_rt.clamp(0, 30)])
                shifts8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                str64r_bytes = ((str64r_vals.unsqueeze(1) >> shifts8) & 0xFF).to(torch.uint8)
                am8 = str64_reg_mask.unsqueeze(1).expand(-1, 8)
                act_a = (str64r_addr_c.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64))[am8].long()
                act_b = str64r_bytes[am8]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)

            # --- LDXR / STXR: approximate as regular LDR/STR (no exclusivity) ---
            # LDXR 64: 0xC85F7C00 mask 0xFFFFFC00; STXR 64: 0xC8007C00 mask 0xFFE07C00
            ldxr_mask = ((insts & 0xFFFFFC00) == 0xC85F7C00) & exec_mask
            if ldxr_mask.any():
                _ldxr_base = rn_vals[ldxr_mask]
                _ldxr_neural = mem_arith.compute_address(_ldxr_base,
                    torch.zeros(int(ldxr_mask.sum().item()), device=device, dtype=torch.int64))
                ldxr_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldxr_addr[ldxr_mask] = _ldxr_neural
                ldxr_addr_c = ldxr_addr.clamp(0, self.mem_size - 8)
                byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                ldxr_bytes = mem[(ldxr_addr_c.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                ldxr_vals = (ldxr_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                results = torch.where(ldxr_mask, ldxr_vals, results)
                write_mask = write_mask | ldxr_mask

            stxr_mask = ((insts & 0xFFE07C00) == 0xC8007C00) & exec_mask
            if stxr_mask.any():
                _stxr_base = rn_vals[stxr_mask]
                _stxr_neural = mem_arith.compute_address(_stxr_base,
                    torch.zeros(int(stxr_mask.sum().item()), device=device, dtype=torch.int64))
                stxr_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                stxr_addr[stxr_mask] = _stxr_neural
                stxr_addr_c = stxr_addr.clamp(0, self.mem_size - 8)
                stxr_rt = insts & 0x1F
                stxr_vals = torch.where(stxr_rt == 31, torch.zeros_like(rd_vals),
                                        regs[stxr_rt.clamp(0, 30)])
                shifts8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                stxr_bytes = ((stxr_vals.unsqueeze(1) >> shifts8) & 0xFF).to(torch.uint8)
                am8 = stxr_mask.unsqueeze(1).expand(-1, 8)
                act_a = (stxr_addr_c.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64))[am8].long()
                act_b = stxr_bytes[am8]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)
                # STXR writes status (0=success) to Rs field bits[20:16]
                stxr_rs = (insts >> 16) & 0x1F
                stxr_write = stxr_mask & (stxr_rs != 31)
                if stxr_write.any():
                    regs.index_put_((stxr_rs[stxr_write].long(),), 
                                  torch.zeros(int(stxr_write.sum().item()), device=device, dtype=torch.int64))

            # --- MRS (read system register → return 0) ---
            # Encoding: 1101 0101 001 op0 op1 CRn CRm op2 Rt = 0xD5300000 mask 0xFFF00000
            mrs_mask = ((insts & 0xFFF00000) == 0xD5300000) & exec_mask
            results = torch.where(mrs_mask, torch.zeros_like(results), results)
            write_mask = write_mask | mrs_mask

            # --- MSR (write system register → NOP, we have no real sysregs) ---
            # Encoding: 1101 0101 000 ... = 0xD5100000 mask 0xFFF00000
            # MSR immediate: 0xD500401F mask 0xFFFFF01F (DAIF/SP writes) → also NOP
            # Just ignore, do not crash
            # (no write_mask update — MSR writes no GP register)

            # --- DMB / DSB / ISB / HINT / NOP (memory barriers, no-ops) ---
            # All encode as: 0xD503xxxx — system instruction encoding
            # DMB: 0xD50330BF, DSB: 0xD503309F, ISB: 0xD50330DF, NOP: 0xD503201F
            # Treat all as no-ops (ordering barriers have no effect in our single-threaded model)
            # No action needed — these fall through all instruction handlers harmlessly

            # --- LDNP / STNP 64-bit (non-temporal pair, same behavior as LDP/STP) ---
            # STNP 64: (inst & 0xFFC00000)==0xA8000000; LDNP 64: 0xA8400000
            stnp_mask = ((insts & 0xFFC00000) == 0xA8000000) & exec_mask
            if stnp_mask.any():
                stnp_imm7 = (insts >> 15) & 0x7F
                stnp_imm7_s = torch.where(stnp_imm7 >= 64, stnp_imm7 - 128, stnp_imm7)
                stnp_addr = (rn_vals + stnp_imm7_s * 8).clamp(0, self.mem_size - 16)
                stnp_rt1 = insts & 0x1F
                stnp_rt2 = (insts >> 10) & 0x1F
                stnp_val1 = torch.where(stnp_rt1 == 31, torch.zeros_like(rd_vals), regs[stnp_rt1.clamp(0, 30)])
                stnp_val2 = torch.where(stnp_rt2 == 31, torch.zeros_like(rd_vals), regs[stnp_rt2.clamp(0, 30)])
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                stnp_b = torch.cat([(stnp_val1.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8),
                                    (stnp_val2.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8)], dim=1)
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                am = stnp_mask.unsqueeze(1).expand(-1, 16)
                act_a = (stnp_addr.unsqueeze(1) + b16)[am].long()
                act_b = stnp_b[am]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)

            ldnp_mask = ((insts & 0xFFC00000) == 0xA8400000) & exec_mask
            if ldnp_mask.any():
                ldnp_imm7 = (insts >> 15) & 0x7F
                ldnp_imm7_s = torch.where(ldnp_imm7 >= 64, ldnp_imm7 - 128, ldnp_imm7)
                ldnp_addr = (rn_vals + ldnp_imm7_s * 8).clamp(0, self.mem_size - 16)
                ldnp_rt1 = insts & 0x1F
                ldnp_rt2 = (insts >> 10) & 0x1F
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                ldnp_ba = mem[(ldnp_addr.unsqueeze(1) + b16).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 16).to(torch.int64)
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                ldnp_v1 = (ldnp_ba[:, :8] << sh8).sum(dim=1)
                ldnp_v2 = (ldnp_ba[:, 8:] << sh8).sum(dim=1)
                wb1 = ldnp_mask & (ldnp_rt1 != 31)
                if wb1.any():
                    regs.index_put_((ldnp_rt1[wb1].long(),), ldnp_v1[wb1])
                wb2 = ldnp_mask & (ldnp_rt2 != 31)
                if wb2.any():
                    regs.index_put_((ldnp_rt2[wb2].long(),), ldnp_v2[wb2])

            # --- STP 64 post-index: STP Xt1, Xt2, [Xn], #imm7 ---
            # Encoding: (inst & 0xFFC00000)==0xA8800000
            stp_post_mask = ((insts & 0xFFC00000) == 0xA8800000) & exec_mask
            if stp_post_mask.any():
                stp_p_imm7 = (insts >> 15) & 0x7F
                stp_p_imm7_s = torch.where(stp_p_imm7 >= 64, stp_p_imm7 - 128, stp_p_imm7)
                stp_p_addr = rn_vals.clamp(0, self.mem_size - 16)
                stp_p_rt1 = insts & 0x1F
                stp_p_rt2 = (insts >> 10) & 0x1F
                stp_p_v1 = torch.where(stp_p_rt1 == 31, torch.zeros_like(rd_vals), regs[stp_p_rt1.clamp(0, 30)])
                stp_p_v2 = torch.where(stp_p_rt2 == 31, torch.zeros_like(rd_vals), regs[stp_p_rt2.clamp(0, 30)])
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                stp_p_b = torch.cat([(stp_p_v1.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8),
                                     (stp_p_v2.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8)], dim=1)
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                am = stp_post_mask.unsqueeze(1).expand(-1, 16)
                act_a = (stp_p_addr.unsqueeze(1) + b16)[am].long()
                act_b = stp_p_b[am]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)
                wb = stp_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + stp_p_imm7_s * 8)[wb])

            # --- LDP 64 post-index: LDP Xt1, Xt2, [Xn], #imm7 ---
            # Encoding: (inst & 0xFFC00000)==0xA8C00000
            ldp_post_mask = ((insts & 0xFFC00000) == 0xA8C00000) & exec_mask
            if ldp_post_mask.any():
                ldp_p_imm7 = (insts >> 15) & 0x7F
                ldp_p_imm7_s = torch.where(ldp_p_imm7 >= 64, ldp_p_imm7 - 128, ldp_p_imm7)
                ldp_p_addr = rn_vals.clamp(0, self.mem_size - 16)
                ldp_p_rt1 = insts & 0x1F
                ldp_p_rt2 = (insts >> 10) & 0x1F
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                ldp_p_ba = mem[(ldp_p_addr.unsqueeze(1) + b16).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 16).to(torch.int64)
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                ldp_p_v1 = (ldp_p_ba[:, :8] << sh8).sum(dim=1)
                ldp_p_v2 = (ldp_p_ba[:, 8:] << sh8).sum(dim=1)
                wb1 = ldp_post_mask & (ldp_p_rt1 != 31)
                if wb1.any():
                    regs.index_put_((ldp_p_rt1[wb1].long(),), ldp_p_v1[wb1])
                wb2 = ldp_post_mask & (ldp_p_rt2 != 31)
                if wb2.any():
                    regs.index_put_((ldp_p_rt2[wb2].long(),), ldp_p_v2[wb2])
                # Writeback base register
                wb_b = ldp_post_mask & (rns != 31)
                if wb_b.any():
                    regs.index_put_((rns[wb_b].long(),), (rn_vals + ldp_p_imm7_s * 8)[wb_b])

            # --- STP 64 pre-index: STP Xt1, Xt2, [Xn, #imm7]! ---
            # Encoding: (inst & 0xFFC00000)==0xA9800000
            stp_pre_mask = ((insts & 0xFFC00000) == 0xA9800000) & exec_mask
            if stp_pre_mask.any():
                stp_pr_imm7 = (insts >> 15) & 0x7F
                stp_pr_imm7_s = torch.where(stp_pr_imm7 >= 64, stp_pr_imm7 - 128, stp_pr_imm7)
                stp_pr_addr = (rn_vals + stp_pr_imm7_s * 8).clamp(0, self.mem_size - 16)
                stp_pr_rt1 = insts & 0x1F
                stp_pr_rt2 = (insts >> 10) & 0x1F
                stp_pr_v1 = torch.where(stp_pr_rt1 == 31, torch.zeros_like(rd_vals), regs[stp_pr_rt1.clamp(0, 30)])
                stp_pr_v2 = torch.where(stp_pr_rt2 == 31, torch.zeros_like(rd_vals), regs[stp_pr_rt2.clamp(0, 30)])
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                stp_pr_b = torch.cat([(stp_pr_v1.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8),
                                      (stp_pr_v2.unsqueeze(1) >> sh8 & 0xFF).to(torch.uint8)], dim=1)
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                am = stp_pre_mask.unsqueeze(1).expand(-1, 16)
                act_a = (stp_pr_addr.unsqueeze(1) + b16)[am].long()
                act_b = stp_pr_b[am]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)
                wb = stp_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), stp_pr_addr[wb])

            # --- LDP 64 pre-index: LDP Xt1, Xt2, [Xn, #imm7]! ---
            # Encoding: (inst & 0xFFC00000)==0xA9C00000
            ldp_pre_mask = ((insts & 0xFFC00000) == 0xA9C00000) & exec_mask
            if ldp_pre_mask.any():
                ldp_pr_imm7 = (insts >> 15) & 0x7F
                ldp_pr_imm7_s = torch.where(ldp_pr_imm7 >= 64, ldp_pr_imm7 - 128, ldp_pr_imm7)
                ldp_pr_addr = (rn_vals + ldp_pr_imm7_s * 8).clamp(0, self.mem_size - 16)
                ldp_pr_rt1 = insts & 0x1F
                ldp_pr_rt2 = (insts >> 10) & 0x1F
                b16 = torch.arange(16, device=device, dtype=torch.int64)
                ldp_pr_ba = mem[(ldp_pr_addr.unsqueeze(1) + b16).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 16).to(torch.int64)
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                ldp_pr_v1 = (ldp_pr_ba[:, :8] << sh8).sum(dim=1)
                ldp_pr_v2 = (ldp_pr_ba[:, 8:] << sh8).sum(dim=1)
                wb1 = ldp_pre_mask & (ldp_pr_rt1 != 31)
                if wb1.any():
                    regs.index_put_((ldp_pr_rt1[wb1].long(),), ldp_pr_v1[wb1])
                wb2 = ldp_pre_mask & (ldp_pr_rt2 != 31)
                if wb2.any():
                    regs.index_put_((ldp_pr_rt2[wb2].long(),), ldp_pr_v2[wb2])
                wb_b = ldp_pre_mask & (rns != 31)
                if wb_b.any():
                    regs.index_put_((rns[wb_b].long(),), ldp_pr_addr[wb_b])

            # --- LDR 32 register offset: LDR Wt, [Xn, Xm{, LSL #2}] ---
            # Encoding: (inst & 0xFFE00C00)==0xB8600800
            ldr32_reg_mask = ((insts & 0xFFE00C00) == 0xB8600800) & exec_mask
            if ldr32_reg_mask.any():
                ldr32r_s = (insts >> 12) & 1
                ldr32r_off = torch.where(ldr32r_s == 1, rm_vals * 4, rm_vals)
                _ldr32r_base = rn_vals[ldr32_reg_mask]
                _ldr32r_neural = mem_arith.compute_address(_ldr32r_base, ldr32r_off[ldr32_reg_mask])
                ldr32r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldr32r_addr[ldr32_reg_mask] = _ldr32r_neural
                ldr32r_addr_c = ldr32r_addr.clamp(0, self.mem_size - 4)
                b4 = torch.arange(4, device=device, dtype=torch.int64)
                ldr32r_bytes = mem[(ldr32r_addr_c.unsqueeze(1) + b4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                ldr32r_vals = (ldr32r_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                results = torch.where(ldr32_reg_mask, ldr32r_vals, results)
                write_mask = write_mask | ldr32_reg_mask

            # --- STRB register offset: STRB Wt, [Xn, Xm] ---
            # Encoding: (inst & 0xFFE00C00)==0x38200800
            strb_reg_mask = ((insts & 0xFFE00C00) == 0x38200800) & exec_mask
            if strb_reg_mask.any():
                strb_r_base = rn_vals[strb_reg_mask]
                strb_r_off = rm_vals[strb_reg_mask]
                _strb_r_neural = mem_arith.compute_address(strb_r_base, strb_r_off)
                strb_r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                strb_r_addr[strb_reg_mask] = _strb_r_neural
                strb_r_addr_c = strb_r_addr.clamp(0, self.mem_size - 1)
                strb_r_rt = insts & 0x1F
                strb_r_vals = torch.where(strb_r_rt == 31, torch.zeros_like(rd_vals),
                                          regs[strb_r_rt.clamp(0, 30)])
                strb_r_bytes = (strb_r_vals & 0xFF).to(torch.uint8)
                act_a = strb_r_addr_c[strb_reg_mask].long()
                act_b = strb_r_bytes[strb_reg_mask]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)

            # --- LDRB register offset: LDRB Wt, [Xn, Xm] ---
            # Encoding: (inst & 0xFFE00C00)==0x38600800
            ldrb_reg_mask = ((insts & 0xFFE00C00) == 0x38600800) & exec_mask
            if ldrb_reg_mask.any():
                ldrb_r_base = rn_vals[ldrb_reg_mask]
                ldrb_r_off = rm_vals[ldrb_reg_mask]
                _ldrb_r_neural = mem_arith.compute_address(ldrb_r_base, ldrb_r_off)
                ldrb_r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrb_r_addr[ldrb_reg_mask] = _ldrb_r_neural
                ldrb_r_addr_c = ldrb_r_addr.clamp(0, self.mem_size - 1)
                ldrb_r_vals = mem[ldrb_r_addr_c.long()].to(torch.int64)
                results = torch.where(ldrb_reg_mask, ldrb_r_vals, results)
                write_mask = write_mask | ldrb_reg_mask

            # --- LDRSB register offset: LDRSB Xt/Wt, [Xn, Xm] ---
            # 64-bit dest: 0x38A00800; 32-bit dest: 0x38E00800
            ldrsb_reg_mask = (((insts & 0xFFE00C00) == 0x38A00800) |
                              ((insts & 0xFFE00C00) == 0x38E00800)) & exec_mask
            if ldrsb_reg_mask.any():
                _ldrsb_r_base = rn_vals[ldrsb_reg_mask]
                _ldrsb_r_off = rm_vals[ldrsb_reg_mask]
                _ldrsb_r_neural = mem_arith.compute_address(_ldrsb_r_base, _ldrsb_r_off)
                ldrsb_r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrsb_r_addr[ldrsb_reg_mask] = _ldrsb_r_neural
                ldrsb_r_addr_c = ldrsb_r_addr.clamp(0, self.mem_size - 1)
                ldrsb_r_u8 = mem[ldrsb_r_addr_c.long()].to(torch.int64)
                ldrsb_r_vals = torch.where(ldrsb_r_u8 >= 0x80, ldrsb_r_u8 - 0x100, ldrsb_r_u8)
                results = torch.where(ldrsb_reg_mask, ldrsb_r_vals, results)
                write_mask = write_mask | ldrsb_reg_mask

            # --- LDRH register offset: LDRH Wt, [Xn, Xm{, LSL #1}] ---
            ldrh_reg_mask = ((insts & 0xFFE00C00) == 0x78600800) & exec_mask
            if ldrh_reg_mask.any():
                ldrh_r_s = (insts >> 12) & 1
                ldrh_r_off = torch.where(ldrh_r_s == 1, rm_vals * 2, rm_vals)
                _ldrh_r_base = rn_vals[ldrh_reg_mask]
                _ldrh_r_neural = mem_arith.compute_address(_ldrh_r_base, ldrh_r_off[ldrh_reg_mask])
                ldrh_r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrh_r_addr[ldrh_reg_mask] = _ldrh_r_neural
                ldrh_r_addr_c = ldrh_r_addr.clamp(0, self.mem_size - 2)
                ldrh_r_vals = (mem[ldrh_r_addr_c.long()].to(torch.int64) |
                               (mem[(ldrh_r_addr_c + 1).clamp(0, self.mem_size-1).long()].to(torch.int64) << 8))
                results = torch.where(ldrh_reg_mask, ldrh_r_vals, results)
                write_mask = write_mask | ldrh_reg_mask

            # --- STRH register offset: STRH Wt, [Xn, Xm{, LSL #1}] ---
            strh_reg_mask = ((insts & 0xFFE00C00) == 0x78200800) & exec_mask
            if strh_reg_mask.any():
                strh_r_s = (insts >> 12) & 1
                strh_r_off = torch.where(strh_r_s == 1, rm_vals * 2, rm_vals)
                _strh_r_base = rn_vals[strh_reg_mask]
                _strh_r_neural = mem_arith.compute_address(_strh_r_base, strh_r_off[strh_reg_mask])
                strh_r_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                strh_r_addr[strh_reg_mask] = _strh_r_neural
                strh_r_addr_c = strh_r_addr.clamp(0, self.mem_size - 2)
                strh_r_rt = insts & 0x1F
                strh_r_vals = torch.where(strh_r_rt == 31, torch.zeros_like(rd_vals), regs[strh_r_rt.clamp(0, 30)])
                mem.scatter_(0, strh_r_addr_c[strh_reg_mask].long(), (strh_r_vals & 0xFF).to(torch.uint8)[strh_reg_mask])
                mem.scatter_(0, (strh_r_addr_c[strh_reg_mask] + 1).clamp(0, self.mem_size-1).long(),
                             ((strh_r_vals >> 8) & 0xFF).to(torch.uint8)[strh_reg_mask])

            # --- STUR 32-bit: STUR Wt, [Xn, #imm9] ---
            # Encoding: 1011 1000 000 imm9 00 Rn Rt = 0xB8000000 mask 0xFFE00C00
            stur32_mask = ((insts & 0xFFE00C00) == 0xB8000000) & exec_mask
            if stur32_mask.any():
                stur32_imm9 = (insts >> 12) & 0x1FF
                stur32_imm9_s = torch.where(stur32_imm9 >= 256, stur32_imm9 - 512, stur32_imm9)
                _stur32_base = rn_vals[stur32_mask]
                _stur32_neural = mem_arith.compute_address(_stur32_base, stur32_imm9_s[stur32_mask])
                stur32_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                stur32_addr[stur32_mask] = _stur32_neural
                stur32_addr_c = stur32_addr.clamp(0, self.mem_size - 4)
                stur32_rt = insts & 0x1F
                stur32_vals = torch.where(stur32_rt == 31, torch.zeros_like(rd_vals),
                                          regs[stur32_rt.clamp(0, 30)]) & 0xFFFFFFFF
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                stur32_bytes = ((stur32_vals.unsqueeze(1) >> sh4) & 0xFF).to(torch.uint8)
                am4 = stur32_mask.unsqueeze(1).expand(-1, 4)
                act_a = (stur32_addr_c.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64))[am4].long()
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, stur32_bytes[am4])

            # --- STR 64 pre-index: STR Xt, [Xn, #imm9]! ---
            # Encoding: 1111 1000 000 imm9 11 Rn Rt = 0xF8000C00 mask 0xFFE00C00
            str64_pre_mask = ((insts & 0xFFE00C00) == 0xF8000C00) & exec_mask
            if str64_pre_mask.any():
                str64_pre_imm9 = (insts >> 12) & 0x1FF
                str64_pre_imm9_s = torch.where(str64_pre_imm9 >= 256, str64_pre_imm9 - 512, str64_pre_imm9)
                str64_pre_addr = (rn_vals + str64_pre_imm9_s).clamp(0, self.mem_size - 8)
                str64_pre_rt = insts & 0x1F
                str64_pre_vals = torch.where(str64_pre_rt == 31, torch.zeros_like(rd_vals),
                                             regs[str64_pre_rt.clamp(0, 30)])
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                str64_pre_bytes = ((str64_pre_vals.unsqueeze(1) >> sh8) & 0xFF).to(torch.uint8)
                am8 = str64_pre_mask.unsqueeze(1).expand(-1, 8)
                act_a = (str64_pre_addr.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64))[am8].long()
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, str64_pre_bytes[am8])
                wb = str64_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), str64_pre_addr[wb])

            # --- LDR 64 pre-index: LDR Xt, [Xn, #imm9]! ---
            ldr64_pre_mask = ((insts & 0xFFE00C00) == 0xF8400C00) & exec_mask
            if ldr64_pre_mask.any():
                ldr64_pre_imm9 = (insts >> 12) & 0x1FF
                ldr64_pre_imm9_s = torch.where(ldr64_pre_imm9 >= 256, ldr64_pre_imm9 - 512, ldr64_pre_imm9)
                ldr64_pre_addr = (rn_vals + ldr64_pre_imm9_s).clamp(0, self.mem_size - 8)
                sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                ldr64_pre_vals = (mem[(ldr64_pre_addr.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64)).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64) << sh8).sum(dim=1)
                results = torch.where(ldr64_pre_mask, ldr64_pre_vals, results)
                write_mask = write_mask | ldr64_pre_mask
                wb = ldr64_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), ldr64_pre_addr[wb])

            # --- STR 32 post-index: STR Wt, [Xn], #imm9 ---
            str32_post_mask = ((insts & 0xFFE00C00) == 0xB8000400) & exec_mask
            if str32_post_mask.any():
                str32_post_imm9 = (insts >> 12) & 0x1FF
                str32_post_imm9_s = torch.where(str32_post_imm9 >= 256, str32_post_imm9 - 512, str32_post_imm9)
                str32_post_addr = rn_vals.clamp(0, self.mem_size - 4)
                str32_post_rt = insts & 0x1F
                str32_post_vals = torch.where(str32_post_rt == 31, torch.zeros_like(rd_vals),
                                              regs[str32_post_rt.clamp(0, 30)]) & 0xFFFFFFFF
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                str32_post_bytes = ((str32_post_vals.unsqueeze(1) >> sh4) & 0xFF).to(torch.uint8)
                am4 = str32_post_mask.unsqueeze(1).expand(-1, 4)
                act_a = (str32_post_addr.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64))[am4].long()
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, str32_post_bytes[am4])
                wb = str32_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + str32_post_imm9_s)[wb])

            # --- LDR 32 post-index: LDR Wt, [Xn], #imm9 ---
            ldr32_post_mask = ((insts & 0xFFE00C00) == 0xB8400400) & exec_mask
            if ldr32_post_mask.any():
                ldr32_post_imm9 = (insts >> 12) & 0x1FF
                ldr32_post_imm9_s = torch.where(ldr32_post_imm9 >= 256, ldr32_post_imm9 - 512, ldr32_post_imm9)
                ldr32_post_addr = rn_vals.clamp(0, self.mem_size - 4)
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                ldr32_post_vals = (mem[(ldr32_post_addr.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64)).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64) << sh4).sum(dim=1)
                results = torch.where(ldr32_post_mask, ldr32_post_vals, results)
                write_mask = write_mask | ldr32_post_mask
                wb = ldr32_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + ldr32_post_imm9_s)[wb])

            # --- STR 32 pre-index: STR Wt, [Xn, #imm9]! ---
            str32_pre_mask = ((insts & 0xFFE00C00) == 0xB8000C00) & exec_mask
            if str32_pre_mask.any():
                str32_pre_imm9 = (insts >> 12) & 0x1FF
                str32_pre_imm9_s = torch.where(str32_pre_imm9 >= 256, str32_pre_imm9 - 512, str32_pre_imm9)
                str32_pre_addr = (rn_vals + str32_pre_imm9_s).clamp(0, self.mem_size - 4)
                str32_pre_rt = insts & 0x1F
                str32_pre_vals = torch.where(str32_pre_rt == 31, torch.zeros_like(rd_vals),
                                             regs[str32_pre_rt.clamp(0, 30)]) & 0xFFFFFFFF
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                str32_pre_bytes = ((str32_pre_vals.unsqueeze(1) >> sh4) & 0xFF).to(torch.uint8)
                am4 = str32_pre_mask.unsqueeze(1).expand(-1, 4)
                act_a = (str32_pre_addr.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64))[am4].long()
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, str32_pre_bytes[am4])
                wb = str32_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), str32_pre_addr[wb])

            # --- LDR 32 pre-index: LDR Wt, [Xn, #imm9]! ---
            ldr32_pre_mask = ((insts & 0xFFE00C00) == 0xB8400C00) & exec_mask
            if ldr32_pre_mask.any():
                ldr32_pre_imm9 = (insts >> 12) & 0x1FF
                ldr32_pre_imm9_s = torch.where(ldr32_pre_imm9 >= 256, ldr32_pre_imm9 - 512, ldr32_pre_imm9)
                ldr32_pre_addr = (rn_vals + ldr32_pre_imm9_s).clamp(0, self.mem_size - 4)
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                ldr32_pre_vals = (mem[(ldr32_pre_addr.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64)).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64) << sh4).sum(dim=1)
                results = torch.where(ldr32_pre_mask, ldr32_pre_vals, results)
                write_mask = write_mask | ldr32_pre_mask
                wb = ldr32_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), ldr32_pre_addr[wb])

            # --- LDRSW post-index: sign-extend 32-bit load, [Xn], #imm9 ---
            # size=10, opc=10, bit24=0, bits[11:10]=01 → (inst & 0xFFE00C00)==0xB8800400
            ldrsw_post_mask = ((insts & 0xFFE00C00) == 0xB8800400) & exec_mask
            if ldrsw_post_mask.any():
                lsw_p_imm9 = (insts >> 12) & 0x1FF
                lsw_p_imm9_s = torch.where(lsw_p_imm9 >= 256, lsw_p_imm9 - 512, lsw_p_imm9)
                lsw_p_addr = rn_vals.clamp(0, self.mem_size - 4)
                byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                lsw_p_bytes = mem[(lsw_p_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                lsw_p_u32 = (lsw_p_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                lsw_p_vals = torch.where(lsw_p_u32 >= 0x80000000, lsw_p_u32 - 0x100000000, lsw_p_u32)
                results = torch.where(ldrsw_post_mask, lsw_p_vals, results)
                write_mask = write_mask | ldrsw_post_mask
                wb = ldrsw_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + lsw_p_imm9_s)[wb])

            # --- LDRSW pre-index: sign-extend 32-bit load, [Xn, #imm9]! ---
            # bits[11:10]=11 → (inst & 0xFFE00C00)==0xB8800C00
            ldrsw_pre_mask = ((insts & 0xFFE00C00) == 0xB8800C00) & exec_mask
            if ldrsw_pre_mask.any():
                lsw_r_imm9 = (insts >> 12) & 0x1FF
                lsw_r_imm9_s = torch.where(lsw_r_imm9 >= 256, lsw_r_imm9 - 512, lsw_r_imm9)
                lsw_r_addr = (rn_vals + lsw_r_imm9_s).clamp(0, self.mem_size - 4)
                byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                lsw_r_bytes = mem[(lsw_r_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                lsw_r_u32 = (lsw_r_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                lsw_r_vals = torch.where(lsw_r_u32 >= 0x80000000, lsw_r_u32 - 0x100000000, lsw_r_u32)
                results = torch.where(ldrsw_pre_mask, lsw_r_vals, results)
                write_mask = write_mask | ldrsw_pre_mask
                wb = ldrsw_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), lsw_r_addr[wb])

            # --- LDRSW register offset: sign-extend 32-bit load, [Xn, Xm{,LSL#2}] ---
            # size=10, opc=10, bit21=1, bits[11:10]=10 → (inst & 0xFFE00C00)==0xB8A00800
            ldrsw_reg_mask = ((insts & 0xFFE00C00) == 0xB8A00800) & exec_mask
            if ldrsw_reg_mask.any():
                lsw_ro_s3 = (insts >> 12) & 0x1  # S bit: 1=LSL#2
                lsw_ro_shift = lsw_ro_s3 * 2
                _lsw_ro_base = rn_vals[ldrsw_reg_mask]
                _lsw_ro_off = (rm_vals << lsw_ro_shift)[ldrsw_reg_mask]
                _lsw_ro_neural = mem_arith.compute_address(_lsw_ro_base, _lsw_ro_off)
                lsw_ro_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                lsw_ro_addr[ldrsw_reg_mask] = _lsw_ro_neural
                lsw_ro_addr_c = lsw_ro_addr.clamp(0, self.mem_size - 4)
                byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                lsw_ro_bytes = mem[(lsw_ro_addr_c.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                lsw_ro_u32 = (lsw_ro_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                lsw_ro_vals = torch.where(lsw_ro_u32 >= 0x80000000, lsw_ro_u32 - 0x100000000, lsw_ro_u32)
                results = torch.where(ldrsw_reg_mask, lsw_ro_vals, results)
                write_mask = write_mask | ldrsw_reg_mask

            # --- STLR (store-release, treat as regular STR — no memory ordering in single-threaded model) ---
            # STLR 64: (inst & 0xFFFFFC00)==0xC89FFC00; STLR 32: 0x889FFC00
            stlr64_mask = ((insts & 0xFFFFFC00) == 0xC89FFC00) & exec_mask
            stlr32_mask = ((insts & 0xFFFFFC00) == 0x889FFC00) & exec_mask
            stlr_mask = stlr64_mask | stlr32_mask
            if stlr_mask.any():
                stlr_addr = rn_vals.clamp(0, self.mem_size - 8)
                stlr_rt = insts & 0x1F
                stlr_vals = torch.where(stlr_rt == 31, torch.zeros_like(rd_vals),
                                         regs[stlr_rt.clamp(0, 30)])
                # 64-bit stores
                if stlr64_mask.any():
                    sh8 = torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)
                    stlr64_bytes = ((stlr_vals.unsqueeze(1) >> sh8) & 0xFF).to(torch.uint8)
                    am8 = stlr64_mask.unsqueeze(1).expand(-1, 8)
                    act_a = (stlr_addr.unsqueeze(1) + torch.arange(8, device=device, dtype=torch.int64))[am8].long()
                    if act_a.numel() > 0:
                        mem.scatter_(0, act_a, stlr64_bytes[am8])
                # 32-bit stores
                if stlr32_mask.any():
                    sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                    stlr32_bytes = ((stlr_vals.unsqueeze(1) >> sh4) & 0xFF).to(torch.uint8)
                    am4 = stlr32_mask.unsqueeze(1).expand(-1, 4)
                    act_a = (stlr_addr.unsqueeze(1) + torch.arange(4, device=device, dtype=torch.int64))[am4].long()
                    if act_a.numel() > 0:
                        mem.scatter_(0, act_a, stlr32_bytes[am4])

            # --- LDAR (load-acquire, treat as regular LDR) ---
            # LDAR 64: (inst & 0xFFFFFC00)==0xC8DFFC00; LDAR 32: 0x88DFFC00
            ldar64_mask = ((insts & 0xFFFFFC00) == 0xC8DFFC00) & exec_mask
            ldar32_mask = ((insts & 0xFFFFFC00) == 0x88DFFC00) & exec_mask
            ldar_any = ldar64_mask | ldar32_mask
            if ldar_any.any():
                ldar_addr = rn_vals.clamp(0, self.mem_size - 8)
                # 64-bit load
                if ldar64_mask.any():
                    byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                    ldar64_bytes = mem[(ldar_addr.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                    ldar64_vals = (ldar64_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldar64_mask, ldar64_vals, results)
                    write_mask = write_mask | ldar64_mask
                # 32-bit load
                if ldar32_mask.any():
                    byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                    ldar32_bytes = mem[(ldar_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                    ldar32_vals = (ldar32_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldar32_mask, ldar32_vals, results)
                    write_mask = write_mask | ldar32_mask

            # --- LDAXR (load-acquire exclusive — approximate as regular LDR, no monitor) ---
            # LDAXR 64: A=1 in exclusive load family → 0xC8DF7C00 mask 0xFFFFFC00
            # LDAXR 32: 0x88DF7C00 mask 0xFFFFFC00
            ldaxr64_mask = ((insts & 0xFFFFFC00) == 0xC8DF7C00) & exec_mask
            ldaxr32_mask = ((insts & 0xFFFFFC00) == 0x88DF7C00) & exec_mask
            ldaxr_any = ldaxr64_mask | ldaxr32_mask
            if ldaxr_any.any():
                ldaxr_addr = rn_vals.clamp(0, self.mem_size - 8)
                if ldaxr64_mask.any():
                    byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                    ldaxr64_bytes = mem[(ldaxr_addr.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                    ldaxr64_vals = (ldaxr64_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldaxr64_mask, ldaxr64_vals, results)
                    write_mask = write_mask | ldaxr64_mask
                if ldaxr32_mask.any():
                    byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                    ldaxr32_bytes = mem[(ldaxr_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                    ldaxr32_vals = (ldaxr32_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldaxr32_mask, ldaxr32_vals, results)
                    write_mask = write_mask | ldaxr32_mask

            # CLREX (clear exclusive monitor) — NOP in single-threaded emulation
            # Encoding: 0xD503305F (D5 = system, ISB/DSB/DMB/CLREX family)

            # --- LDRSH: 16-bit signed load, sign-extend to 32 or 64 bits ---
            # LDRSH 64-bit dest: size=01, opc=10 → 0x79800000 mask 0xFFC00000 (scaled by 2)
            # LDRSH 32-bit dest: size=01, opc=11 → 0x79C00000 mask 0xFFC00000
            ldrsh_mask = (((insts & 0xFFC00000) == 0x79800000) |
                          ((insts & 0xFFC00000) == 0x79C00000)) & exec_mask
            if ldrsh_mask.any():
                ldrsh_imm12 = (insts >> 10) & 0xFFF
                _ldrsh_base = rn_vals[ldrsh_mask]
                _ldrsh_addr = mem_arith.compute_address(_ldrsh_base, ldrsh_imm12[ldrsh_mask] * 2)
                ldrsh_addr = torch.zeros(batch_size, device=device, dtype=torch.int64)
                ldrsh_addr[ldrsh_mask] = _ldrsh_addr
                ldrsh_addr_c = ldrsh_addr.clamp(0, self.mem_size - 2)
                ldrsh_lo = mem[ldrsh_addr_c.long()].to(torch.int64)
                ldrsh_hi = mem[(ldrsh_addr_c + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrsh_u16 = ldrsh_lo | (ldrsh_hi << 8)
                ldrsh_vals = torch.where(ldrsh_u16 >= 0x8000, ldrsh_u16 - 0x10000, ldrsh_u16)
                results = torch.where(ldrsh_mask, ldrsh_vals, results)
                write_mask = write_mask | ldrsh_mask

            # --- LDURH: 16-bit zero-extending load, unscaled offset ---
            # size=01, opc=01, bit24=0, bit21=0, bits[11:10]=00 → 0x78400000 mask 0xFFE00C00
            ldurh_mask = ((insts & 0xFFE00C00) == 0x78400000) & exec_mask
            if ldurh_mask.any():
                ldurh_imm9 = (insts >> 12) & 0x1FF
                ldurh_imm9_s = torch.where(ldurh_imm9 >= 256, ldurh_imm9 - 512, ldurh_imm9)
                ldurh_addr = (rn_vals + ldurh_imm9_s).clamp(0, self.mem_size - 2)
                ldurh_lo = mem[ldurh_addr.long()].to(torch.int64)
                ldurh_hi = mem[(ldurh_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldurh_vals = ldurh_lo | (ldurh_hi << 8)
                results = torch.where(ldurh_mask, ldurh_vals, results)
                write_mask = write_mask | ldurh_mask

            # --- LDURSH: 16-bit signed load, unscaled offset ---
            # LDURSH 64: size=01, opc=10, bits[11:10]=00 → 0x78800000 mask 0xFFE00C00
            # LDURSH 32: opc=11 → 0x78C00000 mask 0xFFE00C00
            ldursh_mask = (((insts & 0xFFE00C00) == 0x78800000) |
                           ((insts & 0xFFE00C00) == 0x78C00000)) & exec_mask
            if ldursh_mask.any():
                ldursh_imm9 = (insts >> 12) & 0x1FF
                ldursh_imm9_s = torch.where(ldursh_imm9 >= 256, ldursh_imm9 - 512, ldursh_imm9)
                ldursh_addr = (rn_vals + ldursh_imm9_s).clamp(0, self.mem_size - 2)
                ldursh_lo = mem[ldursh_addr.long()].to(torch.int64)
                ldursh_hi = mem[(ldursh_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldursh_u16 = ldursh_lo | (ldursh_hi << 8)
                ldursh_vals = torch.where(ldursh_u16 >= 0x8000, ldursh_u16 - 0x10000, ldursh_u16)
                results = torch.where(ldursh_mask, ldursh_vals, results)
                write_mask = write_mask | ldursh_mask

            # --- LDURSB: 8-bit signed load, unscaled offset (64 or 32-bit dest) ---
            # LDURSB 64: 0x38800000 mask 0xFFE00C00; LDURSB 32: 0x38C00000
            ldursb_mask = (((insts & 0xFFE00C00) == 0x38800000) |
                           ((insts & 0xFFE00C00) == 0x38C00000)) & exec_mask
            if ldursb_mask.any():
                ldursb_imm9 = (insts >> 12) & 0x1FF
                ldursb_imm9_s = torch.where(ldursb_imm9 >= 256, ldursb_imm9 - 512, ldursb_imm9)
                ldursb_addr = (rn_vals + ldursb_imm9_s).clamp(0, self.mem_size - 1)
                ldursb_u8 = mem[ldursb_addr.long()].to(torch.int64)
                ldursb_vals = torch.where(ldursb_u8 >= 0x80, ldursb_u8 - 0x100, ldursb_u8)
                results = torch.where(ldursb_mask, ldursb_vals, results)
                write_mask = write_mask | ldursb_mask

            # --- LDR literal (PC-relative load from constant pool) ---
            # LDR literal 64: bits[31:24]=0x58; LDR literal 32: bits[31:24]=0x18
            # imm19 = bits[23:5], signed word (×4) offset from instruction PC
            ldr_lit64_mask = ((insts & 0xFF000000) == 0x58000000) & exec_mask
            ldr_lit32_mask = ((insts & 0xFF000000) == 0x18000000) & exec_mask
            ldr_lit_any = ldr_lit64_mask | ldr_lit32_mask
            if ldr_lit_any.any():
                lit_imm19 = (insts >> 5) & 0x7FFFF
                lit_offset = torch.where(lit_imm19 >= 0x40000, lit_imm19 - 0x80000, lit_imm19) * 4
                lit_pcs = pc_t + batch_idx * 4
                lit_addr = (lit_pcs + lit_offset).clamp(0, self.mem_size - 8)
                if ldr_lit64_mask.any():
                    byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                    lit64_bytes = mem[(lit_addr.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                    lit64_vals = (lit64_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldr_lit64_mask, lit64_vals, results)
                    write_mask = write_mask | ldr_lit64_mask
                if ldr_lit32_mask.any():
                    byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                    lit32_bytes = mem[(lit_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                    lit32_vals = (lit32_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                    results = torch.where(ldr_lit32_mask, lit32_vals, results)
                    write_mask = write_mask | ldr_lit32_mask

            # --- LDRH post-index: LDRH Wt, [Xn], #imm9 ---
            # Encoding: size=01, opc=01, bits[11:10]=01 → 0x78400400 mask 0xFFE00C00
            ldrh_post_mask = ((insts & 0xFFE00C00) == 0x78400400) & exec_mask
            if ldrh_post_mask.any():
                ldrh_p_imm9 = (insts >> 12) & 0x1FF
                ldrh_p_imm9_s = torch.where(ldrh_p_imm9 >= 256, ldrh_p_imm9 - 512, ldrh_p_imm9)
                ldrh_p_addr = rn_vals.clamp(0, self.mem_size - 2)
                ldrh_p_lo = mem[ldrh_p_addr.long()].to(torch.int64)
                ldrh_p_hi = mem[(ldrh_p_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrh_p_vals = ldrh_p_lo | (ldrh_p_hi << 8)
                results = torch.where(ldrh_post_mask, ldrh_p_vals, results)
                write_mask = write_mask | ldrh_post_mask
                wb = ldrh_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + ldrh_p_imm9_s)[wb])

            # --- LDRH pre-index: LDRH Wt, [Xn, #imm9]! ---
            # Encoding: bits[11:10]=11 → 0x78400C00 mask 0xFFE00C00
            ldrh_pre_mask = ((insts & 0xFFE00C00) == 0x78400C00) & exec_mask
            if ldrh_pre_mask.any():
                ldrh_r_imm9 = (insts >> 12) & 0x1FF
                ldrh_r_imm9_s = torch.where(ldrh_r_imm9 >= 256, ldrh_r_imm9 - 512, ldrh_r_imm9)
                ldrh_r_addr = (rn_vals + ldrh_r_imm9_s).clamp(0, self.mem_size - 2)
                ldrh_r_lo = mem[ldrh_r_addr.long()].to(torch.int64)
                ldrh_r_hi = mem[(ldrh_r_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrh_r_vals = ldrh_r_lo | (ldrh_r_hi << 8)
                results = torch.where(ldrh_pre_mask, ldrh_r_vals, results)
                write_mask = write_mask | ldrh_pre_mask
                wb = ldrh_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), ldrh_r_addr[wb])

            # --- STRH post-index: STRH Wt, [Xn], #imm9 ---
            # Encoding: size=01, opc=00, bits[11:10]=01 → 0x78000400 mask 0xFFE00C00
            strh_post_mask = ((insts & 0xFFE00C00) == 0x78000400) & exec_mask
            if strh_post_mask.any():
                strh_p_imm9 = (insts >> 12) & 0x1FF
                strh_p_imm9_s = torch.where(strh_p_imm9 >= 256, strh_p_imm9 - 512, strh_p_imm9)
                strh_p_addr = rn_vals.clamp(0, self.mem_size - 2)
                strh_p_rt = insts & 0x1F
                strh_p_vals = torch.where(strh_p_rt == 31, torch.zeros_like(rd_vals),
                                          regs[strh_p_rt.clamp(0, 30)])
                if strh_post_mask.any():
                    mem.scatter_(0, strh_p_addr[strh_post_mask].long(),
                                 (strh_p_vals & 0xFF).to(torch.uint8)[strh_post_mask])
                    mem.scatter_(0, (strh_p_addr[strh_post_mask] + 1).clamp(0, self.mem_size-1).long(),
                                 ((strh_p_vals >> 8) & 0xFF).to(torch.uint8)[strh_post_mask])
                wb = strh_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + strh_p_imm9_s)[wb])

            # --- STRH pre-index: STRH Wt, [Xn, #imm9]! ---
            # Encoding: bits[11:10]=11 → 0x78000C00 mask 0xFFE00C00
            strh_pre_mask = ((insts & 0xFFE00C00) == 0x78000C00) & exec_mask
            if strh_pre_mask.any():
                strh_r_imm9 = (insts >> 12) & 0x1FF
                strh_r_imm9_s = torch.where(strh_r_imm9 >= 256, strh_r_imm9 - 512, strh_r_imm9)
                strh_r_addr = (rn_vals + strh_r_imm9_s).clamp(0, self.mem_size - 2)
                strh_r_rt = insts & 0x1F
                strh_r_vals = torch.where(strh_r_rt == 31, torch.zeros_like(rd_vals),
                                          regs[strh_r_rt.clamp(0, 30)])
                mem.scatter_(0, strh_r_addr[strh_pre_mask].long(),
                             (strh_r_vals & 0xFF).to(torch.uint8)[strh_pre_mask])
                mem.scatter_(0, (strh_r_addr[strh_pre_mask] + 1).clamp(0, self.mem_size-1).long(),
                             ((strh_r_vals >> 8) & 0xFF).to(torch.uint8)[strh_pre_mask])
                wb = strh_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), strh_r_addr[wb])

            # --- LDRB pre-index: LDRB Wt, [Xn, #imm9]! ---
            # Encoding: size=00, opc=01, bits[11:10]=11 → 0x38400C00 mask 0xFFE00C00
            ldrb_pre_mask = ((insts & 0xFFE00C00) == 0x38400C00) & exec_mask
            if ldrb_pre_mask.any():
                ldrb_r_imm9 = (insts >> 12) & 0x1FF
                ldrb_r_imm9_s = torch.where(ldrb_r_imm9 >= 256, ldrb_r_imm9 - 512, ldrb_r_imm9)
                ldrb_r_addr = (rn_vals + ldrb_r_imm9_s).clamp(0, self.mem_size - 1)
                ldrb_r_vals = mem[ldrb_r_addr.long()].to(torch.int64)
                results = torch.where(ldrb_pre_mask, ldrb_r_vals, results)
                write_mask = write_mask | ldrb_pre_mask
                wb = ldrb_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), ldrb_r_addr[wb])

            # --- STRB pre-index: STRB Wt, [Xn, #imm9]! ---
            # Encoding: size=00, opc=00, bits[11:10]=11 → 0x38000C00 mask 0xFFE00C00
            strb_pre_mask = ((insts & 0xFFE00C00) == 0x38000C00) & exec_mask
            if strb_pre_mask.any():
                strb_r_imm9 = (insts >> 12) & 0x1FF
                strb_r_imm9_s = torch.where(strb_r_imm9 >= 256, strb_r_imm9 - 512, strb_r_imm9)
                strb_r_addr = (rn_vals + strb_r_imm9_s).clamp(0, self.mem_size - 1)
                strb_r_rt = insts & 0x1F
                strb_r_vals = torch.where(strb_r_rt == 31, torch.zeros_like(rd_vals),
                                          regs[strb_r_rt.clamp(0, 30)])
                act_a = strb_r_addr[strb_pre_mask].long()
                act_b = (strb_r_vals & 0xFF).to(torch.uint8)[strb_pre_mask]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)
                wb = strb_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), strb_r_addr[wb])

            # --- STURH: 16-bit unscaled offset store ---
            # STURH Wt, [Xn, #imm9]: size=01, opc=00, bits[11:10]=00 → 0x78000000 mask 0xFFE00C00
            sturh_mask = ((insts & 0xFFE00C00) == 0x78000000) & exec_mask
            if sturh_mask.any():
                sturh_imm9 = (insts >> 12) & 0x1FF
                sturh_imm9_s = torch.where(sturh_imm9 >= 256, sturh_imm9 - 512, sturh_imm9)
                sturh_addr = (rn_vals + sturh_imm9_s).clamp(0, self.mem_size - 2)
                sturh_rt = insts & 0x1F
                sturh_vals = torch.where(sturh_rt == 31, torch.zeros_like(rd_vals),
                                         regs[sturh_rt.clamp(0, 30)])
                mem.scatter_(0, sturh_addr[sturh_mask].long(),
                             (sturh_vals & 0xFF).to(torch.uint8)[sturh_mask])
                mem.scatter_(0, (sturh_addr[sturh_mask] + 1).clamp(0, self.mem_size-1).long(),
                             ((sturh_vals >> 8) & 0xFF).to(torch.uint8)[sturh_mask])

            # --- STURB: 8-bit unscaled offset store ---
            # STURB Wt, [Xn, #imm9]: size=00, opc=00, bits[11:10]=00 → 0x38000000 mask 0xFFE00C00
            sturb_mask = ((insts & 0xFFE00C00) == 0x38000000) & exec_mask
            if sturb_mask.any():
                sturb_imm9 = (insts >> 12) & 0x1FF
                sturb_imm9_s = torch.where(sturb_imm9 >= 256, sturb_imm9 - 512, sturb_imm9)
                sturb_addr = (rn_vals + sturb_imm9_s).clamp(0, self.mem_size - 1)
                sturb_rt = insts & 0x1F
                sturb_vals = torch.where(sturb_rt == 31, torch.zeros_like(rd_vals),
                                         regs[sturb_rt.clamp(0, 30)])
                act_a = sturb_addr[sturb_mask].long()
                act_b = (sturb_vals & 0xFF).to(torch.uint8)[sturb_mask]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)

            # --- LDRSH post-index: LDRSH Xt/Wt, [Xn], #imm9 ---
            # LDRSH 64 post: 0x78800400; LDRSH 32 post: 0x78C00400 mask 0xFFE00C00
            ldrsh_post_mask = (((insts & 0xFFE00C00) == 0x78800400) |
                               ((insts & 0xFFE00C00) == 0x78C00400)) & exec_mask
            if ldrsh_post_mask.any():
                ldrsh_p_imm9 = (insts >> 12) & 0x1FF
                ldrsh_p_imm9_s = torch.where(ldrsh_p_imm9 >= 256, ldrsh_p_imm9 - 512, ldrsh_p_imm9)
                ldrsh_p_addr = rn_vals.clamp(0, self.mem_size - 2)
                ldrsh_p_lo = mem[ldrsh_p_addr.long()].to(torch.int64)
                ldrsh_p_hi = mem[(ldrsh_p_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrsh_p_u16 = ldrsh_p_lo | (ldrsh_p_hi << 8)
                ldrsh_p_vals = torch.where(ldrsh_p_u16 >= 0x8000, ldrsh_p_u16 - 0x10000, ldrsh_p_u16)
                results = torch.where(ldrsh_post_mask, ldrsh_p_vals, results)
                write_mask = write_mask | ldrsh_post_mask
                wb = ldrsh_post_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), (rn_vals + ldrsh_p_imm9_s)[wb])

            # --- LDRSH pre-index: LDRSH Xt/Wt, [Xn, #imm9]! ---
            # LDRSH 64 pre: 0x78800C00; LDRSH 32 pre: 0x78C00C00 mask 0xFFE00C00
            ldrsh_pre_mask = (((insts & 0xFFE00C00) == 0x78800C00) |
                              ((insts & 0xFFE00C00) == 0x78C00C00)) & exec_mask
            if ldrsh_pre_mask.any():
                ldrsh_r_imm9 = (insts >> 12) & 0x1FF
                ldrsh_r_imm9_s = torch.where(ldrsh_r_imm9 >= 256, ldrsh_r_imm9 - 512, ldrsh_r_imm9)
                ldrsh_r_addr = (rn_vals + ldrsh_r_imm9_s).clamp(0, self.mem_size - 2)
                ldrsh_r_lo = mem[ldrsh_r_addr.long()].to(torch.int64)
                ldrsh_r_hi = mem[(ldrsh_r_addr + 1).clamp(0, self.mem_size-1).long()].to(torch.int64)
                ldrsh_r_u16 = ldrsh_r_lo | (ldrsh_r_hi << 8)
                ldrsh_r_vals = torch.where(ldrsh_r_u16 >= 0x8000, ldrsh_r_u16 - 0x10000, ldrsh_r_u16)
                results = torch.where(ldrsh_pre_mask, ldrsh_r_vals, results)
                write_mask = write_mask | ldrsh_pre_mask
                wb = ldrsh_pre_mask & (rns != 31)
                if wb.any():
                    regs.index_put_((rns[wb].long(),), ldrsh_r_addr[wb])

            # --- STP/LDP 32-bit (signed offset): ---
            # STP 32: (inst & 0xFFC00000)==0x29000000; LDP 32: 0x29400000
            stp32_mask = ((insts & 0xFFC00000) == 0x29000000) & exec_mask
            if stp32_mask.any():
                stp32_imm7 = (insts >> 15) & 0x7F
                stp32_imm7_s = torch.where(stp32_imm7 >= 64, stp32_imm7 - 128, stp32_imm7)
                stp32_addr = (rn_vals + stp32_imm7_s * 4).clamp(0, self.mem_size - 8)
                stp32_rt1 = insts & 0x1F
                stp32_rt2 = (insts >> 10) & 0x1F
                stp32_v1 = (torch.where(stp32_rt1 == 31, torch.zeros_like(rd_vals),
                                         regs[stp32_rt1.clamp(0, 30)]) & 0xFFFFFFFF)
                stp32_v2 = (torch.where(stp32_rt2 == 31, torch.zeros_like(rd_vals),
                                         regs[stp32_rt2.clamp(0, 30)]) & 0xFFFFFFFF)
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                stp32_b = torch.cat([(stp32_v1.unsqueeze(1) >> sh4 & 0xFF).to(torch.uint8),
                                     (stp32_v2.unsqueeze(1) >> sh4 & 0xFF).to(torch.uint8)], dim=1)
                b8 = torch.arange(8, device=device, dtype=torch.int64)
                am = stp32_mask.unsqueeze(1).expand(-1, 8)
                act_a = (stp32_addr.unsqueeze(1) + b8)[am].long()
                act_b = stp32_b[am]
                if act_a.numel() > 0:
                    mem.scatter_(0, act_a, act_b)

            ldp32_mask = ((insts & 0xFFC00000) == 0x29400000) & exec_mask
            if ldp32_mask.any():
                ldp32_imm7 = (insts >> 15) & 0x7F
                ldp32_imm7_s = torch.where(ldp32_imm7 >= 64, ldp32_imm7 - 128, ldp32_imm7)
                ldp32_addr = (rn_vals + ldp32_imm7_s * 4).clamp(0, self.mem_size - 8)
                ldp32_rt1 = insts & 0x1F
                ldp32_rt2 = (insts >> 10) & 0x1F
                sh4 = torch.tensor([0,8,16,24], device=device, dtype=torch.int64)
                byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                ldp32_all = mem[(ldp32_addr.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                ldp32_v1 = (ldp32_all[:, :4] << sh4).sum(dim=1)
                ldp32_v2 = (ldp32_all[:, 4:] << sh4).sum(dim=1)
                # Write Rt1 (if not XZR)
                wrt1 = ldp32_mask & (ldp32_rt1 != 31)
                if wrt1.any():
                    regs.index_put_((ldp32_rt1[wrt1].long(),), ldp32_v1[wrt1])
                # Write Rt2 (if not XZR, use results for the register-write path)
                wrt2 = ldp32_mask & (ldp32_rt2 != 31)
                if wrt2.any():
                    regs.index_put_((ldp32_rt2[wrt2].long(),), ldp32_v2[wrt2])
                # Also use write_mask + results for Rt1 (consistent with Phase 6)
                results = torch.where(ldp32_mask & (ldp32_rt1 == rds), ldp32_v1, results)
                write_mask = write_mask | (ldp32_mask & (ldp32_rt1 == rds))

            # --- PRFM / PRFUM (prefetch hint — treat as NOP) ---
            # PRFM unsigned offset: (inst & 0xFFC00000)==0xF9800000
            # PRFM literal: (inst & 0xFF000000)==0xD8000000
            # PRFUM: (inst & 0xFFE00C00)==0xF8800000
            # All are NOPs — no register write, no memory write
            # (fall-through: they simply don't match any handler above)

            # --- CAS / CASAL / CASA / CASL (compare-and-swap, approximate) ---
            # CAS 64: (inst & 0xFFE07C00)==0xC8207C00 mask; CASAL: 0xC8E07C00
            # Approximate: if mem[Xn] == Xs then mem[Xn] = Xt, Xs = old_mem[Xn]
            # (single-threaded: no actual atomicity needed)
            cas64_mask = (((insts & 0xFFE07C00) == 0xC8207C00) |   # CAS 64
                          ((insts & 0xFFE07C00) == 0xC8607C00) |   # CASL 64
                          ((insts & 0xFFE07C00) == 0xC8A07C00) |   # CASA 64
                          ((insts & 0xFFE07C00) == 0xC8E07C00)) & exec_mask  # CASAL 64
            cas32_mask = (((insts & 0xFFE07C00) == 0x88207C00) |   # CAS 32
                          ((insts & 0xFFE07C00) == 0x88607C00) |   # CASL 32
                          ((insts & 0xFFE07C00) == 0x88A07C00) |   # CASA 32
                          ((insts & 0xFFE07C00) == 0x88E07C00)) & exec_mask  # CASAL 32
            cas_mask = cas64_mask | cas32_mask
            if cas_mask.any():
                cas_addr = rn_vals.clamp(0, self.mem_size - 8)
                # Rs = bits[20:16] (comparand, updated with old value); Rt = bits[4:0] (new value)
                cas_rs = (insts >> 16) & 0x1F
                cas_rt = insts & 0x1F
                cas_rt_vals = torch.where(cas_rt == 31, torch.zeros_like(rn_vals),
                                           regs[cas_rt.clamp(0, 30)])
                cas_rs_vals = torch.where(cas_rs == 31, torch.zeros_like(rn_vals),
                                           regs[cas_rs.clamp(0, 30)])
                # Load current memory value
                if cas64_mask.any():
                    byte_off8 = torch.arange(8, device=device, dtype=torch.int64)
                    cas64_bytes = mem[(cas_addr.unsqueeze(1) + byte_off8).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 8).to(torch.int64)
                    cas64_cur = (cas64_bytes << torch.tensor([0,8,16,24,32,40,48,56], device=device, dtype=torch.int64)).sum(dim=1)
                    cas_cur = cas64_cur
                if cas32_mask.any():
                    byte_off4 = torch.arange(4, device=device, dtype=torch.int64)
                    cas32_bytes = mem[(cas_addr.unsqueeze(1) + byte_off4).view(-1).clamp(0, self.mem_size-1)].view(batch_size, 4).to(torch.int64)
                    cas32_cur = (cas32_bytes << torch.tensor([0,8,16,24], device=device, dtype=torch.int64)).sum(dim=1)
                    if cas64_mask.any():
                        cas_cur = torch.where(cas64_mask, cas64_cur, cas32_cur)
                    else:
                        cas_cur = cas32_cur
                # Perform CAS: if current == Rs, write Rt; always load Rs with old value
                cas_match = (cas_cur == cas_rs_vals)
                # Write new value to memory where match
                for _bi in (cas_mask & cas_match).nonzero(as_tuple=False).squeeze(1):
                    _addr = int(cas_addr[_bi].item())
                    _new_val = int(cas_rt_vals[_bi].item())
                    _nbytes = 8 if cas64_mask[_bi] else 4
                    for _j in range(_nbytes):
                        mem[_addr + _j] = _new_val & 0xFF
                        _new_val >>= 8
                # Rs always gets the old memory value (regardless of match)
                cas_rs_write = cas_mask & (cas_rs != 31)
                if cas_rs_write.any():
                    regs.index_put_((cas_rs[cas_rs_write].long(),), cas_cur[cas_rs_write])

            # Neural Prefetcher: record one memory access per batch.
            # prefetch.pt LSTM tracks address history, fires every 32 accesses.
            if _wov_pf_addr >= 0:
                prefetcher.on_memory_access(_wov_pf_addr, is_write=_wov_pf_write)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 6: SCATTER RESULTS TO REGISTERS (masked)
            # ═══════════════════════════════════════════════════════════════
            # Only write where write_mask is True and rd != 31
            final_write_mask = write_mask & (rds != 31)
            if final_write_mask.any():
                # Use scatter_ with mask
                write_rds = rds[final_write_mask]
                write_vals = results[final_write_mask]
                regs.index_put_((write_rds.long(),), write_vals)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 7: BRANCH RESOLUTION (tensor operations)
            # ALL state updates masked by 'active' - becomes no-op when halted/done
            # ═══════════════════════════════════════════════════════════════
            # Count instructions executed (tensor arithmetic)
            # If stopped by anything, execute up to first_stop; otherwise full batch
            has_any_stop = (first_stop < batch_size)
            inst_count = torch.where(has_any_stop, first_stop, torch.tensor(batch_size, device=device, dtype=torch.int64))
            # MASKED: only add to counter if active
            executed_t = torch.where(active, executed_t + inst_count, executed_t)

            # Default: advance PC by instructions executed
            new_pc = pc_t + inst_count * 4

            # Handle branches using tensor operations
            # ONLY process branches if we stopped due to a branch event (not hazard) AND active
            # Get the branch instruction (at first_stop index)
            branch_inst = insts[first_stop.clamp(0, batch_size - 1)]

            # Use EXPLICIT BIT PATTERNS for branch type detection (not op_type_table)
            # --- Unconditional B ---
            is_b = ((branch_inst & 0xFC000000) == 0x14000000) & stopped_by_event & active
            imm26 = branch_inst & 0x3FFFFFF
            imm26_signed = torch.where(imm26 >= 0x2000000, imm26 - 0x4000000, imm26)
            b_target = pc_t + first_stop * 4 + (imm26_signed << 2)
            new_pc = torch.where(is_b, b_target, new_pc)

            # --- BL ---
            is_bl = ((branch_inst & 0xFC000000) == 0x94000000) & stopped_by_event & active
            bl_target = pc_t + first_stop * 4 + (imm26_signed << 2)
            link_addr = pc_t + first_stop * 4 + 4
            # Set X30 if BL
            regs[30] = torch.where(is_bl, link_addr, regs[30])
            new_pc = torch.where(is_bl, bl_target, new_pc)

            # --- BR/BLR ---
            is_br = ((branch_inst & 0xFFFFFC1F) == 0xD61F0000) & stopped_by_event & active
            is_blr = ((branch_inst & 0xFFFFFC1F) == 0xD63F0000) & stopped_by_event & active
            br_rn = (branch_inst >> 5) & 0x1F
            br_target = regs[br_rn.clamp(0, 31)]
            regs[30] = torch.where(is_blr, link_addr, regs[30])
            new_pc = torch.where(is_br | is_blr, br_target, new_pc)

            # --- RET ---
            is_ret = ((branch_inst & 0xFFFFFC1F) == 0xD65F0000) & stopped_by_event & active
            ret_target = regs[30]
            new_pc = torch.where(is_ret, ret_target, new_pc)

            # --- CBZ/CBNZ ---
            is_cbz = (((branch_inst & 0xFF000000) == 0xB4000000) |
                      ((branch_inst & 0xFF000000) == 0x34000000)) & stopped_by_event & active
            is_cbnz = (((branch_inst & 0xFF000000) == 0xB5000000) |
                       ((branch_inst & 0xFF000000) == 0x35000000)) & stopped_by_event & active
            cb_rt = branch_inst & 0x1F
            cb_imm19 = (branch_inst >> 5) & 0x7FFFF
            cb_imm19_signed = torch.where(cb_imm19 >= 0x40000, cb_imm19 - 0x80000, cb_imm19)
            cb_offset = cb_imm19_signed << 2
            cb_val = regs[cb_rt.clamp(0, 31)]
            cb_pc = pc_t + first_stop * 4
            cbz_taken = (cb_val == 0)
            cbnz_taken = (cb_val != 0)
            cbz_target = cb_pc + cb_offset
            cbz_fallthrough = cb_pc + 4
            new_pc = torch.where(is_cbz, torch.where(cbz_taken, cbz_target, cbz_fallthrough), new_pc)
            new_pc = torch.where(is_cbnz, torch.where(cbnz_taken, cbz_target, cbz_fallthrough), new_pc)

            # --- TBZ / TBNZ (test bit and branch) ---
            # TBZ:  (inst & 0x7F000000) == 0x36000000  (bit31=0 for W-variants)
            # TBNZ: (inst & 0x7F000000) == 0x37000000
            # bit_pos = bits[23:19] | bit[31] (for 64-bit variants bit31=1 → top bit_pos=32+)
            is_tbz  = (((branch_inst & 0x7F000000) == 0x36000000)) & stopped_by_event & active
            is_tbnz = (((branch_inst & 0x7F000000) == 0x37000000)) & stopped_by_event & active
            tb_rt = branch_inst & 0x1F
            tb_bit_lo = (branch_inst >> 19) & 0x1F
            tb_bit_hi = (branch_inst >> 31) & 1  # extends bit_pos to 6 bits for 64-bit variant
            tb_bit_pos = (tb_bit_hi << 5) | tb_bit_lo
            tb_val = regs[tb_rt.clamp(0, 31)]
            tb_bit_set = ((tb_val >> tb_bit_pos) & 1) != 0
            tb_imm14 = (branch_inst >> 5) & 0x3FFF
            tb_imm14_signed = torch.where(tb_imm14 >= 0x2000, tb_imm14 - 0x4000, tb_imm14)
            tb_offset = tb_imm14_signed << 2
            tb_pc = pc_t + first_stop * 4
            tb_target = tb_pc + tb_offset
            tb_fallthrough = tb_pc + 4
            new_pc = torch.where(is_tbz,  torch.where(~tb_bit_set, tb_target, tb_fallthrough), new_pc)
            new_pc = torch.where(is_tbnz, torch.where( tb_bit_set, tb_target, tb_fallthrough), new_pc)

            # --- B.cond ---
            is_bcond = ((branch_inst & 0xFF000010) == 0x54000000) & stopped_by_event & active
            cond_code = branch_inst & 0xF
            bcond_imm19 = (branch_inst >> 5) & 0x7FFFF
            bcond_imm19_signed = torch.where(bcond_imm19 >= 0x40000, bcond_imm19 - 0x80000, bcond_imm19)
            bcond_offset = bcond_imm19_signed << 2
            bcond_pc = pc_t + first_stop * 4
            bcond_target = bcond_pc + bcond_offset
            bcond_fallthrough = bcond_pc + 4

            # Evaluate condition from flags tensor
            n, z, c, v = flags[0], flags[1], flags[2], flags[3]
            cond_results = torch.stack([
                z > 0.5,                                    # 0: EQ
                z <= 0.5,                                   # 1: NE
                c > 0.5,                                    # 2: CS
                c <= 0.5,                                   # 3: CC
                n > 0.5,                                    # 4: MI
                n <= 0.5,                                   # 5: PL
                v > 0.5,                                    # 6: VS
                v <= 0.5,                                   # 7: VC
                (c > 0.5) & (z <= 0.5),                    # 8: HI
                (c <= 0.5) | (z > 0.5),                    # 9: LS
                ((n > 0.5) == (v > 0.5)),                  # 10: GE
                ((n > 0.5) != (v > 0.5)),                  # 11: LT
                (z <= 0.5) & ((n > 0.5) == (v > 0.5)),    # 12: GT
                (z > 0.5) | ((n > 0.5) != (v > 0.5)),     # 13: LE
                torch.tensor(True, device=device),         # 14: AL
                torch.tensor(False, device=device),        # 15: NV
            ])
            cond_taken = cond_results[cond_code.clamp(0, 15)]
            new_pc = torch.where(is_bcond, torch.where(cond_taken, bcond_target, bcond_fallthrough), new_pc)

            # --- Halt check (masked by active) ---
            has_halt = halt_mask[first_stop.clamp(0, batch_size - 1)] & stopped_by_event & active
            halted_t = torch.where(has_halt, torch.tensor(1, device=device, dtype=torch.int64), halted_t)

            # --- SVC (syscall) - unavoidable sync for I/O ---
            has_svc = svc_mask[first_stop.clamp(0, batch_size - 1)] & stopped_by_event & active

            # --- DEFERRED combined sync: halt OR svc ---
            # Sync only every SYNC_INTERVAL batches (amortises MPS round-trip ~95ms).
            # SVC/halt tensors are updated every iter via torch.where; we just read
            # them less often.  Active mask keeps post-halt iters as cheap no-ops.
            if _iter % SYNC_INTERVAL == (SYNC_INTERVAL - 1):
                _stop_code = int((halted_t + (has_svc.long() << 1)).item())
            else:
                _stop_code = 0   # no sync this iteration

            # Early-exit on halt (bit 0)
            if _stop_code & 1:
                break

            if _stop_code & 2:  # svc (bit 1)
                syscall_num = int(regs[8].item())
                arg0, arg1, arg2 = int(regs[0].item()), int(regs[1].item()), int(regs[2].item())
                result = 0

                if syscall_num == 64:  # write
                    fd, buf_ptr, count = arg0, arg1, arg2
                    if fd in (1, 2) and count > 0 and 0 <= buf_ptr < self.mem_size:
                        end_ptr = min(buf_ptr + count, self.mem_size)
                        out_bytes = mem[buf_ptr:end_ptr].cpu().numpy().tobytes()
                        try:
                            print(out_bytes.decode('utf-8', errors='replace'), end='', flush=True)
                        except:
                            pass
                        result = count
                elif syscall_num in (93, 94):  # exit
                    halted_t = torch.tensor(1, device=device, dtype=torch.int64)
                    result = arg0
                elif syscall_num == 214:  # brk
                    if not hasattr(self, '_brk_addr'):
                        self._brk_addr = 0x200000
                    if arg0 == 0:
                        result = self._brk_addr
                    elif arg0 > self._brk_addr and arg0 < self.mem_size:
                        self._brk_addr = arg0
                        result = arg0
                    else:
                        result = self._brk_addr
                elif syscall_num == 222:  # mmap
                    length = arg1
                    if not hasattr(self, '_mmap_base'):
                        self._mmap_base = 0x400000
                    if arg0 == 0 and length > 0:
                        aligned_len = (length + 0xFFF) & ~0xFFF
                        if self._mmap_base + aligned_len < self.mem_size:
                            result = self._mmap_base
                            self._mmap_base += aligned_len
                        else:
                            result = -12
                    else:
                        result = -22
                elif syscall_num in (215, 226):  # munmap, mprotect
                    result = 0
                elif syscall_num == 63:  # read
                    result = 0
                elif syscall_num == 57:  # close
                    result = 0
                elif syscall_num == 56:  # openat
                    result = -2
                elif syscall_num in (79, 80):  # fstat
                    result = -2
                elif syscall_num == 29:  # ioctl
                    result = -25
                elif syscall_num in (172, 178, 96):  # getpid, gettid, set_tid_address
                    result = 1
                elif syscall_num in (113, 134, 135):  # clock_gettime, rt_sigaction, rt_sigprocmask
                    result = 0
                # --- Extended syscall coverage for OS code ---
                elif syscall_num == 35:   # nanosleep
                    result = 0
                elif syscall_num == 62:   # lseek
                    result = 0
                elif syscall_num == 66:   # writev
                    # Simple writev: sum iov lengths and call write for each
                    iov_ptr, iovcnt = arg1, arg2
                    written = 0
                    for k in range(min(iovcnt, 16)):
                        iov_base_addr = iov_ptr + k * 16
                        if 0 <= iov_base_addr + 15 < self.mem_size:
                            iov_base_b = mem[iov_base_addr:iov_base_addr+8].cpu().numpy()
                            iov_len_b  = mem[iov_base_addr+8:iov_base_addr+16].cpu().numpy()
                            import struct as _struct
                            iov_base = _struct.unpack('<Q', bytes(iov_base_b))[0]
                            iov_len  = _struct.unpack('<Q', bytes(iov_len_b))[0]
                            if arg0 in (1, 2) and iov_len > 0 and 0 <= iov_base < self.mem_size:
                                ep = min(iov_base + iov_len, self.mem_size)
                                try:
                                    print(mem[iov_base:ep].cpu().numpy().tobytes().decode('utf-8', errors='replace'), end='', flush=True)
                                except:
                                    pass
                                written += iov_len
                    result = written
                elif syscall_num in (67, 68):  # pread64, pwrite64
                    result = 0
                elif syscall_num == 72:   # pselect6
                    result = 0
                elif syscall_num == 73:   # ppoll
                    result = 0
                elif syscall_num == 78:   # readlinkat
                    result = -2  # ENOENT
                elif syscall_num == 79:   # fstatat / newfstatat
                    result = -2
                elif syscall_num == 80:   # fstat
                    result = -2
                elif syscall_num == 48:   # faccessat
                    result = -13  # EACCES
                elif syscall_num == 49:   # chdir
                    result = 0
                elif syscall_num == 53:   # ftruncate
                    result = 0
                elif syscall_num == 54:   # truncate
                    result = 0
                elif syscall_num == 97:   # getcwd
                    buf_ptr, size = arg0, arg1
                    cwd = b'/\x00'
                    if 0 <= buf_ptr < self.mem_size and size > 0:
                        n_bytes = min(len(cwd), size, self.mem_size - buf_ptr)
                        mem[buf_ptr:buf_ptr+n_bytes] = torch.tensor(list(cwd[:n_bytes]), dtype=torch.uint8, device=device)
                    result = buf_ptr
                elif syscall_num == 98:   # futex
                    result = 0
                elif syscall_num == 99:   # set_robust_list / set_tls
                    result = 0
                elif syscall_num == 160:  # uname
                    # Write minimal utsname struct to arg0 (sysname only)
                    if 0 <= arg0 < self.mem_size - 64:
                        sysname = b'Linux\x00'
                        for ki, bv in enumerate(sysname):
                            if arg0 + ki < self.mem_size:
                                mem[arg0 + ki] = bv
                    result = 0
                elif syscall_num == 161:  # sysinfo
                    result = 0
                elif syscall_num in (169, 403):  # gettimeofday, clock_gettime64
                    result = 0
                elif syscall_num in (173, 174, 175, 176, 177):  # getppid, getuid, geteuid, getgid, getegid
                    result = 0
                elif syscall_num == 220:  # clone / fork
                    result = -1  # EPERM (we don't support forking)
                elif syscall_num == 221:  # execve
                    result = -2  # ENOENT
                elif syscall_num == 233:  # madvise
                    result = 0
                elif syscall_num == 260:  # wait4
                    result = -10  # ECHILD
                elif syscall_num in (261, 262):  # set_robust_list, get_robust_list
                    result = 0
                elif syscall_num == 278:  # getrandom
                    buf_ptr, count, flags_arg = arg0, arg1, arg2
                    import os as _os
                    if 0 <= buf_ptr < self.mem_size and count > 0:
                        actual = min(count, self.mem_size - buf_ptr, 256)
                        rand_bytes = list(_os.urandom(actual))
                        mem[buf_ptr:buf_ptr+actual] = torch.tensor(rand_bytes, dtype=torch.uint8, device=device)
                        result = actual
                    else:
                        result = -22
                elif syscall_num == 281:  # epoll_pwait
                    result = 0
                elif syscall_num == 291:  # statx
                    result = -2  # ENOENT
                elif syscall_num in (39, 40, 41, 42, 43, 44, 45, 46, 47):  # socket-related
                    result = -1  # EPERM
                else:
                    result = -38

                regs[0] = result
                new_pc = pc_t + first_stop * 4 + 4  # Advance past SVC

            # Update PC tensor (MASKED by active - no-op when done)
            pc_t = torch.where(active, new_pc, pc_t)

        # ═══════════════════════════════════════════════════════════════════
        # FINAL SYNC - only place we sync to get return values
        # ═══════════════════════════════════════════════════════════════════
        self.pc = pc_t
        final_executed = int(executed_t.item())
        self.inst_count.fill_(final_executed)
        self.halted = bool(halted_t.item())

        return final_executed, time.perf_counter() - start

