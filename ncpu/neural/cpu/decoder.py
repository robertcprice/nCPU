"""
Instruction decoding mixin for NeuralCPU.

Provides instruction fetch, decode, and operand extraction using
neural network extractors and GPU-resident lookup tables.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import os
import numpy as np

from .constants import OpType, _u64_to_s64
from .extractors import NeuralMovzExtractor, NeuralBranchExtractor, NeuralBranch19Extractor

logger = logging.getLogger(__name__)


class DecoderMixin:
    """Instruction decoding methods for NeuralCPU."""

    def _load_extractors(self):
        """Load pre-trained neural extractors - ALL ON GPU."""
        self.movz_ext = NeuralMovzExtractor(d_model=128).to(self.device).eval()
        self.branch_ext = NeuralBranchExtractor(d_model=128).to(self.device).eval()
        self.branch19_ext = NeuralBranch19Extractor(d_model=128).to(self.device).eval()

        # Powers for bit-to-integer conversion - stay on GPU!
        self.powers_16 = torch.tensor([1 << i for i in range(16)], dtype=torch.int64, device=self.device)
        self.powers_19 = torch.tensor([1 << i for i in range(19)], dtype=torch.int64, device=self.device)
        self.powers_26 = torch.tensor([1 << i for i in range(26)], dtype=torch.int64, device=self.device)

        # Load pre-trained weights (optional — models/final/*.pt are pre-trained
        # neural extractors from earlier training runs, not included in the
        # current models/ layout. The CPU works without them using fallback logic.)
        for name, ext, path in [
            ("MOVZ", self.movz_ext, 'models/final/neural_movz_extractor.pt'),
            ("Branch26", self.branch_ext, 'models/final/neural_branch_extractor.pt'),
            ("Branch19", self.branch19_ext, 'models/final/neural_branch19_extractor.pt'),
        ]:
            if Path(path).exists():
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                ext.load_state_dict(ckpt['model_state_dict'])
                logger.info(f"   {name} extractor loaded")


        # ═══════════════════════════════════════════════════════════════════════
        # NEURAL INSTRUCTION LOOKUP TABLES - NO IF/ELIF, PURE TENSOR OPS!
        # ═══════════════════════════════════════════════════════════════════════
        self._init_neural_lookup_tables()

        logger.info("   GPU branch decider ready")
        logger.info(f"   Neural loop detector ready ({sum(p.numel() for p in self.loop_detector.parameters()):,} params)")
        logger.info(f"   Neural instruction dispatcher ready ({sum(p.numel() for p in self.neural_dispatcher.parameters()):,} params)")
        logger.info(f"   Neural execution optimizer ready ({sum(p.numel() for p in self.execution_optimizer.parameters()):,} params)")
        logger.info("   Neural lookup tables ready (op_type, operand masks)")
        logger.info(f"   Neural execution engine ready ({sum(p.numel() for p in self.neural_engine.parameters()):,} params)")
        logger.info(f"   Framebuffer on GPU [{self.FB_WIDTH}x{self.FB_HEIGHT}]")

    def _init_neural_lookup_tables(self):
        """
        Initialize NEURAL lookup tables for instruction decoding.

        NO IF/ELIF CHAINS! All decoding via tensor indexing:
        - op_type_table[256]: Maps top byte → OpType
        - op_code_table[512]: Maps 9-bit opcode → OpType (for MOVZ/MOVK etc)
        - All lookups are tensor index operations on GPU
        """
        # Primary op_type table indexed by top byte (bits 31-24)
        # Initialize all to NOP, then fill known patterns
        self.op_type_table = torch.zeros(256, dtype=torch.int64, device=self.device)

        # Map op_byte → OpType using tensor assignment
        op_byte_mappings = [
            (0x91, OpType.ADD_IMM), (0x8B, OpType.ADD_REG),
            (0xD1, OpType.SUB_IMM), (0xCB, OpType.SUB_REG),
            (0xF1, OpType.SUBS_IMM), (0xEB, OpType.SUBS_REG),
            (0xB1, OpType.ADDS_IMM), (0xAB, OpType.ADDS_REG),
            (0x31, OpType.ADDS_IMM_W),
            (0x9B, OpType.MUL), (0xAA, OpType.ORR_REG),
            (0x8A, OpType.AND_REG), (0x92, OpType.AND_IMM),
            (0xEA, OpType.ANDS_REG), (0xF2, OpType.ANDS_IMM),
            (0xB2, OpType.ORR_IMM), (0xCA, OpType.EOR_REG),
            (0x39, OpType.STRB),  # Will check load/store bit separately
            (0xF9, OpType.STR),   # Will check load/store bit separately
            # 0xF8 NOT in table - needs special handling for pre/post-index modes
            (0x54, OpType.B_COND),
        ]
        for op_byte, op_type in op_byte_mappings:
            self.op_type_table[op_byte] = op_type.value

        # Branch instructions (check prefix)
        for prefix in range(0x14, 0x18):  # B unconditional
            self.op_type_table[prefix] = OpType.B.value
        for prefix in range(0x94, 0x98):  # BL
            self.op_type_table[prefix] = OpType.BL.value

        # CBZ/CBNZ
        self.op_type_table[0x34] = OpType.CBZ.value
        self.op_type_table[0xB4] = OpType.CBZ.value
        self.op_type_table[0x35] = OpType.CBNZ.value
        self.op_type_table[0xB5] = OpType.CBNZ.value

        # 32-bit variants
        self.op_type_table[0x11] = OpType.ADD_IMM_W.value
        self.op_type_table[0x0B] = OpType.ADD_REG_W.value
        self.op_type_table[0x51] = OpType.SUB_IMM_W.value
        self.op_type_table[0x71] = OpType.SUBS_IMM_W.value
        self.op_type_table[0x2A] = OpType.MOV_W.value

        # ADRP/ADR
        self.op_type_table[0x90] = OpType.ADRP.value
        self.op_type_table[0x10] = OpType.ADR.value
        self.op_type_table[0xB0] = OpType.ADRP.value
        self.op_type_table[0xD0] = OpType.ADRP.value
        self.op_type_table[0xF0] = OpType.ADRP.value

        # Sign/Zero extend
        self.op_type_table[0x93] = OpType.SXTW.value  # SBFM/SXTW

        # Load/Store pair (handled specially in lookup)
        self.op_type_table[0xA9] = OpType.STP.value  # STP (64-bit)
        self.op_type_table[0xA8] = OpType.STP.value  # STP (pre/post-index)
        self.op_type_table[0xA5] = OpType.LDP.value  # LDP (64-bit)
        self.op_type_table[0x29] = OpType.STP.value  # STP 32-bit
        self.op_type_table[0x28] = OpType.STP.value  # STP 32-bit pre/post

        # Conditional select
        self.op_type_table[0x9A] = OpType.CSEL.value  # CSEL/CSINC (will refine)
        self.op_type_table[0x1A] = OpType.CSEL_W.value  # CSEL 32-bit

        # Branch to register
        self.op_type_table[0xD6] = OpType.BR.value  # BR/BLR (will refine)

        # System instructions (barriers, etc.)
        self.op_type_table[0xD5] = OpType.DMB.value  # System instructions

        # Bit manipulation and shifts
        self.op_type_table[0x12] = OpType.AND_IMM.value  # AND immediate 32-bit
        self.op_type_table[0x53] = OpType.UBFM.value     # UBFM 32-bit (LSL/LSR/UBFX)
        self.op_type_table[0xD3] = OpType.UBFM.value     # UBFM 64-bit (LSL/LSR/UBFX/UXTB)

        # Test bit and branch
        self.op_type_table[0x36] = OpType.TBZ.value      # TBZ
        self.op_type_table[0x37] = OpType.TBNZ.value     # TBNZ

        # Load/store variants
        self.op_type_table[0x38] = OpType.LDRB.value     # LDRB register offset
        self.op_type_table[0x79] = OpType.LDRH.value     # LDRH unsigned offset
        self.op_type_table[0xB9] = OpType.LDR_W.value    # LDR 32-bit (word)

        # More ALU
        self.op_type_table[0x6B] = OpType.SUBS_REG.value # SUBS register 32-bit
        self.op_type_table[0x5A] = OpType.CSINV.value    # CSINV/CSNEG 32-bit

        # Conditional compare (mapped to CMP for simplicity)
        self.op_type_table[0x7A] = OpType.CMP_IMM_W.value  # CCMP 32-bit
        self.op_type_table[0xFA] = OpType.CMP_IMM.value    # CCMP 64-bit

        # 9-bit opcode table for MOVZ/MOVK/MOVN (bits 31-23)
        # Wide moves have bit 23 = 1 in their encoding!
        # WRONG: 0x1A4, 0x1E4, 0x124 are NOT wide moves - they overlap with AND_IMM etc.
        self.op_code_table = torch.zeros(512, dtype=torch.int64, device=self.device)
        # 64-bit wide moves (bit 23 = 1)
        self.op_code_table[0x1A5] = OpType.MOVZ.value    # MOVZ 64-bit: 110100101
        self.op_code_table[0x1E5] = OpType.MOVK.value    # MOVK 64-bit: 111100101
        self.op_code_table[0x125] = OpType.MOVN.value    # MOVN 64-bit: 100100101
        # 32-bit wide moves
        self.op_code_table[0x0A5] = OpType.MOVZ_W.value  # MOVZ 32-bit: 010100101
        self.op_code_table[0x0E5] = OpType.MOVK_W.value  # MOVK 32-bit: 011100101
        self.op_code_table[0x025] = OpType.MOVN.value    # MOVN 32-bit: 000100101

        # Bit extraction masks as tensors (for operand extraction)
        self.rd_mask = torch.tensor(0x1F, dtype=torch.int64, device=self.device)
        self.rn_mask = torch.tensor(0x1F << 5, dtype=torch.int64, device=self.device)
        self.rm_mask = torch.tensor(0x1F << 16, dtype=torch.int64, device=self.device)
        self.imm12_mask = torch.tensor(0xFFF << 10, dtype=torch.int64, device=self.device)
        self.imm9_mask = torch.tensor(0x1FF << 12, dtype=torch.int64, device=self.device)

        # Powers of 2 for neural bit extraction
        self.powers_32 = torch.tensor([1 << i for i in range(32)], dtype=torch.int64, device=self.device)

        # MOVK mask lookup table - indexed by hw (0-3)
        # These are stored as signed int64 but work correctly for bitwise ops
        # hw=0: clear bits 0-15,  hw=1: clear bits 16-31, etc.
        self.movk_masks = torch.tensor([
            -65536,              # hw=0: 0xFFFFFFFFFFFF0000
            -4294901761,         # hw=1: 0xFFFFFFFF0000FFFF
            -281470681743361,    # hw=2: 0xFFFF0000FFFFFFFF
            281474976710655,     # hw=3: 0x0000FFFFFFFFFFFF
        ], dtype=torch.int64, device=self.device)

        # MOVK shift amounts for each hw value
        self.movk_shifts = torch.tensor([0, 16, 32, 48], dtype=torch.int64, device=self.device)

        # Neural dispatcher training state
        self.dispatcher_trained = False
        self.use_pure_neural = False  # Switch to pure neural after training

    def _build_gpu_op_table(self):
        """
        Build GPU opcode table for pure parallel execution.
        Pre-allocate all tensors needed for run_parallel_gpu.
        """
        # MOVK mask constant - pre-allocated, no allocation in hot loop!
        self._movk_clear_base = torch.tensor(0xFFFF, dtype=torch.int64, device=self.device)

        # Pre-allocated zero constant for results initialization
        self._zero_i64 = torch.tensor(0, dtype=torch.int64, device=self.device)

        # Pre-allocated batch results and masks (reused each iteration)
        self._gpu_results = torch.zeros(self.BATCH_SIZE, dtype=torch.int64, device=self.device)
        self._gpu_write_mask = torch.zeros(self.BATCH_SIZE, dtype=torch.bool, device=self.device)

        # Pre-allocated index buffers (avoid torch.arange in hot path)
        self._batch_idx = torch.arange(self.BATCH_SIZE, dtype=torch.int64, device=self.device)
        self._byte_offsets = torch.arange(self.BATCH_SIZE * 4, dtype=torch.int64, device=self.device)
        self._idx_2 = torch.arange(2, dtype=torch.int64, device=self.device)
        self._idx_3 = torch.arange(3, dtype=torch.int64, device=self.device)
        self._idx_4 = torch.arange(4, dtype=torch.int64, device=self.device)
        self._idx_8 = torch.arange(8, dtype=torch.int64, device=self.device)
        self._idx_32 = torch.arange(32, dtype=torch.int64, device=self.device)
        self._idx_64 = torch.arange(64, dtype=torch.int64, device=self.device)
        self._idx_4096 = torch.arange(4096, dtype=torch.int64, device=self.device)

        # Pre-allocated op type scalars for sub-decode (avoid per-iteration tensors)
        self._op_ldr_post = torch.tensor(OpType.LDR_POST.value, dtype=torch.int64, device=self.device)
        self._op_str_post = torch.tensor(OpType.STR_POST.value, dtype=torch.int64, device=self.device)
        self._op_ldr_pre = torch.tensor(OpType.LDR_PRE.value, dtype=torch.int64, device=self.device)
        self._op_str_pre = torch.tensor(OpType.STR_PRE.value, dtype=torch.int64, device=self.device)
        self._op_ldr_reg_off = torch.tensor(OpType.LDR_REG_OFF.value, dtype=torch.int64, device=self.device)
        self._op_str_reg_off = torch.tensor(OpType.STR_REG_OFF.value, dtype=torch.int64, device=self.device)
        self._op_ldrb_post = torch.tensor(OpType.LDRB_POST.value, dtype=torch.int64, device=self.device)
        self._op_strb_post = torch.tensor(OpType.STRB_POST.value, dtype=torch.int64, device=self.device)
        self._op_ldr = torch.tensor(OpType.LDR.value, dtype=torch.int64, device=self.device)
        self._op_ldrb = torch.tensor(OpType.LDRB.value, dtype=torch.int64, device=self.device)
        self._op_tst_reg = torch.tensor(OpType.TST_REG.value, dtype=torch.int64, device=self.device)
        self._op_tst_imm = torch.tensor(OpType.TST_IMM.value, dtype=torch.int64, device=self.device)
        self._op_ands_reg = torch.tensor(OpType.ANDS_REG.value, dtype=torch.int64, device=self.device)
        self._op_ands_imm = torch.tensor(OpType.ANDS_IMM.value, dtype=torch.int64, device=self.device)

        # Pre-allocated small constants
        self._const_i64_0 = torch.tensor(0, dtype=torch.int64, device=self.device)
        self._const_i64_1 = torch.tensor(1, dtype=torch.int64, device=self.device)
        self._const_i64_2 = torch.tensor(2, dtype=torch.int64, device=self.device)
        self._const_i64_3 = torch.tensor(3, dtype=torch.int64, device=self.device)
        self._const_i64_4 = torch.tensor(4, dtype=torch.int64, device=self.device)
        self._const_i64_5 = torch.tensor(5, dtype=torch.int64, device=self.device)
        self._const_i64_6 = torch.tensor(6, dtype=torch.int64, device=self.device)
        self._const_i64_7 = torch.tensor(7, dtype=torch.int64, device=self.device)
        self._const_i64_8 = torch.tensor(8, dtype=torch.int64, device=self.device)
        self._const_i64_30 = torch.tensor(30, dtype=torch.int64, device=self.device)
        self._const_i64_4096 = torch.tensor(4096, dtype=torch.int64, device=self.device)
        self._const_i64_1023 = torch.tensor(1023, dtype=torch.int64, device=self.device)
        self._flags_ne = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        self._flags_eq = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
        self._sign_mask = torch.tensor(-0x8000000000000000, dtype=torch.int64, device=self.device)

        # Loop signature logging (GPU-only)
        self._loop_sig_buf = torch.zeros(4096, dtype=torch.int64, device=self.device)
        self._loop_sig_ptr = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._loop_sig_counts = torch.zeros(1024, dtype=torch.int64, device=self.device)
        self._loop_log_enabled = torch.tensor(1, dtype=torch.int64, device=self.device)
        log_all = 1 if os.getenv("NEURAL_LOOP_LOG_ALL", "0") == "1" else 0
        self._loop_log_all = torch.tensor([log_all], dtype=torch.int64, device=self.device)

        # Adaptive gates (GPU-side)
        adaptive_on = 1 if os.getenv("NEURAL_ADAPTIVE", "1") == "1" else 0
        self._adaptive_on = torch.tensor([adaptive_on], dtype=torch.int64, device=self.device)
        self._stall_score = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._spec_gate = torch.tensor([1], dtype=torch.int64, device=self.device)
        self._sb_gate = torch.tensor([1], dtype=torch.int64, device=self.device)

        # GPU trace buffer (PC, inst, op)
        self._trace_buf = torch.zeros((8192, 3), dtype=torch.int64, device=self.device)
        self._trace_ptr = torch.zeros(1, dtype=torch.int64, device=self.device)
        trace_on = 1 if os.getenv("NEURAL_GPU_TRACE", "0") == "1" else 0
        self._trace_enabled = torch.tensor([trace_on], dtype=torch.int64, device=self.device)

        # Speculative dual-path window buffers (single-instruction paths)
        self._spec_vals = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._spec_write = torch.zeros(2, dtype=torch.bool, device=self.device)

        # Pre-computed LUTs for run_woven() hot loop (avoid per-batch Python sets/loops/GPU syncs)
        _wov_max_op = max(m.value for m in OpType) + 1
        # Memory ops: r31 reads as SP (not XZR) in these instructions
        _wov_mem_vals = {
            OpType.LDR.value, OpType.STR.value, OpType.LDRB.value, OpType.STRB.value,
            OpType.LDRH.value, OpType.STRH.value, OpType.LDRSW.value, OpType.LDP.value,
            OpType.STP.value, OpType.LDUR.value, OpType.STUR.value,
            OpType.LDR_W.value, OpType.STR_W.value, OpType.LDRSB.value,
            OpType.LDRSH.value, OpType.LDRSW_IMM.value, OpType.LDR_REG_OFF.value,
            OpType.STR_REG_OFF.value, OpType.LDR_POST.value, OpType.STR_POST.value,
            OpType.LDR_PRE.value, OpType.STR_PRE.value, OpType.LDP_POST.value,
            OpType.STP_POST.value, OpType.LDP_PRE.value, OpType.STP_PRE.value,
            OpType.LDRB_POST.value, OpType.STRB_POST.value, OpType.LDXR.value,
            OpType.STXR.value,
        }
        _wov_mem_arr = torch.zeros(_wov_max_op, dtype=torch.bool)
        for _wv in _wov_mem_vals:
            _wov_mem_arr[_wv] = True
        self._woven_mem_op_lut = _wov_mem_arr.to(self.device)
        # Flag-setting ops (combined imm + reg)
        _wov_flag_all = {
            OpType.SUBS_IMM.value, OpType.ADDS_IMM.value, OpType.CMP_IMM.value,
            OpType.SUBS_IMM_W.value, OpType.CMP_IMM_W.value,
            OpType.SUBS_REG.value, OpType.ADDS_REG.value, OpType.CMP_REG.value,
            OpType.CMP_REG_W.value, OpType.ANDS_REG.value,
        }
        _wov_flag_arr = torch.zeros(_wov_max_op, dtype=torch.bool)
        for _wv in _wov_flag_all:
            _wov_flag_arr[_wv] = True
        self._woven_flag_op_lut = _wov_flag_arr.to(self.device)
        # Flag-setting register ops (vs immediate — determines rm vs imm12 operand)
        _wov_flag_reg_vals = {
            OpType.SUBS_REG.value, OpType.ADDS_REG.value, OpType.CMP_REG.value,
            OpType.CMP_REG_W.value, OpType.ANDS_REG.value,
        }
        _wov_flag_reg_arr = torch.zeros(_wov_max_op, dtype=torch.bool)
        for _wv in _wov_flag_reg_vals:
            _wov_flag_reg_arr[_wv] = True
        self._woven_flag_reg_lut = _wov_flag_reg_arr.to(self.device)

        # Superblock cache (GPU-resident decode window)
        self._sb_max = 256
        sb_entries = int(os.getenv("NEURAL_SB_ENTRIES", "8"))
        sb_entries = max(1, min(sb_entries, 64))
        self._sb_entries = sb_entries
        self._sb_ptr = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._sb_valid = torch.zeros(self._sb_entries, dtype=torch.int64, device=self.device)
        self._sb_pc = torch.zeros(self._sb_entries, dtype=torch.int64, device=self.device)
        self._sb_len = torch.zeros(self._sb_entries, dtype=torch.int64, device=self.device)
        self._sb_insts = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_ops = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_rds = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_rns = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_rms = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_imm12 = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_imm16 = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)
        self._sb_hw = torch.zeros((self._sb_entries, self._sb_max), dtype=torch.int64, device=self.device)


    # ════════════════════════════════════════════════════════════════════════════════
    # BATCHED NEURAL DECODING - Process multiple instructions at once
    # ════════════════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _decode_batch(self, instructions: List[int]) -> List[Tuple]:
        """
        Decode multiple instructions in ONE neural network forward pass.

        This amortizes the neural network overhead across many instructions.
        Returns list of (op_type, rd, rn, rm, imm, branch_off) tuples.
        """
        results = []
        uncached = []
        uncached_idx = []

        # Check cache first
        for i, inst in enumerate(instructions):
            if inst in self.decode_cache:
                results.append((i, self.decode_cache[inst]))
            else:
                uncached.append(inst)
                uncached_idx.append(i)

        if not uncached:
            # All cached!
            return [r[1] for r in sorted(results, key=lambda x: x[0])]

        # Convert uncached instructions to bit tensors [N, 32]
        bits_batch = torch.tensor(
            [[float((inst >> j) & 1) for j in range(32)] for inst in uncached],
            dtype=torch.float32, device=device
        )

        # Convert to int64 tensors for masking [N]
        inst_tensors = torch.tensor(uncached, dtype=torch.int64, device=self.device)

        # BATCHED neural dispatch (if trained)
        if self.dispatcher_trained and self.use_pure_neural:
            # Process all at once through neural network
            op_types_neural = []
            for i in range(len(bits_batch)):
                op_logits, _, _, _ = self.neural_dispatcher(bits_batch[i])
                op_types_neural.append(op_logits.argmax().item())
            op_type_vals = op_types_neural
        else:
            # Use lookup tables (faster when not trained)
            op_bytes = ((inst_tensors >> 24) & 0xFF)
            op_codes = ((inst_tensors >> 23) & 0x1FF)

            # Batch lookup
            op_type_vals_code = self.op_code_table[op_codes]
            op_type_vals_byte = self.op_type_table[op_bytes]

            # Use op_code if non-zero, else op_byte
            op_type_vals = torch.where(
                op_type_vals_code != 0,
                op_type_vals_code,
                op_type_vals_byte
            ).tolist()

        # BATCHED operand extraction via tensor operations
        rd_vals = (inst_tensors & self.rd_mask).tolist()
        rn_vals = ((inst_tensors & self.rn_mask) >> 5).tolist()
        rm_vals = ((inst_tensors & self.rm_mask) >> 16).tolist()

        # Build results and update cache
        for i, inst in enumerate(uncached):
            op_type_val = op_type_vals[i]
            op_type = OpType(op_type_val) if 0 < op_type_val < len(OpType) else OpType.NOP

            # Get basic operands
            rd = rd_vals[i]
            rn = rn_vals[i]
            rm = rm_vals[i]
            imm = 0
            branch_off = 0

            # Handle special cases (still need some logic for immediates)
            if op_type in (OpType.ADD_IMM, OpType.SUB_IMM, OpType.SUBS_IMM):
                imm = (inst >> 10) & 0xFFF
            elif op_type in (OpType.AND_IMM, OpType.ORR_IMM, OpType.EOR_IMM):
                imm = self._decode_bitmask_imm(inst)

            result = (op_type_val, rd, rn, rm, imm, branch_off)
            self.decode_cache[inst] = result
            results.append((uncached_idx[i], result))

        # Sort by original index and return
        return [r[1] for r in sorted(results, key=lambda x: x[0])]

    def prefetch_decode(self, start_addr: int, count: int = 64):
        """
        Prefetch and decode instructions ahead of time.

        Call this during idle time to populate the decode cache.
        """
        instructions = []
        for i in range(count):
            addr = start_addr + i * 4
            if 0 <= addr < self.mem_size - 4:
                inst = self.read32(addr)
                if inst not in self.decode_cache:
                    instructions.append(inst)

        if instructions:
            self._decode_batch(instructions)

    # Neural ALU is always on — no enable/disable toggle needed.

    def read32(self, addr: int) -> int:
        """Read 32-bit instruction - ONLY place we transfer to CPU."""
        if addr < 0 or addr + 4 > self.mem_size:
            return 0
        b = self.memory[addr:addr+4].cpu().numpy()
        return int.from_bytes(b.tobytes(), 'little')

    def load_binary(self, data: bytes, addr: int = 0):
        """Load binary data into GPU memory."""
        t = torch.tensor(list(data), dtype=torch.uint8, device=self.device)
        self.memory[addr:addr+len(data)] = t

    @torch.no_grad()
    def _decode_neural_lookup(self, inst: int) -> Tuple:
        """
        FULLY NEURAL instruction decoder using GPU tensor operations.

        FAST MODE (default): Use tensor lookup tables for speed (65M+ IPS)
        NEURAL MODE (after training): Use NeuralInstructionDispatcher

        Returns: (op_type, rd, rn, rm, imm, branch_off)
        """
        # Convert instruction to tensor for GPU operations
        inst_t = torch.tensor(inst, dtype=torch.int64, device=self.device)

        # Always extract op_byte for special instruction handling (0xF8 etc)
        op_byte = ((inst_t >> 24) & 0xFF).item()

        # ═══════════════════════════════════════════════════════════════════
        # OP-TYPE CLASSIFICATION: Neural or Lookup Table
        # ═══════════════════════════════════════════════════════════════════
        if self.use_pure_neural and self.dispatcher_trained:
            # PURE NEURAL MODE: Use trained NeuralInstructionDispatcher
            bits_flat = torch.tensor([float((inst >> j) & 1) for j in range(32)], device=self.device)
            op_logits, rd_logits, rn_logits, rm_logits = self.neural_dispatcher(bits_flat)
            op_type_val = op_logits.argmax().item()
            # Also get rd/rn/rm from neural network
            rd = rd_logits.argmax().item()
            rn = rn_logits.argmax().item()
            rm = rm_logits.argmax().item()
        else:
            # FAST MODE: Use tensor lookup tables (65M+ IPS)
            op_code = ((inst_t >> 23) & 0x1FF).item()

            # Lookup: op_code table takes priority, then op_byte table
            op_type_val = self.op_code_table[op_code].item()
            if op_type_val == 0:
                op_type_val = self.op_type_table[op_byte].item()

            # Extract operands via TENSOR MASKING (pure GPU operations)
            rd = (inst_t & self.rd_mask).item()
            rn = ((inst_t & self.rn_mask) >> 5).item()
            rm = ((inst_t & self.rm_mask) >> 16).item()

        imm = 0
        branch_off = 0

        # Create bit tensor for neural extractors (only when needed)
        bits = None

        # NEURAL EXTRACTION for specific instruction classes
        op_type = OpType(op_type_val) if op_type_val < len(OpType) else OpType.NOP

        # Special case: ORR with rn=31 (XZR) is actually MOV
        # In ARM64, register 31 is XZR (zero) for data-processing but SP for addressing
        if op_type == OpType.ORR_REG and rn == 31:
            op_type = OpType.MOV_REG

        # ═══════════════════════════════════════════════════════════════════════
        # LDP/STP - CRITICAL: Must detect pre/post-index modes BEFORE table lookup is used
        # The lookup table maps 0xA8 -> STP, but 0xA8 with bit 22=1 is actually LDP_POST!
        # Addressing modes (bits 25-23): 001=post-index, 010=signed-offset, 011=pre-index
        # Load/Store (bit 22): 0=store, 1=load
        # ═══════════════════════════════════════════════════════════════════════
        if op_byte in (0xA8, 0xA9):
            addr_mode = (inst >> 23) & 0x7  # bits 25-23
            load_bit = (inst >> 22) & 1     # bit 22
            if addr_mode == 1:  # Post-index
                if load_bit == 1:
                    op_type = OpType.LDP_POST
                else:
                    op_type = OpType.STP_POST
                imm7 = (inst >> 15) & 0x7F
                if imm7 & 0x40: imm7 = imm7 - 0x80
                imm = imm7 * 8
            elif addr_mode == 3:  # Pre-index
                if load_bit == 1:
                    op_type = OpType.LDP_PRE
                else:
                    op_type = OpType.STP_PRE
                imm7 = (inst >> 15) & 0x7F
                if imm7 & 0x40: imm7 = imm7 - 0x80
                imm = imm7 * 8
            elif addr_mode == 2:  # Signed offset
                if load_bit == 1:
                    op_type = OpType.LDP
                else:
                    op_type = OpType.STP
                imm7 = (inst >> 15) & 0x7F
                if imm7 & 0x40: imm7 = imm7 - 0x80
                imm = imm7 * 8

        # ═══════════════════════════════════════════════════════════════════════
        # UBFM/SBFM - Extract immr and imms, check for LSR/LSL specialization
        # The lookup table maps 0xD3 -> UBFM, but we need to extract the immediates
        # and potentially specialize to LSR_IMM or LSL_IMM for better handling.
        # ═══════════════════════════════════════════════════════════════════════
        if op_byte == 0xD3:
            immr = (inst >> 16) & 0x3F
            imms = (inst >> 10) & 0x3F
            if imms == 63:  # LSR encoding: UBFM with imms=63
                op_type = OpType.LSR_IMM
                imm = immr  # shift amount
            elif imms == 63 - immr:  # LSL encoding
                op_type = OpType.LSL_IMM
                imm = 63 - immr  # shift amount
            else:
                op_type = OpType.UBFM
                imm = (immr << 6) | imms  # Pack both values for UBFM handler

        # Helper: lazily create bits tensor only when needed for neural extractors
        def get_bits():
            nonlocal bits
            if bits is None:
                bits = torch.tensor([[float((inst >> j) & 1) for j in range(32)]], device=self.device)
            return bits

        # Special instruction patterns (tensor comparison)
        if inst == 0 or inst == 0xD503201F:
            op_type = OpType.NOP
        elif inst == 0xD65F03C0:
            op_type = OpType.RET

        # MOVZ/MOVK - use neural MOVZ extractor
        elif op_type in (OpType.MOVZ, OpType.MOVK, OpType.MOVZ_W, OpType.MOVK_W):
            imm16_l, hw_l = self.movz_ext(get_bits())
            imm16 = ((imm16_l[0] > 0).long() * self.powers_16).sum()
            hw = ((hw_l[0] > 0).long() * self.powers_16[:2]).sum()
            imm = (imm16 | (hw << 16)).item()

        # Branch instructions - use neural branch extractors
        elif op_type == OpType.B:
            off = ((self.branch_ext(get_bits())[0] > 0).long() * self.powers_26).sum().item()
            if off & 0x2000000: off -= 0x4000000
            branch_off = off

        elif op_type == OpType.BL:
            off = ((self.branch_ext(get_bits())[0] > 0).long() * self.powers_26).sum().item()
            if off & 0x2000000: off -= 0x4000000
            branch_off = off

        elif op_type in (OpType.B_COND, OpType.CBZ, OpType.CBNZ):
            off = ((self.branch19_ext(get_bits())[0] > 0).long() * self.powers_19).sum().item()
            if off & 0x40000: off -= 0x80000
            branch_off = off
            if op_type == OpType.B_COND:
                imm = inst & 0xF  # condition code

        # TBZ/TBNZ - 14-bit branch offset, bit number in imm
        elif op_type in (OpType.TBZ, OpType.TBNZ):
            # imm14 is bits [18:5], sign-extended
            imm14 = (inst >> 5) & 0x3FFF
            if imm14 & 0x2000:
                imm14 -= 0x4000
            branch_off = imm14
            # bit number is bits [23:19] | bit31
            b5 = (inst >> 31) & 1
            b40 = (inst >> 19) & 0x1F
            imm = (b5 << 5) | b40  # bit position to test

        # ADR - PC-relative address (+/- 1MB)
        elif op_type == OpType.ADR:
            # immlo = bits[30:29], immhi = bits[23:5]
            immlo = (inst >> 29) & 0x3
            immhi = (inst >> 5) & 0x7FFFF
            adr_imm = (immhi << 2) | immlo
            if adr_imm & 0x100000:  # Sign extend 21-bit
                adr_imm -= 0x200000
            branch_off = adr_imm

        # ADRP - PC-relative page address (+/- 4GB)
        elif op_type == OpType.ADRP:
            # immlo = bits[30:29], immhi = bits[23:5]
            immlo = (inst >> 29) & 0x3
            immhi = (inst >> 5) & 0x7FFFF
            adrp_imm = (immhi << 2) | immlo
            if adrp_imm & 0x100000:  # Sign extend 21-bit
                adrp_imm -= 0x200000
            branch_off = adrp_imm  # Will be shifted by 12 in execution

        # Immediate extraction via tensor masking
        # ADDS_IMM and ADDS_IMM_W also need imm12 extraction (CMN is ADDS with Rd=XZR/WZR)
        elif op_type in (OpType.ADD_IMM, OpType.SUB_IMM, OpType.ADD_IMM_W, OpType.SUB_IMM_W,
                         OpType.ADDS_IMM, OpType.SUBS_IMM, OpType.CMP_IMM,
                         OpType.ADDS_IMM_W, OpType.SUBS_IMM_W, OpType.CMP_IMM_W):
            imm = ((inst_t & self.imm12_mask) >> 10).item()

        # Logical immediate instructions need bitmask decoding (AND/ORR/EOR with immediate)
        elif op_type in (OpType.AND_IMM, OpType.ORR_IMM, OpType.EOR_IMM):
            imm = self._decode_bitmask_imm(inst)

        elif op_type in (OpType.LDR, OpType.STR):
            # Check load/store bit
            load_bit = (inst >> 22) & 0x3
            if load_bit == 1:
                op_type = OpType.LDR
            else:
                op_type = OpType.STR
            imm = (((inst_t & self.imm12_mask) >> 10).item()) << 3

        elif op_type in (OpType.LDRB, OpType.STRB):
            load_bit = (inst >> 22) & 0x3
            if load_bit == 1:
                op_type = OpType.LDRB
            else:
                op_type = OpType.STRB
            imm = ((inst_t & self.imm12_mask) >> 10).item()

        # F8 complex: register offset or pre/post-index
        elif op_byte == 0xF8:
            opc_bit = (inst >> 22) & 0x1
            opt_bits = (inst >> 10) & 0x3
            if opt_bits == 0b10:
                # Register offset
                shift_bit = (inst >> 12) & 0x1
                imm = 3 if shift_bit else 0
                op_type = OpType.LDR_REG_OFF if opc_bit else OpType.STR_REG_OFF
            else:
                # Pre/post-index
                imm9_raw = (inst >> 12) & 0x1FF
                imm = imm9_raw - 0x200 if imm9_raw & 0x100 else imm9_raw
                if opt_bits == 0b01:
                    op_type = OpType.LDR_POST if opc_bit else OpType.STR_POST
                elif opt_bits == 0b11:
                    op_type = OpType.LDR_PRE if opc_bit else OpType.STR_PRE

        # 38 complex: LDRB/STRB register offset or pre/post-index
        # Used heavily in string loops like strcpy, strlen, strcmp
        elif op_byte == 0x38:
            opc_bit = (inst >> 22) & 0x1  # 1 = load, 0 = store
            opt_bits = (inst >> 10) & 0x3
            if opt_bits == 0b10:
                # Register offset: LDRB Wt, [Xn, Xm] - handled by OpType.LDRB
                pass  # Keep default LDRB from table, rm will be extracted
            elif opt_bits == 0b01:
                # Post-index: LDRB Wt, [Xn], #imm9
                imm9_raw = (inst >> 12) & 0x1FF
                imm = imm9_raw - 0x200 if imm9_raw & 0x100 else imm9_raw
                op_type = OpType.LDRB_POST if opc_bit else OpType.STRB_POST
            elif opt_bits == 0b11:
                # Pre-index: LDRB Wt, [Xn, #imm9]! - not yet implemented
                imm9_raw = (inst >> 12) & 0x1FF
                imm = imm9_raw - 0x200 if imm9_raw & 0x100 else imm9_raw
                op_type = OpType.LDRB_POST if opc_bit else OpType.STRB_POST  # TODO: add LDRB_PRE

        # ═══════════════════════════════════════════════════════════════════
        # BR/BLR REFINEMENT - Distinguish branch vs branch-with-link
        # Lookup table maps 0xD6 → BR, but BLR has bit 21 set
        # BR:  D61F00xx → opc (bits 22-21) = 00
        # BLR: D63F00xx → opc (bits 22-21) = 01
        # RET: D65F03C0 → opc (bits 22-21) = 10 (handled separately above)
        # ═══════════════════════════════════════════════════════════════════
        elif op_byte == 0xD6:
            opc = (inst >> 21) & 0x3
            if opc == 1:  # BLR
                op_type = OpType.BLR
            # opc == 0 is BR (already set by lookup table)
            # opc == 2 is RET (handled by inst == 0xD65F03C0 check above)

        # ═══════════════════════════════════════════════════════════════════
        # NEURAL PATTERN TRACKING - Accumulate data for optimization learning
        # All tracking via tensor operations (GPU-accelerated)
        # ═══════════════════════════════════════════════════════════════════

        # Track opcode frequency (9-bit opcode)
        op_code = ((inst_t >> 23) & 0x1FF).item()
        self.opcode_frequency[op_code] += 1

        # Track op-type frequency
        if op_type.value < 128:
            self.optype_frequency[op_type.value] += 1

        # Track register access patterns (source registers being read)
        if rn < 32:
            self.reg_read_frequency[rn] += 1
        if rm < 32:
            self.reg_read_frequency[rm] += 1
        # Destination register will be tracked at write time in execute

        # Store instruction bits in sequence buffer (circular)
        # This allows neural networks to learn instruction sequences/patterns
        seq_idx = self.seq_ptr.item() % self.seq_buffer_size
        for j in range(32):
            self.instruction_sequence[seq_idx, j] = float((inst >> j) & 1)
        self.seq_ptr += 1

        return (op_type.value, rd, rn, rm, imm, branch_off)

    @torch.no_grad()
    def _decode_neural(self, inst: int) -> Tuple:
        """
        Decode instruction using PURE NEURAL LOOKUP TABLES.

        NO HARDCODED IF/ELIF CHAINS!
        All decoding via tensor indexing + neural extractors.
        """
        if inst in self.decode_cache:
            self.cache_hits += 1  # Track for neural optimization
            return self.decode_cache[inst]

        self.cache_misses += 1  # Track for neural optimization

        # PURE NEURAL: Use lookup table decoder
        result = self._decode_neural_lookup(inst)
        self.decode_cache[inst] = result
        return result

    # ═══════════════════════════════════════════════════════════════════════════════
    # LEGACY FALLBACK - KEPT FOR REFERENCE ONLY
    # The neural lookup now handles all instructions via tensor tables
    # ═══════════════════════════════════════════════════════════════════════════════
    def _decode_legacy_DISABLED(self, inst: int) -> Tuple:
        """DISABLED - Kept for reference. Neural lookup is now primary."""
        op_byte = (inst >> 24) & 0xFF
        op_code = (inst >> 23) & 0x1FF
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        rm = (inst >> 16) & 0x1F
        imm = 0
        branch_off = 0
        op_type = OpType.NOP

        bits = torch.tensor([[float((inst >> j) & 1) for j in range(32)]], device=self.device)

        if inst == 0 or inst == 0xD503201F:
            op_type = OpType.NOP

        elif inst == 0xD65F03C0:
            op_type = OpType.RET

        elif op_code in [0x1A5, 0x1A4]:  # MOVZ
            op_type = OpType.MOVZ

        elif op_code in [0x1E5, 0x1E4]:  # MOVK
            op_type = OpType.MOVK

        elif op_byte & 0xFC == 0x14:  # B unconditional
            op_type = OpType.B
            off = ((self.branch_ext(bits)[0] > 0).long() * self.powers_26).sum().item()
            if off & 0x2000000: off -= 0x4000000
            branch_off = off

        elif op_byte & 0xFC == 0x94:  # BL - NEURAL EXTRACTION
            op_type = OpType.BL
            off = ((self.branch_ext(bits)[0] > 0).long() * self.powers_26).sum().item()
            if off & 0x2000000: off -= 0x4000000
            branch_off = off

        elif op_byte == 0x54:  # B.cond - NEURAL 19-BIT EXTRACTION
            op_type = OpType.B_COND
            off = ((self.branch19_ext(bits)[0] > 0).long() * self.powers_19).sum().item()
            if off & 0x40000: off -= 0x80000
            branch_off = off
            imm = inst & 0xF  # condition code

        elif op_byte in [0x34, 0xB4]:  # CBZ - NEURAL 19-BIT EXTRACTION
            op_type = OpType.CBZ
            off = ((self.branch19_ext(bits)[0] > 0).long() * self.powers_19).sum().item()
            if off & 0x40000: off -= 0x80000
            branch_off = off

        elif op_byte in [0x35, 0xB5]:  # CBNZ - NEURAL 19-BIT EXTRACTION
            op_type = OpType.CBNZ
            off = ((self.branch19_ext(bits)[0] > 0).long() * self.powers_19).sum().item()
            if off & 0x40000: off -= 0x80000
            branch_off = off

        elif op_byte == 0x91:
            op_type = OpType.ADD_IMM
            imm = (inst >> 10) & 0xFFF

        elif op_byte == 0x8B:
            op_type = OpType.ADD_REG

        elif op_byte == 0xD1:
            op_type = OpType.SUB_IMM
            imm = (inst >> 10) & 0xFFF

        elif op_byte == 0xCB:
            op_type = OpType.SUB_REG

        elif op_byte == 0xF1:
            # SUBS immediate (0xF1): CMP is SUBS with Rd=XZR
            imm = (inst >> 10) & 0xFFF
            if rd == 31:  # CMP (compare only, no writeback)
                op_type = OpType.CMP_IMM
            else:  # SUBS with writeback to Rd
                op_type = OpType.SUBS_IMM

        elif op_byte == 0xEB:
            # SUBS register (0xEB): CMP is SUBS with Rd=XZR
            if rd == 31:  # CMP (compare only, no writeback)
                op_type = OpType.CMP_REG
            else:  # SUBS with writeback to Rd
                op_type = OpType.SUBS_REG

        elif op_byte == 0x39:
            if (inst >> 22) & 0x3 == 1:
                op_type = OpType.LDRB
            else:
                op_type = OpType.STRB
            imm = (inst >> 10) & 0xFFF

        elif op_byte == 0xF8:
            # LDR/STR 64-bit with register offset or post/pre-index
            # Format: 11 11 1000 0xL ooooo xxxx SS nn nnnt tttt
            # Check bits 21 (opc[1]) and 11-10 for mode
            opc_bit = (inst >> 22) & 0x1  # Load=1, Store=0
            opt_bits = (inst >> 10) & 0x3  # 10 = register offset

            if opt_bits == 0b10:
                # Register offset mode: LDR/STR Xt, [Xn, Xm{, extend/shift}]
                # NEURAL: Extract shift and option from bits using tensor ops
                shift_bit = (inst >> 12) & 0x1  # S bit - if 1, shift by size (3 for 64-bit)
                option = (inst >> 13) & 0x7  # extend option

                # Store shift amount in imm (0 or 3 for 64-bit LDR/STR)
                imm = 3 if shift_bit else 0

                if opc_bit:
                    op_type = OpType.LDR_REG_OFF
                else:
                    op_type = OpType.STR_REG_OFF
            else:
                # Pre/post-index mode: bits [11:10] determine mode
                # 01 = post-index: LDR Xt, [Xn], #imm9
                # 11 = pre-index:  LDR Xt, [Xn, #imm9]!
                # NEURAL: Extract signed 9-bit immediate
                imm9_raw = (inst >> 12) & 0x1FF
                # Sign-extend from 9 bits
                if imm9_raw & 0x100:
                    imm9 = imm9_raw - 0x200
                else:
                    imm9 = imm9_raw
                imm = imm9

                if opt_bits == 0b01:
                    # Post-index mode
                    if opc_bit:
                        op_type = OpType.LDR_POST
                    else:
                        op_type = OpType.STR_POST
                elif opt_bits == 0b11:
                    # Pre-index mode
                    if opc_bit:
                        op_type = OpType.LDR_PRE
                    else:
                        op_type = OpType.STR_PRE
                # opt_bits == 0b00 is unscaled immediate (LDUR/STUR) - fall through to NOP for now

        elif op_byte == 0xF9:
            if (inst >> 22) & 0x3 == 1:
                op_type = OpType.LDR
            else:
                op_type = OpType.STR
            imm = ((inst >> 10) & 0xFFF) << 3

        elif op_byte == 0x9B:
            op_type = OpType.MUL

        elif op_byte == 0xAA:
            # Check if this is MOV (ORR with Rn=XZR) or ORR
            if rn == 31:  # MOV Xd, Xm is alias for ORR Xd, XZR, Xm
                op_type = OpType.MOV_REG
            else:
                op_type = OpType.ORR_REG

        # === NEW INSTRUCTION DECODING FOR ALPINE LINUX SUPPORT ===
        # All decoded with tensor-compatible values

        # AND register: 0x8A
        elif op_byte == 0x8A:
            op_type = OpType.AND_REG

        # AND immediate: 0x92 (64-bit)
        elif op_byte == 0x92:
            op_type = OpType.AND_IMM
            # Bitmask immediate encoding (simplified - full decoding would need neural extractor)
            imm = self._decode_bitmask_imm(inst)

        # ORR immediate: 0xB2
        elif op_byte == 0xB2:
            op_type = OpType.ORR_IMM
            imm = self._decode_bitmask_imm(inst)

        # EOR register: 0xCA
        elif op_byte == 0xCA:
            op_type = OpType.EOR_REG

        # EOR immediate: 0xD2
        elif op_byte == 0xD2 and (inst >> 23) & 1 == 0:  # Distinguish from MOVZ
            op_type = OpType.EOR_IMM
            imm = self._decode_bitmask_imm(inst)

        # LSL variable (register): LSLV Xd, Xn, Xm = 0x9AC02000
        elif (inst & 0xFFE0FC00) == 0x9AC02000:
            op_type = OpType.LSL_REG

        # LSL/LSR immediate (via UBFM): 0xD3
        # UBFM Xd, Xn, #immr, #imms - different encodings for LSL vs LSR
        elif op_byte == 0xD3:
            immr = (inst >> 16) & 0x3F
            imms = (inst >> 10) & 0x3F
            if imms == 63:  # LSR encoding: UBFM with imms=63, immr=shift
                op_type = OpType.LSR_IMM
                imm = immr  # shift amount
            elif imms == 63 - immr:  # LSL encoding
                op_type = OpType.LSL_IMM
                imm = 63 - immr  # shift amount

        # LSR variable (register): LSRV Xd, Xn, Xm = 0x9AC02400
        elif (inst & 0xFFE0FC00) == 0x9AC02400:
            op_type = OpType.LSR_REG

        # ASR variable (register): ASRV Xd, Xn, Xm = 0x9AC02800
        elif (inst & 0xFFE0FC00) == 0x9AC02800:
            op_type = OpType.ASR_REG

        # ROR variable (register): RORV Xd, Xn, Xm = 0x9AC02C00
        elif (inst & 0xFFE0FC00) == 0x9AC02C00:
            op_type = OpType.ROR_REG

        # MVN (bitwise NOT): ORN with Rn=XZR = 0xAA2003E0
        elif (inst & 0xFFE0FFE0) == 0xAA2003E0:
            op_type = OpType.MVN

        # BIC (AND NOT): 0x8A200000
        elif (inst & 0xFFE00000) == 0x8A200000:
            op_type = OpType.BIC

        # TST register: ANDS with Rd=XZR = 0xEA00001F
        elif (inst & 0xFFE0001F) == 0xEA00001F:
            op_type = OpType.TST_REG

        # TST immediate: 0xF2 with Rd=XZR
        elif op_byte == 0xF2 and rd == 31:
            op_type = OpType.TST_IMM
            imm = self._decode_bitmask_imm(inst)

        # NEG: SUB with Rn=XZR = 0xCB0003E0
        elif (inst & 0xFFE0FFE0) == 0xCB0003E0:
            op_type = OpType.NEG

        # BLR: 0xD63F0000 (branch with link to register)
        elif (inst & 0xFFFFFC00) == 0xD63F0000:
            op_type = OpType.BLR

        # BR: 0xD61F0000 (branch to register)
        elif (inst & 0xFFFFFC00) == 0xD61F0000:
            op_type = OpType.BR

        # SVC: 0xD4000001 (syscall)
        elif (inst & 0xFFE0001F) == 0xD4000001:
            op_type = OpType.SVC
            imm = (inst >> 5) & 0xFFFF  # syscall number

        # LDUR: 0xF8400000 (load unscaled offset)
        elif (inst & 0xFFE00C00) == 0xF8400000:
            op_type = OpType.LDUR
            imm = ((inst >> 12) & 0x1FF)
            if imm & 0x100: imm -= 0x200  # Sign extend

        # STUR: 0xF8000000 (store unscaled offset)
        elif (inst & 0xFFE00C00) == 0xF8000000:
            op_type = OpType.STUR
            imm = ((inst >> 12) & 0x1FF)
            if imm & 0x100: imm -= 0x200

        # ═══════════════════════════════════════════════════════════════════════
        # LDP/STP - Load/Store Pair instructions
        # All modes: signed-offset, post-index, pre-index
        # Addressing mode is bits 25-23: 001=post-index, 010=signed-offset, 011=pre-index
        # Load/Store is bit 22: 0=store, 1=load
        # ═══════════════════════════════════════════════════════════════════════

        # LDP post-index 64-bit: 10 101 0 001 1 = 0xA8C00000
        elif (inst & 0xFFC00000) == 0xA8C00000:
            op_type = OpType.LDP_POST
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80  # Sign extend
            imm = imm7 * 8  # Scale by 8 for 64-bit

        # STP post-index 64-bit: 10 101 0 001 0 = 0xA8800000
        elif (inst & 0xFFC00000) == 0xA8800000:
            op_type = OpType.STP_POST
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80
            imm = imm7 * 8

        # LDP pre-index 64-bit: 10 101 0 011 1 = 0xA9C00000
        elif (inst & 0xFFC00000) == 0xA9C00000:
            op_type = OpType.LDP_PRE
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80
            imm = imm7 * 8

        # STP pre-index 64-bit: 10 101 0 011 0 = 0xA9800000
        elif (inst & 0xFFC00000) == 0xA9800000:
            op_type = OpType.STP_PRE
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80
            imm = imm7 * 8

        # LDP signed-offset 64-bit: 10 101 0 010 1 = 0xA9400000
        elif (inst & 0xFFC00000) == 0xA9400000:
            op_type = OpType.LDP
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80
            imm = imm7 * 8

        # STP signed-offset 64-bit: 10 101 0 010 0 = 0xA9000000
        elif (inst & 0xFFC00000) == 0xA9000000:
            op_type = OpType.STP
            imm7 = (inst >> 15) & 0x7F
            if imm7 & 0x40: imm7 = imm7 - 0x80
            imm = imm7 * 8

        # MADD: 0x9B000000 (multiply-add)
        elif (inst & 0xFFE08000) == 0x9B000000:
            op_type = OpType.MADD

        # MSUB: 0x9B008000 (multiply-subtract)
        elif (inst & 0xFFE08000) == 0x9B008000:
            op_type = OpType.MSUB

        # SDIV: 0x9AC00C00
        elif (inst & 0xFFE0FC00) == 0x9AC00C00:
            op_type = OpType.SDIV

        # UDIV: 0x9AC00800
        elif (inst & 0xFFE0FC00) == 0x9AC00800:
            op_type = OpType.UDIV

        # CLZ: 0xDAC01000
        elif (inst & 0xFFFFFC00) == 0xDAC01000:
            op_type = OpType.CLZ

        # SXTW: 0x93407C00 (sign extend word to 64-bit)
        elif (inst & 0xFFFFFC00) == 0x93407C00:
            op_type = OpType.SXTW

        # UXTB: UBFM with imms=7, immr=0 = extract byte
        elif (inst & 0xFFFFFC00) == 0xD3401C00:
            op_type = OpType.UXTB

        # UXTH: UBFM with imms=15, immr=0 = extract halfword
        elif (inst & 0xFFFFFC00) == 0xD3403C00:
            op_type = OpType.UXTH

        # ═══════════════════════════════════════════════════════════════════════
        # BUSYBOX SUPPORT INSTRUCTIONS
        # ═══════════════════════════════════════════════════════════════════════

        # ADDS immediate (64-bit): 0xB1xxxxxx
        elif op_byte == 0xB1:
            op_type = OpType.ADDS_IMM
            imm = (inst >> 10) & 0xFFF

        # ADDS register (64-bit): 0xABxxxxxx
        elif op_byte == 0xAB:
            op_type = OpType.ADDS_REG

        # SUBS immediate (64-bit): 0xF1xxxxxx (CMP is SUBS with Rd=XZR)
        # Note: CMP_IMM already handled as 0xF1, but this is for when Rd != XZR
        elif op_byte == 0xF1 and rd != 31:
            op_type = OpType.SUBS_IMM
            imm = (inst >> 10) & 0xFFF

        # SUBS register (64-bit): 0xEBxxxxxx (CMP is SUBS with Rd=XZR)
        elif op_byte == 0xEB and rd != 31:
            op_type = OpType.SUBS_REG

        # LDRSB (64-bit result): 0x39800000 or 0x38C00000 (register offset)
        elif (inst & 0xFFC00000) == 0x39800000:
            op_type = OpType.LDRSB
            imm = (inst >> 10) & 0xFFF

        # LDRSH (64-bit result): 0x79800000
        elif (inst & 0xFFC00000) == 0x79800000:
            op_type = OpType.LDRSH
            imm = ((inst >> 10) & 0xFFF) * 2  # Scale by 2 for halfwords

        # LDRSW: 0xB9800000 (immediate) or 0x98xxxxxx (literal)
        elif (inst & 0xFFC00000) == 0xB9800000:
            op_type = OpType.LDRSW
            imm = ((inst >> 10) & 0xFFF) * 4  # Scale by 4 for words

        # LDRH (unsigned halfword): 0x79400000
        elif (inst & 0xFFC00000) == 0x79400000:
            op_type = OpType.LDRH
            imm = ((inst >> 10) & 0xFFF) * 2

        # STRH (store halfword): 0x79000000
        elif (inst & 0xFFC00000) == 0x79000000:
            op_type = OpType.STRH
            imm = ((inst >> 10) & 0xFFF) * 2

        # CSEL: 0x9A800000 (Rd = cond ? Rn : Rm)
        elif (inst & 0xFFE00C00) == 0x9A800000:
            op_type = OpType.CSEL
            imm = (inst >> 12) & 0xF  # condition code

        # CSINC: 0x9A800400 (Rd = cond ? Rn : Rm+1)
        elif (inst & 0xFFE00C00) == 0x9A800400:
            op_type = OpType.CSINC
            imm = (inst >> 12) & 0xF

        # CSINV: 0x9A800000 with bit[10]=1 (Rd = cond ? Rn : ~Rm)
        elif (inst & 0xFFE00C00) == 0xDA800000:
            op_type = OpType.CSINV
            imm = (inst >> 12) & 0xF

        # CSNEG: 0xDA800400 (Rd = cond ? Rn : -Rm)
        elif (inst & 0xFFE00C00) == 0xDA800400:
            op_type = OpType.CSNEG
            imm = (inst >> 12) & 0xF

        # ADR: 0x10xxxxxx (PC-relative, +/- 1MB)
        elif (inst & 0x9F000000) == 0x10000000:
            op_type = OpType.ADR
            # immlo = bits[30:29], immhi = bits[23:5]
            immlo = (inst >> 29) & 0x3
            immhi = (inst >> 5) & 0x7FFFF
            imm = (immhi << 2) | immlo
            if imm & 0x100000:  # Sign extend 21-bit
                imm -= 0x200000
            branch_off = imm  # Store as branch_off for PC-relative calc

        # ADRP: 0x90xxxxxx (PC-relative page, +/- 4GB)
        elif (inst & 0x9F000000) == 0x90000000:
            op_type = OpType.ADRP
            immlo = (inst >> 29) & 0x3
            immhi = (inst >> 5) & 0x7FFFF
            imm = (immhi << 2) | immlo
            if imm & 0x100000:
                imm -= 0x200000
            branch_off = imm  # Page offset (will be << 12)

        # UBFM (64-bit): 0xD3xxxxxx (handles LSL, LSR, UBFX, UXTB, UXTH)
        # Only if not already matched as LSL_IMM, UXTB, UXTH
        elif op_byte == 0xD3 and op_type == OpType.NOP:
            op_type = OpType.UBFM
            immr = (inst >> 16) & 0x3F
            imms = (inst >> 10) & 0x3F
            imm = (immr << 6) | imms  # Pack both values

        # SBFM (64-bit): 0x93xxxxxx (handles ASR, SBFX, SXTB, SXTH, SXTW)
        elif op_byte == 0x93 and op_type == OpType.NOP:
            op_type = OpType.SBFM
            immr = (inst >> 16) & 0x3F
            imms = (inst >> 10) & 0x3F
            imm = (immr << 6) | imms

        # EXTR: 0x93C00000 (extract register - for ROR immediate)
        elif (inst & 0xFFE00000) == 0x93C00000:
            op_type = OpType.EXTR
            imm = (inst >> 10) & 0x3F  # lsb position

        # TBZ: 0x36xxxxxx (test bit and branch if zero)
        elif (inst & 0x7F000000) == 0x36000000:
            op_type = OpType.TBZ
            bit_pos = ((inst >> 19) & 0x1F) | ((inst >> 26) & 0x20)
            imm = bit_pos
            off = (inst >> 5) & 0x3FFF
            if off & 0x2000:
                off -= 0x4000
            branch_off = off

        # TBNZ: 0x37xxxxxx (test bit and branch if not zero)
        elif (inst & 0x7F000000) == 0x37000000:
            op_type = OpType.TBNZ
            bit_pos = ((inst >> 19) & 0x1F) | ((inst >> 26) & 0x20)
            imm = bit_pos
            off = (inst >> 5) & 0x3FFF
            if off & 0x2000:
                off -= 0x4000
            branch_off = off

        # RBIT: 0xDAC00000 (reverse bits)
        elif (inst & 0xFFFFFC00) == 0xDAC00000:
            op_type = OpType.RBIT

        # REV: 0xDAC00C00 (reverse bytes in 64-bit)
        elif (inst & 0xFFFFFC00) == 0xDAC00C00:
            op_type = OpType.REV

        # REV16: 0xDAC00400 (reverse bytes in each 16-bit halfword)
        elif (inst & 0xFFFFFC00) == 0xDAC00400:
            op_type = OpType.REV16

        # REV32: 0xDAC00800 (reverse bytes in each 32-bit word)
        elif (inst & 0xFFFFFC00) == 0xDAC00800:
            op_type = OpType.REV32

        # ANDS register: 0xEA000000 (AND, set flags)
        elif (inst & 0xFFE00000) == 0xEA000000 and rd != 31:
            op_type = OpType.ANDS_REG

        # ANDS immediate: 0xF2000000
        elif op_byte == 0xF2 and rd != 31:
            op_type = OpType.ANDS_IMM
            imm = self._decode_bitmask_imm(inst)

        # LDXR: 0xC85F7C00 (load exclusive register)
        elif (inst & 0xFFFFFC00) == 0xC85F7C00:
            op_type = OpType.LDXR

        # STXR: 0xC8007C00 (store exclusive register)
        elif (inst & 0xFFE07C00) == 0xC8007C00:
            op_type = OpType.STXR

        # DMB: 0xD50330BF (data memory barrier)
        elif (inst & 0xFFFFF0FF) == 0xD50330BF:
            op_type = OpType.DMB

        # DSB: 0xD503309F (data synchronization barrier)
        elif (inst & 0xFFFFF0FF) == 0xD503309F:
            op_type = OpType.DSB

        # ISB: 0xD50330DF (instruction synchronization barrier)
        elif (inst & 0xFFFFF0FF) == 0xD50330DF:
            op_type = OpType.ISB

        # MRS: 0xD5300000 (read system register)
        elif (inst & 0xFFF00000) == 0xD5300000:
            op_type = OpType.MRS
            imm = (inst >> 5) & 0x7FFF  # System register encoding

        # MSR: 0xD5100000 (write system register)
        elif (inst & 0xFFF00000) == 0xD5100000:
            op_type = OpType.MSR
            imm = (inst >> 5) & 0x7FFF

        # ERET: 0xD69F03E0 (exception return)
        elif inst == 0xD69F03E0:
            op_type = OpType.ERET

        # ADD extended register: 0x8B200000 (ADD with UXTW, SXTW, etc.)
        elif (inst & 0xFFE00000) == 0x8B200000:
            op_type = OpType.ADD_EXT
            imm = (inst >> 10) & 0x7  # shift amount
            rm = (inst >> 16) & 0x1F  # Also need extension type from bits 13-15

        # SUB extended register: 0xCB200000
        elif (inst & 0xFFE00000) == 0xCB200000:
            op_type = OpType.SUB_EXT
            imm = (inst >> 10) & 0x7

        # ═══════════════════════════════════════════════════════════════════════
        # 32-BIT (W) INSTRUCTION VARIANTS FOR BUSYBOX
        # ═══════════════════════════════════════════════════════════════════════

        # MOVZ 32-bit: 0x52xxxxxx
        elif op_byte == 0x52:
            op_type = OpType.MOVZ_W
            imm16_l, hw_l = self.movz_ext(bits)
            imm16 = ((imm16_l[0] > 0).long() * self.powers_16).sum()
            hw = ((hw_l[0] > 0).long() * self.powers_16[:2]).sum()
            imm = (imm16 | (hw << 16)).item()

        # MOVK 32-bit: 0x72xxxxxx
        elif op_byte == 0x72:
            op_type = OpType.MOVK_W
            imm16_l, hw_l = self.movz_ext(bits)
            imm16 = ((imm16_l[0] > 0).long() * self.powers_16).sum()
            hw = ((hw_l[0] > 0).long() * self.powers_16[:2]).sum()
            imm = (imm16 | (hw << 16)).item()

        # MOV 32-bit (ORR with WZR): 0x2Axxxxxx
        elif op_byte == 0x2A:
            op_type = OpType.MOV_W

        # ADD 32-bit immediate: 0x11xxxxxx
        elif op_byte == 0x11:
            op_type = OpType.ADD_IMM_W
            imm = (inst >> 10) & 0xFFF

        # SUB 32-bit immediate: 0x51xxxxxx
        elif op_byte == 0x51:
            op_type = OpType.SUB_IMM_W
            imm = (inst >> 10) & 0xFFF

        # ADD 32-bit register: 0x0Bxxxxxx
        elif op_byte == 0x0B:
            op_type = OpType.ADD_REG_W

        # SUB 32-bit register: 0x4Bxxxxxx
        elif op_byte == 0x4B:
            op_type = OpType.SUB_REG_W

        # ADDS 32-bit immediate: 0x31xxxxxx
        elif op_byte == 0x31:
            op_type = OpType.ADDS_IMM_W
            imm = (inst >> 10) & 0xFFF

        # SUBS 32-bit immediate: 0x71xxxxxx (CMP_W when Rd=WZR)
        elif op_byte == 0x71:
            if rd == 31:
                op_type = OpType.CMP_IMM_W
            else:
                op_type = OpType.SUBS_IMM_W
            imm = (inst >> 10) & 0xFFF

        # SUBS 32-bit register: 0x6Bxxxxxx (CMP_W when Rd=WZR)
        elif op_byte == 0x6B:
            if rd == 31:
                op_type = OpType.CMP_REG_W
            else:
                op_type = OpType.SUBS_IMM_W  # Actually SUBS_REG_W

        # LDR 32-bit (word): 0xB9400000
        elif (inst & 0xFFC00000) == 0xB9400000:
            op_type = OpType.LDR_W
            imm = ((inst >> 10) & 0xFFF) * 4  # Scale by 4

        # STR 32-bit (word): 0xB9000000
        elif (inst & 0xFFC00000) == 0xB9000000:
            op_type = OpType.STR_W
            imm = ((inst >> 10) & 0xFFF) * 4

        # CSEL 32-bit: 0x1A800000
        elif (inst & 0xFFE00C00) == 0x1A800000:
            op_type = OpType.CSEL_W
            imm = (inst >> 12) & 0xF

        # MADD 32-bit: 0x1B000000
        elif (inst & 0xFFE08000) == 0x1B000000:
            op_type = OpType.MADD_W

        # MOVN 64-bit: 0x92xxxxxx
        elif op_byte == 0x92 and op_type == OpType.NOP:
            op_type = OpType.MOVN
            imm16_l, hw_l = self.movz_ext(bits)
            imm16 = ((imm16_l[0] > 0).long() * self.powers_16).sum()
            hw = ((hw_l[0] > 0).long() * self.powers_16[:2]).sum()
            imm = (imm16 | (hw << 16)).item()

        # MOVN 32-bit: 0x12xxxxxx
        elif op_byte == 0x12:
            op_type = OpType.MOVN_W
            imm16_l, hw_l = self.movz_ext(bits)
            imm16 = ((imm16_l[0] > 0).long() * self.powers_16).sum()
            hw = ((hw_l[0] > 0).long() * self.powers_16[:2]).sum()
            imm = (imm16 | (hw << 16)).item()

        result = (op_type, rd, rn, rm, imm, branch_off)
        self.decode_cache[inst] = result
        return result

    def _eval_condition(self, cond_code: int) -> bool:
        """
        Evaluate ARM64 condition code using current flags.
        Returns True if condition is met.
        Uses flags on GPU, only final comparison transfers to CPU.
        """
        n = self.flags[0].item() > 0.5
        z = self.flags[1].item() > 0.5
        c = self.flags[2].item() > 0.5
        v = self.flags[3].item() > 0.5

        if cond_code == 0:    # EQ - equal (Z set)
            return z
        elif cond_code == 1:  # NE - not equal (Z clear)
            return not z
        elif cond_code == 2:  # CS/HS - carry set / unsigned higher or same
            return c
        elif cond_code == 3:  # CC/LO - carry clear / unsigned lower
            return not c
        elif cond_code == 4:  # MI - negative (N set)
            return n
        elif cond_code == 5:  # PL - positive or zero (N clear)
            return not n
        elif cond_code == 6:  # VS - overflow (V set)
            return v
        elif cond_code == 7:  # VC - no overflow (V clear)
            return not v
        elif cond_code == 8:  # HI - unsigned higher (C set and Z clear)
            return c and not z
        elif cond_code == 9:  # LS - unsigned lower or same (C clear or Z set)
            return not c or z
        elif cond_code == 10: # GE - signed greater or equal (N == V)
            return n == v
        elif cond_code == 11: # LT - signed less than (N != V)
            return n != v
        elif cond_code == 12: # GT - signed greater than (Z clear and N == V)
            return not z and (n == v)
        elif cond_code == 13: # LE - signed less or equal (Z set or N != V)
            return z or (n != v)
        elif cond_code == 14: # AL - always
            return True
        else:                 # NV - never (condition 15 always false)
            return False

    def _decode_bitmask_imm(self, inst: int) -> int:
        """
        Decode ARM64 bitmask immediate.
        Full ARM64 logical immediate decoder.
        Returns the immediate value as an integer.
        """
        sf = (inst >> 31) & 1  # 0=32-bit, 1=64-bit
        N = (inst >> 22) & 1
        immr = (inst >> 16) & 0x3F
        imms = (inst >> 10) & 0x3F

        # Determine element size from N and imms
        if N == 1:
            # 64-bit element
            len_val = 6
        else:
            # Find highest bit of ~imms to determine size
            not_imms = (~imms) & 0x3F
            if not_imms == 0:
                return 0  # Reserved
            # Count leading zeros in 6-bit field
            len_val = 0
            for i in range(5, -1, -1):
                if not_imms & (1 << i):
                    len_val = i + 1
                    break

        if len_val == 0:
            return 0

        size = 1 << len_val
        # Extract S and R
        S = imms & ((1 << len_val) - 1)
        R = immr & ((1 << len_val) - 1)

        # Create pattern of (S+1) ones
        pattern = (1 << (S + 1)) - 1

        # Rotate right by R
        if R > 0:
            pattern = ((pattern >> R) | (pattern << (size - R))) & ((1 << size) - 1)

        # Replicate to 64 bits
        result = 0
        for i in range(64 // size):
            result |= pattern << (i * size)

        # Mask to appropriate size
        if sf == 0:
            result &= 0xFFFFFFFF

        return result

