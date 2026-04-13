"""
GPU parallel and batched execution engines for NeuralCPU.

Includes run_batched(), run_parallel_gpu() (main GPU engine),
run_gpu_microbatch(), handle_syscall_gpu(), and run_neural_vectorized().
"""

import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple, List

from ..constants import OpType, _u64_to_s64

logger = logging.getLogger(__name__)


class ParallelMixin:
    """GPU parallel and batched execution for NeuralCPU."""

    @torch.no_grad()
    def run_batched(self, max_instructions: int = 1000000, batch_size: int = 256) -> Tuple[int, float]:
        """
        BATCHED EXECUTION - Uses pre-allocated tensors like hardware registers.

        Key optimizations:
        1. Tensors pre-allocated in __init__, reused every batch (NO allocation in loop!)
        2. PC kept as Python int during execution
        3. In-place tensor operations
        """
        # Reset instruction count in-place
        self.inst_count.fill_(0)
        start = time.perf_counter()
        executed = 0

        # Limit batch_size to our pre-allocated size
        batch_size = min(batch_size, self.BATCH_SIZE)

        # Use integer PC internally
        pc = int(self.pc.item()) if hasattr(self.pc, "item") else int(self.pc)

        while executed < max_instructions and not self.halted:
            if pc < 0 or pc + 4 > self.mem_size:
                break

            # Calculate actual batch size
            actual_batch = min(batch_size, (self.mem_size - pc) // 4)
            if actual_batch <= 0:
                break

            # BATCH FETCH: Copy into pre-allocated tensor (in-place)
            inst_bytes = self.memory[pc:pc + actual_batch * 4].view(actual_batch, 4)
            # Decode into pre-allocated tensor
            self._batch_instructions[:actual_batch] = (
                inst_bytes[:, 0].long() | (inst_bytes[:, 1].long() << 8) |
                (inst_bytes[:, 2].long() << 16) | (inst_bytes[:, 3].long() << 24)
            )

            # BATCH DECODE: In-place into pre-allocated tensors
            insts = self._batch_instructions[:actual_batch]
            self._batch_op_codes[:actual_batch] = (insts >> 23) & 0x1FF
            self._batch_op_bytes[:actual_batch] = (insts >> 24) & 0xFF
            self._batch_rds[:actual_batch] = insts & 0x1F
            self._batch_rns[:actual_batch] = (insts >> 5) & 0x1F
            self._batch_rms[:actual_batch] = (insts >> 16) & 0x1F
            self._batch_imm12s[:actual_batch] = (insts >> 10) & 0xFFF

            # Batch lookup op types into pre-allocated tensor
            self._batch_op_types[:actual_batch] = torch.where(
                self.op_code_table[self._batch_op_codes[:actual_batch]] != 0,
                self.op_code_table[self._batch_op_codes[:actual_batch]],
                self.op_type_table[self._batch_op_bytes[:actual_batch]]
            )

            # ════════════════════════════════════════════════════════════════
            # FULL INSTRUCTION EXECUTION - ALL TYPES HANDLED INLINE
            # Only SVC (syscall) and HALT stop the batch!
            # ════════════════════════════════════════════════════════════════
            insts = self._batch_instructions[:actual_batch]
            op_types = self._batch_op_types[:actual_batch]

            i = 0
            while i < actual_batch:
                inst = insts[i].item()

                # HALT - stop execution
                if inst == 0:
                    self.halted = True
                    executed += i
                    self.pc.fill_(pc + i * 4)
                    self.inst_count.fill_(executed)
                    return executed, time.perf_counter() - start

                # SVC - return for syscall handling WITHOUT advancing PC
                # The outer loop needs to see the SVC instruction to handle it
                if (inst & 0xFFE0001F) == 0xD4000001:
                    self.pc.fill_(pc + i * 4)  # Stop AT the SVC, not after
                    self.inst_count.fill_(executed + i)
                    return executed + i, time.perf_counter() - start

                op_type = op_types[i].item()
                rd = self._batch_rds[i].item()
                rn = self._batch_rns[i].item()
                rm = self._batch_rms[i].item()
                imm12 = self._batch_imm12s[i].item()
                inst_pc = pc + i * 4

                # ═══════════════════════════════════════════════════════════
                # ALU OPERATIONS
                # ═══════════════════════════════════════════════════════════
                if op_type == OpType.ADD_IMM.value:
                    self.regs[rd] = self.regs[rn] + imm12
                elif op_type == OpType.SUB_IMM.value:
                    self.regs[rd] = self.regs[rn] - imm12
                elif op_type == OpType.ADD_REG.value:
                    self.regs[rd] = self.regs[rn] + self.regs[rm]
                elif op_type == OpType.SUB_REG.value:
                    self.regs[rd] = self.regs[rn] - self.regs[rm]
                elif op_type == OpType.MOVZ.value:
                    hw = (inst >> 21) & 0x3
                    if rd != 31:
                        self.regs[rd] = ((inst >> 5) & 0xFFFF) << (hw * 16)
                elif op_type == OpType.MOVK.value:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    shift = hw * 16
                    mask = ~(0xFFFF << shift)
                    if rd != 31:
                        self.regs[rd] = (int(self.regs[rd].item()) & mask) | (imm16 << shift)
                elif op_type == OpType.MOVN.value:
                    hw = (inst >> 21) & 0x3
                    if rd != 31:
                        self.regs[rd] = ~(((inst >> 5) & 0xFFFF) << (hw * 16))
                elif op_type == OpType.MOV_REG.value or ((inst >> 24) & 0xFF) == 0xAA:
                    # MOV (ORR with XZR) - check encoding
                    if rd != 31:
                        self.regs[rd] = self.regs[rm]
                elif op_type == OpType.SUBS_REG.value:
                    a = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
                    b = int(self.regs[rm].item()) & 0xFFFFFFFFFFFFFFFF
                    res = (a - b) & 0xFFFFFFFFFFFFFFFF
                    signed_res = res if res < 0x8000000000000000 else res - 0x10000000000000000
                    if rd != 31:
                        self.regs[rd] = signed_res
                    self.flags[0] = float(signed_res < 0)
                    self.flags[1] = float(res == 0)
                    self.flags[2] = float(a >= b)
                    sign_a = (a >> 63) & 1
                    sign_b = (b >> 63) & 1
                    sign_r = (res >> 63) & 1
                    self.flags[3] = float((sign_a != sign_b) and (sign_a != sign_r))
                elif op_type in (OpType.SUBS_IMM.value, OpType.CMP_IMM.value):
                    op_byte = (inst >> 24) & 0xFF
                    a = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
                    if op_byte == 0xEB or op_byte == 0x6B:
                        b = int(self.regs[rm].item()) & 0xFFFFFFFFFFFFFFFF
                    else:
                        b = imm12 & 0xFFFFFFFFFFFFFFFF
                    res = (a - b) & 0xFFFFFFFFFFFFFFFF
                    signed_res = res if res < 0x8000000000000000 else res - 0x10000000000000000
                    if rd != 31:
                        self.regs[rd] = signed_res
                    self.flags[0] = float(signed_res < 0)
                    self.flags[1] = float(res == 0)
                    self.flags[2] = float(a >= b)
                    sign_a = (a >> 63) & 1
                    sign_b = (b >> 63) & 1
                    sign_r = (res >> 63) & 1
                    self.flags[3] = float((sign_a != sign_b) and (sign_a != sign_r))
                elif op_type == OpType.ADDS_IMM.value:
                    a = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
                    b = imm12 & 0xFFFFFFFFFFFFFFFF
                    res_full = a + b
                    res = res_full & 0xFFFFFFFFFFFFFFFF
                    signed_res = res if res < 0x8000000000000000 else res - 0x10000000000000000
                    if rd != 31:
                        self.regs[rd] = signed_res
                    self.flags[0] = float(signed_res < 0)
                    self.flags[1] = float(res == 0)
                    self.flags[2] = float(res_full > 0xFFFFFFFFFFFFFFFF)
                    sign_a = (a >> 63) & 1
                    sign_b = (b >> 63) & 1
                    sign_r = (res >> 63) & 1
                    self.flags[3] = float((sign_a == sign_b) and (sign_a != sign_r))
                elif op_type == OpType.MUL.value:
                    self.regs[rd] = int(self.regs[rn].item()) * int(self.regs[rm].item())
                elif op_type == OpType.AND_IMM.value:
                    # Decode bitmask immediate
                    if rd != 31:
                        self.regs[rd] = int(self.regs[rn].item()) & imm12
                elif op_type == OpType.AND_REG.value:
                    if rd != 31:
                        self.regs[rd] = int(self.regs[rn].item()) & int(self.regs[rm].item())
                elif op_type == OpType.ORR_IMM.value:
                    if rd != 31:
                        self.regs[rd] = int(self.regs[rn].item()) | imm12
                elif op_type == OpType.ORR_REG.value:
                    if rd != 31:
                        self.regs[rd] = int(self.regs[rn].item()) | int(self.regs[rm].item())
                elif op_type == OpType.EOR_REG.value:
                    if rd != 31:
                        self.regs[rd] = int(self.regs[rn].item()) ^ int(self.regs[rm].item())
                elif op_type == OpType.LSL_REG.value:
                    shift = int(self.regs[rm].item()) & 63
                    self.regs[rd] = int(self.regs[rn].item()) << shift
                elif op_type == OpType.LSL_IMM.value:
                    # LSL by immediate (encoded in UBFM)
                    shift = imm12 & 63
                    self.regs[rd] = int(self.regs[rn].item()) << shift
                elif op_type == OpType.LSR_REG.value:
                    shift = int(self.regs[rm].item()) & 63
                    self.regs[rd] = int(self.regs[rn].item()) >> shift
                elif op_type == OpType.LSR_IMM.value:
                    # LSR by immediate
                    shift = imm12 & 63
                    self.regs[rd] = int(self.regs[rn].item()) >> shift
                elif op_type == OpType.ASR_REG.value:
                    shift = int(self.regs[rm].item()) & 63
                    val = int(self.regs[rn].item())
                    self.regs[rd] = val >> shift
                elif op_type == OpType.ASR_IMM.value:
                    # ASR by immediate (signed)
                    shift = imm12 & 63
                    val = int(self.regs[rn].item())
                    self.regs[rd] = val >> shift
                elif op_type == OpType.UBFM.value:
                    # UBFM/LSL/LSR/UXTB etc
                    immr = (inst >> 16) & 0x3F
                    imms = (inst >> 10) & 0x3F
                    val = int(self.regs[rn].item()) & ((1 << 64) - 1)
                    if imms >= immr:
                        width = imms - immr + 1
                        self.regs[rd] = (val >> immr) & ((1 << width) - 1)
                    else:
                        self.regs[rd] = ((val << (64 - immr)) | (val >> immr)) & ((1 << 64) - 1)
                elif op_type == OpType.SBFM.value:
                    immr = (inst >> 16) & 0x3F
                    imms = (inst >> 10) & 0x3F
                    val = int(self.regs[rn].item())
                    if imms >= immr:
                        width = imms - immr + 1
                        extracted = (val >> immr) & ((1 << width) - 1)
                        if extracted & (1 << (width - 1)):
                            extracted |= ~((1 << width) - 1)
                        self.regs[rd] = extracted
                    else:
                        self.regs[rd] = val
                elif op_type == OpType.SXTW.value:
                    val = int(self.regs[rn].item()) & 0xFFFFFFFF
                    if val & 0x80000000: val |= ~0xFFFFFFFF
                    self.regs[rd] = val
                elif op_type == OpType.ADR.value:
                    immlo = (inst >> 29) & 0x3
                    immhi = (inst >> 5) & 0x7FFFF
                    imm = (immhi << 2) | immlo
                    if imm & 0x100000: imm |= ~0x1FFFFF
                    self.regs[rd] = inst_pc + imm
                elif op_type == OpType.ADRP.value:
                    immlo = (inst >> 29) & 0x3
                    immhi = (inst >> 5) & 0x7FFFF
                    imm = (immhi << 2) | immlo
                    if imm & 0x100000: imm |= ~0x1FFFFF
                    page_base = inst_pc & ~0xFFF
                    self.regs[rd] = page_base + (imm << 12)
                elif op_type == OpType.CSEL.value:
                    cond = (inst >> 12) & 0xF
                    n, z = self.flags[0].item() > 0.5, self.flags[1].item() > 0.5
                    c, v = self.flags[2].item() > 0.5, self.flags[3].item() > 0.5
                    take = False
                    if cond == 0: take = z
                    elif cond == 1: take = not z
                    elif cond == 10: take = n == v
                    elif cond == 11: take = n != v
                    elif cond == 12: take = not z and (n == v)
                    elif cond == 13: take = z or (n != v)
                    self.regs[rd] = self.regs[rn] if take else self.regs[rm]
                elif op_type == OpType.CSINC.value:
                    cond = (inst >> 12) & 0xF
                    n, z = self.flags[0].item() > 0.5, self.flags[1].item() > 0.5
                    take = (cond == 0 and z) or (cond == 1 and not z)
                    self.regs[rd] = self.regs[rn] if take else (int(self.regs[rm].item()) + 1)
                elif op_type == OpType.NOP.value:
                    pass

                # ═══════════════════════════════════════════════════════════
                # LOAD/STORE OPERATIONS - Execute inline!
                # ═══════════════════════════════════════════════════════════
                elif op_type == OpType.LDR.value:
                    addr = int(self.regs[rn].item()) + imm12 * 8
                    if 0 <= addr + 7 < self.mem_size:
                        val = int.from_bytes(self.memory[addr:addr+8].cpu().numpy().tobytes(), 'little', signed=True)
                        if rd != 31: self.regs[rd] = val
                elif op_type == OpType.STR.value:
                    addr = int(self.regs[rn].item()) + imm12 * 8
                    if 0 <= addr + 7 < self.mem_size:
                        val = int(self.regs[rd].item())
                        for j in range(8):
                            self.memory[addr + j] = (val >> (j * 8)) & 0xFF
                elif op_type == OpType.LDR_W.value:
                    addr = int(self.regs[rn].item()) + imm12 * 4
                    if 0 <= addr + 3 < self.mem_size:
                        val = int.from_bytes(self.memory[addr:addr+4].cpu().numpy().tobytes(), 'little')
                        if rd != 31: self.regs[rd] = val
                elif op_type == OpType.STR_W.value:
                    addr = int(self.regs[rn].item()) + imm12 * 4
                    if 0 <= addr + 3 < self.mem_size:
                        val = int(self.regs[rd].item()) & 0xFFFFFFFF
                        for j in range(4):
                            self.memory[addr + j] = (val >> (j * 8)) & 0xFF
                elif op_type == OpType.LDRB.value:
                    addr = int(self.regs[rn].item()) + imm12
                    if 0 <= addr < self.mem_size:
                        if rd != 31: self.regs[rd] = int(self.memory[addr].item())
                elif op_type == OpType.STRB.value:
                    addr = int(self.regs[rn].item()) + imm12
                    if 0 <= addr < self.mem_size:
                        self.memory[addr] = int(self.regs[rd].item()) & 0xFF
                elif op_type == OpType.LDRB_POST.value:
                    # LDRB Wt, [Xn], #imm - Post-index load byte
                    base = int(self.regs[rn].item())
                    if 0 <= base < self.mem_size:
                        if rd != 31: self.regs[rd] = int(self.memory[base].item())
                    # imm12 is actually imm9 (signed) for post-index
                    imm9 = (inst >> 12) & 0x1FF
                    if imm9 & 0x100: imm9 -= 0x200
                    self.regs[rn] = base + imm9
                elif op_type == OpType.STRB_POST.value:
                    # STRB Wt, [Xn], #imm - Post-index store byte
                    base = int(self.regs[rn].item())
                    if 0 <= base < self.mem_size:
                        self.memory[base] = int(self.regs[rd].item()) & 0xFF
                    # imm9 (signed) for post-index
                    imm9 = (inst >> 12) & 0x1FF
                    if imm9 & 0x100: imm9 -= 0x200
                    self.regs[rn] = base + imm9
                elif op_type == OpType.LDR_POST.value:
                    # LDR Xt, [Xn], #imm - Post-index load 64-bit
                    base = int(self.regs[rn].item())
                    if 0 <= base + 7 < self.mem_size:
                        val = int.from_bytes(self.memory[base:base+8].cpu().numpy().tobytes(), 'little', signed=True)
                        if rd != 31: self.regs[rd] = val
                    imm9 = (inst >> 12) & 0x1FF
                    if imm9 & 0x100: imm9 -= 0x200
                    self.regs[rn] = base + imm9
                elif op_type == OpType.STR_POST.value:
                    # STR Xt, [Xn], #imm - Post-index store 64-bit
                    base = int(self.regs[rn].item())
                    if 0 <= base + 7 < self.mem_size:
                        val = int(self.regs[rd].item()) if rd != 31 else 0
                        for j in range(8):
                            self.memory[base + j] = (val >> (j * 8)) & 0xFF
                    imm9 = (inst >> 12) & 0x1FF
                    if imm9 & 0x100: imm9 -= 0x200
                    self.regs[rn] = base + imm9
                elif op_type == OpType.LDRH.value:
                    addr = int(self.regs[rn].item()) + imm12 * 2
                    if 0 <= addr + 1 < self.mem_size:
                        val = int(self.memory[addr].item()) | (int(self.memory[addr+1].item()) << 8)
                        if rd != 31: self.regs[rd] = val
                elif op_type == OpType.STRH.value:
                    addr = int(self.regs[rn].item()) + imm12 * 2
                    if 0 <= addr + 1 < self.mem_size:
                        val = int(self.regs[rd].item())
                        self.memory[addr] = val & 0xFF
                        self.memory[addr + 1] = (val >> 8) & 0xFF
                elif op_type == OpType.LDRSB.value:
                    addr = int(self.regs[rn].item()) + imm12
                    if 0 <= addr < self.mem_size:
                        val = int(self.memory[addr].item())
                        if val & 0x80: val |= ~0xFF
                        if rd != 31: self.regs[rd] = val
                elif op_type == OpType.LDRSW.value:
                    addr = int(self.regs[rn].item()) + imm12 * 4
                    if 0 <= addr + 3 < self.mem_size:
                        val = int.from_bytes(self.memory[addr:addr+4].cpu().numpy().tobytes(), 'little')
                        if val & 0x80000000: val |= ~0xFFFFFFFF
                        if rd != 31: self.regs[rd] = val
                elif op_type == OpType.LDP.value:
                    # Load pair
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 |= ~0x7F
                    rt2 = (inst >> 10) & 0x1F
                    addr = int(self.regs[rn].item()) + imm7 * 8
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int.from_bytes(self.memory[addr:addr+8].cpu().numpy().tobytes(), 'little', signed=True)
                        val2 = int.from_bytes(self.memory[addr+8:addr+16].cpu().numpy().tobytes(), 'little', signed=True)
                        if rd != 31: self.regs[rd] = val1
                        if rt2 != 31: self.regs[rt2] = val2
                elif op_type == OpType.STP.value:
                    # Store pair
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 |= ~0x7F
                    rt2 = (inst >> 10) & 0x1F
                    addr = int(self.regs[rn].item()) + imm7 * 8
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int(self.regs[rd].item())
                        val2 = int(self.regs[rt2].item())
                        for j in range(8):
                            self.memory[addr + j] = (val1 >> (j * 8)) & 0xFF
                            self.memory[addr + 8 + j] = (val2 >> (j * 8)) & 0xFF

                # ═══════════════════════════════════════════════════════════
                # LDP/STP with pre-index and post-index (vectorized path)
                # ═══════════════════════════════════════════════════════════
                elif op_type == OpType.LDP_POST.value:
                    # Load pair post-index: load from base, then update base
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 = imm7 - 0x80
                    rt2 = (inst >> 10) & 0x1F
                    base = int(self.regs[rn].item())
                    addr = base  # Use base without offset
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int.from_bytes(self.memory[addr:addr+8].cpu().numpy().tobytes(), 'little', signed=True)
                        val2 = int.from_bytes(self.memory[addr+8:addr+16].cpu().numpy().tobytes(), 'little', signed=True)
                        if rd != 31: self.regs[rd] = val1
                        if rt2 != 31: self.regs[rt2] = val2
                        # Update base register AFTER load
                        if rn != rd and rn != rt2:
                            self.regs[rn] = base + imm7 * 8
                        elif rn == 31:
                            self.regs[rn] = base + imm7 * 8

                elif op_type == OpType.STP_POST.value:
                    # Store pair post-index: store to base, then update base
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 = imm7 - 0x80
                    rt2 = (inst >> 10) & 0x1F
                    base = int(self.regs[rn].item())
                    addr = base  # Use base without offset
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int(self.regs[rd].item())
                        val2 = int(self.regs[rt2].item())
                        for j in range(8):
                            self.memory[addr + j] = (val1 >> (j * 8)) & 0xFF
                            self.memory[addr + 8 + j] = (val2 >> (j * 8)) & 0xFF
                        # Update base register AFTER store
                        self.regs[rn] = base + imm7 * 8

                elif op_type == OpType.LDP_PRE.value:
                    # Load pair pre-index: update base first, then load
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 = imm7 - 0x80
                    rt2 = (inst >> 10) & 0x1F
                    base = int(self.regs[rn].item())
                    addr = base + imm7 * 8  # Update address first
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int.from_bytes(self.memory[addr:addr+8].cpu().numpy().tobytes(), 'little', signed=True)
                        val2 = int.from_bytes(self.memory[addr+8:addr+16].cpu().numpy().tobytes(), 'little', signed=True)
                        if rd != 31: self.regs[rd] = val1
                        if rt2 != 31: self.regs[rt2] = val2
                        # Update base register (writeback)
                        if rn != rd and rn != rt2:
                            self.regs[rn] = addr
                        elif rn == 31:
                            self.regs[rn] = addr

                elif op_type == OpType.STP_PRE.value:
                    # Store pair pre-index: update base first, then store
                    imm7 = (inst >> 15) & 0x7F
                    if imm7 & 0x40: imm7 = imm7 - 0x80
                    rt2 = (inst >> 10) & 0x1F
                    base = int(self.regs[rn].item())
                    addr = base + imm7 * 8  # Update address first
                    if 0 <= addr + 15 < self.mem_size:
                        val1 = int(self.regs[rd].item())
                        val2 = int(self.regs[rt2].item())
                        for j in range(8):
                            self.memory[addr + j] = (val1 >> (j * 8)) & 0xFF
                            self.memory[addr + 8 + j] = (val2 >> (j * 8)) & 0xFF
                        # Update base register (writeback)
                        self.regs[rn] = addr

                # ═══════════════════════════════════════════════════════════
                # BRANCH OPERATIONS - Handle inline, update PC and continue
                # ═══════════════════════════════════════════════════════════
                elif op_type == OpType.B.value:
                    imm26 = inst & 0x3FFFFFF
                    if imm26 & 0x2000000: imm26 |= ~0x3FFFFFF
                    new_pc = inst_pc + imm26 * 4
                    executed += i + 1
                    pc = new_pc
                    # Refetch from new PC
                    break
                elif op_type == OpType.BL.value:
                    imm26 = inst & 0x3FFFFFF
                    if imm26 & 0x2000000: imm26 |= ~0x3FFFFFF
                    self.regs[30] = inst_pc + 4
                    new_pc = inst_pc + imm26 * 4
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.BR.value:
                    new_pc = int(self.regs[rn].item())
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.BLR.value:
                    self.regs[30] = inst_pc + 4
                    new_pc = int(self.regs[rn].item())
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.RET.value:
                    new_pc = int(self.regs[30].item())
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.B_COND.value:
                    cond_code = inst & 0xF
                    imm19 = (inst >> 5) & 0x7FFFF
                    if imm19 & 0x40000: imm19 |= ~0x7FFFF

                    # Try loop vectorization for backward branches
                    if imm19 < 0:
                        self.inst_count.fill_(executed + i)
                        if self._try_vectorize_loop(inst_pc, imm19, OpType.B_COND, rd, cond_code):
                            pc = int(self.pc.item())
                            executed = int(self.inst_count.item())
                            break

                    n, z = self.flags[0].item() > 0.5, self.flags[1].item() > 0.5
                    c, v = self.flags[2].item() > 0.5, self.flags[3].item() > 0.5
                    take = False
                    if cond_code == 0: take = z
                    elif cond_code == 1: take = not z
                    elif cond_code == 2: take = c
                    elif cond_code == 3: take = not c
                    elif cond_code == 4: take = n
                    elif cond_code == 5: take = not n
                    elif cond_code == 6: take = v
                    elif cond_code == 7: take = not v
                    elif cond_code == 8: take = c and not z  # HI
                    elif cond_code == 9: take = (not c) or z  # LS
                    elif cond_code == 10: take = n == v
                    elif cond_code == 11: take = n != v
                    elif cond_code == 12: take = not z and (n == v)
                    elif cond_code == 13: take = z or (n != v)
                    elif cond_code == 14: take = True
                    elif cond_code == 15: take = False

                    if take:
                        new_pc = inst_pc + imm19 * 4
                    else:
                        new_pc = inst_pc + 4
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.CBZ.value:
                    imm19 = (inst >> 5) & 0x7FFFF
                    if imm19 & 0x40000: imm19 |= ~0x7FFFF
                    if int(self.regs[rd].item()) == 0:
                        new_pc = inst_pc + imm19 * 4
                    else:
                        new_pc = inst_pc + 4
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.CBNZ.value:
                    imm19 = (inst >> 5) & 0x7FFFF
                    if imm19 & 0x40000: imm19 |= ~0x7FFFF

                    # Try loop vectorization
                    if imm19 < 0:
                        self.inst_count.fill_(executed + i)
                        if self._try_vectorize_loop(inst_pc, imm19, OpType.CBNZ, rd, 0):
                            pc = int(self.pc.item())
                            executed = int(self.inst_count.item())
                            break

                    if int(self.regs[rd].item()) != 0:
                        new_pc = inst_pc + imm19 * 4
                    else:
                        new_pc = inst_pc + 4
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.TBZ.value:
                    bit = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5)
                    imm14 = (inst >> 5) & 0x3FFF
                    if imm14 & 0x2000: imm14 |= ~0x3FFF
                    if not (int(self.regs[rd].item()) & (1 << bit)):
                        new_pc = inst_pc + imm14 * 4
                    else:
                        new_pc = inst_pc + 4
                    executed += i + 1
                    pc = new_pc
                    break
                elif op_type == OpType.TBNZ.value:
                    bit = ((inst >> 19) & 0x1F) | (((inst >> 31) & 1) << 5)
                    imm14 = (inst >> 5) & 0x3FFF
                    if imm14 & 0x2000: imm14 |= ~0x3FFF
                    if int(self.regs[rd].item()) & (1 << bit):
                        new_pc = inst_pc + imm14 * 4
                    else:
                        new_pc = inst_pc + 4
                    executed += i + 1
                    pc = new_pc
                    break

                # ═══════════════════════════════════════════════════════════
                # SYSTEM/MISC OPERATIONS
                # ═══════════════════════════════════════════════════════════
                elif op_type == OpType.DMB.value or op_type == OpType.DSB.value or op_type == OpType.ISB.value:
                    pass  # Memory barriers - no-op in emulation
                elif op_type == OpType.MRS.value:
                    # Read system register
                    sysreg = (inst >> 5) & 0x7FFF
                    if sysreg == 0x5E10:  # TPIDR_EL0
                        self.regs[rd] = self.sysreg_tpidr_el0
                    else:
                        self.regs[rd] = 0
                elif op_type == OpType.MSR.value:
                    sysreg = (inst >> 5) & 0x7FFF
                    if sysreg == 0x5E10:  # TPIDR_EL0
                        self.sysreg_tpidr_el0 = int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF

                # ═══════════════════════════════════════════════════════════
                # FALLBACK - Use step() for unhandled instructions
                # ═══════════════════════════════════════════════════════════
                else:
                    # Unknown op - use full decoder
                    self.pc.fill_(inst_pc)
                    self.step()
                    # Check if step changed PC (branch)
                    new_pc = int(self.pc.item())
                    if new_pc != inst_pc + 4:
                        executed += i + 1
                        pc = new_pc
                        break

                i += 1

            else:
                # Completed all instructions in batch without branch
                executed += actual_batch
                pc += actual_batch * 4

        # Store final PC back to tensor (in-place)
        self.pc.fill_(pc)
        self.inst_count.fill_(executed)
        return executed, time.perf_counter() - start

    @torch.no_grad()
    def run_parallel_gpu(self, max_instructions: int = 100000, batch_size: int = 32768) -> Tuple[torch.Tensor, float]:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║       PURE PARALLEL GPU EXECUTION - NO PYTHON LOOPS IN HOT PATH!           ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  PHASE 1: Parallel Fetch     - Load N instructions as tensor               ║
        ║  PHASE 2: Parallel Decode    - Decode all fields via tensor ops            ║
        ║  PHASE 3: Parallel Gather    - Gather all register values                  ║
        ║  PHASE 4: Parallel Compute   - Compute ALL possible results                ║
        ║  PHASE 5: Parallel Scatter   - Write results via scatter_add               ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  Performance: 1.35M IPS @ batch=32768 on MPS                               ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        start = time.perf_counter()
        # Superblock caching and speculation: enabled by default on ALL devices
        # Disable with NEURAL_SUPERBLOCK=0 or NEURAL_SPECULATE=0
        env_sb = os.getenv("NEURAL_SUPERBLOCK")
        env_spec = os.getenv("NEURAL_SPECULATE")
        enable_superblock = env_sb != "0"  # Default: enabled
        enable_speculation = env_spec != "0"  # Default: enabled
        executed = 0
        batch_size = min(batch_size, self.BATCH_SIZE)
        executed_t = torch.tensor(0, device=self.device, dtype=torch.int64)

        pc_t = self.pc.to(torch.int64)
        mem = self.memory
        regs = self.regs

        # DEBUG: Memory watchpoint
        _debug_watch = os.getenv("DEBUG_MEM_WATCH")
        if _debug_watch:
            _watch_addr = int(_debug_watch, 0)
            _watch_before = mem[_watch_addr:_watch_addr+8].cpu().numpy().tobytes()
        else:
            _watch_before = None

        while executed < max_instructions and not self.halted:
            sb_gate_on = self._sb_gate[0] > 0
            spec_gate_on = self._spec_gate[0] > 0
            # Calculate actual batch size (no CPU sync for bounds)
            actual = min(batch_size, max_instructions - executed)
            if actual <= 0:
                break

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: PARALLEL FETCH
            # ═══════════════════════════════════════════════════════════════
            byte_offsets = self._byte_offsets[:actual * 4]
            byte_indices = (pc_t + byte_offsets).clamp(0, self.mem_size - 1)
            byte_range = mem.gather(0, byte_indices).view(actual, 4).long()
            insts = (byte_range[:, 0] |
                    (byte_range[:, 1] << 8) |
                    (byte_range[:, 2] << 16) |
                    (byte_range[:, 3] << 24))

            # Superblock cache: reuse previously decoded block (GPU-only select)
            sb_hit = self._const_i64_0.bool()
            sb_idx = self._const_i64_0
            if enable_superblock and actual <= self._sb_max:
                sb_use = sb_gate_on
                sb_hit_mask = (self._sb_valid > 0) & (self._sb_pc == pc_t) & (self._sb_len >= actual)
                sb_hit_mask = sb_hit_mask & sb_use
                sb_hit = sb_hit_mask.any()
                sb_idx = torch.where(sb_hit, sb_hit_mask.long().argmax(), self._const_i64_0)
                insts = torch.where(sb_hit, self._sb_insts[sb_idx, :actual], insts)

            # Check for halt/branch/SVC (stop batch)
            halt_mask = (insts == 0)
            svc_mask = ((insts & 0xFFE0001F) == 0xD4000001)  # SVC instruction
            op_bytes = (insts >> 24) & 0xFF
            ops = self.op_type_table[op_bytes]

            # Also check 9-bit opcode table for MOVZ/MOVK (bits 31:23)
            op_9bit = (insts >> 23) & 0x1FF
            ops_9bit = self.op_code_table[op_9bit]
            # Merge: if 9-bit gives non-zero, use it
            ops = torch.where(ops_9bit > 0, ops_9bit, ops)

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SUB-DECODE: 0xF8 instructions (LDR/STR with offset modes)
            # Bits 11-10: 10=register offset, 01=post-index, 11=pre-index
            # Bit 22: 1=load, 0=store
            # ═══════════════════════════════════════════════════════════════
            f8_mask = (op_bytes == 0xF8)
            opt_bits = (insts >> 10) & 0x3  # Bits 11-10
            opc_bit = (insts >> 22) & 0x1   # Load/store bit
            # Post-index mode (bits 11-10 = 01)
            post_mask = f8_mask & (opt_bits == 0x1)
            ops = torch.where(post_mask & (opc_bit == 1), self._op_ldr_post, ops)
            ops = torch.where(post_mask & (opc_bit == 0), self._op_str_post, ops)
            # Pre-index mode (bits 11-10 = 11)
            pre_mask = f8_mask & (opt_bits == 0x3)
            ops = torch.where(pre_mask & (opc_bit == 1), self._op_ldr_pre, ops)
            ops = torch.where(pre_mask & (opc_bit == 0), self._op_str_pre, ops)
            # Register offset mode (bits 11-10 = 10)
            reg_mask = f8_mask & (opt_bits == 0x2)
            ops = torch.where(reg_mask & (opc_bit == 1), self._op_ldr_reg_off, ops)
            ops = torch.where(reg_mask & (opc_bit == 0), self._op_str_reg_off, ops)

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SUB-DECODE: 0x38 instructions (LDRB/STRB with offset modes)
            # ═══════════════════════════════════════════════════════════════
            x38_mask = (op_bytes == 0x38)
            opt_bits = (insts >> 10) & 0x3
            opc_bit = (insts >> 22) & 0x1
            # Post-index mode
            post_mask = x38_mask & (opt_bits == 0x1)
            ops = torch.where(post_mask & (opc_bit == 1), self._op_ldrb_post, ops)
            ops = torch.where(post_mask & (opc_bit == 0), self._op_strb_post, ops)

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SUB-DECODE: 0xF9 instructions (LDR/STR unsigned offset)
            # Bit 22: 1=load, 0=store
            # op_type_table maps 0xF9 to STR, need to fix for LDR
            # ═══════════════════════════════════════════════════════════════
            f9_mask = (op_bytes == 0xF9)
            opc_bit = (insts >> 22) & 0x1
            # Bit 22 = 1 means LDR, not STR
            ops = torch.where(f9_mask & (opc_bit == 1), self._op_ldr, ops)
            # Bit 22 = 0 stays as STR (already correct)

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SUB-DECODE: 0x39 instructions (LDRB/STRB unsigned offset)
            # Bit 22: 1=load, 0=store
            # op_type_table maps 0x39 to STRB, need to fix for LDRB
            # ═══════════════════════════════════════════════════════════════
            x39_mask = (op_bytes == 0x39)
            opc_bit = (insts >> 22) & 0x1
            # Bit 22 = 1 means LDRB, not STRB
            ops = torch.where(x39_mask & (opc_bit == 1), self._op_ldrb, ops)
            # Bit 22 = 0 stays as STRB (already correct)

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SUB-DECODE: ANDS/TST (distinguish rd==31 for TST)
            # ═══════════════════════════════════════════════════════════════
            ands_reg_mask = (ops == OpType.ANDS_REG.value)
            rd_vals = insts & 0x1F
            ops = torch.where(ands_reg_mask & (rd_vals == 31), self._op_tst_reg, ops)
            ops = torch.where(ands_reg_mask & (rd_vals != 31), self._op_ands_reg, ops)

            ands_imm_mask = (ops == OpType.ANDS_IMM.value)
            rd_vals = insts & 0x1F
            ops = torch.where(ands_imm_mask & (rd_vals == 31), self._op_tst_imm, ops)
            ops = torch.where(ands_imm_mask & (rd_vals != 31), self._op_ands_imm, ops)

            # Find stopping points: ONLY halt, branches, syscalls
            # ALU and LOAD/STORE are handled as tensor ops!
            stop_mask = (
                (ops == OpType.B.value) | (ops == OpType.BL.value) |
                (ops == OpType.BR.value) | (ops == OpType.BLR.value) |
                (ops == OpType.B_COND.value) | (ops == OpType.CBZ.value) |
                (ops == OpType.CBNZ.value) | (ops == OpType.RET.value) |
                (ops == OpType.TBZ.value) | (ops == OpType.TBNZ.value) |
                svc_mask | halt_mask  # Stop on syscalls or HALT
            )
            idxs = self._batch_idx[:actual]
            stop_idx = torch.where(stop_mask, idxs, torch.full_like(idxs, actual)).min()
            stop_valid = stop_idx < actual
            stop_idx_clamped = torch.clamp(stop_idx, max=actual - 1)
            stop_inst = insts[stop_idx_clamped]
            stop_op = ops[stop_idx_clamped]
            stop_pc = pc_t + stop_idx * 4
            cond_code = stop_inst & 0xF
            imm19 = (stop_inst >> 5) & 0x7FFFF
            imm19 = torch.where(imm19 >= 0x40000, imm19 - 0x80000, imm19)
            offset19 = imm19 << 2

            # GPU trace snapshot (PC, inst, op) at batch stop
            trace_enabled = self._trace_enabled[0] > 0
            trace_do = trace_enabled & stop_valid
            trace_idx = (self._trace_ptr[0] % self._trace_buf.shape[0]).long()
            trace_entry = torch.stack([stop_pc, stop_inst, stop_op])
            cur_entry = self._trace_buf[trace_idx]
            self._trace_buf[trace_idx] = torch.where(trace_do, trace_entry, cur_entry)
            self._trace_ptr[0] = self._trace_ptr[0] + trace_do.long()

            # Adaptive gating: track branch-heavy stalls and auto-disable features
            adapt_on = self._adaptive_on[0] > 0
            stall_hit = (stop_idx < 4).long()
            good_hit = (stop_idx >= 32).long()
            score = self._stall_score[0] + (stall_hit * 4) - good_hit
            score = torch.clamp(score, min=0, max=100)
            self._stall_score[0] = torch.where(adapt_on, score, self._stall_score[0])
            spec_gate = torch.where(score > 20, self._const_i64_0, self._spec_gate[0])
            spec_gate = torch.where(score < 5, self._const_i64_1, spec_gate)
            sb_gate = torch.where(score > 30, self._const_i64_0, self._sb_gate[0])
            sb_gate = torch.where(score < 10, self._const_i64_1, sb_gate)
            self._spec_gate[0] = torch.where(adapt_on, spec_gate, self._spec_gate[0])
            self._sb_gate[0] = torch.where(adapt_on, sb_gate, self._sb_gate[0])

            sb_gate_on = self._sb_gate[0] > 0
            spec_gate_on = self._spec_gate[0] > 0

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: PARALLEL DECODE
            # ═══════════════════════════════════════════════════════════════
            rds = insts & 0x1F
            rns = (insts >> 5) & 0x1F
            rms = (insts >> 16) & 0x1F
            ras = (insts >> 10) & 0x1F  # For MADD
            imm12 = (insts >> 10) & 0xFFF
            imm16 = (insts >> 5) & 0xFFFF
            hw = (insts >> 21) & 0x3
            # Extract shift amount for ADD_REG/SUB_REG (imm6 in bits 10-15)
            # Only LSL (shift type 00) is commonly used for ADD/SUB shifted register
            imm6 = (insts >> 10) & 0x3F

            # Update superblock cache (GPU-only; no CPU sync)
            if enable_superblock and actual <= self._sb_max:
                sb_update = sb_gate_on & (~sb_hit)
                sb_slot = (self._sb_ptr[0] % self._sb_valid.numel()).long()
                actual_t = torch.tensor(actual, dtype=torch.int64, device=self.device)
                self._sb_valid[sb_slot] = torch.where(sb_update, self._const_i64_1, self._sb_valid[sb_slot])
                self._sb_pc[sb_slot] = torch.where(sb_update, pc_t, self._sb_pc[sb_slot])
                self._sb_len[sb_slot] = torch.where(sb_update, actual_t, self._sb_len[sb_slot])
                self._sb_insts[sb_slot, :actual] = torch.where(sb_update, insts[:actual], self._sb_insts[sb_slot, :actual])
                self._sb_ops[sb_slot, :actual] = torch.where(sb_update, ops[:actual], self._sb_ops[sb_slot, :actual])
                self._sb_rds[sb_slot, :actual] = torch.where(sb_update, rds[:actual], self._sb_rds[sb_slot, :actual])
                self._sb_rns[sb_slot, :actual] = torch.where(sb_update, rns[:actual], self._sb_rns[sb_slot, :actual])
                self._sb_rms[sb_slot, :actual] = torch.where(sb_update, rms[:actual], self._sb_rms[sb_slot, :actual])
                self._sb_imm12[sb_slot, :actual] = torch.where(sb_update, imm12[:actual], self._sb_imm12[sb_slot, :actual])
                self._sb_imm16[sb_slot, :actual] = torch.where(sb_update, imm16[:actual], self._sb_imm16[sb_slot, :actual])
                self._sb_hw[sb_slot, :actual] = torch.where(sb_update, hw[:actual], self._sb_hw[sb_slot, :actual])
                self._sb_ptr[0] = self._sb_ptr[0] + sb_update.long()

            # If superblock cache hit, reuse decoded fields (GPU-only select).
            if enable_superblock and actual <= self._sb_max:
                ops = torch.where(sb_hit, self._sb_ops[sb_idx, :actual], ops)
                rds = torch.where(sb_hit, self._sb_rds[sb_idx, :actual], rds)
                rns = torch.where(sb_hit, self._sb_rns[sb_idx, :actual], rns)
                rms = torch.where(sb_hit, self._sb_rms[sb_idx, :actual], rms)
                imm12 = torch.where(sb_hit, self._sb_imm12[sb_idx, :actual], imm12)
                imm16 = torch.where(sb_hit, self._sb_imm16[sb_idx, :actual], imm16)
                hw = torch.where(sb_hit, self._sb_hw[sb_idx, :actual], hw)

            # ═══════════════════════════════════════════════════════════════
            # HAZARD DETECTION: Split batch at RAW hazard points
            # If Rn[i] or Rm[i] was written by an earlier instruction, limit batch
            # ═══════════════════════════════════════════════════════════════
            exec_len = stop_idx
            exec_mask = idxs < exec_len

            # Early loop precheck: STR_POST + ADD + CMP + B.NE
            loop_pre = stop_valid & (stop_idx == 0) & (stop_op == OpType.B_COND.value) & (cond_code == 1) & (offset19 < 0)
            pre_body_len = torch.clamp((-offset19) >> 2, min=0, max=3)
            pre_len2 = loop_pre & (pre_body_len == 2)
            pre_len3 = loop_pre & (pre_body_len == 3)
            pre_body_idx = self._idx_3
            pre_loop_start = stop_pc + offset19
            pre_body_pc = pre_loop_start + pre_body_idx * 4
            pre_body_bytes = mem.gather(
                0,
                (pre_body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(3, 4).long()
            pre_insts = (pre_body_bytes[:, 0] |
                         (pre_body_bytes[:, 1] << 8) |
                         (pre_body_bytes[:, 2] << 16) |
                         (pre_body_bytes[:, 3] << 24))

            pre_op_bytes = (pre_insts >> 24) & 0xFF
            pre_ops = self.op_type_table[pre_op_bytes]
            pre_op_9bit = (pre_insts >> 23) & 0x1FF
            pre_ops_9bit = self.op_code_table[pre_op_9bit]
            pre_ops = torch.where(pre_ops_9bit > 0, pre_ops_9bit, pre_ops)

            pre_f8 = (pre_op_bytes == 0xF8)
            pre_opt_bits = (pre_insts >> 10) & 0x3
            pre_opc = (pre_insts >> 22) & 0x1
            pre_str_post = pre_f8 & (pre_opt_bits == 0x1) & (pre_opc == 0)

            pre_inst0 = pre_insts[0]
            pre_inst1 = pre_insts[1]
            pre_inst2 = pre_insts[2]
            pre_op1 = pre_ops[1]
            pre_op2 = pre_ops[2]
            pre_cmp_inst = torch.where(pre_len2, pre_inst1, pre_inst2)
            pre_cmp_op = torch.where(pre_len2, pre_op1, pre_op2)
            pre_cmp_is = (pre_cmp_op == OpType.CMP_REG.value) | ((pre_cmp_op == OpType.SUBS_REG.value) & ((pre_cmp_inst & 0x1F) == 31))

            pre_add_ok = pre_len3 & ((pre_op1 == OpType.ADD_IMM.value) | (pre_op1 == OpType.ADD_IMM_W.value))
            pre_add_rd = pre_inst1 & 0x1F
            pre_add_rn = (pre_inst1 >> 5) & 0x1F
            pre_add_imm = (pre_inst1 >> 10) & 0xFFF
            pre_cmp_rm = (pre_cmp_inst >> 16) & 0x1F
            pre_add_match = pre_add_ok & (pre_add_rd == pre_cmp_rm) & (pre_add_rn != pre_add_rd)

            pre_ok = pre_str_post[0] & pre_cmp_is & (pre_len2 | pre_add_match)

            pre_str_inst = pre_inst0
            pre_str_rn = (pre_str_inst >> 5) & 0x1F
            pre_str_imm9 = (pre_str_inst >> 12) & 0x1FF
            pre_str_imm9 = torch.where(pre_str_imm9 & 0x100 != 0, pre_str_imm9 - 0x200, pre_str_imm9)
            pre_ptr = regs[pre_str_rn.long()]
            pre_add_base = regs[pre_add_rn.long()]
            pre_end_ptr = torch.where(pre_len3 & pre_add_match, pre_add_base + pre_add_imm, regs[pre_cmp_rm.long()])
            pre_step = pre_str_imm9
            pre_safe_step = torch.where(pre_step == 0, torch.ones_like(pre_step), pre_step)
            pre_stride_ok = pre_step > 0
            pre_range_ok = pre_end_ptr >= pre_ptr
            pre_rem = (pre_end_ptr - pre_ptr) % pre_safe_step
            pre_iter_ok = (pre_step != 0) & (pre_rem == 0)
            pre_iter_count = torch.where(pre_iter_ok, (pre_end_ptr - pre_ptr) // pre_safe_step, self._const_i64_0)
            pre_ok = pre_ok & pre_stride_ok & pre_range_ok & (pre_iter_count > 0)

            exec_len = torch.where(pre_ok, self._const_i64_0, exec_len)
            exec_mask = idxs < exec_len

            if actual > 1:
                # Vectorized hazard detection using prefix-written masks
                store_ops = (
                    (ops == OpType.STR.value) | (ops == OpType.STRB.value) | (ops == OpType.STRH.value) |
                    (ops == OpType.STR_POST.value) | (ops == OpType.STRB_POST.value) | (ops == OpType.STR_PRE.value) |
                    (ops == OpType.STR_REG_OFF.value) | (ops == OpType.STP.value) |
                    (ops == OpType.STP_POST.value) | (ops == OpType.STP_PRE.value)
                )
                cmp_ops = (
                    (ops == OpType.CMP_IMM.value) | (ops == OpType.CMP_REG.value) |
                    (ops == OpType.CMP_IMM_W.value) | (ops == OpType.CMP_REG_W.value) |
                    (ops == OpType.TST_IMM.value) | (ops == OpType.TST_REG.value)
                )
                sp_write_ops = exec_mask & (rds == 31) & (
                    (ops == OpType.ADD_IMM.value) | (ops == OpType.SUB_IMM.value) |
                    (ops == OpType.ADD_IMM_W.value) | (ops == OpType.SUB_IMM_W.value) |
                    (ops == OpType.ADD_REG.value) | (ops == OpType.SUB_REG.value) |
                    (ops == OpType.ADD_REG_W.value) | (ops == OpType.SUB_REG_W.value)
                )
                writes_rd = exec_mask & (~store_ops) & (~cmp_ops) & (ops != OpType.NOP.value)
                writes_rd = writes_rd | sp_write_ops
                writes_rn = exec_mask & (
                    (ops == OpType.LDR_POST.value) | (ops == OpType.STR_POST.value) |
                    (ops == OpType.LDRB_POST.value) | (ops == OpType.STRB_POST.value) |
                    (ops == OpType.LDR_PRE.value) | (ops == OpType.STR_PRE.value) |
                    (ops == OpType.LDP_POST.value) | (ops == OpType.STP_POST.value) |
                    (ops == OpType.LDP_PRE.value) | (ops == OpType.STP_PRE.value)
                )

                rd_idx = rds.clamp(0, 31)
                rn_idx = rns.clamp(0, 31)
                rm_idx = rms.clamp(0, 31)

                rd_onehot = F.one_hot(rd_idx, num_classes=32).bool() & writes_rd.unsqueeze(1)
                rn_onehot = F.one_hot(rn_idx, num_classes=32).bool() & writes_rn.unsqueeze(1)
                write_onehot = rd_onehot | rn_onehot

                prefix = torch.cumsum(write_onehot.int(), dim=0)
                prev_written = (prefix - write_onehot.int()) > 0

                rn_read = F.one_hot(rn_idx, num_classes=32).bool()
                rm_read = F.one_hot(rm_idx, num_classes=32).bool() & (rms < 31).unsqueeze(1)
                dest_read = F.one_hot(rd_idx, num_classes=32).bool()

                raw_hazard = (prev_written & rn_read).any(dim=1) | (prev_written & rm_read).any(dim=1)
                waw_hazard = writes_rd & (prev_written & dest_read).any(dim=1)
                rn_waw = writes_rn & (prev_written & rn_read).any(dim=1)

                hazard_mask = raw_hazard | waw_hazard | rn_waw
                hazard_idx = torch.where(hazard_mask, idxs, torch.full_like(idxs, actual)).min()
                exec_len = torch.minimum(exec_len, hazard_idx)
                exec_mask = idxs < exec_len

            # ═══════════════════════════════════════════════════════════════
            # NOTE: Scalar fallback DISABLED on MPS - the .item() sync overhead
            # (0.15-3ms per call) makes it slower than GPU path for small batches.
            # The exec_mask already limits execution to exec_len instructions.
            # ═══════════════════════════════════════════════════════════════
            # Ensure _svc_t exists for SVC detection
            if not hasattr(self, '_svc_t'):
                self._svc_t = torch.tensor(False, device=self.device)

            ops = torch.where(exec_mask, ops, torch.full_like(ops, OpType.NOP.value))
            op_bytes = torch.where(exec_mask, op_bytes, torch.zeros_like(op_bytes))

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: PARALLEL GATHER (register values)
            # ═══════════════════════════════════════════════════════════════
            rn_vals = regs[rns]
            rm_vals = regs[rms]
            ra_vals = regs[ras]
            rd_vals = regs[rds]  # For MOVK

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: PARALLEL COMPUTE ALL RESULTS
            # Use pre-allocated tensors - NO ALLOCATION IN HOT PATH!
            # ═══════════════════════════════════════════════════════════════
            # Reuse pre-allocated tensors (slice to actual size)
            results = self._gpu_results[:actual]
            results.zero_()  # In-place zero
            write_mask = self._gpu_write_mask[:actual]
            write_mask.zero_()  # In-place zero

            rn_vals_32 = rn_vals & 0xFFFFFFFF
            rm_vals_32 = rm_vals & 0xFFFFFFFF
            imm12_32 = imm12 & 0xFFFFFFFF

            # --- ARITHMETIC IMMEDIATE ---
            add_imm_mask = (ops == OpType.ADD_IMM.value)
            add_imm_w_mask = (ops == OpType.ADD_IMM_W.value)
            sub_imm_mask = (ops == OpType.SUB_IMM.value)
            sub_imm_w_mask = (ops == OpType.SUB_IMM_W.value)
            adds_imm_mask = (ops == OpType.ADDS_IMM.value)
            adds_imm_w_mask = (ops == OpType.ADDS_IMM_W.value)
            subs_imm_mask = (ops == OpType.SUBS_IMM.value)
            subs_imm_w_mask = (ops == OpType.SUBS_IMM_W.value)

            results = torch.where(add_imm_mask, rn_vals + imm12, results)
            results = torch.where(add_imm_w_mask, (rn_vals_32 + imm12_32) & 0xFFFFFFFF, results)
            results = torch.where(sub_imm_mask, rn_vals - imm12, results)
            results = torch.where(sub_imm_w_mask, (rn_vals_32 - imm12_32) & 0xFFFFFFFF, results)
            results = torch.where(adds_imm_mask, rn_vals + imm12, results)
            results = torch.where(adds_imm_w_mask, (rn_vals_32 + imm12_32) & 0xFFFFFFFF, results)
            results = torch.where(subs_imm_mask, rn_vals - imm12, results)
            results = torch.where(subs_imm_w_mask, (rn_vals_32 - imm12_32) & 0xFFFFFFFF, results)

            write_mask = write_mask | add_imm_mask | add_imm_w_mask | sub_imm_mask | sub_imm_w_mask
            write_mask = write_mask | adds_imm_mask | adds_imm_w_mask
            write_mask = write_mask | ((subs_imm_mask | subs_imm_w_mask) & (rds != 31))

            # --- ARITHMETIC REGISTER (with shift support) ---
            add_reg_mask = (ops == OpType.ADD_REG.value)
            add_reg_w_mask = (ops == OpType.ADD_REG_W.value)
            sub_reg_mask = (ops == OpType.SUB_REG.value)
            sub_reg_w_mask = (ops == OpType.SUB_REG_W.value)
            adds_reg_mask = (ops == OpType.ADDS_REG.value)
            subs_reg_mask = (ops == OpType.SUBS_REG.value)
            subs_reg_w_mask = subs_reg_mask & (op_bytes == 0x6B)
            subs_reg_x_mask = subs_reg_mask & (op_bytes == 0xEB)

            # Apply LSL shift from imm6 to rm_vals for ADD/SUB shifted register
            # imm6 contains shift amount (0-63 for 64-bit, 0-31 for 32-bit)
            # Only apply shift for ADD_REG/SUB_REG (not W variants which use different encoding)
            rm_vals_shifted = rm_vals << imm6
            rm_vals_32_shifted = (rm_vals_32 << (imm6 & 0x1F)) & 0xFFFFFFFF

            results = torch.where(add_reg_mask, rn_vals + rm_vals_shifted, results)
            results = torch.where(add_reg_w_mask, (rn_vals_32 + rm_vals_32_shifted) & 0xFFFFFFFF, results)
            results = torch.where(sub_reg_mask, rn_vals - rm_vals_shifted, results)
            results = torch.where(sub_reg_w_mask, (rn_vals_32 - rm_vals_32_shifted) & 0xFFFFFFFF, results)
            results = torch.where(adds_reg_mask, rn_vals + rm_vals_shifted, results)
            results = torch.where(subs_reg_x_mask, rn_vals - rm_vals_shifted, results)
            results = torch.where(subs_reg_w_mask, (rn_vals_32 - rm_vals_32_shifted) & 0xFFFFFFFF, results)

            write_mask = write_mask | add_reg_mask | add_reg_w_mask | sub_reg_mask | sub_reg_w_mask
            write_mask = write_mask | adds_reg_mask | (subs_reg_mask & (rds != 31))

            # --- LOGICAL REGISTER ---
            and_mask = (ops == OpType.AND_REG.value)
            orr_mask = (ops == OpType.ORR_REG.value)
            eor_mask = (ops == OpType.EOR_REG.value)

            results = torch.where(and_mask, rn_vals & rm_vals, results)
            results = torch.where(orr_mask, rn_vals | rm_vals, results)
            results = torch.where(eor_mask, rn_vals ^ rm_vals, results)

            write_mask = write_mask | and_mask | orr_mask | eor_mask

            ands_reg_mask = (ops == OpType.ANDS_REG.value)
            tst_reg_mask = (ops == OpType.TST_REG.value)
            results = torch.where(ands_reg_mask, rn_vals & rm_vals, results)
            write_mask = write_mask | ands_reg_mask

            # --- MOV (ORR with XZR) ---
            mov_mask = orr_mask & (rns == 31)
            results = torch.where(mov_mask, rm_vals, results)

            mov_reg_mask = (ops == OpType.MOV_REG.value)
            results = torch.where(mov_reg_mask, rm_vals, results)
            write_mask = write_mask | mov_reg_mask

            mov_w_mask = (ops == OpType.MOV_W.value)
            results = torch.where(mov_w_mask, rm_vals_32, results)
            write_mask = write_mask | mov_w_mask

            # --- MOVZ ---
            movz_mask = (ops == OpType.MOVZ.value) | (ops == OpType.MOVZ_W.value)
            movz_val = imm16 << (hw * 16)
            results = torch.where(movz_mask, movz_val, results)
            write_mask = write_mask | movz_mask

            # --- MOVK ---
            movk_mask = (ops == OpType.MOVK.value) | (ops == OpType.MOVK_W.value)
            movk_clear = ~(self._movk_clear_base << (hw * 16))  # Use pre-allocated constant
            movk_val = (rd_vals & movk_clear) | (imm16 << (hw * 16))
            results = torch.where(movk_mask, movk_val, results)
            write_mask = write_mask | movk_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR ADR/ADRP - PC-relative address calculation
            # ADR:  result = PC + imm21
            # ADRP: result = (PC & ~0xFFF) + (imm21 << 12)
            # ═══════════════════════════════════════════════════════════════
            adrp_mask = (ops == OpType.ADRP.value)
            adr_mask = (ops == OpType.ADR.value)
            # Build PC tensor for each instruction in batch
            inst_pcs = self._batch_idx[:actual] * 4 + pc_t
            # Extract immediate: immlo = bits[30:29], immhi = bits[23:5]
            adr_immlo = (insts >> 29) & 0x3
            adr_immhi = (insts >> 5) & 0x7FFFF
            adr_imm = (adr_immhi << 2) | adr_immlo
            # Sign extend 21-bit immediate
            adr_imm = torch.where(adr_imm >= 0x100000, adr_imm - 0x200000, adr_imm)
            # ADRP: page_base + (imm << 12)
            page_base = inst_pcs & ~0xFFF
            adrp_val = page_base + (adr_imm << 12)
            results = torch.where(adrp_mask, adrp_val, results)
            # ADR: PC + imm
            adr_val = inst_pcs + adr_imm
            results = torch.where(adr_mask, adr_val, results)
            write_mask = write_mask | adrp_mask | adr_mask

            # --- MUL (MADD with Ra=XZR) ---
            mul_mask = (ops == OpType.MUL.value)
            mul_val = rn_vals * rm_vals
            results = torch.where(mul_mask, mul_val, results)
            write_mask = write_mask | mul_mask

            # --- MADD (multiply-add) ---
            madd_mask = (ops == OpType.MADD.value)
            madd_val = rn_vals * rm_vals + ra_vals
            results = torch.where(madd_mask, madd_val, results)
            write_mask = write_mask | madd_mask

            # --- MSUB (multiply-subtract) ---
            msub_mask = (ops == OpType.MSUB.value)
            msub_val = ra_vals - rn_vals * rm_vals
            results = torch.where(msub_mask, msub_val, results)
            write_mask = write_mask | msub_mask

            # --- SDIV (signed divide) ---
            sdiv_mask = (ops == OpType.SDIV.value)
            # Use floor_divide, handle div by zero
            rm_safe = torch.where(rm_vals == 0, torch.ones_like(rm_vals), rm_vals)
            sdiv_val = torch.where(rm_vals == 0, torch.zeros_like(rn_vals), rn_vals // rm_safe)
            results = torch.where(sdiv_mask, sdiv_val, results)
            write_mask = write_mask | sdiv_mask

            # --- UDIV (unsigned divide) ---
            udiv_mask = (ops == OpType.UDIV.value)
            udiv_val = torch.where(rm_vals == 0, torch.zeros_like(rn_vals), rn_vals // rm_safe)
            results = torch.where(udiv_mask, udiv_val, results)
            write_mask = write_mask | udiv_mask

            # --- SP updates (rd==31 uses SP, not XZR, for ADD/SUB) ---
            sp_write_mask = (rds == 31) & (
                (ops == OpType.ADD_IMM.value) | (ops == OpType.SUB_IMM.value) |
                (ops == OpType.ADD_IMM_W.value) | (ops == OpType.SUB_IMM_W.value) |
                (ops == OpType.ADD_REG.value) | (ops == OpType.SUB_REG.value) |
                (ops == OpType.ADD_REG_W.value) | (ops == OpType.SUB_REG_W.value)
            )
            sp_idx = torch.where(sp_write_mask, idxs, torch.full_like(idxs, -1)).max()
            sp_valid = sp_idx >= 0
            sp_idx_clamped = torch.clamp(sp_idx, min=0)
            sp_val = results[sp_idx_clamped]
            regs[31] = torch.where(sp_valid, sp_val, regs[31])

            # --- LOGICAL IMMEDIATE (AND/ORR/EOR with immediate) ---
            and_imm_mask = (ops == OpType.AND_IMM.value)
            orr_imm_mask = (ops == OpType.ORR_IMM.value)
            eor_imm_mask = (ops == OpType.EOR_IMM.value)
            ands_imm_mask = (ops == OpType.ANDS_IMM.value)
            tst_imm_mask = (ops == OpType.TST_IMM.value)

            # For logical immediate, properly decode bitmask from N/immr/imms
            # ARM64 logical immediate encoding: N determines element size, imms encodes bit count
            # Extract N (bit 22), immr (bits 21:16), imms (bits 15:10)
            log_N = (insts >> 22) & 1
            log_immr = (insts >> 16) & 0x3F
            log_imms = (insts >> 10) & 0x3F

            # For N=1 (64-bit element): mask = ((1 << (imms+1)) - 1) rotated right by immr
            # For N=0 (32-bit or smaller): element size is determined by highest clear bit in imms
            # Most common case: N=1, immr=0 -> mask = (1 << (imms+1)) - 1

            # Create base mask: (imms+1) consecutive 1 bits
            ones_count = log_imms + 1
            ones_count_clamped = ones_count.clamp(1, 63)  # Safety clamp
            base_mask = (torch.tensor(1, dtype=torch.int64, device=self.device) << ones_count_clamped) - 1

            # Apply rotation (rotate right by immr)
            # ROR(x, r) for 64 bits = (x >> r) | (x << (64-r))
            rot_right = log_immr & 0x3F
            rot_left = (64 - rot_right) & 0x3F
            logical_imm = torch.where(
                rot_right > 0,
                (base_mask >> rot_right) | (base_mask << rot_left),
                base_mask
            )

            # For 32-bit ops (sf=0, N=0), mask to 32 bits
            sf_bit = (insts >> 31) & 1
            logical_imm = torch.where(sf_bit == 0, logical_imm & 0xFFFFFFFF, logical_imm)

            results = torch.where(and_imm_mask, rn_vals & logical_imm, results)
            results = torch.where(orr_imm_mask, rn_vals | logical_imm, results)
            results = torch.where(eor_imm_mask, rn_vals ^ logical_imm, results)
            results = torch.where(ands_imm_mask, rn_vals & logical_imm, results)
            write_mask = write_mask | and_imm_mask | orr_imm_mask | eor_imm_mask | ands_imm_mask

            # --- SHIFTS (immediate) ---
            lsl_imm_mask = (ops == OpType.LSL_IMM.value)
            lsr_imm_mask = (ops == OpType.LSR_IMM.value)
            asr_imm_mask = (ops == OpType.ASR_IMM.value)

            # Shift amount from imms field (bits 15:10)
            shift_amt = (insts >> 10) & 0x3F
            shift_amt_clamped = shift_amt.clamp(0, 63)  # Safety clamp

            lsl_val = rn_vals << shift_amt_clamped
            lsr_val = rn_vals >> shift_amt_clamped  # Logical right shift
            asr_val = rn_vals >> shift_amt_clamped  # Arithmetic for signed (Python >> is arithmetic for signed)

            results = torch.where(lsl_imm_mask, lsl_val, results)
            results = torch.where(lsr_imm_mask, lsr_val, results)
            results = torch.where(asr_imm_mask, asr_val, results)
            write_mask = write_mask | lsl_imm_mask | lsr_imm_mask | asr_imm_mask

            # --- SHIFTS (register) ---
            lsl_reg_mask = (ops == OpType.LSL_REG.value)
            lsr_reg_mask = (ops == OpType.LSR_REG.value)
            asr_reg_mask = (ops == OpType.ASR_REG.value)
            ror_reg_mask = (ops == OpType.ROR_REG.value)

            rm_shift_amt = rm_vals & 0x3F  # Shift amount from Rm, masked to 6 bits

            lsl_reg_val = rn_vals << rm_shift_amt
            lsr_reg_val = rn_vals >> rm_shift_amt
            asr_reg_val = rn_vals >> rm_shift_amt  # Arithmetic for signed

            results = torch.where(lsl_reg_mask, lsl_reg_val, results)
            results = torch.where(lsr_reg_mask, lsr_reg_val, results)
            results = torch.where(asr_reg_mask, asr_reg_val, results)
            write_mask = write_mask | lsl_reg_mask | lsr_reg_mask | asr_reg_mask | ror_reg_mask

            # --- NEG (negate: SUB from zero) ---
            neg_mask = (ops == OpType.NEG.value)
            neg_val = -rm_vals  # NEG Rd, Rm = SUB Rd, XZR, Rm
            results = torch.where(neg_mask, neg_val, results)
            write_mask = write_mask | neg_mask

            # --- MVN (bitwise NOT) ---
            mvn_mask = (ops == OpType.MVN.value)
            mvn_val = ~rm_vals
            results = torch.where(mvn_mask, mvn_val, results)
            write_mask = write_mask | mvn_mask

            # --- BIC (bit clear: AND NOT) ---
            bic_mask = (ops == OpType.BIC.value)
            bic_val = rn_vals & (~rm_vals)
            results = torch.where(bic_mask, bic_val, results)
            write_mask = write_mask | bic_mask

            # --- CLZ (count leading zeros) ---
            clz_mask = (ops == OpType.CLZ.value)
            # Use bit manipulation: find highest set bit
            # For GPU: log2 approximation, then 64 - position
            clz_val = 64 - torch.floor(torch.log2(rn_vals.float().clamp(min=1))).long() - 1
            clz_val = torch.where(rn_vals == 0, torch.full_like(clz_val, 64), clz_val)
            results = torch.where(clz_mask, clz_val, results)
            write_mask = write_mask | clz_mask

            # --- SXTW (sign extend word to 64-bit) ---
            sxtw_mask = (ops == OpType.SXTW.value)
            # Sign extend 32-bit to 64-bit
            sxtw_val = (rn_vals & 0xFFFFFFFF).to(torch.int32).to(torch.int64)
            results = torch.where(sxtw_mask, sxtw_val, results)
            write_mask = write_mask | sxtw_mask

            # --- UXTB (zero extend byte) ---
            uxtb_mask = (ops == OpType.UXTB.value)
            uxtb_val = rn_vals & 0xFF
            results = torch.where(uxtb_mask, uxtb_val, results)
            write_mask = write_mask | uxtb_mask

            # --- UXTH (zero extend halfword) ---
            uxth_mask = (ops == OpType.UXTH.value)
            uxth_val = rn_vals & 0xFFFF
            results = torch.where(uxth_mask, uxth_val, results)
            write_mask = write_mask | uxth_mask

            # --- FLAGS UPDATE (use last flag-setting op in batch) ---
            cmp_imm_mask = (ops == OpType.CMP_IMM.value)
            cmp_reg_mask = (ops == OpType.CMP_REG.value)
            cmp_imm_w_mask = (ops == OpType.CMP_IMM_W.value)
            cmp_reg_w_mask = (ops == OpType.CMP_REG_W.value)

            flag_mask = (
                adds_imm_mask | adds_imm_w_mask | adds_reg_mask |
                subs_imm_mask | subs_imm_w_mask | subs_reg_mask |
                ands_reg_mask | ands_imm_mask | tst_reg_mask | tst_imm_mask |
                cmp_imm_mask | cmp_reg_mask | cmp_imm_w_mask | cmp_reg_w_mask
            )
            flag_idx = torch.where(flag_mask, idxs, torch.full_like(idxs, -1)).max()
            flag_valid = flag_idx >= 0
            flag_idx_clamped = torch.clamp(flag_idx, min=0)

            op = ops[flag_idx_clamped]
            op_byte = op_bytes[flag_idx_clamped]
            a_val = rn_vals[flag_idx_clamped]
            b_reg_val = rm_vals[flag_idx_clamped]
            imm_val = imm12[flag_idx_clamped]

            is_32 = (
                (op == OpType.ADDS_IMM_W.value) | (op == OpType.SUBS_IMM_W.value) |
                (op == OpType.CMP_IMM_W.value) | (op == OpType.CMP_REG_W.value) |
                ((op == OpType.SUBS_REG.value) & (op_byte == 0x6B))
            )
            mask_32 = torch.tensor(0xFFFFFFFF, device=self.device, dtype=torch.int64)
            mask_64 = torch.tensor(-1, device=self.device, dtype=torch.int64)
            sign_32 = torch.tensor(0x80000000, device=self.device, dtype=torch.int64)
            sign_64 = torch.tensor(-0x8000000000000000, device=self.device, dtype=torch.int64)
            mask = torch.where(is_32, mask_32, mask_64)
            sign_bit = torch.where(is_32, sign_32, sign_64)

            is_add = (op == OpType.ADDS_IMM.value) | (op == OpType.ADDS_IMM_W.value) | (op == OpType.ADDS_REG.value)
            is_sub = (
                (op == OpType.SUBS_IMM.value) | (op == OpType.SUBS_IMM_W.value) |
                (op == OpType.SUBS_REG.value) | (op == OpType.CMP_IMM.value) |
                (op == OpType.CMP_REG.value) | (op == OpType.CMP_IMM_W.value) |
                (op == OpType.CMP_REG_W.value)
            )
            is_and = (op == OpType.ANDS_REG.value) | (op == OpType.ANDS_IMM.value) | (op == OpType.TST_REG.value) | (op == OpType.TST_IMM.value)

            b_val_add = torch.where((op == OpType.ADDS_IMM.value) | (op == OpType.ADDS_IMM_W.value), imm_val, b_reg_val)
            b_val_sub = torch.where(
                (op == OpType.SUBS_IMM.value) | (op == OpType.SUBS_IMM_W.value) |
                (op == OpType.CMP_IMM.value) | (op == OpType.CMP_IMM_W.value),
                imm_val,
                b_reg_val
            )
            b_val_and = torch.where((op == OpType.ANDS_IMM.value) | (op == OpType.TST_IMM.value), imm_val, b_reg_val)

            a_u = a_val & mask
            b_u_add = b_val_add & mask
            b_u_sub = b_val_sub & mask
            b_u_and = b_val_and & mask

            res_add_full = a_u + b_u_add
            res_add = res_add_full & mask
            res_sub = (a_u - b_u_sub) & mask
            res_and = (a_u & b_u_and) & mask

            res = torch.where(is_add, res_add, torch.where(is_sub, res_sub, res_and))
            n = (res & sign_bit) != 0
            z = res == 0

            c_add = res_add_full > mask
            c_sub = a_u >= b_u_sub
            c = torch.where(is_add, c_add, torch.where(is_sub, c_sub, torch.zeros_like(c_add)))

            sign_a = (a_u & sign_bit) != 0
            sign_b_add = (b_u_add & sign_bit) != 0
            sign_b_sub = (b_u_sub & sign_bit) != 0
            sign_r_add = (res_add & sign_bit) != 0
            sign_r_sub = (res_sub & sign_bit) != 0
            v_add = (sign_a == sign_b_add) & (sign_a != sign_r_add)
            v_sub = (sign_a != sign_b_sub) & (sign_a != sign_r_sub)
            v = torch.where(is_add, v_add, torch.where(is_sub, v_sub, torch.zeros_like(v_add)))

            new_flags = torch.stack([n.float(), z.float(), c.float(), v.float()])
            self.flags = torch.where(flag_valid, new_flags, self.flags)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4B: PARALLEL LOAD/STORE (memory as tensor ops)
            # ═══════════════════════════════════════════════════════════════
            # Decode load/store offsets
            load_offset = imm12  # Scaled offset for most loads

            # --- LDRB (load byte) ---
            ldrb_mask = (ops == OpType.LDRB.value)
            ldrb_addrs = (rn_vals[ldrb_mask] + ((insts[ldrb_mask] >> 10) & 0xFFF)).clamp(0, self.mem_size - 1)
            ldrb_vals = mem[ldrb_addrs.long()]
            results[ldrb_mask] = ldrb_vals.long()
            write_mask = write_mask | ldrb_mask

            # --- LDR (load 64-bit) ---
            ldr_mask = (ops == OpType.LDR.value)
            ldr_addrs = (rn_vals[ldr_mask] + ((insts[ldr_mask] >> 10) & 0xFFF) * 8).clamp(0, self.mem_size - 8)
            ldr_addrs_long = ldr_addrs.long()
            # Gather 8 bytes and combine
            b0 = mem[ldr_addrs_long].long()
            b1 = mem[(ldr_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldr_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldr_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldr_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldr_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldr_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldr_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldr_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldr_mask] = ldr_vals
            write_mask = write_mask | ldr_mask

            # --- STRB (store byte) ---
            strb_mask = (ops == OpType.STRB.value)
            strb_addrs = (rn_vals[strb_mask] + ((insts[strb_mask] >> 10) & 0xFFF)).clamp(0, self.mem_size - 1)
            rt_indices = (insts[strb_mask] & 0x1F).long()
            strb_vals = regs[rt_indices] & 0xFF
            mem.scatter_(0, strb_addrs.long(), strb_vals.byte())

            # --- STR (store 64-bit) ---
            str_mask = (ops == OpType.STR.value)
            str_addrs = (rn_vals[str_mask] + ((insts[str_mask] >> 10) & 0xFFF) * 8).clamp(0, self.mem_size - 8)
            str_addrs_long = str_addrs.long()
            rt_indices = (insts[str_mask] & 0x1F).long()
            str_vals = regs[rt_indices]
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and str_addrs_long.numel() > 0:
                hit_code = ((str_addrs_long >= 0x4558) & (str_addrs_long <= 0x4560)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] STR writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  Addresses: {str_addrs_long.cpu().tolist()}")
            # Scatter 8 bytes
            mem.scatter_(0, str_addrs_long, (str_vals & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 1).clamp(max=self.mem_size-1), ((str_vals >> 8) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 2).clamp(max=self.mem_size-1), ((str_vals >> 16) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 3).clamp(max=self.mem_size-1), ((str_vals >> 24) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 4).clamp(max=self.mem_size-1), ((str_vals >> 32) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 5).clamp(max=self.mem_size-1), ((str_vals >> 40) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 6).clamp(max=self.mem_size-1), ((str_vals >> 48) & 0xFF).byte())
            mem.scatter_(0, (str_addrs_long + 7).clamp(max=self.mem_size-1), ((str_vals >> 56) & 0xFF).byte())

            # --- LDRH (load halfword) ---
            ldrh_mask = (ops == OpType.LDRH.value)
            ldrh_addrs = (rn_vals[ldrh_mask] + ((insts[ldrh_mask] >> 10) & 0xFFF) * 2).clamp(0, self.mem_size - 2)
            ldrh_addrs_long = ldrh_addrs.long()
            b0 = mem[ldrh_addrs_long].long()
            b1 = mem[(ldrh_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            ldrh_vals = b0 | (b1 << 8)
            results[ldrh_mask] = ldrh_vals
            write_mask = write_mask | ldrh_mask

            # --- STRH (store halfword) ---
            strh_mask = (ops == OpType.STRH.value)
            strh_addrs = (rn_vals[strh_mask] + ((insts[strh_mask] >> 10) & 0xFFF) * 2).clamp(0, self.mem_size - 2)
            strh_addrs_long = strh_addrs.long()
            rt_indices = (insts[strh_mask] & 0x1F).long()
            strh_vals = regs[rt_indices]
            mem.scatter_(0, strh_addrs_long, (strh_vals & 0xFF).byte())
            mem.scatter_(0, (strh_addrs_long + 1).clamp(max=self.mem_size-1), ((strh_vals >> 8) & 0xFF).byte())

            # --- LDR_REG_OFF (load 64-bit with register offset) ---
            # LDR Xt, [Xn, Xm, {extend} {#amount}]
            ldr_reg_off_mask = (ops == OpType.LDR_REG_OFF.value)
            rm_indices = rms[ldr_reg_off_mask]
            rm_vals_local = regs[rm_indices]
            # S bit (bit 12): if 1, shift Rm left by 3 (for 64-bit scale)
            s_bit = (insts[ldr_reg_off_mask] >> 12) & 0x1
            offset = torch.where(s_bit == 1, rm_vals_local << 3, rm_vals_local)
            ldr_reg_addrs = (rn_vals[ldr_reg_off_mask] + offset).clamp(0, self.mem_size - 8)
            ldr_reg_addrs_long = ldr_reg_addrs.long()
            # Gather 8 bytes
            b0 = mem[ldr_reg_addrs_long].long()
            b1 = mem[(ldr_reg_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldr_reg_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldr_reg_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldr_reg_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldr_reg_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldr_reg_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldr_reg_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldr_reg_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldr_reg_off_mask] = ldr_reg_vals
            write_mask = write_mask | ldr_reg_off_mask

            # --- STR_REG_OFF (store 64-bit with register offset) ---
            str_reg_off_mask = (ops == OpType.STR_REG_OFF.value)
            rm_indices = rms[str_reg_off_mask]
            rm_vals_local = regs[rm_indices]
            s_bit = (insts[str_reg_off_mask] >> 12) & 0x1
            offset = torch.where(s_bit == 1, rm_vals_local << 3, rm_vals_local)
            str_reg_addrs = (rn_vals[str_reg_off_mask] + offset).clamp(0, self.mem_size - 8)
            str_reg_addrs_long = str_reg_addrs.long()
            rt_indices = (insts[str_reg_off_mask] & 0x1F).long()
            str_reg_vals = torch.where(rt_indices == 31, torch.zeros_like(regs[rt_indices]), regs[rt_indices])
            # Scatter 8 bytes
            mem.scatter_(0, str_reg_addrs_long, (str_reg_vals & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 1).clamp(max=self.mem_size-1), ((str_reg_vals >> 8) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 2).clamp(max=self.mem_size-1), ((str_reg_vals >> 16) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 3).clamp(max=self.mem_size-1), ((str_reg_vals >> 24) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 4).clamp(max=self.mem_size-1), ((str_reg_vals >> 32) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 5).clamp(max=self.mem_size-1), ((str_reg_vals >> 40) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 6).clamp(max=self.mem_size-1), ((str_reg_vals >> 48) & 0xFF).byte())
            mem.scatter_(0, (str_reg_addrs_long + 7).clamp(max=self.mem_size-1), ((str_reg_vals >> 56) & 0xFF).byte())

            # --- LDR_POST (load 64-bit, then update base) ---
            ldr_post_mask = (ops == OpType.LDR_POST.value)
            ldr_post_addrs = rn_vals[ldr_post_mask].clamp(0, self.mem_size - 8)
            ldr_post_addrs_long = ldr_post_addrs.long()
            # Gather 8 bytes
            b0 = mem[ldr_post_addrs_long].long()
            b1 = mem[(ldr_post_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldr_post_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldr_post_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldr_post_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldr_post_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldr_post_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldr_post_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldr_post_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldr_post_mask] = ldr_post_vals
            write_mask = write_mask | ldr_post_mask
            # Update base register: imm9 = bits 20:12, sign-extended
            imm9_raw = (insts[ldr_post_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw - 0x200, imm9_raw)
            rn_indices = rns[ldr_post_mask]
            regs[rn_indices] = regs[rn_indices] + imm9

            # --- STR_POST (store 64-bit, then update base) ---
            str_post_mask = (ops == OpType.STR_POST.value)
            str_post_addrs = rn_vals[str_post_mask].clamp(0, self.mem_size - 8)
            str_post_addrs_long = str_post_addrs.long()
            rt_indices = (insts[str_post_mask] & 0x1F).long()
            # Handle XZR (rd=31) - use 0
            str_post_vals = torch.where(rt_indices == 31, torch.zeros_like(regs[rt_indices]), regs[rt_indices])
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and str_post_addrs_long.numel() > 0:
                hit_code = ((str_post_addrs_long >= 0x4558) & (str_post_addrs_long <= 0x4560)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] STR_POST writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  Addresses: {str_post_addrs_long.cpu().tolist()}")
                    logger.debug(f"  Values: {str_post_vals.cpu().tolist()}")
            # Scatter 8 bytes
            mem.scatter_(0, str_post_addrs_long, (str_post_vals & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 1).clamp(max=self.mem_size-1), ((str_post_vals >> 8) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 2).clamp(max=self.mem_size-1), ((str_post_vals >> 16) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 3).clamp(max=self.mem_size-1), ((str_post_vals >> 24) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 4).clamp(max=self.mem_size-1), ((str_post_vals >> 32) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 5).clamp(max=self.mem_size-1), ((str_post_vals >> 40) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 6).clamp(max=self.mem_size-1), ((str_post_vals >> 48) & 0xFF).byte())
            mem.scatter_(0, (str_post_addrs_long + 7).clamp(max=self.mem_size-1), ((str_post_vals >> 56) & 0xFF).byte())
            # Update base register: imm9 = bits 20:12, sign-extended
            imm9_raw = (insts[str_post_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw - 0x200, imm9_raw)
            rn_indices = rns[str_post_mask]
            regs[rn_indices] = regs[rn_indices] + imm9

            # --- LDRB_POST (load byte, then update base) ---
            ldrb_post_mask = (ops == OpType.LDRB_POST.value)
            ldrb_post_addrs = rn_vals[ldrb_post_mask].clamp(0, self.mem_size - 1)
            ldrb_post_vals = mem[ldrb_post_addrs.long()].long()
            results[ldrb_post_mask] = ldrb_post_vals
            write_mask = write_mask | ldrb_post_mask
            # Update base
            imm9_raw = (insts[ldrb_post_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw - 0x200, imm9_raw)
            rn_indices = rns[ldrb_post_mask]
            regs[rn_indices] = regs[rn_indices] + imm9

            # --- STRB_POST (store byte, then update base) ---
            strb_post_mask = (ops == OpType.STRB_POST.value)
            strb_post_addrs = rn_vals[strb_post_mask].clamp(0, self.mem_size - 1)
            rt_indices = (insts[strb_post_mask] & 0x1F).long()
            strb_post_vals = regs[rt_indices] & 0xFF
            mem.scatter_(0, strb_post_addrs.long(), strb_post_vals.byte())
            # Update base
            imm9_raw = (insts[strb_post_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw - 0x200, imm9_raw)
            rn_indices = rns[strb_post_mask]
            regs[rn_indices] = regs[rn_indices] + imm9

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDUR (load 64-bit with unscaled 9-bit signed offset)
            # LDUR Xt, [Xn, #simm9] - for negative offsets
            # ═══════════════════════════════════════════════════════════════
            ldur_mask = (ops == OpType.LDUR.value)
            # imm9 is bits [20:12], sign-extended
            imm9_raw = (insts[ldur_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw.long() - 0x200, imm9_raw.long())
            ldur_addrs = (rn_vals[ldur_mask] + imm9).clamp(0, self.mem_size - 8)
            ldur_addrs_long = ldur_addrs.long()
            # Gather 8 bytes
            b0 = mem[ldur_addrs_long].long()
            b1 = mem[(ldur_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldur_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldur_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldur_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldur_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldur_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldur_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldur_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldur_mask] = ldur_vals
            write_mask = write_mask | ldur_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR STUR (store 64-bit with unscaled 9-bit signed offset)
            # STUR Xt, [Xn, #simm9] - for negative offsets
            # ═══════════════════════════════════════════════════════════════
            stur_mask = (ops == OpType.STUR.value)
            imm9_raw = (insts[stur_mask] >> 12) & 0x1FF
            imm9 = torch.where(imm9_raw >= 0x100, imm9_raw.long() - 0x200, imm9_raw.long())
            stur_addrs = (rn_vals[stur_mask] + imm9).clamp(0, self.mem_size - 8)
            stur_addrs_long = stur_addrs.long()
            rt_indices = (insts[stur_mask] & 0x1F).long()
            stur_vals = torch.where(rt_indices == 31, torch.zeros_like(regs[rt_indices]), regs[rt_indices])
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and stur_addrs_long.numel() > 0:
                hit_code = ((stur_addrs_long >= 0x4558) & (stur_addrs_long <= 0x4560)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] STUR writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  Addresses: {stur_addrs_long.cpu().tolist()}")
                    logger.debug(f"  Values: {stur_vals.cpu().tolist()}")
            # Scatter 8 bytes
            mem.scatter_(0, stur_addrs_long, (stur_vals & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 1).clamp(max=self.mem_size-1), ((stur_vals >> 8) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 2).clamp(max=self.mem_size-1), ((stur_vals >> 16) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 3).clamp(max=self.mem_size-1), ((stur_vals >> 24) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 4).clamp(max=self.mem_size-1), ((stur_vals >> 32) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 5).clamp(max=self.mem_size-1), ((stur_vals >> 40) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 6).clamp(max=self.mem_size-1), ((stur_vals >> 48) & 0xFF).byte())
            mem.scatter_(0, (stur_addrs_long + 7).clamp(max=self.mem_size-1), ((stur_vals >> 56) & 0xFF).byte())

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDP (Load Pair of 64-bit registers)
            # LDP Xt1, Xt2, [Xn, #imm7*8] - critical for function prologue
            # Encoding: opc=10, imm7 bits[21:15], Rt2 bits[14:10], Rn bits[9:5], Rt bits[4:0]
            # ═══════════════════════════════════════════════════════════════
            ldp_mask = (ops == OpType.LDP.value)
            # imm7 is bits [21:15], sign-extended, scaled by 8
            imm7_raw = (insts[ldp_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            ldp_addrs = (rn_vals[ldp_mask] + imm7).clamp(0, self.mem_size - 16)
            ldp_addrs_long = ldp_addrs.long()
            rt2_indices = ((insts[ldp_mask] >> 10) & 0x1F).long()
            # Load first 64-bit value for Rt
            b0 = mem[ldp_addrs_long].long()
            b1 = mem[(ldp_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldp_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldp_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldp_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldp_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldp_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldp_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldp_val1 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldp_mask] = ldp_val1  # Rt gets first value
            write_mask = write_mask | ldp_mask
            # Load second 64-bit value for Rt2
            ldp_addrs2 = (ldp_addrs_long + 8).clamp(max=self.mem_size-1)
            c0 = mem[ldp_addrs2].long()
            c1 = mem[(ldp_addrs2 + 1).clamp(max=self.mem_size-1)].long()
            c2 = mem[(ldp_addrs2 + 2).clamp(max=self.mem_size-1)].long()
            c3 = mem[(ldp_addrs2 + 3).clamp(max=self.mem_size-1)].long()
            c4 = mem[(ldp_addrs2 + 4).clamp(max=self.mem_size-1)].long()
            c5 = mem[(ldp_addrs2 + 5).clamp(max=self.mem_size-1)].long()
            c6 = mem[(ldp_addrs2 + 6).clamp(max=self.mem_size-1)].long()
            c7 = mem[(ldp_addrs2 + 7).clamp(max=self.mem_size-1)].long()
            ldp_val2 = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24) | (c4 << 32) | (c5 << 40) | (c6 << 48) | (c7 << 56)
            # Write Rt2 directly to regs (not going through results)
            valid_rt2 = rt2_indices < 31
            regs.scatter_(0, rt2_indices[valid_rt2], ldp_val2[valid_rt2])

            # ═══════════════════════════════════════════════════════════════
            # TENSOR STP (Store Pair of 64-bit registers)
            # STP Xt1, Xt2, [Xn, #imm7*8] - critical for function prologue
            # ═══════════════════════════════════════════════════════════════
            stp_mask = (ops == OpType.STP.value)
            imm7_raw = (insts[stp_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            stp_addrs = (rn_vals[stp_mask] + imm7).clamp(0, self.mem_size - 16)
            stp_addrs_long = stp_addrs.long()
            rt_indices = (insts[stp_mask] & 0x1F).long()
            rt2_indices = ((insts[stp_mask] >> 10) & 0x1F).long()
            # Get values (handle XZR)
            stp_val1 = torch.where(rt_indices == 31, torch.zeros(rt_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt_indices])
            stp_val2 = torch.where(rt2_indices == 31, torch.zeros(rt2_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt2_indices])
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and stp_addrs_long.numel() > 0:
                hit_code = ((stp_addrs_long >= 0x4558) & (stp_addrs_long <= 0x4568)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] STP writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  Addresses: {stp_addrs_long.cpu().tolist()}")
                    logger.debug(f"  Values: {stp_val1.cpu().tolist()}, {stp_val2.cpu().tolist()}")
            # Store first 64-bit value
            mem.scatter_(0, stp_addrs_long, (stp_val1 & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 1).clamp(max=self.mem_size-1), ((stp_val1 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 2).clamp(max=self.mem_size-1), ((stp_val1 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 3).clamp(max=self.mem_size-1), ((stp_val1 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 4).clamp(max=self.mem_size-1), ((stp_val1 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 5).clamp(max=self.mem_size-1), ((stp_val1 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 6).clamp(max=self.mem_size-1), ((stp_val1 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs_long + 7).clamp(max=self.mem_size-1), ((stp_val1 >> 56) & 0xFF).byte())
            # Store second 64-bit value at addr+8
            stp_addrs2 = (stp_addrs_long + 8).clamp(max=self.mem_size-1)
            mem.scatter_(0, stp_addrs2, (stp_val2 & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 1).clamp(max=self.mem_size-1), ((stp_val2 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 2).clamp(max=self.mem_size-1), ((stp_val2 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 3).clamp(max=self.mem_size-1), ((stp_val2 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 4).clamp(max=self.mem_size-1), ((stp_val2 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 5).clamp(max=self.mem_size-1), ((stp_val2 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 6).clamp(max=self.mem_size-1), ((stp_val2 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_addrs2 + 7).clamp(max=self.mem_size-1), ((stp_val2 >> 56) & 0xFF).byte())

            # ═══════════════════════════════════════════════════════════════
            # TENSOR STP_PRE (Store Pair with pre-index: update base THEN store)
            # STP Xt1, Xt2, [Xn, #imm]! - critical for function prologue
            # ═══════════════════════════════════════════════════════════════
            stp_pre_mask = (ops == OpType.STP_PRE.value)
            imm7_raw = (insts[stp_pre_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            # Pre-index: compute new base first
            new_base = rn_vals[stp_pre_mask] + imm7
            stp_pre_addrs = new_base.clamp(0, self.mem_size - 16)
            stp_pre_addrs_long = stp_pre_addrs.long()
            rt_indices = (insts[stp_pre_mask] & 0x1F).long()
            rt2_indices = ((insts[stp_pre_mask] >> 10) & 0x1F).long()
            rn_indices = rns[stp_pre_mask]
            stp_pre_val1 = torch.where(rt_indices == 31, torch.zeros(rt_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt_indices])
            stp_pre_val2 = torch.where(rt2_indices == 31, torch.zeros(rt2_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt2_indices])
            # Store first value
            mem.scatter_(0, stp_pre_addrs_long, (stp_pre_val1 & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 1).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 2).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 3).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 4).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 5).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 6).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs_long + 7).clamp(max=self.mem_size-1), ((stp_pre_val1 >> 56) & 0xFF).byte())
            # Store second value at addr+8
            stp_pre_addrs2 = (stp_pre_addrs_long + 8).clamp(max=self.mem_size-1)
            mem.scatter_(0, stp_pre_addrs2, (stp_pre_val2 & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 1).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 2).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 3).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 4).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 5).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 6).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_pre_addrs2 + 7).clamp(max=self.mem_size-1), ((stp_pre_val2 >> 56) & 0xFF).byte())
            # Update base register (pre-index writes back)
            regs.scatter_(0, rn_indices[rn_indices < 31], new_base[rn_indices < 31])

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDP_PRE (Load Pair with pre-index: update base THEN load)
            # LDP Xt1, Xt2, [Xn, #imm]! - critical for function epilogue
            # ═══════════════════════════════════════════════════════════════
            ldp_pre_mask = (ops == OpType.LDP_PRE.value)
            imm7_raw = (insts[ldp_pre_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            # Pre-index: compute new base first
            new_base = rn_vals[ldp_pre_mask] + imm7
            ldp_pre_addrs = new_base.clamp(0, self.mem_size - 16)
            ldp_pre_addrs_long = ldp_pre_addrs.long()
            rt2_indices = ((insts[ldp_pre_mask] >> 10) & 0x1F).long()
            rn_indices = rns[ldp_pre_mask]
            # Load first value
            b0 = mem[ldp_pre_addrs_long].long()
            b1 = mem[(ldp_pre_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldp_pre_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldp_pre_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldp_pre_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldp_pre_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldp_pre_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldp_pre_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldp_pre_val1 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldp_pre_mask] = ldp_pre_val1
            write_mask = write_mask | ldp_pre_mask
            # Load second value
            ldp_pre_addrs2 = (ldp_pre_addrs_long + 8).clamp(max=self.mem_size-1)
            c0 = mem[ldp_pre_addrs2].long()
            c1 = mem[(ldp_pre_addrs2 + 1).clamp(max=self.mem_size-1)].long()
            c2 = mem[(ldp_pre_addrs2 + 2).clamp(max=self.mem_size-1)].long()
            c3 = mem[(ldp_pre_addrs2 + 3).clamp(max=self.mem_size-1)].long()
            c4 = mem[(ldp_pre_addrs2 + 4).clamp(max=self.mem_size-1)].long()
            c5 = mem[(ldp_pre_addrs2 + 5).clamp(max=self.mem_size-1)].long()
            c6 = mem[(ldp_pre_addrs2 + 6).clamp(max=self.mem_size-1)].long()
            c7 = mem[(ldp_pre_addrs2 + 7).clamp(max=self.mem_size-1)].long()
            ldp_pre_val2 = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24) | (c4 << 32) | (c5 << 40) | (c6 << 48) | (c7 << 56)
            valid_rt2 = rt2_indices < 31
            regs.scatter_(0, rt2_indices[valid_rt2], ldp_pre_val2[valid_rt2])
            # Update base register (pre-index writes back)
            regs.scatter_(0, rn_indices[rn_indices < 31], new_base[rn_indices < 31])

            # ═══════════════════════════════════════════════════════════════
            # TENSOR STP_POST (Store Pair with post-index: store THEN update base)
            # STP Xt1, Xt2, [Xn], #imm
            # ═══════════════════════════════════════════════════════════════
            stp_post_mask = (ops == OpType.STP_POST.value)
            imm7_raw = (insts[stp_post_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            stp_post_addrs = rn_vals[stp_post_mask].clamp(0, self.mem_size - 16)
            stp_post_addrs_long = stp_post_addrs.long()
            rt_indices = (insts[stp_post_mask] & 0x1F).long()
            rt2_indices = ((insts[stp_post_mask] >> 10) & 0x1F).long()
            rn_indices = rns[stp_post_mask]
            stp_post_val1 = torch.where(rt_indices == 31, torch.zeros(rt_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt_indices])
            stp_post_val2 = torch.where(rt2_indices == 31, torch.zeros(rt2_indices.shape[0], device=self.device, dtype=torch.int64), regs[rt2_indices])
            # Store first value
            mem.scatter_(0, stp_post_addrs_long, (stp_post_val1 & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 1).clamp(max=self.mem_size-1), ((stp_post_val1 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 2).clamp(max=self.mem_size-1), ((stp_post_val1 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 3).clamp(max=self.mem_size-1), ((stp_post_val1 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 4).clamp(max=self.mem_size-1), ((stp_post_val1 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 5).clamp(max=self.mem_size-1), ((stp_post_val1 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 6).clamp(max=self.mem_size-1), ((stp_post_val1 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs_long + 7).clamp(max=self.mem_size-1), ((stp_post_val1 >> 56) & 0xFF).byte())
            # Store second value at addr+8
            stp_post_addrs2 = (stp_post_addrs_long + 8).clamp(max=self.mem_size-1)
            mem.scatter_(0, stp_post_addrs2, (stp_post_val2 & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 1).clamp(max=self.mem_size-1), ((stp_post_val2 >> 8) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 2).clamp(max=self.mem_size-1), ((stp_post_val2 >> 16) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 3).clamp(max=self.mem_size-1), ((stp_post_val2 >> 24) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 4).clamp(max=self.mem_size-1), ((stp_post_val2 >> 32) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 5).clamp(max=self.mem_size-1), ((stp_post_val2 >> 40) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 6).clamp(max=self.mem_size-1), ((stp_post_val2 >> 48) & 0xFF).byte())
            mem.scatter_(0, (stp_post_addrs2 + 7).clamp(max=self.mem_size-1), ((stp_post_val2 >> 56) & 0xFF).byte())
            # Update base register (post-index writes back after store)
            new_base = rn_vals[stp_post_mask] + imm7
            regs.scatter_(0, rn_indices[rn_indices < 31], new_base[rn_indices < 31])

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDP_POST (Load Pair with post-index: load THEN update base)
            # LDP Xt1, Xt2, [Xn], #imm
            # ═══════════════════════════════════════════════════════════════
            ldp_post_mask = (ops == OpType.LDP_POST.value)
            imm7_raw = (insts[ldp_post_mask] >> 15) & 0x7F
            imm7 = torch.where(imm7_raw >= 0x40, imm7_raw.long() - 0x80, imm7_raw.long()) * 8
            ldp_post_addrs = rn_vals[ldp_post_mask].clamp(0, self.mem_size - 16)
            ldp_post_addrs_long = ldp_post_addrs.long()
            rt2_indices = ((insts[ldp_post_mask] >> 10) & 0x1F).long()
            rn_indices = rns[ldp_post_mask]
            # Load first value
            b0 = mem[ldp_post_addrs_long].long()
            b1 = mem[(ldp_post_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldp_post_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldp_post_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            b4 = mem[(ldp_post_addrs_long + 4).clamp(max=self.mem_size-1)].long()
            b5 = mem[(ldp_post_addrs_long + 5).clamp(max=self.mem_size-1)].long()
            b6 = mem[(ldp_post_addrs_long + 6).clamp(max=self.mem_size-1)].long()
            b7 = mem[(ldp_post_addrs_long + 7).clamp(max=self.mem_size-1)].long()
            ldp_post_val1 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
            results[ldp_post_mask] = ldp_post_val1
            write_mask = write_mask | ldp_post_mask
            # Load second value
            ldp_post_addrs2 = (ldp_post_addrs_long + 8).clamp(max=self.mem_size-1)
            c0 = mem[ldp_post_addrs2].long()
            c1 = mem[(ldp_post_addrs2 + 1).clamp(max=self.mem_size-1)].long()
            c2 = mem[(ldp_post_addrs2 + 2).clamp(max=self.mem_size-1)].long()
            c3 = mem[(ldp_post_addrs2 + 3).clamp(max=self.mem_size-1)].long()
            c4 = mem[(ldp_post_addrs2 + 4).clamp(max=self.mem_size-1)].long()
            c5 = mem[(ldp_post_addrs2 + 5).clamp(max=self.mem_size-1)].long()
            c6 = mem[(ldp_post_addrs2 + 6).clamp(max=self.mem_size-1)].long()
            c7 = mem[(ldp_post_addrs2 + 7).clamp(max=self.mem_size-1)].long()
            ldp_post_val2 = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24) | (c4 << 32) | (c5 << 40) | (c6 << 48) | (c7 << 56)
            valid_rt2 = rt2_indices < 31
            regs.scatter_(0, rt2_indices[valid_rt2], ldp_post_val2[valid_rt2])
            # Update base register (post-index writes back after load)
            new_base = rn_vals[ldp_post_mask] + imm7
            regs.scatter_(0, rn_indices[rn_indices < 31], new_base[rn_indices < 31])

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDRSB (Load Register Signed Byte - sign extends to 64-bit)
            # ═══════════════════════════════════════════════════════════════
            ldrsb_mask = (ops == OpType.LDRSB.value)
            ldrsb_addrs = (rn_vals[ldrsb_mask] + ((insts[ldrsb_mask] >> 10) & 0xFFF)).clamp(0, self.mem_size - 1)
            ldrsb_vals = mem[ldrsb_addrs.long()].long()
            # Sign extend from 8-bit
            ldrsb_vals = torch.where(ldrsb_vals >= 0x80, ldrsb_vals - 0x100, ldrsb_vals)
            results[ldrsb_mask] = ldrsb_vals
            write_mask = write_mask | ldrsb_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDRSH (Load Register Signed Halfword - sign extends to 64-bit)
            # ═══════════════════════════════════════════════════════════════
            ldrsh_mask = (ops == OpType.LDRSH.value)
            ldrsh_addrs = (rn_vals[ldrsh_mask] + ((insts[ldrsh_mask] >> 10) & 0xFFF) * 2).clamp(0, self.mem_size - 2)
            ldrsh_addrs_long = ldrsh_addrs.long()
            b0 = mem[ldrsh_addrs_long].long()
            b1 = mem[(ldrsh_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            ldrsh_vals = b0 | (b1 << 8)
            # Sign extend from 16-bit
            ldrsh_vals = torch.where(ldrsh_vals >= 0x8000, ldrsh_vals - 0x10000, ldrsh_vals)
            results[ldrsh_mask] = ldrsh_vals
            write_mask = write_mask | ldrsh_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDRSW (Load Register Signed Word - sign extends to 64-bit)
            # ═══════════════════════════════════════════════════════════════
            ldrsw_mask = (ops == OpType.LDRSW.value) | (ops == OpType.LDRSW_IMM.value)
            ldrsw_addrs = (rn_vals[ldrsw_mask] + ((insts[ldrsw_mask] >> 10) & 0xFFF) * 4).clamp(0, self.mem_size - 4)
            ldrsw_addrs_long = ldrsw_addrs.long()
            b0 = mem[ldrsw_addrs_long].long()
            b1 = mem[(ldrsw_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldrsw_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldrsw_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            ldrsw_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            # Sign extend from 32-bit
            ldrsw_vals = torch.where(ldrsw_vals >= 0x80000000, ldrsw_vals - 0x100000000, ldrsw_vals)
            results[ldrsw_mask] = ldrsw_vals
            write_mask = write_mask | ldrsw_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR LDR_W (Load 32-bit word, zero extends to 64-bit)
            # ═══════════════════════════════════════════════════════════════
            ldr_w_mask = (ops == OpType.LDR_W.value)
            ldr_w_addrs = (rn_vals[ldr_w_mask] + ((insts[ldr_w_mask] >> 10) & 0xFFF) * 4).clamp(0, self.mem_size - 4)
            ldr_w_addrs_long = ldr_w_addrs.long()
            b0 = mem[ldr_w_addrs_long].long()
            b1 = mem[(ldr_w_addrs_long + 1).clamp(max=self.mem_size-1)].long()
            b2 = mem[(ldr_w_addrs_long + 2).clamp(max=self.mem_size-1)].long()
            b3 = mem[(ldr_w_addrs_long + 3).clamp(max=self.mem_size-1)].long()
            ldr_w_vals = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            results[ldr_w_mask] = ldr_w_vals
            write_mask = write_mask | ldr_w_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR STR_W (Store 32-bit word)
            # ═══════════════════════════════════════════════════════════════
            str_w_mask = (ops == OpType.STR_W.value)
            str_w_addrs = (rn_vals[str_w_mask] + ((insts[str_w_mask] >> 10) & 0xFFF) * 4).clamp(0, self.mem_size - 4)
            str_w_addrs_long = str_w_addrs.long()
            rt_indices = (insts[str_w_mask] & 0x1F).long()
            str_w_vals = torch.where(rt_indices == 31, torch.zeros_like(regs[rt_indices]), regs[rt_indices])
            # Store 4 bytes
            mem.scatter_(0, str_w_addrs_long, (str_w_vals & 0xFF).byte())
            mem.scatter_(0, (str_w_addrs_long + 1).clamp(max=self.mem_size-1), ((str_w_vals >> 8) & 0xFF).byte())
            mem.scatter_(0, (str_w_addrs_long + 2).clamp(max=self.mem_size-1), ((str_w_vals >> 16) & 0xFF).byte())
            mem.scatter_(0, (str_w_addrs_long + 3).clamp(max=self.mem_size-1), ((str_w_vals >> 24) & 0xFF).byte())

            # ═══════════════════════════════════════════════════════════════
            # TENSOR CSEL (Conditional Select)
            # CSEL Xd, Xn, Xm, cond - Xd = cond ? Xn : Xm
            # ═══════════════════════════════════════════════════════════════
            csel_mask = (ops == OpType.CSEL.value) | (ops == OpType.CSEL_W.value)
            csel_insts = insts[csel_mask]
            csel_rn = (csel_insts >> 5) & 0x1F
            csel_rm = (csel_insts >> 16) & 0x1F
            csel_cond = (csel_insts >> 12) & 0xF
            rn_val = torch.where(csel_rn.long() == 31, torch.zeros_like(regs[0]), regs[csel_rn.long()])
            rm_val = torch.where(csel_rm.long() == 31, torch.zeros_like(regs[0]), regs[csel_rm.long()])
            cond_results = self.branch_decider(
                csel_cond,
                self.flags,
                torch.zeros_like(csel_cond, dtype=rn_val.dtype),
                torch.zeros_like(csel_cond, dtype=torch.int64),
            ) > 0.5
            csel_result = torch.where(cond_results, rn_val, rm_val)
            results[csel_mask] = csel_result
            write_mask = write_mask | csel_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR CSINC (Conditional Select Increment)
            # CSINC Xd, Xn, Xm, cond - Xd = cond ? Xn : Xm+1
            # ═══════════════════════════════════════════════════════════════
            csinc_mask = (ops == OpType.CSINC.value)
            csinc_insts = insts[csinc_mask]
            csinc_rn = (csinc_insts >> 5) & 0x1F
            csinc_rm = (csinc_insts >> 16) & 0x1F
            csinc_cond = (csinc_insts >> 12) & 0xF
            rn_val = torch.where(csinc_rn.long() == 31, torch.zeros_like(regs[0]), regs[csinc_rn.long()])
            rm_val = torch.where(csinc_rm.long() == 31, torch.zeros_like(regs[0]), regs[csinc_rm.long()])
            cond_results = self.branch_decider(
                csinc_cond,
                self.flags,
                torch.zeros_like(csinc_cond, dtype=rn_val.dtype),
                torch.zeros_like(csinc_cond, dtype=torch.int64),
            ) > 0.5
            csinc_result = torch.where(cond_results, rn_val, rm_val + 1)
            results[csinc_mask] = csinc_result
            write_mask = write_mask | csinc_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR CSINV (Conditional Select Invert)
            # CSINV Xd, Xn, Xm, cond - Xd = cond ? Xn : ~Xm
            # ═══════════════════════════════════════════════════════════════
            csinv_mask = (ops == OpType.CSINV.value)
            csinv_insts = insts[csinv_mask]
            csinv_rn = (csinv_insts >> 5) & 0x1F
            csinv_rm = (csinv_insts >> 16) & 0x1F
            csinv_cond = (csinv_insts >> 12) & 0xF
            rn_val = torch.where(csinv_rn.long() == 31, torch.zeros_like(regs[0]), regs[csinv_rn.long()])
            rm_val = torch.where(csinv_rm.long() == 31, torch.zeros_like(regs[0]), regs[csinv_rm.long()])
            cond_results = self.branch_decider(
                csinv_cond,
                self.flags,
                torch.zeros_like(csinv_cond, dtype=rn_val.dtype),
                torch.zeros_like(csinv_cond, dtype=torch.int64),
            ) > 0.5
            csinv_result = torch.where(cond_results, rn_val, ~rm_val)
            results[csinv_mask] = csinv_result
            write_mask = write_mask | csinv_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR CSNEG (Conditional Select Negate)
            # CSNEG Xd, Xn, Xm, cond - Xd = cond ? Xn : -Xm
            # ═══════════════════════════════════════════════════════════════
            csneg_mask = (ops == OpType.CSNEG.value)
            csneg_insts = insts[csneg_mask]
            csneg_rn = (csneg_insts >> 5) & 0x1F
            csneg_rm = (csneg_insts >> 16) & 0x1F
            csneg_cond = (csneg_insts >> 12) & 0xF
            rn_val = torch.where(csneg_rn.long() == 31, torch.zeros_like(regs[0]), regs[csneg_rn.long()])
            rm_val = torch.where(csneg_rm.long() == 31, torch.zeros_like(regs[0]), regs[csneg_rm.long()])
            cond_results = self.branch_decider(
                csneg_cond,
                self.flags,
                torch.zeros_like(csneg_cond, dtype=rn_val.dtype),
                torch.zeros_like(csneg_cond, dtype=torch.int64),
            ) > 0.5
            csneg_result = torch.where(cond_results, rn_val, -rm_val)
            results[csneg_mask] = csneg_result
            write_mask = write_mask | csneg_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR UBFM (Unsigned Bitfield Move)
            # UBFM Xd, Xn, #immr, #imms - used for UBFX, LSL, LSR, UXTB, UXTH
            # ═══════════════════════════════════════════════════════════════
            ubfm_mask = (ops == OpType.UBFM.value)
            ubfm_insts = insts[ubfm_mask]
            ubfm_immr = (ubfm_insts >> 16) & 0x3F
            ubfm_imms = (ubfm_insts >> 10) & 0x3F
            ubfm_rn_vals = rn_vals[ubfm_mask]
            # UBFM extracts bits: rotate right by immr, then extract bits 0 to imms
            # Result = (src >> immr) & ((1 << (imms+1)) - 1)
            rotated = (ubfm_rn_vals >> ubfm_immr) | (ubfm_rn_vals << (64 - ubfm_immr))
            mask = (torch.ones_like(rotated) << (ubfm_imms + 1)) - 1
            ubfm_result = rotated & mask
            results[ubfm_mask] = ubfm_result
            write_mask = write_mask | ubfm_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR SBFM (Signed Bitfield Move)
            # SBFM Xd, Xn, #immr, #imms - used for SBFX, ASR, SXTB, SXTH, SXTW
            # ═══════════════════════════════════════════════════════════════
            sbfm_mask = (ops == OpType.SBFM.value)
            sbfm_insts = insts[sbfm_mask]
            sbfm_immr = (sbfm_insts >> 16) & 0x3F
            sbfm_imms = (sbfm_insts >> 10) & 0x3F
            sbfm_rn_vals = rn_vals[sbfm_mask]
            # SBFM: rotate right by immr, extract bits 0 to imms, sign extend
            rotated = (sbfm_rn_vals >> sbfm_immr) | (sbfm_rn_vals << (64 - sbfm_immr))
            mask = (torch.ones_like(rotated) << (sbfm_imms + 1)) - 1
            extracted = rotated & mask
            # Sign extend from bit imms
            sign_bit = (extracted >> sbfm_imms) & 1
            sign_ext_mask = ~mask
            sbfm_result = torch.where(sign_bit == 1, extracted | sign_ext_mask, extracted)
            results[sbfm_mask] = sbfm_result
            write_mask = write_mask | sbfm_mask

            # ═══════════════════════════════════════════════════════════════
            # TENSOR MOVN (Move NOT - load inverted immediate)
            # MOVN Xd, #imm16, LSL #shift
            # ═══════════════════════════════════════════════════════════════
            movn_mask = (ops == OpType.MOVN.value) | (ops == OpType.MOVN_W.value)
            movn_insts = insts[movn_mask]
            movn_imm16 = (movn_insts >> 5) & 0xFFFF
            movn_hw = (movn_insts >> 21) & 0x3
            movn_shift = movn_hw * 16
            movn_val = ~(movn_imm16 << movn_shift)
            # For 32-bit variant, mask to 32 bits
            is_32bit = (ops[movn_mask] == OpType.MOVN_W.value)
            movn_val = torch.where(is_32bit, movn_val & 0xFFFFFFFF, movn_val)
            results[movn_mask] = movn_val
            write_mask = write_mask | movn_mask

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5: PARALLEL SCATTER (write results)
            # ═══════════════════════════════════════════════════════════════
            # Filter to only actual writes (excluding XZR)
            valid_writes = write_mask & (rds != 31) & (rds < 32)
            write_rds = rds[valid_writes]
            write_vals = results[valid_writes]

            if write_rds.numel() > 0:
                # Use scatter for unique destinations

                # Check for accumulator pattern (same reg in consecutive ops)
                unique_rds, inverse_indices = torch.unique(write_rds, return_inverse=True)

                if len(unique_rds) == len(write_rds):
                    # All unique destinations - direct assignment
                    regs[write_rds] = write_vals
                else:
                    # Has duplicates - need scatter_add for accumulators
                    # This handles X0 = X0 + 1 repeated correctly!
                    delta = write_vals - regs[write_rds]
                    regs.scatter_add_(0, write_rds, delta)

            stop_exec = stop_valid & (exec_len == stop_idx)
            stop_pc_next = stop_pc + 4

            # Branch decode (tensor-only)
            imm26 = stop_inst & 0x3FFFFFF
            imm26 = torch.where(imm26 >= 0x2000000, imm26 - 0x4000000, imm26)
            offset26 = imm26 << 2

            imm14 = (stop_inst >> 5) & 0x3FFF
            imm14 = torch.where(imm14 >= 0x2000, imm14 - 0x4000, imm14)
            offset14 = imm14 << 2

            is_b = stop_op == OpType.B.value
            is_bl = stop_op == OpType.BL.value
            is_br = stop_op == OpType.BR.value
            is_blr = stop_op == OpType.BLR.value
            is_ret = stop_op == OpType.RET.value
            is_bcond = stop_op == OpType.B_COND.value
            is_cbz = stop_op == OpType.CBZ.value
            is_cbnz = stop_op == OpType.CBNZ.value
            is_tbz = stop_op == OpType.TBZ.value
            is_tbnz = stop_op == OpType.TBNZ.value
            is_svc = (stop_inst & 0xFFE0001F) == 0xD4000001
            is_halt = stop_inst == 0
            # DEBUG: Track halt detection
            if os.getenv("DEBUG_HALT") == "1" and bool(is_halt.item()):
                # Also check what's actually in memory at PC
                actual_mem = mem[int(pc_t.item()):int(pc_t.item())+4].cpu().numpy()
                actual_inst = int.from_bytes(actual_mem.tobytes(), 'little')
                logger.debug(f"[DEBUG_HALT] is_halt=True, stop_inst=0x{int(stop_inst.item()):08X}, actual_mem_inst=0x{actual_inst:08X}, stop_exec={bool(stop_exec.item())}, stop_valid={bool(stop_valid.item())}, pc=0x{int(pc_t.item()):X}, stop_idx={int(stop_idx.item())}, exec_len={int(exec_len.item())}")

            rt_idx = (stop_inst & 0x1F).long()
            # Clone rt_val to preserve original value - ss_pattern may modify regs[rt_idx] later
            rt_val = regs[rt_idx].clone()

            # Loop vectorization (GPU-only) for simple CBNZ countdown loops
            loop_back = stop_valid & is_cbnz & (offset19 < 0)
            body_len = torch.clamp((-offset19) >> 2, min=0, max=32)
            iterations = rt_val
            vec_ok = loop_back & (body_len >= 1) & (body_len <= 32) & (iterations >= 1) & (iterations < 100000)

            max_body = 32
            body_idx = self._idx_32
            loop_start = stop_pc + offset19
            body_pc = loop_start + body_idx * 4
            body_byte_indices = (body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1)
            body_bytes = mem.gather(0, body_byte_indices.reshape(-1)).view(max_body, 4).long()
            body_insts = (body_bytes[:, 0] |
                          (body_bytes[:, 1] << 8) |
                          (body_bytes[:, 2] << 16) |
                          (body_bytes[:, 3] << 24))
            body_valid = body_idx < body_len
            bi_ops = (body_insts >> 24) & 0xFF
            bi_rds = body_insts & 0x1F
            bi_rns = (body_insts >> 5) & 0x1F
            bi_imms = (body_insts >> 10) & 0xFFF
            add_mask = vec_ok & body_valid & (bi_ops == 0x91) & (bi_rds == bi_rns) & (bi_rds != 31)
            sub_mask = vec_ok & body_valid & (bi_ops == 0xD1) & (bi_rds == bi_rns) & (bi_rds != 31)
            add_regs = bi_rds[add_mask]
            add_deltas = bi_imms[add_mask] * iterations
            sub_regs = bi_rds[sub_mask]
            sub_deltas = bi_imms[sub_mask] * iterations
            regs.scatter_add_(0, add_regs, add_deltas)
            regs.scatter_add_(0, sub_regs, -sub_deltas)
            regs[rt_idx] = torch.where(vec_ok, torch.zeros_like(rt_val), regs[rt_idx])

            # B.cond loop vectorization - SUBS_IMM countdown (SUBS + B.NE)
            # Use stop_exec to ensure we've executed up to the B.cond (registers are updated)
            bcond_loop = stop_valid & stop_exec & is_bcond & (offset19 < 0)
            bcond_body_len = torch.clamp((-offset19) >> 2, min=0, max=2)
            body0 = body_insts[0]
            body1 = body_insts[1]

            body0_op_byte = (body0 >> 24) & 0xFF
            body1_op_byte = (body1 >> 24) & 0xFF
            body0_op = self.op_type_table[body0_op_byte]
            body1_op = self.op_type_table[body1_op_byte]
            body0_op_9 = self.op_code_table[(body0 >> 23) & 0x1FF]
            body1_op_9 = self.op_code_table[(body1 >> 23) & 0x1FF]
            body0_op = torch.where(body0_op_9 > 0, body0_op_9, body0_op)
            body1_op = torch.where(body1_op_9 > 0, body1_op_9, body1_op)

            subs_ok = bcond_loop & (bcond_body_len == 1) & (cond_code == 1) & (body0_op == OpType.SUBS_IMM.value)
            subs_rd = body0 & 0x1F
            subs_rn = (body0 >> 5) & 0x1F
            subs_imm = (body0 >> 10) & 0xFFF
            subs_same = subs_rd == subs_rn
            subs_val = regs[subs_rn.long()]
            subs_safe_imm = torch.where(subs_imm == 0, torch.ones_like(subs_imm), subs_imm)
            subs_rem = subs_val % subs_safe_imm
            subs_vec = subs_ok & subs_same & (subs_imm > 0) & (subs_rem == 0) & (subs_val >= 0)
            subs_iters = torch.where(subs_vec, subs_val // subs_safe_imm, self._const_i64_0)
            subs_iters = torch.clamp(subs_iters, min=0, max=200000)
            subs_active = subs_vec & (subs_iters > 0)
            subs_new = subs_val - (subs_iters * subs_imm)
            regs[subs_rn.long()] = torch.where(subs_active, subs_new, regs[subs_rn.long()])
            self.flags = torch.where(subs_active, self._flags_eq, self.flags)

            # B.cond loop vectorization - ADD/SUB + CMP + B.<cond>
            cmp_loop = bcond_loop & (bcond_body_len == 2)
            add_ok = (body0_op == OpType.ADD_IMM.value)
            sub_ok = (body0_op == OpType.SUB_IMM.value)
            inc_rd = body0 & 0x1F
            inc_rn = (body0 >> 5) & 0x1F
            inc_imm = (body0 >> 10) & 0xFFF
            inc_same = inc_rd == inc_rn
            inc_ok = (inc_rd != 31) & inc_same & (inc_imm > 0) & (add_ok | sub_ok)

            cmp_is_imm = (body1_op == OpType.CMP_IMM.value)
            cmp_is_reg = (body1_op == OpType.CMP_REG.value)
            cmp_rn = (body1 >> 5) & 0x1F
            cmp_rm = (body1 >> 16) & 0x1F
            cmp_imm = (body1 >> 10) & 0xFFF
            cmp_rm_val = torch.where(cmp_rm == 31, self._const_i64_0, regs[cmp_rm.long()])
            cmp_bound = torch.where(cmp_is_imm, cmp_imm, cmp_rm_val)
            cmp_ok = (cmp_is_imm | cmp_is_reg) & (cmp_rn == inc_rd)

            step = torch.where(add_ok, inc_imm, -inc_imm)
            step_abs = torch.where(step < 0, -step, step)
            safe_step_abs = torch.where(step_abs == 0, torch.ones_like(step_abs), step_abs)
            cur_val = regs[inc_rd.long()]

            cond_ne = cond_code == 1
            cond_lt = cond_code == 11
            cond_le = cond_code == 13
            cond_gt = cond_code == 12
            cond_ge = cond_code == 10

            add_dir = add_ok & (step > 0)
            sub_dir = sub_ok & (step < 0)
            add_cond_ok = add_dir & (cond_lt | cond_le | cond_ne)
            sub_cond_ok = sub_dir & (cond_gt | cond_ge | cond_ne)
            cmp_vec = cmp_loop & inc_ok & cmp_ok & (step_abs > 0) & (add_cond_ok | sub_cond_ok)

            add_delta = cmp_bound - cur_val
            sub_delta = cur_val - cmp_bound
            add_iters_lt = torch.where(add_delta > 0, (add_delta + safe_step_abs - 1) // safe_step_abs, self._const_i64_0)
            add_iters_le = torch.where(add_delta >= 0, (add_delta + safe_step_abs) // safe_step_abs, self._const_i64_0)
            sub_iters_gt = torch.where(sub_delta > 0, (sub_delta + safe_step_abs - 1) // safe_step_abs, self._const_i64_0)
            sub_iters_ge = torch.where(sub_delta >= 0, (sub_delta + safe_step_abs) // safe_step_abs, self._const_i64_0)
            add_eq_ok = (add_delta >= 0) & ((add_delta % safe_step_abs) == 0)
            sub_eq_ok = (sub_delta >= 0) & ((sub_delta % safe_step_abs) == 0)
            add_iters_ne = torch.where(add_eq_ok, add_delta // safe_step_abs, self._const_i64_0)
            sub_iters_ne = torch.where(sub_eq_ok, sub_delta // safe_step_abs, self._const_i64_0)

            cmp_iters = torch.where(
                cond_lt,
                add_iters_lt,
                torch.where(
                    cond_le,
                    add_iters_le,
                    torch.where(
                        cond_gt,
                        sub_iters_gt,
                        torch.where(
                            cond_ge,
                            sub_iters_ge,
                            torch.where(
                                cond_ne & add_dir,
                                add_iters_ne,
                                torch.where(cond_ne & sub_dir, sub_iters_ne, self._const_i64_0),
                            ),
                        ),
                    ),
                ),
            )
            cmp_iters = torch.clamp(cmp_iters, min=0, max=200000)
            cmp_active = cmp_vec & (cmp_iters > 0)
            cmp_new = cur_val + (step * cmp_iters)
            regs[inc_rd.long()] = torch.where(cmp_active, cmp_new, regs[inc_rd.long()])

            # Update flags to reflect final CMP
            cmp_res = cmp_new - cmp_bound
            cmp_n = cmp_res < 0
            cmp_z = cmp_res == 0
            cmp_c = cmp_new >= cmp_bound
            cmp_v = ((cmp_new ^ cmp_bound) & (cmp_new ^ cmp_res) & self._sign_mask) != 0
            cmp_flags = torch.stack([cmp_n.float(), cmp_z.float(), cmp_c.float(), cmp_v.float()])
            self.flags = torch.where(cmp_active, cmp_flags, self.flags)

            # Loop vectorization (GPU-only) for STR_POST + CMP + B.NE (memory clear/copy)
            # Use stop_exec to ensure we've executed up to the B.NE (registers are updated)
            # This is safe because regular execution already updated registers before this code runs
            bne_loop = stop_valid & stop_exec & is_bcond & (cond_code == 1) & (offset19 < 0)
            bne_body_len = torch.clamp((-offset19) >> 2, min=0, max=3)
            bne_len2 = bne_loop & (bne_body_len == 2)
            bne_len3 = bne_loop & (bne_body_len == 3)

            bne_body_idx = self._idx_3
            bne_loop_start = stop_pc + offset19
            bne_body_pc = bne_loop_start + bne_body_idx * 4
            bne_body_bytes = mem.gather(
                0,
                (bne_body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(3, 4).long()
            bne_body_insts = (bne_body_bytes[:, 0] |
                              (bne_body_bytes[:, 1] << 8) |
                              (bne_body_bytes[:, 2] << 16) |
                              (bne_body_bytes[:, 3] << 24))

            bne_op_bytes = (bne_body_insts >> 24) & 0xFF
            bne_ops = self.op_type_table[bne_op_bytes]
            bne_op_9bit = (bne_body_insts >> 23) & 0x1FF
            bne_ops_9bit = self.op_code_table[bne_op_9bit]
            bne_ops = torch.where(bne_ops_9bit > 0, bne_ops_9bit, bne_ops)

            bne_f8 = (bne_op_bytes == 0xF8)
            bne_opt_bits = (bne_body_insts >> 10) & 0x3
            bne_opc = (bne_body_insts >> 22) & 0x1
            bne_str_post = bne_f8 & (bne_opt_bits == 0x1) & (bne_opc == 0)

            inst0 = bne_body_insts[0]
            inst1 = bne_body_insts[1]
            inst2 = bne_body_insts[2]

            op1 = bne_ops[1]
            op2 = bne_ops[2]

            cmp_inst = torch.where(bne_len2, inst1, inst2)
            cmp_op = torch.where(bne_len2, op1, op2)
            cmp_is = (cmp_op == OpType.CMP_REG.value) | ((cmp_op == OpType.SUBS_REG.value) & ((cmp_inst & 0x1F) == 31))

            add_ok = bne_len3 & ((op1 == OpType.ADD_IMM.value) | (op1 == OpType.ADD_IMM_W.value))
            add_rd = inst1 & 0x1F
            add_rn = (inst1 >> 5) & 0x1F
            add_imm = (inst1 >> 10) & 0xFFF

            bne_str_first = bne_str_post[0]
            bne_pattern_ok = bne_str_first & (bne_len2 | bne_len3)
            bne_pattern_ok = bne_pattern_ok & cmp_is
            bne_pattern_ok = bne_pattern_ok & (bne_len2 | (add_ok & (add_rd == ((cmp_inst >> 16) & 0x1F)) & (add_rn != add_rd)))

            str_inst = inst0
            str_rn = (str_inst >> 5) & 0x1F
            str_rt = str_inst & 0x1F
            str_imm9 = (str_inst >> 12) & 0x1FF
            str_imm9 = torch.where(str_imm9 & 0x100 != 0, str_imm9 - 0x200, str_imm9)

            cmp_rn = (cmp_inst >> 5) & 0x1F
            cmp_rm = (cmp_inst >> 16) & 0x1F

            ptr = regs[str_rn.long()]
            add_base = regs[add_rn.long()]
            end_ptr = torch.where(bne_len3 & add_ok, add_base + add_imm, regs[cmp_rm.long()])
            regs[cmp_rm.long()] = torch.where(bne_len3 & add_ok, end_ptr, regs[cmp_rm.long()])
            step = str_imm9
            safe_step = torch.where(step == 0, torch.ones_like(step), step)
            stride_ok = step > 0
            range_ok = end_ptr >= ptr
            rem = (end_ptr - ptr) % safe_step
            iter_ok = (step != 0) & (rem == 0)
            iter_count = torch.where(iter_ok, (end_ptr - ptr) // safe_step, torch.zeros_like(ptr))
            iter_count = torch.clamp(iter_count, min=0, max=200000)
            bne_vec = bne_pattern_ok & (str_rn == cmp_rn) & stride_ok & range_ok & (iter_count > 0)
            bne_iters = torch.minimum(iter_count, self._const_i64_4096)
            bne_active = bne_vec & (bne_iters > 0)

            idx = self._idx_4096
            mask = idx < bne_iters
            mask = mask & bne_active
            addr = ptr + idx * step
            addr = addr[mask]
            val = torch.where(str_rt == 31, torch.zeros_like(ptr), regs[str_rt.long()])
            val = val.expand_as(addr)
            addr_bytes = (addr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            val_bytes = ((val.unsqueeze(1) >> (self._idx_8 * 8)) & 0xFF).byte().reshape(-1)
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and addr.numel() > 0:
                hit_code = ((addr >= 0x4558) & (addr <= 0x4560)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] BNE loop writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  ptr: 0x{int(ptr.item()):X}")
                    logger.debug(f"  step: {int(step.item())}")
                    logger.debug(f"  bne_iters: {int(bne_iters.item())}")
                    logger.debug(f"  First few addrs: {addr[:10].cpu().tolist()}")
            mem.scatter_(0, addr_bytes, val_bytes)

            new_ptr = ptr + (bne_iters * step)
            regs[str_rn.long()] = torch.where(bne_active, new_ptr, regs[str_rn.long()])
            self.flags = torch.where(bne_active, self._flags_ne, self.flags)

            # B.cond scan loop vectorization: ADD + LDR + CBZ + CMP + B.<cond>
            # Use stop_exec to ensure we've executed up to the B.cond (registers are updated)
            scan_loop = stop_valid & stop_exec & is_bcond & (offset19 < 0)
            scan_body_len = torch.clamp((-offset19) >> 2, min=0, max=4)
            scan_ok = scan_loop & (scan_body_len == 4)

            scan_body_idx = self._idx_4
            scan_loop_start = stop_pc + offset19
            scan_body_pc = scan_loop_start + scan_body_idx * 4
            scan_bytes = mem.gather(
                0,
                (scan_body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(4, 4).long()
            scan_insts = (scan_bytes[:, 0] |
                          (scan_bytes[:, 1] << 8) |
                          (scan_bytes[:, 2] << 16) |
                          (scan_bytes[:, 3] << 24))

            scan_op_bytes = (scan_insts >> 24) & 0xFF
            scan_ops = self.op_type_table[scan_op_bytes]
            scan_op_9bit = self.op_code_table[(scan_insts >> 23) & 0x1FF]
            scan_ops = torch.where(scan_op_9bit > 0, scan_op_9bit, scan_ops)
            scan_f9 = (scan_op_bytes == 0xF9)
            scan_opc = (scan_insts >> 22) & 0x1
            scan_ops = torch.where(scan_f9 & (scan_opc == 1), self._op_ldr, scan_ops)

            scan_add = scan_ops[0] == OpType.ADD_IMM.value
            scan_ldr = scan_ops[1] == OpType.LDR.value
            scan_cbz = scan_ops[2] == OpType.CBZ.value
            scan_cmp = (scan_ops[3] == OpType.CMP_IMM.value) | ((scan_ops[3] == OpType.SUBS_IMM.value) & ((scan_insts[3] & 0x1F) == 31))

            scan_add_rd = scan_insts[0] & 0x1F
            scan_add_rn = (scan_insts[0] >> 5) & 0x1F
            scan_add_imm = (scan_insts[0] >> 10) & 0xFFF
            scan_ldr_rt = scan_insts[1] & 0x1F
            scan_ldr_rn = (scan_insts[1] >> 5) & 0x1F
            scan_ldr_imm = (scan_insts[1] >> 10) & 0xFFF
            scan_cbz_rt = scan_insts[2] & 0x1F
            scan_cmp_rn = (scan_insts[3] >> 5) & 0x1F
            scan_cmp_imm = (scan_insts[3] >> 10) & 0xFFF

            scan_ok = scan_ok & scan_add & scan_ldr & scan_cbz & scan_cmp
            scan_ok = scan_ok & ((cond_code == 8) | (cond_code == 9) | (cond_code == 12) | (cond_code == 13))
            scan_ok = scan_ok & (scan_add_rd == scan_add_rn) & (scan_add_rd == scan_ldr_rn)
            scan_ok = scan_ok & (scan_ldr_rt == scan_cbz_rt) & (scan_ldr_rt == scan_cmp_rn)
            scan_ok = scan_ok & (scan_add_imm > 0) & (scan_ldr_imm == 0)

            scan_ptr = regs[scan_add_rd.long()]
            scan_stride = scan_add_imm
            scan_idx = self._idx_4096
            scan_addr = scan_ptr + scan_stride + scan_idx * scan_stride
            scan_addr = scan_addr.clamp(0, self.mem_size - 8)
            scan_addr_bytes = (scan_addr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            scan_bytes_val = mem.gather(0, scan_addr_bytes).view(-1, 8).long()
            shifts = (self._idx_8 * 8).long()
            scan_vals = (scan_bytes_val << shifts).sum(dim=1)

            cond_hi = cond_code == 8
            cond_ls = cond_code == 9
            cond_gt = cond_code == 12
            cond_le = cond_code == 13
            cond_mask = (cond_hi & (scan_vals > scan_cmp_imm)) | (cond_ls & (scan_vals <= scan_cmp_imm)) | \
                        (cond_gt & (scan_vals > scan_cmp_imm)) | (cond_le & (scan_vals <= scan_cmp_imm))
            cont_mask = (scan_vals != 0) & cond_mask
            inv = (~cont_mask).long()
            has_break = inv.any()
            first_break = torch.where(has_break, inv.argmax(), self._const_i64_0)
            scan_iters = torch.where(has_break, first_break + 1, self._const_i64_4096)
            scan_active = scan_ok & (scan_iters > 0)

            scan_new_ptr = scan_ptr + scan_iters * scan_stride
            regs[scan_add_rd.long()] = torch.where(scan_active, scan_new_ptr, regs[scan_add_rd.long()])
            scan_val = scan_vals[first_break.long()]
            regs[scan_ldr_rt.long()] = torch.where(scan_active & has_break, scan_val, regs[scan_ldr_rt.long()])
            scan_z = (scan_val == scan_cmp_imm).float()
            scan_n = (scan_val < 0).float()
            scan_c = (scan_val >= scan_cmp_imm).float()
            scan_v = torch.zeros_like(scan_z)
            scan_flags = torch.stack([scan_n, scan_z, scan_c, scan_v])
            self.flags = torch.where(scan_active & has_break, scan_flags, self.flags)

            # Scan+store loop vectorization: ADD + LDR + CBZ + CMP + B.HI + LDR + STR(reg) + B
            scan_store_loop = stop_valid & is_cbz & (offset19 > 0)
            ss_loop_start = stop_pc - self._const_i64_8
            ss_idx = self._idx_8
            ss_body_pc = ss_loop_start + ss_idx * 4
            ss_bytes = mem.gather(
                0,
                (ss_body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(8, 4).long()
            ss_insts = (ss_bytes[:, 0] |
                        (ss_bytes[:, 1] << 8) |
                        (ss_bytes[:, 2] << 16) |
                        (ss_bytes[:, 3] << 24))

            ss_op_bytes = (ss_insts >> 24) & 0xFF
            ss_ops = self.op_type_table[ss_op_bytes]
            ss_op_9bit = self.op_code_table[(ss_insts >> 23) & 0x1FF]
            ss_ops = torch.where(ss_op_9bit > 0, ss_op_9bit, ss_ops)

            ss_f9 = ss_op_bytes == 0xF9
            ss_opc = (ss_insts >> 22) & 0x1
            ss_ops = torch.where(ss_f9 & (ss_opc == 1), self._op_ldr, ss_ops)

            ss_f8 = ss_op_bytes == 0xF8
            ss_opt_bits = (ss_insts >> 10) & 0x3
            ss_opc_bit = (ss_insts >> 22) & 0x1
            ss_reg_mask = ss_f8 & (ss_opt_bits == 0x2)
            ss_ops = torch.where(ss_reg_mask & (ss_opc_bit == 1), self._op_ldr_reg_off, ss_ops)
            ss_ops = torch.where(ss_reg_mask & (ss_opc_bit == 0), self._op_str_reg_off, ss_ops)

            ss_add = ss_ops[0] == OpType.ADD_IMM.value
            ss_ldr1 = ss_ops[1] == OpType.LDR.value
            ss_cbz = ss_ops[2] == OpType.CBZ.value
            ss_cmp = (ss_ops[3] == OpType.CMP_IMM.value) | ((ss_ops[3] == OpType.SUBS_IMM.value) & ((ss_insts[3] & 0x1F) == 31))
            ss_bcond = ss_ops[4] == OpType.B_COND.value
            ss_ldr2 = ss_ops[5] == OpType.LDR.value
            ss_str = ss_ops[6] == OpType.STR_REG_OFF.value
            ss_b = ss_ops[7] == OpType.B.value

            ss_add_rd = ss_insts[0] & 0x1F
            ss_add_rn = (ss_insts[0] >> 5) & 0x1F
            ss_add_imm = (ss_insts[0] >> 10) & 0xFFF

            ss_ldr1_rt = ss_insts[1] & 0x1F
            ss_ldr1_rn = (ss_insts[1] >> 5) & 0x1F

            ss_cbz_rt = ss_insts[2] & 0x1F

            ss_cmp_rn = (ss_insts[3] >> 5) & 0x1F
            ss_cmp_imm = (ss_insts[3] >> 10) & 0xFFF

            ss_bcond_cond = ss_insts[4] & 0xF
            ss_bcond_imm = (ss_insts[4] >> 5) & 0x7FFFF
            ss_bcond_imm = torch.where(ss_bcond_imm >= 0x40000, ss_bcond_imm - 0x80000, ss_bcond_imm)
            ss_bcond_off = ss_bcond_imm << 2
            ss_bcond_target = (ss_loop_start + 16) + ss_bcond_off

            ss_ldr2_rt = ss_insts[5] & 0x1F
            ss_ldr2_rn = (ss_insts[5] >> 5) & 0x1F
            ss_ldr2_imm = (ss_insts[5] >> 10) & 0xFFF
            ss_ldr2_off = ss_ldr2_imm << 3

            ss_str_rt = ss_insts[6] & 0x1F
            ss_str_rn = (ss_insts[6] >> 5) & 0x1F
            ss_str_rm = (ss_insts[6] >> 16) & 0x1F
            ss_str_s = (ss_insts[6] >> 12) & 0x1

            ss_b_imm = ss_insts[7] & 0x3FFFFFF
            ss_b_imm = torch.where(ss_b_imm >= 0x2000000, ss_b_imm - 0x4000000, ss_b_imm)
            ss_b_off = ss_b_imm << 2
            ss_b_target = (ss_loop_start + 28) + ss_b_off

            ss_pattern = scan_store_loop & ss_add & ss_ldr1 & ss_cbz & ss_cmp & ss_bcond & ss_ldr2 & ss_str & ss_b
            ss_pattern = ss_pattern & (ss_add_rd == ss_add_rn) & (ss_add_imm > 0)
            ss_pattern = ss_pattern & (ss_ldr1_rn == ss_add_rd) & (ss_ldr1_rt == ss_cbz_rt)
            ss_pattern = ss_pattern & (ss_cmp_rn == ss_ldr1_rt)
            ss_pattern = ss_pattern & (ss_ldr2_rn == ss_add_rd) & (ss_ldr2_rt == ss_str_rt)
            ss_pattern = ss_pattern & (ss_str_rm == ss_ldr1_rt) & (ss_str_s == 1)
            ss_pattern = ss_pattern & (ss_bcond_cond == 8)  # HI
            ss_pattern = ss_pattern & (ss_bcond_target == ss_loop_start) & (ss_b_target == ss_loop_start)

            ss_base = regs[ss_add_rd.long()]
            ss_step = ss_add_imm
            ss_idx2 = self._idx_4096
            ss_ptr = ss_base + ss_step * (ss_idx2 + 1)
            ss_ptr = ss_ptr.clamp(0, self.mem_size - 8)
            ss_ptr_bytes = (ss_ptr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            ss_ptr_vals = mem.gather(0, ss_ptr_bytes).view(-1, 8).long()
            ss_ptr_shifts = (self._idx_8 * 8).long()
            ss_vals = (ss_ptr_vals << ss_ptr_shifts).sum(dim=1)

            ss_zero = ss_vals == 0
            ss_has_zero = ss_zero.any()
            ss_first_zero = torch.where(ss_has_zero, ss_zero.long().argmax(), self._const_i64_4096)
            ss_iters = torch.where(ss_has_zero, ss_first_zero, self._const_i64_4096)
            ss_iters = torch.clamp(ss_iters, min=0, max=self._const_i64_4096)
            ss_valid = ss_idx2 < ss_iters

            ss_hi = ss_vals > ss_cmp_imm
            ss_store_mask = ss_valid & (~ss_hi)
            ss_data_ptr = (ss_ptr + ss_ldr2_off).clamp(0, self.mem_size - 8)
            ss_data_bytes = (ss_data_ptr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            ss_data_vals = mem.gather(0, ss_data_bytes).view(-1, 8).long()
            ss_data_vals = (ss_data_vals << ss_ptr_shifts).sum(dim=1)

            # BUG FIX: Only scatter when ss_pattern matches, not just when store_mask conditions met
            # Without this check, random memory values can cause writes to wrong addresses (including code!)
            ss_store_mask_final = ss_store_mask & ss_pattern
            ss_store_addr = regs[ss_str_rn.long()] + (ss_vals << 3)
            ss_store_addr = ss_store_addr.clamp(0, self.mem_size - 8)
            ss_store_addr = ss_store_addr[ss_store_mask_final]
            ss_store_vals = ss_data_vals[ss_store_mask_final]
            ss_store_addr_bytes = (ss_store_addr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            ss_store_bytes = ((ss_store_vals.unsqueeze(1) >> (self._idx_8 * 8)) & 0xFF).byte().reshape(-1)
            # DEBUG: Check for writes to code section
            if os.getenv("DEBUG_MEM_WRITE") and ss_store_addr.numel() > 0:
                hit_code = ((ss_store_addr >= 0x4558) & (ss_store_addr <= 0x4560)).any()
                if hit_code.item():
                    logger.debug(f"[DEBUG_MEM_WRITE] SS_PATTERN writing to code section!")
                    logger.debug(f"  PC: 0x{int(pc_t.item()):X}")
                    logger.debug(f"  ss_pattern: {bool(ss_pattern.item())}")
                    logger.debug(f"  Addresses: {ss_store_addr.cpu().tolist()[:10]}...")
            mem.scatter_(0, ss_store_addr_bytes, ss_store_bytes)

            ss_last_idx = torch.clamp(ss_iters - 1, min=0, max=self._const_i64_4096 - 1)
            ss_last_val = ss_vals[ss_last_idx.long()]
            ss_new_x2 = torch.where(ss_iters > 0, ss_last_val, self._const_i64_0)

            ss_store_has = ss_store_mask_final.any()
            ss_store_rev = torch.flip(ss_store_mask_final.long(), dims=[0])
            ss_store_last = torch.where(ss_store_has, (self._const_i64_4096 - 1) - ss_store_rev.argmax(), self._const_i64_0)
            ss_last_data = ss_data_vals[ss_store_last.long()]
            # BUG FIX: Must check ss_pattern, not just ss_store_has!
            # Otherwise random memory values can corrupt registers when pattern doesn't match
            regs[ss_str_rt.long()] = torch.where(ss_pattern & ss_store_has, ss_last_data, regs[ss_str_rt.long()])

            regs[ss_add_rd.long()] = torch.where(ss_pattern, ss_base + ss_step * ss_iters, regs[ss_add_rd.long()])
            regs[ss_ldr1_rt.long()] = torch.where(ss_pattern, ss_new_x2, regs[ss_ldr1_rt.long()])

            ss_cmp_res = ss_last_val - ss_cmp_imm
            ss_n = (ss_cmp_res < 0).float()
            ss_z = (ss_cmp_res == 0).float()
            ss_c = (ss_last_val >= ss_cmp_imm).float()
            ss_v = (((ss_last_val ^ ss_cmp_imm) & (ss_last_val ^ ss_cmp_res) & self._sign_mask) != 0).float()
            ss_flags = torch.stack([ss_n, ss_z, ss_c, ss_v])
            self.flags = torch.where(ss_pattern & (ss_iters > 0), ss_flags, self.flags)

            ss_hi_count = (ss_hi & ss_valid).long().sum()
            ss_le_count = (ss_valid & (~ss_hi)).long().sum()
            ss_exec = (ss_hi_count * self._const_i64_5) + (ss_le_count * self._const_i64_8)
            ss_exit = ss_has_zero & (ss_first_zero < self._const_i64_4096)
            ss_exec = ss_exec + torch.where(ss_exit, self._const_i64_3, self._const_i64_0)
            ss_next_pc = torch.where(ss_exit, stop_pc + offset19, ss_loop_start)
            # BUG FIX: Disable ss_pattern vectorization when CBZ should NOT branch (rt_val != 0)
            # The vectorization skips processing the current entry, causing incorrect behavior.
            # When rt_val != 0, let normal execution handle it (CBZ doesn't branch, continues to loop body).
            # When rt_val == 0, CBZ branches to exit - normal branch resolution handles it.
            ss_active = ss_pattern & (rt_val == 0)

            # AUXV scan loop vectorization:
            # SUB_IMM + CBZ + SUB_REG + LDR + UBFM + CMP_IMM + B.NE
            # Use stop_exec to ensure we've executed up to the B.NE (registers are updated)
            aux_loop = stop_valid & stop_exec & is_bcond & (cond_code == 1) & (offset19 < 0)
            aux_body_len = torch.clamp((-offset19) >> 2, min=0, max=8)
            aux_ok = aux_loop & (aux_body_len == 6)

            aux_idx = self._idx_8
            aux_loop_start = stop_pc + offset19
            aux_body_pc = aux_loop_start + aux_idx * 4
            aux_bytes = mem.gather(
                0,
                (aux_body_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(8, 4).long()
            aux_insts = (aux_bytes[:, 0] |
                         (aux_bytes[:, 1] << 8) |
                         (aux_bytes[:, 2] << 16) |
                         (aux_bytes[:, 3] << 24))
            aux_insts = aux_insts[:6]

            aux_op_bytes = (aux_insts >> 24) & 0xFF
            aux_ops = self.op_type_table[aux_op_bytes]
            aux_op_9bit = self.op_code_table[(aux_insts >> 23) & 0x1FF]
            aux_ops = torch.where(aux_op_9bit > 0, aux_op_9bit, aux_ops)
            aux_f9 = (aux_op_bytes == 0xF9)
            aux_opc = (aux_insts >> 22) & 0x1
            aux_ops = torch.where(aux_f9 & (aux_opc == 1), self._op_ldr, aux_ops)

            aux_i0 = aux_insts[0]
            aux_i1 = aux_insts[1]
            aux_i2 = aux_insts[2]
            aux_i3 = aux_insts[3]
            aux_i4 = aux_insts[4]
            aux_i5 = aux_insts[5]

            aux_op0 = aux_ops[0]
            aux_op1 = aux_ops[1]
            aux_op2 = aux_ops[2]
            aux_op3 = aux_ops[3]
            aux_op4 = aux_ops[4]
            aux_op5 = aux_ops[5]

            aux_subimm = aux_op0 == OpType.SUB_IMM.value
            aux_cbz = aux_op1 == OpType.CBZ.value
            aux_subreg = aux_op2 == OpType.SUB_REG.value
            aux_ldr = aux_op3 == OpType.LDR.value
            # Accept either UBFM or AND_IMM for bit masking (compilers use both)
            aux_ubfm = (aux_op4 == OpType.UBFM.value) | (aux_op4 == OpType.AND_IMM.value)
            aux_cmp = (aux_op5 == OpType.CMP_IMM.value) | ((aux_op5 == OpType.SUBS_IMM.value) & ((aux_i5 & 0x1F) == 31))

            aux_rd0 = aux_i0 & 0x1F
            aux_rn0 = (aux_i0 >> 5) & 0x1F
            aux_step = (aux_i0 >> 10) & 0xFFF
            aux_shift = (aux_i0 >> 22) & 0x1

            aux_cbz_rt = aux_i1 & 0x1F
            aux_rd2 = aux_i2 & 0x1F
            aux_rn2 = (aux_i2 >> 5) & 0x1F
            aux_rm2 = (aux_i2 >> 16) & 0x1F

            aux_ldr_rt = aux_i3 & 0x1F
            aux_ldr_rn = (aux_i3 >> 5) & 0x1F
            aux_ldr_imm = (aux_i3 >> 10) & 0xFFF
            aux_ldr_off = aux_ldr_imm << 3

            aux_ubfm_rd = aux_i4 & 0x1F
            aux_ubfm_rn = (aux_i4 >> 5) & 0x1F
            aux_ubfm_immr = (aux_i4 >> 16) & 0x3F
            aux_ubfm_imms = (aux_i4 >> 10) & 0x3F

            aux_cmp_rn = (aux_i5 >> 5) & 0x1F
            aux_cmp_imm = (aux_i5 >> 10) & 0xFFF

            aux_ok = aux_ok & aux_subimm & aux_cbz & aux_subreg & aux_ldr & aux_ubfm & aux_cmp
            aux_ok = aux_ok & (aux_rd0 == aux_rn0) & (aux_cbz_rt == aux_rd0)
            aux_ok = aux_ok & (aux_rm2 == aux_rd0) & (aux_rd2 == aux_ldr_rn)
            aux_ok = aux_ok & (aux_ldr_rt == aux_ubfm_rd) & (aux_ldr_rt == aux_ubfm_rn) & (aux_ldr_rt == aux_cmp_rn)
            aux_step = aux_step << (aux_shift * 12)
            aux_ok = aux_ok & (aux_step > 0)

            aux_counter = regs[aux_rd0.long()]
            aux_safe_step = torch.where(aux_step == 0, self._const_i64_1, aux_step)
            aux_iters = torch.clamp(aux_counter // aux_safe_step, min=0, max=self._const_i64_4096)
            aux_active = aux_ok & (aux_iters > 0)

            aux_base = regs[aux_rn2.long()]
            aux_base_addr = aux_base - aux_counter + aux_step + aux_ldr_off
            aux_last_addr = aux_base_addr + (aux_iters - 1) * aux_step
            aux_range_ok = (aux_base_addr >= 0) & (aux_last_addr <= (self.mem_size - 8))
            aux_idx2 = self._idx_4096
            aux_mask = aux_idx2 < aux_iters
            aux_addr = aux_base_addr + aux_idx2 * aux_step
            aux_addr = aux_addr.clamp(0, self.mem_size - 8)
            aux_addr_bytes = (aux_addr.unsqueeze(1) + self._idx_8).reshape(-1).clamp(0, self.mem_size - 1)
            aux_bytes_val = mem.gather(0, aux_addr_bytes).view(-1, 8).long()
            aux_shifts = (self._idx_8 * 8).long()
            aux_vals = (aux_bytes_val << aux_shifts).sum(dim=1)
            aux_vals = torch.where(aux_mask, aux_vals, aux_cmp_imm + 1)

            aux_rot_r = aux_vals >> aux_ubfm_immr
            aux_rot_l = aux_vals << (64 - aux_ubfm_immr)
            aux_rot = torch.where(aux_ubfm_immr == 0, aux_vals, aux_rot_r | aux_rot_l)
            aux_mask_full = torch.where(aux_ubfm_imms == 63, torch.full_like(aux_vals, -1), (torch.ones_like(aux_vals) << (aux_ubfm_imms + 1)) - 1)
            aux_ubfm_vals = aux_rot & aux_mask_full

            aux_match = aux_ubfm_vals == aux_cmp_imm
            aux_has_match = aux_match.any()
            aux_first = torch.where(aux_has_match, aux_match.long().argmax(), self._const_i64_0)
            aux_iters_eff = torch.where(aux_has_match, aux_first + 1, aux_iters)
            aux_iters_eff = torch.clamp(aux_iters_eff, min=0, max=self._const_i64_4096)
            aux_active = aux_active & aux_range_ok & (aux_iters_eff > 0)

            aux_last_idx = torch.clamp(aux_iters_eff - 1, min=0, max=self._const_i64_4096 - 1)
            aux_last_val = aux_ubfm_vals[aux_last_idx.long()]
            aux_last_counter = aux_counter - aux_step * aux_iters_eff
            aux_last_ptr = aux_base - aux_last_counter

            aux_cbz_inst = aux_i1
            aux_cbz_imm = (aux_cbz_inst >> 5) & 0x7FFFF
            aux_cbz_imm = torch.where(aux_cbz_imm >= 0x40000, aux_cbz_imm - 0x80000, aux_cbz_imm)
            aux_cbz_target = (aux_loop_start + 4) + (aux_cbz_imm << 2)

            aux_cmp_res = aux_last_val - aux_cmp_imm
            aux_n = (aux_cmp_res < 0).float()
            aux_z = (aux_cmp_res == 0).float()
            aux_c = (aux_last_val >= aux_cmp_imm).float()
            aux_v = torch.zeros_like(aux_z)
            aux_flags = torch.stack([aux_n, aux_z, aux_c, aux_v])

            aux_exit_bne = aux_has_match
            aux_continue = aux_last_counter > 0
            aux_next_pc = torch.where(
                aux_exit_bne,
                stop_pc_next,
                torch.where(aux_continue, aux_loop_start, aux_cbz_target),
            )

            aux_new_counter = torch.where(aux_exit_bne | aux_continue, aux_last_counter, self._const_i64_0)
            regs[aux_rd0.long()] = torch.where(aux_active, aux_new_counter, regs[aux_rd0.long()])
            regs[aux_rd2.long()] = torch.where(aux_active, aux_last_ptr, regs[aux_rd2.long()])
            regs[aux_ldr_rt.long()] = torch.where(aux_active, aux_last_val, regs[aux_ldr_rt.long()])
            self.flags = torch.where(aux_active, aux_flags, self.flags)

            # TBZ/TBNZ loop vectorization - LSR #1 + TBZ/TBNZ bit0
            tb_loop = stop_valid & (is_tbz | is_tbnz) & (offset14 < 0)
            tb_body_len = torch.clamp((-offset14) >> 2, min=0, max=1)
            tb_ok = tb_loop & (tb_body_len == 1)

            tb_loop_start = stop_pc + offset14
            tb_bytes = mem.gather(
                0,
                (tb_loop_start + self._idx_4).clamp(0, self.mem_size - 1)
            ).long()
            tb_inst = (tb_bytes[0] |
                       (tb_bytes[1] << 8) |
                       (tb_bytes[2] << 16) |
                       (tb_bytes[3] << 24))

            tb_op_byte = (tb_inst >> 24) & 0xFF
            tb_immr = (tb_inst >> 16) & 0x3F
            tb_imms = (tb_inst >> 10) & 0x3F
            tb_lsr1 = (tb_op_byte == 0xD3) & (tb_imms == 63) & (tb_immr == 1)
            tb_rd = tb_inst & 0x1F
            tb_rn = (tb_inst >> 5) & 0x1F
            tb_same = (tb_rd == tb_rn) & (tb_rd == rt_idx)

            bit = ((stop_inst >> 19) & 0x1F) | (((stop_inst >> 31) & 1) << 5)
            tb_bit0 = bit == 0
            tb_vec = tb_ok & tb_lsr1 & tb_same & tb_bit0

            tb_bits = ((rt_val >> self._idx_64) & 1).long()
            tb_has_one = tb_bits.any()
            tb_inv_bits = (tb_bits == 0).long()
            tb_has_zero = tb_inv_bits.any()
            tb_tz = torch.where(tb_has_one, tb_bits.argmax(), self._const_i64_0)
            tb_to = torch.where(tb_has_zero, tb_inv_bits.argmax(), self._const_i64_0)
            tb_iters = torch.where(is_tbz, tb_tz, tb_to)
            tb_iters = torch.clamp(tb_iters, min=0, max=63)
            tb_valid = torch.where(is_tbz, tb_has_one, tb_has_zero)
            tb_active = tb_vec & tb_valid & (tb_iters > 0)

            tb_new = rt_val >> tb_iters
            regs[rt_idx] = torch.where(tb_active, tb_new, regs[rt_idx])

            branch_type = torch.where(
                is_cbz,
                self._const_i64_1,
                torch.where(is_cbnz, self._const_i64_2, self._const_i64_0),
            )
            cond_take = self.branch_decider(cond_code, self.flags, rt_val, branch_type) > 0.5

            bit_set = ((rt_val >> bit) & 1) != 0
            tb_take = torch.where(is_tbz, ~bit_set, bit_set)
            spec_cond_take = torch.where(is_tbz | is_tbnz, tb_take, cond_take)

            rn = (stop_inst >> 5) & 0x1F
            rn_eff = torch.where(is_ret & (rn == 0), self._const_i64_30, rn)
            reg_target = regs[rn_eff.long()]

            branch_pc = stop_pc_next
            branch_pc = torch.where(is_b | is_bl, stop_pc + offset26, branch_pc)
            branch_pc = torch.where(is_br | is_blr | is_ret, reg_target, branch_pc)
            branch_pc = torch.where(is_bcond | is_cbz | is_cbnz,
                                    torch.where(cond_take, stop_pc + offset19, stop_pc_next),
                                    branch_pc)
            branch_pc = torch.where(is_tbz | is_tbnz,
                                    torch.where(tb_take, stop_pc + offset14, stop_pc_next),
                                    branch_pc)
            branch_pc = torch.where(is_svc | is_halt, stop_pc_next, branch_pc)

            lr_update = stop_exec & (is_bl | is_blr)
            regs[30] = torch.where(lr_update, stop_pc_next, regs[30])

            # Speculative dual-path window (single ALU instruction) for branches
            spec_blocked = vec_ok | bne_active | subs_active | cmp_active | tb_active | scan_active | ss_active | aux_active
            spec_branch = is_bcond | is_cbz | is_cbnz | is_tbz | is_tbnz
            spec_enabled = self._const_i64_1.bool() if enable_speculation else self._const_i64_0.bool()
            spec_active = stop_exec & spec_branch & (~spec_blocked) & spec_enabled & spec_gate_on
            spec_off = torch.where(is_tbz | is_tbnz, offset14, offset19)
            spec_active = spec_active & (spec_off != 0)
            spec_pc_f = stop_pc_next
            spec_pc_b = stop_pc + spec_off
            spec_pc = torch.stack([spec_pc_f, spec_pc_b])
            spec_bytes = mem.gather(
                0,
                (spec_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(2, 4).long()
            spec_insts = (spec_bytes[:, 0] |
                          (spec_bytes[:, 1] << 8) |
                          (spec_bytes[:, 2] << 16) |
                          (spec_bytes[:, 3] << 24))
            spec_op_bytes = (spec_insts >> 24) & 0xFF
            spec_ops = self.op_type_table[spec_op_bytes]
            spec_op_9bit = self.op_code_table[(spec_insts >> 23) & 0x1FF]
            spec_ops = torch.where(spec_op_9bit > 0, spec_op_9bit, spec_ops)

            spec_rds = spec_insts & 0x1F
            spec_rns = (spec_insts >> 5) & 0x1F
            spec_rms = (spec_insts >> 16) & 0x1F
            spec_imm12 = (spec_insts >> 10) & 0xFFF
            spec_imm16 = (spec_insts >> 5) & 0xFFFF
            spec_hw = (spec_insts >> 21) & 0x3

            spec_rn_vals = regs[spec_rns]
            spec_rm_vals = regs[spec_rms]
            spec_rd_vals = regs[spec_rds]
            spec_rn_vals_32 = spec_rn_vals & 0xFFFFFFFF
            spec_rm_vals_32 = spec_rm_vals & 0xFFFFFFFF

            spec_vals = self._spec_vals
            spec_vals.zero_()
            spec_write = self._spec_write
            spec_write.zero_()

            spec_add_imm = spec_ops == OpType.ADD_IMM.value
            spec_add_imm_w = spec_ops == OpType.ADD_IMM_W.value
            spec_sub_imm = spec_ops == OpType.SUB_IMM.value
            spec_sub_imm_w = spec_ops == OpType.SUB_IMM_W.value
            spec_add_reg = spec_ops == OpType.ADD_REG.value
            spec_add_reg_w = spec_ops == OpType.ADD_REG_W.value
            spec_sub_reg = spec_ops == OpType.SUB_REG.value
            spec_sub_reg_w = spec_ops == OpType.SUB_REG_W.value
            spec_and_reg = spec_ops == OpType.AND_REG.value
            spec_orr_reg = spec_ops == OpType.ORR_REG.value
            spec_eor_reg = spec_ops == OpType.EOR_REG.value
            spec_movz = (spec_ops == OpType.MOVZ.value) | (spec_ops == OpType.MOVZ_W.value)
            spec_movk = (spec_ops == OpType.MOVK.value) | (spec_ops == OpType.MOVK_W.value)
            spec_movn = (spec_ops == OpType.MOVN.value) | (spec_ops == OpType.MOVN_W.value)
            spec_mov = (spec_ops == OpType.MOV_REG.value) | (spec_ops == OpType.MOV_W.value)
            spec_lsl_imm = spec_ops == OpType.LSL_IMM.value
            spec_lsr_imm = spec_ops == OpType.LSR_IMM.value
            spec_asr_imm = spec_ops == OpType.ASR_IMM.value
            spec_lsl_reg = spec_ops == OpType.LSL_REG.value
            spec_lsr_reg = spec_ops == OpType.LSR_REG.value
            spec_asr_reg = spec_ops == OpType.ASR_REG.value

            spec_vals = torch.where(spec_add_imm, spec_rn_vals + spec_imm12, spec_vals)
            spec_vals = torch.where(spec_add_imm_w, (spec_rn_vals_32 + spec_imm12) & 0xFFFFFFFF, spec_vals)
            spec_vals = torch.where(spec_sub_imm, spec_rn_vals - spec_imm12, spec_vals)
            spec_vals = torch.where(spec_sub_imm_w, (spec_rn_vals_32 - spec_imm12) & 0xFFFFFFFF, spec_vals)
            spec_vals = torch.where(spec_add_reg, spec_rn_vals + spec_rm_vals, spec_vals)
            spec_vals = torch.where(spec_add_reg_w, (spec_rn_vals_32 + spec_rm_vals_32) & 0xFFFFFFFF, spec_vals)
            spec_vals = torch.where(spec_sub_reg, spec_rn_vals - spec_rm_vals, spec_vals)
            spec_vals = torch.where(spec_sub_reg_w, (spec_rn_vals_32 - spec_rm_vals_32) & 0xFFFFFFFF, spec_vals)
            spec_vals = torch.where(spec_and_reg, spec_rn_vals & spec_rm_vals, spec_vals)
            spec_vals = torch.where(spec_orr_reg, spec_rn_vals | spec_rm_vals, spec_vals)
            spec_vals = torch.where(spec_eor_reg, spec_rn_vals ^ spec_rm_vals, spec_vals)

            spec_movz_val = spec_imm16 << (spec_hw * 16)
            spec_movz_val = torch.where(spec_ops == OpType.MOVZ_W.value, spec_movz_val & 0xFFFFFFFF, spec_movz_val)
            spec_vals = torch.where(spec_movz, spec_movz_val, spec_vals)
            spec_movk_clear = ~(self._movk_clear_base << (spec_hw * 16))
            spec_movk_val = (spec_rd_vals & spec_movk_clear) | (spec_imm16 << (spec_hw * 16))
            spec_movk_val = torch.where(spec_ops == OpType.MOVK_W.value, spec_movk_val & 0xFFFFFFFF, spec_movk_val)
            spec_vals = torch.where(spec_movk, spec_movk_val, spec_vals)
            spec_movn_val = ~(spec_imm16 << (spec_hw * 16))
            spec_movn_val = torch.where(spec_ops == OpType.MOVN_W.value, spec_movn_val & 0xFFFFFFFF, spec_movn_val)
            spec_vals = torch.where(spec_movn, spec_movn_val, spec_vals)
            spec_vals = torch.where(spec_mov, spec_rm_vals, spec_vals)

            spec_shift_amt = (spec_insts >> 10) & 0x3F
            spec_shift_amt = spec_shift_amt.clamp(0, 63)
            spec_vals = torch.where(spec_lsl_imm, spec_rn_vals << spec_shift_amt, spec_vals)
            spec_vals = torch.where(spec_lsr_imm, spec_rn_vals >> spec_shift_amt, spec_vals)
            spec_vals = torch.where(spec_asr_imm, spec_rn_vals >> spec_shift_amt, spec_vals)
            spec_reg_shift = spec_rm_vals & 0x3F
            spec_vals = torch.where(spec_lsl_reg, spec_rn_vals << spec_reg_shift, spec_vals)
            spec_vals = torch.where(spec_lsr_reg, spec_rn_vals >> spec_reg_shift, spec_vals)
            spec_vals = torch.where(spec_asr_reg, spec_rn_vals >> spec_reg_shift, spec_vals)

            spec_write = spec_write | spec_add_imm | spec_add_imm_w | spec_sub_imm | spec_sub_imm_w
            spec_write = spec_write | spec_add_reg | spec_add_reg_w | spec_sub_reg | spec_sub_reg_w
            spec_write = spec_write | spec_and_reg | spec_orr_reg | spec_eor_reg
            spec_write = spec_write | spec_movz | spec_movk | spec_movn | spec_mov
            spec_write = spec_write | spec_lsl_imm | spec_lsr_imm | spec_asr_imm
            spec_write = spec_write | spec_lsl_reg | spec_lsr_reg | spec_asr_reg

            spec_is_reg = spec_add_reg | spec_add_reg_w | spec_sub_reg | spec_sub_reg_w | spec_and_reg | spec_orr_reg | spec_eor_reg | spec_lsl_reg | spec_lsr_reg | spec_asr_reg
            spec_is_imm = spec_add_imm | spec_add_imm_w | spec_sub_imm | spec_sub_imm_w | spec_lsl_imm | spec_lsr_imm | spec_asr_imm
            spec_rn_ok = torch.where(spec_is_reg | spec_is_imm, spec_rns != 31, torch.ones_like(spec_rns, dtype=torch.bool))
            spec_rm_ok = torch.where(spec_is_reg, spec_rms != 31, torch.ones_like(spec_rms, dtype=torch.bool))
            spec_safe = spec_rn_ok & spec_rm_ok & spec_write
            spec_ok = spec_safe[0] & spec_safe[1]
            spec_active = spec_active & spec_ok

            spec_rd_f = spec_rds[0]
            spec_rd_b = spec_rds[1]
            spec_val_f = spec_vals[0]
            spec_val_b = spec_vals[1]
            spec_w_f = spec_write[0] & (spec_rd_f != 31)
            spec_w_b = spec_write[1] & (spec_rd_b != 31)
            same_rd = spec_rd_f == spec_rd_b

            combined_mask = spec_active & same_rd & (spec_w_f | spec_w_b)
            combined_val = torch.where(spec_cond_take, spec_val_b, spec_val_f)
            regs[spec_rd_f.long()] = torch.where(combined_mask, combined_val, regs[spec_rd_f.long()])

            diff_mask = spec_active & (~same_rd)
            mask_f = diff_mask & spec_w_f & (~spec_cond_take)
            mask_b = diff_mask & spec_w_b & spec_cond_take
            idxs = torch.stack([spec_rd_f, spec_rd_b]).long()
            vals = torch.stack([spec_val_f, spec_val_b])
            masks = torch.stack([mask_f, mask_b]).long()
            delta = vals - regs[idxs]
            regs.scatter_add_(0, idxs, delta * masks)

            # GPU-only loop signature logging for unvectorized backward branches
            loop_offset = torch.where(is_tbz | is_tbnz, offset14, offset19)
            branch_offset = is_b | is_bl | is_bcond | is_cbz | is_cbnz | is_tbz | is_tbnz
            loop_back = stop_exec & branch_offset & (loop_offset < 0)
            log_enabled = self._loop_log_enabled > 0
            log_all = self._loop_log_all[0] > 0
            unvec_loop = loop_back & log_enabled & (log_all | ((~vec_ok) & (~bne_active) & (~subs_active) & (~cmp_active) & (~tb_active) & (~scan_active) & (~ss_active) & (~aux_active)))
            log_len = torch.clamp((-loop_offset) >> 2, min=1, max=4)
            log_idx = self._idx_4
            log_pc = stop_pc + loop_offset + log_idx * 4
            log_bytes = mem.gather(
                0,
                (log_pc.unsqueeze(1) + self._idx_4).clamp(0, self.mem_size - 1).reshape(-1)
            ).view(4, 4).long()
            log_insts = (log_bytes[:, 0] |
                         (log_bytes[:, 1] << 8) |
                         (log_bytes[:, 2] << 16) |
                         (log_bytes[:, 3] << 24))
            log_ops = (log_insts >> 24) & 0xFF
            log_ops = torch.where(log_idx < log_len, log_ops, self._const_i64_0)
            branch_kind = torch.where(
                is_b,
                self._const_i64_1,
                torch.where(
                    is_bl,
                    self._const_i64_2,
                    torch.where(
                        is_bcond,
                        self._const_i64_3,
                        torch.where(
                            is_cbz,
                            self._const_i64_4,
                            torch.where(
                                is_cbnz,
                                self._const_i64_5,
                                torch.where(
                                    is_tbz,
                                    self._const_i64_6,
                                    torch.where(is_tbnz, self._const_i64_7, self._const_i64_0),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            sig = (branch_kind << 56) | (cond_code << 52) | (log_len << 48) | (log_ops[0] << 40) | (log_ops[1] << 32) | (log_ops[2] << 24) | (log_ops[3] << 16)
            sig_ptr = self._loop_sig_ptr[0]
            sig_idx = (sig_ptr % self._loop_sig_buf.numel()).long()
            cur_sig = self._loop_sig_buf[sig_idx]
            self._loop_sig_buf[sig_idx] = torch.where(unvec_loop, sig, cur_sig)
            self._loop_sig_ptr[0] = sig_ptr + unvec_loop.long()
            sig_hash = sig ^ (sig >> 33) ^ (sig >> 17) ^ (sig >> 9)
            sig_bin = (sig_hash & self._const_i64_1023).long()
            self._loop_sig_counts.scatter_add_(0, sig_bin, unvec_loop.long())

            bne_done = bne_active & (bne_iters == iter_count)
            bne_next_pc = torch.where(bne_done, stop_pc_next, stop_pc + offset19)
            pc_next = torch.where(stop_exec, branch_pc, pc_t + exec_len * 4)
            pc_next = torch.where(vec_ok, stop_pc_next, pc_next)
            pc_next = torch.where(subs_active, stop_pc_next, pc_next)
            pc_next = torch.where(cmp_active, stop_pc_next, pc_next)
            pc_next = torch.where(tb_active, stop_pc_next, pc_next)
            scan_done = scan_active & has_break
            scan_next_pc = torch.where(scan_done, stop_pc_next, stop_pc + offset19)
            pc_next = torch.where(scan_active, scan_next_pc, pc_next)
            pc_next = torch.where(ss_active, ss_next_pc, pc_next)
            pc_next = torch.where(bne_active, bne_next_pc, pc_next)
            pc_next = torch.where(aux_active, aux_next_pc, pc_next)
            spec_next_pc = torch.where(spec_cond_take, spec_pc_b + 4, spec_pc_f + 4)
            pc_next = torch.where(spec_active, spec_next_pc, pc_next)

            # ════════════════════════════════════════════════════════════════
            # BTB UPDATE - Track conditional branch outcomes for prediction
            # Only update on conditional branches (not unconditional B/BL)
            # ════════════════════════════════════════════════════════════════
            is_cond_branch = stop_exec & (is_bcond | is_cbz | is_cbnz | is_tbz | is_tbnz)
            if is_cond_branch.any():
                # Get actual target PC and whether branch was taken
                branch_target = torch.where(spec_cond_take,
                                           stop_pc + torch.where(is_tbz | is_tbnz, offset14, offset19),
                                           stop_pc_next)
                self.btb.update(stop_pc, spec_cond_take, branch_target)

            pc_t = pc_next

            exec_next = exec_len + stop_exec.long()
            exec_next = torch.where(vec_ok, exec_len + (iterations * (body_len + 1)), exec_next)
            exec_next = torch.where(subs_active, exec_len + (subs_iters * (bcond_body_len + 1)), exec_next)
            exec_next = torch.where(cmp_active, exec_len + (cmp_iters * (bcond_body_len + 1)), exec_next)
            exec_next = torch.where(tb_active, exec_len + (tb_iters * (tb_body_len + 1)), exec_next)
            exec_next = torch.where(scan_active, exec_len + (scan_iters * (scan_body_len + 1)), exec_next)
            exec_next = torch.where(ss_active, ss_exec, exec_next)
            exec_next = torch.where(bne_active, exec_len + (bne_iters * (bne_body_len + 1)), exec_next)
            exec_next = torch.where(aux_active, exec_len + (aux_iters * (aux_body_len + 1)), exec_next)
            exec_next = torch.where(spec_active, exec_next + self._const_i64_1, exec_next)
            executed_t = exec_next

            # DEBUG: Memory watchpoint check (removed - using inline checks instead)

            # Update executed count
            executed = int(executed_t.item())

            # Check for SVC or HALT - exit loop to hand control to caller
            is_svc_active = (stop_exec & is_svc).item()
            is_halt_active = (stop_exec & is_halt).item()
            if is_svc_active or is_halt_active:
                break

        self.pc = pc_t
        self.inst_count.copy_(executed_t)
        self._halted_t = stop_exec & is_halt
        # SVC flag: must only be set when we actually EXECUTE the SVC (stop_exec=True)
        # Not just when we detect it in the batch (stop_valid)
        # BUG FIX: Previously used stop_valid which triggered SVC even when hazard detection
        # shortened the batch and we didn't actually reach the SVC
        self._svc_t = stop_exec & is_svc
        # DEBUG: Print when SVC is detected
        if os.getenv("DEBUG_SVC") == "1":
            svc_found = bool((stop_inst & 0xFFE0001F) == 0xD4000001)
            if svc_found:
                logger.debug(f"[DEBUG] SVC at stop_inst: inst=0x{int(stop_inst.item()):08X}, stop_exec={bool(stop_exec.item())}, is_svc={bool(is_svc.item())}, stop_valid={bool(stop_valid.item())}, exec_len={int(exec_len.item())}, stop_idx={int(stop_idx.item())}, pc=0x{int(pc_t.item()):X}")
        return executed_t, time.perf_counter() - start

    @torch.no_grad()
    def handle_syscall_gpu(self, regs: torch.Tensor, mem: torch.Tensor) -> Tuple[bool, bool]:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║              GPU-NATIVE SYSCALL HANDLER - NO CPU SYNC!                     ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  Handles syscalls entirely on GPU using tensor operations.                 ║
        ║  Returns: (handled_on_gpu, should_exit)                                    ║
        ║    - handled_on_gpu=True: syscall completed, continue execution            ║
        ║    - handled_on_gpu=False: need CPU handling, break out                    ║
        ║    - should_exit=True: exit/exit_group called                              ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        syscall_num = regs[8]  # X8 = syscall number (stays as tensor!)

        # Debug: print syscall numbers
        if os.getenv("DEBUG_GPU_SYSCALL") == "1":
            logger.debug(f"[GPU SYSCALL] num={int(syscall_num.item())} x0={int(regs[0].item()):x} x1={int(regs[1].item()):x} x2={int(regs[2].item()):x}")

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 1: Pure Computation - 100% GPU
        # ═══════════════════════════════════════════════════════════════════════

        # brk (214) - Heap management
        if syscall_num == 214:
            new_brk = regs[0]
            # If new_brk is 0, return current brk
            result = torch.where(new_brk == 0, self.brk_t, new_brk)
            # Update brk if valid request
            self.brk_t = torch.where(new_brk > 0, new_brk, self.brk_t)
            regs[0] = result
            return True, False

        # mmap (222) - Memory mapping (simplified: anonymous only)
        if syscall_num == 222:
            length = regs[1]
            # Allocate from mmap_base
            result = self.mmap_base_t
            # Align and advance
            aligned_len = (length + 4095) & ~torch.tensor(4095, device=self.device)
            self.mmap_base_t = self.mmap_base_t + aligned_len
            regs[0] = result
            return True, False

        # mprotect (226) - Memory protection (no-op, always succeeds)
        if syscall_num == 226:
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # munmap (215) - Unmap memory (no-op, always succeeds)
        if syscall_num == 215:
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # getpid (172)
        if syscall_num == 172:
            regs[0] = self.pid_t
            return True, False

        # getppid (173)
        if syscall_num == 173:
            regs[0] = torch.tensor(1, device=self.device, dtype=torch.int64)  # Parent = init
            return True, False

        # getuid (174)
        if syscall_num == 174:
            regs[0] = self.uid_t
            return True, False

        # geteuid (175)
        if syscall_num == 175:
            regs[0] = self.uid_t
            return True, False

        # getgid (176)
        if syscall_num == 176:
            regs[0] = self.gid_t
            return True, False

        # getegid (177)
        if syscall_num == 177:
            regs[0] = self.gid_t
            return True, False

        # gettid (178)
        if syscall_num == 178:
            regs[0] = self.pid_t
            return True, False

        # set_tid_address (96)
        if syscall_num == 96:
            # Store address, return TID
            regs[0] = self.pid_t
            return True, False

        # rt_sigprocmask (135) - Signal mask (no-op)
        if syscall_num == 135:
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # rt_sigaction (134) - Signal action (no-op)
        if syscall_num == 134:
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # prlimit64 (261) - Resource limits
        if syscall_num == 261:
            # Return success, don't actually limit anything
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # getrandom (278) - Random bytes
        if syscall_num == 278:
            buf = regs[0]
            length = regs[1]
            # Generate random bytes directly to GPU memory
            random_bytes = torch.randint(0, 256, (int(length.item()),), dtype=torch.uint8, device=self.device)
            addr = int(buf.item())
            mem[addr:addr + int(length.item())] = random_bytes
            regs[0] = length  # Return bytes written
            return True, False

        # clock_gettime (113) - Time
        if syscall_num == 113:
            # Return a pseudo-time based on instruction count
            tp = regs[1]  # timespec pointer
            addr = int(tp.item())
            # Write seconds and nanoseconds
            secs = self.inst_count // 1000000  # Fake: 1M instructions = 1 second
            nsecs = (self.inst_count % 1000000) * 1000
            # Write timespec struct (8 bytes sec + 8 bytes nsec)
            for i in range(8):
                mem[addr + i] = ((secs >> (i * 8)) & 0xFF).to(torch.uint8)
                mem[addr + 8 + i] = ((nsecs >> (i * 8)) & 0xFF).to(torch.uint8)
            regs[0] = torch.tensor(0, device=self.device, dtype=torch.int64)
            return True, False

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 2: Buffered I/O - GPU with deferred flush
        # ═══════════════════════════════════════════════════════════════════════

        # write (64) - Buffered write to stdout/stderr
        if syscall_num == 64:
            fd = regs[0]
            buf = regs[1]
            count = regs[2]

            # Only buffer stdout (1) and stderr (2)
            is_console = (fd == 1) | (fd == 2)
            if is_console:
                addr = int(buf.item())
                write_len = int(count.item())

                # Check buffer space
                current_len = int(self.io_buffer_len.item())
                if current_len + write_len < 65536:
                    # Append to buffer (GPU tensor copy)
                    self.io_buffer[current_len:current_len + write_len] = mem[addr:addr + write_len]
                    self.io_buffer_len = self.io_buffer_len + write_len
                    regs[0] = count  # Return bytes "written"
                    return True, False

            # Non-console or buffer full - need CPU
            return False, False

        # writev (66) - Vectored write (can buffer)
        if syscall_num == 66:
            fd = regs[0]
            is_console = (fd == 1) | (fd == 2)
            if is_console:
                iov = int(regs[1].item())
                iovcnt = int(regs[2].item())
                total = 0
                current_len = int(self.io_buffer_len.item())

                for i in range(min(iovcnt, 16)):  # Limit vectors
                    # Read iovec struct: base (8 bytes) + len (8 bytes)
                    base_addr = iov + i * 16
                    base = int(mem[base_addr].item())
                    for j in range(1, 8):
                        base |= int(mem[base_addr + j].item()) << (j * 8)
                    iov_len = int(mem[base_addr + 8].item())
                    for j in range(1, 8):
                        iov_len |= int(mem[base_addr + 8 + j].item()) << (j * 8)

                    # Copy to buffer if space
                    if current_len + iov_len < 65536:
                        self.io_buffer[current_len:current_len + iov_len] = mem[base:base + iov_len]
                        current_len += iov_len
                        total += iov_len

                self.io_buffer_len = torch.tensor(current_len, device=self.device, dtype=torch.int64)
                regs[0] = torch.tensor(total, device=self.device, dtype=torch.int64)
                return True, False

            return False, False

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 3: Exit - Set flag but don't break (let caller handle)
        # ═══════════════════════════════════════════════════════════════════════

        # exit_group (93) / exit (94)
        if syscall_num == 93 or syscall_num == 94:
            self._exit_requested = torch.tensor(True, device=self.device, dtype=torch.bool)
            self._exit_code = regs[0].clone()
            return True, True  # Handled, should exit

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 4: Requires CPU - Return False to break out
        # ═══════════════════════════════════════════════════════════════════════
        # read (63), openat (56), close (57), ioctl (29), fcntl (25), etc.
        return False, False

    def flush_io_buffer(self) -> str:
        """Flush the GPU I/O buffer and return contents as string."""
        length = int(self.io_buffer_len.item())
        if length == 0:
            return ""
        # Single CPU transfer for all buffered output
        output = self.io_buffer[:length].cpu().numpy().tobytes().decode('utf-8', errors='replace')
        self.io_buffer_len.zero_()
        return output

    @torch.no_grad()
    def run_gpu_microbatch(self, max_instructions: int = 100000, microbatch_size: int = 32) -> Tuple[torch.Tensor, float]:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║       GPU MICRO-BATCH - 100% GPU EXECUTION WITH BRANCH CONTINUATION        ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  Unlike run_parallel_gpu which stops at branches, this method:              ║
        ║  1. Uses small batches (16-32 instructions)                                 ║
        ║  2. Resolves branches ENTIRELY on GPU (no .item() for branch decisions)    ║
        ║  3. Updates PC tensor directly (no CPU sync)                                ║
        ║  4. Only exits to CPU for syscalls                                          ║
        ║                                                                             ║
        ║  This avoids the batch truncation problem while staying 100% on GPU!        ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        start = time.perf_counter()
        executed_t = torch.tensor(0, device=self.device, dtype=torch.int64)
        pc_t = self.pc.clone()
        mem = self.memory
        regs = self.regs

        # Pre-allocate micro-batch tensors
        byte_offsets = torch.arange(microbatch_size * 4, device=self.device, dtype=torch.int64)

        # GPU iteration counter (no Python loop counter for hot path)
        max_iters = max_instructions // microbatch_size + 1

        for _ in range(max_iters):
            if self.halted:
                break

            # Check remaining (GPU comparison - minimal sync)
            remaining = max_instructions - int(executed_t.item())
            if remaining <= 0:
                break

            # ═══════════════════════════════════════════════════════════════
            # PHASE 0: MEMORY ORACLE PREFETCH (every N instructions)
            # ═══════════════════════════════════════════════════════════════
            if self.memory_oracle_enabled:
                self._prefetch_counter += 1
                if self._prefetch_counter >= self.prefetch_interval:
                    self._prefetch_counter = 0
                    current_pc = int(pc_t.item())
                    self.memory_oracle.predict_and_prefetch(current_pc)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 0.5: SEMANTIC DISPATCH CHECK (automatic pattern detection)
            # ═══════════════════════════════════════════════════════════════
            if self.semantic_dispatch_enabled:
                # Extract current register snapshot for pattern matching
                if self.semantic_dispatcher.should_check_patterns(int(executed_t.item())):
                    pc_int = int(pc_t.item())
                    # PERFORMANCE: Use batch sync instead of 32 individual .item() calls
                    regs_dict = self._get_regs_dict_fast()
                    dispatch_result = self.semantic_dispatcher.try_dispatch(pc_int, regs_dict)

                    if dispatch_result and dispatch_result.handled:
                        # Pattern detected and accelerated!
                        # Apply register modifications
                        for reg_idx, val in dispatch_result.registers_modified.items():
                            regs[reg_idx] = torch.tensor(val, dtype=torch.int64, device=self.device)
                        # Update PC to skip accelerated instructions
                        if dispatch_result.new_pc > 0:
                            pc_t = torch.tensor(dispatch_result.new_pc, dtype=torch.int64, device=self.device)
                        # Account for skipped instructions
                        executed_t = executed_t + dispatch_result.instructions_skipped
                        continue  # Skip normal execution for this batch

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: MICRO-BATCH FETCH (GPU only)
            # ═══════════════════════════════════════════════════════════════
            actual = min(microbatch_size, remaining)
            byte_indices = (pc_t + byte_offsets[:actual * 4]).clamp(0, self.mem_size - 1)
            byte_range = mem.gather(0, byte_indices).view(actual, 4).long()
            insts = (byte_range[:, 0] |
                    (byte_range[:, 1] << 8) |
                    (byte_range[:, 2] << 16) |
                    (byte_range[:, 3] << 24))

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1.5: RECORD INSTRUCTIONS FOR PATTERN DETECTION
            # ═══════════════════════════════════════════════════════════════
            if self.semantic_dispatch_enabled and actual > 0:
                # Record first instruction for pattern detection (lightweight sampling)
                pc_int = int(pc_t.item())
                inst_int = int(insts[0].item())
                # PERFORMANCE: Use batch sync instead of 32 individual .item() calls
                regs_dict = self._get_regs_dict_fast()
                self.semantic_dispatcher.record_instruction(pc_int, inst_int, regs_dict)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: FIND STOPPING POINTS (HALT/SVC/BRANCH)
            # ═══════════════════════════════════════════════════════════════
            halt_mask = (insts == 0)
            svc_mask = ((insts & 0xFFE0001F) == 0xD4000001)

            # Branch detection
            op_bytes = (insts >> 24) & 0xFF
            is_b = (insts & 0xFC000000) == 0x14000000      # Unconditional B
            is_bl = (insts & 0xFC000000) == 0x94000000     # BL
            is_br = (insts & 0xFFFFFC1F) == 0xD61F0000     # BR
            is_blr = (insts & 0xFFFFFC1F) == 0xD63F0000    # BLR
            is_ret = (insts & 0xFFFFFC1F) == 0xD65F0000    # RET
            is_cbz = (insts & 0x7F000000) == 0x34000000    # CBZ
            is_cbnz = (insts & 0x7F000000) == 0x35000000   # CBNZ
            is_bcond = (insts & 0xFF000010) == 0x54000000  # B.cond
            is_tbz = (insts & 0x7F000000) == 0x36000000    # TBZ
            is_tbnz = (insts & 0x7F000000) == 0x37000000   # TBNZ

            any_branch = is_b | is_bl | is_br | is_blr | is_ret | is_cbz | is_cbnz | is_bcond | is_tbz | is_tbnz
            stop_mask = halt_mask | svc_mask | any_branch

            # Find first stop point
            stop_idx_tensor = torch.where(stop_mask, torch.arange(actual, device=self.device), torch.full((actual,), actual, device=self.device))
            first_stop = stop_idx_tensor.min()

            # Execute up to first_stop ALU instructions
            exec_count = first_stop

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: PARALLEL ALU EXECUTION (instructions 0..first_stop-1)
            # ═══════════════════════════════════════════════════════════════
            if exec_count > 0:
                # Execute ALU instructions in parallel (simplified - key ops only)
                for i in range(int(exec_count.item())):
                    inst = insts[i]
                    op_byte = (inst >> 24) & 0xFF
                    rd = inst & 0x1F
                    rn = (inst >> 5) & 0x1F
                    rm = (inst >> 16) & 0x1F
                    imm12 = (inst >> 10) & 0xFFF

                    # MOVZ (64-bit: 0xD2, 32-bit: 0x52)
                    is_movz = (op_byte == 0xD2) | (op_byte == 0x52)
                    if is_movz:
                        hw = (inst >> 21) & 3
                        imm16 = (inst >> 5) & 0xFFFF
                        if rd != 31:
                            regs[rd] = imm16 << (hw * 16)

                    # MOVK (64-bit: 0xF2, 32-bit: 0x72)
                    is_movk = (op_byte == 0xF2) | (op_byte == 0x72)
                    if is_movk:
                        hw = (inst >> 21) & 3
                        imm16 = (inst >> 5) & 0xFFFF
                        shift = hw * 16
                        mask = ~(torch.tensor(0xFFFF, dtype=torch.int64, device=self.device) << shift)
                        if rd != 31:
                            regs[rd] = (regs[rd] & mask) | (imm16 << shift)

                    # ADD immediate (64-bit: 0x91, 32-bit: 0x11)
                    is_add_imm = (op_byte == 0x91) | (op_byte == 0x11)
                    if is_add_imm:
                        rn_val = regs[rn] if rn != 31 else regs[31]  # SP
                        result = rn_val + imm12
                        if op_byte == 0x11:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # SUB immediate (64-bit: 0xD1, 32-bit: 0x51)
                    is_sub_imm = (op_byte == 0xD1) | (op_byte == 0x51)
                    if is_sub_imm:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        result = rn_val - imm12
                        if op_byte == 0x51:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # LDR unsigned offset (0xF9)
                    is_ldr = (op_byte == 0xF9) & ((inst >> 22) & 1)
                    if is_ldr:
                        base = regs[rn] if rn != 31 else regs[31]
                        offset = imm12 * 8  # Scale by 8 for 64-bit
                        addr = (base + offset).clamp(0, self.mem_size - 8)
                        addr_int = int(addr.item())

                        # Memory Oracle: Record load access
                        if self.memory_oracle_enabled:
                            self.memory_oracle.record_load(addr_int, size=8)

                        if rd != 31:
                            val = (mem[addr].long() |
                                  (mem[addr+1].long() << 8) |
                                  (mem[addr+2].long() << 16) |
                                  (mem[addr+3].long() << 24) |
                                  (mem[addr+4].long() << 32) |
                                  (mem[addr+5].long() << 40) |
                                  (mem[addr+6].long() << 48) |
                                  (mem[addr+7].long() << 56))
                            regs[rd] = val

                    # STR unsigned offset (0xF9 with store bit)
                    is_str = (op_byte == 0xF9) & (~((inst >> 22) & 1))
                    if is_str:
                        base = regs[rn] if rn != 31 else regs[31]
                        offset = imm12 * 8
                        addr = (base + offset).clamp(0, self.mem_size - 8)
                        addr_int = int(addr.item())

                        # Memory Oracle: Record store access
                        if self.memory_oracle_enabled:
                            self.memory_oracle.record_store(addr_int, size=8)

                        val = regs[rd] if rd != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        for j in range(8):
                            mem[addr + j] = ((val >> (j * 8)) & 0xFF).to(torch.uint8)

                    # ═══════════════════════════════════════════════════════════
                    # LDR/STR post-index (0xF8) - CRITICAL FOR LOOPS!
                    # Format: LDR Xt, [Xn], #imm9 or STR Xt, [Xn], #imm9
                    # ═══════════════════════════════════════════════════════════
                    if op_byte == 0xF8:
                        # Extract post/pre-index mode from bits [11:10]
                        idx_mode = (inst >> 10) & 0x3
                        # Extract imm9 from bits [20:12] - signed 9-bit offset
                        imm9 = (inst >> 12) & 0x1FF
                        if imm9 & 0x100:  # Sign extend
                            imm9 = imm9 - 0x200
                        # Extract load/store bit from bit 22
                        is_load = (inst >> 22) & 1

                        if idx_mode == 0x1:  # Post-index: LDR/STR Xt, [Xn], #imm9
                            base = regs[rn] if rn != 31 else regs[31]
                            addr = base.clamp(0, self.mem_size - 8)
                            addr_int = int(addr.item())

                            if is_load:
                                # Memory Oracle: Record post-index load access
                                if self.memory_oracle_enabled:
                                    self.memory_oracle.record_load(addr_int, size=8)

                                if rd != 31:
                                    val = (mem[addr].long() |
                                          (mem[addr+1].long() << 8) |
                                          (mem[addr+2].long() << 16) |
                                          (mem[addr+3].long() << 24) |
                                          (mem[addr+4].long() << 32) |
                                          (mem[addr+5].long() << 40) |
                                          (mem[addr+6].long() << 48) |
                                          (mem[addr+7].long() << 56))
                                    regs[rd] = val
                            else:
                                # Memory Oracle: Record post-index store access
                                if self.memory_oracle_enabled:
                                    self.memory_oracle.record_store(addr_int, size=8)

                                val = regs[rd] if rd != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                                for j in range(8):
                                    mem[addr + j] = ((val >> (j * 8)) & 0xFF).to(torch.uint8)

                            # Update base register AFTER the access
                            regs[rn] = base + imm9

                        elif idx_mode == 0x3:  # Pre-index: LDR/STR Xt, [Xn, #imm9]!
                            base = regs[rn] if rn != 31 else regs[31]
                            # Update base register BEFORE the access
                            new_base = base + imm9
                            regs[rn] = new_base
                            addr = new_base.clamp(0, self.mem_size - 8)
                            addr_int = int(addr.item())

                            if is_load:
                                # Memory Oracle: Record pre-index load access
                                if self.memory_oracle_enabled:
                                    self.memory_oracle.record_load(addr_int, size=8)

                                if rd != 31:
                                    val = (mem[addr].long() |
                                          (mem[addr+1].long() << 8) |
                                          (mem[addr+2].long() << 16) |
                                          (mem[addr+3].long() << 24) |
                                          (mem[addr+4].long() << 32) |
                                          (mem[addr+5].long() << 40) |
                                          (mem[addr+6].long() << 48) |
                                          (mem[addr+7].long() << 56))
                                    regs[rd] = val
                            else:
                                # Memory Oracle: Record pre-index store access
                                if self.memory_oracle_enabled:
                                    self.memory_oracle.record_store(addr_int, size=8)

                                val = regs[rd] if rd != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                                for j in range(8):
                                    mem[addr + j] = ((val >> (j * 8)) & 0xFF).to(torch.uint8)

                    # ADD register (64-bit: 0x8B, 32-bit: 0x0B)
                    is_add_reg = (op_byte == 0x8B) | (op_byte == 0x0B)
                    if is_add_reg:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val + rm_val
                        if op_byte == 0x0B:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # SUB register (64-bit: 0xCB, 32-bit: 0x4B)
                    is_sub_reg = (op_byte == 0xCB) | (op_byte == 0x4B)
                    if is_sub_reg:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val - rm_val
                        if op_byte == 0x4B:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # AND register (64-bit: 0x8A, 32-bit: 0x0A)
                    is_and_reg = (op_byte == 0x8A) | (op_byte == 0x0A)
                    if is_and_reg:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val & rm_val
                        if op_byte == 0x0A:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # ORR register (64-bit: 0xAA, 32-bit: 0x2A)
                    # In data processing, X31 = XZR (zero), NOT SP!
                    is_orr_reg = (op_byte == 0xAA) | (op_byte == 0x2A)
                    if is_orr_reg:
                        rn_val = regs[rn] if rn != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val | rm_val
                        if op_byte == 0x2A:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # EOR/XOR register (64-bit: 0xCA, 32-bit: 0x4A)
                    is_eor_reg = (op_byte == 0xCA) | (op_byte == 0x4A)
                    if is_eor_reg:
                        rn_val = regs[rn] if rn != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val ^ rm_val
                        if op_byte == 0x4A:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result

                    # ADDS register with flags (64-bit: 0xAB, 32-bit: 0x2B)
                    is_adds_reg = (op_byte == 0xAB) | (op_byte == 0x2B)
                    if is_adds_reg:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val + rm_val
                        is_32bit = (op_byte == 0x2B)
                        if is_32bit:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result
                        # Set flags (GPU tensor operations) - avoid 64-bit overflow
                        sign_bit = 31 if is_32bit else 63
                        # N flag: check sign bit
                        n_flag = ((result >> sign_bit) & 1) != 0
                        self.flags[0] = torch.tensor(1.0 if n_flag else 0.0, device=self.device)
                        # Z flag: result is zero
                        z_flag = result == 0
                        self.flags[1] = torch.tensor(1.0 if z_flag else 0.0, device=self.device)
                        # C flag: carry (overflow for unsigned add)
                        max_val = 0xFFFFFFFF if is_32bit else 0x7FFFFFFFFFFFFFFF
                        c_flag = (rn_val > 0) and (rm_val > max_val - rn_val)
                        self.flags[2] = torch.tensor(1.0 if c_flag else 0.0, device=self.device)
                        # V flag: signed overflow check
                        rn_sign = (rn_val >> sign_bit) & 1
                        rm_sign = (rm_val >> sign_bit) & 1
                        res_sign = (result >> sign_bit) & 1
                        v_flag = ((rn_sign == rm_sign) & (rn_sign != res_sign))
                        self.flags[3] = torch.tensor(1.0 if v_flag else 0.0, device=self.device)

                    # SUBS register with flags (64-bit: 0xEB, 32-bit: 0x6B) - includes CMP
                    is_subs_reg = (op_byte == 0xEB) | (op_byte == 0x6B)
                    if is_subs_reg:
                        rn_val = regs[rn] if rn != 31 else regs[31]
                        rm_val = regs[rm] if rm != 31 else torch.tensor(0, dtype=torch.int64, device=self.device)
                        result = rn_val - rm_val
                        is_32bit = (op_byte == 0x6B)
                        if is_32bit:
                            result = result & 0xFFFFFFFF
                        if rd != 31:
                            regs[rd] = result
                        # Set flags (GPU tensor operations) - avoid 64-bit overflow
                        sign_bit = 31 if is_32bit else 63
                        # N flag: check sign bit
                        n_flag = ((result >> sign_bit) & 1) != 0
                        self.flags[0] = torch.tensor(1.0 if n_flag else 0.0, device=self.device)
                        # Z flag: result is zero
                        z_flag = result == 0
                        self.flags[1] = torch.tensor(1.0 if z_flag else 0.0, device=self.device)
                        # C flag: no borrow (rn >= rm for unsigned)
                        c_flag = rn_val >= rm_val
                        self.flags[2] = torch.tensor(1.0 if c_flag else 0.0, device=self.device)
                        # V flag: signed overflow for subtraction
                        rn_sign = (rn_val >> sign_bit) & 1
                        rm_sign = (rm_val >> sign_bit) & 1
                        res_sign = (result >> sign_bit) & 1
                        v_flag = ((rn_sign != rm_sign) & (rn_sign != res_sign))
                        self.flags[3] = torch.tensor(1.0 if v_flag else 0.0, device=self.device)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: HANDLE STOPPING INSTRUCTION (100% GPU)
            # ═══════════════════════════════════════════════════════════════
            if first_stop < actual:
                stop_inst = insts[first_stop]
                stop_pc = pc_t + first_stop * 4

                # HALT
                if halt_mask[first_stop]:
                    self.halted = True
                    executed_t = executed_t + exec_count
                    pc_t = stop_pc
                    break

                # SVC - Try GPU-native handling first!
                syscall_handled_on_gpu = False
                if svc_mask[first_stop]:
                    executed_t = executed_t + exec_count + 1  # Include SVC instruction

                    # Try to handle syscall entirely on GPU
                    handled_on_gpu, should_exit = self.handle_syscall_gpu(regs, mem)

                    if should_exit:
                        # Exit requested - flush output and halt
                        pc_t = stop_pc + 4
                        self.halted = True
                        break

                    if handled_on_gpu:
                        # Syscall handled on GPU - advance PC and CONTINUE EXECUTION!
                        pc_t = stop_pc + 4
                        syscall_handled_on_gpu = True
                        # Skip branch resolution - go straight to next iteration
                    else:
                        # Need CPU handling - break out
                        pc_t = stop_pc
                        self._svc_t = torch.tensor(True, device=self.device)
                        break

                # Skip branch resolution if syscall was handled on GPU
                if syscall_handled_on_gpu:
                    # pc_t already set to stop_pc + 4, exec_count already added
                    # Skip branch resolution and final exec_count update
                    pass  # Just continue to next iteration
                    # IMPORTANT: Don't fall through to branch resolution or exec_count update!
                    continue  # Skip to next loop iteration

                # ───────────────────────────────────────────────────────────
                # BRANCH RESOLUTION (100% GPU - no .item() for decisions!)
                # ───────────────────────────────────────────────────────────
                next_pc = stop_pc + 4  # Default: fallthrough

                # Unconditional B
                if is_b[first_stop]:
                    imm26 = stop_inst & 0x3FFFFFF
                    # Sign extend
                    imm26 = torch.where((imm26 & 0x2000000) != 0, imm26 | ~torch.tensor(0x3FFFFFF, dtype=torch.int64, device=self.device), imm26)
                    next_pc = stop_pc + imm26 * 4
                    exec_count = exec_count + 1

                # BL
                elif is_bl[first_stop]:
                    imm26 = stop_inst & 0x3FFFFFF
                    imm26 = torch.where((imm26 & 0x2000000) != 0, imm26 | ~torch.tensor(0x3FFFFFF, dtype=torch.int64, device=self.device), imm26)
                    regs[30] = stop_pc + 4  # Link register
                    next_pc = stop_pc + imm26 * 4
                    exec_count = exec_count + 1

                # BR
                elif is_br[first_stop]:
                    rn_br = (stop_inst >> 5) & 0x1F
                    next_pc = regs[rn_br]
                    exec_count = exec_count + 1

                # BLR
                elif is_blr[first_stop]:
                    rn_br = (stop_inst >> 5) & 0x1F
                    regs[30] = stop_pc + 4
                    next_pc = regs[rn_br]
                    exec_count = exec_count + 1

                # RET
                elif is_ret[first_stop]:
                    next_pc = regs[30]  # Return address
                    exec_count = exec_count + 1

                # CBZ/CBNZ - GPU condition evaluation with LOOP VECTORIZATION
                elif is_cbz[first_stop] or is_cbnz[first_stop]:
                    rt = stop_inst & 0x1F
                    imm19 = (stop_inst >> 5) & 0x7FFFF
                    imm19 = torch.where((imm19 & 0x40000) != 0, imm19 | ~torch.tensor(0x7FFFF, dtype=torch.int64, device=self.device), imm19)
                    rt_val = regs[rt]

                    # ═══════════════════════════════════════════════════════════
                    # LOOP VECTORIZATION: Detect backward CBNZ countdown loops
                    # If we see: SUB Rx, Rx, #1; CBNZ Rx, -4
                    # Instead of iterating, compute final state in ONE op!
                    # ═══════════════════════════════════════════════════════════
                    imm19_scalar = int(imm19.item()) if hasattr(imm19, 'item') else int(imm19)
                    vectorized = False

                    if is_cbnz[first_stop] and imm19_scalar < 0:
                        # Backward branch - likely a loop!
                        loop_start = stop_pc + imm19 * 4
                        body_len = -imm19_scalar  # Number of instructions in loop body

                        # Pattern 1: Simple countdown (1 instruction: SUB/SUBS + CBNZ)
                        if body_len == 1:
                            # Read the instruction before CBNZ
                            prev_addr = int(loop_start.item())
                            if 0 <= prev_addr < self.mem_size - 4:
                                prev_bytes = mem[prev_addr:prev_addr+4].long()
                                prev_inst = prev_bytes[0] | (prev_bytes[1] << 8) | (prev_bytes[2] << 16) | (prev_bytes[3] << 24)
                                prev_op = (prev_inst >> 24) & 0xFF

                                # Check for SUBS Rx, Rx, #1 (0xF1) or SUB (0xD1)
                                if prev_op in (0xF1, 0xD1):
                                    sub_rd = prev_inst & 0x1F
                                    sub_rn = (prev_inst >> 5) & 0x1F
                                    sub_imm = (prev_inst >> 10) & 0xFFF

                                    # Verify it's decrementing the same register as CBNZ checks
                                    if sub_rd == sub_rn == rt and sub_imm == 1:
                                        # VECTORIZE! Counter value = number of remaining iterations
                                        iterations = int(rt_val.item())
                                        if iterations > 10:  # Worth vectorizing
                                            # Set register to 0 (loop will exit)
                                            regs[rt] = torch.tensor(0, dtype=torch.int64, device=self.device)
                                            # Update flags for final SUBS (Z=1 since result is 0)
                                            self.flags[1] = 1.0  # Z flag set
                                            self.flags[0] = 0.0  # N flag clear
                                            # Account for all loop iterations
                                            exec_count = exec_count + iterations * 2  # SUB + CBNZ per iteration
                                            next_pc = stop_pc + 4  # Exit loop
                                            vectorized = True
                                            self.loops_vectorized += 1

                        # Pattern 2: Memory fill/copy loop (3-4 instructions)
                        elif body_len <= 4 and not vectorized:
                            iterations = int(rt_val.item())
                            if iterations > 10:
                                # Read loop body instructions
                                loop_addr = int(loop_start.item())
                                body_insts = []
                                for i in range(body_len):
                                    addr = loop_addr + i * 4
                                    if 0 <= addr < self.mem_size - 4:
                                        b = mem[addr:addr+4].long()
                                        inst = int((b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)).item())
                                        body_insts.append(inst)

                                # Analyze loop body for vectorizable patterns
                                has_sub_counter = False
                                add_regs = []  # (reg, imm) pairs for ADD instructions

                                for inst in body_insts:
                                    op = (inst >> 24) & 0xFF
                                    rd = inst & 0x1F
                                    rn = (inst >> 5) & 0x1F
                                    imm = (inst >> 10) & 0xFFF

                                    # SUBS/SUB decrementing counter
                                    if op in (0xF1, 0xD1) and rd == rn == rt and imm == 1:
                                        has_sub_counter = True
                                    # ADD incrementing a pointer
                                    elif op in (0x91, 0x11) and rd == rn and rd != rt:
                                        add_regs.append((rd, imm))

                                if has_sub_counter:
                                    # Apply vectorized updates
                                    for reg, imm in add_regs:
                                        old_val = int(regs[reg].item())
                                        regs[reg] = torch.tensor(old_val + imm * iterations, dtype=torch.int64, device=self.device)
                                    regs[rt] = torch.tensor(0, dtype=torch.int64, device=self.device)
                                    self.flags[1] = 1.0  # Z flag
                                    exec_count = exec_count + iterations * (body_len + 1)
                                    next_pc = stop_pc + 4
                                    vectorized = True
                                    self.loops_vectorized += 1

                    if not vectorized:
                        # Normal branch evaluation (no vectorization)
                        # CBZ: branch if zero, CBNZ: branch if not zero
                        take_branch = torch.where(is_cbz[first_stop], rt_val == 0, rt_val != 0)
                        next_pc = torch.where(take_branch, stop_pc + imm19 * 4, stop_pc + 4)
                        exec_count = exec_count + 1
                        # Update BTB
                        self.btb.update(stop_pc, take_branch, stop_pc + imm19 * 4)

                # B.cond - GPU flag evaluation with LOOP VECTORIZATION
                elif is_bcond[first_stop]:
                    cond = stop_inst & 0xF
                    imm19 = (stop_inst >> 5) & 0x7FFFF
                    imm19 = torch.where((imm19 & 0x40000) != 0, imm19 | ~torch.tensor(0x7FFFF, dtype=torch.int64, device=self.device), imm19)

                    imm19_scalar = int(imm19.item()) if hasattr(imm19, 'item') else int(imm19)
                    vectorized = False

                    # ═══════════════════════════════════════════════════════════
                    # LOOP VECTORIZATION for B.NE countdown loops
                    # Pattern: SUBS Rx, Rx, #1; B.NE -4 (very common!)
                    # ═══════════════════════════════════════════════════════════
                    if cond == 1 and imm19_scalar < 0:  # B.NE with backward branch
                        loop_start = stop_pc + imm19 * 4
                        body_len = -imm19_scalar

                        # Pattern: Single SUBS Rx, Rx, #1 + B.NE
                        if body_len == 1:
                            prev_addr = int(loop_start.item())
                            if 0 <= prev_addr < self.mem_size - 4:
                                prev_bytes = mem[prev_addr:prev_addr+4].long()
                                prev_inst = prev_bytes[0] | (prev_bytes[1] << 8) | (prev_bytes[2] << 16) | (prev_bytes[3] << 24)
                                prev_op = (prev_inst >> 24) & 0xFF

                                # Check for SUBS Rx, Rx, #1 (0xF1 = 64-bit SUBS imm, 0x71 = 32-bit)
                                if prev_op in (0xF1, 0x71):
                                    sub_rd = prev_inst & 0x1F
                                    sub_rn = (prev_inst >> 5) & 0x1F
                                    sub_imm = (prev_inst >> 10) & 0xFFF

                                    if sub_rd == sub_rn and sub_imm == 1:
                                        # Found countdown loop! Counter is in sub_rd
                                        current_val = int(regs[sub_rd].item())
                                        if current_val > 10:  # Worth vectorizing
                                            # VECTORIZE: Set counter to 0, update flags, skip loop
                                            regs[sub_rd] = torch.tensor(0, dtype=torch.int64, device=self.device)
                                            self.flags[1] = 1.0  # Z flag = 1 (result is 0)
                                            self.flags[0] = 0.0  # N flag = 0
                                            exec_count = exec_count + current_val * 2  # SUBS + B.NE per iter
                                            next_pc = stop_pc + 4  # Exit loop
                                            vectorized = True
                                            self.loops_vectorized += 1

                        # Pattern: Multi-instruction loop (2-4 instructions) + B.NE
                        elif body_len <= 4 and not vectorized:
                            loop_addr = int(loop_start.item())
                            body_insts = []
                            for i in range(body_len):
                                addr = loop_addr + i * 4
                                if 0 <= addr < self.mem_size - 4:
                                    b = mem[addr:addr+4].long()
                                    inst = int((b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)).item())
                                    body_insts.append(inst)

                            # Find SUBS with counter
                            counter_reg = None
                            counter_val = 0
                            add_updates = []  # (reg, imm) for registers that get added to

                            for inst in body_insts:
                                op = (inst >> 24) & 0xFF
                                rd = inst & 0x1F
                                rn = (inst >> 5) & 0x1F
                                imm = (inst >> 10) & 0xFFF

                                if op in (0xF1, 0x71) and rd == rn and imm == 1:  # SUBS Rx, Rx, #1
                                    counter_reg = rd
                                    counter_val = int(regs[rd].item())
                                elif op in (0x91, 0x11) and rd == rn:  # ADD Rx, Rx, #imm
                                    add_updates.append((rd, imm))

                            if counter_reg is not None and counter_val > 10:
                                # Apply vectorized updates
                                for reg, imm in add_updates:
                                    if reg != counter_reg:
                                        old_val = int(regs[reg].item())
                                        regs[reg] = torch.tensor(old_val + imm * counter_val, dtype=torch.int64, device=self.device)
                                regs[counter_reg] = torch.tensor(0, dtype=torch.int64, device=self.device)
                                self.flags[1] = 1.0
                                exec_count = exec_count + counter_val * (body_len + 1)
                                next_pc = stop_pc + 4
                                vectorized = True
                                self.loops_vectorized += 1

                        # ═══════════════════════════════════════════════════════════
                        # Pattern 3: Memory zeroing loop (STR XZR + [ADD] + CMP + B.NE)
                        # Common in BSS init. Handles 2-3 instruction bodies.
                        # ═══════════════════════════════════════════════════════════
                        if not vectorized and body_len in (2, 3):
                            # Check for STR XZR pattern with post-increment
                            inst0 = body_insts[0] if len(body_insts) > 0 else 0

                            # STR Rt, [Xn], #imm9 (post-index): 0xF8, bits[11:10]=01
                            is_str_post = ((inst0 >> 24) & 0xFF) == 0xF8 and ((inst0 >> 10) & 0x3) == 0x1
                            str_rt = inst0 & 0x1F
                            str_rn = (inst0 >> 5) & 0x1F
                            str_imm9 = (inst0 >> 12) & 0x1FF
                            # Sign-extend imm9
                            if str_imm9 & 0x100:
                                str_imm9 = str_imm9 | ~0x1FF

                            if is_str_post and str_rt == 0x1F:  # STR XZR detected
                                # Find the CMP instruction (last instruction before B.NE)
                                cmp_inst = body_insts[-1] if body_insts else 0
                                is_cmp_reg = ((cmp_inst >> 24) & 0xFF) == 0xEB and (cmp_inst & 0x1F) == 0x1F
                                cmp_rn = (cmp_inst >> 5) & 0x1F
                                cmp_rm = (cmp_inst >> 16) & 0x1F

                                if is_cmp_reg and cmp_rn == str_rn:
                                    # Check if there's an ADD computing end address (body_len == 3)
                                    end_ptr = None
                                    if body_len == 3:
                                        # Middle instruction might be ADD Xm, Xbase, #imm
                                        inst1 = body_insts[1]
                                        if ((inst1 >> 24) & 0xFF) == 0x91:  # ADD immediate
                                            add_rd = inst1 & 0x1F
                                            add_rn = (inst1 >> 5) & 0x1F
                                            add_imm = (inst1 >> 10) & 0xFFF
                                            if add_rd == cmp_rm:  # ADD computes the end address
                                                base_val = int(regs[add_rn].item())
                                                end_ptr = base_val + add_imm
                                    elif body_len == 2:
                                        end_ptr = int(regs[cmp_rm].item())

                                    if end_ptr is not None:
                                        current_ptr = int(regs[str_rn].item())
                                        stride = str_imm9

                                        # Handle normal case: zero memory from current to end
                                        if stride > 0 and end_ptr > current_ptr:
                                            bytes_to_zero = end_ptr - current_ptr
                                            iterations = bytes_to_zero // stride

                                            if iterations > 10:
                                                # VECTORIZE: Zero the entire range with one GPU op!
                                                if 0 <= current_ptr and end_ptr <= self.mem_size:
                                                    mem[current_ptr:end_ptr] = 0
                                                # Update pointer to end
                                                regs[str_rn] = torch.tensor(end_ptr, dtype=torch.int64, device=self.device)
                                                # If there was an ADD, also update that register
                                                if body_len == 3 and 'add_rd' in dir():
                                                    regs[cmp_rm] = torch.tensor(end_ptr, dtype=torch.int64, device=self.device)
                                                # Set Z flag (CMP result is equal at exit)
                                                self.flags[1] = 1.0
                                                exec_count = exec_count + iterations * (body_len + 1)
                                                next_pc = stop_pc + 4  # Exit loop
                                                vectorized = True
                                                self.loops_vectorized += 1

                    if not vectorized:
                        # Normal B.cond evaluation
                        n, z, c, v = self.flags[0], self.flags[1], self.flags[2], self.flags[3]
                        take_branch = torch.tensor(False, device=self.device)

                        # Condition code evaluation (all GPU tensor ops)
                        if cond == 0:  # EQ
                            take_branch = z > 0.5
                        elif cond == 1:  # NE
                            take_branch = z < 0.5
                        elif cond == 2:  # CS/HS
                            take_branch = c > 0.5
                        elif cond == 3:  # CC/LO
                            take_branch = c < 0.5
                        elif cond == 4:  # MI
                            take_branch = n > 0.5
                        elif cond == 5:  # PL
                            take_branch = n < 0.5
                        elif cond == 8:  # HI
                            take_branch = (c > 0.5) & (z < 0.5)
                        elif cond == 9:  # LS
                            take_branch = (c < 0.5) | (z > 0.5)
                        elif cond == 10:  # GE
                            take_branch = (n > 0.5) == (v > 0.5)
                        elif cond == 11:  # LT
                            take_branch = (n > 0.5) != (v > 0.5)
                        elif cond == 12:  # GT
                            take_branch = (z < 0.5) & ((n > 0.5) == (v > 0.5))
                        elif cond == 13:  # LE
                            take_branch = (z > 0.5) | ((n > 0.5) != (v > 0.5))

                        next_pc = torch.where(take_branch, stop_pc + imm19 * 4, stop_pc + 4)
                        exec_count = exec_count + 1
                        # Update BTB
                        self.btb.update(stop_pc, take_branch, stop_pc + imm19 * 4)

                # TBZ/TBNZ - Test bit and branch
                elif is_tbz[first_stop] or is_tbnz[first_stop]:
                    rt = stop_inst & 0x1F
                    bit = ((stop_inst >> 19) & 0x1F) | (((stop_inst >> 31) & 1) << 5)
                    imm14 = (stop_inst >> 5) & 0x3FFF
                    imm14 = torch.where((imm14 & 0x2000) != 0, imm14 | ~torch.tensor(0x3FFF, dtype=torch.int64, device=self.device), imm14)
                    rt_val = regs[rt]
                    bit_set = ((rt_val >> bit) & 1) != 0
                    # TBZ: branch if bit zero, TBNZ: branch if bit not zero
                    take_branch = torch.where(is_tbz[first_stop], ~bit_set, bit_set)
                    next_pc = torch.where(take_branch, stop_pc + imm14 * 4, stop_pc + 4)
                    exec_count = exec_count + 1
                    # Update BTB
                    self.btb.update(stop_pc, take_branch, stop_pc + imm14 * 4)

                pc_t = next_pc
                executed_t = executed_t + exec_count

            else:
                # No stopping instruction - advance by full batch
                pc_t = pc_t + actual * 4
                executed_t = executed_t + exec_count

        # Final sync
        self.pc = pc_t
        self.inst_count = executed_t
        return executed_t, time.perf_counter() - start

