"""
Legacy single-step execution engine for NeuralCPU.

Provides the detailed `step()` method that processes one instruction
at a time with full dispatch logic.
"""

import logging
import torch
import numpy as np
from typing import Optional

from ..constants import OpType, _u64_to_s64

logger = logging.getLogger(__name__)


class StepMixin:
    """Legacy single-step execution for NeuralCPU."""

    @torch.no_grad()
    def step(self) -> bool:
        """
        Execute one instruction with NEURAL EXTRACTION and GPU BRANCH DECISIONS.
        """
        if self.halted:
            return False

        pc_val = int(self.pc.item()) if hasattr(self.pc, "item") else int(self.pc)  # One .item() for instruction fetch
        inst = self.read32(pc_val)

        if inst == 0:
            self.halted = True
            return False

        # ════════════════════════════════════════════════════════════════════
        # SVC detection by bit pattern (op tables may not map 0xD4 → SVC)
        # ════════════════════════════════════════════════════════════════════
        if (inst & 0xFFE0001F) == 0xD4000001:
            op_type = OpType.SVC
            rd, rn, rm, imm, branch_off = 0, 0, 0, 0, 0
        else:
            op_type, rd, rn, rm, imm, branch_off = self._decode_neural(inst)

        # Try loop vectorization
        if op_type in [OpType.B_COND, OpType.CBZ, OpType.CBNZ] and branch_off < 0:
            if self._try_vectorize_loop(pc_val, branch_off, op_type, rd, imm):
                return True

        # ════════════════════════════════════════════════════════════════════
        # EXECUTION - ALL TENSOR OPERATIONS
        # ════════════════════════════════════════════════════════════════════

        if op_type == OpType.NOP:
            pass

        elif op_type == OpType.RET:
            self.pc = self.regs[30].clone()
            self.inst_count += 1
            return True

        elif op_type == OpType.MOVZ:
            if rd != 31:
                self.regs[rd] = imm

        elif op_type == OpType.MOVK:
            imm16 = imm & 0xFFFF
            hw = (imm >> 16) & 0x3
            # Use pre-computed mask tensor lookup - fully neural!
            mask = self.movk_masks[hw]
            shift = self.movk_shifts[hw]
            if rd != 31:
                self.regs[rd] = (self.regs[rd] & mask) | (imm16 << shift)

        elif op_type == OpType.ADD_IMM:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.add(self.regs[rn], imm)
            else:
                self.regs[rd] = self.regs[rn] + imm

        elif op_type == OpType.ADD_REG:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.add(self.regs[rn], self.regs[rm])
            else:
                self.regs[rd] = self.regs[rn] + self.regs[rm]

        elif op_type == OpType.SUB_IMM:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.sub(self.regs[rn], imm)
            else:
                self.regs[rd] = self.regs[rn] - imm

        elif op_type == OpType.SUB_REG:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.sub(self.regs[rn], self.regs[rm])
            else:
                self.regs[rd] = self.regs[rn] - self.regs[rm]

        elif op_type == OpType.MUL:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.mul(self.regs[rn], self.regs[rm])
            else:
                self.regs[rd] = self.regs[rn] * self.regs[rm]

        elif op_type == OpType.MOV_REG:
            if rd != 31:
                self.regs[rd] = self.regs[rm].clone()

        elif op_type in [OpType.CMP_IMM, OpType.CMP_REG]:
            a = self.regs[rn]
            b = imm if op_type == OpType.CMP_IMM else self.regs[rm]
            if self.use_neural_alu:
                n, z, c = self._neural_alu.cmp(a, b)
                self.flags[0] = n
                self.flags[1] = z
                self.flags[2] = c
            else:
                diff = a - b
                self.flags[0] = (diff < 0).float()  # N
                self.flags[1] = (diff == 0).float()  # Z
                self.flags[2] = (a >= b).float()  # C

        elif op_type == OpType.B:
            self.pc = self.pc + branch_off * 4
            self.inst_count += 1
            return True

        elif op_type == OpType.BL:
            self.regs[30] = self.pc + 4
            self.pc = self.pc + branch_off * 4
            self.inst_count += 1
            return True

        elif op_type == OpType.B_COND:
            # GPU BRANCH DECISION
            cond = torch.tensor(imm & 0xF, device=self.device)
            branch_type = torch.tensor(0, device=self.device)
            reg_val = torch.tensor(0.0, device=self.device)

            take = self.branch_decider(cond, self.flags, reg_val, branch_type)
            self.gpu_branch_decisions += 1

            if take.item() > 0.5:
                self.pc = self.pc + branch_off * 4
                self.inst_count += 1
                return True

        elif op_type == OpType.CBZ:
            # GPU BRANCH DECISION
            branch_type = torch.tensor(1, device=self.device)
            take = self.branch_decider(
                torch.tensor(0, device=self.device),
                self.flags,
                self.regs[rd].float(),
                branch_type
            )
            self.gpu_branch_decisions += 1

            if take.item() > 0.5:
                self.pc = self.pc + branch_off * 4
                self.inst_count += 1
                return True

        elif op_type == OpType.CBNZ:
            # GPU BRANCH DECISION
            branch_type = torch.tensor(2, device=self.device)
            take = self.branch_decider(
                torch.tensor(0, device=self.device),
                self.flags,
                self.regs[rd].float(),
                branch_type
            )
            self.gpu_branch_decisions += 1

            if take.item() > 0.5:
                self.pc = self.pc + branch_off * 4
                self.inst_count += 1
                return True

        elif op_type == OpType.STRB:
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size:
                # Memory Oracle: Record byte store access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=1)
                val = 0 if rd == 31 else (int(self.regs[rd].item()) & 0xFF)
                self.memory[addr] = val

                # Update framebuffer if in FB range
                if self.FB_BASE <= addr < self.FB_BASE + self.FB_SIZE:
                    fb_off = addr - self.FB_BASE
                    row = fb_off // self.FB_WIDTH
                    col = fb_off % self.FB_WIDTH
                    if 0 <= row < self.FB_HEIGHT and 0 <= col < self.FB_WIDTH:
                        self.framebuffer[row, col] = val

        elif op_type == OpType.LDRB:
            # LDRB has two main forms:
            # 1. Register offset: LDRB Wt, [Xn, Xm] - op byte 0x38, bit 21=1
            # 2. Immediate offset: LDRB Wt, [Xn, #imm] - op byte 0x39
            # For register offset (0x38), rm contains the offset register
            if rm != 0:  # Register offset mode - use rm as index register
                offset = int(self.regs[rm].item())
            else:
                offset = imm
            addr = int(self.regs[rn].item()) + offset
            if 0 <= addr < self.mem_size:
                # Memory Oracle: Record byte load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=1)
                self.regs[rd] = self.memory[addr].long()

        elif op_type == OpType.LDRB_POST:
            # LDRB Wt, [Xn], #imm - Load byte from base, then increment base
            # Used in string loops: while (*dst++ = *src++);
            base = int(self.regs[rn].item())
            addr = base  # Post-index: use base without offset first
            if 0 <= addr < self.mem_size:
                # Memory Oracle: Record byte load access (post-increment pattern)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=1)
                self.regs[rd] = self.memory[addr].long()
            # Update base AFTER the load
            self.regs[rn] = base + imm

        elif op_type == OpType.STRB_POST:
            # STRB Wt, [Xn], #imm - Store byte to base, then increment base
            base = int(self.regs[rn].item())
            addr = base  # Post-index: use base without offset first
            if 0 <= addr < self.mem_size:
                # Memory Oracle: Record byte store access (post-increment pattern)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=1)
                val = 0 if rd == 31 else (int(self.regs[rd].item()) & 0xFF)
                self.memory[addr] = val
            # Update base AFTER the store
            self.regs[rn] = base + imm

        elif op_type == OpType.STR:
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 7:
                # Memory Oracle: Record 64-bit store access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=8)
                val = 0 if rd == 31 else int(self.regs[rd].item())
                for i in range(8):
                    self.memory[addr + i] = (val >> (i * 8)) & 0xFF

        elif op_type == OpType.LDR:
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 7:
                # Memory Oracle: Record 64-bit load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=8)
                val = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                # Mask to signed 64-bit range to avoid overflow
                val = val & 0xFFFFFFFFFFFFFFFF
                if val > 0x7FFFFFFFFFFFFFFF:
                    val = val - 0x10000000000000000  # Convert to signed
                self.regs[rd] = val

        elif op_type == OpType.LDR_REG_OFF:
            # LDR Xt, [Xn, Xm, LSL #shift] - PURE TENSOR OPERATIONS ON GPU
            # Base + (offset_reg << shift), shift stored in imm (0 or 3)
            base = int(self.regs[rn].item())
            offset = int(self.regs[rm].item()) << imm  # imm is shift amount (0 or 3)
            addr = base + offset
            if 0 <= addr < self.mem_size - 7:
                # Memory Oracle: Record 64-bit load access (register offset)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=8)
                # GPU: Tensor slice read + tensor dot product for byte assembly
                bytes_tensor = self.memory[addr:addr+8].to(torch.int64)
                # Shift multipliers as tensor: [1, 256, 65536, ...]
                shifts = torch.tensor([1 << (i * 8) for i in range(8)], dtype=torch.int64, device=self.device)
                val = (bytes_tensor * shifts).sum().item()
                # Handle signed values
                if val > 0x7FFFFFFFFFFFFFFF:
                    val = val - 0x10000000000000000
                self.regs[rd] = val

        elif op_type == OpType.STR_REG_OFF:
            # STR Xt, [Xn, Xm, LSL #shift] - PURE TENSOR OPERATIONS ON GPU
            base = int(self.regs[rn].item())
            offset = int(self.regs[rm].item()) << imm
            addr = base + offset
            if 0 <= addr < self.mem_size - 7:
                # Memory Oracle: Record 64-bit store access (register offset)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=8)
                val = 0 if rd == 31 else (int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF)
                # GPU: Create byte tensor and write with tensor slicing
                byte_vals = torch.tensor([(val >> (i * 8)) & 0xFF for i in range(8)],
                                         dtype=torch.uint8, device=self.device)
                self.memory[addr:addr+8] = byte_vals

        # ═══════════════════════════════════════════════════════════════════════
        # POST/PRE-INDEX LOAD/STORE - CRITICAL FOR BUSYBOX
        # These modify the base register after/before the memory access
        # ═══════════════════════════════════════════════════════════════════════

        elif op_type == OpType.LDR_POST:
            # LDR Xt, [Xn], #imm - Load then update base
            # GPU: Load from base address, then add imm to base register
            base = int(self.regs[rn].item())
            if 0 <= base < self.mem_size - 7:
                # Memory Oracle: Record 64-bit load access (post-increment - common in loops!)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(base, size=8)
                # GPU tensor read
                bytes_tensor = self.memory[base:base+8].to(torch.int64)
                shifts = torch.tensor([1 << (i * 8) for i in range(8)], dtype=torch.int64, device=self.device)
                val = (bytes_tensor * shifts).sum().item()
                if val > 0x7FFFFFFFFFFFFFFF:
                    val = val - 0x10000000000000000
                self.regs[rd] = val
            # Update base register AFTER load
            self.regs[rn] = int(self.regs[rn].item()) + imm

        elif op_type == OpType.STR_POST:
            # STR Xt, [Xn], #imm - Store then update base
            base = int(self.regs[rn].item())
            if 0 <= base < self.mem_size - 7:
                # Memory Oracle: Record 64-bit store access (post-increment - common in loops!)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(base, size=8)
                val = 0 if rd == 31 else (int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF)
                byte_vals = torch.tensor([(val >> (i * 8)) & 0xFF for i in range(8)],
                                         dtype=torch.uint8, device=self.device)
                self.memory[base:base+8] = byte_vals
            # Update base register AFTER store
            self.regs[rn] = int(self.regs[rn].item()) + imm

        elif op_type == OpType.LDR_PRE:
            # LDR Xt, [Xn, #imm]! - Update base then load
            # Update base register BEFORE load
            new_base = int(self.regs[rn].item()) + imm
            self.regs[rn] = new_base
            if 0 <= new_base < self.mem_size - 7:
                # Memory Oracle: Record 64-bit load access (pre-increment)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(new_base, size=8)
                bytes_tensor = self.memory[new_base:new_base+8].to(torch.int64)
                shifts = torch.tensor([1 << (i * 8) for i in range(8)], dtype=torch.int64, device=self.device)
                val = (bytes_tensor * shifts).sum().item()
                if val > 0x7FFFFFFFFFFFFFFF:
                    val = val - 0x10000000000000000
                self.regs[rd] = val

        elif op_type == OpType.STR_PRE:
            # STR Xt, [Xn, #imm]! - Update base then store
            # Update base register BEFORE store
            new_base = int(self.regs[rn].item()) + imm
            self.regs[rn] = new_base
            if 0 <= new_base < self.mem_size - 7:
                # Memory Oracle: Record 64-bit store access (pre-increment)
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(new_base, size=8)
                val = 0 if rd == 31 else (int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF)
                byte_vals = torch.tensor([(val >> (i * 8)) & 0xFF for i in range(8)],
                                         dtype=torch.uint8, device=self.device)
                self.memory[new_base:new_base+8] = byte_vals

        # ═══════════════════════════════════════════════════════════════════════
        # NEW INSTRUCTIONS - ALL TENSOR OPERATIONS ON GPU
        # ═══════════════════════════════════════════════════════════════════════

        elif op_type == OpType.AND_REG:
            if rd != 31:
                if self.use_neural_alu:
                    self.regs[rd] = self._neural_alu.and_(self.regs[rn], self.regs[rm])
                else:
                    self.regs[rd] = self.regs[rn] & self.regs[rm]

        elif op_type == OpType.AND_IMM:
            # Handle large bitmask immediates by using Python int operations
            val = int(self.regs[rn].item()) & imm
            if rd != 31:
                self.regs[rd] = _u64_to_s64(val)

        elif op_type == OpType.ORR_REG:
            if rd != 31:
                if self.use_neural_alu:
                    self.regs[rd] = self._neural_alu.or_(self.regs[rn], self.regs[rm])
                else:
                    self.regs[rd] = self.regs[rn] | self.regs[rm]

        elif op_type == OpType.ORR_IMM:
            val = int(self.regs[rn].item()) | imm
            if rd != 31:
                self.regs[rd] = _u64_to_s64(val)

        elif op_type == OpType.EOR_REG:
            if rd != 31:
                if self.use_neural_alu:
                    self.regs[rd] = self._neural_alu.xor_(self.regs[rn], self.regs[rm])
                else:
                    self.regs[rd] = self.regs[rn] ^ self.regs[rm]

        elif op_type == OpType.EOR_IMM:
            val = int(self.regs[rn].item()) ^ imm
            if rd != 31:
                self.regs[rd] = _u64_to_s64(val)

        elif op_type == OpType.LSL_REG:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.shl(self.regs[rn], self.regs[rm])
            else:
                shift = int(self.regs[rm].item()) & 63
                self.regs[rd] = self.regs[rn] << shift

        elif op_type == OpType.LSL_IMM:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.shl(self.regs[rn], imm)
            else:
                self.regs[rd] = self.regs[rn] << (imm & 63)

        elif op_type == OpType.LSR_REG:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.shr(self.regs[rn], self.regs[rm])
            else:
                shift = int(self.regs[rm].item()) & 63
                val = int(self.regs[rn].item())
                self.regs[rd] = (val >> shift) if val >= 0 else ((val & 0xFFFFFFFFFFFFFFFF) >> shift)

        elif op_type == OpType.LSR_IMM:
            if self.use_neural_alu:
                self.regs[rd] = self._neural_alu.shr(self.regs[rn], imm)
            else:
                val = int(self.regs[rn].item())
                shift = imm & 63
                self.regs[rd] = (val >> shift) if val >= 0 else ((val & 0xFFFFFFFFFFFFFFFF) >> shift)

        elif op_type == OpType.ASR_REG:
            shift = int(self.regs[rm].item()) & 63
            # Arithmetic shift right (sign-extends)
            val = int(self.regs[rn].item())
            # Python handles sign extension automatically for >>
            self.regs[rd] = val >> shift

        elif op_type == OpType.ASR_IMM:
            val = int(self.regs[rn].item())
            self.regs[rd] = val >> (imm & 63)

        elif op_type == OpType.ROR_REG:
            shift = int(self.regs[rm].item()) & 63
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            if shift > 0:
                self.regs[rd] = ((val >> shift) | (val << (64 - shift))) & 0xFFFFFFFFFFFFFFFF
            else:
                self.regs[rd] = val

        elif op_type == OpType.MVN:
            self.regs[rd] = ~self.regs[rm]

        elif op_type == OpType.BIC:
            self.regs[rd] = self.regs[rn] & (~self.regs[rm])

        elif op_type == OpType.TST_REG:
            result = self.regs[rn] & self.regs[rm]
            self.flags[0] = (result < 0).float()  # N
            self.flags[1] = (result == 0).float()  # Z
            self.flags[2] = 0.0  # C (cleared for TST)
            # Result is discarded

        elif op_type == OpType.TST_IMM:
            result = self.regs[rn] & imm
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            self.flags[2] = 0.0

        elif op_type == OpType.NEG:
            self.regs[rd] = -self.regs[rm]

        elif op_type == OpType.BLR:
            self.regs[30] = self.pc + 4
            self.pc = self.regs[rn].clone()
            self.inst_count += 1
            return True

        elif op_type == OpType.BR:
            self.pc = self.regs[rn].clone()
            self.inst_count += 1
            return True

        elif op_type == OpType.SVC:
            # Syscall dispatch — X8 = syscall number, args in X0-X5
            syscall_nr = int(self.regs[8].item())

            if syscall_nr == 64:  # SYS_WRITE
                fd = int(self.regs[0].item())
                buf_addr = int(self.regs[1].item())
                length = int(self.regs[2].item())
                # Read bytes from memory
                length = min(length, self.mem_size - buf_addr)
                if length > 0 and 0 <= buf_addr < self.mem_size:
                    data = bytes(int(self.memory[buf_addr + i].item()) for i in range(length))
                    if fd in (1, 2):  # stdout / stderr
                        import sys as _sys
                        _sys.stdout.buffer.write(data)
                        _sys.stdout.buffer.flush()
                        if self._neural_display is not None:
                            self._neural_display.write(data)
                    self.regs[0] = length  # return bytes written
                else:
                    self.regs[0] = 0

            elif syscall_nr == 93:  # SYS_EXIT
                self.halted = True
                self.inst_count += 1
                self.pc += 4
                return False

            elif syscall_nr == 214:  # SYS_BRK
                if not hasattr(self, '_brk'):
                    self._brk = 0x60000
                req = int(self.regs[0].item())
                if req > 0:
                    self._brk = req
                self.regs[0] = self._brk

            else:
                logger.debug("SVC: unhandled syscall %d", syscall_nr)

            self.inst_count += 1
            self.pc += 4
            return True

        elif op_type == OpType.LDUR:
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 7:
                val = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                self.regs[rd] = val

        elif op_type == OpType.STUR:
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr + 7 < self.mem_size:
                val = int(self.regs[rd].item())
                for i in range(8):
                    self.memory[addr + i] = (val >> (i * 8)) & 0xFF

        elif op_type == OpType.LDP:
            # Load pair: load two registers from memory
            rt2 = (inst >> 10) & 0x1F  # Second destination register
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 15:
                val1 = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                val2 = sum(int(self.memory[addr + 8 + i].item()) << (i * 8) for i in range(8))
                # Convert to signed 64-bit (mask to prevent overflow)
                val1 = val1 & 0xFFFFFFFFFFFFFFFF
                val2 = val2 & 0xFFFFFFFFFFFFFFFF
                if val1 >= 0x8000000000000000: val1 -= 0x10000000000000000
                if val2 >= 0x8000000000000000: val2 -= 0x10000000000000000
                self.regs[rd] = val1
                self.regs[rt2] = val2

        elif op_type == OpType.STP:
            # Store pair: store two registers to memory
            rt2 = (inst >> 10) & 0x1F
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr + 15 < self.mem_size:
                val1 = int(self.regs[rd].item())
                val2 = int(self.regs[rt2].item())
                for i in range(8):
                    self.memory[addr + i] = (val1 >> (i * 8)) & 0xFF
                    self.memory[addr + 8 + i] = (val2 >> (i * 8)) & 0xFF

        # ═══════════════════════════════════════════════════════════════════════
        # LDP/STP with pre-index and post-index addressing
        # Post-index: load/store from base, THEN update base
        # Pre-index: update base FIRST, then load/store from base
        # ═══════════════════════════════════════════════════════════════════════

        elif op_type == OpType.LDP_POST:
            # Load pair post-index: LDP Xt1, Xt2, [Xn], #imm
            # Load from base address, then update base = base + imm
            rt2 = (inst >> 10) & 0x1F
            base = int(self.regs[rn].item())
            addr = base  # Use base without offset for post-index
            if 0 <= addr < self.mem_size - 15:
                val1 = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                val2 = sum(int(self.memory[addr + 8 + i].item()) << (i * 8) for i in range(8))
                # Convert to signed 64-bit (mask to prevent overflow)
                val1 = val1 & 0xFFFFFFFFFFFFFFFF
                val2 = val2 & 0xFFFFFFFFFFFFFFFF
                if val1 >= 0x8000000000000000: val1 -= 0x10000000000000000
                if val2 >= 0x8000000000000000: val2 -= 0x10000000000000000
                self.regs[rd] = val1
                self.regs[rt2] = val2
                # Update base register AFTER load (if not SP being used as dest)
                if rn != rd and rn != rt2:
                    self.regs[rn] = base + imm
                elif rn == 31:  # SP always gets updated
                    self.regs[rn] = base + imm

        elif op_type == OpType.STP_POST:
            # Store pair post-index: STP Xt1, Xt2, [Xn], #imm
            # Store to base address, then update base = base + imm
            rt2 = (inst >> 10) & 0x1F
            base = int(self.regs[rn].item())
            addr = base  # Use base without offset for post-index
            if 0 <= addr + 15 < self.mem_size:
                val1 = int(self.regs[rd].item())
                val2 = int(self.regs[rt2].item())
                for i in range(8):
                    self.memory[addr + i] = (val1 >> (i * 8)) & 0xFF
                    self.memory[addr + 8 + i] = (val2 >> (i * 8)) & 0xFF
                # Update base register AFTER store
                self.regs[rn] = base + imm

        elif op_type == OpType.LDP_PRE:
            # Load pair pre-index: LDP Xt1, Xt2, [Xn, #imm]!
            # Update base FIRST, then load from new address
            rt2 = (inst >> 10) & 0x1F
            base = int(self.regs[rn].item())
            addr = base + imm  # Update address first
            if 0 <= addr < self.mem_size - 15:
                val1 = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                val2 = sum(int(self.memory[addr + 8 + i].item()) << (i * 8) for i in range(8))
                # Convert to signed 64-bit (mask to prevent overflow)
                val1 = val1 & 0xFFFFFFFFFFFFFFFF
                val2 = val2 & 0xFFFFFFFFFFFFFFFF
                if val1 >= 0x8000000000000000: val1 -= 0x10000000000000000
                if val2 >= 0x8000000000000000: val2 -= 0x10000000000000000
                self.regs[rd] = val1
                self.regs[rt2] = val2
                # Update base register (writeback)
                if rn != rd and rn != rt2:
                    self.regs[rn] = addr
                elif rn == 31:  # SP always gets updated
                    self.regs[rn] = addr

        elif op_type == OpType.STP_PRE:
            # Store pair pre-index: STP Xt1, Xt2, [Xn, #imm]!
            # Update base FIRST, then store to new address
            rt2 = (inst >> 10) & 0x1F
            base = int(self.regs[rn].item())
            addr = base + imm  # Update address first
            if 0 <= addr + 15 < self.mem_size:
                val1 = int(self.regs[rd].item())
                val2 = int(self.regs[rt2].item())
                for i in range(8):
                    self.memory[addr + i] = (val1 >> (i * 8)) & 0xFF
                    self.memory[addr + 8 + i] = (val2 >> (i * 8)) & 0xFF
                # Update base register (writeback)
                self.regs[rn] = addr

        elif op_type == OpType.MADD:
            # Rd = Ra + Rn * Rm
            ra = (inst >> 10) & 0x1F
            self.regs[rd] = self.regs[ra] + self.regs[rn] * self.regs[rm]

        elif op_type == OpType.MSUB:
            # Rd = Ra - Rn * Rm
            ra = (inst >> 10) & 0x1F
            self.regs[rd] = self.regs[ra] - self.regs[rn] * self.regs[rm]

        elif op_type == OpType.SDIV:
            divisor = int(self.regs[rm].item())
            if divisor != 0:
                dividend = int(self.regs[rn].item())
                # Python handles signed division
                self.regs[rd] = dividend // divisor
            else:
                self.regs[rd] = 0  # Division by zero returns 0 on ARM64

        elif op_type == OpType.UDIV:
            divisor = int(self.regs[rm].item()) & 0xFFFFFFFFFFFFFFFF
            if divisor != 0:
                dividend = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
                self.regs[rd] = dividend // divisor
            else:
                self.regs[rd] = 0

        elif op_type == OpType.CLZ:
            # Count leading zeros
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            if val == 0:
                self.regs[rd] = 64
            else:
                count = 0
                mask = 1 << 63
                while (val & mask) == 0:
                    count += 1
                    mask >>= 1
                self.regs[rd] = count

        elif op_type == OpType.SXTW:
            # Sign extend 32-bit word to 64-bit
            val = int(self.regs[rn].item()) & 0xFFFFFFFF
            if val & 0x80000000:  # Negative in 32-bit
                # Convert to signed 64-bit representation for PyTorch
                val = val - 0x100000000  # Two's complement: subtract 2^32
            self.regs[rd] = val

        elif op_type == OpType.UXTB:
            # Zero extend byte to 64-bit
            self.regs[rd] = self.regs[rn] & 0xFF

        elif op_type == OpType.UXTH:
            # Zero extend halfword to 64-bit
            self.regs[rd] = self.regs[rn] & 0xFFFF

        # ═══════════════════════════════════════════════════════════════════════
        # BUSYBOX SUPPORT INSTRUCTIONS - ALL TENSOR OPERATIONS ON GPU
        # ═══════════════════════════════════════════════════════════════════════

        elif op_type == OpType.ADDS_IMM:
            a = self.regs[rn]
            b = imm
            result = a + b
            # rd=31 in ADDS means XZR (discard result, only set flags)
            if rd != 31:
                self.regs[rd] = result
            # Set flags as tensor operations
            self.flags[0] = (result < 0).float()  # N
            self.flags[1] = (result == 0).float()  # Z
            # C: carry out (simplified - check if result wrapped)
            a_val = int(a.item())
            self.flags[2] = float((a_val >= 0 and result.item() < a_val) or (a_val < 0 and result.item() >= 0))
            # V: signed overflow
            a_neg = a_val < 0
            b_neg = b < 0
            r_neg = int(result.item()) < 0
            self.flags[3] = float((a_neg == b_neg) and (a_neg != r_neg))

        elif op_type == OpType.ADDS_REG:
            a = self.regs[rn]
            b = self.regs[rm]
            result = a + b
            # rd=31 in ADDS means XZR (discard result, only set flags)
            if rd != 31:
                self.regs[rd] = result
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            a_val = int(a.item())
            b_val = int(b.item())
            r_val = int(result.item())
            self.flags[2] = float((a_val >= 0 and b_val >= 0 and r_val < 0) or (a_val < 0 and b_val < 0 and r_val >= 0))
            a_neg = a_val < 0
            b_neg = b_val < 0
            r_neg = r_val < 0
            self.flags[3] = float((a_neg == b_neg) and (a_neg != r_neg))

        elif op_type == OpType.SUBS_IMM:
            a = self.regs[rn]
            b = imm
            result = a - b
            # rd=31 in SUBS means XZR (discard result, only set flags)
            if rd != 31:
                self.regs[rd] = result
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            a_val = int(a.item())
            self.flags[2] = float(a_val >= b)  # C: no borrow for unsigned subtraction
            # V: signed overflow on subtraction
            a_neg = a_val < 0
            b_neg = b < 0
            r_neg = int(result.item()) < 0
            self.flags[3] = float((a_neg != b_neg) and (a_neg != r_neg))

        elif op_type == OpType.SUBS_REG:
            a = self.regs[rn]
            b = self.regs[rm]
            result = a - b
            # rd=31 in SUBS means XZR (discard result, only set flags)
            # This is different from data instructions where rd=31 means SP!
            if rd != 31:
                self.regs[rd] = result
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            a_val = int(a.item())
            b_val = int(b.item())
            r_val = int(result.item())
            self.flags[2] = float(a_val >= b_val)
            a_neg = a_val < 0
            b_neg = b_val < 0
            r_neg = r_val < 0
            self.flags[3] = float((a_neg != b_neg) and (a_neg != r_neg))

        elif op_type == OpType.LDRSB:
            # Load signed byte and sign-extend to 64 bits
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size:
                # Memory Oracle: Record signed byte load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=1)
                val = int(self.memory[addr].item())
                if val & 0x80:  # Negative in signed byte
                    val |= 0xFFFFFFFFFFFFFF00  # Sign extend
                self.regs[rd] = val

        elif op_type == OpType.LDRSH:
            # Load signed halfword and sign-extend to 64 bits
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 1:
                # Memory Oracle: Record signed halfword load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=2)
                val = int(self.memory[addr].item()) | (int(self.memory[addr + 1].item()) << 8)
                if val & 0x8000:  # Negative in signed halfword
                    val |= 0xFFFFFFFFFFFF0000
                self.regs[rd] = val

        elif op_type == OpType.LDRSW:
            # Load signed word and sign-extend to 64 bits
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 3:
                # Memory Oracle: Record signed word load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=4)
                val = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(4))
                if val & 0x80000000:  # Negative in signed word
                    val |= 0xFFFFFFFF00000000
                self.regs[rd] = val

        elif op_type == OpType.LDRH:
            # Load unsigned halfword
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 1:
                # Memory Oracle: Record unsigned halfword load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=2)
                val = int(self.memory[addr].item()) | (int(self.memory[addr + 1].item()) << 8)
                self.regs[rd] = val

        elif op_type == OpType.STRH:
            # Store halfword
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 1:
                # Memory Oracle: Record halfword store access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=2)
                val = int(self.regs[rd].item())
                self.memory[addr] = val & 0xFF
                self.memory[addr + 1] = (val >> 8) & 0xFF

        elif op_type == OpType.CSEL:
            # Conditional select: Rd = cond ? Rn : Rm
            cond_code = imm & 0xF
            take = self._eval_condition(cond_code)
            if take:
                self.regs[rd] = self.regs[rn].clone()
            else:
                self.regs[rd] = self.regs[rm].clone()

        elif op_type == OpType.CSINC:
            # Conditional select increment: Rd = cond ? Rn : Rm + 1
            cond_code = imm & 0xF
            take = self._eval_condition(cond_code)
            if take:
                self.regs[rd] = self.regs[rn].clone()
            else:
                self.regs[rd] = self.regs[rm] + 1

        elif op_type == OpType.CSINV:
            # Conditional select invert: Rd = cond ? Rn : ~Rm
            cond_code = imm & 0xF
            take = self._eval_condition(cond_code)
            if take:
                self.regs[rd] = self.regs[rn].clone()
            else:
                self.regs[rd] = ~self.regs[rm]

        elif op_type == OpType.CSNEG:
            # Conditional select negate: Rd = cond ? Rn : -Rm
            cond_code = imm & 0xF
            take = self._eval_condition(cond_code)
            if take:
                self.regs[rd] = self.regs[rn].clone()
            else:
                self.regs[rd] = -self.regs[rm]

        elif op_type == OpType.ADR:
            # PC-relative address
            pc_val = int(self.pc.item()) if hasattr(self.pc, "item") else int(self.pc)
            self.regs[rd] = pc_val + branch_off

        elif op_type == OpType.ADRP:
            # PC-relative page address (offset * 4KB)
            pc_val = int(self.pc.item()) if hasattr(self.pc, "item") else int(self.pc)
            page_base = pc_val & ~0xFFF  # Clear bottom 12 bits
            self.regs[rd] = page_base + (branch_off << 12)

        elif op_type == OpType.UBFM:
            # Unsigned bitfield move
            # ARM64: When imms >= immr, it's a simple shift right + mask (includes LSR alias)
            # When imms < immr, it's a rotate + mask (bit replication pattern)
            immr = (imm >> 6) & 0x3F
            imms = imm & 0x3F
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            if imms >= immr:
                # Simple shift right (handles LSR, UXTB, UXTH, UBFX)
                shifted = val >> immr
                mask = (1 << (imms + 1)) - 1 if imms < 63 else 0xFFFFFFFFFFFFFFFF
                self.regs[rd] = shifted & mask
            else:
                # Rotate right for bit replication pattern
                rotated = ((val >> immr) | (val << (64 - immr))) & 0xFFFFFFFFFFFFFFFF
                mask = (1 << (imms + 1)) - 1
                self.regs[rd] = rotated & mask

        elif op_type == OpType.SBFM:
            # Signed bitfield move
            immr = (imm >> 6) & 0x3F
            imms = imm & 0x3F
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            rotated = ((val >> immr) | (val << (64 - immr))) & 0xFFFFFFFFFFFFFFFF
            mask = (1 << (imms + 1)) - 1
            result = rotated & mask
            # Sign extend from bit imms
            if result & (1 << imms):
                result |= ~mask & 0xFFFFFFFFFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.EXTR:
            # Extract register (concatenate and extract bits)
            val_n = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            val_m = int(self.regs[rm].item()) & 0xFFFFFFFFFFFFFFFF
            lsb = imm
            # Concatenate [Rn:Rm] and extract 64 bits starting at lsb
            concat = (val_n << 64) | val_m
            result = (concat >> lsb) & 0xFFFFFFFFFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.TBZ:
            # Test bit and branch if zero
            val = int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF
            bit_pos = imm
            if not (val & (1 << bit_pos)):  # Bit is zero
                self.pc = self.pc + branch_off * 4
                self.inst_count += 1
                return True

        elif op_type == OpType.TBNZ:
            # Test bit and branch if not zero
            val = int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF
            bit_pos = imm
            if val & (1 << bit_pos):  # Bit is not zero
                self.pc = self.pc + branch_off * 4
                self.inst_count += 1
                return True

        elif op_type == OpType.RBIT:
            # Reverse bits in 64-bit value
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            result = 0
            for i in range(64):
                if val & (1 << i):
                    result |= 1 << (63 - i)
            self.regs[rd] = result

        elif op_type == OpType.REV:
            # Reverse bytes in 64-bit value
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            result = 0
            for i in range(8):
                byte = (val >> (i * 8)) & 0xFF
                result |= byte << ((7 - i) * 8)
            self.regs[rd] = result

        elif op_type == OpType.REV16:
            # Reverse bytes in each 16-bit halfword
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            result = 0
            for i in range(4):
                hw = (val >> (i * 16)) & 0xFFFF
                b0, b1 = hw & 0xFF, (hw >> 8) & 0xFF
                result |= ((b0 << 8) | b1) << (i * 16)
            self.regs[rd] = result

        elif op_type == OpType.REV32:
            # Reverse bytes in each 32-bit word
            val = int(self.regs[rn].item()) & 0xFFFFFFFFFFFFFFFF
            result = 0
            for i in range(2):
                word = (val >> (i * 32)) & 0xFFFFFFFF
                rev_word = 0
                for j in range(4):
                    byte = (word >> (j * 8)) & 0xFF
                    rev_word |= byte << ((3 - j) * 8)
                result |= rev_word << (i * 32)
            self.regs[rd] = result

        elif op_type == OpType.ANDS_REG:
            result = self.regs[rn] & self.regs[rm]
            self.regs[rd] = result
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            self.flags[2] = 0.0
            self.flags[3] = 0.0

        elif op_type == OpType.ANDS_IMM:
            result = self.regs[rn] & imm
            self.regs[rd] = result
            self.flags[0] = (result < 0).float()
            self.flags[1] = (result == 0).float()
            self.flags[2] = 0.0
            self.flags[3] = 0.0

        elif op_type == OpType.LDXR:
            # Load exclusive register (for atomics - simplified, no actual exclusivity tracking)
            addr = int(self.regs[rn].item())
            if 0 <= addr < self.mem_size - 7:
                val = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(8))
                self.regs[rd] = val

        elif op_type == OpType.STXR:
            # Store exclusive register (simplified - always succeeds)
            rs = (inst >> 16) & 0x1F  # Status register
            addr = int(self.regs[rn].item())
            if 0 <= addr + 7 < self.mem_size:
                val = int(self.regs[rd].item())
                for i in range(8):
                    self.memory[addr + i] = (val >> (i * 8)) & 0xFF
                self.regs[rs] = 0  # Success

        elif op_type in [OpType.DMB, OpType.DSB, OpType.ISB]:
            # Memory barriers - no-op in our single-threaded emulator
            pass

        elif op_type == OpType.MRS:
            # Move from system register - FULL KERNEL BOOT SUPPORT
            sysreg = imm
            result = self._read_system_register(sysreg)
            self.regs[rd] = result

        elif op_type == OpType.MSR:
            # Move to system register - FULL KERNEL BOOT SUPPORT
            val = int(self.regs[rd].item()) & 0xFFFFFFFFFFFFFFFF
            self._write_system_register(imm, val)

        elif op_type == OpType.ERET:
            # Exception return - restore state and return to ELR_EL1
            if self._exception_stack:
                ctx = self._exception_stack.pop()
                self.pc = torch.tensor(ctx['elr'], dtype=torch.int64, device=self.device)
                self.flags = ctx['spsr_flags'].clone()
                self.current_el = ctx['return_el']
                return True  # Continue execution at ELR_EL1
            else:
                # No saved context, use ELR_EL1 directly
                self.pc = torch.tensor(self.elr_el1, dtype=torch.int64, device=self.device)
                return True

        elif op_type == OpType.ADD_EXT:
            # ADD with extension (UXTW, SXTW, UXTB, etc.)
            ext_type = (inst >> 13) & 0x7  # Extension type
            shift = imm  # Shift amount (0-4)
            val = int(self.regs[rm].item())
            # Apply extension
            if ext_type == 0:  # UXTB
                val = val & 0xFF
            elif ext_type == 1:  # UXTH
                val = val & 0xFFFF
            elif ext_type == 2:  # UXTW
                val = val & 0xFFFFFFFF
            elif ext_type == 3:  # UXTX (no change)
                val = val & 0xFFFFFFFFFFFFFFFF
            elif ext_type == 4:  # SXTB
                val = val & 0xFF
                if val & 0x80:
                    val |= 0xFFFFFFFFFFFFFF00
            elif ext_type == 5:  # SXTH
                val = val & 0xFFFF
                if val & 0x8000:
                    val |= 0xFFFFFFFFFFFF0000
            elif ext_type == 6:  # SXTW
                val = val & 0xFFFFFFFF
                if val & 0x80000000:
                    val |= 0xFFFFFFFF00000000
            # Apply shift
            val = (val << shift) & 0xFFFFFFFFFFFFFFFF
            result = (int(self.regs[rn].item()) + val) & 0xFFFFFFFFFFFFFFFF
            self.regs[rd] = _u64_to_s64(result)

        elif op_type == OpType.SUB_EXT:
            ext_type = (inst >> 13) & 0x7
            shift = imm
            val = int(self.regs[rm].item())
            # Same extension logic as ADD_EXT
            if ext_type == 0:
                val = val & 0xFF
            elif ext_type == 1:
                val = val & 0xFFFF
            elif ext_type == 2:
                val = val & 0xFFFFFFFF
            elif ext_type == 3:
                val = val & 0xFFFFFFFFFFFFFFFF
            elif ext_type == 4:
                val = val & 0xFF
                if val & 0x80:
                    val |= 0xFFFFFFFFFFFFFF00
            elif ext_type == 5:
                val = val & 0xFFFF
                if val & 0x8000:
                    val |= 0xFFFFFFFFFFFF0000
            elif ext_type == 6:
                val = val & 0xFFFFFFFF
                if val & 0x80000000:
                    val |= 0xFFFFFFFF00000000
            val = (val << shift) & 0xFFFFFFFFFFFFFFFF
            result = (int(self.regs[rn].item()) - val) & 0xFFFFFFFFFFFFFFFF
            self.regs[rd] = _u64_to_s64(result)

        # ═══════════════════════════════════════════════════════════════════════
        # 32-BIT (W) INSTRUCTION EXECUTION - ALL ON GPU
        # 32-bit operations mask result to 32 bits
        # ═══════════════════════════════════════════════════════════════════════

        elif op_type == OpType.MOVZ_W:
            # MOVZ 32-bit - zero extend to 64-bit
            self.regs[rd] = imm & 0xFFFFFFFF

        elif op_type == OpType.MOVK_W:
            # MOVK 32-bit - keep other 16-bit portions
            imm16 = imm & 0xFFFF
            hw = (imm >> 16) & 0x1
            # Mask to 32-bit to avoid Python negative int overflow with PyTorch
            mask = (~(0xFFFF << (hw * 16))) & 0xFFFFFFFF
            val = int(self.regs[rd].item()) & 0xFFFFFFFF
            self.regs[rd] = ((val & mask) | (imm16 << (hw * 16))) & 0xFFFFFFFF

        elif op_type == OpType.MOV_W:
            # MOV 32-bit (ORR with WZR)
            self.regs[rd] = self.regs[rm] & 0xFFFFFFFF

        elif op_type == OpType.ADD_IMM_W:
            result = (self.regs[rn] + imm) & 0xFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.SUB_IMM_W:
            result = (self.regs[rn] - imm) & 0xFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.ADD_REG_W:
            result = (self.regs[rn] + self.regs[rm]) & 0xFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.SUB_REG_W:
            result = (self.regs[rn] - self.regs[rm]) & 0xFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.ADDS_IMM_W:
            a = int(self.regs[rn].item()) & 0xFFFFFFFF
            b = imm & 0xFFFFFFFF
            result = (a + b) & 0xFFFFFFFF
            # rd=31 in ADDS means WZR (discard result, only set flags)
            if rd != 31:
                self.regs[rd] = result
            # Set flags for 32-bit operation
            self.flags[0] = float((result & 0x80000000) != 0)  # N
            self.flags[1] = float(result == 0)  # Z
            self.flags[2] = float((a + b) > 0xFFFFFFFF)  # C
            a_neg = (a & 0x80000000) != 0
            r_neg = (result & 0x80000000) != 0
            self.flags[3] = float(a_neg != r_neg and imm > 0)  # V simplified

        elif op_type == OpType.SUBS_IMM_W:
            a = int(self.regs[rn].item()) & 0xFFFFFFFF
            b = imm & 0xFFFFFFFF
            result = (a - b) & 0xFFFFFFFF
            # rd=31 in SUBS means WZR (discard result, only set flags)
            if rd != 31:
                self.regs[rd] = result
            self.flags[0] = float((result & 0x80000000) != 0)
            self.flags[1] = float(result == 0)
            self.flags[2] = float(a >= b)
            a_neg = (a & 0x80000000) != 0
            b_neg = (b & 0x80000000) != 0
            r_neg = (result & 0x80000000) != 0
            self.flags[3] = float((a_neg != b_neg) and (a_neg != r_neg))

        elif op_type in [OpType.CMP_IMM_W, OpType.CMP_REG_W]:
            a = int(self.regs[rn].item()) & 0xFFFFFFFF
            b = (imm if op_type == OpType.CMP_IMM_W else int(self.regs[rm].item())) & 0xFFFFFFFF
            result = (a - b) & 0xFFFFFFFF
            self.flags[0] = float((result & 0x80000000) != 0)
            self.flags[1] = float(result == 0)
            self.flags[2] = float(a >= b)
            a_neg = (a & 0x80000000) != 0
            b_neg = (b & 0x80000000) != 0
            r_neg = (result & 0x80000000) != 0
            self.flags[3] = float((a_neg != b_neg) and (a_neg != r_neg))

        elif op_type == OpType.LDR_W:
            # Load 32-bit word
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr < self.mem_size - 3:
                # Memory Oracle: Record 32-bit word load access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_load(addr, size=4)
                val = sum(int(self.memory[addr + i].item()) << (i * 8) for i in range(4))
                self.regs[rd] = val  # Zero-extended to 64-bit

        elif op_type == OpType.STR_W:
            # Store 32-bit word
            addr = int(self.regs[rn].item()) + imm
            if 0 <= addr + 3 < self.mem_size:
                # Memory Oracle: Record 32-bit word store access
                if self.memory_oracle_enabled:
                    self.memory_oracle.record_store(addr, size=4)
                val = int(self.regs[rd].item()) & 0xFFFFFFFF
                for i in range(4):
                    self.memory[addr + i] = (val >> (i * 8)) & 0xFF

        elif op_type == OpType.CSEL_W:
            cond_code = imm & 0xF
            take = self._eval_condition(cond_code)
            if take:
                self.regs[rd] = self.regs[rn] & 0xFFFFFFFF
            else:
                self.regs[rd] = self.regs[rm] & 0xFFFFFFFF

        elif op_type == OpType.MADD_W:
            ra = (inst >> 10) & 0x1F
            result = (int(self.regs[ra].item()) + int(self.regs[rn].item()) * int(self.regs[rm].item())) & 0xFFFFFFFF
            self.regs[rd] = result

        elif op_type == OpType.MOVN:
            # Move NOT 64-bit - NEURAL: keep as GPU tensor compatible
            imm16 = imm & 0xFFFF
            hw = (imm >> 16) & 0x3
            val = ~(imm16 << (hw * 16)) & 0xFFFFFFFFFFFFFFFF
            # Convert to signed for int64 tensor
            if val > 0x7FFFFFFFFFFFFFFF:
                val = val - 0x10000000000000000
            self.regs[rd] = val

        elif op_type == OpType.MOVN_W:
            # Move NOT 32-bit
            imm16 = imm & 0xFFFF
            hw = (imm >> 16) & 0x1
            val = ~(imm16 << (hw * 16)) & 0xFFFFFFFF
            # No sign conversion needed for 32-bit (fits in int64)
            self.regs[rd] = val

        self.pc += 4
        self.inst_count += 1
        return True

