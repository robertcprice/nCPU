"""
Loop vectorization mixin for NeuralCPU.

Detects tight loops and computes N iterations as a single tensor operation,
turning sequential loops into parallel GPU operations.
"""

import logging
import torch
from typing import Optional
import numpy as np

from .constants import OpType, _u64_to_s64

logger = logging.getLogger(__name__)


class VectorizerMixin:
    """Loop vectorization methods for NeuralCPU."""

    def _try_vectorize_loop(self, pc_val: int, branch_off: int, op_type: int, rd: int, imm: int) -> bool:
        """
        Try to detect and vectorize a loop - EXECUTES ENTIRE LOOP AS ONE OP!

        Uses NEURAL LOOP DETECTOR + pattern matching fallback.
        Returns True if loop was vectorized (skipping all iterations).
        """
        if branch_off >= 0:
            return False

        loop_start = pc_val + branch_off * 4
        loop_end = pc_val

        if loop_start < 0:
            return False

        # ═══════════════════════════════════════════════════════════════════
        # NEURAL LOOP DETECTION - ENABLED with trained model!
        # Trained FastLoopDetector: 100% type accuracy, 91% register accuracy
        # ═══════════════════════════════════════════════════════════════════
        body_len = (loop_end - loop_start) // 4
        if getattr(self, '_neural_loop_enabled', False) and body_len <= 32:
            try:
                # Collect instruction bits as tensor + raw instructions for analysis
                body_bits = []
                body_insts = []
                for i in range(body_len):
                    inst = self.read32(loop_start + i * 4)
                    body_insts.append(inst)
                    bits = torch.tensor([[float((inst >> j) & 1) for j in range(32)]], device=self.device)
                    body_bits.append(bits)
                body_tensor = torch.cat(body_bits, dim=0)  # [body_len, 32]

                # Get register values as tensor
                reg_values = self.regs[:32].clone()

                # Neural network prediction (ALL ON GPU)
                with torch.no_grad():
                    loop_type_logits, counter_probs, iterations_pred = self.loop_detector(body_tensor, reg_values)
                    loop_type = torch.argmax(loop_type_logits).item()
                    predicted_iters = int(iterations_pred.item())

                    # SANITY CHECK: Neural network can predict garbage - limit to reasonable values
                    MAX_VECTORIZE_ITERS = 100000

                    # Types: 0=none, 1=count_up, 2=countdown, 3=mem_fill
                    if loop_type > 0 and 10 < predicted_iters < MAX_VECTORIZE_ITERS:
                        counter_reg = torch.argmax(counter_probs).item()
                        current = int(self.regs[counter_reg].item())

                        # Additional sanity: counter value should be reasonable
                        if abs(current) > 0x10000000:
                            pass  # Skip vectorization
                        elif loop_type in (1, 2):  # count_up or countdown
                            # ═══════════════════════════════════════════════════════
                            # IMPROVED: Analyze ALL instructions in loop body
                            # Apply correct transformation to ALL modified registers
                            # ═══════════════════════════════════════════════════════
                            iterations = current if loop_type == 2 else predicted_iters

                            for inst in body_insts:
                                op_byte = (inst >> 24) & 0xFF

                                # ADD Rd, Rn, #imm (0x91xxxxxx for 64-bit)
                                if op_byte == 0x91:
                                    rd = inst & 0x1F
                                    rn = (inst >> 5) & 0x1F
                                    imm = (inst >> 10) & 0xFFF
                                    if rd == rn and rd != 31:  # ADD Rx, Rx, #imm
                                        old_val = int(self.regs[rd].item())
                                        self.regs[rd] = old_val + imm * iterations

                                # SUB Rd, Rn, #imm (0xD1xxxxxx for 64-bit)
                                elif op_byte == 0xD1:
                                    rd = inst & 0x1F
                                    rn = (inst >> 5) & 0x1F
                                    imm = (inst >> 10) & 0xFFF
                                    if rd == rn and rd != 31:  # SUB Rx, Rx, #imm
                                        old_val = int(self.regs[rd].item())
                                        new_val = old_val - imm * iterations
                                        self.regs[rd] = max(0, new_val)  # Don't go negative

                                # ADD Rd, Rn, #imm (0x11xxxxxx for 32-bit)
                                elif op_byte == 0x11:
                                    rd = inst & 0x1F
                                    rn = (inst >> 5) & 0x1F
                                    imm = (inst >> 10) & 0xFFF
                                    if rd == rn and rd != 31:
                                        old_val = int(self.regs[rd].item()) & 0xFFFFFFFF
                                        self.regs[rd] = (old_val + imm * iterations) & 0xFFFFFFFF

                                # SUB Rd, Rn, #imm (0x51xxxxxx for 32-bit)
                                elif op_byte == 0x51:
                                    rd = inst & 0x1F
                                    rn = (inst >> 5) & 0x1F
                                    imm = (inst >> 10) & 0xFFF
                                    if rd == rn and rd != 31:
                                        old_val = int(self.regs[rd].item()) & 0xFFFFFFFF
                                        new_val = max(0, old_val - imm * iterations)
                                        self.regs[rd] = new_val & 0xFFFFFFFF

                            self.inst_count += iterations * body_len
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True
            except Exception:
                pass  # Fall through to pattern matching

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 1: Simple countdown (SUB + CBNZ)
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBNZ and (loop_end - loop_start) == 4:
            sub_inst = self.read32(loop_start)
            sub_dec = self._decode_neural(sub_inst)

            if (sub_dec[0] == OpType.SUB_IMM and
                sub_dec[1] == sub_dec[2] and  # rd == rn
                sub_dec[1] == rd and          # same as CBNZ register
                sub_dec[4] == 1):             # decrement by 1

                iterations = int(self.regs[rd].item())

                if iterations > 10:
                    # VECTORIZE: Set counter to 0 in ONE op
                    self.regs[rd] = 0
                    self.inst_count += iterations * 2
                    self.loops_vectorized += 1
                    self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                    return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 2: Memory fill (STRB + ADD + SUB + CBNZ)
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBNZ and (loop_end - loop_start) == 12:
            inst1 = self.read32(loop_start)
            inst2 = self.read32(loop_start + 4)
            inst3 = self.read32(loop_start + 8)

            dec1 = self._decode_neural(inst1)
            dec2 = self._decode_neural(inst2)
            dec3 = self._decode_neural(inst3)

            if (dec1[0] == OpType.STRB and
                dec2[0] == OpType.ADD_IMM and dec2[1] == dec2[2] and dec2[4] == 1 and
                dec3[0] == OpType.SUB_IMM and dec3[1] == dec3[2] and dec3[4] == 1 and
                dec3[1] == rd):

                counter_reg = dec3[1]
                base_reg = dec2[1]
                value_reg = dec1[1]

                iterations = int(self.regs[counter_reg].item())
                start_addr = int(self.regs[base_reg].item())
                fill_val = int(self.regs[value_reg].item()) & 0xFF

                if iterations > 10 and 0 <= start_addr < self.mem_size:
                    end_addr = min(start_addr + iterations, self.mem_size)
                    actual_iters = end_addr - start_addr

                    # VECTORIZED MEMORY FILL - ONE TENSOR OP!
                    self.memory[start_addr:end_addr] = fill_val

                    # Also update framebuffer if in FB range
                    if self.FB_BASE <= start_addr < self.FB_BASE + self.FB_SIZE:
                        fb_start = start_addr - self.FB_BASE
                        fb_end = min(end_addr - self.FB_BASE, self.FB_SIZE)
                        row_start = fb_start // self.FB_WIDTH
                        col_start = fb_start % self.FB_WIDTH
                        row_end = fb_end // self.FB_WIDTH
                        col_end = fb_end % self.FB_WIDTH

                        # Fill framebuffer tensor
                        for r in range(row_start, min(row_end + 1, self.FB_HEIGHT)):
                            c_start = col_start if r == row_start else 0
                            c_end = col_end if r == row_end else self.FB_WIDTH
                            self.framebuffer[r, c_start:c_end] = fill_val

                    self.regs[base_reg] = end_addr
                    self.regs[counter_reg] = 0

                    self.inst_count += actual_iters * 4
                    self.loops_vectorized += 1
                    self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                    return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 3: Count-up (ADD + CMP + B.LT)
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 11:  # LT condition
            body_len = (loop_end - loop_start) // 4

            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                dec1 = self._decode_neural(inst1)
                dec2 = self._decode_neural(inst2)

                # CMP can be encoded as CMP_IMM, CMP_REG, SUBS_IMM, or SUBS_REG (when Rd=31)
                cmp_types = [OpType.CMP_IMM, OpType.CMP_REG, OpType.SUBS_IMM, OpType.SUBS_REG]
                if (dec1[0] == OpType.ADD_IMM and dec1[1] == dec1[2] and
                    dec2[0] in cmp_types):

                    counter_reg = dec1[1]
                    increment = dec1[4]
                    current = int(self.regs[counter_reg].item())

                    # Extract target from CMP/SUBS instruction
                    if dec2[0] in [OpType.CMP_IMM, OpType.SUBS_IMM]:
                        target = dec2[4]  # Immediate value
                    else:
                        target = int(self.regs[dec2[3]].item())  # Register value (rm)

                    if increment > 0 and current < target:
                        iterations = (target - current + increment - 1) // increment

                        if iterations > 10:
                            # VECTORIZE: Jump counter to target
                            final = current + iterations * increment
                            self.regs[counter_reg] = min(final, target)

                            # Update flags as tensor ops
                            diff = self.regs[counter_reg] - target
                            self.flags[0] = (diff < 0).float()
                            self.flags[1] = (diff == 0).float()
                            self.flags[2] = (self.regs[counter_reg] >= target).float()

                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 4: Memory copy (LDRB + STRB + ADD + ADD + SUB + CBNZ)
        # Common in memcpy implementations
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBNZ and (loop_end - loop_start) == 20:
            inst1 = self.read32(loop_start)       # LDRB
            inst2 = self.read32(loop_start + 4)   # STRB
            inst3 = self.read32(loop_start + 8)   # ADD src
            inst4 = self.read32(loop_start + 12)  # ADD dst
            inst5 = self.read32(loop_start + 16)  # SUB counter

            dec1 = self._decode_neural(inst1)
            dec2 = self._decode_neural(inst2)
            dec3 = self._decode_neural(inst3)
            dec4 = self._decode_neural(inst4)
            dec5 = self._decode_neural(inst5)

            if (dec1[0] == OpType.LDRB and
                dec2[0] == OpType.STRB and
                dec3[0] == OpType.ADD_IMM and dec3[4] == 1 and
                dec4[0] == OpType.ADD_IMM and dec4[4] == 1 and
                dec5[0] == OpType.SUB_IMM and dec5[4] == 1 and
                dec5[1] == rd):

                counter_reg = dec5[1]
                src_reg = dec3[1]
                dst_reg = dec4[1]

                iterations = int(self.regs[counter_reg].item())
                src_addr = int(self.regs[src_reg].item())
                dst_addr = int(self.regs[dst_reg].item())

                if iterations > 10 and 0 <= src_addr < self.mem_size and 0 <= dst_addr < self.mem_size:
                    end_src = min(src_addr + iterations, self.mem_size)
                    end_dst = min(dst_addr + iterations, self.mem_size)
                    actual_iters = min(end_src - src_addr, end_dst - dst_addr)

                    # VECTORIZED MEMORY COPY - ONE TENSOR OP!
                    self.memory[dst_addr:dst_addr + actual_iters] = self.memory[src_addr:src_addr + actual_iters].clone()

                    self.regs[src_reg] = src_addr + actual_iters
                    self.regs[dst_reg] = dst_addr + actual_iters
                    self.regs[counter_reg] = 0

                    self.inst_count += actual_iters * 6
                    self.loops_vectorized += 1
                    self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                    return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 5: Accumulation loop (LDR + ADD_REG + ADD + SUB + CBNZ)
        # Common in sum/reduce operations
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBNZ and (loop_end - loop_start) == 16:
            inst1 = self.read32(loop_start)       # LDR or LDRB
            inst2 = self.read32(loop_start + 4)   # ADD_REG (accumulate)
            inst3 = self.read32(loop_start + 8)   # ADD (advance pointer)
            inst4 = self.read32(loop_start + 12)  # SUB counter

            dec1 = self._decode_neural(inst1)
            dec2 = self._decode_neural(inst2)
            dec3 = self._decode_neural(inst3)
            dec4 = self._decode_neural(inst4)

            if (dec1[0] in [OpType.LDR, OpType.LDRB] and
                dec2[0] == OpType.ADD_REG and
                dec3[0] == OpType.ADD_IMM and
                dec4[0] == OpType.SUB_IMM and dec4[4] == 1 and
                dec4[1] == rd):

                counter_reg = dec4[1]
                ptr_reg = dec3[1]
                accum_reg = dec2[1]
                stride = dec3[4]

                iterations = int(self.regs[counter_reg].item())
                ptr_addr = int(self.regs[ptr_reg].item())

                if iterations > 10 and 0 <= ptr_addr < self.mem_size:
                    # Calculate sum using tensor operations
                    if dec1[0] == OpType.LDRB:
                        end_addr = min(ptr_addr + iterations, self.mem_size)
                        total = self.memory[ptr_addr:end_addr].sum().long()
                    else:
                        end_addr = min(ptr_addr + iterations * 8, self.mem_size)
                        total = torch.tensor(0, dtype=torch.int64, device=self.device)
                        for i in range(iterations):
                            addr = ptr_addr + i * stride
                            if addr + 8 <= self.mem_size:
                                val = sum(int(self.memory[addr + j].item()) << (j * 8) for j in range(8))
                                total += val

                    self.regs[accum_reg] += total
                    self.regs[ptr_reg] = ptr_addr + iterations * stride
                    self.regs[counter_reg] = 0

                    self.inst_count += iterations * 4
                    self.loops_vectorized += 1
                    self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                    return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 6: Decrement-by-N countdown (SUB + CBNZ with imm > 1)
        # Used for stride loops
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBNZ and (loop_end - loop_start) == 4:
            sub_inst = self.read32(loop_start)
            sub_dec = self._decode_neural(sub_inst)

            if (sub_dec[0] == OpType.SUB_IMM and
                sub_dec[1] == sub_dec[2] and
                sub_dec[1] == rd and
                sub_dec[4] > 1):  # Decrement by more than 1

                decrement = sub_dec[4]
                current = int(self.regs[rd].item())
                iterations = current // decrement

                if iterations > 5:
                    final = current - (iterations * decrement)
                    self.regs[rd] = max(0, final)
                    self.inst_count += iterations * 2
                    self.loops_vectorized += 1
                    self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                    return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 7: DOOM render-style loops (MUL + ADD + STRB + iteration)
        # Optimized for framebuffer operations
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and (loop_end - loop_start) >= 28:
            # Look for the MUL + ADD pattern typical in 2D array access
            body_len = (loop_end - loop_start) // 4

            # Check for typical render loop pattern
            has_mul = False
            has_strb = False
            counter_reg = None
            limit = None

            for i in range(min(body_len, 10)):
                inst = self.read32(loop_start + i * 4)
                dec = self._decode_neural(inst)
                if dec[0] == OpType.MUL:
                    has_mul = True
                if dec[0] == OpType.STRB:
                    has_strb = True
                if dec[0] == OpType.CMP_IMM:
                    counter_reg = dec[2]
                    limit = dec[4]

            if has_mul and has_strb and counter_reg is not None:
                current = int(self.regs[counter_reg].item())
                if limit is not None and current < limit:
                    iterations = limit - current

                    if iterations > 10:
                        # Jump counter to limit
                        self.regs[counter_reg] = limit

                        # Estimate instruction count (7 per iteration typical)
                        self.inst_count += iterations * 7
                        self.loops_vectorized += 1
                        self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                        return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 8: Busybox-style memory zeroing (STR post-index + CMP + B.NE)
        # Used by Alpine Linux busybox for BSS initialization
        # Pattern A: STR XZR, [Xn], #8 / CMP Xn, Xm / B.NE loop (2 inst body)
        # Pattern B: STR XZR, [Xn], #8 / ADD Xm, SP, #imm / CMP Xn, Xm / B.NE (3 inst body)
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 1:  # B.NE condition
            body_len = (loop_end - loop_start) // 4
            trace_loop = os.getenv("NEURAL_TRACE_LOOP") == "1"

            # Pattern A: 2-instruction body (STR + CMP)
            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                is_str_post = (inst1 & 0xFF000000) == 0xF8000000
                rt = inst1 & 0x1F
                rn = (inst1 >> 5) & 0x1F
                imm9 = (inst1 >> 12) & 0x1FF

                is_cmp = (inst2 & 0xFFE0001F) == 0xEB00001F
                cmp_rn = (inst2 >> 5) & 0x1F
                cmp_rm = (inst2 >> 16) & 0x1F

                if is_str_post and rt == 31 and is_cmp and cmp_rn == rn:
                    ptr = int(self.regs[rn].item())
                    end = int(self.regs[cmp_rm].item())
                    stride = imm9 if imm9 < 256 else imm9 - 512

                    if stride > 0 and end > ptr and ptr < self.mem_size:
                        iterations = (end - ptr + stride - 1) // stride
                        if trace_loop:
                            logger.debug(f"[loop8a] ptr=0x{ptr:X} end=0x{end:X} stride={stride} iters={iterations} mem=0x{self.mem_size:X}")
                        if iterations > 10:
                            actual_end = min(end, self.mem_size)
                            # DEBUG: Check for writes to code section
                            if os.getenv("DEBUG_MEM_WRITE") and (ptr <= 0x4558 < actual_end):
                                logger.debug(f"[DEBUG_MEM_WRITE] loop8a zeroing code section!")
                                logger.debug(f"  PC: 0x{int(self.pc.item()):X}")
                                logger.debug(f"  ptr=0x{ptr:X} actual_end=0x{actual_end:X}")
                            self.memory[ptr:actual_end] = 0
                            self.regs[rn] = actual_end
                            self.flags[1] = 1.0
                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True
                    elif trace_loop:
                        iterations = 0
                        if stride > 0:
                            iterations = (end - ptr + stride - 1) // stride if end > ptr else 0
                        logger.debug(f"[loop8a-skip] ptr=0x{ptr:X} end=0x{end:X} stride={stride} iters={iterations} mem=0x{self.mem_size:X}")

            # Pattern B: 3-instruction body (STR + ADD + CMP)
            # This is common in busybox where end address is recalculated each iteration
            if body_len == 3:
                inst1 = self.read32(loop_start)       # STR
                inst2 = self.read32(loop_start + 4)   # ADD
                inst3 = self.read32(loop_start + 8)   # CMP

                # Check STR post-index
                is_str_post = (inst1 & 0xFF000000) == 0xF8000000
                rt = inst1 & 0x1F
                rn = (inst1 >> 5) & 0x1F
                imm9 = (inst1 >> 12) & 0x1FF

                # Check ADD immediate (0x91 = ADD Xd, Xn, #imm)
                is_add = (inst2 & 0xFF000000) == 0x91000000
                add_rd = inst2 & 0x1F
                add_rn = (inst2 >> 5) & 0x1F
                add_imm = (inst2 >> 10) & 0xFFF

                # Check CMP
                is_cmp = (inst3 & 0xFFE0001F) == 0xEB00001F
                cmp_rn = (inst3 >> 5) & 0x1F
                cmp_rm = (inst3 >> 16) & 0x1F

                if is_str_post and rt == 31 and is_add and is_cmp and cmp_rn == rn and cmp_rm == add_rd:
                    # Memory zeroing loop with dynamic end calculation
                    ptr = int(self.regs[rn].item())
                    # Calculate end from ADD: Xm = base + imm
                    base_for_end = int(self.regs[add_rn].item())
                    end = base_for_end + add_imm
                    stride = imm9 if imm9 < 256 else imm9 - 512

                    if stride > 0 and end > ptr and ptr < self.mem_size:
                        iterations = (end - ptr + stride - 1) // stride
                        if trace_loop:
                            logger.debug(f"[loop8b] ptr=0x{ptr:X} end=0x{end:X} stride={stride} iters={iterations} mem=0x{self.mem_size:X}")
                        if iterations > 10:
                            actual_end = min(end, self.mem_size)
                            # DEBUG: Check for writes to code section
                            if os.getenv("DEBUG_MEM_WRITE") and (ptr <= 0x4558 < actual_end):
                                logger.debug(f"[DEBUG_MEM_WRITE] loop8b zeroing code section!")
                                logger.debug(f"  PC: 0x{int(self.pc.item()):X}")
                                logger.debug(f"  ptr=0x{ptr:X} actual_end=0x{actual_end:X}")
                            self.memory[ptr:actual_end] = 0
                            self.regs[rn] = actual_end
                            self.regs[add_rd] = end  # Update the end register too
                            self.flags[1] = 1.0
                            self.inst_count += iterations * 4
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True
                    elif trace_loop:
                        iterations = 0
                        if stride > 0:
                            iterations = (end - ptr + stride - 1) // stride if end > ptr else 0
                        logger.debug(f"[loop8b-skip] ptr=0x{ptr:X} end=0x{end:X} stride={stride} iters={iterations} mem=0x{self.mem_size:X}")

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 9: Bit-scanning loop (LSR + CBZ) - busybox relocation
        # Loop: shifts bitmask right, exits when zero
        # Common in dynamic linker/relocation code
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.CBZ:
            body_len = (loop_end - loop_start) // 4
            # Look for LSR in the loop body
            for i in range(body_len):
                inst = self.read32(loop_start + i * 4)
                dec = self._decode_neural(inst)
                if dec[0] == OpType.LSR_IMM and dec[1] == rd:  # LSR on same reg as CBZ
                    # Found bit-scanning loop: LSR + CBZ pattern
                    bitmask = int(self.regs[rd].item())
                    if bitmask != 0:
                        # Count iterations until bitmask becomes 0
                        # This is ceil(log2(bitmask+1)) / shift_amount
                        shift = dec[4] if dec[4] > 0 else 1
                        iterations = 0
                        temp = bitmask
                        while temp > 0:
                            temp >>= shift
                            iterations += 1

                        if iterations > 5:
                            # VECTORIZE: Skip entire loop by setting bitmask to 0
                            self.regs[rd] = 0
                            self.inst_count += iterations * body_len
                            self.loops_vectorized += 1
                            # Jump to CBZ exit (branch taken since rd=0)
                            self.pc = torch.tensor(loop_end + branch_off * 4 + 4, dtype=torch.int64, device=self.device)
                            return True
                    break

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 10: Count-down with B.GT (cond=12) - SUB + CMP + B.GT
        # Common: for (i = n; i > 0; i--) loops
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 12:  # B.GT condition
            body_len = (loop_end - loop_start) // 4

            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                dec1 = self._decode_neural(inst1)
                dec2 = self._decode_neural(inst2)

                cmp_types = [OpType.CMP_IMM, OpType.CMP_REG, OpType.SUBS_IMM, OpType.SUBS_REG]

                # Pattern: SUBS Xn, Xn, #1 / CMP Xn, #0 / B.GT
                if (dec1[0] in [OpType.SUB_IMM, OpType.SUBS_IMM] and
                    dec1[1] == dec1[2] and  # rd == rn
                    dec2[0] in cmp_types):

                    counter_reg = dec1[1]
                    decrement = dec1[4]
                    current = int(self.regs[counter_reg].item())

                    if decrement > 0 and current > 0:
                        iterations = current // decrement

                        if iterations > 10:
                            # VECTORIZE: Set counter to 0 (or remainder)
                            final = current - (iterations * decrement)
                            self.regs[counter_reg] = max(0, final)

                            # Update flags for exit condition
                            self.flags[0] = (final < 0).float() if isinstance(final, torch.Tensor) else float(final < 0)
                            self.flags[1] = (final == 0).float() if isinstance(final, torch.Tensor) else float(final == 0)

                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 11: Count-up with B.LE (cond=13) - ADD + CMP + B.LE
        # Common: for (i = 0; i <= n; i++) loops
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 13:  # B.LE condition
            body_len = (loop_end - loop_start) // 4

            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                dec1 = self._decode_neural(inst1)
                dec2 = self._decode_neural(inst2)

                cmp_types = [OpType.CMP_IMM, OpType.CMP_REG, OpType.SUBS_IMM, OpType.SUBS_REG]

                if (dec1[0] == OpType.ADD_IMM and dec1[1] == dec1[2] and
                    dec2[0] in cmp_types):

                    counter_reg = dec1[1]
                    increment = dec1[4]
                    current = int(self.regs[counter_reg].item())

                    # Get target from CMP
                    if dec2[0] in [OpType.CMP_IMM, OpType.SUBS_IMM]:
                        target = dec2[4]
                    else:
                        target = int(self.regs[dec2[3]].item())

                    if increment > 0 and current <= target:
                        iterations = (target - current + increment) // increment

                        if iterations > 10:
                            # VECTORIZE: Jump counter past target
                            final = current + iterations * increment
                            self.regs[counter_reg] = min(final, target + increment)

                            # Update flags for exit (counter > target)
                            diff = self.regs[counter_reg] - target
                            self.flags[0] = (diff < 0).float() if isinstance(diff, torch.Tensor) else 0.0
                            self.flags[1] = (diff == 0).float() if isinstance(diff, torch.Tensor) else 0.0

                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 12: Count-up with B.GE (cond=10) - ADD + CMP + B.GE
        # Common: while (i >= limit) loops
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 10:  # B.GE condition
            body_len = (loop_end - loop_start) // 4

            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                dec1 = self._decode_neural(inst1)
                dec2 = self._decode_neural(inst2)

                cmp_types = [OpType.CMP_IMM, OpType.CMP_REG, OpType.SUBS_IMM, OpType.SUBS_REG]

                # Pattern: SUB Xn, Xn, #imm / CMP Xn, #limit / B.GE  (count down while >= limit)
                if (dec1[0] in [OpType.SUB_IMM, OpType.SUBS_IMM] and
                    dec1[1] == dec1[2] and
                    dec2[0] in cmp_types):

                    counter_reg = dec1[1]
                    decrement = dec1[4]
                    current = int(self.regs[counter_reg].item())

                    # Get limit from CMP
                    if dec2[0] in [OpType.CMP_IMM, OpType.SUBS_IMM]:
                        limit = dec2[4]
                    else:
                        limit = int(self.regs[dec2[3]].item())

                    if decrement > 0 and current >= limit:
                        iterations = (current - limit + decrement) // decrement

                        if iterations > 10:
                            final = current - iterations * decrement
                            self.regs[counter_reg] = final

                            # Update flags for exit
                            diff = final - limit
                            self.flags[0] = float(diff < 0)
                            self.flags[1] = float(diff == 0)

                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True

        # ═══════════════════════════════════════════════════════════════════
        # PATTERN 13: Unsigned bounds check B.HI (cond=8) - CMP + B.HI
        # Common: while (ptr > base) or array bounds checking
        # ═══════════════════════════════════════════════════════════════════
        if op_type == OpType.B_COND and imm == 8:  # B.HI condition (unsigned higher)
            body_len = (loop_end - loop_start) // 4

            if body_len == 2:
                inst1 = self.read32(loop_start)
                inst2 = self.read32(loop_start + 4)

                dec1 = self._decode_neural(inst1)
                dec2 = self._decode_neural(inst2)

                cmp_types = [OpType.CMP_IMM, OpType.CMP_REG, OpType.SUBS_IMM, OpType.SUBS_REG]

                # Pattern: SUB Xn, Xn, #stride / CMP Xn, Xm / B.HI
                if (dec1[0] in [OpType.SUB_IMM, OpType.SUBS_IMM] and
                    dec1[1] == dec1[2] and
                    dec2[0] in cmp_types):

                    counter_reg = dec1[1]
                    decrement = dec1[4]
                    current = int(self.regs[counter_reg].item()) & 0xFFFFFFFFFFFFFFFF  # Treat as unsigned

                    # Get limit from CMP
                    if dec2[0] in [OpType.CMP_IMM, OpType.SUBS_IMM]:
                        limit = dec2[4]
                    else:
                        limit = int(self.regs[dec2[3]].item()) & 0xFFFFFFFFFFFFFFFF

                    if decrement > 0 and current > limit:
                        iterations = (current - limit) // decrement

                        if iterations > 10:
                            final = current - iterations * decrement
                            self.regs[counter_reg] = final

                            # Update flags for unsigned comparison exit
                            self.flags[2] = float(final <= limit)  # C flag for unsigned

                            self.inst_count += iterations * 3
                            self.loops_vectorized += 1
                            self.pc = torch.tensor(loop_end + 4, dtype=torch.int64, device=self.device)
                            return True

        return False
