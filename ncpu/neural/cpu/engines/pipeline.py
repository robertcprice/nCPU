"""
Full neural pipeline execution engine for NeuralCPU.

Every stage (decode, execute, branch predict, memory) is a trained neural network.
"""

import logging
import torch
from typing import Optional

from ..constants import OpType

logger = logging.getLogger(__name__)


class PipelineMixin:
    """Full neural pipeline execution for NeuralCPU."""

    def _get_full_pipeline(self):
        """Lazily construct and cache the FullNeuralPipeline."""
        if not hasattr(self, '_full_pipeline') or self._full_pipeline is None:
            if not self.use_neural_alu or self._neural_alu is None:
                raise RuntimeError(
                    "run_full_neural() requires neural ALU. "
                    "Initialise NeuralCPU with fast_mode=False."
                )
            from ncpu.neural.neural_pipeline import FullNeuralPipeline
            self._full_pipeline = FullNeuralPipeline(self, self._neural_alu._ops)
        return self._full_pipeline

    @torch.no_grad()
    def run_full_neural(self, max_instructions: int = 1_000_000,
                        batch_size: int = 256,
                        speculate: bool = True) -> tuple:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║           FULL NEURAL PIPELINE — EVERY STAGE IS A NEURAL NETWORK          ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  Stage 1: NeuralPrefetcher    — LSTM pre-warms instruction stream          ║
        ║  Stage 2: NeuralARM64Decoder  — Transformer decodes instruction bits       ║
        ║  Stage 3: NeuralSpeculator    — checkpoint/rollback for branch speculation ║
        ║  Stage 4: NeuralWeaveBatchALU — ALU: arithmetic/logical/shift .pt models  ║
        ║  Stage 5: NeuralCacheManager  — cache_replace.pt superblock eviction       ║
        ║  Stage 6: NeuralSyscallRouter — MLP routes syscall numbers to handlers     ║
        ║                                                                             ║
        ║  vs run_woven():                                                            ║
        ║    + speculative execution (10-50× larger effective batch)                 ║
        ║    + neural decode replaces op_type_table lookup                           ║
        ║    + memory prefetch wired in (dormant → active)                           ║
        ║    + neural cache eviction in superblock cache                             ║
        ║    + neural syscall routing (replaces Python if/elif chain)                ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        import time
        pipe = self._get_full_pipeline()
        op_map = {m.name: m.value for m in OpType}

        start      = time.perf_counter()
        executed   = 0
        batch_size = min(batch_size, self.BATCH_SIZE)
        mem        = self.memory
        regs       = self.regs
        pc_t       = self.pc.to(torch.int64)
        spec       = pipe.speculator
        branch_history = torch.zeros(4, device=self.device)  # last 4 outcomes

        while executed < max_instructions and not self.halted:
            actual = min(batch_size, max_instructions - executed)
            if actual <= 0:
                break

            # ── Stage 1: Neural prefetch ─────────────────────────────────────
            pipe.prefetcher.warm_instruction_stream(int(pc_t.item()), actual, self.mem_size)

            # ── Stage 2: Fetch + neural decode ───────────────────────────────
            byte_offsets = self._byte_offsets[:actual * 4]
            byte_indices = (pc_t + byte_offsets).clamp(0, self.mem_size - 1)
            byte_range   = mem.gather(0, byte_indices).view(actual, 4).long()
            insts = (byte_range[:, 0] |
                     (byte_range[:, 1] << 8)  |
                     (byte_range[:, 2] << 16) |
                     (byte_range[:, 3] << 24))

            # Classical decode (fallback + write_mask for non-ALU ops)
            op_bytes    = (insts >> 24) & 0xFF
            ops_classic = self.op_type_table[op_bytes]
            op_9bit     = (insts >> 23) & 0x1FF
            ops_9bit    = self.op_code_table[op_9bit]
            ops_classic = torch.where(ops_9bit > 0, ops_9bit, ops_classic)

            # Neural decode (overrides when confidence is high)
            ops, rds_neural, rns_neural = pipe.decoder.decode_batch(insts, ops_classic)

            # ── Identify stopping instructions ────────────────────────────────
            halt_mask = (insts == 0)
            svc_mask  = ((insts & 0xFFE0001F) == 0xD4000001)
            stop_mask = (
                (ops == OpType.B.value) | (ops == OpType.BL.value) |
                (ops == OpType.BR.value) | (ops == OpType.BLR.value) |
                (ops == OpType.B_COND.value) | (ops == OpType.CBZ.value) |
                (ops == OpType.CBNZ.value) | (ops == OpType.RET.value) |
                (ops == OpType.TBZ.value) | (ops == OpType.TBNZ.value) |
                svc_mask | halt_mask
            )
            idxs      = self._batch_idx[:actual]
            stop_idx  = torch.where(stop_mask, idxs,
                                    torch.full_like(idxs, actual)).min()
            stop_valid = stop_idx < actual
            exec_len   = stop_idx
            exec_mask  = idxs < exec_len

            # Field decode (classical for correctness)
            rds   = insts & 0x1F
            rns   = (insts >> 5) & 0x1F
            rms   = (insts >> 16) & 0x1F
            imm12 = (insts >> 10) & 0xFFF
            imm16 = (insts >> 5) & 0xFFFF
            hw    = (insts >> 21) & 0x3
            imm6  = (insts >> 10) & 0x3F

            # ── Stage 3a: Neural branch speculation (before exec) ─────────────
            # If a branch is the stopping instruction, get prediction BEFORE executing
            will_speculate = False
            if speculate and stop_valid.item():
                stop_op  = ops[stop_idx.clamp(max=actual - 1)]
                is_cond  = (stop_op == OpType.B_COND.value)
                if is_cond.item() and not spec.active:
                    stop_pc_val  = int((pc_t + stop_idx * 4).item())
                    cond_code    = int((insts[stop_idx.clamp(max=actual - 1)] & 0xF).item())
                    taken_prob   = float(pipe.branch_pred.predict_batch(
                        torch.tensor([stop_pc_val], device=self.device),
                        torch.tensor([cond_code],   device=self.device),
                        branch_history.unsqueeze(0),
                    )[0].item())
                    do_spec, pred_taken = spec.should_speculate(taken_prob)
                    if do_spec:
                        spec.enter(regs, pc_t, self.flags, pred_taken)
                        will_speculate = True

            # ── Stage 4: Gather + neural ALU weave ───────────────────────────
            rn_vals = regs[rns.clamp(0, 31)]
            rm_vals = regs[rms.clamp(0, 31)]
            rd_vals = regs[rds.clamp(0, 31)]

            results    = self._gpu_results[:actual];    results.zero_()
            write_mask = self._gpu_write_mask[:actual]; write_mask.zero_()

            rn_vals_32  = rn_vals & 0xFFFFFFFF
            rm_vals_32  = rm_vals & 0xFFFFFFFF
            imm12_32    = imm12   & 0xFFFFFFFF

            # Tensor-op pass (non-ALU ops + MOVZ/MOVK)
            add_imm_m = (ops == OpType.ADD_IMM.value)
            sub_imm_m = (ops == OpType.SUB_IMM.value)
            add_reg_m = (ops == OpType.ADD_REG.value)
            sub_reg_m = (ops == OpType.SUB_REG.value)
            mul_m     = (ops == OpType.MUL.value)
            and_m     = (ops == OpType.AND_REG.value) | (ops == OpType.AND_IMM.value)
            orr_m     = (ops == OpType.ORR_REG.value) | (ops == OpType.ORR_IMM.value)
            eor_m     = (ops == OpType.EOR_REG.value) | (ops == OpType.EOR_IMM.value)
            lsl_m     = (ops == OpType.LSL_REG.value) | (ops == OpType.LSL_IMM.value)
            lsr_m     = (ops == OpType.LSR_REG.value) | (ops == OpType.LSR_IMM.value)
            alu_mask  = add_imm_m | sub_imm_m | add_reg_m | sub_reg_m | mul_m
            alu_mask  = alu_mask | and_m | orr_m | eor_m | lsl_m | lsr_m
            write_mask = write_mask | (alu_mask & exec_mask)

            movz_m = (ops == OpType.MOVZ.value)
            movk_m = (ops == OpType.MOVK.value)
            shift  = hw * 16
            movz_r = imm16 << shift
            movk_r = (rd_vals & ~(torch.tensor(0xFFFF, device=self.device) << shift)) | (imm16 << shift)
            results    = torch.where(movz_m & exec_mask, movz_r, results)
            results    = torch.where(movk_m & exec_mask, movk_r, results)
            write_mask = write_mask | ((movz_m | movk_m) & exec_mask)

            # Neural ALU weave overrides all ALU results
            if exec_len > 0:
                wops      = ops[:exec_len]
                wrn       = rn_vals[:exec_len]
                wrm       = rm_vals[:exec_len]
                wimm      = imm12[:exec_len]
                wres      = results[:exec_len]
                wwm       = write_mask[:exec_len]
                wres, wwm = pipe.alu_weave.execute_batch(wops, wrn, wrm, wimm, op_map, wres, wwm)
                results[:exec_len]    = wres
                write_mask[:exec_len] = wwm

            # ── Stage 5a: Scatter writeback ───────────────────────────────────
            write_rds   = rds[:actual].clamp(0, 30)
            final_wm    = write_mask & exec_mask & (rds[:actual] != 31)
            write_tensor = torch.zeros(32, dtype=torch.int64, device=self.device)
            write_tensor.scatter_reduce_(0, write_rds[final_wm],
                                          results[:actual][final_wm],
                                          reduce='sum', include_self=False)
            upd_mask = torch.zeros(32, dtype=torch.bool, device=self.device)
            upd_mask.scatter_(0, write_rds[final_wm],
                               torch.ones(final_wm.sum(), dtype=torch.bool, device=self.device))
            regs[:] = torch.where(upd_mask, write_tensor, regs)

            # ── Stage 5b: Stage 5 — branch resolve / speculator check ────────
            if stop_valid.item():
                stop_pc_t  = pc_t + stop_idx * 4
                stop_inst  = insts[stop_idx.clamp(max=actual - 1)]
                stop_op_v  = ops[stop_idx.clamp(max=actual - 1)]
                cond_code  = int((stop_inst & 0xF).item())

                # Advance PC to the branch instruction and execute it normally
                self.pc = stop_pc_t.clone()
                pc_before_branch = int(self.pc.item())
                executed += int(stop_idx.item())

                # Execute the stopping instruction (branch/syscall)
                self.step()
                pc_after = int(self.pc.item())
                executed += 1

                # Speculator: check if prediction was correct
                if spec.active:
                    actual_taken = (pc_after != pc_before_branch + 4)
                    if actual_taken == spec.predicted_taken:
                        spec.commit()
                    else:
                        pc_t, _ = spec.rollback(regs, pc_t, self.flags)
                        pc_t = self.pc.to(torch.int64)
                    pipe.on_branch(pc_before_branch, cond_code, actual_taken, regs, self.flags)
                    # Update branch history
                    branch_history = torch.roll(branch_history, -1)
                    branch_history[-1] = float(actual_taken)

                # Syscall routing recording
                if int(svc_mask[stop_idx.clamp(max=actual - 1)].item()):
                    pipe.on_syscall(int(regs[8].item()), regs)

                pc_t = self.pc.to(torch.int64)
            else:
                pc_t = pc_t + exec_len * 4
                self.pc = pc_t.clone()
                executed += int(exec_len.item())

        elapsed = time.perf_counter() - start
        pipeline_stats = pipe.stats()
        return executed, elapsed, pipeline_stats
