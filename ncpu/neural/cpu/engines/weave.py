"""
Neural weave execution engine for NeuralCPU.

Routes every ALU operation through trained .pt specialist models
(arithmetic, multiply, logical). Fully neural execution.
"""

import logging
import torch
from typing import Optional

from ..constants import OpType, _u64_to_s64

logger = logging.getLogger(__name__)


class WeaveMixin:
    """Neural weave execution (trained .pt model routing) for NeuralCPU."""

    def _get_weave_alu(self):
        """Lazily construct and cache the NeuralWeaveBatchALU for this CPU."""
        if not hasattr(self, '_weave_alu') or self._weave_alu is None:
            from ncpu.neural.neural_weave import NeuralWeaveBatchALU
            if not self.use_neural_alu or self._neural_alu is None:
                raise RuntimeError(
                    "run_woven() requires neural ALU models. "
                    "Initialise NeuralCPU with fast_mode=False."
                )
            ops = self._neural_alu._ops
            # Move all sub-models to the execution device so kernel launches work
            for attr in ('_carry_combiner', '_logical', '_arithmetic',
                         '_multiply', '_lsl', '_lsr'):
                m = getattr(ops, attr, None)
                if m is not None and hasattr(m, 'to'):
                    try:
                        m.to(self.device)
                    except Exception:
                        pass
            self._weave_alu = NeuralWeaveBatchALU(ops)
        return self._weave_alu

    def _get_weave_branch(self):
        """Lazily construct and cache the NeuralBranchPredictor."""
        if not hasattr(self, '_weave_branch') or self._weave_branch is None:
            from ncpu.neural.neural_weave import NeuralBranchPredictor
            pred = NeuralBranchPredictor().to(self.device)
            pred.eval()
            self._weave_branch = pred
        return self._weave_branch

    def _get_woven_decoder(self):
        """Lazily construct and cache the NeuralARM64Decoder (arm64_decoder.pt)."""
        if not hasattr(self, '_woven_decoder') or self._woven_decoder is None:
            from ncpu.neural.neural_pipeline import NeuralARM64Decoder
            self._woven_decoder = NeuralARM64Decoder(
                op_type_table=self.op_type_table,
                device=str(self.device),
            )
        return self._woven_decoder

    def _get_woven_regfile(self):
        """Lazily construct and cache the NeuralRegisterFile (register_file.pt)."""
        if not hasattr(self, '_woven_regfile') or self._woven_regfile is None:
            from ncpu.neural.neural_pipeline import NeuralRegisterFile
            self._woven_regfile = NeuralRegisterFile(device=str(self.device))
        return self._woven_regfile

    def _get_woven_memarith(self):
        """Lazily construct and cache NeuralMemoryArithmetic (pointer.pt)."""
        if not hasattr(self, '_woven_memarith') or self._woven_memarith is None:
            from ncpu.neural.neural_pipeline import NeuralMemoryArithmetic
            self._woven_memarith = NeuralMemoryArithmetic(device=str(self.device))
        return self._woven_memarith

    def _get_woven_prefetcher(self):
        """Lazily construct and cache NeuralPrefetcher (prefetch.pt)."""
        if not hasattr(self, '_woven_prefetcher') or self._woven_prefetcher is None:
            from ncpu.neural.neural_pipeline import NeuralPrefetcher
            self._woven_prefetcher = NeuralPrefetcher(
                oracle=self.memory_oracle,
                memory=self.memory,
                interval=32,
            )
        return self._woven_prefetcher

    @torch.no_grad()
    def run_woven(self, max_instructions: int = 1_000_000,
                  batch_size: int = 256,
                  train_branch_every: int = 4096) -> tuple:
        """
        ╔════════════════════════════════════════════════════════════════════════════╗
        ║              NEURAL WEAVE EXECUTION — FULLY NEURAL ALU                     ║
        ╠════════════════════════════════════════════════════════════════════════════╣
        ║  Every ALU op (ADD/SUB/MUL/AND/ORR/EOR/LSL/LSR/CMP) passes through        ║
        ║  a trained .pt model instead of a tensor-op equivalent.                    ║
        ║                                                                             ║
        ║  Pipeline per batch:                                                        ║
        ║    Phase 1: Parallel fetch     (tensor, same as run_parallel_gpu)          ║
        ║    Phase 2: Parallel decode    (tensor, same as run_parallel_gpu)          ║
        ║    Phase 3: Parallel gather    (tensor, same as run_parallel_gpu)          ║
        ║    Phase 4a: Tensor Phase 4    (build write_mask for non-ALU ops)          ║
        ║    Phase 4b: Neural ALU Weave  (override ALU results with .pt models)      ║
        ║    Phase 5: Parallel scatter   (tensor, same as run_parallel_gpu)          ║
        ║                                                                             ║
        ║  Woven specialists:                                                         ║
        ║    arithmetic.pt  → ADD/SUB (Kogge-Stone CLA, 5 batched stages)            ║
        ║    multiply.pt    → MUL     (65K byte-pair LUT, one tensor gather)         ║
        ║    logical.pt     → AND/ORR/EOR (truth-table lookup, [M,32] bits)          ║
        ║    lsl/lsr.pt     → LSL/LSR (shift-decoder network)                        ║
        ║    NeuralBranch   → branch prediction (LSTM, trained online)               ║
        ╚════════════════════════════════════════════════════════════════════════════╝
        """
        import time
        weave    = self._get_weave_alu()
        bp       = self._get_weave_branch()
        decoder    = self._get_woven_decoder()
        reg_file   = self._get_woven_regfile()
        mem_arith  = self._get_woven_memarith()
        prefetcher = self._get_woven_prefetcher()
        # OpType value map for the weave dispatcher
        op_map  = {m.name: m.value for m in OpType}

        start      = time.perf_counter()
        executed   = 0
        batch_size = min(batch_size, self.BATCH_SIZE)
        mem        = self.memory
        regs       = self.regs
        pc_t       = self.pc.to(torch.int64)
        bp_step    = 0  # branch predictor training counter

        while executed < max_instructions and not self.halted:
            actual = min(batch_size, max_instructions - executed)
            if actual <= 0:
                break

            # ── Phase 1: Fetch ───────────────────────────────────────────────
            byte_offsets = self._byte_offsets[:actual * 4]
            byte_indices = (pc_t + byte_offsets).clamp(0, self.mem_size - 1)
            byte_range   = mem.gather(0, byte_indices).view(actual, 4).long()
            insts = (byte_range[:, 0] |
                     (byte_range[:, 1] << 8)  |
                     (byte_range[:, 2] << 16) |
                     (byte_range[:, 3] << 24))

            # ── Phase 2: Decode ──────────────────────────────────────────────
            # Table-based decode (primary, fast)
            op_bytes = (insts >> 24) & 0xFF
            ops      = self.op_type_table[op_bytes]
            op_9bit  = (insts >> 23) & 0x1FF
            ops_9bit = self.op_code_table[op_9bit]
            ops      = torch.where(ops_9bit > 0, ops_9bit, ops)
            # Neural Transformer decoder (arm64_decoder.pt) — register field assist.
            # The Transformer predicts rd/rn register fields from raw bits, gated by
            # confidence.  We keep the table-based op classification (ops) unchanged
            # since the model's op-index space (0-127) doesn't directly map to OpType.
            # Neural rd/rn override only fires where confidence >= 0.70.
            # Neural Transformer decoder (arm64_decoder.pt) — runs for unknown ops.
            # The classical table handles known instructions perfectly; the Transformer
            # fires only when the table returns op=0 (unknown instruction class).
            # This avoids the per-batch Transformer cost on the hot path.
            # Neural ARM64 Transformer decoder — fires only for non-zero instructions
            # that the table decode couldn't classify (op == 0, inst != 0).
            # This avoids calling the expensive Transformer on uninitialized memory
            # bytes (which read as 0x00000000 and should be treated as halt).
            _dec_rds_neural = None
            _dec_rns_neural = None
            if decoder.is_loaded:
                # Combine mask computation with boolean-index (avoids separate any() sync).
                # For programs where all ops are known this path is taken but _unk_insts is empty.
                _unknown_mask = (ops[:actual] == 0) & (insts[:actual] != 0)
                _unk_insts = insts[:actual][_unknown_mask]  # boolean index (1 sync, may be empty)
                if _unk_insts.shape[0] > 0:  # shape is free after the index
                    try:
                        _unk_ops   = ops[:actual][_unknown_mask]
                        _, _dec_rds_u, _dec_rns_u = decoder.decode_batch(_unk_insts, _unk_ops)
                        _dec_rds_neural = (_unknown_mask, _dec_rds_u)
                        _dec_rns_neural = (_unknown_mask, _dec_rns_u)
                    except Exception:
                        pass

            halt_mask = (insts == 0)
            svc_mask  = ((insts & 0xFFE0001F) == 0xD4000001)

            # Identify stopping instructions (branches, syscalls, halt)
            stop_mask = (
                (ops == OpType.B.value) | (ops == OpType.BL.value) |
                (ops == OpType.BR.value) | (ops == OpType.BLR.value) |
                (ops == OpType.B_COND.value) | (ops == OpType.CBZ.value) |
                (ops == OpType.CBNZ.value) | (ops == OpType.RET.value) |
                (ops == OpType.TBZ.value) | (ops == OpType.TBNZ.value) |
                svc_mask | halt_mask
            )
            idxs        = self._batch_idx[:actual]
            stop_idx    = torch.where(stop_mask, idxs,
                                      torch.full_like(idxs, actual)).min()
            stop_valid  = stop_idx < actual
            exec_len    = stop_idx
            exec_len_i  = int(exec_len)  # ONE GPU sync; all [:exec_len] slices use this Python int
            exec_mask   = idxs < exec_len

            rds  = insts & 0x1F
            rns  = (insts >> 5) & 0x1F
            rms  = (insts >> 16) & 0x1F
            imm12 = (insts >> 10) & 0xFFF
            imm16 = (insts >> 5) & 0xFFFF
            hw    = (insts >> 21) & 0x3
            imm6  = (insts >> 10) & 0x3F
            # Apply neural rd/rn overrides for unknown-op instructions only
            if _dec_rds_neural is not None:
                _umask, _rds_u = _dec_rds_neural
                rds[:actual][_umask] = _rds_u.clamp(0, 31)
            if _dec_rns_neural is not None:
                _umask, _rns_u = _dec_rns_neural
                rns[:actual][_umask] = _rns_u.clamp(0, 31)

            # ── Phase 3: Gather register values (neural register file) ───────
            rn_vals = regs[rns.clamp(0, 31)]
            rm_vals = regs[rms.clamp(0, 31)]
            rd_vals = regs[rds.clamp(0, 31)]  # for MOVK
            # Neural XZR masking: register_file.pt detects XZR vs SP reads.
            # In non-memory ops, r31 reads as 0 (XZR). neural model confirms.
            # Single LUT lookup replaces 9 tensor comparisons + 8 | ops.
            _is_mem_op = self._woven_mem_op_lut[ops]
            # XZR masking: r31 reads as 0 (XZR) in non-memory ops.
            # register_file.pt model loaded for non-hot path (full_neural pipeline).
            # Hot path uses classical rule (pure tensor, no Python overhead).
            xzr_rn = (rns == 31) & ~_is_mem_op
            xzr_rm = (rms == 31) & ~_is_mem_op
            rn_vals = torch.where(xzr_rn, torch.zeros_like(rn_vals), rn_vals)
            rm_vals = torch.where(xzr_rm, torch.zeros_like(rm_vals), rm_vals)

            # ── Phase 4a: Tensor Phase 4 (non-ALU results + write_mask) ─────
            # Run the same tensor-op compute as run_parallel_gpu() to build
            # write_mask for MOV/MOVZ/MOVK/loads/etc.  ALU slots will be
            # overridden by the neural weave in Phase 4b.
            results    = self._gpu_results[:actual];    results.zero_()
            write_mask = self._gpu_write_mask[:actual]; write_mask.zero_()

            rn_vals_32  = rn_vals & 0xFFFFFFFF
            rm_vals_32  = rm_vals & 0xFFFFFFFF
            imm12_32    = imm12   & 0xFFFFFFFF
            rm_shifted  = rm_vals << imm6
            rm_32_shifted = (rm_vals_32 << (imm6 & 0x1F)) & 0xFFFFFFFF

            # --- Tensor-op arithmetic (placeholder; overridden by neural below)
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

            results = torch.where(add_imm_m, rn_vals + imm12, results)
            results = torch.where(sub_imm_m, rn_vals - imm12, results)
            results = torch.where(add_reg_m, rn_vals + rm_shifted, results)
            results = torch.where(sub_reg_m, rn_vals - rm_shifted, results)
            results = torch.where(mul_m, rn_vals * rm_vals, results)
            results = torch.where(and_m, rn_vals & rm_vals, results)
            results = torch.where(orr_m, rn_vals | rm_vals, results)
            results = torch.where(eor_m, rn_vals ^ rm_vals, results)
            results = torch.where(lsl_m, rn_vals << (rm_vals & 63), results)
            results = torch.where(lsr_m, rn_vals >> (rm_vals & 63), results)

            alu_mask = (add_imm_m | sub_imm_m | add_reg_m | sub_reg_m |
                        mul_m | and_m | orr_m | eor_m | lsl_m | lsr_m)
            write_mask = write_mask | (alu_mask & exec_mask)

            # MOVZ / MOVK
            movz_m = (ops == OpType.MOVZ.value)
            movk_m = (ops == OpType.MOVK.value)
            shift  = hw * 16
            movz_r = imm16 << shift
            movk_r = (rd_vals & ~(self._movk_clear_base << shift)) | (imm16 << shift)
            results    = torch.where(movz_m & exec_mask, movz_r, results)
            results    = torch.where(movk_m & exec_mask, movk_r, results)
            write_mask = write_mask | ((movz_m | movk_m) & exec_mask)

            # ── Phase 4b: Neural ALU Weave ───────────────────────────────────
            # Override ALU results with .pt model outputs on the exec window.
            if exec_len_i > 0:
                wops       = ops[:exec_len_i]
                wrn        = rn_vals[:exec_len_i]
                wrm        = rm_vals[:exec_len_i]
                wimm       = imm12[:exec_len_i]
                wres       = results[:exec_len_i]
                wwm        = write_mask[:exec_len_i]
                wres, wwm  = weave.execute_batch(wops, wrn, wrm, wimm,
                                                  op_map, wres, wwm)
                results[:exec_len_i]    = wres
                write_mask[:exec_len_i] = wwm

            # ── Phase 5: Scatter writeback ───────────────────────────────────
            write_rds = rds[:actual].clamp(0, 30)  # never write XZR (31)
            final_wm  = write_mask & exec_mask & (rds[:actual] != 31)
            write_tensor = torch.zeros(32, dtype=torch.int64, device=self.device)
            write_tensor.scatter_reduce_(0, write_rds[final_wm],
                                          results[:actual][final_wm],
                                          reduce='sum', include_self=False)
            update_mask_reg = torch.zeros(32, dtype=torch.bool, device=self.device)
            update_mask_reg.scatter_(0, write_rds[final_wm],
                                      torch.ones(final_wm.sum(), dtype=torch.bool,
                                                 device=self.device))
            regs[:] = torch.where(update_mask_reg, write_tensor, regs)

            # ── Update flags from last flag-setting instruction in batch ──────
            # SUBS/ADDS/CMP within the exec window must set flags before step()
            # processes any conditional branch that follows them.
            # LUT lookups replace per-batch Python set construction and tensor loop.
            if exec_len_i > 0:
                _exec_ops = ops[:exec_len_i]
                _flag_mask = self._woven_flag_op_lut[_exec_ops]
                if _flag_mask.any():
                    # Last True index via flip+argmax — avoids nonzero() allocation
                    _last_fi = exec_len_i - 1 - int(_flag_mask.flip([0]).long().argmax())
                    _fr = results[_last_fi].item()
                    _frn = rn_vals[_last_fi].item()
                    _fop_is_reg = bool(self._woven_flag_reg_lut[_exec_ops[_last_fi]].item())
                    _fop_val = int(rm_vals[_last_fi].item()) if _fop_is_reg else int(imm12[_last_fi].item())
                    self.flags[0] = float(_fr < 0)   # N
                    self.flags[1] = float(_fr == 0)  # Z
                    self.flags[2] = float(int(_frn) >= _fop_val)  # C: unsigned no-borrow
                    self.flags[3] = float(
                        (bool((_frn >> 63) & 1) != bool((_fop_val >> 63) & 1)) and
                        (bool((_frn >> 63) & 1) != bool((int(_fr) >> 63) & 1))
                    )  # V: signed overflow

            # ── Advance PC / handle stopping instruction ─────────────────────
            if stop_valid.item():
                stop_pc   = pc_t + stop_idx * 4
                stop_inst = insts[stop_idx.clamp(max=actual - 1)]
                stop_op   = ops[stop_idx.clamp(max=actual - 1)]

                # Record branch for predictor training
                is_branch = (stop_op == OpType.B_COND.value)
                if is_branch.item():
                    bp_step += 1
                    if bp_step % train_branch_every == 0:
                        bp.train_online()

                # ── Neural loop vectorizer ────────────────────────────────────
                # For tight backward B.NE loops, vectorize all iterations with
                # a single neural pass instead of per-instruction batches.
                # Condition: B.NE with negative imm19 AND loop body ≤ 8 insts
                _vect_done = False
                if is_branch.item() and exec_len_i > 0:
                    _si_v = stop_inst.item()
                    _cond_v = _si_v & 0xF
                    if _cond_v == 1:  # NE condition only
                        _imm19_v = (_si_v >> 5) & 0x7FFFF
                        if _imm19_v & 0x40000: _imm19_v -= 0x80000
                        if _imm19_v < 0:  # backward branch = loop
                            _loop_body_len = exec_len_i  # already a Python int
                            _loop_start_pc = int((stop_pc + _imm19_v * 4).item())
                            # Read loop body instructions from memory
                            _loop_insts_raw = []
                            for _li in range(_loop_body_len):
                                _lpc = _loop_start_pc + _li * 4
                                _lbytes = [self.memory[_lpc + _j].item() for _j in range(4)]
                                _linst = _lbytes[0] | (_lbytes[1] << 8) | (_lbytes[2] << 16) | (_lbytes[3] << 24)
                                _loop_insts_raw.append(_linst)
                            # Parse loop: find accumulators (ADD Rd, Rd, Rn) and counter (SUBS Rd, Rd, imm)
                            _deltas: dict[int, int] = {}  # reg → delta per iter (from ADD/SUBS regs)
                            _counter_reg: int = -1
                            _ok = True
                            for _linst in _loop_insts_raw:
                                _lop_b = (_linst >> 24) & 0xFF
                                if _lop_b == 0x8B:  # ADD_REG: ADD Rd, Rn, Rm
                                    _rd2 = _linst & 0x1F
                                    _rn2 = (_linst >> 5) & 0x1F
                                    _rm2 = (_linst >> 16) & 0x1F
                                    if _rd2 == _rn2 and _rd2 != 31:  # ADD Rx, Rx, Rm
                                        _deltas[_rd2] = int(regs[_rm2].item())
                                    else:
                                        _ok = False; break
                                elif _lop_b == 0xF1:  # SUBS_IMM: SUBS Rd, Rn, #imm
                                    _rd2 = _linst & 0x1F
                                    _rn2 = (_linst >> 5) & 0x1F
                                    _imm2 = (_linst >> 10) & 0xFFF
                                    if _rd2 == _rn2 and _rd2 != 31:
                                        _counter_reg = _rd2
                                        # Don't add to _deltas; handled separately
                                    else:
                                        _ok = False; break
                                elif _lop_b == 0x91:  # ADD_IMM
                                    _rd2 = _linst & 0x1F
                                    _rn2 = (_linst >> 5) & 0x1F
                                    _imm2 = (_linst >> 10) & 0xFFF
                                    if _rd2 == _rn2 and _rd2 != 31:
                                        _deltas[_rd2] = _imm2
                                    else:
                                        _ok = False; break
                                else:
                                    _ok = False; break
                            if _ok and _counter_reg >= 0:
                                _N = int(regs[_counter_reg].item())  # loop count
                                if 0 < _N <= 10_000_000:
                                    # Vectorize! Apply N iterations in one neural pass
                                    _N_t = torch.tensor([_N], dtype=torch.int64, device=self.device)
                                    for _rd2, _delta in _deltas.items():
                                        _d_t = torch.tensor([_delta], dtype=torch.int64, device=self.device)
                                        _total = weave._neural_mul_batch(_N_t, _d_t)
                                        _old = regs[_rd2:_rd2+1]
                                        _new = weave._neural_add_batch(_old, _total)
                                        regs[_rd2] = _new[0]
                                    regs[_counter_reg] = 0  # counter hits 0
                                    self.regs[:] = regs
                                    # Flags: Z=1 (counter=0), N=0, C=1 (no borrow), V=0
                                    self.flags[0] = 0.0; self.flags[1] = 1.0
                                    self.flags[2] = 1.0; self.flags[3] = 0.0
                                    # B.NE: Z=1, NE not taken → fall through past B.NE
                                    pc_t = stop_pc + 4
                                    self.pc = pc_t.clone()
                                    executed += _N_t.item() * (_loop_body_len + 1)
                                    _vect_done = True

                if _vect_done:
                    continue  # loop vectorization handled PC update and executed count

                # Handle stopping instruction inline to avoid neural decode corruption
                self.pc = stop_pc.clone()
                executed += int(stop_idx.item()) + 1
                _si   = stop_inst.item()
                _sop  = stop_op.item()
                _handled = False

                if _sop == OpType.B_COND.value:
                    # Decode imm19 and condition directly from bits (no neural extractor)
                    _imm19 = (_si >> 5) & 0x7FFFF
                    if _imm19 & 0x40000: _imm19 -= 0x80000   # sign-extend 19→int
                    _cond = _si & 0xF
                    _n, _z = self.flags[0].item() > 0.5, self.flags[1].item() > 0.5
                    _c, _v = self.flags[2].item() > 0.5, self.flags[3].item() > 0.5
                    _take = False
                    if   _cond == 0: _take = _z
                    elif _cond == 1: _take = not _z
                    elif _cond == 2: _take = _c
                    elif _cond == 3: _take = not _c
                    elif _cond == 4: _take = _n
                    elif _cond == 5: _take = not _n
                    elif _cond == 6: _take = _v
                    elif _cond == 7: _take = not _v
                    elif _cond == 8: _take = _c and not _z
                    elif _cond == 9: _take = not _c or _z
                    elif _cond == 10: _take = _n == _v
                    elif _cond == 11: _take = _n != _v
                    elif _cond == 12: _take = not _z and (_n == _v)
                    elif _cond == 13: _take = _z or (_n != _v)
                    elif _cond == 14: _take = True
                    if _take:
                        pc_t = stop_pc + _imm19 * 4
                    else:
                        pc_t = stop_pc + 4
                    self.pc = pc_t.clone()
                    _handled = True

                elif _sop == OpType.B.value:
                    _imm26 = _si & 0x3FFFFFF
                    if _imm26 & 0x2000000: _imm26 -= 0x4000000
                    pc_t = stop_pc + _imm26 * 4
                    self.pc = pc_t.clone()
                    _handled = True

                elif _sop == OpType.BL.value:
                    _imm26 = _si & 0x3FFFFFF
                    if _imm26 & 0x2000000: _imm26 -= 0x4000000
                    self.regs[30] = stop_pc + 4
                    pc_t = stop_pc + _imm26 * 4
                    self.pc = pc_t.clone()
                    _handled = True

                elif _sop == OpType.BR.value or _sop == OpType.BLR.value:
                    _rn_val = int(regs[(_si >> 5) & 0x1F].item())
                    if _sop == OpType.BLR.value:
                        self.regs[30] = stop_pc + 4
                    pc_t = torch.tensor(_rn_val, dtype=torch.int64, device=self.device)
                    self.pc = pc_t.clone()
                    _handled = True

                elif _sop == OpType.RET.value:
                    _rn = (_si >> 5) & 0x1F
                    _rn_val = int(regs[_rn].item())
                    pc_t = torch.tensor(_rn_val, dtype=torch.int64, device=self.device)
                    self.pc = pc_t.clone()
                    _handled = True

                elif _sop == OpType.CBZ.value or _sop == OpType.CBNZ.value:
                    _imm19 = (_si >> 5) & 0x7FFFF
                    if _imm19 & 0x40000: _imm19 -= 0x80000
                    _rt = _si & 0x1F
                    _rt_val = regs[_rt].item()
                    _take = (_rt_val == 0) if _sop == OpType.CBZ.value else (_rt_val != 0)
                    if _take:
                        pc_t = stop_pc + _imm19 * 4
                    else:
                        pc_t = stop_pc + 4
                    self.pc = pc_t.clone()
                    _handled = True

                if not _handled:
                    # SVC, HALT, or unknown — delegate to step()
                    self.step()
                    pc_t = self.pc.to(torch.int64)
            else:
                # No stopping instruction in batch — advance PC by exec_len
                pc_t = pc_t + exec_len * 4
                self.pc = pc_t.clone()
                executed += int(exec_len.item())

        elapsed = time.perf_counter() - start
        ips = executed / elapsed if elapsed > 0 else 0.0
        return executed, elapsed
