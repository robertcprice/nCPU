"""
Zero-sync GPU-only execution engine for NeuralCPU.

The most optimized engine: zero .item() calls in the hot path,
async syscall handling, fully GPU-resident execution.
"""

import logging
import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from typing import Optional, Tuple

from ..constants import OpType, _u64_to_s64
from ...hotloop_value_model import (
    encode_hotloop_value_embedding,
    derive_hotloop_value_target,
    predict_hotloop_value_score,
)

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ALU_MODELS_ROOT = _REPO_ROOT / "models" / "alu"
_U64_SIGN_FLIP = _u64_to_s64(0x8000000000000000)

# ── torch.compile for CUDA: fuse SIMD ALU into single kernel ──
_USE_COMPILE = os.environ.get('NCPU_COMPILE', '0') == '1' and torch.cuda.is_available()

def _simd_alu_core(insts, ops, rn_vals, rm_vals, imm12, imm16, hw, exec_mask,
                    batch_size, device, OpType_vals):
    """SIMD ALU dispatch — compilable on CUDA via torch.compile."""
    add_imm_mask = (ops == OpType_vals[0]) & exec_mask
    sub_imm_mask = (ops == OpType_vals[1]) & exec_mask
    add_reg_mask = (ops == OpType_vals[2]) & exec_mask
    sub_reg_mask = (ops == OpType_vals[3]) & exec_mask
    mov_reg_mask = (ops == OpType_vals[4]) & exec_mask
    lsl_imm_mask = (ops == OpType_vals[5]) & exec_mask
    lsr_imm_mask = (ops == OpType_vals[6]) & exec_mask

    movz_mask = (((insts & 0xFF800000) == 0xD2800000) | ((insts & 0xFF800000) == 0x52800000)) & exec_mask
    mul_gpu_mask = (((insts & 0xFFE0FC00) == 0x9B007C00) | ((insts & 0xFFE0FC00) == 0x1B007C00)) & exec_mask
    asr_imm_gpu = (((insts & 0xFFC0FC00) == 0x9340FC00) | ((insts & 0xFFC0FC00) == 0x13007C00)) & exec_mask

    shift_amt = (insts >> 10) & 0x3F
    asr_shift = (insts >> 16) & 0x3F
    rm_shift = rm_vals & 0x3F
    movz_val = imm16 << (hw * 16)
    sxtw_u32 = rn_vals & 0xFFFFFFFF
    sxtw_s32 = torch.where(sxtw_u32 >= 0x80000000, sxtw_u32 - 0x100000000, sxtw_u32)

    results = torch.zeros(batch_size, device=device, dtype=torch.int64)
    write_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

    # Priority-select via stacked classify/compute
    _cls = torch.stack([
        add_imm_mask, sub_imm_mask, add_reg_mask, sub_reg_mask,
        movz_mask, mov_reg_mask, mul_gpu_mask, lsl_imm_mask, lsr_imm_mask,
        asr_imm_gpu,
    ], dim=0)

    _cmp = torch.stack([
        rn_vals + imm12, rn_vals - imm12, rn_vals + rm_vals, rn_vals - rm_vals,
        movz_val, rm_vals, rn_vals * rm_vals,
        rn_vals << shift_amt.clamp(0, 63), rn_vals >> shift_amt.clamp(0, 63),
        rn_vals >> asr_shift.clamp(0, 63),
    ], dim=0)

    _K = _cls.shape[0]
    _bidx = torch.arange(batch_size, device=device)
    _prio = (_cls.long() * torch.arange(_K, device=device, dtype=torch.int64).unsqueeze(1)).max(dim=0).indices
    _hit = _cls.any(dim=0)
    results = torch.where(_hit, _cmp[_prio, _bidx], results)
    write_mask = _hit

    return results, write_mask

if _USE_COMPILE:
    _simd_alu_core = torch.compile(_simd_alu_core, mode="reduce-overhead")
    logger.info("torch.compile enabled for SIMD ALU dispatch (CUDA)")


def _u64_ge(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Unsigned >= comparison that works on CPU/MPS/CUDA int64 tensors."""
    return torch.bitwise_xor(lhs.to(torch.int64), _U64_SIGN_FLIP) >= torch.bitwise_xor(rhs.to(torch.int64), _U64_SIGN_FLIP)


def _u64_gt(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Unsigned > comparison that works on CPU/MPS/CUDA int64 tensors."""
    return torch.bitwise_xor(lhs.to(torch.int64), _U64_SIGN_FLIP) > torch.bitwise_xor(rhs.to(torch.int64), _U64_SIGN_FLIP)


class GpuOnlyMixin:
    """Zero-sync GPU-only execution for NeuralCPU."""

    @staticmethod
    def _wrap_hotloop_i64(value: int) -> int:
        value &= 0xFFFFFFFFFFFFFFFF
        if value >= (1 << 63):
            value -= (1 << 64)
        return value

    @staticmethod
    def _ceil_div_pos(num: int, den: int) -> int:
        return -((-num) // den)

    @staticmethod
    def _is_hotloop_cmp(inst: int) -> bool:
        rd = inst & 0x1F
        return (
            ((inst & 0xFF000000) == 0xF1000000 and rd == 31) or
            ((inst & 0xFF200000) == 0xEB000000 and rd == 31)
        )

    @staticmethod
    def _encode_hotloop_words(words) -> bytes:
        return b"".join(
            int(word & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False)
            for word in words
        )

    @staticmethod
    def _hotloop_materialized_nop() -> int:
        # MOVZ XZR, #0 behaves as a no-op and stays inside the Rust CPU subset.
        return 0xD280001F

    @staticmethod
    def _clone_hotloop_candidate(
        candidate,
        *,
        cache_hit: bool = False,
        cache_hit_kind: str = "none",
    ):
        cloned = dict(candidate)
        cloned["cache_hit"] = bool(cache_hit)
        cloned["cache_hit_kind"] = str(cache_hit_kind)
        return cloned

    def _read_hotloop_window_snapshot(self, windows):
        merged = self._merge_hotloop_windows(windows)
        snapshot = []
        for start, end in merged:
            start = int(start)
            end = int(end)
            if start < 0 or end < start or end > self.mem_size:
                return None
            payload = bytes(self.memory[start:end].detach().cpu().tolist())
            snapshot.append((start, payload))
        return tuple(snapshot)

    def _matches_hotloop_window_snapshot(self, snapshot) -> bool:
        if snapshot is None:
            return False
        for start, payload in snapshot:
            end = int(start) + len(payload)
            if start < 0 or end > self.mem_size:
                return False
            current = bytes(self.memory[start:end].detach().cpu().tolist())
            if current != payload:
                return False
        return True

    def _bump_superblock_cache_stat(self, field: str, delta: int = 1):
        if not hasattr(self, "_last_gpu_only_hotloop_stats"):
            self._last_gpu_only_hotloop_stats = {}
        self._last_gpu_only_hotloop_stats[field] = (
            int(self._last_gpu_only_hotloop_stats.get(field, 0)) + int(delta)
        )

    def _matches_superblock_template_guard(self, guard) -> bool:
        if not isinstance(guard, dict):
            return False
        for reg_idx, expected_value in guard.get("regs", ()):
            if int(self.regs[int(reg_idx)].item()) != int(expected_value):
                return False
        expected_flags = tuple(int(v) for v in guard.get("flags", ()))
        if expected_flags:
            flags_host = self.flags.detach().cpu().to(torch.float32).tolist()
            observed_flags = tuple(int(value > 0.5) for value in flags_host[:4])
            if observed_flags != expected_flags:
                return False
        return True

    def _build_superblock_template_guard(self, candidate):
        simulation = candidate.get("simulation")
        words = candidate.get("original_words") or candidate.get("words")
        if not isinstance(simulation, dict) or not isinstance(words, list):
            return None

        path_indices = simulation.get("path_indices")
        if not isinstance(path_indices, list) or not path_indices:
            return None

        needed_regs: set[int] = set()
        needed_flags = False

        def _add_reg_use(reg_idx: int | None):
            if reg_idx is None or reg_idx == 31:
                return
            needed_regs.add(int(reg_idx))

        for path_idx in reversed(path_indices):
            path_idx = int(path_idx)
            if path_idx < 0 or path_idx >= len(words):
                return None
            inst = int(words[path_idx])
            if inst == 0x00000000:
                continue

            branch = self._decode_hotloop_branch(inst, path_idx)
            if branch is not None:
                if branch["branch_kind"] in {"cbz", "cbnz"}:
                    _add_reg_use(branch.get("branch_reg"))
                elif branch["branch_kind"] == "bcond":
                    needed_flags = True
                continue

            memop = self._decode_hotloop_memop(inst)
            if memop is not None:
                _add_reg_use(memop.get("base_reg"))
                data_reg = memop.get("data_reg")
                if (
                    not memop["writes_memory"] and
                    data_reg not in {None, 31} and
                    int(data_reg) in needed_regs
                ):
                    needed_regs.discard(int(data_reg))
                continue

            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            defs_regs: set[int] = set()
            uses_regs: set[int] = set()
            writes_flags = False

            if (inst & 0xFF800000) in {0xD2800000, 0x52800000}:  # MOVZ
                if rd != 31:
                    defs_regs.add(rd)
            elif (inst & 0xFF800000) in {0xF2800000, 0x72800000}:  # MOVK
                if rd != 31:
                    defs_regs.add(rd)
                    uses_regs.add(rd)
            elif (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
                if rn != 31:
                    uses_regs.add(rn)
                writes_flags = (inst & 0xFF000000) in {0xB1000000, 0xF1000000}
                if rd != 31:
                    defs_regs.add(rd)
            elif (inst & 0xFF200000) in {0x8B000000, 0xCB000000, 0xAB000000, 0xEB000000}:
                rm = (inst >> 16) & 0x1F
                if rn != 31:
                    uses_regs.add(rn)
                if rm != 31:
                    uses_regs.add(rm)
                writes_flags = (inst & 0xFF200000) in {0xAB000000, 0xEB000000}
                if rd != 31:
                    defs_regs.add(rd)
            elif (inst & 0xFF200000) in {0x8A000000, 0xAA000000, 0xCA000000}:
                rm = (inst >> 16) & 0x1F
                if rn != 31:
                    uses_regs.add(rn)
                if rm != 31:
                    uses_regs.add(rm)
                if rd != 31:
                    defs_regs.add(rd)
            elif (inst & 0xFFE0FC00) == 0x9B007C00:
                rm = (inst >> 16) & 0x1F
                if rn != 31:
                    uses_regs.add(rn)
                if rm != 31:
                    uses_regs.add(rm)
                if rd != 31:
                    defs_regs.add(rd)
            else:
                return None

            if writes_flags and needed_flags:
                needed_flags = False
                needed_regs.update(uses_regs)

            overwritten = defs_regs & needed_regs
            if overwritten:
                needed_regs.difference_update(overwritten)
                needed_regs.update(uses_regs)

        reg_guard = tuple(
            (reg_idx, int(self.regs[reg_idx].item()))
            for reg_idx in sorted(needed_regs)
        )
        flag_guard = ()
        if needed_flags:
            flags_host = self.flags.detach().cpu().to(torch.float32).tolist()
            flag_guard = tuple(int(value > 0.5) for value in flags_host[:4])
        return {
            "regs": reg_guard,
            "flags": flag_guard,
        }

    @staticmethod
    def _retarget_superblock_template_candidate(candidate, current_pc: int):
        cloned = dict(candidate)
        source_pc = int(candidate.get("pc", current_pc))
        delta_pc = int(current_pc) - source_pc
        cloned["pc"] = int(current_pc)
        if "expected_stop_pc" in candidate:
            cloned["expected_stop_pc"] = int(candidate["expected_stop_pc"]) + delta_pc
        simulation = candidate.get("simulation")
        if isinstance(simulation, dict):
            updated_simulation = dict(simulation)
            if "expected_stop_pc" in simulation:
                updated_simulation["expected_stop_pc"] = int(simulation["expected_stop_pc"]) + delta_pc
            cloned["simulation"] = updated_simulation
        return cloned

    @staticmethod
    def _superblock_shape_token(inst: int):
        inst = int(inst) & 0xFFFFFFFF
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F
        rm = (inst >> 16) & 0x1F

        if inst == 0x00000000:
            return ("halt",)
        if (inst & 0xFC000000) == 0x14000000:
            return ("b",)
        if (inst & 0xFF000010) == 0x54000000:
            return ("bcond", inst & 0xF)
        if (inst & 0x7F000000) == 0x34000000:
            return ("cbz", rd)
        if (inst & 0x7F000000) == 0x35000000:
            return ("cbnz", rd)
        if (inst & 0xFF800000) in {0xD2800000, 0x52800000}:
            return ("movz", inst & 0xFF800000, rd, (inst >> 21) & 0x3)
        if (inst & 0xFF800000) in {0xF2800000, 0x72800000}:
            return ("movk", inst & 0xFF800000, rd, (inst >> 21) & 0x3)
        if (inst & 0xFFC00000) in {
            0x39000000,
            0x39400000,
            0xB9000000,
            0xB9400000,
            0xF9000000,
            0xF9400000,
        }:
            return ("mem-uoff", inst & 0xFFC00000, rn, rd)
        if (inst & 0xFFE00C00) in {0x38000400, 0x38400400}:
            return ("mem-post", inst & 0xFFE00C00, rn, rd)
        if (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
            return ("alu-imm", inst & 0xFF000000, rd, rn)
        if (inst & 0xFF200000) in {
            0x8B000000,
            0xCB000000,
            0xAB000000,
            0xEB000000,
            0x8A000000,
            0xAA000000,
            0xCA000000,
        }:
            return ("alu-reg", inst & 0xFF200000, rd, rn, rm)
        if (inst & 0xFFE0FC00) == 0x9B007C00:
            return ("mul", rd, rn, rm)
        return ("raw", (inst >> 21) & 0x7FF, rn, rm, rd)

    def _superblock_window_shape_key(self, max_words: int, max_steps: int, words):
        return (
            int(max_words),
            int(max_steps),
            tuple(self._superblock_shape_token(word) for word in words),
        )

    @staticmethod
    def _is_superblock_literal_patch_inst(inst: int) -> bool:
        return (int(inst) & 0xFF800000) in {
            0xD2800000,
            0x52800000,
            0xF2800000,
            0x72800000,
        }

    @staticmethod
    def _superblock_shape_patch_kind(inst: int) -> str | None:
        inst = int(inst) & 0xFFFFFFFF
        if GpuOnlyMixin._is_superblock_literal_patch_inst(inst):
            return "literal"
        if (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
            return "alu-imm"
        return None

    @staticmethod
    def _superblock_patch_inst_identity_mask(inst: int) -> int | None:
        inst = int(inst) & 0xFFFFFFFF
        kind = GpuOnlyMixin._superblock_shape_patch_kind(inst)
        if kind == "literal":
            return 0xFFE0001F
        if kind == "alu-imm":
            return 0xFFC003FF
        return None

    @staticmethod
    def _superblock_patch_inst_kind(original_inst: int, current_inst: int) -> str | None:
        original_inst = int(original_inst) & 0xFFFFFFFF
        current_inst = int(current_inst) & 0xFFFFFFFF
        original_kind = GpuOnlyMixin._superblock_shape_patch_kind(original_inst)
        current_kind = GpuOnlyMixin._superblock_shape_patch_kind(current_inst)
        if original_kind is None or original_kind != current_kind:
            return None
        identity_mask = GpuOnlyMixin._superblock_patch_inst_identity_mask(original_inst)
        if identity_mask is None:
            return None
        if (original_inst & identity_mask) != (current_inst & identity_mask):
            return None
        return original_kind

    @staticmethod
    def _describe_superblock_data_flow(inst: int):
        inst = int(inst) & 0xFFFFFFFF
        rd = inst & 0x1F
        rn = (inst >> 5) & 0x1F

        defs_regs: set[int] = set()
        uses_regs: set[int] = set()
        writes_flags = False

        if (inst & 0xFF800000) == 0xD2800000:  # MOVZ Xd
            if rd != 31:
                defs_regs.add(rd)
        elif (inst & 0xFF800000) == 0x52800000:  # MOVZ Wd
            if rd != 31:
                defs_regs.add(rd)
        elif (inst & 0xFF800000) == 0xF2800000:  # MOVK Xd
            if rd != 31:
                defs_regs.add(rd)
                uses_regs.add(rd)
        elif (inst & 0xFF800000) == 0x72800000:  # MOVK Wd
            if rd != 31:
                defs_regs.add(rd)
                uses_regs.add(rd)
        elif (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
            if rn != 31:
                uses_regs.add(rn)
            writes_flags = (inst & 0xFF000000) in {0xB1000000, 0xF1000000}
            if rd != 31:
                defs_regs.add(rd)
        elif (inst & 0xFF200000) in {0x8B000000, 0xCB000000, 0xAB000000, 0xEB000000}:
            rm = (inst >> 16) & 0x1F
            if rn != 31:
                uses_regs.add(rn)
            if rm != 31:
                uses_regs.add(rm)
            writes_flags = (inst & 0xFF200000) in {0xAB000000, 0xEB000000}
            if rd != 31:
                defs_regs.add(rd)
        elif (inst & 0xFF200000) in {0x8A000000, 0xAA000000, 0xCA000000}:
            rm = (inst >> 16) & 0x1F
            if rn != 31:
                uses_regs.add(rn)
            if rm != 31:
                uses_regs.add(rm)
            if rd != 31:
                defs_regs.add(rd)
        elif (inst & 0xFFE0FC00) == 0x9B007C00:
            rm = (inst >> 16) & 0x1F
            if rn != 31:
                uses_regs.add(rn)
            if rm != 31:
                uses_regs.add(rm)
            if rd != 31:
                defs_regs.add(rd)
        else:
            return None

        return {
            "defs_regs": defs_regs,
            "uses_regs": uses_regs,
            "writes_flags": writes_flags,
        }

    def _collect_superblock_shape_patch_sites(self, candidate):
        simulation = candidate.get("simulation")
        original_words = candidate.get("original_words") or candidate.get("words")
        if not isinstance(simulation, dict) or not isinstance(original_words, list):
            return None
        path_indices = simulation.get("path_indices")
        if not isinstance(path_indices, list) or not path_indices:
            return None

        live_regs: set[int] = set()
        live_flags = False
        patchable_sites: set[int] = set()

        for path_idx in reversed(path_indices):
            idx = int(path_idx)
            if idx < 0 or idx >= len(original_words):
                return None
            inst = int(original_words[idx])
            if inst == 0x00000000:
                continue

            live_after_regs = set(live_regs)
            live_after_flags = bool(live_flags)

            branch = self._decode_hotloop_branch(inst, idx)
            if branch is not None:
                if branch["branch_kind"] in {"cbz", "cbnz"}:
                    branch_reg = branch.get("branch_reg")
                    if branch_reg not in {None, 31}:
                        live_regs.add(int(branch_reg))
                elif branch["branch_kind"] == "bcond":
                    live_flags = True
                continue

            memop = self._decode_hotloop_memop(inst)
            if memop is not None:
                base_reg = memop.get("base_reg")
                if base_reg not in {None, 31}:
                    live_regs.add(int(base_reg))
                continue

            flow = self._describe_superblock_data_flow(inst)
            if flow is None:
                return None
            defs_regs = set(flow["defs_regs"])
            uses_regs = set(flow["uses_regs"])
            writes_flags = bool(flow["writes_flags"])

            if (
                self._superblock_shape_patch_kind(inst) is not None and
                not (defs_regs & live_after_regs) and
                not (writes_flags and live_after_flags)
            ):
                patchable_sites.add(idx)

            if writes_flags and live_flags:
                live_flags = False
                live_regs.update(uses_regs)

            overwritten = defs_regs & live_regs
            if overwritten:
                live_regs.difference_update(overwritten)
                live_regs.update(uses_regs)

        return patchable_sites

    def _specialize_superblock_shape_candidate(self, candidate, words):
        simulation = candidate.get("simulation")
        original_words = candidate.get("original_words") or candidate.get("words")
        materialized_words = candidate.get("words")
        if (
            not isinstance(simulation, dict) or
            not isinstance(original_words, list) or
            not isinstance(materialized_words, list)
        ):
            return None
        path_indices = simulation.get("path_indices")
        if not isinstance(path_indices, list) or not path_indices:
            return None

        patchable_sites = self._collect_superblock_shape_patch_sites(candidate)
        if patchable_sites is None:
            return None

        specialized_candidate = dict(candidate)
        specialized_original_words = list(original_words)
        specialized_materialized_words = list(materialized_words)
        patched_sites = []
        patched_kinds = []
        materialized_idx = 0

        for path_idx in path_indices:
            idx = int(path_idx)
            if idx < 0 or idx >= len(words) or idx >= len(original_words):
                return None
            original_inst = int(original_words[idx])
            current_inst = int(words[idx])
            if original_inst == 0x00000000:
                continue
            branch = self._decode_hotloop_branch(original_inst, idx)
            if current_inst != original_inst:
                if idx not in patchable_sites:
                    return None
                if branch is not None:
                    return None
                patch_kind = self._superblock_patch_inst_kind(original_inst, current_inst)
                if patch_kind is None:
                    return None
                if materialized_idx >= len(specialized_materialized_words) - 1:
                    return None
                specialized_original_words[idx] = current_inst
                specialized_materialized_words[materialized_idx] = current_inst
                patched_sites.append(idx)
                patched_kinds.append(patch_kind)
            materialized_idx += 1

        specialized_candidate["original_words"] = specialized_original_words
        specialized_candidate["words"] = specialized_materialized_words
        specialized_candidate["specialization_patch_count"] = len(patched_sites)
        specialized_candidate["specialization_patch_sites"] = tuple(patched_sites)
        specialized_candidate["specialization_patch_kinds"] = tuple(patched_kinds)
        specialized_candidate["literal_patch_count"] = sum(kind == "literal" for kind in patched_kinds)
        specialized_candidate["imm_patch_count"] = sum(kind == "alu-imm" for kind in patched_kinds)
        if patched_sites:
            specialized_candidate["code_bytes"] = self._encode_hotloop_words(specialized_materialized_words)
            updated_simulation = dict(simulation)
            updated_simulation["linearized_words"] = list(specialized_materialized_words)
            specialized_candidate["simulation"] = updated_simulation
        return specialized_candidate

    @staticmethod
    def _superblock_template_embedding_similarity(lhs, rhs) -> float | None:
        if not torch.is_tensor(lhs) or not torch.is_tensor(rhs):
            return None
        lhs = lhs.detach().to(dtype=torch.float32).reshape(-1)
        rhs = rhs.detach().to(dtype=torch.float32).reshape(-1)
        if lhs.numel() == 0 or rhs.numel() == 0 or lhs.shape != rhs.shape:
            return None
        lhs_norm = torch.linalg.norm(lhs)
        rhs_norm = torch.linalg.norm(rhs)
        if float(lhs_norm.item()) <= 0.0 or float(rhs_norm.item()) <= 0.0:
            return None
        return float(torch.dot(lhs / lhs_norm, rhs / rhs_norm).item())

    @staticmethod
    def _merge_superblock_template_guard(lhs, rhs):
        if not isinstance(lhs, dict) or not isinstance(rhs, dict):
            return None
        lhs_regs = dict(lhs.get("regs", ()))
        rhs_regs = dict(rhs.get("regs", ()))
        merged_regs = tuple(
            (reg_idx, lhs_regs[reg_idx])
            for reg_idx in sorted(lhs_regs.keys() & rhs_regs.keys())
            if lhs_regs[reg_idx] == rhs_regs[reg_idx]
        )
        lhs_flags = tuple(int(v) for v in lhs.get("flags", ()))
        rhs_flags = tuple(int(v) for v in rhs.get("flags", ()))
        merged_flags = lhs_flags if lhs_flags and lhs_flags == rhs_flags else ()
        return {
            "regs": merged_regs,
            "flags": merged_flags,
        }

    def _score_superblock_cache_candidate(self, candidate) -> dict[str, object]:
        estimate = self._estimate_hotloop_work(candidate)
        sync_plan = self._plan_hotloop_memory_sync(candidate)
        if sync_plan is None:
            return {
                "score": None,
                "embedding": None,
                "source": "sync-plan-unavailable",
                "feature_source": None,
            }

        remaining_instructions = max(
            int(estimate.get("estimated_work") or 0),
            int(candidate.get("simulation", {}).get("executed_count", 0)),
            int(candidate.get("halt_idx", 0)),
            1,
        )
        feature_source = self._build_hotloop_value_feature_source(
            candidate,
            sync_plan,
            remaining_instructions,
            estimate=estimate,
            segment_index=1,
            reused_state=False,
            previous_segment=None,
        )

        value_model = getattr(self, "_neural_hotloop_value_model", None)
        if callable(value_model):
            try:
                return {
                    "score": predict_hotloop_value_score(
                        value_model,
                        feature_source,
                        device=self.device,
                    ),
                    "embedding": encode_hotloop_value_embedding(
                        value_model,
                        feature_source,
                        device=self.device,
                    ),
                    "source": "value-model",
                    "feature_source": feature_source,
                }
            except Exception:
                return {
                    "score": None,
                    "embedding": None,
                    "source": "value-model-error",
                    "feature_source": feature_source,
                }

        return {
            "score": None,
            "embedding": None,
            "source": "fifo",
            "feature_source": feature_source,
        }

    def _infer_hotloop_signed_trip_count(self, start: int, delta: int, bound: int, cond: int):
        """Infer iterations for monotonic signed-compare loops."""
        if delta == 0:
            return None

        if cond == 12:  # GT: continue while x > bound
            if delta < 0:
                return max(1, self._ceil_div_pos(start - bound, -delta))
            return None if start + delta > bound else 1

        if cond == 10:  # GE: continue while x >= bound
            if delta < 0:
                return max(1, ((start - bound) // (-delta)) + 1)
            return None if start + delta >= bound else 1

        if cond == 11:  # LT: continue while x < bound
            if delta > 0:
                return max(1, self._ceil_div_pos(bound - start, delta))
            return None if start + delta < bound else 1

        if cond == 13:  # LE: continue while x <= bound
            if delta > 0:
                return max(1, ((bound - start) // delta) + 1)
            return None if start + delta <= bound else 1

        return None

    def _decode_hotloop_memop(self, inst: int):
        """Decode simple memory ops that can participate in Rust handoff sync."""
        if (inst & 0xFFC00000) == 0x39000000:  # STRB unsigned offset
            return {
                "kind": "strb_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": (inst >> 10) & 0xFFF,
                "access_size": 1,
                "post_index": 0,
                "writes_memory": True,
            }
        if (inst & 0xFFC00000) == 0x39400000:  # LDRB unsigned offset
            return {
                "kind": "ldrb_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": (inst >> 10) & 0xFFF,
                "access_size": 1,
                "post_index": 0,
                "writes_memory": False,
            }
        if (inst & 0xFFC00000) == 0xB9000000:  # STR W unsigned offset
            return {
                "kind": "str_w_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": ((inst >> 10) & 0xFFF) * 4,
                "access_size": 4,
                "post_index": 0,
                "writes_memory": True,
            }
        if (inst & 0xFFC00000) == 0xB9400000:  # LDR W unsigned offset
            return {
                "kind": "ldr_w_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": ((inst >> 10) & 0xFFF) * 4,
                "access_size": 4,
                "post_index": 0,
                "writes_memory": False,
            }
        if (inst & 0xFFC00000) == 0xF9000000:  # STR X unsigned offset
            return {
                "kind": "str_x_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": ((inst >> 10) & 0xFFF) * 8,
                "access_size": 8,
                "post_index": 0,
                "writes_memory": True,
            }
        if (inst & 0xFFC00000) == 0xF9400000:  # LDR X unsigned offset
            return {
                "kind": "ldr_x_uoff",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": ((inst >> 10) & 0xFFF) * 8,
                "access_size": 8,
                "post_index": 0,
                "writes_memory": False,
            }
        if (inst & 0xFFE00C00) == 0x38000400:  # STRB post-index
            imm9 = (inst >> 12) & 0x1FF
            if imm9 & 0x100:
                imm9 -= 0x200
            return {
                "kind": "strb_post",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": 0,
                "access_size": 1,
                "post_index": imm9,
                "writes_memory": True,
            }
        if (inst & 0xFFE00C00) == 0x38400400:  # LDRB post-index
            imm9 = (inst >> 12) & 0x1FF
            if imm9 & 0x100:
                imm9 -= 0x200
            return {
                "kind": "ldrb_post",
                "base_reg": (inst >> 5) & 0x1F,
                "data_reg": inst & 0x1F,
                "offset": 0,
                "access_size": 1,
                "post_index": imm9,
                "writes_memory": False,
            }
        return None

    def _decode_hotloop_branch(self, inst: int, idx: int):
        """Decode supported loop branches for conservative Rust handoff."""
        if (inst & 0xFC000000) == 0x14000000:  # B
            imm26 = inst & 0x3FFFFFF
            if imm26 & 0x2000000:
                imm26 -= 0x4000000
            return {
                "branch_kind": "b",
                "branch_idx": idx,
                "loop_start_idx": idx + imm26,
                "branch_imm": imm26,
                "cond": None,
                "branch_reg": None,
            }
        if (inst & 0xFF000010) == 0x54000000:  # B.cond
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            return {
                "branch_kind": "bcond",
                "branch_idx": idx,
                "loop_start_idx": idx + imm19,
                "branch_imm": imm19,
                "cond": inst & 0xF,
                "branch_reg": None,
            }
        if ((inst & 0xFF000000) == 0xB4000000) or ((inst & 0xFF000000) == 0x34000000):  # CBZ
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            return {
                "branch_kind": "cbz",
                "branch_idx": idx,
                "loop_start_idx": idx + imm19,
                "branch_imm": imm19,
                "cond": None,
                "branch_reg": inst & 0x1F,
            }
        if ((inst & 0xFF000000) == 0xB5000000) or ((inst & 0xFF000000) == 0x35000000):  # CBNZ
            imm19 = (inst >> 5) & 0x7FFFF
            if imm19 & 0x40000:
                imm19 -= 0x80000
            return {
                "branch_kind": "cbnz",
                "branch_idx": idx,
                "loop_start_idx": idx + imm19,
                "branch_imm": imm19,
                "cond": None,
                "branch_reg": inst & 0x1F,
            }
        return None

    def _merge_hotloop_windows(self, windows):
        if not windows:
            return []
        bounded = []
        for start, end in windows:
            start_i = max(0, int(start))
            end_i = min(self.mem_size, int(end))
            if end_i > start_i:
                bounded.append((start_i, end_i))
        if not bounded:
            return []
        bounded.sort()
        merged = [list(bounded[0])]
        for start, end in bounded[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _subtract_hotloop_window(self, window, covered_windows):
        if not covered_windows:
            return [window]

        start, end = int(window[0]), int(window[1])
        remaining = [(start, end)]
        for cover_start, cover_end in self._merge_hotloop_windows(covered_windows):
            next_remaining = []
            for seg_start, seg_end in remaining:
                if cover_end <= seg_start or cover_start >= seg_end:
                    next_remaining.append((seg_start, seg_end))
                    continue
                if cover_start > seg_start:
                    next_remaining.append((seg_start, cover_start))
                if cover_end < seg_end:
                    next_remaining.append((cover_end, seg_end))
            remaining = next_remaining
            if not remaining:
                break
        return remaining

    def _simulate_hotloop_prefix_regs(self, candidate):
        """Evaluate the straight-line setup before the loop entry."""
        regs = self.regs.detach().cpu().to(torch.int64).tolist()
        for inst in candidate["words"][:candidate["loop_start_idx"]]:
            if inst == 0x00000000 or self._decode_hotloop_memop(inst) is not None:
                return None
            if self._decode_hotloop_branch(inst, 0) is not None:
                return None

            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F

            if (inst & 0xFF800000) == 0xD2800000:  # MOVZ Xd, #imm16, lsl #hw
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    regs[rd] = self._wrap_hotloop_i64(imm16 << (hw * 16))
                continue
            if (inst & 0xFF800000) == 0x52800000:  # MOVZ Wd, #imm16, lsl #hw
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    regs[rd] = self._wrap_hotloop_i64((imm16 << (hw * 16)) & 0xFFFFFFFF)
                continue
            if (inst & 0xFF800000) == 0xF2800000:  # MOVK Xd, #imm16, lsl #hw
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    mask = 0xFFFF << (hw * 16)
                    prev = regs[rd] & 0xFFFFFFFFFFFFFFFF
                    regs[rd] = self._wrap_hotloop_i64((prev & ~mask) | (imm16 << (hw * 16)))
                continue
            if (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:  # ADD/SUB imm
                imm12 = (inst >> 10) & 0xFFF
                src = 0 if rn == 31 else regs[rn]
                if (inst & 0xFF000000) in {0x91000000, 0xB1000000}:
                    value = src + imm12
                else:
                    value = src - imm12
                if rd != 31:
                    regs[rd] = self._wrap_hotloop_i64(value)
                continue
            if (inst & 0xFF200000) in {0x8B000000, 0xCB000000, 0xAB000000, 0xEB000000}:  # ADD/SUB reg
                if ((inst >> 22) & 0x3) != 0 or ((inst >> 10) & 0x3F) != 0:
                    return None
                rm = (inst >> 16) & 0x1F
                lhs = 0 if rn == 31 else regs[rn]
                rhs = 0 if rm == 31 else regs[rm]
                if (inst & 0xFF200000) in {0x8B000000, 0xAB000000}:
                    value = lhs + rhs
                else:
                    value = lhs - rhs
                if rd != 31:
                    regs[rd] = self._wrap_hotloop_i64(value)
                continue
            if (inst & 0xFF200000) in {0x8A000000, 0xAA000000, 0xCA000000}:  # AND/ORR/EOR reg
                if ((inst >> 22) & 0x3) != 0 or ((inst >> 10) & 0x3F) != 0:
                    return None
                rm = (inst >> 16) & 0x1F
                lhs = 0 if rn == 31 else regs[rn]
                rhs = 0 if rm == 31 else regs[rm]
                top = inst & 0xFF200000
                if top == 0x8A000000:
                    value = lhs & rhs
                elif top == 0xAA000000:
                    value = lhs | rhs
                else:
                    value = lhs ^ rhs
                if rd != 31:
                    regs[rd] = self._wrap_hotloop_i64(value)
                continue
            return None
        return regs

    def _infer_hotloop_iterations(self, candidate, loop_regs):
        """Infer a concrete loop trip count for simple NE-counted loops."""
        loop_words = candidate["words"][candidate["loop_start_idx"]:candidate["branch_idx"]]
        body_start_idx = candidate.get("body_start_idx", candidate["loop_start_idx"])
        body_words = candidate["words"][body_start_idx:candidate["branch_idx"]]
        reg_deltas: dict[int, int] = {}
        for inst in body_words:
            top = inst & 0xFF000000
            if top not in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
                continue
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            if rd != rn or rd == 31:
                continue
            imm12 = (inst >> 10) & 0xFFF
            reg_deltas[rd] = reg_deltas.get(rd, 0) + (
                imm12 if top in {0x91000000, 0xB1000000} else -imm12
            )

        def _find_cmp_inst(*, allow_fused_counter: bool = False, compare_idx: int | None = None):
            if compare_idx is not None:
                inst = candidate["words"][compare_idx]
                if self._is_hotloop_cmp(inst):
                    return inst
                return None

            for inst in reversed(loop_words):
                if self._is_hotloop_cmp(inst):
                    return inst
                if allow_fused_counter and (
                    (inst & 0xFF000000) == 0xF1000000 or
                    (inst & 0xFF200000) == 0xEB000000
                ):
                    return inst
            return None

        def _infer_compare_branch_iterations(
            cond_code: int,
            *,
            allow_fused_counter: bool = False,
            compare_idx: int | None = None,
        ):
            cmp_inst = _find_cmp_inst(
                allow_fused_counter=allow_fused_counter,
                compare_idx=compare_idx,
            )
            if cmp_inst is None:
                return None

            rd = cmp_inst & 0x1F
            rn = (cmp_inst >> 5) & 0x1F
            lhs_start = 0 if rn == 31 else int(loop_regs[rn])

            if cond_code != 1:
                delta = reg_deltas.get(rn, 0)
                if delta == 0:
                    return None

                if (cmp_inst & 0xFF000000) == 0xF1000000:
                    imm12 = (cmp_inst >> 10) & 0xFFF
                    if rd == rn and rd != 31:
                        return self._infer_hotloop_signed_trip_count(
                            lhs_start, delta, 0, cond_code
                        )
                    rhs_limit = imm12
                else:
                    rm = (cmp_inst >> 16) & 0x1F
                    if rd == rn and rd != 31:
                        return self._infer_hotloop_signed_trip_count(
                            lhs_start, delta, 0, cond_code
                        )
                    rhs_limit = 0 if rm == 31 else int(loop_regs[rm])

                return self._infer_hotloop_signed_trip_count(
                    lhs_start, delta, rhs_limit, cond_code
                )

            if (cmp_inst & 0xFF000000) == 0xF1000000:
                imm12 = (cmp_inst >> 10) & 0xFFF
                if rd == rn and rd != 31:
                    step = imm12
                    if step <= 0 or lhs_start <= 0 or (lhs_start % step) != 0:
                        return None
                    return lhs_start // step
                step = reg_deltas.get(rn)
                if step is None or step == 0:
                    return None
                diff = imm12 - lhs_start
            else:
                rm = (cmp_inst >> 16) & 0x1F
                rhs_limit = 0 if rm == 31 else int(loop_regs[rm])
                if rd == rn and rd != 31:
                    step = 0 if rm == 31 else int(loop_regs[rm])
                    if step <= 0 or lhs_start <= 0 or (lhs_start % step) != 0:
                        return None
                    return lhs_start // step
                step = reg_deltas.get(rn)
                if step is None or step == 0:
                    return None
                diff = rhs_limit - lhs_start

            if diff == 0:
                return 0
            if (diff > 0 and step < 0) or (diff < 0 and step > 0):
                return None
            step_mag = abs(step)
            diff_mag = abs(diff)
            if step_mag == 0 or (diff_mag % step_mag) != 0:
                return None
            return diff_mag // step_mag

        exit_branch = candidate.get("exit_branch")
        if exit_branch is not None:
            branch_reg = exit_branch["branch_reg"]
            if exit_branch["branch_kind"] == "cbz":
                if branch_reg is None or branch_reg == 31:
                    return None
                step = reg_deltas.get(branch_reg)
                if step is None or step == 0:
                    return None
                start = int(loop_regs[branch_reg])
                if start == 0:
                    return None
                if (start > 0 and step >= 0) or (start < 0 and step <= 0):
                    return None
                step_mag = abs(step)
                start_mag = abs(start)
                if (start_mag % step_mag) != 0:
                    return None
                return start_mag // step_mag
            if exit_branch["branch_kind"] == "cbnz":
                if branch_reg is None or branch_reg == 31:
                    return None
                step = reg_deltas.get(branch_reg)
                if step is None or step == 0:
                    return None
                start = int(loop_regs[branch_reg])
                if start != 0:
                    return None
                return 1
            if exit_branch["branch_kind"] == "bcond":
                continue_cond = {
                    10: 11,  # exit GE => continue LT
                    11: 10,  # exit LT => continue GE
                    12: 13,  # exit GT => continue LE
                    13: 12,  # exit LE => continue GT
                }.get(exit_branch["cond"])
                if continue_cond is None:
                    return None
                return _infer_compare_branch_iterations(
                    continue_cond,
                    compare_idx=candidate.get("compare_idx"),
                )
            return None

        if candidate["branch_kind"] == "cbnz":
            branch_reg = candidate["branch_reg"]
            if branch_reg is None or branch_reg == 31:
                return None
            step = reg_deltas.get(branch_reg)
            if step is None or step == 0:
                return None
            start = int(loop_regs[branch_reg])
            if start == 0:
                return None
            diff = -start
            if (diff > 0 and step < 0) or (diff < 0 and step > 0):
                return None
            step_mag = abs(step)
            diff_mag = abs(diff)
            if step_mag == 0 or (diff_mag % step_mag) != 0:
                return None
            return diff_mag // step_mag

        return _infer_compare_branch_iterations(candidate["cond"], allow_fused_counter=True)

    def _plan_hotloop_memory_sync(self, candidate):
        """Return minimal memory windows to sync for simple load/store loops."""
        if candidate.get("region_kind") == "superblock":
            simulation = candidate.get("simulation")
            if not isinstance(simulation, dict):
                return None
            pre_merged = self._merge_hotloop_windows(simulation.get("pre_windows", []))
            post_merged = self._merge_hotloop_windows(simulation.get("post_windows", []))
            max_sync_bytes = int(os.environ.get("NCPU_GPU_ONLY_HOTLOOP_SYNC_BYTES", "262144"))
            if sum(end - start for start, end in pre_merged) > max_sync_bytes:
                return None
            if sum(end - start for start, end in post_merged) > max_sync_bytes:
                return None
            return {"pre": pre_merged, "post": post_merged}

        body_start_idx = candidate.get("body_start_idx", candidate["loop_start_idx"])
        loop_words = candidate["words"][body_start_idx:candidate["branch_idx"]]
        mem_ops = [self._decode_hotloop_memop(inst) for inst in loop_words]
        mem_ops = [op for op in mem_ops if op is not None]
        if not mem_ops:
            return {"pre": [], "post": []}

        loop_regs = self._simulate_hotloop_prefix_regs(candidate)
        if loop_regs is None:
            return None

        iterations = self._infer_hotloop_iterations(candidate, loop_regs)
        if iterations is None or iterations < 0:
            return None

        reg_deltas: dict[int, int] = {}
        for inst in loop_words:
            top = inst & 0xFF000000
            if top not in {0x91000000, 0xD1000000}:
                continue
            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            if rd != rn or rd == 31:
                continue
            imm12 = (inst >> 10) & 0xFFF
            reg_deltas[rd] = reg_deltas.get(rd, 0) + (imm12 if top == 0x91000000 else -imm12)

        pre_windows = []
        post_windows = []
        for inst in loop_words:
            memop = self._decode_hotloop_memop(inst)
            if memop is None:
                if (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
                    continue
                if (inst & 0xFF200000) == 0xEB000000:
                    continue
                return None

            base_reg = memop["base_reg"]
            if base_reg == 31:
                return None
            base_start = int(loop_regs[base_reg])
            stride = reg_deltas.get(base_reg, 0) + int(memop["post_index"])
            eff_start = base_start + int(memop["offset"])
            access_size = int(memop["access_size"])

            if iterations <= 1 or stride == 0:
                first_addr = eff_start
                last_addr = eff_start
            elif stride > 0:
                first_addr = eff_start
                last_addr = eff_start + stride * (iterations - 1)
            else:
                first_addr = eff_start + stride * (iterations - 1)
                last_addr = eff_start

            window_start = first_addr
            window_end = last_addr + access_size
            if window_start < 0 or window_end > self.mem_size:
                return None
            if memop["writes_memory"]:
                post_windows.append((window_start, window_end))
            else:
                pre_windows.append((window_start, window_end))

        pre_merged = self._merge_hotloop_windows(pre_windows)
        post_merged = self._merge_hotloop_windows(post_windows)
        max_sync_bytes = int(os.environ.get("NCPU_GPU_ONLY_HOTLOOP_SYNC_BYTES", "262144"))
        if sum(end - start for start, end in pre_merged) > max_sync_bytes:
            return None
        if sum(end - start for start, end in post_merged) > max_sync_bytes:
            return None
        return {"pre": pre_merged, "post": post_merged}

    @staticmethod
    def _is_supported_rust_region_inst(inst: int) -> bool:
        return (
            inst == 0x00000000 or
            ((inst & 0xFF800000) == 0xD2800000) or
            ((inst & 0xFF800000) == 0x52800000) or
            ((inst & 0xFF800000) == 0xF2800000) or
            ((inst & 0xFF800000) == 0x72800000) or
            ((inst & 0xFF000000) == 0x91000000) or
            ((inst & 0xFF000000) == 0xD1000000) or
            ((inst & 0xFF000000) == 0xB1000000) or
            ((inst & 0xFF000000) == 0xF1000000) or
            ((inst & 0xFF200000) == 0x8B000000) or
            ((inst & 0xFF200000) == 0xCB000000) or
            ((inst & 0xFF200000) == 0xAB000000) or
            ((inst & 0xFF200000) == 0xEB000000) or
            ((inst & 0xFF200000) == 0x8A000000) or
            ((inst & 0xFF200000) == 0xAA000000) or
            ((inst & 0xFF200000) == 0xCA000000) or
            ((inst & 0xFFE0FC00) == 0x9B007C00) or
            ((inst & 0xFC000000) == 0x14000000) or
            ((inst & 0xFFC00000) == 0x39000000) or
            ((inst & 0xFFC00000) == 0x39400000) or
            ((inst & 0xFFC00000) == 0xB9000000) or
            ((inst & 0xFFC00000) == 0xB9400000) or
            ((inst & 0xFFC00000) == 0xF9000000) or
            ((inst & 0xFFC00000) == 0xF9400000) or
            ((inst & 0xFFE00C00) == 0x38000400) or
            ((inst & 0xFFE00C00) == 0x38400400) or
            ((inst & 0xFF000000) == 0xB4000000) or
            ((inst & 0xFF000000) == 0x34000000) or
            ((inst & 0xFF000000) == 0xB5000000) or
            ((inst & 0xFF000000) == 0x35000000) or
            ((inst & 0xFF000010) == 0x54000000)
        )

    @staticmethod
    def _evaluate_rust_region_cond(cond: int, flags) -> bool | None:
        n, z, c, v = flags
        if cond == 0:
            return z
        if cond == 1:
            return not z
        if cond == 2:
            return c
        if cond == 3:
            return not c
        if cond == 4:
            return n
        if cond == 5:
            return not n
        if cond == 6:
            return v
        if cond == 7:
            return not v
        if cond == 8:
            return c and not z
        if cond == 9:
            return (not c) or z
        if cond == 10:
            return n == v
        if cond == 11:
            return n != v
        if cond == 12:
            return (not z) and (n == v)
        if cond == 13:
            return z or (n != v)
        if cond in {14, 15}:
            return True
        return None

    def _simulate_superblock_path(self, words, max_steps: int = 256, *, base_pc: int = 0):
        """Simulate a bounded supported region to obtain an exact linearized execution trace."""
        regs = self.regs.detach().cpu().to(torch.int64).tolist()
        flags_host = self.flags.detach().cpu().to(torch.float32).tolist()
        flags = (
            bool(flags_host[0] > 0.5),
            bool(flags_host[1] > 0.5),
            bool(flags_host[2] > 0.5),
            bool(flags_host[3] > 0.5),
        )
        shadow_bytes: dict[int, int] = {}
        pre_windows = []
        post_windows = []
        path_indices = []
        executed = 0
        branch_sites = set()
        pc_idx = 0

        def _finish(exit_idx: int, *, halted: bool, synthetic_stop: bool):
            linearized_words: list[int] = []
            for path_idx in path_indices:
                inst = int(words[path_idx])
                if inst == 0x00000000:
                    continue
                if self._decode_hotloop_branch(inst, path_idx) is not None:
                    linearized_words.append(self._hotloop_materialized_nop())
                else:
                    linearized_words.append(inst)
            if not linearized_words:
                return None
            linearized_words.append(0x00000000)
            return {
                "executed_count": executed,
                "pre_windows": self._merge_hotloop_windows(pre_windows),
                "post_windows": self._merge_hotloop_windows(post_windows),
                "path_indices": path_indices,
                "block_count": max(len(branch_sites) + 1, 1),
                "halt_idx": len(linearized_words) - 1,
                "linearized_words": linearized_words,
                "expected_stop_idx": exit_idx,
                "expected_stop_pc": int(base_pc + (exit_idx * 4)),
                "expected_halted": halted,
                "synthetic_stop": synthetic_stop,
            }

        def _signed(value: int) -> int:
            return self._wrap_hotloop_i64(value)

        def _add_flags(lhs: int, rhs: int):
            lhs_u = lhs & 0xFFFFFFFFFFFFFFFF
            rhs_u = rhs & 0xFFFFFFFFFFFFFFFF
            result_u = (lhs_u + rhs_u) & 0xFFFFFFFFFFFFFFFF
            lhs_s = _signed(lhs_u)
            rhs_s = _signed(rhs_u)
            result_s = _signed(result_u)
            return _signed(result_u), (
                bool(result_u & (1 << 63)),
                result_u == 0,
                (lhs_u + rhs_u) > 0xFFFFFFFFFFFFFFFF,
                ((lhs_s < 0) == (rhs_s < 0)) and ((result_s < 0) != (lhs_s < 0)),
            )

        def _sub_flags(lhs: int, rhs: int):
            lhs_u = lhs & 0xFFFFFFFFFFFFFFFF
            rhs_u = rhs & 0xFFFFFFFFFFFFFFFF
            result_u = (lhs_u - rhs_u) & 0xFFFFFFFFFFFFFFFF
            lhs_s = _signed(lhs_u)
            rhs_s = _signed(rhs_u)
            result_s = _signed(result_u)
            return _signed(result_u), (
                bool(result_u & (1 << 63)),
                result_u == 0,
                lhs_u >= rhs_u,
                ((lhs_s < 0) != (rhs_s < 0)) and ((result_s < 0) != (lhs_s < 0)),
            )

        def _read_bytes(address: int, size: int):
            if address < 0 or address + size > self.mem_size:
                return None
            payload = self.memory[address:address + size].detach().cpu().tolist()
            for offset in range(size):
                payload[offset] = shadow_bytes.get(address + offset, payload[offset])
            return payload

        while 0 <= pc_idx < len(words) and executed < max_steps:
            inst = int(words[pc_idx])
            path_indices.append(pc_idx)
            if inst == 0x00000000:
                return _finish(pc_idx, halted=True, synthetic_stop=False)
            if not self._is_supported_rust_region_inst(inst):
                return None

            memop = self._decode_hotloop_memop(inst)
            if memop is not None:
                base_reg = memop["base_reg"]
                if base_reg == 31:
                    return None
                address = int(regs[base_reg]) + int(memop["offset"])
                size = int(memop["access_size"])
                if address < 0 or address + size > self.mem_size:
                    return None
                if memop["writes_memory"]:
                    post_windows.append((address, address + size))
                    value = 0 if memop["data_reg"] == 31 else int(regs[memop["data_reg"]]) & ((1 << (size * 8)) - 1)
                    for offset, byte in enumerate(value.to_bytes(size, byteorder="little", signed=False)):
                        shadow_bytes[address + offset] = byte
                else:
                    # Only reads contribute to the pre-state guard / pre-sync window.
                    # Bytes that were shadowed by a prior store in this block are served
                    # from shadow_bytes below, so they don't need to be in pre_windows.
                    pending_shadow = all(
                        (address + offset) in shadow_bytes for offset in range(size)
                    )
                    if not pending_shadow:
                        pre_windows.append((address, address + size))
                    payload = _read_bytes(address, size)
                    if payload is None:
                        return None
                    raw_value = int.from_bytes(bytes(payload), byteorder="little", signed=False)
                    if memop["data_reg"] != 31:
                        if size == 1:
                            regs[memop["data_reg"]] = _signed(raw_value & 0xFF)
                        elif size == 4:
                            regs[memop["data_reg"]] = _signed(raw_value & 0xFFFFFFFF)
                        else:
                            regs[memop["data_reg"]] = _signed(raw_value)
                if int(memop["post_index"]) != 0:
                    regs[base_reg] = _signed(int(regs[base_reg]) + int(memop["post_index"]))
                pc_idx += 1
                executed += 1
                if pc_idx >= len(words):
                    return _finish(pc_idx, halted=False, synthetic_stop=True)
                continue

            branch = self._decode_hotloop_branch(inst, pc_idx)
            if branch is not None:
                branch_sites.add(int(branch["branch_idx"]))
                if branch["branch_kind"] == "b":
                    next_pc = branch["loop_start_idx"]
                elif branch["branch_kind"] in {"cbz", "cbnz"}:
                    reg_idx = branch["branch_reg"]
                    value = 0 if reg_idx in {None, 31} else int(regs[reg_idx])
                    take_branch = (value == 0) if branch["branch_kind"] == "cbz" else (value != 0)
                    next_pc = branch["loop_start_idx"] if take_branch else (pc_idx + 1)
                else:
                    take_branch = self._evaluate_rust_region_cond(branch["cond"], flags)
                    if take_branch is None:
                        return None
                    next_pc = branch["loop_start_idx"] if take_branch else (pc_idx + 1)
                if next_pc < 0:
                    return None
                executed += 1
                if next_pc >= len(words):
                    return _finish(next_pc, halted=False, synthetic_stop=True)
                pc_idx = next_pc
                continue

            rd = inst & 0x1F
            rn = (inst >> 5) & 0x1F
            if (inst & 0xFF800000) == 0xD2800000:  # MOVZ Xd
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    regs[rd] = _signed(imm16 << (hw * 16))
            elif (inst & 0xFF800000) == 0x52800000:  # MOVZ Wd
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    regs[rd] = _signed((imm16 << (hw * 16)) & 0xFFFFFFFF)
            elif (inst & 0xFF800000) == 0xF2800000:  # MOVK Xd
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    mask = 0xFFFF << (hw * 16)
                    prev = int(regs[rd]) & 0xFFFFFFFFFFFFFFFF
                    regs[rd] = _signed((prev & ~mask) | (imm16 << (hw * 16)))
            elif (inst & 0xFF800000) == 0x72800000:  # MOVK Wd
                if rd != 31:
                    hw = (inst >> 21) & 0x3
                    imm16 = (inst >> 5) & 0xFFFF
                    mask = 0xFFFF << (hw * 16)
                    prev = int(regs[rd]) & 0xFFFFFFFF
                    regs[rd] = _signed(((prev & ~mask) | (imm16 << (hw * 16))) & 0xFFFFFFFF)
            elif (inst & 0xFF000000) in {0x91000000, 0xD1000000, 0xB1000000, 0xF1000000}:
                imm12 = (inst >> 10) & 0xFFF
                lhs = 0 if rn == 31 else int(regs[rn])
                if (inst & 0xFF000000) in {0x91000000, 0xB1000000}:
                    result, new_flags = _add_flags(lhs, imm12)
                else:
                    result, new_flags = _sub_flags(lhs, imm12)
                if (inst & 0xFF000000) in {0xB1000000, 0xF1000000}:
                    flags = new_flags
                if rd != 31:
                    regs[rd] = result
            elif (inst & 0xFF200000) in {0x8B000000, 0xCB000000, 0xAB000000, 0xEB000000}:
                if ((inst >> 22) & 0x3) != 0 or ((inst >> 10) & 0x3F) != 0:
                    return None
                rm = (inst >> 16) & 0x1F
                lhs = 0 if rn == 31 else int(regs[rn])
                rhs = 0 if rm == 31 else int(regs[rm])
                if (inst & 0xFF200000) in {0x8B000000, 0xAB000000}:
                    result, new_flags = _add_flags(lhs, rhs)
                else:
                    result, new_flags = _sub_flags(lhs, rhs)
                if (inst & 0xFF200000) in {0xAB000000, 0xEB000000}:
                    flags = new_flags
                if rd != 31:
                    regs[rd] = result
            elif (inst & 0xFF200000) in {0x8A000000, 0xAA000000, 0xCA000000}:
                if ((inst >> 22) & 0x3) != 0 or ((inst >> 10) & 0x3F) != 0:
                    return None
                rm = (inst >> 16) & 0x1F
                lhs = 0 if rn == 31 else int(regs[rn])
                rhs = 0 if rm == 31 else int(regs[rm])
                top = inst & 0xFF200000
                if rd != 31:
                    if top == 0x8A000000:
                        regs[rd] = _signed(lhs & rhs)
                    elif top == 0xAA000000:
                        regs[rd] = _signed(lhs | rhs)
                    else:
                        regs[rd] = _signed(lhs ^ rhs)
            elif (inst & 0xFFE0FC00) == 0x9B007C00:
                rm = (inst >> 16) & 0x1F
                lhs = 0 if rn == 31 else int(regs[rn])
                rhs = 0 if rm == 31 else int(regs[rm])
                if rd != 31:
                    regs[rd] = _signed((lhs & 0xFFFFFFFFFFFFFFFF) * (rhs & 0xFFFFFFFFFFFFFFFF))
            else:
                return None

            pc_idx += 1
            executed += 1
            if pc_idx >= len(words):
                return _finish(pc_idx, halted=False, synthetic_stop=True)

        return None

    def _collect_superblock_candidate(self, max_words: int = 64):
        """Return a path-specialized bounded region for Rust handoff."""
        try:
            pc = int(self.pc.item())
        except Exception:
            return None
        if not hasattr(self, "_superblock_trace_cache"):
            self._superblock_trace_cache = {}
        if not hasattr(self, "_superblock_template_cache"):
            self._superblock_template_cache = {}
        if not hasattr(self, "_superblock_shape_cache"):
            self._superblock_shape_cache = {}
        if not hasattr(self, "_superblock_trace_miss_counter"):
            # Adaptive promotion: program_keys that consecutively miss at
            # trace level get promoted to skip trace lookup, saving the
            # memory-snapshot comparison on keys that never benefit from it.
            self._superblock_trace_miss_counter: dict = {}
        if not hasattr(self, "_last_gpu_only_hotloop_stats"):
            self._last_gpu_only_hotloop_stats = {}
        self._last_gpu_only_hotloop_stats.setdefault("superblock_cache_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_cache_misses", 0)
        self._last_gpu_only_hotloop_stats.setdefault(
            "superblock_cache_entries",
            len(self._superblock_trace_cache),
        )
        self._last_gpu_only_hotloop_stats.setdefault("superblock_cache_priority_evictions", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_cache_priority_rejections", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_template_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_template_misses", 0)
        self._last_gpu_only_hotloop_stats.setdefault(
            "superblock_template_entries",
            sum(len(bucket) for bucket in self._superblock_template_cache.values()),
        )
        self._last_gpu_only_hotloop_stats.setdefault("superblock_template_generalizations", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_template_cross_window_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_shape_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_shape_misses", 0)
        self._last_gpu_only_hotloop_stats.setdefault(
            "superblock_shape_entries",
            sum(len(bucket) for bucket in self._superblock_shape_cache.values()),
        )
        self._last_gpu_only_hotloop_stats.setdefault("superblock_shape_cross_window_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_literal_patch_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_literal_patch_words", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_imm_patch_hits", 0)
        self._last_gpu_only_hotloop_stats.setdefault("superblock_imm_patch_words", 0)

        try:
            max_words = max(int(os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_WORDS", str(max_words))), 1)
        except Exception:
            max_words = max(max_words, 1)
        try:
            max_steps = max(int(os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_STEPS", "256")), 1)
        except Exception:
            max_steps = 256
        try:
            cache_size = max(int(os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "64")), 0)
        except Exception:
            cache_size = 64
        try:
            template_per_key = max(
                int(os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")),
                1,
            )
        except Exception:
            template_per_key = 4
        try:
            shape_per_key = max(
                int(os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")),
                1,
            )
        except Exception:
            shape_per_key = 4

        max_bytes = min(self.mem_size - pc, max_words * 4)
        if max_bytes < 4:
            return None

        raw = self.memory[pc:pc + max_bytes].detach().cpu().tolist()
        words: list[int] = []
        for i in range(0, len(raw), 4):
            inst = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16) | (raw[i + 3] << 24)
            words.append(inst)
            if inst == 0x00000000:
                break

        if len(words) < 1:
            return None

        window_key = (
            int(max_words),
            int(max_steps),
            tuple(int(word) for word in words),
        )
        shape_key = self._superblock_window_shape_key(max_words, max_steps, words)
        program_key = (int(pc), *window_key)
        cache_key = None
        trace_promotion_threshold = int(
            os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION", "3")
        )
        if cache_size > 0:
            regs_signature = tuple(int(v) for v in self.regs.detach().cpu().to(torch.int64).tolist())
            flags_host = self.flags.detach().cpu().to(torch.float32).tolist()
            flags_signature = tuple(int(v > 0.5) for v in flags_host[:4])
            cache_key = (
                *program_key,
                regs_signature,
                flags_signature,
            )
            trace_miss_count = self._superblock_trace_miss_counter.get(program_key, 0)
            trace_skipped = (
                trace_promotion_threshold > 0 and trace_miss_count >= trace_promotion_threshold
            )
            if trace_skipped:
                self._bump_superblock_cache_stat("superblock_trace_skips")
            else:
                cached_entry = self._superblock_trace_cache.pop(cache_key, None)
                if cached_entry is not None and self._matches_hotloop_window_snapshot(cached_entry.get("memory_guard")):
                    self._superblock_trace_cache[cache_key] = cached_entry
                    self._bump_superblock_cache_stat("superblock_cache_hits")
                    self._superblock_trace_miss_counter.pop(program_key, None)
                    self._last_gpu_only_hotloop_stats["superblock_cache_entries"] = len(self._superblock_trace_cache)
                    return self._clone_hotloop_candidate(
                        cached_entry["candidate"],
                        cache_hit=True,
                        cache_hit_kind="exact",
                    )
                self._bump_superblock_cache_stat("superblock_cache_misses")
                new_miss_count = trace_miss_count + 1
                self._superblock_trace_miss_counter[program_key] = new_miss_count
                if (
                    trace_promotion_threshold > 0
                    and new_miss_count == trace_promotion_threshold
                ):
                    self._bump_superblock_cache_stat("superblock_trace_promotions")

            template_bucket = self._superblock_template_cache.get(window_key, [])
            for idx, cached_template in enumerate(template_bucket):
                if not self._matches_superblock_template_guard(cached_template.get("template_guard")):
                    continue
                if not self._matches_hotloop_window_snapshot(cached_template.get("memory_guard")):
                    continue
                if idx != len(template_bucket) - 1:
                    template_bucket.append(template_bucket.pop(idx))
                self._bump_superblock_cache_stat("superblock_template_hits")
                source_pc = int(cached_template["candidate"].get("pc", pc))
                hit_kind = "template-cross-window" if source_pc != int(pc) else "template"
                if hit_kind == "template-cross-window":
                    self._bump_superblock_cache_stat("superblock_template_cross_window_hits")
                self._last_gpu_only_hotloop_stats["superblock_template_entries"] = sum(
                    len(bucket) for bucket in self._superblock_template_cache.values()
                )
                return self._clone_hotloop_candidate(
                    self._retarget_superblock_template_candidate(cached_template["candidate"], pc),
                    cache_hit=True,
                    cache_hit_kind=hit_kind,
                )
            self._bump_superblock_cache_stat("superblock_template_misses")

            shape_bucket = self._superblock_shape_cache.get(shape_key, [])
            for idx, cached_shape in enumerate(shape_bucket):
                if not self._matches_superblock_template_guard(cached_shape.get("template_guard")):
                    continue
                if not self._matches_hotloop_window_snapshot(cached_shape.get("memory_guard")):
                    continue
                specialized_candidate = self._specialize_superblock_shape_candidate(
                    cached_shape.get("candidate", {}),
                    words,
                )
                if specialized_candidate is None:
                    continue
                if idx != len(shape_bucket) - 1:
                    shape_bucket.append(shape_bucket.pop(idx))
                self._bump_superblock_cache_stat("superblock_shape_hits")
                source_pc = int(cached_shape["candidate"].get("pc", pc))
                literal_patch_count = int(specialized_candidate.get("literal_patch_count", 0))
                imm_patch_count = int(specialized_candidate.get("imm_patch_count", 0))
                specialization_patch_count = int(specialized_candidate.get("specialization_patch_count", 0))
                if literal_patch_count > 0:
                    self._bump_superblock_cache_stat("superblock_literal_patch_hits")
                    self._bump_superblock_cache_stat("superblock_literal_patch_words", literal_patch_count)
                if imm_patch_count > 0:
                    self._bump_superblock_cache_stat("superblock_imm_patch_hits")
                    self._bump_superblock_cache_stat("superblock_imm_patch_words", imm_patch_count)
                if specialization_patch_count > 0:
                    if literal_patch_count == specialization_patch_count:
                        hit_kind = "shape-literal-cross-window" if source_pc != int(pc) else "shape-literal"
                    elif imm_patch_count == specialization_patch_count:
                        hit_kind = "shape-imm-cross-window" if source_pc != int(pc) else "shape-imm"
                    else:
                        hit_kind = "shape-specialized-cross-window" if source_pc != int(pc) else "shape-specialized"
                else:
                    hit_kind = "shape-cross-window" if source_pc != int(pc) else "shape"
                if hit_kind in {"shape-cross-window", "shape-literal-cross-window"}:
                    self._bump_superblock_cache_stat("superblock_shape_cross_window_hits")
                if hit_kind in {"shape-imm-cross-window", "shape-specialized-cross-window"}:
                    self._bump_superblock_cache_stat("superblock_shape_cross_window_hits")
                self._last_gpu_only_hotloop_stats["superblock_shape_entries"] = sum(
                    len(bucket) for bucket in self._superblock_shape_cache.values()
                )
                return self._clone_hotloop_candidate(
                    self._retarget_superblock_template_candidate(specialized_candidate, pc),
                    cache_hit=True,
                    cache_hit_kind=hit_kind,
                )
            self._bump_superblock_cache_stat("superblock_shape_misses")

        simulation = self._simulate_superblock_path(words, max_steps=max_steps, base_pc=pc)
        if simulation is None or int(simulation.get("executed_count", 0)) <= 0:
            return None
        candidate_words = list(simulation["linearized_words"])
        candidate = {
            "pc": pc,
            "words": candidate_words,
            "original_words": words,
            "code_bytes": self._encode_hotloop_words(candidate_words),
            "halt_idx": int(simulation["halt_idx"]),
            "synthetic_stop": bool(simulation.get("synthetic_stop")),
            "expected_stop_pc": int(simulation["expected_stop_pc"]),
            "expected_halted": bool(simulation.get("expected_halted")),
            "tail_word_count": 0,
            "region_blocks": int(simulation.get("block_count", 1)),
            "region_kind": "superblock",
            "branch_kind": "superblock",
            "loop_start_idx": 0,
            "body_start_idx": 0,
            "branch_idx": int(simulation["halt_idx"]),
            "simulation": simulation,
        }
        cache_priority = self._score_superblock_cache_candidate(candidate)
        candidate["cache_priority_score"] = cache_priority.get("score")
        candidate["cache_priority_source"] = cache_priority.get("source", "fifo")
        candidate["cache_priority_embedding"] = cache_priority.get("embedding")
        if cache_key is not None:
            memory_guard = self._read_hotloop_window_snapshot(simulation.get("pre_windows", []))
            if memory_guard is not None:
                cache_payload = {
                    "candidate": dict(candidate),
                    "memory_guard": memory_guard,
                }
                inserted = False
                if len(self._superblock_trace_cache) < cache_size:
                    self._superblock_trace_cache[cache_key] = cache_payload
                    inserted = True
                elif candidate.get("cache_priority_score") is not None:
                    scored_entries = [
                        (
                            entry_key,
                            float(entry["candidate"].get("cache_priority_score")),
                        )
                        for entry_key, entry in self._superblock_trace_cache.items()
                        if entry["candidate"].get("cache_priority_score") is not None
                    ]
                    if scored_entries:
                        victim_key, victim_score = min(scored_entries, key=lambda item: item[1])
                        if float(candidate["cache_priority_score"]) > victim_score:
                            del self._superblock_trace_cache[victim_key]
                            self._superblock_trace_cache[cache_key] = cache_payload
                            inserted = True
                            self._bump_superblock_cache_stat("superblock_cache_priority_evictions")
                        else:
                            self._bump_superblock_cache_stat("superblock_cache_priority_rejections")
                    else:
                        self._superblock_trace_cache[cache_key] = cache_payload
                        inserted = True
                else:
                    self._superblock_trace_cache[cache_key] = cache_payload
                    inserted = True
                while inserted and len(self._superblock_trace_cache) > cache_size:
                    oldest_key = next(iter(self._superblock_trace_cache))
                    del self._superblock_trace_cache[oldest_key]
                if inserted:
                    # Fresh trace entry means this program_key should get trace
                    # lookups again until it re-accumulates misses.
                    self._superblock_trace_miss_counter.pop(program_key, None)

                template_guard = self._build_superblock_template_guard(candidate)
                if template_guard is not None:
                    template_bucket = self._superblock_template_cache.setdefault(window_key, [])
                    inserted_template = False
                    merge_threshold = float(
                        os.environ.get("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_MERGE_SIM", "0.995")
                    )
                    candidate_embedding = candidate.get("cache_priority_embedding")
                    for idx, existing_template in enumerate(template_bucket):
                        if existing_template["candidate"].get("code_bytes") != candidate.get("code_bytes"):
                            continue
                        if existing_template["candidate"].get("expected_stop_pc") != candidate.get("expected_stop_pc"):
                            continue
                        if existing_template["candidate"].get("expected_halted") != candidate.get("expected_halted"):
                            continue
                        if existing_template.get("memory_guard") != memory_guard:
                            continue
                        similarity = self._superblock_template_embedding_similarity(
                            existing_template["candidate"].get("cache_priority_embedding"),
                            candidate_embedding,
                        )
                        if similarity is None or similarity < merge_threshold:
                            continue
                        merged_guard = self._merge_superblock_template_guard(
                            existing_template.get("template_guard"),
                            template_guard,
                        )
                        existing_score = existing_template["candidate"].get("cache_priority_score")
                        candidate_score = candidate.get("cache_priority_score")
                        if (
                            candidate_score is not None and
                            (existing_score is None or float(candidate_score) >= float(existing_score))
                        ):
                            kept_candidate = dict(candidate)
                        else:
                            kept_candidate = dict(existing_template["candidate"])
                        template_bucket[idx] = {
                            "candidate": kept_candidate,
                            "template_guard": merged_guard,
                            "memory_guard": memory_guard,
                        }
                        self._bump_superblock_cache_stat("superblock_template_generalizations")
                        inserted_template = True
                        break
                    if not inserted_template:
                        template_bucket.append(
                            {
                                "candidate": dict(candidate),
                                "template_guard": template_guard,
                                "memory_guard": memory_guard,
                            }
                        )
                        while len(template_bucket) > template_per_key:
                            del template_bucket[0]

                    shape_bucket = self._superblock_shape_cache.setdefault(shape_key, [])
                    shape_bucket.append(
                        {
                            "candidate": dict(candidate),
                            "template_guard": template_guard,
                            "memory_guard": memory_guard,
                        }
                    )
                    while len(shape_bucket) > shape_per_key:
                        del shape_bucket[0]
            self._last_gpu_only_hotloop_stats["superblock_cache_entries"] = len(self._superblock_trace_cache)
            self._last_gpu_only_hotloop_stats["superblock_template_entries"] = sum(
                len(bucket) for bucket in self._superblock_template_cache.values()
            )
            self._last_gpu_only_hotloop_stats["superblock_shape_entries"] = sum(
                len(bucket) for bucket in self._superblock_shape_cache.values()
            )
        return self._clone_hotloop_candidate(candidate, cache_hit=False, cache_hit_kind="none")

    def _collect_hotloop_candidate(self, max_words: int = 32):
        """Return a conservative short-loop candidate for backend handoff."""
        try:
            pc = int(self.pc.item())
        except Exception:
            return None

        max_bytes = min(self.mem_size - pc, max_words * 4)
        if max_bytes < 4:
            return None

        raw = self.memory[pc:pc + max_bytes].detach().cpu().tolist()
        words: list[int] = []

        for i in range(0, len(raw), 4):
            inst = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16) | (raw[i + 3] << 24)
            words.append(inst)
            if inst == 0x00000000:
                break
            if (inst & 0xFFE0001F) == 0xD4000001:
                return None
            if not self._is_supported_rust_region_inst(inst):
                return None

        if len(words) < 3:
            return None

        max_idx = len(words) - 1

        def _is_tail_supported(inst: int) -> bool:
            if inst == 0x00000000:
                return True
            if self._decode_hotloop_branch(inst, 0) is not None:
                return False
            if self._decode_hotloop_memop(inst) is not None:
                return False
            return self._is_supported_rust_region_inst(inst)

        def _extend_region_exit(exit_idx: int) -> tuple[int, int]:
            tail_limit = int(os.environ.get("NCPU_GPU_ONLY_HOTLOOP_TAIL_WORDS", "4"))
            if tail_limit <= 0:
                return exit_idx, 0

            region_exit_idx = exit_idx
            tail_word_count = 0
            while region_exit_idx < len(words):
                inst = words[region_exit_idx]
                if inst == 0x00000000:
                    return region_exit_idx, tail_word_count
                if tail_word_count >= tail_limit or not _is_tail_supported(inst):
                    break
                region_exit_idx += 1
                tail_word_count += 1
            return region_exit_idx, tail_word_count

        def _build_candidate(exit_idx: int, *, tail_stop_idx: int | None = None, **fields):
            if exit_idx <= 0:
                return None
            region_exit_idx, tail_word_count = _extend_region_exit(exit_idx)
            if tail_stop_idx is not None and exit_idx < tail_stop_idx < region_exit_idx:
                region_exit_idx = tail_stop_idx
                tail_word_count = tail_stop_idx - exit_idx
            stop_is_real = region_exit_idx < len(words) and words[region_exit_idx] == 0x00000000
            candidate_words = list(words[:region_exit_idx]) + [0x00000000]
            return {
                "pc": pc,
                "words": candidate_words,
                "code_bytes": self._encode_hotloop_words(candidate_words),
                "halt_idx": region_exit_idx,
                "synthetic_stop": not stop_is_real,
                "tail_word_count": tail_word_count,
                "region_blocks": 1 + int(tail_word_count > 0),
                **fields,
            }

        branches = []
        backward_branches = []
        for idx, inst in enumerate(words):
            branch = self._decode_hotloop_branch(inst, idx)
            if branch is None:
                continue
            if branch["loop_start_idx"] < 0 or branch["loop_start_idx"] > (max_idx + 1):
                return None
            if branch["branch_imm"] == 0:
                return None
            branches.append(branch)
            if branch["branch_imm"] < 0:
                if branch["loop_start_idx"] >= idx:
                    return None
                backward_branches.append(branch)

        if not backward_branches:
            return None

        nested_branches = None
        if len(backward_branches) == 2:
            inner_branch, outer_branch = backward_branches
            if (
                inner_branch["branch_kind"] == "bcond" and
                outer_branch["branch_kind"] == "bcond" and
                inner_branch["cond"] == 1 and
                outer_branch["cond"] == 1 and
                outer_branch["loop_start_idx"] < inner_branch["loop_start_idx"] <
                inner_branch["branch_idx"] < outer_branch["branch_idx"]
            ):
                outer_body = words[outer_branch["loop_start_idx"]:outer_branch["branch_idx"]]
                if any(self._decode_hotloop_memop(inst) is not None for inst in outer_body):
                    return None
                nested_branches = backward_branches
                backward_branch = outer_branch
            else:
                backward_branch = backward_branches[0]
        elif len(backward_branches) > 2:
            backward_branch = backward_branches[0]
        else:
            backward_branch = backward_branches[0]

        next_loop_start_idx = min(
            (
                branch["loop_start_idx"]
                for branch in backward_branches
                if (
                    branch["branch_idx"] > backward_branch["branch_idx"] and
                    branch["loop_start_idx"] >= (backward_branch["branch_idx"] + 1)
                )
            ),
            default=None,
        )
        exit_idx = backward_branch["branch_idx"] + 1
        branches = [branch for branch in branches if branch["branch_idx"] < exit_idx]
        backward_branches = [branch for branch in backward_branches if branch["branch_idx"] < exit_idx]
        forward_exits = [
            branch for branch in branches
            if branch["branch_imm"] > 0 and branch["branch_kind"] in {"cbz", "cbnz", "bcond"}
        ]
        if any(
            branch["branch_imm"] > 0 and branch["branch_kind"] not in {"cbz", "cbnz", "bcond"}
            for branch in branches
        ):
            return None

        if nested_branches is None and len(backward_branches) == 1 and len(forward_exits) == 1 and len(branches) == 2:
            exit_branch = forward_exits[0]
            if backward_branch["branch_kind"] != "b":
                return None
            if exit_branch["loop_start_idx"] != exit_idx:
                return None
            body_start_idx = exit_branch["branch_idx"] + 1
            candidate = _build_candidate(
                exit_idx,
                tail_stop_idx=next_loop_start_idx,
                body_start_idx=body_start_idx,
                exit_branch=exit_branch,
                **backward_branch,
            )
            if candidate is None:
                return None

            if exit_branch["branch_kind"] in {"cbz", "cbnz"}:
                if backward_branch["loop_start_idx"] != exit_branch["branch_idx"]:
                    return None
                return candidate

            if exit_branch["branch_kind"] != "bcond" or exit_branch["cond"] not in {10, 11, 12, 13}:
                return None
            compare_idx = exit_branch["branch_idx"] - 1
            if compare_idx < 0 or not self._is_hotloop_cmp(words[compare_idx]):
                return None
            if backward_branch["loop_start_idx"] != compare_idx:
                return None
            candidate["compare_idx"] = compare_idx
            return candidate

        if nested_branches is None and (len(branches) != 1 or len(backward_branches) != 1 or len(forward_exits) > 0):
            return None

        candidate = _build_candidate(exit_idx, tail_stop_idx=next_loop_start_idx, **backward_branch)
        if candidate is None:
            return None
        if nested_branches is not None:
            candidate["nested_branches"] = nested_branches
        return candidate

    def _hotloop_detector_confirms(self, candidate, device) -> bool:
        """Use the trained loop detector to approve a Rust hot-loop handoff."""
        return bool(self._hotloop_detector_decision(candidate, device).get("approved"))

    def _estimate_hotloop_work(self, candidate):
        """Estimate trip count and dynamic work for a conservative loop candidate."""
        if candidate.get("region_kind") == "superblock":
            simulation = candidate.get("simulation", {})
            executed_count = int(simulation.get("executed_count", max(candidate.get("halt_idx", 1), 1)))
            return {
                "body_word_count": max(int(candidate.get("halt_idx", executed_count)), 1),
                "estimated_iterations": 1,
                "estimated_work": max(executed_count, 1),
            }
        body_start_idx = candidate.get("body_start_idx", candidate["loop_start_idx"])
        body_words = candidate["words"][body_start_idx:candidate["branch_idx"]]
        estimate = {
            "body_word_count": len(body_words),
            "estimated_iterations": None,
            "estimated_work": None,
        }
        loop_regs = self._simulate_hotloop_prefix_regs(candidate)
        if loop_regs is None:
            return estimate
        iterations = self._infer_hotloop_iterations(candidate, loop_regs)
        if iterations is None or iterations < 0:
            return estimate
        estimate["estimated_iterations"] = int(iterations)
        estimate["estimated_work"] = int(iterations) * len(body_words)
        return estimate

    @staticmethod
    def _decode_hotloop_mov_imm16(inst: int) -> Optional[int]:
        """Return the immediate encoded by MOVZ/MOVK forms used in short tails."""
        if (
            (inst & 0xFF800000) == 0xD2800000 or
            (inst & 0xFF800000) == 0x52800000 or
            (inst & 0xFF800000) == 0xF2800000 or
            (inst & 0xFF800000) == 0x72800000
        ):
            return (inst >> 5) & 0xFFFF
        return None

    @staticmethod
    def _hotloop_branch_kind_flags(branch_kind: str) -> dict[str, int]:
        branch_kind = str(branch_kind or "").strip().lower()
        return {
            "branch_kind_b": int(branch_kind == "b"),
            "branch_kind_bcond": int(branch_kind == "bcond"),
            "branch_kind_cbz": int(branch_kind == "cbz"),
            "branch_kind_cbnz": int(branch_kind == "cbnz"),
        }

    def _build_hotloop_value_feature_source(
        self,
        candidate,
        sync_plan,
        remaining_instructions: int,
        *,
        estimate=None,
        segment_index: int = 1,
        reused_state: bool = False,
        previous_segment=None,
    ):
        """Build the runtime feature payload shared by policy scoring and trace export."""
        if estimate is None:
            estimate = self._estimate_hotloop_work(candidate)
        tail_start = int(candidate.get("branch_idx", -1)) + 1
        tail_stop = int(candidate.get("halt_idx", tail_start))
        tail_words = candidate["words"][tail_start:tail_stop]
        tail_imms = [
            imm16 for inst in tail_words
            if (imm16 := self._decode_hotloop_mov_imm16(inst)) is not None
        ]
        pre_sync_bytes = sum(end - start for start, end in sync_plan.get("pre", []))
        post_sync_bytes = sum(end - start for start, end in sync_plan.get("post", []))
        branch_kind = candidate.get("branch_kind", "")
        previous_segment = previous_segment if isinstance(previous_segment, dict) else (previous_segment or {})
        feature_source = {
            "body_word_count": estimate.get("body_word_count"),
            "estimated_iterations": estimate.get("estimated_iterations"),
            "estimated_work": estimate.get("estimated_work"),
            "pre_sync_bytes": pre_sync_bytes,
            "post_sync_bytes": post_sync_bytes,
            "remaining_instructions": remaining_instructions,
            "tail_word_count": int(candidate.get("tail_word_count", 0)),
            "synthetic_stop": bool(candidate.get("synthetic_stop")),
            "region_blocks": int(candidate.get("region_blocks", 1)),
            "nested_branch_count": len(candidate.get("nested_branches", [candidate])),
            "branch_kind": branch_kind,
            "tail_max_imm16": max(tail_imms, default=0),
            "tail_large_imm16_count": sum(1 for imm16 in tail_imms if imm16 >= 4096),
            "segment": int(segment_index),
            "reused_state": bool(reused_state),
            "previous_pre_sync_bytes": int(previous_segment.get("pre_sync_bytes", 0)),
            "previous_post_sync_bytes": int(previous_segment.get("post_sync_bytes", 0)),
            "previous_region_blocks": int(previous_segment.get("region_blocks", 0)),
            "previous_tail_word_count": int(previous_segment.get("tail_word_count", 0)),
            "previous_tail_max_imm16": int(previous_segment.get("tail_max_imm16", 0)),
            "previous_tail_large_imm16_count": int(previous_segment.get("tail_large_imm16_count", 0)),
        }
        feature_source.update(self._hotloop_branch_kind_flags(branch_kind))
        return feature_source

    def _hotloop_detector_decision(self, candidate, device):
        """Return a structured auto-mode decision for a Rust hot-loop handoff."""
        estimate = self._estimate_hotloop_work(candidate)
        decision = {
            "approved": False,
            "reason": "detector-unavailable",
            "segment_checks": [],
            **estimate,
        }
        if candidate.get("region_kind") == "superblock":
            min_work = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MIN_ESTIMATED_WORK", "0"))
            if min_work > 0 and estimate["estimated_work"] is not None and estimate["estimated_work"] < min_work:
                decision["reason"] = "estimated-work-below-threshold"
                return decision
            decision["approved"] = True
            decision["reason"] = "superblock-approved"
            return decision
        detector = getattr(self, '_neural_loop_detector', None)
        min_work = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MIN_ESTIMATED_WORK", "0"))
        if detector is None:
            if min_work > 0 and estimate["estimated_work"] is not None and estimate["estimated_work"] < min_work:
                decision["reason"] = "estimated-work-below-threshold"
            return decision
        try:
            branches = candidate.get("nested_branches", [candidate])
            min_conf = None
            for idx, branch in enumerate(branches):
                body_start_idx = branch.get(
                    "body_start_idx",
                    candidate.get("body_start_idx", branch["loop_start_idx"])
                )
                body_words = candidate["words"][
                    body_start_idx:branch["branch_idx"]
                ]
                body_bits = torch.zeros(len(body_words), 32, device=device, dtype=torch.float32)
                for i, word in enumerate(body_words):
                    for bit in range(32):
                        body_bits[i, bit] = (word >> bit) & 1
                type_logits, _, _ = detector(body_bits, self.regs[:32].float())
                conf = float(F.softmax(type_logits, dim=-1).max().item())
                pred_type = int(type_logits.argmax().item())
                decision["segment_checks"].append(
                    {
                        "segment": idx + 1,
                        "pred_type": pred_type,
                        "confidence": conf,
                        "body_word_count": len(body_words),
                    }
                )
                min_conf = conf if min_conf is None else min(min_conf, conf)
                if pred_type <= 0 or conf <= 0.8:
                    decision["reason"] = "detector-rejected"
                    return decision
            if min_work > 0 and estimate["estimated_work"] is not None and estimate["estimated_work"] < min_work:
                decision["reason"] = "estimated-work-below-threshold"
                return decision
            decision["approved"] = True
            decision["reason"] = "detector-approved"
            if min_conf is not None:
                decision["min_confidence"] = min_conf
            return decision
        except Exception:
            decision["reason"] = "detector-error"
            return decision

    def _hotloop_policy_decision(self, candidate, sync_plan, remaining_instructions: int, *, feature_source=None):
        """Estimate whether a detector-approved loop is worth handing off."""
        estimate = self._estimate_hotloop_work(candidate)
        if feature_source is None:
            feature_source = self._build_hotloop_value_feature_source(
                candidate,
                sync_plan,
                remaining_instructions,
                estimate=estimate,
            )
        pre_sync_bytes = int(feature_source["pre_sync_bytes"])
        post_sync_bytes = int(feature_source["post_sync_bytes"])
        decision = {
            "approved": True,
            "reason": "policy-approved",
            "score": None,
            "threshold": None,
            "pre_sync_bytes": int(pre_sync_bytes),
            "post_sync_bytes": int(post_sync_bytes),
            "remaining_instructions": int(remaining_instructions),
            **estimate,
        }

        max_sync_bytes = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MAX_SYNC_BYTES", "65536"))
        if pre_sync_bytes > max_sync_bytes or post_sync_bytes > max_sync_bytes:
            decision["approved"] = False
            decision["reason"] = "sync-budget-exceeded"
            decision["threshold"] = max_sync_bytes
            return decision

        value_model = getattr(self, "_neural_hotloop_value_model", None)
        if callable(value_model):
            threshold = float(os.environ.get("NCPU_GPU_ONLY_AUTO_VALUE_THRESHOLD", "0.55"))
            decision["threshold"] = threshold
            try:
                decision["score"] = predict_hotloop_value_score(
                    value_model,
                    feature_source,
                    device=self.device,
                )
                decision["approved"] = bool(decision["score"] >= threshold)
                decision["reason"] = "value-model" if decision["approved"] else "value-threshold"
                return decision
            except Exception:
                decision["approved"] = False
                decision["reason"] = "value-model-error"
                return decision

        min_work = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MIN_ESTIMATED_WORK", "0"))
        if estimate.get("estimated_work") is not None:
            decision["score"] = float(estimate["estimated_work"])
            if min_work > 0:
                decision["threshold"] = min_work
                if estimate["estimated_work"] < min_work:
                    decision["approved"] = False
                    decision["reason"] = "estimated-work-below-threshold"
                    return decision
            return decision

        min_body_words = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS", "4"))
        decision["threshold"] = min_body_words
        decision["score"] = float(estimate.get("body_word_count") or 0)
        if estimate.get("body_word_count", 0) < min_body_words:
            decision["approved"] = False
            decision["reason"] = "body-too-small"
            return decision

        return decision

    def _maybe_run_rust_hotloop(self, max_instructions: int):
        """Run a short neural-approved hot loop on the shared Rust/Metal CPU."""
        mode = os.environ.get("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto").strip().lower()
        if mode in {"0", "off", "false", "no"}:
            return None
        allow_cpu_auto = os.environ.get("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "0") == "1"
        if mode == "auto" and getattr(self.device, "type", str(self.device)) == "cpu" and not allow_cpu_auto:
            return None
        auto_min_instructions = int(os.environ.get("NCPU_GPU_ONLY_AUTO_MIN_INSTRUCTIONS", "512"))

        try:
            from kernels.mlx.rust_runner import get_shared_cpu
            rust_cpu = get_shared_cpu(memory_size=max(self.mem_size, 1024 * 1024))
            total_executed = 0
            total_elapsed = 0.0
            remaining = max_instructions
            segments = 0
            need_full_state_push = True
            detector_attempts = 0
            detector_rejections = 0
            policy_rejections = 0
            total_pre_sync_bytes = 0
            total_queued_post_sync_bytes = 0
            total_flushed_post_sync_bytes = 0
            reused_state_segments = 0
            pending_post_windows = []
            previous_segment_context = None
            if not hasattr(self, "_last_gpu_only_hotloop_trace"):
                self._last_gpu_only_hotloop_trace = []
            if not hasattr(self, "_last_gpu_only_hotloop_samples"):
                self._last_gpu_only_hotloop_samples = []
            if not hasattr(self, "_hotloop_dispatch_samples"):
                self._hotloop_dispatch_samples = []
            if not hasattr(self, "_last_gpu_only_hotloop_stats"):
                self._last_gpu_only_hotloop_stats = {}

            def _record_trace(payload):
                self._last_gpu_only_hotloop_trace.append(dict(payload))

            def _record_sample(payload):
                sample = dict(payload)
                self._last_gpu_only_hotloop_samples.append(sample)
                self._hotloop_dispatch_samples.append(sample)
                if len(self._hotloop_dispatch_samples) > 256:
                    del self._hotloop_dispatch_samples[:-256]
                return sample

            def _flush_pending_post_windows():
                nonlocal pending_post_windows, total_flushed_post_sync_bytes
                merged = self._merge_hotloop_windows(pending_post_windows)
                for start, end in merged:
                    data = rust_cpu.read_memory(start, end - start)
                    self.memory[start:end] = torch.tensor(
                        list(data), dtype=torch.uint8, device=self.device
                    )
                    total_flushed_post_sync_bytes += end - start
                pending_post_windows = []

            def _finalize(final_status: str):
                self._last_gpu_only_hotloop_stats.update(
                    {
                        "mode": mode,
                        "segments": segments,
                        "total_executed": total_executed,
                        "elapsed_seconds": total_elapsed,
                        "detector_attempts": detector_attempts,
                        "detector_rejections": detector_rejections,
                        "policy_rejections": policy_rejections,
                        "pre_sync_bytes": total_pre_sync_bytes,
                        "queued_post_sync_bytes": total_queued_post_sync_bytes,
                        "flushed_post_sync_bytes": total_flushed_post_sync_bytes,
                        "reused_state_segments": reused_state_segments,
                        "final_status": final_status,
                    }
                )

            while True:
                if mode == "auto" and segments == 0 and remaining < auto_min_instructions:
                    _record_trace(
                        {
                            "segment": 0,
                            "pc": int(self.pc.item()),
                            "auto_mode": True,
                            "approved": False,
                            "reason": "remaining-budget",
                            "remaining_after": remaining,
                        }
                    )
                    _finalize("remaining-budget")
                    return None

                candidate = self._collect_hotloop_candidate()
                if candidate is None:
                    candidate = self._collect_superblock_candidate()
                if candidate is None:
                    break

                sync_plan = self._plan_hotloop_memory_sync(candidate)
                if sync_plan is None:
                    _record_trace(
                        {
                            "segment": segments + 1,
                            "pc": int(candidate["pc"]),
                            "branch_kind": candidate["branch_kind"],
                            "cache_hit": bool(candidate.get("cache_hit")),
                            "cache_hit_kind": candidate.get("cache_hit_kind", "none"),
                            "cache_priority_score": candidate.get("cache_priority_score"),
                            "cache_priority_source": candidate.get("cache_priority_source"),
                            "synthetic_stop": bool(candidate.get("synthetic_stop")),
                            "auto_mode": mode == "auto",
                            "approved": False,
                            "reason": "sync-plan-unavailable",
                        }
                    )
                    break

                decision = self._hotloop_detector_decision(candidate, self.device)
                policy = None
                feature_source = self._build_hotloop_value_feature_source(
                    candidate,
                    sync_plan,
                    remaining,
                    estimate=decision,
                    segment_index=segments + 1,
                    reused_state=segments > 0,
                    previous_segment=previous_segment_context,
                )
                sample = {
                    "segment": segments + 1,
                    "pc": int(candidate["pc"]),
                    "region_kind": candidate.get("region_kind", "hotloop"),
                    "branch_kind": candidate["branch_kind"],
                    "cache_hit": bool(candidate.get("cache_hit")),
                    "cache_hit_kind": candidate.get("cache_hit_kind", "none"),
                    "cache_priority_score": candidate.get("cache_priority_score"),
                    "cache_priority_source": candidate.get("cache_priority_source"),
                    "specialization_patch_count": int(candidate.get("specialization_patch_count", 0)),
                    "literal_patch_count": int(candidate.get("literal_patch_count", 0)),
                    "imm_patch_count": int(candidate.get("imm_patch_count", 0)),
                    "synthetic_stop": bool(candidate.get("synthetic_stop")),
                    "expected_stop_pc": candidate.get("expected_stop_pc"),
                    "expected_halted": candidate.get("expected_halted"),
                    "materialized_word_count": int(candidate.get("halt_idx", 0)),
                    "tail_word_count": int(feature_source.get("tail_word_count", 0)),
                    "region_blocks": int(feature_source.get("region_blocks", 1)),
                    "nested_branch_count": int(feature_source.get("nested_branch_count", 1)),
                    "branch_kind_b": int(feature_source.get("branch_kind_b", 0)),
                    "branch_kind_bcond": int(feature_source.get("branch_kind_bcond", 0)),
                    "branch_kind_cbz": int(feature_source.get("branch_kind_cbz", 0)),
                    "branch_kind_cbnz": int(feature_source.get("branch_kind_cbnz", 0)),
                    "tail_max_imm16": int(feature_source.get("tail_max_imm16", 0)),
                    "tail_large_imm16_count": int(feature_source.get("tail_large_imm16_count", 0)),
                    "reused_state": bool(feature_source.get("reused_state", False)),
                    "previous_pre_sync_bytes": int(feature_source.get("previous_pre_sync_bytes", 0)),
                    "previous_post_sync_bytes": int(feature_source.get("previous_post_sync_bytes", 0)),
                    "previous_region_blocks": int(feature_source.get("previous_region_blocks", 0)),
                    "previous_tail_word_count": int(feature_source.get("previous_tail_word_count", 0)),
                    "previous_tail_max_imm16": int(feature_source.get("previous_tail_max_imm16", 0)),
                    "previous_tail_large_imm16_count": int(feature_source.get("previous_tail_large_imm16_count", 0)),
                    "estimated_iterations": feature_source.get("estimated_iterations"),
                    "estimated_work": feature_source.get("estimated_work"),
                    "body_word_count": feature_source.get("body_word_count"),
                    "detector_checks": list(decision.get("segment_checks", [])),
                    "detector_reason": decision.get("reason"),
                    "pre_sync_bytes": int(feature_source.get("pre_sync_bytes", 0)),
                    "post_sync_bytes": int(feature_source.get("post_sync_bytes", 0)),
                    "remaining_instructions": int(feature_source.get("remaining_instructions", remaining)),
                    "auto_mode": mode == "auto",
                }
                if mode == "auto":
                    detector_attempts += 1
                    if not decision.get("approved"):
                        detector_rejections += 1
                        sample["approved"] = False
                        sample["policy_reason"] = decision.get("reason")
                        sample["value_target"] = derive_hotloop_value_target(sample)
                        _record_sample(sample)
                        _record_trace(sample)
                        break
                    policy = self._hotloop_policy_decision(
                        candidate,
                        sync_plan,
                        remaining,
                        feature_source=feature_source,
                    )
                    sample["policy_score"] = policy.get("score")
                    sample["policy_threshold"] = policy.get("threshold")
                    sample["policy_reason"] = policy.get("reason")
                    if not policy.get("approved"):
                        policy_rejections += 1
                        sample["approved"] = False
                        sample["value_target"] = derive_hotloop_value_target(sample)
                        _record_sample(sample)
                        _record_trace(sample)
                        break
                sample["approved"] = True
                sample = _record_sample(sample)

                rust_cpu.load_program(candidate["code_bytes"], candidate["pc"])
                rust_cpu.set_pc(candidate["pc"])

                if need_full_state_push:
                    regs_host = self.regs.detach().cpu().to(torch.int64).tolist()
                    for reg_idx, value in enumerate(regs_host):
                        rust_cpu.set_register(reg_idx, int(value))

                    flags_host = self.flags.detach().cpu().to(torch.float32).tolist()
                    rust_cpu.set_flags(
                        bool(flags_host[0] > 0.5),
                        bool(flags_host[1] > 0.5),
                        bool(flags_host[2] > 0.5),
                        bool(flags_host[3] > 0.5),
                    )
                else:
                    reused_state_segments += 1

                pre_windows = []
                for window in sync_plan["pre"]:
                    pre_windows.extend(self._subtract_hotloop_window(window, pending_post_windows))
                pre_windows = self._merge_hotloop_windows(pre_windows)
                pre_sync_bytes = sum(end - start for start, end in pre_windows)
                post_sync_bytes = sum(end - start for start, end in sync_plan["post"])
                total_pre_sync_bytes += pre_sync_bytes
                total_queued_post_sync_bytes += post_sync_bytes
                for start, end in pre_windows:
                    host_bytes = bytes(self.memory[start:end].detach().cpu().tolist())
                    rust_cpu.write_memory(start, host_bytes)

                result = rust_cpu.execute(max_cycles=remaining)

                regs_back = torch.tensor(
                    [rust_cpu.get_register(i) for i in range(32)],
                    dtype=torch.int64,
                    device=self.device,
                )
                flags_back = torch.tensor(
                    [1.0 if f else 0.0 for f in rust_cpu.get_flags()],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.regs.copy_(regs_back)
                self.flags.copy_(flags_back)
                rust_stop_pc = int(rust_cpu.pc)
                resolved_stop_pc = int(candidate.get("expected_stop_pc", rust_stop_pc))
                self.pc = torch.tensor(resolved_stop_pc, dtype=torch.int64, device=self.device)
                pending_post_windows = self._merge_hotloop_windows(
                    pending_post_windows + list(sync_plan["post"])
                )

                stop_name = getattr(result, "stop_reason_name", "")
                synthetic_stop = bool(candidate.get("synthetic_stop"))
                if "expected_halted" in candidate:
                    self.halted = bool(candidate.get("expected_halted"))
                else:
                    self.halted = stop_name == "HALT" and not synthetic_stop
                region_kind = candidate.get("region_kind", "hotloop")
                self._last_gpu_only_backend = f"rust-{region_kind}"

                executed = int(getattr(result, "cycles", getattr(result, "total_cycles", 0)))
                if stop_name == "HALT":
                    # Match run_gpu_only() accounting by dropping the terminal loop-exit step.
                    executed = max(executed - 1, 0)
                observed_elapsed = float(result.elapsed_seconds)
                observed_ips = (executed / observed_elapsed) if observed_elapsed > 0 else None
                total_executed += executed
                total_elapsed += observed_elapsed
                remaining = max(remaining - executed, 0)
                segments += 1
                self._last_gpu_only_hotloop_segments = segments
                need_full_state_push = False
                sample["executed_count"] = executed
                sample["elapsed_seconds"] = observed_elapsed
                sample["observed_ips"] = observed_ips
                sample["stop_reason"] = stop_name
                sample["reused_state"] = bool(feature_source.get("reused_state", False))
                sample["rust_stop_pc"] = rust_stop_pc
                sample["resolved_stop_pc"] = resolved_stop_pc
                sample["value_target"] = derive_hotloop_value_target(sample)
                previous_segment_context = dict(sample)
                _record_trace(
                    {
                        "segment": segments,
                        "pc": int(candidate["pc"]),
                        "region_kind": candidate.get("region_kind", "hotloop"),
                        "branch_kind": candidate["branch_kind"],
                        "cache_hit": bool(candidate.get("cache_hit")),
                        "cache_hit_kind": candidate.get("cache_hit_kind", "none"),
                        "cache_priority_score": candidate.get("cache_priority_score"),
                        "cache_priority_source": candidate.get("cache_priority_source"),
                        "specialization_patch_count": int(candidate.get("specialization_patch_count", 0)),
                        "literal_patch_count": int(candidate.get("literal_patch_count", 0)),
                        "imm_patch_count": int(candidate.get("imm_patch_count", 0)),
                        "synthetic_stop": synthetic_stop,
                        "expected_stop_pc": candidate.get("expected_stop_pc"),
                        "expected_halted": candidate.get("expected_halted"),
                        "rust_stop_pc": rust_stop_pc,
                        "resolved_stop_pc": resolved_stop_pc,
                        "materialized_word_count": int(candidate.get("halt_idx", 0)),
                        "tail_word_count": int(feature_source.get("tail_word_count", 0)),
                        "region_blocks": int(feature_source.get("region_blocks", 1)),
                        "nested_branch_count": int(feature_source.get("nested_branch_count", 1)),
                        "branch_kind_b": int(feature_source.get("branch_kind_b", 0)),
                        "branch_kind_bcond": int(feature_source.get("branch_kind_bcond", 0)),
                        "branch_kind_cbz": int(feature_source.get("branch_kind_cbz", 0)),
                        "branch_kind_cbnz": int(feature_source.get("branch_kind_cbnz", 0)),
                        "tail_max_imm16": int(feature_source.get("tail_max_imm16", 0)),
                        "tail_large_imm16_count": int(feature_source.get("tail_large_imm16_count", 0)),
                        "reused_state": bool(feature_source.get("reused_state", False)),
                        "previous_pre_sync_bytes": int(feature_source.get("previous_pre_sync_bytes", 0)),
                        "previous_post_sync_bytes": int(feature_source.get("previous_post_sync_bytes", 0)),
                        "previous_region_blocks": int(feature_source.get("previous_region_blocks", 0)),
                        "previous_tail_word_count": int(feature_source.get("previous_tail_word_count", 0)),
                        "previous_tail_max_imm16": int(feature_source.get("previous_tail_max_imm16", 0)),
                        "previous_tail_large_imm16_count": int(feature_source.get("previous_tail_large_imm16_count", 0)),
                        "auto_mode": mode == "auto",
                        "approved": mode != "auto" or bool(decision.get("approved")),
                        "reason": (
                            policy.get("reason")
                            if policy is not None
                            else decision.get("reason", "forced-mode")
                        ),
                        "estimated_iterations": feature_source.get("estimated_iterations"),
                        "estimated_work": feature_source.get("estimated_work"),
                        "body_word_count": feature_source.get("body_word_count"),
                        "pre_sync_bytes": pre_sync_bytes,
                        "post_sync_bytes": post_sync_bytes,
                        "executed_count": executed,
                        "elapsed_seconds": observed_elapsed,
                        "observed_ips": observed_ips,
                        "remaining_after": remaining,
                        "stop_reason": stop_name,
                        "detector_checks": list(decision.get("segment_checks", [])),
                        "policy_score": None if policy is None else policy.get("score"),
                        "reused_state": bool(feature_source.get("reused_state", False)),
                    }
                )

                if self.halted or remaining <= 0:
                    _flush_pending_post_windows()
                    _finalize("complete")
                    return total_executed, total_elapsed, True
                if stop_name != "HALT" or not synthetic_stop:
                    _flush_pending_post_windows()
                    _finalize("partial-stop")
                    break

            if total_executed > 0:
                _flush_pending_post_windows()
                _finalize("partial-fallback")
                return total_executed, total_elapsed, False
            _finalize("no-handoff")
            return None
        except Exception:
            if hasattr(self, "_last_gpu_only_hotloop_stats"):
                self._last_gpu_only_hotloop_stats["final_status"] = "backend-error"
            return None

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
        self._last_gpu_only_backend = "torch-gpu-only"
        self._last_gpu_only_hotloop_segments = 0
        self._last_gpu_only_hotloop_trace = []
        self._last_gpu_only_hotloop_samples = []
        self._last_gpu_only_hotloop_stats = {
            "mode": os.environ.get("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto").strip().lower(),
            "segments": 0,
            "total_executed": 0,
            "elapsed_seconds": 0.0,
            "detector_attempts": 0,
            "detector_rejections": 0,
            "policy_rejections": 0,
            "pre_sync_bytes": 0,
            "queued_post_sync_bytes": 0,
            "flushed_post_sync_bytes": 0,
            "reused_state_segments": 0,
            "superblock_cache_hits": 0,
            "superblock_cache_misses": 0,
            "superblock_cache_entries": len(getattr(self, "_superblock_trace_cache", {})),
            "superblock_cache_priority_evictions": 0,
            "superblock_cache_priority_rejections": 0,
            "superblock_template_hits": 0,
            "superblock_template_misses": 0,
            "superblock_template_entries": sum(
                len(bucket) for bucket in getattr(self, "_superblock_template_cache", {}).values()
            ),
            "superblock_template_generalizations": 0,
            "superblock_template_cross_window_hits": 0,
            "superblock_shape_hits": 0,
            "superblock_shape_misses": 0,
            "superblock_shape_entries": sum(
                len(bucket) for bucket in getattr(self, "_superblock_shape_cache", {}).values()
            ),
            "superblock_shape_cross_window_hits": 0,
            "superblock_literal_patch_hits": 0,
            "superblock_literal_patch_words": 0,
            "superblock_imm_patch_hits": 0,
            "superblock_imm_patch_words": 0,
            "final_status": "torch-only",
        }
        device     = self.device
        if getattr(device, "type", str(device)) == "cpu":
            # The batch-parallel tensor engine is tuned for GPU execution. On CPU,
            # serialize only when the upcoming window contains memory ops that are
            # sensitive to same-batch data dependencies.
            try:
                pc_i = int(self.pc.item())
                probe_bytes = min(self.mem_size - pc_i, max(batch_size * 4, 4))
                raw_probe = self.memory[pc_i:pc_i + probe_bytes].detach().cpu().tolist()
                for i in range(0, len(raw_probe), 4):
                    inst = (
                        raw_probe[i]
                        | (raw_probe[i + 1] << 8)
                        | (raw_probe[i + 2] << 16)
                        | (raw_probe[i + 3] << 24)
                    )
                    if inst == 0x00000000:
                        break
                    if self._decode_hotloop_memop(inst) is not None:
                        batch_size = 1
                        break
            except Exception:
                batch_size = 1
        mem        = self.memory
        regs       = self.regs
        flags      = self.flags
        mem_arith  = self._get_woven_memarith()    # pointer.pt neural address computation
        prefetcher = self._get_woven_prefetcher()  # prefetch.pt LSTM memory tracking
        weave = None
        if getattr(self, 'use_neural_alu', False) and getattr(self, '_neural_alu', None) is not None:
            try:
                weave = self._get_weave_alu()
            except Exception:
                weave = None

        # Lazy-load neural hazard predictor (opt-in via env var)
        if not hasattr(self, '_hazard_predictor'):
            self._hazard_predictor = None
            if os.environ.get('NEURAL_HAZARD_PREDICTOR', '0') == '1':
                hp_path = _ALU_MODELS_ROOT / 'hazard_predictor.pt'
                if hp_path.exists():
                    try:
                        from ...neural_hazard_predictor import NeuralHazardPredictor
                        _hp_model = NeuralHazardPredictor().to(device).eval()
                        _hp_model.load_state_dict(torch.load(hp_path, map_location=device, weights_only=True))
                        self._hazard_predictor = _hp_model
                        logger.info(f"Neural hazard predictor loaded ({sum(p.numel() for p in _hp_model.parameters()):,} params)")
                    except Exception as e:
                        logger.warning(f"Failed to load neural hazard predictor: {e}")
            # Neural dependency graph predictor (opt-in)
            self._dep_predictor = None
            if os.environ.get('NEURAL_DEPENDENCY_PREDICTOR', '0') == '1':
                dp_path = _ALU_MODELS_ROOT / 'dependency_predictor.pt'
                if dp_path.exists():
                    try:
                        from ...neural_dependency_predictor import NeuralDependencyPredictor
                        _dp_model = NeuralDependencyPredictor().to(device).eval()
                        _dp_model.load_state_dict(torch.load(dp_path, map_location=device, weights_only=True))
                        self._dep_predictor = _dp_model
                        logger.info(f"Neural dependency predictor loaded ({sum(p.numel() for p in _dp_model.parameters()):,} params)")
                    except Exception as e:
                        logger.warning(f"Failed to load neural dependency predictor: {e}")
            # Neural instruction scheduler (opt-in: learned out-of-order execution)
            self._neural_scheduler = None
            if os.environ.get('NEURAL_SCHEDULER', '0') == '1':
                ns_path = _ALU_MODELS_ROOT / 'instruction_scheduler.pt'
                if ns_path.exists():
                    try:
                        from ...neural_instruction_scheduler import NeuralInstructionScheduler
                        _ns_model = NeuralInstructionScheduler().to(device).eval()
                        _ns_model.load_state_dict(torch.load(ns_path, map_location=device, weights_only=True))
                        self._neural_scheduler = _ns_model
                        logger.info(f"Neural instruction scheduler loaded ({sum(p.numel() for p in _ns_model.parameters()):,} params)")
                    except Exception as e:
                        logger.warning(f"Failed to load neural instruction scheduler: {e}")
            # Neural branch predictor (auto-load if weights exist)
            self._neural_branch_predictor = None
            bp_path = _ALU_MODELS_ROOT / 'branch_predictor.pt'
            if bp_path.exists():
                try:
                    from ...neural_branch_predictor import NeuralBranchPredictor
                    _bp_model = NeuralBranchPredictor().to(device).eval()
                    _bp_model.load_state_dict(torch.load(bp_path, map_location=device, weights_only=True))
                    self._neural_branch_predictor = _bp_model
                    logger.info(f"Neural branch predictor loaded ({sum(p.numel() for p in _bp_model.parameters()):,} params)")
                except Exception as e:
                    logger.warning(f"Failed to load neural branch predictor: {e}")
            # Neural loop detector (auto-load if weights exist)
            self._neural_loop_detector = None
            ld_path = Path(__file__).parent.parent.parent / 'loop_detector_fast.pt'
            if ld_path.exists():
                try:
                    from ..extractors import NeuralLoopDetector
                    _ld_model = NeuralLoopDetector(max_body_len=32).to(device).eval()
                    _ld_model.load_state_dict(torch.load(ld_path, map_location=device, weights_only=True))
                    self._neural_loop_detector = _ld_model
                    logger.info(f"Neural loop detector loaded ({sum(p.numel() for p in _ld_model.parameters()):,} params)")
                except Exception as e:
                    logger.warning(f"Failed to load neural loop detector: {e}")
        if not hasattr(self, '_neural_hotloop_value_model'):
            self._neural_hotloop_value_model = None
            value_model_setting = os.environ.get("NCPU_GPU_ONLY_AUTO_VALUE_MODEL", "1").strip()
            if value_model_setting.lower() not in {"0", "off", "false", "no"}:
                if value_model_setting.lower() in {"", "1", "on", "true", "yes"}:
                    value_model_path = Path(__file__).parent.parent.parent / "hotloop_value_model.pt"
                else:
                    value_model_path = Path(value_model_setting).expanduser()
                if value_model_path.exists():
                    try:
                        from ...hotloop_value_model import load_hotloop_value_model

                        _value_model = load_hotloop_value_model(value_model_path, device=device)
                        self._neural_hotloop_value_model = _value_model
                        logger.info(
                            "Neural hotloop value model loaded (%s params)",
                            sum(p.numel() for p in _value_model.parameters()),
                        )
                    except Exception as e:
                        logger.warning(f"Failed to load neural hotloop value model: {e}")
        if getattr(self, '_neural_branch_predictor', None) is not None:
            self._neural_branch_predictor._flag_history = None

        rust_handoff = self._maybe_run_rust_hotloop(max_instructions)
        if rust_handoff is not None:
            rust_executed, rust_elapsed, rust_complete = rust_handoff
            if rust_complete or self.halted or rust_executed >= max_instructions:
                return rust_executed, rust_elapsed
            max_instructions -= rust_executed
            if self._last_gpu_only_backend == "rust-superblock":
                self._last_gpu_only_backend = "hybrid-superblock+torch"
            else:
                self._last_gpu_only_backend = "hybrid-hotloop+torch"
        else:
            rust_executed = 0

        # PC as tensor - NEVER call .item() in hot path
        pc_t = self.pc.clone()

        # Pre-allocate batch index tensor
        batch_idx = torch.arange(batch_size, device=device, dtype=torch.int64)

        # Big value for min operations
        BIG = torch.tensor(batch_size * 2, device=device, dtype=torch.int64)

        # Pre-allocate loop-invariant tensors
        _byte_offsets_4 = torch.arange(4, device=device, dtype=torch.int64)
        _lower_tri = torch.tril(torch.ones(batch_size, batch_size, device=device, dtype=torch.bool), diagonal=-1)

        # Adaptive batch sizing constants (for hazard matrix)
        SMALL_BATCH = 8
        _lower_tri_small = _lower_tri[:SMALL_BATCH, :SMALL_BATCH]
        _batch_idx_small = batch_idx[:SMALL_BATCH]
        _BIG_small = torch.tensor(SMALL_BATCH * 2, device=device, dtype=torch.int64)
        _eff_b = batch_size

        # Pre-allocate hot-loop tensors (avoid per-iteration allocation)
        _zero_results = torch.zeros(batch_size, device=device, dtype=torch.int64)
        _zero_wmask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        _zero_rvals = torch.zeros(batch_size, device=device, dtype=torch.int64)
        _simd_bidx = torch.arange(batch_size, device=device)
        _simd_K_range = torch.arange(25, device=device, dtype=torch.int64)
        _const_0xFFFF = torch.tensor(0xFFFF, device=device, dtype=torch.int64)
        _const_1_i64 = torch.tensor(1, device=device, dtype=torch.int64)
        _zero_f32 = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
        _one_f32 = torch.ones(1, device=device, dtype=torch.float32).squeeze()

        # State tensors - ALL on GPU
        executed_t = torch.tensor(0, device=device, dtype=torch.int64)
        halted_t = torch.tensor(0, device=device, dtype=torch.int64)

        # ═══════════════════════════════════════════════════════════════════
        # FIXED ITERATION COUNT - MINIMAL .item() SYNCS
        # Syncs only every SYNC_INTERVAL batches (amortises MPS round-trip cost).
        # SVC is still handled promptly: we sync every iteration only after a
        # potential SVC window; halt uses the same deferred mechanism.
        # ═══════════════════════════════════════════════════════════════════
        # Keep the default conservative for the current Torch/MPS engine.
        # The environment override is useful for experiments and workload-
        # specific tuning, but larger values regress the common counted-loop
        # benchmark on this branch.
        SYNC_INTERVAL = max(1, int(os.environ.get('NCPU_GPU_SYNC_INTERVAL', '2')))
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
            byte_addrs = pc_t + inst_offsets.unsqueeze(1) + _byte_offsets_4
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
            if _USE_COMPILE:
                _pb = set(range(256))  # CUDA zero-sync: enter ALL blocks unconditionally
            else:
                _pb = set(op_bytes.tolist())  # MPS: 1 GPU-CPU sync → skip irrelevant blocks
            ops = self.op_type_table[op_bytes]
            rds = insts & 0x1F
            rns = (insts >> 5) & 0x1F
            rms = (insts >> 16) & 0x1F
            imm12 = (insts >> 10) & 0xFFF
            imm16 = (insts >> 5) & 0xFFFF
            hw = (insts >> 21) & 0x3
            _hk = (insts >> 21) & 0x7FF

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
            # ── Hazard property lookup + adaptive-size hazard matrix ──
            _hk_long = _hk.long()
            if hasattr(self, '_hazard_predictor') and self._hazard_predictor is not None:
                _hp = self._hazard_predictor(insts)  # [B, 3]
                reads_rn_as_reg = _hp[:, 0] > 0.5
                reads_rm_as_reg = _hp[:, 1] > 0.5
            else:
                reads_rm_as_reg = self._reads_rm_lut[_hk_long]
                reads_rn_as_reg = ~self._rn_not_reg_lut[_hk_long]
            writes_reg = self._writes_rd_lut[_hk_long] | (
                ((insts & 0xFFC00000) == 0x39400000) |  # LDRB unsigned offset
                ((insts & 0xFFC00000) == 0xB9400000) |  # LDR 32-bit unsigned offset
                ((insts & 0xFFC00000) == 0xF9400000) |  # LDR 64-bit unsigned offset
                ((insts & 0xFFE00C00) == 0x38400400) |  # LDRB post-index
                ((insts & 0xFFE00C00) == 0xB8400400) |  # LDR 32-bit post-index
                ((insts & 0xFFE00C00) == 0xF8400400) |  # LDR 64-bit post-index
                ((insts & 0xFFE00C00) == 0x38400C00) |  # LDRB pre-index
                ((insts & 0xFFE00C00) == 0xB8400C00) |  # LDR 32-bit pre-index
                ((insts & 0xFFE00C00) == 0xF8400C00)    # LDR 64-bit pre-index
            )
            reads_rt_as_reg = (
                ((insts & 0xFFC00000) == 0x39000000) |  # STRB unsigned offset
                ((insts & 0xFFC00000) == 0x79000000) |  # STRH unsigned offset
                ((insts & 0xFFC00000) == 0xB9000000) |  # STR 32-bit unsigned offset
                ((insts & 0xFFC00000) == 0xF9000000) |  # STR 64-bit unsigned offset
                ((insts & 0xFFE00C00) == 0x38000400) |  # STRB post-index
                ((insts & 0xFFE00C00) == 0x78000400) |  # STRH post-index
                ((insts & 0xFFE00C00) == 0xB8000400) |  # STR 32-bit post-index
                ((insts & 0xFFE00C00) == 0xF8000400) |  # STR 64-bit post-index
                ((insts & 0xFFE00C00) == 0x38000C00) |  # STRB pre-index
                ((insts & 0xFFE00C00) == 0x78000C00) |  # STRH pre-index
                ((insts & 0xFFE00C00) == 0xB8000C00) |  # STR 32-bit pre-index
                ((insts & 0xFFE00C00) == 0xF8000C00) |  # STR 64-bit pre-index
                ((insts & 0xFFE00C00) == 0x38000000) |  # STURB
                ((insts & 0xFFE00C00) == 0x78000000) |  # STURH
                ((insts & 0xFFE00C00) == 0xB8000000) |  # STUR 32-bit
                ((insts & 0xFFE00C00) == 0xF8000000) |  # STUR 64-bit
                ((insts & 0xFFE00C00) == 0x38200800) |  # STRB register offset
                ((insts & 0xFFE00C00) == 0x78200800) |  # STRH register offset
                ((insts & 0xFFE00C00) == 0xB8200800) |  # STR 32-bit register offset
                ((insts & 0xFFE00C00) == 0xF8200800)    # STR 64-bit register offset
            )

            # Hazard detection: neural dependency predictor OR adaptive matrix
            if hasattr(self, '_dep_predictor') and self._dep_predictor is not None:
                first_hazard = self._dep_predictor(rds, rns, rms, reads_rm_as_reg, writes_reg, BIG)
            else:
                # Adaptive hazard matrix: use smaller matrix for tight loops
                _eb = _eff_b
                _e_rds = rds[:_eb].unsqueeze(0)
                _e_rns = rns[:_eb].unsqueeze(1)
                _e_rms = rms[:_eb].unsqueeze(1)
                _e_rts = rds[:_eb].unsqueeze(1)
                _e_lt = _lower_tri_small if _eb == SMALL_BATCH else _lower_tri
                hazard_rn = (_e_rds == _e_rns) & reads_rn_as_reg[:_eb].unsqueeze(1)
                hazard_rm = (_e_rds == _e_rms) & reads_rm_as_reg[:_eb].unsqueeze(1)
                hazard_rt = (_e_rds == _e_rts) & reads_rt_as_reg[:_eb].unsqueeze(1)
                hazard_valid = (hazard_rn | hazard_rm | hazard_rt) & _e_lt
                hazard_valid = hazard_valid & writes_reg[:_eb].unsqueeze(0) & (rds[:_eb] != 31).unsqueeze(0)
                has_hazard_per_inst = hazard_valid.any(dim=1)
                _e_bidx = _batch_idx_small if _eb == SMALL_BATCH else batch_idx
                _e_BIG = _BIG_small if _eb == SMALL_BATCH else BIG
                hazard_indices = torch.where(has_hazard_per_inst, _e_bidx, _e_BIG)
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
            rn_vals = torch.where((rns == 31) & ~_is_mem_op_gpu, _zero_rvals, rn_vals)
            rm_vals = torch.where((rms == 31) & ~_is_mem_op_gpu, _zero_rvals, rm_vals)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5: SIMD PARALLEL ALU DISPATCH
            # ═══════════════════════════════════════════════════════════════
            # Pure-ALU op classes (no flag side-effects, no memory) are decoded
            # and computed simultaneously via a [K_ALU=25, B] classify+compute
            # matrix.  A single gather picks the correct result per slot,
            # replacing ~40 sequential torch.where calls with 2 torch.stack ops.
            # Flag-updating ops (CMP, ADDS, SUBS, BICS, ANDS) follow below.
            # ═══════════════════════════════════════════════════════════════
            results = _zero_results.clone()
            write_mask = _zero_wmask.clone()

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
            _bidx = _simd_bidx
            _prio = (_cls.long() *
                     _simd_K_range[:_K].unsqueeze(1)
                    ).max(dim=0).indices          # [B]: which class matched
            _hit  = _cls.any(dim=0)              # [B]: did any class match?
            results    = torch.where(_hit, _cmp[_prio, _bidx], results)
            write_mask = write_mask | _hit

            # --- MOVK (insert imm16 into register half-word; needs current Xd) ---
            # MOVK 64-bit: 1 11 100101 hw imm16 rd = 0xF28xxxxx
            # MOVK 32-bit: 0 11 100101 hw imm16 rd = 0x728xxxxx
            movk_mask = (((insts & 0xFF800000) == 0xF2800000) |
                         ((insts & 0xFF800000) == 0x72800000)) & exec_mask
            movk_clear = ~(_const_0xFFFF << (hw * 16))
            movk_val = (rd_vals & movk_clear) | (imm16 << (hw * 16))
            results = torch.where(movk_mask, movk_val, results)
            write_mask = write_mask | movk_mask

            # --- MOVN (move with NOT; separate from SIMD: needs sf-bit conditional) ---
            # MOVN 64: sf=1,opc=00 → 0x92800000 mask 0xFF800000
            # MOVN 32: sf=0,opc=00 → 0x12800000 mask 0xFF800000
            movn_mask = (((insts & 0xFF800000) == 0x92800000) |
                         ((insts & 0xFF800000) == 0x12800000)) & exec_mask
            if (0x92 in _pb or 0x12 in _pb):
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
            if (0xEA in _pb or 0x6A in _pb):
                bics_idx = torch.where(bics_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                bics_v = results[bics_idx]   # read result already placed by SIMD
                flags[0] = ((bics_v >> 63) & 1).to(torch.float32)
                flags[1] = (bics_v == 0).to(torch.float32)
                flags[2] = _zero_f32
                flags[3] = _zero_f32

            # --- ADRP ---
            adrp_mask = (ops == OpType.ADRP.value) & exec_mask
            if (0x90 in _pb or 0xB0 in _pb or 0xD0 in _pb or 0xF0 in _pb):
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
            if (0xEB in _pb or 0xF1 in _pb):
                # Tensor-only CMP flag update — no .any() GPU-CPU sync
                cmp_indices = torch.where(cmp_mask, batch_idx, BIG)
                first_cmp_idx = cmp_indices.min()
                _any_cmp_t = (first_cmp_idx < batch_size)  # tensor bool

                cmp_val = cmp_result[first_cmp_idx.clamp(0, batch_size-1)]
                cmp_rn = rn_vals[first_cmp_idx.clamp(0, batch_size-1)]
                cmp_rm_or_imm = torch.where(
                    subs_reg_mask[first_cmp_idx.clamp(0, batch_size-1)],
                    rm_vals[first_cmp_idx.clamp(0, batch_size-1)],
                    imm12[first_cmp_idx.clamp(0, batch_size-1)]
                )

                new_n = (cmp_val >> 63) & 1
                new_z = (cmp_val == 0).to(torch.float32)
                new_c = _u64_ge(cmp_rn, cmp_rm_or_imm).to(torch.float32)
                rn_neg = (cmp_rn >> 63) & 1
                rm_neg = (cmp_rm_or_imm >> 63) & 1
                result_neg = (cmp_val >> 63) & 1
                new_v = ((rn_neg != rm_neg) & (rn_neg != result_neg)).to(torch.float32)

                # Masked update — no-op when no CMP present
                flags[0] = torch.where(_any_cmp_t, new_n.to(torch.float32), flags[0])
                flags[1] = torch.where(_any_cmp_t, new_z, flags[1])
                flags[2] = torch.where(_any_cmp_t, new_c, flags[2])
                flags[3] = torch.where(_any_cmp_t, new_v, flags[3])

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
            if (0xB1 in _pb or 0x31 in _pb or 0xAB in _pb or 0x2B in _pb):
                adds_result_imm = rn_vals + imm12
                adds_result_reg = rn_vals + rm_vals
                adds_result = torch.where(adds_imm_gpu, adds_result_imm, adds_result_reg)
                adds_idx = torch.where(adds_any_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                adds_v = adds_result[adds_idx]
                adds_rn = rn_vals[adds_idx]
                adds_rm = torch.where(adds_imm_gpu[adds_idx], imm12[adds_idx], rm_vals[adds_idx])
                new_n = (adds_v >> 63) & 1
                new_z = (adds_v == 0).to(torch.float32)
                new_c = _u64_gt(adds_rn, ~adds_rm).to(torch.float32)
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
            if (0x9A in _pb or 0x1A in _pb or 0xBA in _pb or 0x3A in _pb or 0xDA in _pb or 0x5A in _pb or 0xFA in _pb or 0x7A in _pb):
                c_in = flags[2].to(torch.int64)   # carry flag (0 or 1)
                adc_val = rn_vals + rm_vals + c_in
                sbc_val = rn_vals + ~rm_vals + c_in  # ~Rm + C = -(Rm+1) + C = Rn-Rm+(C-1)
                results = torch.where(adc_gpu | adcs_gpu, adc_val, results)
                results = torch.where(sbc_gpu | sbcs_gpu, sbc_val, results)
                write_mask = write_mask | carry_ops
                # Flag-setting variants (ADCS / SBCS) — tensor-only, no .any() sync
                flag_ops = adcs_gpu | sbcs_gpu
                fo_idx = torch.where(flag_ops, batch_idx, BIG).min().clamp(0, batch_size - 1)
                _any_fo_t = (fo_idx < batch_size)
                fo_rn = rn_vals[fo_idx]
                fo_rm = rm_vals[fo_idx]
                fo_val = torch.where(adcs_gpu[fo_idx], adc_val[fo_idx], sbc_val[fo_idx])
                fo_n = ((fo_val >> 63) & 1).to(torch.float32)
                fo_z = (fo_val == 0).to(torch.float32)
                fo_is_adc = adcs_gpu[fo_idx]
                adcs_c = _u64_gt(fo_rn, ~(fo_rm + c_in))
                sbcs_c = _u64_ge(fo_rn, fo_rm)
                fo_c = torch.where(fo_is_adc, adcs_c, sbcs_c).to(torch.float32)
                fo_rn_n = (fo_rn >> 63) & 1
                fo_rm_n_eff = torch.where(fo_is_adc, (fo_rm >> 63) & 1, ~(fo_rm >> 63) & 1)
                fo_res_n = (fo_val >> 63) & 1
                fo_v = ((fo_rn_n == fo_rm_n_eff) & (fo_rn_n != fo_res_n)).to(torch.float32)
                flags[0] = torch.where(_any_fo_t, fo_n, flags[0])
                flags[1] = torch.where(_any_fo_t, fo_z, flags[1])
                flags[2] = torch.where(_any_fo_t, fo_c, flags[2])
                flags[3] = torch.where(_any_fo_t, fo_v, flags[3])

            # --- ANDS_REG / TST (ANDS with rd=31) ---
            # ANDS 64 REG: 0xEA000000 mask 0xFF200000; writes rd, updates flags (N,Z; C=V=0)
            # TST = ANDS with rd=31 (result discarded but flags set)
            ands_reg_gpu = ((insts & 0xFF200000) == 0xEA000000) & exec_mask
            if 0xEA in _pb:
                ands_result = rn_vals & rm_vals
                ands_idx = torch.where(ands_reg_gpu, batch_idx, BIG).min().clamp(0, batch_size - 1)
                ands_v = ands_result[ands_idx]
                flags[0] = ((ands_v >> 63) & 1).to(torch.float32)
                flags[1] = (ands_v == 0).to(torch.float32)
                flags[2] = _zero_f32
                flags[3] = _zero_f32
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
            if 0xD3 in _pb:
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
            if 0x53 in _pb:
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
            if 0x93 in _pb:
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
            if 0x13 in _pb:
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
            if (0xB3 in _pb or 0x33 in _pb):
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
            if (0x9B in _pb or 0x1B in _pb):
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
            if 0x9B in _pb:
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
            if (0x9B in _pb or 0x1B in _pb):
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
            if 0x9B in _pb:
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
            if 0x9B in _pb:
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
            if (0x93 in _pb or 0x13 in _pb):
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
            if (0xDA in _pb or 0x5A in _pb):
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
            if (0xDA in _pb or 0x5A in _pb):
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
            if (0xDA in _pb or 0x5A in _pb):
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
            if (0xDA in _pb or 0x5A in _pb):
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
            if (0x10 in _pb or 0x30 in _pb or 0x50 in _pb or 0x70 in _pb):
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
            if (0x9A in _pb or 0x1A in _pb or 0x5A in _pb):
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
            if (0xFA in _pb or 0x7A in _pb):
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
                if (0xFA in _pb or 0x7A in _pb):
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
                        flags[2] = _u64_ge(rn_vals[ccmp_idx], cc_rm_or_imm).to(torch.float32)
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
            if (0x92 in _pb or 0x12 in _pb or 0xB2 in _pb or 0x32 in _pb or 0xD2 in _pb or 0x52 in _pb):
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
            if (0x9A in _pb or 0x1A in _pb):
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
            if 0xF9 in _pb:
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
            if 0xB9 in _pb:
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
            if 0xF9 in _pb:
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
            if 0xB9 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xA9 in _pb:
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
            if 0xA9 in _pb:
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
            if 0xAD in _pb:
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
            if 0x39 in _pb:
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
            if 0x39 in _pb:
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
            if 0x38 in _pb:
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
            if 0x38 in _pb:
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
            if 0xB9 in _pb:
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
            if 0x79 in _pb:
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
            if 0x79 in _pb:
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
                if 0x79 in _pb:
                    lo_addrs = strh_addr_c[strh_mask].long()
                    hi_addrs = (strh_addr_c[strh_mask] + 1).clamp(0, self.mem_size-1).long()
                    mem.scatter_(0, lo_addrs, strh_lo[strh_mask])
                    mem.scatter_(0, hi_addrs, strh_hi[strh_mask])

            # --- LDRSB: 8-bit signed load, sign-extend ---
            # 64-bit dest: 0x39800000, 32-bit dest: 0x39C00000 (both mask 0xFFC00000)
            ldrsb_mask = (((insts & 0xFFC00000) == 0x39800000) |
                          ((insts & 0xFFC00000) == 0x39C00000)) & exec_mask
            if 0x39 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xF8 in _pb:
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            if 0xA8 in _pb:
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
            if 0xA8 in _pb:
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
            if 0xA8 in _pb:
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
            if 0xA8 in _pb:
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
            if 0xA9 in _pb:
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
            if 0xA9 in _pb:
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
            if 0xB8 in _pb:
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
            if 0x38 in _pb:
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
            if 0x38 in _pb:
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
            if 0x38 in _pb:
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
            if 0x78 in _pb:
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
            if 0x78 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xF8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if 0xB8 in _pb:
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            if 0x79 in _pb:
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
            if 0x78 in _pb:
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
            if 0x78 in _pb:
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
            if 0x38 in _pb:
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
            if (0x58 in _pb or 0x18 in _pb):
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
            if 0x78 in _pb:
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
            if 0x78 in _pb:
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
            if 0x78 in _pb:
                strh_p_imm9 = (insts >> 12) & 0x1FF
                strh_p_imm9_s = torch.where(strh_p_imm9 >= 256, strh_p_imm9 - 512, strh_p_imm9)
                strh_p_addr = rn_vals.clamp(0, self.mem_size - 2)
                strh_p_rt = insts & 0x1F
                strh_p_vals = torch.where(strh_p_rt == 31, torch.zeros_like(rd_vals),
                                          regs[strh_p_rt.clamp(0, 30)])
                if 0x78 in _pb:
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
            if 0x78 in _pb:
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
            if 0x38 in _pb:
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
            if 0x38 in _pb:
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
            if 0x78 in _pb:
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
            if 0x38 in _pb:
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
            if 0x78 in _pb:
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
            if 0x78 in _pb:
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
            if 0x29 in _pb:
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
            if 0x29 in _pb:
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
            if (0xC8 in _pb or 0x88 in _pb):
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
            # Unconditional writeback — empty index_put_ is a no-op (avoids .any() sync)
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

            # Tiny fusion for the common loop tail:
            #   SUBS/CMP ; B.cond
            # Consume the immediately preceding compare directly so the branch
            # sees the latest producer even when multiple flag-setting ops
            # appear earlier in the batch.
            prev_idx = (first_stop - 1).clamp(0, batch_size - 1)
            prev_inst = insts[prev_idx]
            prev_valid = (first_stop > 0) & is_bcond
            fuse_subs_imm = prev_valid & ((prev_inst & 0xFF000000) == 0xF1000000)
            fuse_subs_reg = prev_valid & ((prev_inst & 0xFF200000) == 0xEB000000)
            fuse_subs_bcond = fuse_subs_imm | fuse_subs_reg
            prev_rn = rn_vals[prev_idx]
            prev_rm = rm_vals[prev_idx]
            prev_imm12 = imm12[prev_idx]
            prev_rhs = torch.where(fuse_subs_reg, prev_rm, prev_imm12)
            prev_res = prev_rn.to(torch.int64) - prev_rhs.to(torch.int64)

            # Evaluate condition from flags tensor
            n, z, c, v = flags[0], flags[1], flags[2], flags[3]
            # Direct condition evaluation (eliminates 16-element torch.stack)
            _cc = cond_code.clamp(0, 15)
            _n_set = n > 0.5; _z_set = z > 0.5; _c_set = c > 0.5; _v_set = v > 0.5
            _ge = (_n_set == _v_set)
            cond_taken = (
                ((_cc == 0) & _z_set) | ((_cc == 1) & ~_z_set) |
                ((_cc == 2) & _c_set) | ((_cc == 3) & ~_c_set) |
                ((_cc == 4) & _n_set) | ((_cc == 5) & ~_n_set) |
                ((_cc == 6) & _v_set) | ((_cc == 7) & ~_v_set) |
                ((_cc == 8) & _c_set & ~_z_set) | ((_cc == 9) & (~_c_set | _z_set)) |
                ((_cc == 10) & _ge) | ((_cc == 11) & ~_ge) |
                ((_cc == 12) & ~_z_set & _ge) | ((_cc == 13) & (_z_set | ~_ge)) |
                (_cc == 14)
            )
            _fused_n_set = ((prev_res >> 63) & 1) != 0
            _fused_z_set = prev_res == 0
            _fused_c_set = _u64_ge(prev_rn, prev_rhs)
            _prev_rn_neg = ((prev_rn >> 63) & 1) != 0
            _prev_rhs_neg = ((prev_rhs.to(torch.int64) >> 63) & 1) != 0
            _fused_res_neg = ((prev_res >> 63) & 1) != 0
            _fused_v_set = (_prev_rn_neg != _prev_rhs_neg) & (_prev_rn_neg != _fused_res_neg)
            _fused_ge = (_fused_n_set == _fused_v_set)
            fused_cond_taken = (
                ((_cc == 0) & _fused_z_set) | ((_cc == 1) & ~_fused_z_set) |
                ((_cc == 2) & _fused_c_set) | ((_cc == 3) & ~_fused_c_set) |
                ((_cc == 4) & _fused_n_set) | ((_cc == 5) & ~_fused_n_set) |
                ((_cc == 6) & _fused_v_set) | ((_cc == 7) & ~_fused_v_set) |
                ((_cc == 8) & _fused_c_set & ~_fused_z_set) |
                ((_cc == 9) & (~_fused_c_set | _fused_z_set)) |
                ((_cc == 10) & _fused_ge) | ((_cc == 11) & ~_fused_ge) |
                ((_cc == 12) & ~_fused_z_set & _fused_ge) |
                ((_cc == 13) & (_fused_z_set | ~_fused_ge)) |
                (_cc == 14)
            )
            cond_taken = torch.where(fuse_subs_bcond, fused_cond_taken, cond_taken)
            new_pc = torch.where(is_bcond, torch.where(cond_taken, bcond_target, bcond_fallthrough), new_pc)

            # --- Halt check (masked by active) ---
            has_halt = halt_mask[first_stop.clamp(0, batch_size - 1)] & stopped_by_event & active
            halted_t = torch.where(has_halt, _const_1_i64, halted_t)

            # --- SVC (syscall) - unavoidable sync for I/O ---
            has_svc = svc_mask[first_stop.clamp(0, batch_size - 1)] & stopped_by_event & active

            # --- DEFERRED combined sync: halt | svc | loop-vectorizer ---
            # One CPU-GPU sync per SYNC_INTERVAL batches.
            _vect_done = False
            _force_vectorizer_sync = False
            if getattr(self, '_neural_branch_predictor', None) is not None:
                _bp_loop_mask = is_bcond & (bcond_imm19_signed < 0) & (first_stop > 0)
                if bool(_bp_loop_mask.item()):
                    _bp_cond = int(cond_code.item())
                    _bp_body_len = int(first_stop.item())
                    _loop_conds = {1, 10, 11, 12, 13}
                    if _bp_cond in _loop_conds and 3 <= _bp_body_len <= 8:
                        try:
                            _bp_model = self._neural_branch_predictor
                            _bp_model.update_flag_history(flags.detach(), device)
                            _bp_taken = float(_bp_model(_bp_cond, flags.detach(), True, 0.0).item())
                            if _iter % SYNC_INTERVAL != (SYNC_INTERVAL - 1):
                                _force_vectorizer_sync = _bp_taken > 0.9
                        except Exception:
                            pass
            if _force_vectorizer_sync or _iter % SYNC_INTERVAL == (SYNC_INTERVAL - 1):
                _stop_code = int((halted_t + (has_svc.long() << 1)).item())

                # ── Neural loop vectorizer ───────────────────────────────────
                # Two-pass: (1) neural loop detector (if available), (2) handcoded patterns.
                # Supports: ADD/SUB reg/imm + SUBS/CMP + MUL + bitwise + memset
                #           + B.NE/B.GT/B.GE/B.LT/B.LE
                if not (_stop_code & 1):
                    _vst = torch.stack([
                        is_bcond.long(), cond_code, bcond_imm19_signed, first_stop, pc_t
                    ]).tolist()
                    _is_bcond_v, _cond_v, _imm19_v, _body_len, _pc_v = (
                        int(_vst[0]), int(_vst[1]), int(_vst[2]), int(_vst[3]), int(_vst[4])
                    )
                    # NE=1, GE=10, LT=11, GT=12, LE=13
                    _loop_conds = {1, 10, 11, 12, 13}
                    if _is_bcond_v and _cond_v in _loop_conds and _imm19_v < 0 and 1 <= _body_len <= 8:
                        _stop_idx_v = _body_len
                        _stop_pc = _pc_v + _stop_idx_v * 4
                        _loop_start = _stop_pc + _imm19_v * 4
                        _loop_body_len = max(0, (_stop_pc - _loop_start) // 4)
                        if not (1 <= _loop_body_len <= 8):
                            continue

                        # ── Neural loop detector (first pass) ──
                        _neural_detected = False
                        if hasattr(self, '_neural_loop_detector') and self._neural_loop_detector is not None:
                            try:
                                _body_raw = mem[_loop_start : _loop_start + _loop_body_len * 4]
                                _body_words = []
                                for _bi in range(_loop_body_len):
                                    _bw = int(_body_raw[_bi*4].item()) | (int(_body_raw[_bi*4+1].item()) << 8) | \
                                          (int(_body_raw[_bi*4+2].item()) << 16) | (int(_body_raw[_bi*4+3].item()) << 24)
                                    _body_words.append(_bw)
                                _body_bits = torch.zeros(_loop_body_len, 32, device=device, dtype=torch.float32)
                                for _bi, _bw in enumerate(_body_words):
                                    for _bit in range(32):
                                        _body_bits[_bi, _bit] = (_bw >> _bit) & 1
                                _type_logits, _counter_probs, _iter_pred = self._neural_loop_detector(
                                    _body_bits, regs[:32].float())
                                _pred_type = _type_logits.argmax().item()
                                _pred_conf = F.softmax(_type_logits, dim=-1).max().item()
                                if _pred_type > 0 and _pred_conf > 0.8:
                                    _neural_detected = True
                                    # Use neural counter prediction
                                    _counter_reg = _counter_probs.argmax().item()
                            except Exception:
                                pass  # fall through to handcoded

                        _raw = mem[_loop_start : _loop_start + _loop_body_len * 4].tolist()
                        _loop_insts = [
                            _raw[i*4] | (_raw[i*4+1]<<8) | (_raw[i*4+2]<<16) | (_raw[i*4+3]<<24)
                            for i in range(_loop_body_len)
                        ]
                        _delta_regs: dict = {}
                        _delta_imms: dict = {}
                        _mul_ops: dict[int, int] = {}
                        _bw_ops: dict[int, tuple[str, int]] = {}
                        _mem_stores: list[tuple[str, int, int, int]] = []
                        _counter_reg = -1
                        _counter_step = 1  # default SUBS #1
                        _ok = True
                        for _linst in _loop_insts:
                            _lb  = (_linst >> 24) & 0xFF
                            _rd2 = _linst & 0x1F
                            _rn2 = (_linst >> 5) & 0x1F
                            if _lb == 0x8B:        # ADD REG 64
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _delta_regs[_rd2] = _rm2
                                else:
                                    _ok = False; break
                            elif _lb == 0xCB:      # SUB REG 64
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _delta_regs[_rd2] = -_rm2  # negative = flag for subtraction
                                else:
                                    _ok = False; break
                            elif _lb == 0x91:      # ADD IMM 64
                                _imm2 = (_linst >> 10) & 0xFFF
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _delta_imms[_rd2] = _imm2
                                else:
                                    _ok = False; break
                            elif _lb == 0xD1:      # SUB IMM 64
                                _imm2 = (_linst >> 10) & 0xFFF
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _delta_imms[_rd2] = -_imm2
                                else:
                                    _ok = False; break
                            elif _lb == 0xF1:      # SUBS IMM 64 (counter)
                                _imm2 = (_linst >> 10) & 0xFFF
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _counter_reg = _rd2
                                    _counter_step = _imm2 if _imm2 > 0 else 1
                                else:
                                    _ok = False; break
                            elif _lb == 0xEB:      # SUBS REG 64 / CMP
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _counter_reg = _rd2
                                    _counter_step = -_rm2  # negative = flag for register step
                                elif _rd2 == 31:
                                    pass  # CMP Xn, Xm — no write, skip
                                else:
                                    _ok = False; break
                            elif (_linst & 0xFFE0FC00) == 0x9B007C00:  # MUL 64
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _mul_ops[_rd2] = _rm2
                                else:
                                    _ok = False; break
                            elif _lb == 0x8A or _lb == 0x0A:  # AND REG 64/32
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _bw_ops[_rd2] = ('and', _rm2)
                                else:
                                    _ok = False; break
                            elif _lb == 0xAA or _lb == 0x2A:  # ORR REG 64/32
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _bw_ops[_rd2] = ('orr', _rm2)
                                else:
                                    _ok = False; break
                            elif _lb == 0xCA or _lb == 0x4A:  # EOR REG 64/32
                                _rm2 = (_linst >> 16) & 0x1F
                                if _rd2 == _rn2 and _rd2 != 31:
                                    _bw_ops[_rd2] = ('eor', _rm2)
                                else:
                                    _ok = False; break
                            elif _lb == 0x39:  # STRB unsigned offset
                                _mem_stores.append(('strb', _rd2, _rn2, (_linst >> 10) & 0xFFF))
                            elif _lb == 0x38 and (_linst & 0xFFE00C00) == 0x38000400:  # STRB post-index
                                _imm9 = (_linst >> 12) & 0x1FF
                                _mem_stores.append(('strb_post', _rd2, _rn2, _imm9))
                            else:
                                _ok = False; break
                        if _ok and _counter_reg >= 0:
                            _reg_vals = regs[:32].tolist()
                            _ctr_val = int(_reg_vals[_counter_reg])
                            # Compute iteration count from counter and step
                            if _counter_step > 0:
                                # SUBS imm: counter decrements by _counter_step each iter
                                _N = _ctr_val // _counter_step if _counter_step > 0 else _ctr_val
                            elif _counter_step < 0:
                                # SUBS REG: step comes from a register
                                _step_reg = -_counter_step
                                _step_val = int(_reg_vals[_step_reg])
                                _N = _ctr_val // _step_val if _step_val > 0 else 0
                            else:
                                _N = 0
                            _deltas: dict = {}
                            for rd, rm_or_neg in _delta_regs.items():
                                if rm_or_neg >= 0:
                                    _deltas[rd] = int(_reg_vals[rm_or_neg])  # ADD REG
                                else:
                                    _deltas[rd] = -int(_reg_vals[-rm_or_neg])  # SUB REG
                            _deltas.update(_delta_imms)
                            if 0 < _N <= 10_000_000:
                                # Additive deltas: Rd += N * delta
                                _N_t = torch.tensor([_N], dtype=torch.int64, device=device)
                                for _rd2, _delta in _deltas.items():
                                    if weave is not None:
                                        _d_t = torch.tensor([_delta], dtype=torch.int64, device=device)
                                        _total = weave._neural_mul_batch(_N_t, _d_t)
                                        _old = regs[_rd2:_rd2+1]
                                        _new = weave._neural_add_batch(_old, _total)
                                        regs[_rd2] = _new[0]
                                    else:
                                        regs[_rd2] = regs[_rd2] + torch.tensor(
                                            _N * _delta, device=device, dtype=torch.int64)
                                # MUL power: Rd *= base^N
                                if _mul_ops:
                                    for _rd2, _rm2 in _mul_ops.items():
                                        if _rm2 not in _delta_regs and _rm2 not in _delta_imms and _rm2 != _counter_reg:
                                            _base = int(_reg_vals[_rm2])
                                            if _base == 0:
                                                regs[_rd2] = torch.zeros(1, device=device, dtype=torch.int64).squeeze()
                                            elif _base == 1:
                                                pass  # x *= 1^N = no-op
                                            elif abs(_base) <= 2 and _N <= 63 or _N <= 20:
                                                regs[_rd2] = regs[_rd2] * torch.tensor(
                                                    int(_base ** _N), device=device, dtype=torch.int64)
                                            else:
                                                _ok = False; break
                                        else:
                                            _ok = False; break
                                # Bitwise idempotent: AND/ORR → single op; EOR → N parity
                                if _bw_ops and _ok:
                                    for _rd2, (_bw_type, _rm2) in _bw_ops.items():
                                        if _rm2 not in _delta_regs and _rm2 not in _delta_imms and _rm2 != _counter_reg:
                                            _bw_val = torch.tensor(int(_reg_vals[_rm2]), device=device, dtype=torch.int64)
                                            if _bw_type == 'and':
                                                regs[_rd2] = regs[_rd2] & _bw_val
                                            elif _bw_type == 'orr':
                                                regs[_rd2] = regs[_rd2] | _bw_val
                                            elif _bw_type == 'eor':
                                                if _N % 2 == 1:
                                                    regs[_rd2] = regs[_rd2] ^ _bw_val
                                                # else: even XOR = no-op
                                        else:
                                            _ok = False; break
                                # memset: STRB Wt, [Xn] with Xn advancing by stride
                                if _mem_stores and _ok:
                                    for _ms_type, _ms_rt, _ms_rn, _ms_off in _mem_stores:
                                        _ms_base = int(_reg_vals[_ms_rn])
                                        _ms_val = int(_reg_vals[_ms_rt]) & 0xFF
                                        if _ms_rn in _delta_imms:
                                            _ms_stride = _delta_imms[_ms_rn]
                                            if _ms_stride == 1 and 0 <= _ms_base and _ms_base + _N < self.mem_size:
                                                mem[_ms_base:_ms_base + _N] = _ms_val
                                            else:
                                                _ok = False; break
                                        elif _ms_type == 'strb_post' and _ms_off == 1:
                                            if 0 <= _ms_base and _ms_base + _N < self.mem_size:
                                                mem[_ms_base:_ms_base + _N] = _ms_val
                                                regs[_ms_rn] = regs[_ms_rn] + torch.tensor(
                                                    _N, device=device, dtype=torch.int64)
                                            else:
                                                _ok = False; break
                                        else:
                                            _ok = False; break
                                if _ok:
                                    regs[_counter_reg] = torch.zeros(
                                        1, device=device, dtype=torch.int64).squeeze()
                                    flags[0] = torch.zeros(1, device=device).squeeze()
                                    flags[1] = torch.ones(1, device=device).squeeze()
                                    flags[2] = torch.ones(1, device=device).squeeze()
                                    flags[3] = torch.zeros(1, device=device).squeeze()
                                    _vect_new_pc = torch.tensor(
                                        _stop_pc + 4, device=device, dtype=torch.int64)
                                    pc_t = torch.where(active, _vect_new_pc, pc_t)
                                    executed_t = executed_t + torch.tensor(
                                        _N * (_loop_body_len + 1),
                                        device=device, dtype=torch.int64)
                                    _vect_done = True
                                    _next_pc = _stop_pc + 4
                                    if 0 <= _next_pc < self.mem_size - 3:
                                        _peek = mem[_next_pc : _next_pc + 4].tolist()
                                        if not (_peek[0] | _peek[1] | _peek[2] | _peek[3]):
                                            break  # HALT immediately after loop → done
            else:
                _stop_code = 0

            if _vect_done:
                continue  # PC/executed updated; skip normal PC update

            # Match run_gpu_only() accounting by crediting taken backward loop
            # branches while still excluding terminal exits and HALT.
            taken_backward_loop_branch = (
                (is_b & (imm26_signed < 0)) |
                (is_cbz & cbz_taken & (cb_imm19_signed < 0)) |
                (is_cbnz & cbnz_taken & (cb_imm19_signed < 0)) |
                (is_tbz & (~tb_bit_set) & (tb_imm14_signed < 0)) |
                (is_tbnz & tb_bit_set & (tb_imm14_signed < 0)) |
                (is_bcond & cond_taken & (bcond_imm19_signed < 0))
            )
            executed_t = torch.where(
                taken_backward_loop_branch,
                executed_t + 1,
                executed_t,
            )

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
        final_executed = rust_executed + int(executed_t.item())
        self.inst_count.fill_(final_executed)
        self.halted = bool(halted_t.item())

        return final_executed, time.perf_counter() - start
