"""Execution-guided diffusion sampling via nCPU's differentiable engine.

Core novel contribution: differentiable execution as classifier guidance
during denoising. Every unmasking step gets gradients from "does this
code actually execute correctly?"
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import MaskedDiffusionTransformer, MASK_TOKEN, PAD_TOKEN, VOCAB_SIZE
from .tokenizer import (
    NUM_OPCODES, OPCODE_OFFSET, REG_OFFSET, NUM_REGISTERS,
    IMM_OFFSET, NUM_IMMEDIATES, BR_OFFSET, NUM_BRANCH_TARGETS,
)
from .sampler import build_slot_masks

# Import nCPU differentiable engine
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ncpu.differentiable.execution import (
    DifferentiableEngine, SoftProgram,
)
from ncpu.execution_training.execution_loss import ExecutionLoss


# ---------------------------------------------------------------------------
# Direct execution loss from token logits (maintains gradient flow)
# ---------------------------------------------------------------------------

class DifferentiableExecutionScorer(nn.Module):
    """Computes execution correctness score from token logits.

    Instead of creating a SoftProgram (which detaches gradients), this
    module directly interprets token logits as a soft program and runs
    it through the differentiable engine, maintaining full gradient flow.
    """

    def __init__(
        self,
        max_instructions: int = 32,
        num_registers: int = 8,
        max_exec_steps: int = 32,
        flag_scale: float = 0.1,
    ):
        super().__init__()
        self.max_instructions = max_instructions
        self.num_registers = num_registers
        self.max_exec_steps = max_exec_steps
        self.flag_scale = flag_scale

        # The DifferentiableALU computes all ops in parallel
        from ncpu.differentiable.execution import DifferentiableALU
        self.alu = DifferentiableALU(n_bits=16, bit_temperature=10.0)

    def soft_execute(
        self,
        token_logits: torch.Tensor,
        input_registers: Dict[int, float],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Execute token logits as a soft program and return final registers.

        This is a simplified differentiable execution that:
        1. Interprets token logits as soft instruction parameters
        2. Runs soft execution with weighted instruction dispatch
        3. Returns differentiable register state

        Args:
            token_logits: (seq_len, vocab_size) with requires_grad
            input_registers: initial register values
            temperature: softmax temperature for instruction selection

        Returns:
            registers: (num_registers,) differentiable final register state
        """
        device = token_logits.device
        L, V = token_logits.shape
        n_instr = min(L // 4, self.max_instructions)
        n_reg = self.num_registers

        # Parse instructions from logits
        opcode_logits_all = []  # [n_instr, NUM_OPCODES]
        dst_probs_all = []      # [n_instr, n_reg]
        src_probs_all = []      # [n_instr, n_reg]
        imm_values_all = []     # [n_instr]
        branch_probs_all = []   # [n_instr, n_instr]

        for i in range(n_instr):
            base = i * 4

            # Opcode
            op_logits = token_logits[base, OPCODE_OFFSET:OPCODE_OFFSET + NUM_OPCODES]
            opcode_logits_all.append(op_logits)

            # Dst register
            dst_logits = token_logits[base + 1, REG_OFFSET:REG_OFFSET + n_reg]
            dst_probs_all.append(F.softmax(dst_logits / temperature, dim=0))

            # Src register
            src_logits = token_logits[base + 2, REG_OFFSET:REG_OFFSET + n_reg]
            src_probs_all.append(F.softmax(src_logits / temperature, dim=0))

            # Immediate value (expected value under softmax)
            imm_logits = token_logits[base + 3, IMM_OFFSET:IMM_OFFSET + NUM_IMMEDIATES]
            imm_probs = F.softmax(imm_logits / temperature, dim=0)
            imm_vals = torch.arange(NUM_IMMEDIATES, dtype=torch.float32, device=device)
            imm_values_all.append((imm_probs * imm_vals).sum())

            # Branch target
            br_logits = token_logits[base + 3, IMM_OFFSET:IMM_OFFSET + n_instr]
            if br_logits.shape[0] < n_instr:
                br_logits = F.pad(br_logits, (0, n_instr - br_logits.shape[0]), value=-1e9)
            else:
                br_logits = br_logits[:n_instr]
            branch_probs_all.append(F.softmax(br_logits / temperature, dim=0))

        opcode_logits_t = torch.stack(opcode_logits_all)   # [n_instr, 14]
        dst_probs_t = torch.stack(dst_probs_all)           # [n_instr, n_reg]
        src_probs_t = torch.stack(src_probs_all)            # [n_instr, n_reg]
        imm_values_t = torch.stack(imm_values_all)         # [n_instr]
        branch_probs_t = torch.stack(branch_probs_all)      # [n_instr, n_instr]

        # Initialize registers
        registers = torch.zeros(n_reg, device=device)
        for idx, val in input_registers.items():
            if 0 <= idx < n_reg:
                registers[idx] = float(val)

        # Soft flags
        flags = torch.zeros(4, device=device)  # N, Z, C, V

        # Soft PC: start at instruction 0
        pc = torch.zeros(n_instr, device=device)
        pc[0] = 1.0

        # Execute
        for step in range(self.max_exec_steps):
            # Weighted instruction fetch (soft attention over instructions)
            # opcode weights for current PC
            opcode_weights = F.softmax(
                (pc.unsqueeze(1) * opcode_logits_t).sum(dim=0) / temperature,
                dim=0
            )  # [NUM_OPCODES]

            # Soft register reads
            dst_weights = (pc.unsqueeze(1) * dst_probs_t).sum(dim=0)   # [n_reg]
            src_weights = (pc.unsqueeze(1) * src_probs_t).sum(dim=0)   # [n_reg]

            src_val = (src_weights * registers).sum()
            dst_val = (dst_weights * registers).sum()
            imm_val = (pc * imm_values_t).sum()

            # Compute all ALU results
            alu_results = self.alu.compute_all(src_val, dst_val, imm_val, skip_bitwise=True)

            # Weighted result based on opcode
            # Map opcode names to weights
            op_names = ["NOP", "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL",
                        "AND", "OR", "XOR", "CMP", "BEQ", "BNE", "BGT", "HALT"]
            result_val = torch.tensor(0.0, device=device)
            for j, name in enumerate(op_names):
                if name in alu_results:
                    result_val = result_val + opcode_weights[j] * alu_results[name]

            # Soft register write: update dst register
            new_registers = registers.clone()
            for r in range(n_reg):
                write_weight = dst_weights[r]
                # Only write for non-branch, non-CMP, non-NOP, non-HALT opcodes
                write_mask = (opcode_weights[1] + opcode_weights[2] + opcode_weights[3] +
                              opcode_weights[4] + opcode_weights[5] + opcode_weights[6] +
                              opcode_weights[7] + opcode_weights[8])
                new_registers[r] = registers[r] + write_weight * write_mask * (result_val - registers[r])

            registers = new_registers

            # Update flags (for CMP)
            cmp_weight = opcode_weights[9]  # CMP
            diff = dst_val - src_val
            new_flags = self.alu.compute_flags(dst_val, src_val, scale=self.flag_scale)
            flags = flags + cmp_weight * (new_flags - flags)

            # Update PC
            # Sequential: pc[i] -> pc[i+1]
            sequential_pc = torch.zeros_like(pc)
            sequential_pc[1:] = pc[:-1]
            # Last instruction wraps (halts)

            # Branch PC
            branch_target = (pc.unsqueeze(1) * branch_probs_t).sum(dim=0)  # [n_instr]

            # Branch conditions
            beq_weight = opcode_weights[10]  # BEQ
            bne_weight = opcode_weights[11]  # BNE
            bgt_weight = opcode_weights[12]  # BGT
            halt_weight = opcode_weights[13]  # HALT

            z_flag = flags[1]  # Z
            n_flag = flags[0]  # N
            gt_flag = 1.0 - n_flag - z_flag  # approximate GT

            # Branch taken probability
            branch_taken = (beq_weight * z_flag +
                            bne_weight * (1.0 - z_flag) +
                            bgt_weight * gt_flag)

            # New PC
            pc = ((1.0 - branch_taken - halt_weight) * sequential_pc +
                  branch_taken * branch_target)

            # Renormalize PC
            pc_sum = pc.sum()
            if pc_sum > 1e-8:
                pc = pc / pc_sum
            else:
                break  # effectively halted

        return registers

    def compute_loss(
        self,
        token_logits: torch.Tensor,
        input_registers: Dict[int, float],
        expected_registers: Dict[int, float],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute differentiable execution loss.

        Args:
            token_logits: (seq_len, vocab_size) with requires_grad
            input_registers: {reg_idx: value}
            expected_registers: {reg_idx: expected_value}

        Returns:
            scalar loss (differentiable w.r.t. token_logits)
        """
        registers = self.soft_execute(token_logits, input_registers, temperature)

        # MSE loss on expected registers
        loss = torch.tensor(0.0, device=token_logits.device)
        n = 0
        for idx, expected_val in expected_registers.items():
            if 0 <= idx < self.num_registers:
                loss = loss + (registers[idx] - expected_val) ** 2
                n += 1

        if n > 0:
            loss = loss / n

        return loss


# ---------------------------------------------------------------------------
# Execution Spec
# ---------------------------------------------------------------------------

@dataclass
class ExecutionSpec:
    """Test cases for guided generation."""
    inputs: List[Dict[int, float]]
    expected: List[Dict[int, float]]

    @classmethod
    def from_data_spec(cls, spec_dict: dict) -> "ExecutionSpec":
        """Convert from data_generator spec format."""
        inputs = []
        expected = []
        for tc in spec_dict.get("test_cases", []):
            inp = {}
            for i, (name, val) in enumerate(tc["inputs"].items()):
                inp[i] = float(val)
            inputs.append(inp)
            expected.append({0: float(tc["expected_output"])})
        return cls(inputs=inputs, expected=expected)


# ---------------------------------------------------------------------------
# Execution-Guided Sampler
# ---------------------------------------------------------------------------

class ExecutionGuidedSampler:
    """Denoising sampler with execution guidance.
    
    Supports two modes:
    - 'gradient': differentiable execution gradients (soft execution)
    - 'rerank': generate multiple candidates per step, pick best by execution
    """

    def __init__(
        self,
        model: MaskedDiffusionTransformer,
        scorer: DifferentiableExecutionScorer,
        gamma: float = 2.0,
        gamma_schedule: str = "cosine_ramp",
        guidance_start: float = 0.3,
        mode: str = "gradient",
    ):
        self.model = model
        self.scorer = scorer
        self.gamma = gamma
        self.gamma_schedule = gamma_schedule
        self.guidance_start = guidance_start
        self.mode = mode

    def get_gamma(self, step: int, total_steps: int) -> float:
        """Guidance strength at this denoising step."""
        progress = step / total_steps
        if progress < self.guidance_start:
            return 0.0

        adj_progress = (progress - self.guidance_start) / (1.0 - self.guidance_start)

        if self.gamma_schedule == "constant":
            return self.gamma
        elif self.gamma_schedule == "cosine_ramp":
            return self.gamma * 0.5 * (1 - math.cos(math.pi * adj_progress))
        elif self.gamma_schedule == "linear_ramp":
            return self.gamma * adj_progress
        else:
            return self.gamma

    def compute_guidance_gradient(
        self,
        token_logits: torch.Tensor,
        exec_spec: ExecutionSpec,
    ) -> Tuple[torch.Tensor, float]:
        """Compute execution gradient w.r.t. token logits.

        Returns: (gradient tensor, loss value)
        """
        logits = token_logits.detach().clone().requires_grad_(True)

        # Average loss over test cases
        total_loss = torch.tensor(0.0, device=logits.device)
        for inp, exp in zip(exec_spec.inputs[:2], exec_spec.expected[:2]):
            loss = self.scorer.compute_loss(logits, inp, exp, temperature=1.0)
            total_loss = total_loss + loss
        total_loss = total_loss / min(len(exec_spec.inputs), 2)

        total_loss.backward()

        grad = logits.grad
        loss_val = total_loss.item()

        if grad is not None:
            return grad.detach(), loss_val
        return torch.zeros_like(token_logits), loss_val

    @torch.no_grad()
    def generate(
        self,
        spec_tokens: torch.Tensor,
        exec_spec: ExecutionSpec,
        seq_len: int = 128,
        num_steps: int = 64,
        temperature: float = 0.3,
        constrained: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Generate with execution-guided diffusion.

        Returns: (tokens, metrics)
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()

        tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
        spec_tokens = spec_tokens.to(device)
        slot_masks = build_slot_masks(seq_len).to(device) if constrained else None

        metrics = {"exec_losses": [], "gammas": [], "grad_norms": []}

        for step in range(num_steps):
            t = 1.0 - (step + 1) / num_steps
            t_tensor = torch.tensor([max(t, 0.01)], device=device)

            # Get model logits
            logits = self.model(tokens, t_tensor, spec_tokens=spec_tokens)[0]  # (L, V)

            # Execution guidance
            gamma = self.get_gamma(step, num_steps)
            if gamma > 0:
                with torch.enable_grad():
                    exec_grad, exec_loss = self.compute_guidance_gradient(logits, exec_spec)

                grad_norm = exec_grad.norm().item()
                metrics["exec_losses"].append(exec_loss)
                metrics["grad_norms"].append(grad_norm)
                metrics["gammas"].append(gamma)

                # Apply guidance: subtract gradient (minimize loss)
                if grad_norm > 1e-8:
                    # Normalize and scale
                    normalized_grad = exec_grad / (grad_norm + 1e-8)
                    logits = logits - gamma * normalized_grad
            else:
                metrics["gammas"].append(0.0)

            # Apply constraints
            if constrained and slot_masks is not None:
                logits = logits.clone()
                logits[~slot_masks] = -1e9

            # Sample
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(flat_probs, num_samples=1).view(seq_len)

            # Unmask most confident
            confidence = probs.max(dim=-1).values
            is_masked = (tokens[0] == MASK_TOKEN)
            num_masked = is_masked.sum().item()
            if num_masked == 0:
                break

            num_to_reveal = max(1, min(
                int(math.ceil(num_masked / max(num_steps - step, 1))),
                num_masked,
            ))

            masked_conf = confidence.clone()
            masked_conf[~is_masked] = -1.0
            _, top_idx = masked_conf.topk(num_to_reveal)
            for idx in top_idx:
                tokens[0, idx] = sampled[idx]

        # Final cleanup
        if (tokens == MASK_TOKEN).any():
            t_tensor = torch.tensor([0.01], device=device)
            logits = self.model(tokens, t_tensor, spec_tokens=spec_tokens)[0]
            if constrained and slot_masks is not None:
                logits = logits.clone()
                logits[~slot_masks] = -1e9
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
            mask = (tokens == MASK_TOKEN)
            tokens = torch.where(mask, final, tokens)

        return tokens, metrics
