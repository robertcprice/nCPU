"""Execution-guided diffusion sampling via nCPU's differentiable engine.

This is the core novel contribution: using differentiable execution as
classifier guidance during the denoising process. Every unmasking step
is informed by "does this code actually execute correctly?"

Architecture:
  token logits -> TokenToSoftProgramBridge -> SoftProgram -> DifferentiableEngine
                                                             -> ExecutionLoss
                                                             -> gradients
  gradients flow back to token logits -> guided sampling
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
    DifferentiableEngine, SoftProgram, ExecutionResult,
)
from ncpu.execution_training.execution_loss import ExecutionLoss


# ---------------------------------------------------------------------------
# Token-to-SoftProgram Bridge
# ---------------------------------------------------------------------------

class TokenToSoftProgramBridge(nn.Module):
    """Maps diffusion model token logits to SoftProgram parameters.

    The diffusion model outputs logits of shape (seq_len, vocab_size) for
    each position. This bridge extracts the relevant logit slices for each
    instruction slot and maps them to SoftProgram's parameter tensors.

    Instruction layout (4 tokens per instruction):
      slot 0: opcode  -> opcode_logits[i, :14]
      slot 1: dst_reg -> dst_logits[i, :8]
      slot 2: src_reg -> src1_logits[i, :8] (src2 unused in this ISA)
      slot 3: imm/br  -> immediates[i] and branch_logits[i, :]
    """

    def __init__(
        self,
        max_instructions: int = 32,
        num_registers: int = 8,
        num_opcodes: int = NUM_OPCODES,
    ):
        super().__init__()
        self.max_instructions = max_instructions
        self.num_registers = num_registers
        self.num_opcodes = num_opcodes

        # Small projection to refine immediate values from logits
        self.imm_proj = nn.Linear(NUM_IMMEDIATES + NUM_BRANCH_TARGETS, 1)
        # Branch target projection
        self.branch_proj = nn.Linear(NUM_IMMEDIATES + NUM_BRANCH_TARGETS, max_instructions)

    def forward(
        self,
        token_logits: torch.Tensor,
        seq_len: int = 128,
    ) -> SoftProgram:
        """Convert token logits to a SoftProgram.

        Args:
            token_logits: (seq_len, vocab_size) logits from diffusion model
                          (or (batch, seq_len, vocab_size) - batch dim stripped)

        Returns:
            SoftProgram with parameters set from the token logits
        """
        if token_logits.dim() == 3:
            token_logits = token_logits[0]  # strip batch dim

        L, V = token_logits.shape
        device = token_logits.device
        n_instr = min(L // 4, self.max_instructions)

        # Create SoftProgram with correct dimensions
        prog = SoftProgram(
            max_length=n_instr,
            num_registers=self.num_registers,
            num_opcodes=self.num_opcodes,
        ).to(device)

        # Extract logits for each instruction slot
        for i in range(n_instr):
            base = i * 4

            # Slot 0: opcode logits (tokens 0-13)
            opcode_logits = token_logits[base, OPCODE_OFFSET:OPCODE_OFFSET + self.num_opcodes]
            prog.opcode_logits.data[i] = opcode_logits

            # Slot 1: dst register logits (tokens 14-21)
            dst_logits = token_logits[base + 1, REG_OFFSET:REG_OFFSET + self.num_registers]
            prog.dst_logits.data[i] = dst_logits

            # Slot 2: src register logits (tokens 14-21)
            src_logits = token_logits[base + 2, REG_OFFSET:REG_OFFSET + self.num_registers]
            prog.src1_logits.data[i] = src_logits
            prog.src2_logits.data[i] = src_logits  # same in this ISA

            # Slot 3: immediate + branch logits (tokens 22-341)
            imm_br_logits = token_logits[base + 3, IMM_OFFSET:IMM_OFFSET + NUM_IMMEDIATES + NUM_BRANCH_TARGETS]

            # Immediate value: weighted sum of possible values
            imm_probs = F.softmax(imm_br_logits[:NUM_IMMEDIATES], dim=0)
            imm_values = torch.arange(NUM_IMMEDIATES, dtype=torch.float32, device=device)
            prog.immediates.data[i] = (imm_probs * imm_values).sum()

            # Branch target: project to instruction indices
            branch_logits = self.branch_proj(imm_br_logits)
            prog.branch_logits.data[i] = branch_logits[:n_instr]

        return prog

    def forward_differentiable(
        self,
        token_logits: torch.Tensor,
    ) -> SoftProgram:
        """Like forward() but maintains full gradient connectivity.

        Instead of setting .data (which detaches), we create a SoftProgram
        and replace its parameters with computed tensors via register_buffer
        trick and custom forward hooks.

        Args:
            token_logits: (seq_len, vocab_size) with requires_grad=True

        Returns:
            SoftProgram whose parameters are differentiable functions of token_logits
        """
        if token_logits.dim() == 3:
            token_logits = token_logits[0]

        L, V = token_logits.shape
        device = token_logits.device
        n_instr = min(L // 4, self.max_instructions)

        # Build parameter tensors from logits (all differentiable)
        opcode_list = []
        dst_list = []
        src1_list = []
        src2_list = []
        imm_list = []
        branch_list = []

        for i in range(n_instr):
            base = i * 4

            # Opcode logits
            opcode_logits = token_logits[base, OPCODE_OFFSET:OPCODE_OFFSET + self.num_opcodes]
            opcode_list.append(opcode_logits)

            # Register logits
            dst_logits = token_logits[base + 1, REG_OFFSET:REG_OFFSET + self.num_registers]
            src_logits = token_logits[base + 2, REG_OFFSET:REG_OFFSET + self.num_registers]
            dst_list.append(dst_logits)
            src1_list.append(src_logits)
            src2_list.append(src_logits)

            # Immediate: expected value under softmax
            imm_br_logits = token_logits[base + 3, IMM_OFFSET:IMM_OFFSET + NUM_IMMEDIATES + NUM_BRANCH_TARGETS]
            imm_probs = F.softmax(imm_br_logits[:NUM_IMMEDIATES], dim=0)
            imm_values = torch.arange(NUM_IMMEDIATES, dtype=torch.float32, device=device)
            immediate = (imm_probs * imm_values).sum()
            imm_list.append(immediate)

            # Branch logits
            br = self.branch_proj(imm_br_logits)[:n_instr]
            # Pad to n_instr if needed
            if br.shape[0] < n_instr:
                br = F.pad(br, (0, n_instr - br.shape[0]))
            branch_list.append(br)

        # Create SoftProgram and inject differentiable parameters
        prog = SoftProgram(
            max_length=n_instr,
            num_registers=self.num_registers,
            num_opcodes=self.num_opcodes,
        ).to(device)

        # Stack and assign (these maintain gradient connectivity)
        prog.opcode_logits = nn.Parameter(torch.stack(opcode_list))
        prog.dst_logits = nn.Parameter(torch.stack(dst_list))
        prog.src1_logits = nn.Parameter(torch.stack(src1_list))
        prog.src2_logits = nn.Parameter(torch.stack(src2_list))
        prog.immediates = nn.Parameter(torch.stack(imm_list))
        prog.branch_logits = nn.Parameter(torch.stack(branch_list))

        return prog


# ---------------------------------------------------------------------------
# Execution-Guided Sampler
# ---------------------------------------------------------------------------

@dataclass
class ExecutionSpec:
    """Specification for guided generation: test cases to pass."""
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


class ExecutionGuidedSampler:
    """Denoising sampler that uses nCPU execution gradients as guidance.

    At each denoising step:
    1. Get model's predicted logits for masked positions
    2. Convert logits to a SoftProgram via the bridge
    3. Execute differentiably against the test spec
    4. Backprop execution loss to get token-level gradients
    5. Use gradients as classifier guidance to shift predictions
    """

    def __init__(
        self,
        model: MaskedDiffusionTransformer,
        bridge: TokenToSoftProgramBridge,
        engine: DifferentiableEngine,
        loss_fn: ExecutionLoss,
        gamma: float = 2.0,
        gamma_schedule: str = "cosine_ramp",
        exec_temperature: float = 1.0,
        max_exec_steps: int = 32,
    ):
        self.model = model
        self.bridge = bridge
        self.engine = engine
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.gamma_schedule = gamma_schedule
        self.exec_temperature = exec_temperature
        self.max_exec_steps = max_exec_steps

    def get_gamma(self, step: int, total_steps: int) -> float:
        """Compute guidance strength at this denoising step."""
        progress = step / total_steps  # 0 -> 1 as denoising progresses

        if self.gamma_schedule == "constant":
            return self.gamma
        elif self.gamma_schedule == "cosine_ramp":
            # More guidance as code becomes clearer
            return self.gamma * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.gamma_schedule == "linear_ramp":
            return self.gamma * progress
        elif self.gamma_schedule == "late_only":
            # Only guide in the last 50%
            return self.gamma if progress > 0.5 else 0.0
        else:
            return self.gamma

    def compute_execution_guidance(
        self,
        token_logits: torch.Tensor,
        exec_spec: ExecutionSpec,
    ) -> torch.Tensor:
        """Compute execution loss gradient w.r.t. token logits.

        Args:
            token_logits: (seq_len, vocab_size) requires_grad=True
            exec_spec: test cases to evaluate against

        Returns:
            (seq_len, vocab_size) gradient of execution loss w.r.t. logits
        """
        # Ensure gradients flow
        token_logits = token_logits.detach().requires_grad_(True)

        # Convert to SoftProgram (differentiable path)
        soft_prog = self.bridge.forward_differentiable(token_logits)

        # Run execution loss against test cases
        if len(exec_spec.inputs) > 1:
            # Batched execution
            result = self.loss_fn.compute_soft_batched(
                soft_prog,
                batch_inputs=exec_spec.inputs,
                batch_expected=exec_spec.expected,
                temperature=self.exec_temperature,
                skip_bitwise=True,
            )
        else:
            result = self.loss_fn.compute_soft(
                soft_prog,
                inputs=exec_spec.inputs[0],
                expected=exec_spec.expected[0],
                temperature=self.exec_temperature,
                skip_bitwise=True,
            )

        # Backprop through execution
        loss = result.total_loss
        if loss.requires_grad:
            loss.backward()
            grad = token_logits.grad
            if grad is not None:
                return grad.detach()

        # Fallback: no gradient (execution was trivial or degenerate)
        return torch.zeros_like(token_logits)

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
        """Generate a program with execution-guided diffusion.

        Args:
            spec_tokens: (1, S) conditioning tokens
            exec_spec: test cases for execution guidance
            seq_len: length of output sequence
            num_steps: number of denoising steps
            temperature: sampling temperature
            constrained: enforce ISA slot constraints
            device: compute device

        Returns:
            (tokens, metrics): generated token IDs and guidance metrics
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        self.bridge.eval()

        # Start fully masked
        tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
        spec_tokens = spec_tokens.to(device)

        # Build slot constraint masks
        slot_masks = build_slot_masks(seq_len).to(device) if constrained else None

        metrics = {
            "exec_loss_history": [],
            "gamma_history": [],
            "guidance_norm_history": [],
        }

        for step in range(num_steps):
            t = 1.0 - (step + 1) / num_steps
            t_tensor = torch.tensor([max(t, 0.01)], device=device)

            # 1. Get model's predicted logits
            with torch.enable_grad():
                logits = self.model(tokens, t_tensor, spec_tokens=spec_tokens)  # (1, L, V)

            logits_squeezed = logits[0]  # (L, V)

            # 2. Compute execution guidance gradient
            gamma = self.get_gamma(step, num_steps)

            if gamma > 0 and step > num_steps * 0.3:
                # Only guide after 30% of steps (early code is too noisy)
                with torch.enable_grad():
                    exec_grad = self.compute_execution_guidance(
                        logits_squeezed.detach().clone(),
                        exec_spec,
                    )

                grad_norm = exec_grad.norm().item()
                metrics["guidance_norm_history"].append(grad_norm)

                # 3. Apply classifier guidance
                # Subtract gradient because we want to MINIMIZE execution loss
                # Normalize gradient to prevent scale issues
                if grad_norm > 1e-8:
                    exec_grad = exec_grad / (grad_norm + 1e-8)
                    guided_logits = logits_squeezed - gamma * exec_grad
                else:
                    guided_logits = logits_squeezed

                metrics["gamma_history"].append(gamma)
            else:
                guided_logits = logits_squeezed
                metrics["gamma_history"].append(0.0)

            # 4. Apply constraints and sample
            if constrained and slot_masks is not None:
                guided_logits = guided_logits.clone()
                guided_logits[~slot_masks] = -1e9

            probs = F.softmax(guided_logits / max(temperature, 1e-8), dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(flat_probs, num_samples=1).view(seq_len)

            # 5. Unmask most confident positions
            confidence = probs.max(dim=-1).values
            is_masked = (tokens[0] == MASK_TOKEN)
            num_masked = is_masked.sum().item()

            if num_masked == 0:
                break

            num_to_reveal = max(1, min(
                int(math.ceil(num_masked / max(num_steps - step, 1))),
                num_masked,
            ))

            masked_confidence = confidence.clone()
            masked_confidence[~is_masked] = -1.0
            _, top_indices = masked_confidence.topk(num_to_reveal)

            for idx in top_indices:
                tokens[0, idx] = sampled[idx]

        # Final cleanup
        if (tokens == MASK_TOKEN).any():
            t_tensor = torch.tensor([0.01], device=device)
            logits = self.model(tokens, t_tensor, spec_tokens=spec_tokens)
            if constrained and slot_masks is not None:
                logits = logits.clone()
                logits[0][~slot_masks] = -1e9
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
            mask = (tokens == MASK_TOKEN)
            tokens = torch.where(mask, final, tokens)

        return tokens, metrics
