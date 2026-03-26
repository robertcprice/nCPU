"""Differentiable CPU execution engine with full gradient flow.

Every operation maintains gradient flow through the computation graph.
Supports two modes:
  - FixedProgram: hard instruction sequence, differentiable immediates
  - SoftProgram: fully differentiable program (Gumbel-softmax over opcodes/registers)

Builds on the coprocessor's soft_alu components for bitwise operations,
and uses native tensor ops for arithmetic (naturally differentiable).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.coprocessor.soft_alu import (
    SoftNeuralLogical,
    soft_bits_to_int,
    soft_int_to_bits,
    ste_threshold,
)

# ---------------------------------------------------------------------------
# Opcode table
# ---------------------------------------------------------------------------

OPCODES = {
    "NOP": 0,
    "MOV_IMM": 1,   # dst = immediate
    "MOV_REG": 2,   # dst = src1
    "ADD": 3,       # dst = src1 + src2
    "SUB": 4,       # dst = src1 - src2
    "MUL": 5,       # dst = src1 * src2
    "AND": 6,       # dst = src1 & src2
    "OR": 7,        # dst = src1 | src2
    "XOR": 8,       # dst = src1 ^ src2
    "CMP": 9,       # flags = compare(src1, src2)
    "BEQ": 10,      # branch if Z=1
    "BNE": 11,      # branch if Z=0
    "BGT": 12,      # branch if N=0 and Z=0
    "HALT": 13,
}

NUM_OPCODES = len(OPCODES)
_OP = {v: k for k, v in OPCODES.items()}


# ---------------------------------------------------------------------------
# Instruction and Program representations
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """A single instruction with hard or soft fields."""

    opcode: int
    dst: int = 0
    src1: int = 0
    src2: int = 0
    immediate: float = 0.0
    branch_target: int = 0


@dataclass
class ExecutionResult:
    """Result of executing a program through the differentiable engine."""

    registers: torch.Tensor        # [num_registers] final register state
    flags: torch.Tensor            # [4] N, Z, C, V
    steps_executed: int
    halted: bool
    register_trace: list[torch.Tensor]  # per-step register snapshots


class FixedProgram(nn.Module):
    """Program with fixed structure but differentiable immediate values.

    Use this for program optimization: the instruction sequence is fixed,
    but immediate values are nn.Parameters that can be optimized via
    gradient descent.
    """

    def __init__(self, instructions: list[Instruction]):
        super().__init__()
        self.instructions = instructions
        self.length = len(instructions)

        # Extract immediates as a differentiable parameter vector
        imm_vals = [inst.immediate for inst in instructions]
        self.immediates = nn.Parameter(torch.tensor(imm_vals, dtype=torch.float32))

    def get_hard_instruction(self, idx: int) -> tuple:
        """Return (opcode, dst, src1, src2, immediate, branch_target) at idx."""
        inst = self.instructions[idx]
        return (
            inst.opcode,
            inst.dst,
            inst.src1,
            inst.src2,
            self.immediates[idx],     # differentiable!
            inst.branch_target,
        )


class SoftProgram(nn.Module):
    """Fully differentiable program representation.

    Every aspect of the program --- opcodes, register selections, immediates,
    branch targets --- is a continuous parameter optimized via gradient descent.
    Gumbel-softmax converts continuous logits to (approximately) discrete
    instruction choices during training, with temperature annealing to
    converge toward a discrete program.
    """

    def __init__(
        self,
        max_length: int = 16,
        num_registers: int = 8,
        num_opcodes: int = NUM_OPCODES,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.max_length = max_length
        self.num_registers = num_registers
        self.num_opcodes = num_opcodes

        # Learnable program parameters
        self.opcode_logits = nn.Parameter(
            torch.randn(max_length, num_opcodes) * init_scale
        )
        self.dst_logits = nn.Parameter(
            torch.randn(max_length, num_registers) * init_scale
        )
        self.src1_logits = nn.Parameter(
            torch.randn(max_length, num_registers) * init_scale
        )
        self.src2_logits = nn.Parameter(
            torch.randn(max_length, num_registers) * init_scale
        )
        self.immediates = nn.Parameter(torch.zeros(max_length))
        self.branch_logits = nn.Parameter(
            torch.randn(max_length, max_length) * init_scale
        )

    def get_soft_instruction(
        self, pc_weights: torch.Tensor, temperature: float = 1.0
    ) -> tuple:
        """Fetch a soft instruction weighted by PC distribution.

        Returns soft (opcode_weights, dst_weights, src1_weights, src2_weights,
                       immediate, branch_weights).
        """
        # Weighted sum of all instruction slots by PC distribution
        # pc_weights: [max_length]
        pc = pc_weights.unsqueeze(0)  # [1, max_length]

        # Soft opcode: weighted sum of per-position Gumbel-softmax
        opcode_probs = F.gumbel_softmax(self.opcode_logits, tau=temperature, hard=False)
        opcode_weights = (pc @ opcode_probs).squeeze(0)  # [num_opcodes]

        dst_probs = F.softmax(self.dst_logits, dim=-1)
        dst_weights = (pc @ dst_probs).squeeze(0)  # [num_registers]

        src1_probs = F.softmax(self.src1_logits, dim=-1)
        src1_weights = (pc @ src1_probs).squeeze(0)

        src2_probs = F.softmax(self.src2_logits, dim=-1)
        src2_weights = (pc @ src2_probs).squeeze(0)

        immediate = (pc_weights * self.immediates).sum()

        branch_probs = F.softmax(self.branch_logits, dim=-1)
        branch_weights = (pc @ branch_probs).squeeze(0)  # [max_length]

        return (
            opcode_weights,
            dst_weights,
            src1_weights,
            src2_weights,
            immediate,
            branch_weights,
        )

    def extract_discrete_program(self) -> list[Instruction]:
        """Extract the most likely discrete program from learned parameters."""
        instructions = []
        for i in range(self.max_length):
            opcode = int(self.opcode_logits[i].argmax().item())
            dst = int(self.dst_logits[i].argmax().item())
            src1 = int(self.src1_logits[i].argmax().item())
            src2 = int(self.src2_logits[i].argmax().item())
            imm = float(self.immediates[i].item())
            branch = int(self.branch_logits[i].argmax().item())
            instructions.append(
                Instruction(opcode, dst, src1, src2, imm, branch)
            )
        return instructions

    def format_program(self) -> str:
        """Pretty-print the most likely discrete program."""
        lines = []
        for i, inst in enumerate(self.extract_discrete_program()):
            name = _OP.get(inst.opcode, f"OP{inst.opcode}")
            if name == "HALT":
                lines.append(f"  {i:2d}: HALT")
            elif name == "NOP":
                lines.append(f"  {i:2d}: NOP")
            elif name == "MOV_IMM":
                lines.append(f"  {i:2d}: MOV R{inst.dst}, #{inst.immediate:.1f}")
            elif name == "MOV_REG":
                lines.append(f"  {i:2d}: MOV R{inst.dst}, R{inst.src1}")
            elif name in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
                lines.append(
                    f"  {i:2d}: {name} R{inst.dst}, R{inst.src1}, R{inst.src2}"
                )
            elif name == "CMP":
                lines.append(f"  {i:2d}: CMP R{inst.src1}, R{inst.src2}")
            elif name in ("BEQ", "BNE", "BGT"):
                lines.append(f"  {i:2d}: {name} @{inst.branch_target}")
            else:
                lines.append(f"  {i:2d}: {name} ???")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Differentiable ALU operations
# ---------------------------------------------------------------------------

class DifferentiableALU(nn.Module):
    """ALU that computes all operations in parallel with full gradient flow.

    Arithmetic (ADD, SUB, MUL): native tensor ops (naturally differentiable).
    Bitwise (AND, OR, XOR): soft truth tables from coprocessor soft_alu.
    Comparison: sigmoid-based soft flags.
    """

    def __init__(self, n_bits: int = 16, bit_temperature: float = 10.0):
        super().__init__()
        self.n_bits = n_bits
        self.bit_temperature = bit_temperature
        self.soft_logical = SoftNeuralLogical(n_bits)

    def compute_all(
        self,
        src1: torch.Tensor,
        src2: torch.Tensor,
        immediate: torch.Tensor,
        skip_bitwise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute all operations in parallel. Each returns a scalar tensor.

        All outputs maintain gradient flow back to src1, src2, and immediate.

        Args:
            skip_bitwise: If True, skip expensive bitwise ops (AND/OR/XOR)
                and return zeros for them. Use when synthesizing programs that
                don't need bitwise operations, for ~10x speedup.
        """
        results = {}

        # --- Arithmetic (native tensor ops, fully differentiable) ---
        results["NOP"] = torch.zeros_like(src1)
        results["MOV_IMM"] = immediate
        results["MOV_REG"] = src1
        results["ADD"] = src1 + src2
        results["SUB"] = src1 - src2
        results["MUL"] = src1 * src2

        if not skip_bitwise:
            # --- Bitwise (soft truth tables, differentiable via bilinear) ---
            bits_a = soft_int_to_bits(
                src1.unsqueeze(0), self.n_bits, self.bit_temperature
            ).squeeze(0)
            bits_b = soft_int_to_bits(
                src2.unsqueeze(0), self.n_bits, self.bit_temperature
            ).squeeze(0)

            and_bits = self.soft_logical.forward_single_op(
                bits_a.unsqueeze(0), bits_b.unsqueeze(0), op_idx=0
            ).squeeze(0)
            results["AND"] = soft_bits_to_int(and_bits.unsqueeze(0)).squeeze(0)

            or_bits = self.soft_logical.forward_single_op(
                bits_a.unsqueeze(0), bits_b.unsqueeze(0), op_idx=1
            ).squeeze(0)
            results["OR"] = soft_bits_to_int(or_bits.unsqueeze(0)).squeeze(0)

            xor_bits = self.soft_logical.forward_single_op(
                bits_a.unsqueeze(0), bits_b.unsqueeze(0), op_idx=2
            ).squeeze(0)
            results["XOR"] = soft_bits_to_int(xor_bits.unsqueeze(0)).squeeze(0)
        else:
            zero = torch.zeros_like(src1)
            results["AND"] = zero
            results["OR"] = zero
            results["XOR"] = zero

        # --- Comparison (produces flags, result is zero) ---
        results["CMP"] = torch.zeros_like(src1)

        # --- Branch/halt (no ALU result) ---
        results["BEQ"] = torch.zeros_like(src1)
        results["BNE"] = torch.zeros_like(src1)
        results["BGT"] = torch.zeros_like(src1)
        results["HALT"] = torch.zeros_like(src1)

        return results

    def compute_flags(
        self, src1: torch.Tensor, src2: torch.Tensor, scale: float = 0.1
    ) -> torch.Tensor:
        """Compute soft NZCV flags from comparison.

        Returns [N, Z, C, V] as soft values in [0, 1].
        """
        diff = src1 - src2

        # N: result is negative
        n_flag = torch.sigmoid(-diff * (1.0 / scale))

        # Z: result is zero (peaked Gaussian around 0)
        z_flag = torch.exp(-(diff ** 2) / (2 * scale ** 2))

        # C: unsigned carry (src1 >= src2 for subtraction)
        c_flag = torch.sigmoid((src1 - src2) * (1.0 / scale))

        # V: signed overflow (simplified)
        v_flag = torch.zeros_like(src1)

        return torch.stack([n_flag, z_flag, c_flag, v_flag])


# ---------------------------------------------------------------------------
# Differentiable Execution Engine
# ---------------------------------------------------------------------------

class DifferentiableEngine(nn.Module):
    """Execute programs with full gradient flow through every instruction.

    This is the core module that proves nCPU's central thesis: because the
    CPU is differentiable, you can backpropagate through program execution.

    Supports:
      - FixedProgram: optimize immediate values via gradient descent
      - SoftProgram: discover entire programs via gradient descent
    """

    def __init__(
        self,
        num_registers: int = 8,
        n_bits: int = 16,
        bit_temperature: float = 10.0,
        flag_scale: float = 0.1,
    ):
        super().__init__()
        self.num_registers = num_registers
        self.alu = DifferentiableALU(n_bits, bit_temperature)
        self.flag_scale = flag_scale

    # -- Register access (soft) ------------------------------------------

    def _soft_read(
        self, registers: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Read a register value via soft attention. Fully differentiable."""
        return (weights * registers).sum()

    def _soft_write(
        self,
        registers: torch.Tensor,
        weights: torch.Tensor,
        value: torch.Tensor,
        write_enable: torch.Tensor,
    ) -> torch.Tensor:
        """Write a value to registers via soft attention. Fully differentiable.

        write_enable in [0,1] controls whether the write happens at all
        (0 for NOP, CMP, branches; 1 for ALU/MOV ops).
        """
        # Blend: new = old * (1 - w*we) + value * w * we
        w = weights * write_enable
        return registers * (1.0 - w) + value * w

    # -- PC management ---------------------------------------------------

    def _advance_pc(self, pc: torch.Tensor) -> torch.Tensor:
        """Advance PC by 1 (shift distribution right). Differentiable."""
        return torch.roll(pc, 1, dims=0)

    def _branch_pc(
        self, pc: torch.Tensor, branch_weights: torch.Tensor
    ) -> torch.Tensor:
        """Set PC to branch target distribution. Differentiable."""
        return branch_weights

    # -- Fixed program execution -----------------------------------------

    def execute_fixed(
        self,
        program: FixedProgram,
        inputs: dict[int, float],
        max_steps: int = 64,
    ) -> ExecutionResult:
        """Execute a FixedProgram with gradient flow through immediates.

        Args:
            program: FixedProgram with differentiable immediates
            inputs: {register_index: initial_value}
            max_steps: maximum execution steps

        Returns:
            ExecutionResult with differentiable register state
        """
        # Initialize registers
        regs = torch.zeros(self.num_registers)
        for idx, val in inputs.items():
            if isinstance(val, torch.Tensor):
                regs[idx] = val
            else:
                regs[idx] = float(val)

        flags = torch.zeros(4)  # N, Z, C, V
        pc = 0  # hard PC for fixed programs
        trace = [regs.clone()]
        halted = False
        steps = 0

        for step in range(max_steps):
            if pc < 0 or pc >= program.length:
                halted = True
                break

            opcode, dst, src1, src2, imm, branch_target = \
                program.get_hard_instruction(pc)

            name = _OP.get(opcode, "NOP")

            # Read source registers (hard indexing, values are differentiable)
            src1_val = regs[src1]
            src2_val = regs[src2]

            # Execute the specific operation
            if name == "HALT":
                halted = True
                steps = step + 1
                break
            elif name == "NOP":
                pc += 1
            elif name == "MOV_IMM":
                regs = regs.clone()
                regs[dst] = imm  # imm is differentiable Parameter
                pc += 1
            elif name == "MOV_REG":
                regs = regs.clone()
                regs[dst] = src1_val
                pc += 1
            elif name in ("ADD", "SUB", "MUL"):
                regs = regs.clone()
                if name == "ADD":
                    regs[dst] = src1_val + src2_val
                elif name == "SUB":
                    regs[dst] = src1_val - src2_val
                elif name == "MUL":
                    regs[dst] = src1_val * src2_val
                pc += 1
            elif name in ("AND", "OR", "XOR"):
                regs = regs.clone()
                all_ops = self.alu.compute_all(src1_val, src2_val, imm)
                regs[dst] = all_ops[name]
                pc += 1
            elif name == "CMP":
                flags = self.alu.compute_flags(src1_val, src2_val, self.flag_scale)
                pc += 1
            elif name == "BEQ":
                if flags[1] > 0.5:  # Z flag
                    pc = branch_target
                else:
                    pc += 1
            elif name == "BNE":
                if flags[1] <= 0.5:
                    pc = branch_target
                else:
                    pc += 1
            elif name == "BGT":
                if flags[0] <= 0.5 and flags[1] <= 0.5:  # not N and not Z
                    pc = branch_target
                else:
                    pc += 1
            else:
                pc += 1

            trace.append(regs.clone())
            steps = step + 1

        return ExecutionResult(
            registers=regs,
            flags=flags,
            steps_executed=steps,
            halted=halted,
            register_trace=trace,
        )

    # -- Soft program execution ------------------------------------------

    def execute_soft(
        self,
        program: SoftProgram,
        inputs: dict[int, float],
        max_steps: int = 32,
        temperature: float = 1.0,
        skip_bitwise: bool = False,
        hard_branch_threshold: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a SoftProgram with full gradient flow through everything.

        Every aspect of the program is differentiable: opcodes, register
        selections, immediates, and branch targets are all continuous
        parameters optimized via gradient descent.

        Args:
            program: SoftProgram with learnable parameters
            inputs: {register_index: initial_value}
            max_steps: maximum execution steps
            temperature: Gumbel-softmax temperature (anneal toward 0)
            skip_bitwise: skip expensive bitwise ops for ~10x speedup
            hard_branch_threshold: When not None and temperature is below
                this threshold, use Straight-Through Estimator (STE) for
                hard branch decisions. The forward pass takes the hard
                argmax path (branch_prob > 0.5), but the backward pass
                uses the original soft branch_prob for gradient flow.
                This prevents blended taken/not-taken paths at low
                temperature, enabling clean branch specialization.

        Returns:
            ExecutionResult with differentiable register state
        """
        # Initialize registers
        regs = torch.zeros(self.num_registers)
        for idx, val in inputs.items():
            if isinstance(val, torch.Tensor):
                regs[idx] = val
            else:
                regs[idx] = float(val)

        # Soft PC: distribution over instruction positions
        pc = torch.zeros(program.max_length)
        pc[0] = 1.0  # start at position 0

        flags = torch.zeros(4)
        trace = [regs.clone()]
        cumulative_halt = torch.tensor(0.0)

        for step in range(max_steps):
            # Save pre-step registers for halt masking
            old_regs = regs.clone()

            # Fetch soft instruction
            (
                opcode_w, dst_w, src1_w, src2_w, immediate, branch_w
            ) = program.get_soft_instruction(pc, temperature)

            # Read source operands via soft attention
            src1_val = self._soft_read(regs, src1_w)
            src2_val = self._soft_read(regs, src2_w)

            # Compute ALL operations in parallel
            all_results = self.alu.compute_all(
                src1_val, src2_val, immediate, skip_bitwise=skip_bitwise
            )

            # Weighted combination by soft opcode
            result = torch.tensor(0.0)
            for op_name, op_idx in OPCODES.items():
                result = result + opcode_w[op_idx] * all_results[op_name]

            # Compute which opcodes produce a register write
            write_ops = {
                "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL",
                "AND", "OR", "XOR",
            }
            write_enable = sum(
                opcode_w[OPCODES[op]] for op in write_ops
            )

            # Soft register write
            regs = self._soft_write(regs, dst_w, result, write_enable)

            # Update flags (weighted by CMP probability)
            cmp_weight = opcode_w[OPCODES["CMP"]]
            new_flags = self.alu.compute_flags(
                src1_val, src2_val, self.flag_scale
            )
            flags = flags * (1.0 - cmp_weight) + new_flags * cmp_weight

            # Update PC
            # Normal advance (all non-branch ops)
            next_pc = self._advance_pc(pc)

            # Branch taken probability
            beq_prob = opcode_w[OPCODES["BEQ"]] * flags[1]  # Z flag
            bne_prob = opcode_w[OPCODES["BNE"]] * (1.0 - flags[1])
            bgt_prob = opcode_w[OPCODES["BGT"]] * (1.0 - flags[0]) * (1.0 - flags[1])
            branch_prob = beq_prob + bne_prob + bgt_prob
            branch_prob = branch_prob.clamp(0.0, 1.0)

            branch_pc = self._branch_pc(pc, branch_w)

            # At low temperature, use STE for hard branching.
            # The forward pass snaps to a hard 0/1 decision, but the
            # backward pass sees the original soft branch_prob, allowing
            # gradient flow to continue shaping the branch parameters.
            if (hard_branch_threshold is not None
                    and temperature < hard_branch_threshold):
                branch_taken = (branch_prob > 0.5).float()
                # STE: forward uses hard value, backward uses soft value
                branch_taken = branch_prob + (branch_taken - branch_prob).detach()
                pc = next_pc * (1.0 - branch_taken) + branch_pc * branch_taken
            else:
                # Blend normal advance and branch
                pc = next_pc * (1.0 - branch_prob) + branch_pc * branch_prob

            # Halt accumulation
            halt_prob = opcode_w[OPCODES["HALT"]]
            cumulative_halt = cumulative_halt + halt_prob * (1.0 - cumulative_halt)

            # Blend with pre-step registers based on halt status to freeze
            # registers after halt. Post-halt instructions must not corrupt output.
            regs = old_regs * cumulative_halt + regs * (1.0 - cumulative_halt)

            trace.append(regs.clone())

            if cumulative_halt > 0.99:
                break

        return ExecutionResult(
            registers=regs,
            flags=flags,
            steps_executed=step + 1,
            halted=cumulative_halt > 0.5,
            register_trace=trace,
        )

    # -- Batched soft program execution ------------------------------------

    def execute_soft_batched(
        self,
        program: SoftProgram,
        batch_inputs: list[dict[int, float]],
        max_steps: int = 32,
        temperature: float = 1.0,
        skip_bitwise: bool = False,
        hard_branch_threshold: Optional[float] = None,
        per_example_pc: bool = False,
    ) -> list[ExecutionResult]:
        """Execute a SoftProgram on multiple input sets simultaneously.

        Batches all examples into a single tensor computation:
        - Registers: [batch_size, num_registers] instead of [num_registers]
        - All soft_read/soft_write/ALU ops broadcast across batch dimension
        - Single backward pass computes gradients for all examples

        ~25x faster than sequential execute_soft for 20+ examples because
        the Python loop over examples is eliminated: all batch_size examples
        execute in parallel at every step.

        The program parameters (opcode_logits, dst_logits, etc.) are shared
        across the batch -- only register state differs per example. The PC
        is also shared (unless per_example_pc is True), with branch decisions
        averaged across the batch.

        Args:
            program: SoftProgram with learnable parameters (shared across batch).
            batch_inputs: List of {register_index: initial_value} dicts, one
                per example. len(batch_inputs) is the batch size.
            max_steps: Maximum execution steps.
            temperature: Gumbel-softmax temperature (anneal toward 0).
            skip_bitwise: Skip expensive bitwise ops for ~10x speedup.
            hard_branch_threshold: When not None and temperature is below
                this threshold, use STE for hard branch decisions (see
                execute_soft for details).
            per_example_pc: When True, each example in the batch maintains
                its own PC distribution [B, max_length] instead of a single
                shared PC [max_length]. This allows different examples to
                follow different branch paths, enabling synthesis of branching
                programs at the cost of O(batch * program_len) PC state.
                When False (default), the PC is shared and branch decisions
                are averaged across examples.

        Returns:
            List of ExecutionResult, one per input example.
        """
        batch_size = len(batch_inputs)

        # Initialize batched registers: [B, num_registers]
        regs = torch.zeros(batch_size, self.num_registers)
        for b, inputs in enumerate(batch_inputs):
            for idx, val in inputs.items():
                regs[b, idx] = float(val) if not isinstance(val, torch.Tensor) else val

        if per_example_pc:
            # Per-example PC: [B, max_length]
            pc = torch.zeros(batch_size, program.max_length)
            pc[:, 0] = 1.0
        else:
            # Shared soft PC (same program for all examples): [max_length]
            pc = torch.zeros(program.max_length)
            pc[0] = 1.0

        flags = torch.zeros(batch_size, 4)  # [B, 4] N, Z, C, V
        cumulative_halt = torch.zeros(batch_size)  # [B]
        step = 0

        for step in range(max_steps):
            old_regs = regs.clone()

            if per_example_pc:
                # With per-example PC, we fetch a soft instruction per example.
                # Average the per-example PC distributions to get shared
                # instruction weights (program is still shared), but branch
                # decisions use per-example flags for per-example PC updates.
                #
                # Instruction fetch uses the mean PC across examples so that
                # gradient flows to all program parameters uniformly, but each
                # example's PC evolves independently based on its own flags.
                avg_pc = pc.mean(dim=0)  # [max_length]
                (opcode_w, dst_w, src1_w, src2_w, immediate, branch_w) = \
                    program.get_soft_instruction(avg_pc, temperature)
            else:
                # Fetch instruction (shared across batch)
                (opcode_w, dst_w, src1_w, src2_w, immediate, branch_w) = \
                    program.get_soft_instruction(pc, temperature)

            # Batched soft read: [B, num_regs] @ [num_regs] -> [B]
            src1_val = (regs * src1_w.unsqueeze(0)).sum(dim=-1)  # [B]
            src2_val = (regs * src2_w.unsqueeze(0)).sum(dim=-1)  # [B]

            # Batched ALU: compute all ops for all examples at once
            # Each result value is [B]
            results: dict[str, torch.Tensor] = {}
            results["NOP"] = torch.zeros(batch_size)
            results["MOV_IMM"] = immediate.expand(batch_size)
            results["MOV_REG"] = src1_val
            results["ADD"] = src1_val + src2_val
            results["SUB"] = src1_val - src2_val
            results["MUL"] = src1_val * src2_val

            if not skip_bitwise:
                # Bitwise ops need per-element bit decomposition.
                # Reshape for the soft_alu functions which expect [batch, n_bits].
                from ncpu.coprocessor.soft_alu import (
                    soft_int_to_bits,
                    soft_bits_to_int,
                )
                bits_a = soft_int_to_bits(
                    src1_val.unsqueeze(-1), self.alu.n_bits, self.alu.bit_temperature
                )  # [B, n_bits]
                bits_b = soft_int_to_bits(
                    src2_val.unsqueeze(-1), self.alu.n_bits, self.alu.bit_temperature
                )  # [B, n_bits]

                and_bits = self.alu.soft_logical.forward_single_op(
                    bits_a, bits_b, op_idx=0
                )
                results["AND"] = soft_bits_to_int(and_bits).squeeze(-1)  # [B]

                or_bits = self.alu.soft_logical.forward_single_op(
                    bits_a, bits_b, op_idx=1
                )
                results["OR"] = soft_bits_to_int(or_bits).squeeze(-1)

                xor_bits = self.alu.soft_logical.forward_single_op(
                    bits_a, bits_b, op_idx=2
                )
                results["XOR"] = soft_bits_to_int(xor_bits).squeeze(-1)
            else:
                zero = torch.zeros(batch_size)
                results["AND"] = zero
                results["OR"] = zero
                results["XOR"] = zero

            # Non-result ops
            zero = torch.zeros(batch_size)
            for name in ("CMP", "BEQ", "BNE", "BGT", "HALT"):
                results[name] = zero

            # Weighted combination by opcode: [B]
            result = torch.zeros(batch_size)
            for op_name, op_idx in OPCODES.items():
                result = result + opcode_w[op_idx] * results[op_name]

            # Batched soft write
            write_ops = {
                "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL",
                "AND", "OR", "XOR",
            }
            write_enable = sum(opcode_w[OPCODES[op]] for op in write_ops)

            # dst_w: [num_regs], result: [B] -> update regs: [B, num_regs]
            w = dst_w.unsqueeze(0) * write_enable  # [1, num_regs] * scalar
            regs = regs * (1.0 - w) + result.unsqueeze(-1) * w  # [B, num_regs]

            # Clamp registers to prevent inf/NaN from MUL accumulation.
            # torch.clamp has well-defined gradients (identity in range, zero
            # outside), so gradient flow is preserved while preventing numerical
            # explosion that corrupts all subsequent computation.
            regs = regs.clamp(-1e6, 1e6)

            # Batched flag update
            cmp_weight = opcode_w[OPCODES["CMP"]]
            diff = src1_val - src2_val  # [B]
            n_flag = torch.sigmoid(-diff * (1.0 / self.flag_scale))
            z_flag = torch.exp(-(diff ** 2) / (2 * self.flag_scale ** 2))
            c_flag = torch.sigmoid(diff * (1.0 / self.flag_scale))
            v_flag = torch.zeros(batch_size)
            new_flags = torch.stack([n_flag, z_flag, c_flag, v_flag], dim=-1)  # [B, 4]
            flags = flags * (1.0 - cmp_weight) + new_flags * cmp_weight

            if per_example_pc:
                # Per-example PC update: each example follows its own path.
                # pc: [B, max_length], branch decisions use per-example flags.

                # Advance each example's PC by rolling right along the
                # instruction dimension (dim=1).
                next_pc = torch.roll(pc, 1, dims=1)  # [B, max_length]

                # Per-example branch probability using per-example flags [B]
                beq_prob = opcode_w[OPCODES["BEQ"]] * flags[:, 1]
                bne_prob = opcode_w[OPCODES["BNE"]] * (1.0 - flags[:, 1])
                bgt_prob = (opcode_w[OPCODES["BGT"]]
                            * (1.0 - flags[:, 0]) * (1.0 - flags[:, 1]))
                branch_prob = (beq_prob + bne_prob + bgt_prob).clamp(0.0, 1.0)
                # branch_prob: [B]

                # Branch target is the same for all examples (program is shared).
                # branch_w: [max_length] -> expand to [B, max_length]
                branch_pc = branch_w.unsqueeze(0).expand(batch_size, -1)

                # STE hard branching for per-example PC
                if (hard_branch_threshold is not None
                        and temperature < hard_branch_threshold):
                    branch_taken = (branch_prob > 0.5).float()
                    branch_taken = branch_prob + (branch_taken - branch_prob).detach()
                    # branch_taken: [B] -> [B, 1] for broadcasting
                    bt = branch_taken.unsqueeze(1)
                    pc = next_pc * (1.0 - bt) + branch_pc * bt
                else:
                    bp = branch_prob.unsqueeze(1)  # [B, 1]
                    pc = next_pc * (1.0 - bp) + branch_pc * bp
            else:
                # PC update (shared across batch, branch decisions averaged)
                next_pc = self._advance_pc(pc)

                # Average flags across batch for shared PC decision
                avg_z = flags[:, 1].mean()
                avg_n = flags[:, 0].mean()
                beq_prob = opcode_w[OPCODES["BEQ"]] * avg_z
                bne_prob = opcode_w[OPCODES["BNE"]] * (1.0 - avg_z)
                bgt_prob = opcode_w[OPCODES["BGT"]] * (1.0 - avg_n) * (1.0 - avg_z)
                branch_prob = (beq_prob + bne_prob + bgt_prob).clamp(0.0, 1.0)

                branch_pc = self._branch_pc(pc, branch_w)

                # STE hard branching for shared PC
                if (hard_branch_threshold is not None
                        and temperature < hard_branch_threshold):
                    branch_taken = (branch_prob > 0.5).float()
                    branch_taken = branch_prob + (branch_taken - branch_prob).detach()
                    pc = next_pc * (1.0 - branch_taken) + branch_pc * branch_taken
                else:
                    pc = next_pc * (1.0 - branch_prob) + branch_pc * branch_prob

            # Halt masking per example
            halt_prob = opcode_w[OPCODES["HALT"]]
            cumulative_halt = cumulative_halt + halt_prob * (1.0 - cumulative_halt)
            regs = (old_regs * cumulative_halt.unsqueeze(-1)
                    + regs * (1.0 - cumulative_halt.unsqueeze(-1)))

            if cumulative_halt.min() > 0.99:
                break

        # Unpack into list of ExecutionResult
        out: list[ExecutionResult] = []
        for b in range(batch_size):
            out.append(ExecutionResult(
                registers=regs[b],
                flags=flags[b],
                steps_executed=step + 1,
                halted=bool(cumulative_halt[b] > 0.5),
                register_trace=[],  # skip trace for batched mode (perf)
            ))
        return out

    # -- Convenience: build programs from text ---------------------------

    @staticmethod
    def assemble(text: str) -> FixedProgram:
        """Assemble a simple text program into a FixedProgram.

        Syntax (one instruction per line):
            MOV R0, #42        ; R0 = 42 (immediate)
            MOV R1, R0         ; R1 = R0 (register)
            ADD R2, R0, R1     ; R2 = R0 + R1
            SUB R3, R2, R0     ; R3 = R2 - R0
            MUL R4, R0, R1     ; R4 = R0 * R1
            AND R5, R0, R1     ; R5 = R0 & R1
            CMP R0, R1         ; compare R0, R1
            BEQ @5              ; branch to line 5 if equal
            BNE @3              ; branch to line 3 if not equal
            BGT @7              ; branch to line 7 if greater
            HALT
        """
        instructions = []
        for line in text.strip().split("\n"):
            line = line.split(";")[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            op = parts[0].upper()

            if op == "HALT":
                instructions.append(Instruction(OPCODES["HALT"]))
            elif op == "NOP":
                instructions.append(Instruction(OPCODES["NOP"]))
            elif op == "MOV":
                dst = int(parts[1][1:])
                if parts[2].startswith("#"):
                    imm = float(parts[2][1:])
                    instructions.append(
                        Instruction(OPCODES["MOV_IMM"], dst=dst, immediate=imm)
                    )
                else:
                    src = int(parts[2][1:])
                    instructions.append(
                        Instruction(OPCODES["MOV_REG"], dst=dst, src1=src)
                    )
            elif op in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
                dst = int(parts[1][1:])
                s1 = int(parts[2][1:])
                s2 = int(parts[3][1:])
                instructions.append(
                    Instruction(OPCODES[op], dst=dst, src1=s1, src2=s2)
                )
            elif op == "CMP":
                s1 = int(parts[1][1:])
                s2 = int(parts[2][1:])
                instructions.append(
                    Instruction(OPCODES["CMP"], src1=s1, src2=s2)
                )
            elif op in ("BEQ", "BNE", "BGT"):
                target = int(parts[1][1:])
                instructions.append(
                    Instruction(OPCODES[op], branch_target=target)
                )
            else:
                raise ValueError(f"Unknown instruction: {op}")

        return FixedProgram(instructions)
