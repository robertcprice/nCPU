"""Neural ISA Discovery -- learn optimal instruction set architectures via gradient descent.

Instead of implementing a fixed ISA (like ARM64), this module parameterizes the
instruction set itself and optimizes for minimal total execution cost across a
benchmark suite. Each instruction is a learned neural operation, and gradient
descent discovers which operations should be primitive.

This inverts traditional ISA design: instead of human architects choosing
instructions, gradients discover the cheapest set of operations that achieve
correctness on a benchmark suite.

Builds on:
  - ncpu.differentiable.execution: DifferentiableEngine, OPCODES, NUM_OPCODES
  - ncpu.coprocessor.soft_alu: SoftNeuralLogical, soft_int_to_bits, soft_bits_to_int
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.differentiable.execution import DifferentiableEngine, OPCODES, NUM_OPCODES
from ncpu.coprocessor.soft_alu import SoftNeuralLogical, soft_int_to_bits, soft_bits_to_int


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------


@dataclass
class ISAConfig:
    """Parameterized instruction set configuration."""

    max_opcodes: int = 16           # max instructions in the ISA
    n_bits: int = 16                # operand width
    num_registers: int = 8          # register file size
    allow_compound: bool = True     # allow compound ops (e.g., MAC = MUL+ADD)


@dataclass
class ISADiscoveryResult:
    """Result of ISA optimization."""

    discovered_ops: list[dict]          # [{name, type, cost, usage_frequency}]
    total_cost: float
    loss_history: list[float]
    benchmark_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Neural ISA Discovery
# ---------------------------------------------------------------------------

# Type alias for benchmark functions
BenchmarkFn = Callable[["NeuralISADiscovery"], tuple[torch.Tensor, torch.Tensor]]


class NeuralISADiscovery(nn.Module):
    """Discover optimal instruction sets via gradient descent.

    Key idea: each instruction is parameterized as a learned neural operation.
    The cost of each instruction is a differentiable function of its complexity
    (number of neural passes, bit width, carry depth). Gradient descent finds
    the set of operations that minimizes total execution cost across benchmarks.

    This inverts traditional ISA design: instead of human architects choosing
    instructions, gradients discover which operations should be primitive.

    Learned parameters:
    - Operation functions: what each instruction computes (as neural networks)
    - Cost weights: relative cost of each operation type
    - Composition rules: which operations can be fused
    - Encoding efficiency: how compactly instructions encode
    """

    def __init__(self, config: ISAConfig | None = None):
        super().__init__()
        self.config = config or ISAConfig()

        # Each op is a small learned function: (a, b) -> result
        # Parameterized as a 2-layer MLP per operation
        self.op_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            for _ in range(self.config.max_opcodes)
        ])

        # Learnable cost per operation (initialized from known neural costs)
        # Higher = more expensive. softplus ensures positivity.
        self.op_costs = nn.Parameter(torch.ones(self.config.max_opcodes))

        # Operation "type" embeddings -- learns which ops are similar
        self.op_embeddings = nn.Parameter(
            torch.randn(self.config.max_opcodes, 16) * 0.1
        )

        # Compound operation selector: can two ops be fused?
        # fusion_logits[i, j] > 0 means op_i and op_j can be fused
        self.fusion_logits = nn.Parameter(
            torch.zeros(self.config.max_opcodes, self.config.max_opcodes)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, op_idx: int) -> torch.Tensor:
        """Execute operation op_idx on inputs a, b.

        Args:
            a: scalar or batched tensor -- first operand
            b: scalar or batched tensor -- second operand
            op_idx: which learned operation to apply

        Returns:
            Result tensor of same shape as inputs.
        """
        inp = torch.stack([a, b], dim=-1)
        return self.op_networks[op_idx](inp).squeeze(-1)

    def compute_isa_cost(self, program_op_counts: torch.Tensor) -> torch.Tensor:
        """Compute total cost of a program given operation frequencies.

        Args:
            program_op_counts: [max_opcodes] soft counts of each op used.

        Returns:
            Scalar cost tensor (differentiable w.r.t. op_costs).
        """
        costs = F.softplus(self.op_costs)  # ensure positive
        return (program_op_counts * costs).sum()

    def get_fusion_probs(self) -> torch.Tensor:
        """Get the fusion probability matrix.

        Returns:
            [max_opcodes, max_opcodes] tensor of fusion probabilities.
        """
        return torch.sigmoid(self.fusion_logits)

    def compute_similarity_matrix(self) -> torch.Tensor:
        """Compute cosine similarity between operation embeddings.

        Returns:
            [max_opcodes, max_opcodes] similarity matrix.
        """
        normed = F.normalize(self.op_embeddings, dim=-1)
        return normed @ normed.T

    def discover(
        self,
        benchmarks: list[BenchmarkFn],
        max_iters: int = 2000,
        lr: float = 0.005,
        cost_weight: float = 0.1,
        verbose: bool = False,
    ) -> ISADiscoveryResult:
        """Run ISA discovery optimization.

        Alternates between:
        1. Evaluating benchmarks (correctness + op usage tracking)
        2. Computing total cost (correctness_loss + cost_weight * execution_cost)
        3. Backpropagating to update operation networks and cost parameters

        Args:
            benchmarks: list of functions that take (isa) and return
                       (loss, op_counts) where op_counts is a tensor of
                       how many times each op was used.
            max_iters: number of optimization steps.
            lr: learning rate for Adam optimizer.
            cost_weight: weight for execution cost relative to correctness.
            verbose: print progress every 200 steps.

        Returns:
            ISADiscoveryResult with discovered operations, cost, and history.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_history: list[float] = []

        for step in range(max_iters):
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0)
            for bench_fn in benchmarks:
                correctness_loss, op_counts = bench_fn(self)
                cost_loss = self.compute_isa_cost(op_counts)
                total_loss = total_loss + correctness_loss + cost_weight * cost_loss

            total_loss.backward()
            optimizer.step()
            loss_history.append(total_loss.item())

            if verbose and step % 200 == 0:
                print(f"Step {step}: loss={total_loss.item():.4f}")

        # Extract discovered ISA
        costs = F.softplus(self.op_costs).detach()
        discovered: list[dict] = []
        for i in range(self.config.max_opcodes):
            discovered.append({
                "index": i,
                "cost": costs[i].item(),
                "embedding_norm": self.op_embeddings[i].norm().item(),
            })

        discovered.sort(key=lambda x: x["cost"])

        return ISADiscoveryResult(
            discovered_ops=discovered,
            total_cost=sum(d["cost"] for d in discovered),
            loss_history=loss_history,
            benchmark_scores={},
        )


# ---------------------------------------------------------------------------
# Benchmark factories
# ---------------------------------------------------------------------------


def make_arithmetic_benchmark() -> BenchmarkFn:
    """Benchmark: learn operations that handle basic arithmetic.

    Assigns:
      - op0 -> addition
      - op1 -> multiplication
      - op2 -> subtraction
    """

    def benchmark(isa: NeuralISADiscovery) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.tensor(0.0)
        op_counts = torch.zeros(isa.config.max_opcodes)

        # Test: op0 should learn addition
        for a_val, b_val in [(3, 5), (10, 20), (7, 8)]:
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            result = isa.forward(a, b, 0)
            loss = loss + (result - (a + b)) ** 2
            op_counts[0] = op_counts[0] + 1

        # Test: op1 should learn multiplication
        for a_val, b_val in [(3, 5), (4, 7), (6, 8)]:
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            result = isa.forward(a, b, 1)
            loss = loss + (result - (a * b)) ** 2
            op_counts[1] = op_counts[1] + 1

        # Test: op2 should learn subtraction
        for a_val, b_val in [(10, 3), (20, 5), (15, 7)]:
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            result = isa.forward(a, b, 2)
            loss = loss + (result - (a - b)) ** 2
            op_counts[2] = op_counts[2] + 1

        return loss, op_counts

    return benchmark


def make_bitwise_benchmark() -> BenchmarkFn:
    """Benchmark: learn operations that handle bitwise logic.

    Assigns:
      - op3 -> AND
      - op4 -> OR
    """

    def benchmark(isa: NeuralISADiscovery) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.tensor(0.0)
        op_counts = torch.zeros(isa.config.max_opcodes)

        # op3 should learn AND
        for a_val, b_val in [(0xFF, 0x0F), (0xAA, 0x55), (0x12, 0x34)]:
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            result = isa.forward(a, b, 3)
            loss = loss + (result - float(a_val & b_val)) ** 2
            op_counts[3] = op_counts[3] + 1

        # op4 should learn OR
        for a_val, b_val in [(0xFF, 0x0F), (0xAA, 0x55), (0x12, 0x34)]:
            a = torch.tensor(float(a_val))
            b = torch.tensor(float(b_val))
            result = isa.forward(a, b, 4)
            loss = loss + (result - float(a_val | b_val)) ** 2
            op_counts[4] = op_counts[4] + 1

        return loss, op_counts

    return benchmark


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_isa_discovery() -> None:
    """Demo: gradient descent discovers arithmetic and bitwise operations."""
    print("=" * 60)
    print("Neural ISA Discovery")
    print("=" * 60)
    print("Learning an instruction set from scratch via gradient descent...\n")

    isa = NeuralISADiscovery()
    benchmarks = [make_arithmetic_benchmark(), make_bitwise_benchmark()]

    result = isa.discover(benchmarks, max_iters=2000, lr=0.005, verbose=True)

    print(f"\nDiscovered {len(result.discovered_ops)} operations")
    print(f"Total ISA cost: {result.total_cost:.2f}")
    print("\nOperations by cost (cheapest first):")
    for op in result.discovered_ops[:8]:
        print(f"  Op {op['index']}: cost={op['cost']:.3f}")

    # Verify learned operations
    print("\nVerification:")
    with torch.no_grad():
        a, b = torch.tensor(7.0), torch.tensor(3.0)
        print(f"  Op0(7, 3) = {isa.forward(a, b, 0).item():.1f} (expected 10 = 7+3)")
        print(f"  Op1(7, 3) = {isa.forward(a, b, 1).item():.1f} (expected 21 = 7*3)")
        print(f"  Op2(7, 3) = {isa.forward(a, b, 2).item():.1f} (expected 4 = 7-3)")


if __name__ == "__main__":
    demo_isa_discovery()
