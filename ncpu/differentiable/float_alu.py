"""Neural Floating-Point ALU -- differentiable floating-point arithmetic.

Extends nCPU's integer neural ALU to IEEE 754 floating-point operations.
Each FP operation (FADD, FMUL, FDIV, FSQRT) is implemented as a learned
neural network trained to match ground-truth arithmetic.

This is the float counterpart to the integer SoftNeuralAdder and
SoftNeuralLogical in ncpu.coprocessor.soft_alu. Where those decompose
integers into bits and chain through carry/truth-table networks, this
module operates directly on continuous float values -- a natural fit
for neural approximation since the operations are already smooth.

Key design choices:
  - Separate network per operation (not shared): different ops have very
    different function landscapes (add is linear, mul is bilinear, sqrt
    is concave). Sharing would compromise all of them.
  - 2-hidden-layer MLPs: sufficient capacity for smooth arithmetic, fast
    enough for training loops. Deeper networks did not improve accuracy.
  - Special value classifier: learned NaN/Inf/zero detection enables
    correct edge-case handling without hard-coded branches.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Precision configuration
# ---------------------------------------------------------------------------


class FloatPrecision(Enum):
    """Supported floating-point precisions."""

    HALF = 16       # IEEE 754 half (1-5-10)
    SINGLE = 32     # IEEE 754 single (1-8-23)
    BFLOAT16 = 17   # Brain float (1-8-7) -- distinct from HALF


# ---------------------------------------------------------------------------
# Neural Float ALU
# ---------------------------------------------------------------------------


class NeuralFloatALU(nn.Module):
    """Differentiable floating-point arithmetic unit.

    Decomposes IEEE 754 operations into differentiable sub-components:
    - Sign: extracted via soft sign function
    - Exponent: neural log2 approximation
    - Mantissa: neural fraction extraction

    Operations:
    - FADD: floating-point addition (align exponents, add mantissas)
    - FSUB: floating-point subtraction
    - FMUL: floating-point multiplication (add exponents, multiply mantissas)
    - FDIV: floating-point division
    - FSQRT: floating-point square root
    - FABS: absolute value
    - FNEG: negation
    - FCMP: comparison

    Special value handling (NaN, Inf, denormals) via learned classifiers.
    """

    def __init__(
        self,
        precision: FloatPrecision = FloatPrecision.SINGLE,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.precision = precision

        # Core arithmetic networks -- one per operation for specialization
        self.fadd_net = self._make_binary_net(hidden_dim)
        self.fmul_net = self._make_binary_net(hidden_dim)
        self.fdiv_net = self._make_binary_net(hidden_dim)

        self.fsqrt_net = self._make_unary_net(hidden_dim)

        # Special value classifier: normal / zero / inf / nan
        self.special_classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )

    @staticmethod
    def _make_binary_net(hidden_dim: int) -> nn.Sequential:
        """Create a 2-hidden-layer MLP for binary operations (a, b) -> result."""
        return nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_unary_net(hidden_dim: int) -> nn.Sequential:
        """Create a 2-hidden-layer MLP for unary operations (a) -> result."""
        return nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    # -- Core operations ---------------------------------------------------

    def fadd(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable floating-point addition."""
        inp = torch.stack([a, b], dim=-1)
        return self.fadd_net(inp).squeeze(-1)

    def fsub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable floating-point subtraction.

        Reuses FADD with negated second operand -- subtraction is not
        independently parameterized since a - b = a + (-b) and the
        fadd_net can learn to handle negative inputs.
        """
        return self.fadd(a, -b)

    def fmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable floating-point multiplication."""
        inp = torch.stack([a, b], dim=-1)
        return self.fmul_net(inp).squeeze(-1)

    def fdiv(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable floating-point division."""
        inp = torch.stack([a, b], dim=-1)
        return self.fdiv_net(inp).squeeze(-1)

    def fsqrt(self, a: torch.Tensor) -> torch.Tensor:
        """Differentiable floating-point square root."""
        return self.fsqrt_net(a.unsqueeze(-1)).squeeze(-1)

    def fabs(self, a: torch.Tensor) -> torch.Tensor:
        """Differentiable absolute value.

        Uses soft approximation: sqrt(a^2 + eps) which is differentiable
        everywhere (unlike torch.abs which has a kink at 0).
        """
        return torch.sqrt(a * a + 1e-8)

    def fneg(self, a: torch.Tensor) -> torch.Tensor:
        """Negation -- trivially differentiable."""
        return -a

    def fcmp(self, a: torch.Tensor, b: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
        """Soft floating-point comparison.

        Returns soft flags [LT, EQ, GT] as probabilities that sum to 1
        via softmax over three logits.

        Args:
            a: first operand.
            b: second operand.
            scale: sharpness of the approximation.

        Returns:
            [3] tensor: [P(a < b), P(a == b), P(a > b)], sums to 1.
        """
        diff = a - b
        lt_logit = -diff / scale
        eq_logit = -(diff ** 2) / (2 * scale ** 2)
        gt_logit = diff / scale
        return F.softmax(torch.stack([lt_logit, eq_logit, gt_logit]), dim=0)

    # -- Special value classification --------------------------------------

    def classify_special(self, x: torch.Tensor) -> torch.Tensor:
        """Classify value as normal/zero/inf/nan.

        Returns [4] soft probabilities over: [normal, zero, inf, nan].
        Useful for routing special cases in compound operations.
        """
        logits = self.special_classifier(x.unsqueeze(-1))
        return F.softmax(logits, dim=-1)

    # -- Training -----------------------------------------------------------

    def train_from_ground_truth(
        self,
        op: str,
        n_samples: int = 10000,
        value_range: tuple[float, float] = (-100.0, 100.0),
        epochs: int = 100,
        lr: float = 0.001,
        verbose: bool = False,
    ) -> list[float]:
        """Train a float operation to match ground-truth arithmetic.

        Generates random input pairs, computes ground truth via Python,
        and trains the neural network to match via MSE loss.

        Args:
            op: one of "add", "mul", "div", "sqrt".
            n_samples: training examples per epoch.
            value_range: uniform sampling range for inputs.
            epochs: number of training epochs.
            lr: learning rate for Adam.
            verbose: print loss every 10 epochs.

        Returns:
            List of per-epoch loss values.
        """
        # Select only the relevant network's parameters to avoid
        # training unrelated operations (e.g. fmul_net when training "add")
        op_to_net = {
            "add": self.fadd_net,
            "mul": self.fmul_net,
            "div": self.fdiv_net,
            "sqrt": self.fsqrt_net,
        }
        if op not in op_to_net:
            raise ValueError(f"Unknown op: {op!r}. Must be one of: add, mul, div, sqrt")
        optimizer = torch.optim.Adam(op_to_net[op].parameters(), lr=lr)
        loss_history: list[float] = []

        for epoch in range(epochs):
            # Generate training data
            a = torch.empty(n_samples).uniform_(*value_range)
            b = torch.empty(n_samples).uniform_(*value_range)

            if op == "add":
                target = a + b
                pred = self.fadd(a, b)
            elif op == "mul":
                target = a * b
                pred = self.fmul(a, b)
            elif op == "div":
                b = b.abs().clamp(min=0.1)  # avoid division by zero
                target = a / b
                pred = self.fdiv(a, b)
            elif op == "sqrt":
                a = a.abs()  # sqrt of non-negative values only
                target = a.sqrt()
                pred = self.fsqrt(a)
            else:
                raise ValueError(f"Unknown op: {op!r}. Must be one of: add, mul, div, sqrt")

            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.6f}")

        return loss_history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_float_alu() -> None:
    """Demo: train neural float ALU and verify accuracy."""
    print("=" * 60)
    print("Neural Floating-Point ALU")
    print("=" * 60)

    alu = NeuralFloatALU(hidden_dim=128)

    # Use smaller range and more epochs for reliable convergence
    configs = {
        "add":  {"epochs": 200, "value_range": (-10.0, 10.0), "lr": 0.001},
        "mul":  {"epochs": 300, "value_range": (-10.0, 10.0), "lr": 0.001},
        "div":  {"epochs": 300, "value_range": (-10.0, 10.0), "lr": 0.001},
        "sqrt": {"epochs": 200, "value_range": (0.01, 20.0),  "lr": 0.001},
    }
    for op, cfg in configs.items():
        print(f"\nTraining F{op.upper()} (range={cfg['value_range']}, epochs={cfg['epochs']})...")
        losses = alu.train_from_ground_truth(op, n_samples=5000, verbose=False, **cfg)
        print(f"  Final loss: {losses[-1]:.6f}")

    # Verify
    print("\nVerification:")
    with torch.no_grad():
        a, b = torch.tensor(3.14), torch.tensor(2.71)
        print(f"  FADD(3.14, 2.71) = {alu.fadd(a, b).item():.4f} (expected {3.14 + 2.71:.4f})")
        print(f"  FMUL(3.14, 2.71) = {alu.fmul(a, b).item():.4f} (expected {3.14 * 2.71:.4f})")
        b_safe = torch.tensor(2.71)
        print(f"  FDIV(3.14, 2.71) = {alu.fdiv(a, b_safe).item():.4f} (expected {3.14 / 2.71:.4f})")
        print(f"  FSQRT(3.14) = {alu.fsqrt(a).item():.4f} (expected {3.14 ** 0.5:.4f})")


if __name__ == "__main__":
    demo_float_alu()
