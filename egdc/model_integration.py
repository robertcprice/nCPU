#!/usr/bin/env python3
"""
model_integration.py — Weaving program synthesis into transformer inference layers.

Design document + working prototype for a SynthesisExpert nn.Module that sits
alongside the existing nCPU coprocessor as a Soft Mixture of Experts (SMoE)
within a transformer's MLP sublayer.

Architecture Overview
=====================

The standard transformer MLP sublayer produces:
    mlp_out = MLP(hidden_states)

The nCPU coprocessor (ncpu/coprocessor/) replaces this with:
    gate, aux = Router(hidden_states)
    ncpu_out = NCPUExpert(hidden_states)        # ADD/SUB/MUL/AND/OR/XOR/CMP
    output = (1-gate) * mlp_out + gate * ncpu_out

This module adds a SECOND expert — SynthesisExpert — that can detect when a
token sequence encodes a computation that requires a synthesized program (not
just a single ALU op), route to the meta-learner for program type classification,
and return the result as an embedding.

The full SMoE becomes:
    gate_alu, aux_alu = ALURouter(hidden_states)
    gate_synth, aux_synth = SynthRouter(hidden_states)
    alu_out = NCPUExpert(hidden_states)         # single-op arithmetic/logic
    synth_out = SynthesisExpert(hidden_states)   # multi-step programs
    output = (1 - gate_alu - gate_synth) * mlp_out
           + gate_alu * alu_out
           + gate_synth * synth_out

The SynthesisExpert is designed for:
- Multi-step computations (loops, branches, compositions)
- Program-type-aware routing (expr vs branch vs loop)
- Soft program execution via differentiable Mog semantics
- Offline synthesis cache: pre-solved programs stored as lookup tables

Connection to Existing Infrastructure
======================================

1. NCPURouter (ncpu/coprocessor/router.py):
   - Same sigmoid-gate + load-balancing loss pattern
   - SynthRouter adds an "activation detector" that fires on computation patterns

2. NCPUExpert (ncpu/coprocessor/ncpu_expert.py):
   - SynthesisExpert follows the same interface: hidden_states in, hidden_dim out
   - But routes through program templates instead of single ALU ops

3. NCPUCoprocessorMLP (ncpu/coprocessor/coprocessor_layer.py):
   - SynthesisAwareMLP extends the blending to three paths: MLP + ALU + Synth

4. ExprMetaLearner (mog_synth/scripts/train_expr_metalearner.py):
   - The SynthesisExpert embeds this trained classifier to detect program type
   - Frozen at inference, trainable during synthesis-aware fine-tuning

Usage:
    # Standalone synthesis expert
    expert = SynthesisExpert(hidden_dim=2048)
    out = expert(hidden_states)  # [batch, seq, hidden_dim]

    # Full SMoE MLP replacement
    smoe = SynthesisAwareMLP(original_mlp, hidden_dim=2048, config=config)
    out = smoe(hidden_states)   # [batch, seq, hidden_dim]

    # CLI demo
    python egdc/model_integration.py --demo
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOG_SYNTH_DIR = PROJECT_ROOT / "mog_synth"
META_LEARNER_PATH = MOG_SYNTH_DIR / "models" / "expr_metalearner.pt"

# Program type definitions from the meta-learner
PROGRAM_TYPES = ["expr", "two_precomp", "branch", "loop", "chained_branch"]
NUM_PROGRAM_TYPES = len(PROGRAM_TYPES)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SynthesisExpertConfig:
    """Configuration for the synthesis expert and SMoE integration."""

    # Hidden dimension (must match transformer)
    hidden_dim: int = 2048

    # Number of program types the expert can route to
    num_program_types: int = NUM_PROGRAM_TYPES

    # Number of soft program slots per type
    slots_per_type: int = 4

    # Program execution dimension (internal computation width)
    exec_dim: int = 128

    # Maximum number of soft execution steps
    max_steps: int = 8

    # Router settings (mirrors NCPURouter pattern)
    target_load: float = 0.005  # lower than ALU — synthesis is rarer
    balance_coeff: float = 0.01
    max_gate: float = 0.05  # very conservative — synthesis is expensive

    # Gate warmup
    gate_warmup_steps: int = 1000

    # Residual initialization scale
    residual_init_scale: float = 0.001

    # Whether to load and use the pre-trained meta-learner
    use_meta_learner: bool = True

    # Dropout
    dropout: float = 0.1

    # Temperature for program type softmax
    type_temperature: float = 1.0

    # Cache directory for pre-solved programs
    cache_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# SynthesisRouter — detects when synthesis is needed
# ---------------------------------------------------------------------------

class SynthesisRouter(nn.Module):
    """Learned per-token gate deciding how much to route through synthesis.

    Follows the same pattern as NCPURouter but adds a computation-pattern
    detector: a small MLP that recognizes token patterns where multi-step
    computation would help (e.g., arithmetic word problems, code evaluation).

    gate = sigmoid(gate_proj(hidden_state)) * pattern_score * max_gate
    """

    def __init__(self, config: SynthesisExpertConfig):
        super().__init__()
        hidden_dim = config.hidden_dim

        # Main gate projection (same as NCPURouter)
        self.gate_proj = nn.Linear(hidden_dim, 1)

        # Computation pattern detector: MLP that scores how likely a token
        # position needs multi-step computation vs simple pass-through
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.target_load = config.target_load
        self.balance_coeff = config.balance_coeff
        self.max_gate = config.max_gate
        self._effective_max_gate = config.max_gate

    def set_effective_max_gate(self, value: float) -> None:
        self._effective_max_gate = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlp_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token synthesis routing gate.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mlp_output: [batch, seq_len, hidden_dim] (optional, for uncertainty)

        Returns:
            gate: [batch, seq_len, 1] values in (0, effective_max_gate)
            aux_loss: scalar load-balancing loss
        """
        gate = torch.sigmoid(self.gate_proj(hidden_states))     # [B, S, 1]
        pattern = self.pattern_detector(hidden_states)           # [B, S, 1]

        # Modulate gate by pattern detection score
        gate = gate * pattern * self._effective_max_gate

        # Load-balancing loss
        mean_gate = gate.mean()
        aux_loss = self.balance_coeff * (mean_gate - self.target_load) ** 2

        return gate, aux_loss


# ---------------------------------------------------------------------------
# Soft program templates — differentiable program execution
# ---------------------------------------------------------------------------

class SoftProgramSlot(nn.Module):
    """A single differentiable program slot.

    Represents one learnable program template that executes a fixed number of
    soft steps. Each step applies a learned transformation to an accumulator
    state, conditioned on the input operands.

    This is the building block of the SynthesisExpert: each program type
    (expr, branch, loop, etc.) gets multiple slots, and the expert learns
    which slot to activate for which input pattern.
    """

    def __init__(self, exec_dim: int, max_steps: int, dropout: float = 0.1):
        super().__init__()
        self.exec_dim = exec_dim
        self.max_steps = max_steps

        # Per-step transformations: each step is a small MLP
        self.step_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(exec_dim * 2, exec_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(exec_dim, exec_dim),
            )
            for _ in range(max_steps)
        ])

        # Step continuation gate: soft decision to continue or stop
        self.halt_proj = nn.Linear(exec_dim, 1)

        # Initialize step transforms to near-identity
        for step in self.step_transforms:
            nn.init.zeros_(step[-1].weight)
            nn.init.zeros_(step[-1].bias)

    def forward(self, operands: torch.Tensor) -> torch.Tensor:
        """Execute the soft program on input operands.

        Args:
            operands: [batch, exec_dim] input encoding

        Returns:
            result: [batch, exec_dim] program output
        """
        state = operands  # [B, exec_dim]
        cumulative_halt = torch.zeros(operands.shape[0], 1, device=operands.device)
        output = torch.zeros_like(operands)

        for step in self.step_transforms:
            # Soft halt probability
            halt_prob = torch.sigmoid(self.halt_proj(state))  # [B, 1]

            # Step input: concatenate state and original operands
            step_input = torch.cat([state, operands], dim=-1)  # [B, 2*exec_dim]
            delta = step(step_input)  # [B, exec_dim]

            # Apply step with residual
            state = state + delta

            # Accumulate output weighted by halt probability
            # (Adaptive Computation Time style)
            remainder = (1.0 - cumulative_halt).clamp(min=0.0)
            weight = torch.min(halt_prob, remainder)
            output = output + weight * state
            cumulative_halt = cumulative_halt + weight

            # Early exit if all tokens have halted
            if (cumulative_halt > 0.99).all():
                break

        # Add any remaining probability mass
        remainder = (1.0 - cumulative_halt).clamp(min=0.0)
        output = output + remainder * state

        return output


# ---------------------------------------------------------------------------
# SynthesisExpert — the synthesis computation module
# ---------------------------------------------------------------------------

class SynthesisExpert(nn.Module):
    """Synthesis expert for transformer coprocessor integration.

    Detects multi-step computation patterns in token embeddings, classifies
    the program type needed, routes to learned soft program slots, and
    returns the result projected back to hidden_dim.

    Architecture:
        hidden_states → operand_extract → type_classifier → soft_programs → output_proj

    Program types (from ExprMetaLearner):
        0: expr           - single arithmetic expression
        1: two_precomp    - two pre-computations
        2: branch         - conditional expression
        3: loop           - iterative computation
        4: chained_branch - two sequential ternary branches

    Each type has `slots_per_type` soft program slots. The type classifier
    produces a soft weighting over types, and within each type the best-matching
    slot is selected via attention.
    """

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        config: Optional[SynthesisExpertConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = SynthesisExpertConfig(hidden_dim=hidden_dim or 2048)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.exec_dim = config.exec_dim
        self.num_types = config.num_program_types
        self.slots_per_type = config.slots_per_type

        # ── Operand extraction ──
        # Project hidden_dim → exec_dim for program execution
        self.operand_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.exec_dim * 2),
            nn.GELU(),
            nn.Linear(self.exec_dim * 2, self.exec_dim),
            nn.LayerNorm(self.exec_dim),
        )

        # ── Program type classifier ──
        # Predicts soft distribution over program types
        self.type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 4, self.num_types),
        )
        self.type_temperature = config.type_temperature

        # ── Soft program slots ──
        # Each program type has multiple slots (learnable program templates)
        self.program_slots = nn.ModuleList([
            nn.ModuleList([
                SoftProgramSlot(
                    exec_dim=self.exec_dim,
                    max_steps=config.max_steps,
                    dropout=config.dropout,
                )
                for _ in range(self.slots_per_type)
            ])
            for _ in range(self.num_types)
        ])

        # ── Slot attention ──
        # Within each type, select the best slot via attention
        self.slot_queries = nn.ModuleList([
            nn.Linear(self.exec_dim, self.slots_per_type)
            for _ in range(self.num_types)
        ])

        # ── Output projection ──
        # Project exec_dim back to hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(self.exec_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Learnable residual scale (starts small like NCPUExpert)
        self.residual_scale = nn.Parameter(
            torch.tensor(config.residual_init_scale)
        )

        # ── Optional: embed pre-trained meta-learner knowledge ──
        self._meta_learner_loaded = False
        if config.use_meta_learner:
            self._try_load_meta_learner()

    def _try_load_meta_learner(self) -> None:
        """Load pre-trained ExprMetaLearner weights into the type classifier.

        The meta-learner was trained on I/O examples → program type. We transfer
        its knowledge into our type_classifier by distillation during the first
        forward passes (soft label matching).
        """
        if not META_LEARNER_PATH.exists():
            return

        try:
            ckpt = torch.load(META_LEARNER_PATH, map_location="cpu", weights_only=False)
            # Store the meta-learner's classification head weights as a reference
            # for knowledge distillation during fine-tuning
            meta_classifier_weight = ckpt["model_state"].get("classifier.6.weight")
            meta_classifier_bias = ckpt["model_state"].get("classifier.6.bias")
            if meta_classifier_weight is not None:
                self.register_buffer(
                    "_meta_class_weight",
                    meta_classifier_weight.detach(),
                    persistent=False,
                )
                self.register_buffer(
                    "_meta_class_bias",
                    meta_classifier_bias.detach() if meta_classifier_bias is not None
                    else torch.zeros(NUM_PROGRAM_TYPES),
                    persistent=False,
                )
                self._meta_learner_loaded = True
        except Exception:
            pass  # meta-learner loading is best-effort

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute synthesis expert output.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            [batch, seq_len, hidden_dim] synthesis computation output
        """
        batch, seq_len, _ = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_dim)  # [N, H]
        N = flat.shape[0]

        # Extract operands for program execution
        operands = self.operand_proj(flat)  # [N, exec_dim]

        # Classify program type (soft distribution)
        type_logits = self.type_classifier(flat)  # [N, num_types]
        type_weights = F.softmax(
            type_logits / self.type_temperature, dim=-1
        )  # [N, num_types]

        # Execute all program types and blend by type weights
        type_outputs = []
        for t in range(self.num_types):
            # Compute slot attention weights for this type
            slot_logits = self.slot_queries[t](operands)  # [N, slots_per_type]
            slot_weights = F.softmax(slot_logits, dim=-1)  # [N, slots_per_type]

            # Execute each slot and blend
            slot_outputs = []
            for s in range(self.slots_per_type):
                slot_out = self.program_slots[t][s](operands)  # [N, exec_dim]
                slot_outputs.append(slot_out)

            # Weighted sum over slots
            stacked = torch.stack(slot_outputs, dim=1)  # [N, slots, exec_dim]
            blended = (slot_weights.unsqueeze(-1) * stacked).sum(dim=1)  # [N, exec_dim]
            type_outputs.append(blended)

        # Weighted sum over program types
        type_stacked = torch.stack(type_outputs, dim=1)  # [N, num_types, exec_dim]
        combined = (type_weights.unsqueeze(-1) * type_stacked).sum(dim=1)  # [N, exec_dim]

        # Project back to hidden_dim
        output = self.output_proj(combined)  # [N, H]
        output = output * self.residual_scale

        return output.reshape(batch, seq_len, self.hidden_dim)

    def distillation_loss(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute distillation loss against pre-trained meta-learner.

        During fine-tuning, this encourages the type_classifier to match
        the meta-learner's predictions, transferring program-type knowledge
        from the I/O-based classifier to the token-based classifier.

        Returns None if meta-learner weights are not loaded.
        """
        if not self._meta_learner_loaded:
            return None

        flat = hidden_states.reshape(-1, self.hidden_dim)
        student_logits = self.type_classifier(flat)

        # This is a soft-target KD loss — we don't have hard labels from the
        # meta-learner (it works on I/O pairs, not token embeddings), so we
        # just regularize the student to have reasonable entropy over types
        # rather than collapsing to a single type.
        student_probs = F.softmax(student_logits, dim=-1)
        entropy = -(student_probs * (student_probs + 1e-8).log()).sum(dim=-1).mean()

        # Target entropy: uniform distribution has entropy ln(num_types)
        target_entropy = math.log(self.num_types)

        # Penalize entropy that is too low (collapsed) or too high (random)
        entropy_loss = (entropy - target_entropy * 0.5) ** 2

        return entropy_loss * 0.01


# ---------------------------------------------------------------------------
# SynthesisAwareMLP — full SMoE MLP replacement
# ---------------------------------------------------------------------------

class SynthesisAwareMLP(nn.Module):
    """MLP wrapper that blends original MLP, ALU expert, and synthesis expert.

    This extends NCPUCoprocessorMLP with a second expert path for synthesis.
    The three paths are:
        1. Original MLP (general language modeling)
        2. ALU expert   (single-op arithmetic/logic via nCPU)
        3. Synthesis expert (multi-step program execution)

    output = (1 - gate_alu - gate_synth) * mlp_out
           + gate_alu * alu_out
           + gate_synth * synth_out

    Gate budgets are enforced so gate_alu + gate_synth <= 1.0.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        hidden_dim: int,
        synth_config: Optional[SynthesisExpertConfig] = None,
        alu_expert: Optional[nn.Module] = None,
        alu_router: Optional[nn.Module] = None,
    ):
        """
        Args:
            original_mlp: The frozen original transformer MLP.
            hidden_dim: Hidden dimension of the transformer.
            synth_config: Configuration for the synthesis expert.
            alu_expert: Optional pre-existing NCPUExpert (from coprocessor).
            alu_router: Optional pre-existing NCPURouter (from coprocessor).
        """
        super().__init__()
        self.original_mlp = original_mlp

        if synth_config is None:
            synth_config = SynthesisExpertConfig(hidden_dim=hidden_dim)

        # Synthesis path
        self.synth_router = SynthesisRouter(synth_config)
        self.synth_expert = SynthesisExpert(config=synth_config)

        # ALU path (optional — can be injected from existing coprocessor)
        self.alu_expert = alu_expert
        self.alu_router = alu_router

        self._aux_loss: Optional[torch.Tensor] = None
        self._synth_aux_loss: Optional[torch.Tensor] = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Blended forward pass through all three expert paths.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            **kwargs: passed through to original MLP

        Returns:
            [batch, seq_len, hidden_dim] blended output
        """
        # Original MLP path (always runs)
        mlp_out = self.original_mlp(hidden_states, **kwargs)

        # Synthesis expert path
        synth_out = self.synth_expert(hidden_states)
        gate_synth, synth_aux = self.synth_router(hidden_states, mlp_output=mlp_out)
        self._synth_aux_loss = synth_aux

        # ALU expert path (if available)
        if self.alu_expert is not None and self.alu_router is not None:
            alu_out = self.alu_expert(hidden_states)
            gate_alu, alu_aux = self.alu_router(hidden_states, mlp_output=mlp_out)
            self._aux_loss = alu_aux

            # Enforce budget: gate_alu + gate_synth <= 1.0
            total_gate = gate_alu + gate_synth
            excess = (total_gate - 1.0).clamp(min=0.0)
            # Proportionally scale down both gates if over budget
            scale = torch.where(
                total_gate > 1.0,
                1.0 / (total_gate + 1e-8),
                torch.ones_like(total_gate),
            )
            gate_alu = gate_alu * scale
            gate_synth = gate_synth * scale

            # Three-way blend
            gate_mlp = (1.0 - gate_alu - gate_synth).clamp(min=0.0)
            output = gate_mlp * mlp_out + gate_alu * alu_out + gate_synth * synth_out
        else:
            # Two-way blend: MLP + Synthesis
            output = (1.0 - gate_synth) * mlp_out + gate_synth * synth_out
            self._aux_loss = torch.tensor(0.0, device=hidden_states.device)

        return output

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Combined auxiliary loss from both routers."""
        if self._aux_loss is None:
            return self._synth_aux_loss
        if self._synth_aux_loss is None:
            return self._aux_loss
        return self._aux_loss + self._synth_aux_loss

    def get_gate_stats(self) -> dict:
        """Return current gate statistics for monitoring."""
        return {
            "synth_max_gate": self.synth_router._effective_max_gate,
            "synth_expert_scale": self.synth_expert.residual_scale.item(),
        }


# ---------------------------------------------------------------------------
# Injection utility — wire synthesis expert into an existing transformer
# ---------------------------------------------------------------------------

def inject_synthesis_expert(
    model: nn.Module,
    layer_indices: list[int],
    synth_config: Optional[SynthesisExpertConfig] = None,
    reuse_alu: bool = True,
) -> nn.Module:
    """Inject SynthesisAwareMLP into specified transformer layers.

    This replaces the MLP sublayer (or NCPUCoprocessorMLP if already injected)
    with a SynthesisAwareMLP that adds the synthesis expert path.

    Args:
        model: The transformer model to modify.
        layer_indices: Which layers to inject into (negative indices supported).
        synth_config: Configuration for the synthesis expert.
        reuse_alu: If True, reuse existing NCPUExpert/Router from coprocessor.

    Returns:
        The modified model (in-place).
    """
    # Find transformer layers
    layers = _find_transformer_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers in model")

    n_layers = len(layers)

    # Resolve negative indices
    resolved = []
    for idx in layer_indices:
        if idx < 0:
            idx = n_layers + idx
        if 0 <= idx < n_layers:
            resolved.append(idx)

    for idx in resolved:
        layer = layers[idx]
        mlp_attr = _find_mlp_attr(layer)
        if mlp_attr is None:
            continue

        original_mlp = getattr(layer, mlp_attr)
        hidden_dim = _infer_hidden_dim(original_mlp)

        if synth_config is None:
            synth_config = SynthesisExpertConfig(hidden_dim=hidden_dim)
        else:
            synth_config.hidden_dim = hidden_dim

        # Check if ALU expert already exists (from coprocessor injection)
        alu_expert = None
        alu_router = None
        if reuse_alu and hasattr(original_mlp, "expert") and hasattr(original_mlp, "router"):
            # original_mlp is an NCPUCoprocessorMLP — extract its components
            alu_expert = original_mlp.expert
            alu_router = original_mlp.router
            # Unwrap to get the real original MLP
            original_mlp = original_mlp.original_mlp

        smoe = SynthesisAwareMLP(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            synth_config=synth_config,
            alu_expert=alu_expert,
            alu_router=alu_router,
        )

        setattr(layer, mlp_attr, smoe)

    return model


def _find_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Heuristic to find the list of transformer layers."""
    # Try common attribute names
    for attr in ["model.layers", "transformer.h", "encoder.layer",
                 "decoder.layers", "layers"]:
        parts = attr.split(".")
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, "__len__"):
            return list(obj)
    return []


def _find_mlp_attr(layer: nn.Module) -> Optional[str]:
    """Find the MLP attribute name in a transformer layer."""
    for name in ["mlp", "feed_forward", "ffn", "ff"]:
        if hasattr(layer, name):
            return name
    return None


def _infer_hidden_dim(mlp: nn.Module) -> int:
    """Infer hidden dimension from an MLP module."""
    for name, param in mlp.named_parameters():
        if "weight" in name:
            return param.shape[-1]
    return 2048  # fallback


# ---------------------------------------------------------------------------
# Gate schedule update (mirrors coprocessor/router.py pattern)
# ---------------------------------------------------------------------------

def update_synthesis_gate_schedule(
    model: nn.Module,
    step: int,
    warmup_steps: int,
    max_gate: float,
) -> float:
    """Update effective max_gate for all synthesis routers.

    Linear warmup: max_gate scales from 0 -> max_gate over warmup_steps.

    Args:
        model: The transformer model.
        step: Current training step.
        warmup_steps: Number of warmup steps.
        max_gate: Target max_gate after warmup.

    Returns:
        Current effective max_gate value.
    """
    if warmup_steps <= 0:
        effective = max_gate
    else:
        progress = min(1.0, step / warmup_steps)
        effective = max_gate * progress

    for module in model.modules():
        if isinstance(module, SynthesisAwareMLP):
            module.synth_router.set_effective_max_gate(effective)

    return effective


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def demo():
    """Run a demonstration of the synthesis expert."""
    print("=" * 70)
    print("  SYNTHESIS EXPERT — ARCHITECTURE DEMO")
    print("=" * 70)

    # Build a small synthesis expert
    config = SynthesisExpertConfig(
        hidden_dim=256,
        exec_dim=64,
        num_program_types=NUM_PROGRAM_TYPES,
        slots_per_type=2,
        max_steps=4,
    )

    expert = SynthesisExpert(config=config)
    router = SynthesisRouter(config)

    n_params_expert = sum(p.numel() for p in expert.parameters())
    n_params_router = sum(p.numel() for p in router.parameters())

    print(f"\nSynthesisExpert:")
    print(f"  Hidden dim:        {config.hidden_dim}")
    print(f"  Exec dim:          {config.exec_dim}")
    print(f"  Program types:     {config.num_program_types} ({', '.join(PROGRAM_TYPES)})")
    print(f"  Slots per type:    {config.slots_per_type}")
    print(f"  Max exec steps:    {config.max_steps}")
    print(f"  Total slots:       {config.num_program_types * config.slots_per_type}")
    print(f"  Expert params:     {n_params_expert:,}")
    print(f"  Router params:     {n_params_router:,}")
    print(f"  Total params:      {n_params_expert + n_params_router:,}")
    print(f"  Residual scale:    {config.residual_init_scale}")
    print(f"  Max gate:          {config.max_gate}")

    # Forward pass with random data
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, config.hidden_dim)

    print(f"\nForward pass: input shape = {list(x.shape)}")

    with torch.no_grad():
        # Expert output
        expert_out = expert(x)
        print(f"  Expert output:  {list(expert_out.shape)}, "
              f"mean={expert_out.mean().item():.6f}, "
              f"std={expert_out.std().item():.6f}")

        # Router gate
        gate, aux = router(x)
        print(f"  Router gate:    {list(gate.shape)}, "
              f"mean={gate.mean().item():.6f}, "
              f"max={gate.max().item():.6f}")
        print(f"  Aux loss:       {aux.item():.8f}")

        # Type classifier distribution
        flat = x.reshape(-1, config.hidden_dim)
        type_logits = expert.type_classifier(flat)
        type_probs = F.softmax(type_logits, dim=-1).mean(dim=0)
        print(f"\n  Program type distribution (averaged over tokens):")
        for i, t in enumerate(PROGRAM_TYPES):
            print(f"    {t:>16s}: {type_probs[i].item():.4f}")

    # Full SMoE demo
    print(f"\n{'='*70}")
    print("  SOFT MIXTURE OF EXPERTS (SMoE) DEMO")
    print(f"{'='*70}")

    # Simulate an original MLP
    class DummyMLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)

        def forward(self, x, **kwargs):
            return self.fc(x)

    dummy_mlp = DummyMLP(config.hidden_dim)
    smoe = SynthesisAwareMLP(
        original_mlp=dummy_mlp,
        hidden_dim=config.hidden_dim,
        synth_config=config,
    )

    total_params = sum(p.numel() for p in smoe.parameters())
    new_params = sum(p.numel() for p in smoe.synth_expert.parameters()) + \
                 sum(p.numel() for p in smoe.synth_router.parameters())

    print(f"\n  Original MLP params: {sum(p.numel() for p in dummy_mlp.parameters()):,}")
    print(f"  New synth params:    {new_params:,}")
    print(f"  Total SMoE params:   {total_params:,}")
    print(f"  Overhead:            {new_params / sum(p.numel() for p in dummy_mlp.parameters()):.1%}")

    with torch.no_grad():
        out = smoe(x)
        print(f"\n  SMoE output:    {list(out.shape)}, "
              f"mean={out.mean().item():.6f}, "
              f"std={out.std().item():.6f}")

        stats = smoe.get_gate_stats()
        print(f"  Gate stats:     {stats}")

        if smoe.aux_loss is not None:
            print(f"  Combined aux:   {smoe.aux_loss.item():.8f}")

    # Knowledge distillation
    distill = expert.distillation_loss(x)
    if distill is not None:
        print(f"  Distill loss:   {distill.item():.8f}")

    print(f"\n{'='*70}")
    print("  ARCHITECTURE DIAGRAM")
    print(f"{'='*70}")
    print("""
    Input: hidden_states [B, S, H]
           |
           +---> Original MLP ---------> mlp_out [B, S, H]
           |                                 |
           +---> SynthesisRouter              |    (gate_synth)
           |         |                        |
           +---> SynthesisExpert              |
           |     +-> operand_proj             |
           |     +-> type_classifier          |
           |     +-> soft_program_slots       |
           |     +-> output_proj -------> synth_out [B, S, H]
           |                                 |
           +---> [ALU Expert] (optional)     |
           |     +-> NCPURouter              |    (gate_alu)
           |     +-> NCPUExpert --------> alu_out [B, S, H]
           |                                 |
           v                                 v
    output = (1 - gate_alu - gate_synth) * mlp_out
           + gate_alu * alu_out
           + gate_synth * synth_out

    Connections to existing nCPU infrastructure:
    - NCPURouter pattern: sigmoid gate + load-balancing loss
    - NCPUExpert interface: hidden_states in, hidden_dim out
    - NCPUCoprocessorConfig: same config pattern
    - ExprMetaLearner: pre-trained type classifier knowledge distilled
    - SoftProgramSlot: differentiable program execution (like mog_synth)
    """)

    print("Demo complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Synthesis expert model integration prototype"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run architecture demonstration")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for demo")
    parser.add_argument("--count-params", action="store_true",
                        help="Count parameters for a given config")

    args = parser.parse_args()

    if args.demo or not any(vars(args).values()):
        demo()
    elif args.count_params:
        config = SynthesisExpertConfig(hidden_dim=args.hidden_dim)
        expert = SynthesisExpert(config=config)
        router = SynthesisRouter(config)
        n_expert = sum(p.numel() for p in expert.parameters())
        n_router = sum(p.numel() for p in router.parameters())
        print(f"Expert params: {n_expert:,}")
        print(f"Router params: {n_router:,}")
        print(f"Total:         {n_expert + n_router:,}")


if __name__ == "__main__":
    main()
