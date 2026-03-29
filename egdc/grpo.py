"""Coupled-GRPO (Group Relative Policy Optimization) for EGDC.

Implements RL fine-tuning of masked diffusion models following the DiffuCoder
paper's coupled-GRPO approach:
  - Group K samples per spec, compute advantages from rewards
  - Complementary mask pairs for variance reduction
  - KL penalty to prevent drift from base model
  - Reward = 2.0*pass_rate + 0.5*syntax_valid + 0.3*halts_cleanly
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (
    MaskedDiffusionTransformer, ModelConfig,
    MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE,
)
from .tokenizer import (
    NCPUTokenizer, OPCODES, NUM_OPCODES,
    REG_OFFSET, NUM_REGISTERS, IMM_OFFSET, NUM_IMMEDIATES,
    BR_OFFSET, NUM_BRANCH_TARGETS, OPCODE_OFFSET,
)
from .sampler import generate, build_slot_masks
from .evaluate import execute_program
from .data_generator import NCPUDataGenerator


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def check_syntax_valid(tokens: List[int]) -> bool:
    """Check if all tokens are valid ISA tokens in the correct slots.

    Each instruction is 4 tokens: [opcode, dst_reg, src_reg, imm/branch].
    """
    clean = [t for t in tokens if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)]
    if len(clean) == 0:
        return False

    # Must be multiple of 4
    if len(clean) % 4 != 0:
        return False

    for i in range(0, len(clean), 4):
        opcode, dst, src, imm = clean[i:i + 4]
        # Opcode must be in range 0..NUM_OPCODES-1
        if not (OPCODE_OFFSET <= opcode < OPCODE_OFFSET + NUM_OPCODES):
            return False
        # dst and src must be registers
        if not (REG_OFFSET <= dst < REG_OFFSET + NUM_REGISTERS):
            return False
        if not (REG_OFFSET <= src < REG_OFFSET + NUM_REGISTERS):
            return False
        # imm must be immediate or branch target
        if not ((IMM_OFFSET <= imm < IMM_OFFSET + NUM_IMMEDIATES) or
                (BR_OFFSET <= imm < BR_OFFSET + NUM_BRANCH_TARGETS)):
            return False

    return True


def check_halts_cleanly(tokens: List[int]) -> bool:
    """Check if the program executes and halts (doesn't loop/crash)."""
    # Try with zero inputs
    result = execute_program(tokens, {})
    return result is not None


def compute_reward(tokens: List[int], spec: Dict[str, Any]) -> float:
    """Compute reward for a generated program against its spec.

    reward = 2.0 * pass_rate + 0.5 * syntax_valid + 0.3 * halts_cleanly
    """
    # Syntax validity
    syntax_valid = 1.0 if check_syntax_valid(tokens) else 0.0

    # Halts cleanly
    halts = 1.0 if check_halts_cleanly(tokens) else 0.0

    # Pass rate: fraction of test cases passed
    test_cases = spec.get("test_cases", [])
    if not test_cases:
        pass_rate = 0.0
    else:
        passed = 0
        for tc in test_cases:
            inputs = tc.get("inputs", {})
            expected = tc.get("expected_output")

            # Map input names to register indices
            reg_map = {}
            for i, (name, val) in enumerate(inputs.items()):
                reg_map[i] = val

            result = execute_program(tokens, reg_map)
            if result is not None:
                actual = result.get(0, None)
                if actual == expected:
                    passed += 1

        pass_rate = passed / len(test_cases)

    return 2.0 * pass_rate + 0.5 * syntax_valid + 0.3 * halts


def encode_spec_tokens(spec: Dict[str, Any], spec_len: int = 32) -> List[int]:
    """Encode a spec dict into a fixed-length token sequence for conditioning.

    Mirrors NCPUDataset._encode_spec().

    Format: [name_hint_opcodes(4)] [test_case_io_values...] [PAD...]
    """
    from .dataset import NCPUDataset

    tokens: List[int] = []

    # Encode program type as opcode hints (first 4 slots)
    name = spec.get("name", "")
    hints = NCPUDataset._NAME_TO_HINT.get(name, [])
    for h in hints[:4]:
        tokens.append(h)
    while len(tokens) < 4:
        tokens.append(PAD_TOKEN)

    # Encode test cases
    test_cases = spec.get("test_cases", [])[:4]
    for tc in test_cases:
        inputs = tc.get("inputs", {})
        expected = tc.get("expected_output", 0)

        input_vals = [v for _, v in sorted(inputs.items())]
        for v in input_vals[:3]:
            val = max(0, min(255, int(v) % 256))
            tokens.append(IMM_OFFSET + val)

        # Pad inputs to 3
        while (len(tokens) - 4) % 4 != 3:
            tokens.append(PAD_TOKEN)

        out_val = max(0, min(255, int(expected) % 256))
        tokens.append(IMM_OFFSET + out_val)

    # Pad to spec_len
    while len(tokens) < spec_len:
        tokens.append(PAD_TOKEN)

    return tokens[:spec_len]


# ---------------------------------------------------------------------------
# Complementary mask pair generation (DiffuCoder's key insight)
# ---------------------------------------------------------------------------

def generate_complementary_masks(
    seq_len: int,
    mask_rate: float,
    exclude_special: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a pair of complementary masks for variance reduction.

    For each code position (excluding special tokens):
      mask1 | mask2 = all_positions (union covers everything)
      mask1 & mask2 = empty (no overlap)

    Args:
        seq_len: length of the sequence
        mask_rate: fraction of positions to mask in mask1 (mask2 gets the rest)
        exclude_special: (seq_len,) bool tensor, True = don't mask this position

    Returns:
        mask1, mask2: (seq_len,) bool tensors
    """
    # Determine maskable positions
    maskable = torch.ones(seq_len, dtype=torch.bool)
    if exclude_special is not None:
        maskable = maskable & ~exclude_special

    maskable_indices = torch.where(maskable)[0]
    num_maskable = len(maskable_indices)

    if num_maskable == 0:
        return torch.zeros(seq_len, dtype=torch.bool), torch.zeros(seq_len, dtype=torch.bool)

    # Randomly partition maskable positions into two complementary sets
    perm = torch.randperm(num_maskable)
    num_mask1 = max(1, int(num_maskable * mask_rate))
    num_mask1 = min(num_mask1, num_maskable - 1)  # ensure mask2 gets at least 1

    mask1 = torch.zeros(seq_len, dtype=torch.bool)
    mask2 = torch.zeros(seq_len, dtype=torch.bool)

    mask1[maskable_indices[perm[:num_mask1]]] = True
    mask2[maskable_indices[perm[num_mask1:]]] = True

    return mask1, mask2


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    num_samples_per_spec: int = 8     # K samples per spec
    num_specs_per_batch: int = 4      # Number of specs per batch
    num_epochs: int = 10
    lr: float = 1e-5
    kl_weight: float = 0.1           # KL penalty weight
    mask_rate: float = 0.5           # Mask rate for coupled masks
    seq_len: int = 128               # Program sequence length
    spec_len: int = 32               # Spec sequence length
    num_sampling_steps: int = 64     # Diffusion sampling steps
    temperature: float = 0.7         # Sampling temperature
    max_grad_norm: float = 1.0       # Gradient clipping
    advantage_eps: float = 1e-8      # Epsilon for advantage normalization
    log_interval: int = 1            # Log every N batches


class GRPOTrainer:
    """Coupled-GRPO trainer for masked diffusion models.

    Implements the DiffuCoder paper's approach:
    1. Sample K programs per spec from the current policy
    2. Score each program via execution (reward function)
    3. Compute group-relative advantages
    4. For each program, create complementary mask pairs
    5. Compute GRPO loss with KL penalty and update
    """

    def __init__(
        self,
        model: MaskedDiffusionTransformer,
        config: GRPOConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Frozen copy of the base model for KL penalty
        self.base_model = copy.deepcopy(model)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=0.01
        )

        # Data generator for specs
        self.data_gen = NCPUDataGenerator()
        self.tokenizer = NCPUTokenizer()

        # Slot masks for constrained sampling
        self.slot_masks = build_slot_masks(config.seq_len).to(device)

        # Stats tracking
        self.stats: Dict[str, List[float]] = {
            "loss": [], "reward": [], "kl": [], "advantage": [],
        }

    def _encode_spec(self, spec: Dict[str, Any]) -> torch.Tensor:
        """Encode a spec to conditioning tokens."""
        tokens = encode_spec_tokens(spec, self.config.spec_len)
        return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _sample_programs(
        self, spec_tokens: torch.Tensor, k: int
    ) -> List[torch.Tensor]:
        """Sample K programs from the current model for a given spec.

        Returns list of (1, seq_len) tensors.
        """
        self.model.eval()
        programs = []
        for _ in range(k):
            tokens = generate(
                self.model,
                spec_tokens,
                seq_len=self.config.seq_len,
                num_steps=self.config.num_sampling_steps,
                temperature=self.config.temperature,
                device=self.device,
                constrained=True,
            )
            programs.append(tokens)
        return programs

    def _compute_log_probs(
        self,
        model: MaskedDiffusionTransformer,
        program: torch.Tensor,
        mask: torch.Tensor,
        spec_tokens: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Compute log probabilities of unmasked tokens at masked positions.

        This is the diffusion model's log P(x_unmasked | x_masked, t, spec).

        Args:
            model: the diffusion model
            program: (1, L) original program tokens
            mask: (L,) bool tensor, True = masked positions
            spec_tokens: (1, S) spec conditioning tokens
            timestep: diffusion timestep (mask_rate serves as proxy)

        Returns:
            scalar: sum of log probs at masked positions
        """
        L = program.shape[1]

        # Create masked input
        masked_input = program.clone()
        masked_input[0, mask] = MASK_TOKEN

        # Forward pass
        t_tensor = torch.tensor([timestep], device=self.device)
        logits = model(masked_input, t_tensor, spec_tokens=spec_tokens)  # (1, L, V)

        # Get log probs at masked positions
        log_probs = F.log_softmax(logits, dim=-1)  # (1, L, V)

        # Gather log probs of the actual tokens at masked positions
        target_tokens = program[0, mask]  # (num_masked,)
        position_log_probs = log_probs[0, mask]  # (num_masked, V)
        token_log_probs = position_log_probs.gather(
            1, target_tokens.unsqueeze(1)
        ).squeeze(1)  # (num_masked,)

        return token_log_probs.sum()

    def _compute_kl_penalty(
        self,
        program: torch.Tensor,
        mask: torch.Tensor,
        spec_tokens: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        """Compute KL(policy || base) at masked positions.

        KL = sum_pos sum_tok P_policy(tok|pos) * [log P_policy - log P_base]
        """
        L = program.shape[1]
        masked_input = program.clone()
        masked_input[0, mask] = MASK_TOKEN

        t_tensor = torch.tensor([timestep], device=self.device)

        # Policy logits
        policy_logits = self.model(masked_input, t_tensor, spec_tokens=spec_tokens)
        policy_log_probs = F.log_softmax(policy_logits[0, mask], dim=-1)

        # Base model logits
        with torch.no_grad():
            base_logits = self.base_model(masked_input, t_tensor, spec_tokens=spec_tokens)
            base_log_probs = F.log_softmax(base_logits[0, mask], dim=-1)

        # KL divergence per masked position
        kl = F.kl_div(base_log_probs, policy_log_probs.exp(), reduction="sum", log_target=False)

        return kl

    def _grpo_step(
        self, specs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """One GRPO training step over a batch of specs.

        For each spec:
          1. Sample K programs
          2. Compute rewards
          3. Compute group-relative advantages
          4. For each program, create complementary mask pair
          5. Compute loss = -sum(A_i * log_prob_i) + kl_weight * KL

        Returns dict of scalar metrics.
        """
        cfg = self.config
        total_loss = torch.tensor(0.0, device=self.device)
        total_reward = 0.0
        total_kl = 0.0
        total_pairs = 0

        for spec in specs:
            spec_tokens = self._encode_spec(spec)

            # Step 1: Sample K programs
            programs = self._sample_programs(spec_tokens, cfg.num_samples_per_spec)

            # Step 2: Compute rewards
            rewards = []
            for prog in programs:
                prog_list = prog[0].tolist()
                r = compute_reward(prog_list, spec)
                rewards.append(r)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            total_reward += rewards_tensor.mean().item()

            # Step 3: Group-relative advantages
            if rewards_tensor.std() < cfg.advantage_eps:
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - rewards_tensor.mean()) / (
                    rewards_tensor.std() + cfg.advantage_eps
                )

            # Step 4 & 5: Complementary mask pairs + GRPO loss
            self.model.train()
            spec_loss = torch.tensor(0.0, device=self.device)
            spec_kl = torch.tensor(0.0, device=self.device)

            for i, prog in enumerate(programs):
                adv = advantages[i].item()

                # Skip zero-advantage samples (no learning signal)
                if abs(adv) < 1e-6:
                    continue

                # Identify special token positions to exclude from masking
                prog_tokens = prog[0]
                special_mask = (
                    (prog_tokens == PAD_TOKEN) |
                    (prog_tokens == BOS_TOKEN) |
                    (prog_tokens == EOS_TOKEN) |
                    (prog_tokens == MASK_TOKEN)
                )

                # Generate complementary mask pair
                mask1, mask2 = generate_complementary_masks(
                    cfg.seq_len, cfg.mask_rate, exclude_special=special_mask
                )
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)

                timestep = cfg.mask_rate  # Use mask rate as timestep proxy

                # Compute log probs for both masks
                for mask in [mask1, mask2]:
                    if mask.sum() == 0:
                        continue

                    log_prob = self._compute_log_probs(
                        self.model, prog, mask, spec_tokens, timestep
                    )

                    # GRPO: -advantage * log_prob
                    spec_loss = spec_loss + (-adv * log_prob)

                    # KL penalty
                    kl = self._compute_kl_penalty(prog, mask, spec_tokens, timestep)
                    spec_kl = spec_kl + kl
                    total_pairs += 1

            if total_pairs > 0:
                total_loss = total_loss + spec_loss + cfg.kl_weight * spec_kl
                total_kl += spec_kl.item()

        # Normalize by number of specs and pairs
        num_specs = len(specs)
        if total_pairs > 0:
            total_loss = total_loss / total_pairs

        # Backward + update
        self.optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "reward": total_reward / max(num_specs, 1),
            "kl": total_kl / max(total_pairs, 1),
            "num_pairs": total_pairs,
        }

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Run GRPO training loop.

        Returns dict of metric histories.
        """
        cfg = self.config
        epochs = num_epochs or cfg.num_epochs

        print(f"Starting coupled-GRPO training")
        print(f"  Epochs: {epochs}")
        print(f"  Specs/batch: {cfg.num_specs_per_batch}")
        print(f"  Samples/spec (K): {cfg.num_samples_per_spec}")
        print(f"  Learning rate: {cfg.lr}")
        print(f"  KL weight: {cfg.kl_weight}")
        print(f"  Mask rate: {cfg.mask_rate}")
        print(f"  Device: {self.device}")
        print()

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_kl = 0.0
            num_batches = 0

            # Generate fresh specs for each epoch
            dataset = self.data_gen.generate_dataset(
                cfg.num_specs_per_batch, balanced=True
            )
            specs = [spec for spec, _tokens in dataset]

            # Process specs in mini-batches
            for batch_start in range(0, len(specs), cfg.num_specs_per_batch):
                batch_specs = specs[batch_start:batch_start + cfg.num_specs_per_batch]

                metrics = self._grpo_step(batch_specs)

                epoch_loss += metrics["loss"]
                epoch_reward += metrics["reward"]
                epoch_kl += metrics["kl"]
                num_batches += 1

                if num_batches % cfg.log_interval == 0:
                    print(
                        f"  [Epoch {epoch+1}/{epochs} Batch {num_batches}] "
                        f"loss={metrics['loss']:.4f} "
                        f"reward={metrics['reward']:.3f} "
                        f"kl={metrics['kl']:.4f} "
                        f"pairs={metrics['num_pairs']}"
                    )

            # Epoch summary
            elapsed = time.time() - epoch_start
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_reward = epoch_reward / max(num_batches, 1)
            avg_kl = epoch_kl / max(num_batches, 1)

            self.stats["loss"].append(avg_loss)
            self.stats["reward"].append(avg_reward)
            self.stats["kl"].append(avg_kl)

            print(
                f"Epoch {epoch+1}/{epochs} complete in {elapsed:.1f}s: "
                f"loss={avg_loss:.4f} reward={avg_reward:.3f} kl={avg_kl:.4f}"
            )

        return self.stats

    def save_checkpoint(self, path: str) -> None:
        """Save the fine-tuned model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved GRPO checkpoint to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def train_grpo(
    checkpoint: str,
    model_size: str = "tiny",
    epochs: int = 10,
    lr: float = 1e-5,
    kl_weight: float = 0.1,
    num_specs_per_batch: int = 4,
    num_samples_per_spec: int = 8,
    mask_rate: float = 0.5,
    temperature: float = 0.7,
    num_sampling_steps: int = 64,
    output_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """Load a pretrained diffusion model and fine-tune with coupled-GRPO.

    Args:
        checkpoint: Path to pretrained model checkpoint.
        model_size: Model config size (tiny/small/medium).
        epochs: Number of training epochs.
        lr: Learning rate.
        kl_weight: Weight for KL penalty.
        num_specs_per_batch: Number of specs per training batch.
        num_samples_per_spec: K samples per spec.
        mask_rate: Mask rate for complementary mask pairs.
        temperature: Sampling temperature.
        num_sampling_steps: Number of diffusion sampling steps.
        output_path: Where to save the fine-tuned model.

    Returns:
        Training statistics dict.
    """
    device = torch.device("cpu")

    # Build model config
    if model_size == "tiny":
        cfg = ModelConfig.tiny()
    elif model_size == "small":
        cfg = ModelConfig.small()
    else:
        cfg = ModelConfig.medium()

    # Load pretrained model
    model = MaskedDiffusionTransformer(cfg)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained model from {checkpoint}")
    print(f"  Config: {model_size} ({model.count_parameters():,} params)")

    # GRPO config
    grpo_config = GRPOConfig(
        num_samples_per_spec=num_samples_per_spec,
        num_specs_per_batch=num_specs_per_batch,
        num_epochs=epochs,
        lr=lr,
        kl_weight=kl_weight,
        mask_rate=mask_rate,
        temperature=temperature,
        num_sampling_steps=num_sampling_steps,
    )

    # Train
    trainer = GRPOTrainer(model, grpo_config, device)
    stats = trainer.train()

    # Save
    if output_path is None:
        output_path = checkpoint.replace(".pt", "_grpo.pt")
    trainer.save_checkpoint(output_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Coupled-GRPO training for EGDC masked diffusion model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--model_size", choices=["tiny", "small", "medium"], default="tiny",
        help="Model configuration size"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL penalty weight")
    parser.add_argument(
        "--num_specs_per_batch", type=int, default=4,
        help="Number of specs per batch"
    )
    parser.add_argument(
        "--num_samples_per_spec", type=int, default=8,
        help="K samples per spec (group size)"
    )
    parser.add_argument("--mask_rate", type=float, default=0.5, help="Complementary mask rate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--num_sampling_steps", type=int, default=64,
        help="Diffusion sampling steps"
    )
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")

    args = parser.parse_args()

    train_grpo(
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        num_specs_per_batch=args.num_specs_per_batch,
        num_samples_per_spec=args.num_samples_per_spec,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        num_sampling_steps=args.num_sampling_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
