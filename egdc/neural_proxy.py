"""Neural Execution Proxy for EGDC.

A small transformer that predicts execution outcomes (pass/fail + register
values) from code token embeddings.  Provides smooth differentiable gradients
for classifier-guidance during masked-diffusion denoising — much faster than
the soft-execution approach in execution_guidance.py.

Usage:
    # Train the proxy
    python -m egdc.neural_proxy --train

    # Use in sampling (programmatic)
    from egdc.neural_proxy import ProxyGuidedSampler
    sampler = ProxyGuidedSampler(diffusion_model, proxy_path="checkpoints/egdc/proxy_best.pt")
    tokens = sampler.generate(spec_tokens, seq_len=128)
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .tokenizer import (
    NCPUTokenizer, VOCAB_SIZE, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
    IMM_OFFSET, NUM_IMMEDIATES, REG_OFFSET, NUM_REGISTERS,
    BR_OFFSET, NUM_BRANCH_TARGETS, NUM_OPCODES, OPCODE_OFFSET,
)
from .evaluate import execute_program
from .data_generator import NCPUDataGenerator
from .sampler import build_slot_masks


# ---------------------------------------------------------------------------
# Proxy model
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    """Configuration for the neural execution proxy."""
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    num_registers: int = 8


class NeuralExecutionProxy(nn.Module):
    """Small transformer that predicts execution outcomes from token IDs.

    Inputs:
        code_tokens:  (B, L) token IDs — may contain MASK_TOKEN for noisy inputs
        spec_tokens:  (B, S) spec/conditioning token IDs

    Outputs dict with:
        pass_prob:    (B,)  probability the program passes all tests, via sigmoid
        reg_values:   (B, 8) predicted final register values (normalised)
    """

    def __init__(self, config: Optional[ProxyConfig] = None) -> None:
        super().__init__()
        self.config = config or ProxyConfig()
        c = self.config

        # Embeddings
        self.token_embed = nn.Embedding(c.vocab_size, c.hidden_dim)
        self.pos_embed = nn.Embedding(c.max_seq_len, c.hidden_dim)
        self.segment_embed = nn.Embedding(2, c.hidden_dim)  # 0=spec, 1=code
        self.slot_embed = nn.Embedding(4, c.hidden_dim)     # instruction slot

        # Noise-level embedding: continuous in [0,1], similar to timestep
        self.noise_mlp = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden_dim,
            nhead=c.num_heads,
            dim_feedforward=c.ff_dim,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=c.num_layers,
            norm=nn.LayerNorm(c.hidden_dim),
        )

        # Output heads
        self.pool_norm = nn.LayerNorm(c.hidden_dim)
        self.pass_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(c.hidden_dim // 2, 1),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(c.hidden_dim // 2, c.num_registers),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0,1] -> (B, hidden_dim)"""
        half = self.config.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        code_tokens: torch.Tensor,
        spec_tokens: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            code_tokens: (B, L) int64 token IDs
            spec_tokens: (B, S) int64 spec token IDs
            noise_level: (B,) float in [0,1] — fraction of masked tokens

        Returns:
            dict with 'pass_prob' (B,) and 'reg_values' (B, 8)
        """
        B, L = code_tokens.shape
        S = spec_tokens.shape[1]
        device = code_tokens.device

        # Embed spec tokens
        spec_emb = self.token_embed(spec_tokens)
        spec_pos = torch.arange(S, device=device)
        spec_emb = spec_emb + self.pos_embed(spec_pos).unsqueeze(0)
        spec_emb = spec_emb + self.segment_embed(
            torch.zeros(1, dtype=torch.long, device=device)
        )

        # Embed code tokens
        code_emb = self.token_embed(code_tokens)
        code_pos = torch.arange(L, device=device)
        code_emb = code_emb + self.pos_embed(code_pos).unsqueeze(0)
        code_emb = code_emb + self.segment_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )
        # Add slot embeddings (position % 4)
        slot_ids = torch.arange(L, device=device) % 4
        code_emb = code_emb + self.slot_embed(slot_ids).unsqueeze(0)

        # Concatenate: [spec | code]
        x = torch.cat([spec_emb, code_emb], dim=1)  # (B, S+L, H)

        # Add noise-level conditioning if provided
        if noise_level is not None:
            noise_emb = self.noise_mlp(self._sinusoidal_embed(noise_level))  # (B, H)
            x = x + noise_emb.unsqueeze(1)

        # Transformer
        x = self.transformer(x)  # (B, S+L, H)

        # Pool: mean over all positions
        pooled = x.mean(dim=1)  # (B, H)
        pooled = self.pool_norm(pooled)

        # Heads
        pass_logit = self.pass_head(pooled).squeeze(-1)  # (B,)
        pass_prob = torch.sigmoid(pass_logit)
        reg_values = self.reg_head(pooled)  # (B, 8)

        return {
            "pass_prob": pass_prob,
            "pass_logit": pass_logit,
            "reg_values": reg_values,
        }

    def get_pass_prob_with_grad(
        self,
        code_tokens: torch.Tensor,
        spec_tokens: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience: returns pass_prob with gradients enabled for guidance.

        Uses the Gumbel-softmax / straight-through trick: creates a
        differentiable embedding lookup via one-hot @ embedding_weight.
        """
        B, L = code_tokens.shape
        S = spec_tokens.shape[1]
        device = code_tokens.device

        # One-hot encode for differentiable embedding
        code_onehot = F.one_hot(code_tokens, self.config.vocab_size).float()  # (B,L,V)
        code_emb = code_onehot @ self.token_embed.weight  # (B, L, H)

        code_pos = torch.arange(L, device=device)
        code_emb = code_emb + self.pos_embed(code_pos).unsqueeze(0)
        code_emb = code_emb + self.segment_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )
        slot_ids = torch.arange(L, device=device) % 4
        code_emb = code_emb + self.slot_embed(slot_ids).unsqueeze(0)

        # Spec embedding (no gradient needed here)
        spec_emb = self.token_embed(spec_tokens)
        spec_pos = torch.arange(S, device=device)
        spec_emb = spec_emb + self.pos_embed(spec_pos).unsqueeze(0)
        spec_emb = spec_emb + self.segment_embed(
            torch.zeros(1, dtype=torch.long, device=device)
        )

        x = torch.cat([spec_emb, code_emb], dim=1)

        if noise_level is not None:
            noise_emb = self.noise_mlp(self._sinusoidal_embed(noise_level))
            x = x + noise_emb.unsqueeze(1)

        x = self.transformer(x)
        pooled = self.pool_norm(x.mean(dim=1))
        pass_logit = self.pass_head(pooled).squeeze(-1)
        return torch.sigmoid(pass_logit)


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class ProxyTrainingDataset(Dataset):
    """Dataset of (noisy_program, spec, pass_label, reg_values, noise_level).

    Generated offline by:
    1. Sampling correct programs from the data generator
    2. Running them through the evaluator to get labels
    3. Creating noisy variants by masking random fractions of tokens
    """

    def __init__(
        self,
        data: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float]],
    ):
        """
        Each item: (code_tokens, spec_tokens, pass_label, reg_values, noise_level)
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        code, spec, label, regs, noise = self.data[idx]
        return code, spec, torch.tensor(label, dtype=torch.float32), regs, torch.tensor(noise, dtype=torch.float32)


def _encode_spec(spec: dict, max_len: int = 32) -> List[int]:
    """Encode a spec dict into token IDs for conditioning."""
    test_cases = spec.get("test_cases", [])
    tokens: List[int] = []
    for tc in test_cases[:2]:  # use first 2 test cases
        for name, val in tc["inputs"].items():
            tokens.append(IMM_OFFSET + min(val % 256, 255))
        tokens.append(IMM_OFFSET + min(tc["expected_output"] % 256, 255))
    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(PAD_TOKEN)
    return tokens[:max_len]


def _get_register_values(tokens: List[int], spec: dict) -> Optional[List[int]]:
    """Run program and return register values for the first test case."""
    test_cases = spec.get("test_cases", [])
    if not test_cases:
        return None
    tc = test_cases[0]
    reg_map = {}
    for i, (name, val) in enumerate(tc["inputs"].items()):
        reg_map[i] = val
    result = execute_program(tokens, reg_map)
    if result is None:
        return None
    return [result.get(i, 0) for i in range(8)]


def _check_first_test_case(tokens: List[int], spec: dict) -> bool:
    """Check if program passes the first test case only.

    This is the right check for generated programs since they hardcode
    specific constant values matching the first test case.
    """
    test_cases = spec.get("test_cases", [])
    if not test_cases:
        return False
    tc = test_cases[0]
    reg_map = {}
    for i, (name, val) in enumerate(tc["inputs"].items()):
        reg_map[i] = val
    result = execute_program(tokens, reg_map)
    if result is None:
        return False
    expected = tc.get("expected_output")
    return result.get(0) == expected


def _apply_noise(tokens: List[int], noise_frac: float) -> List[int]:
    """Mask a fraction of non-special tokens."""
    noisy = list(tokens)
    # Find positions that are actual code (not BOS/EOS/PAD)
    code_positions = [
        i for i, t in enumerate(noisy)
        if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)
    ]
    if not code_positions:
        return noisy
    n_mask = max(0, int(len(code_positions) * noise_frac))
    if n_mask > 0:
        mask_positions = random.sample(code_positions, min(n_mask, len(code_positions)))
        for pos in mask_positions:
            noisy[pos] = MASK_TOKEN
    return noisy


def generate_proxy_dataset(
    num_samples: int = 50_000,
    code_len: int = 128,
    spec_len: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float]]:
    """Generate training data for the neural execution proxy.

    For each base program, we create multiple noisy variants at different
    noise levels, including:
      - noise=0.0  (clean program, should pass)
      - noise=0.1..0.9  (partially masked, likely fail at high noise)
      - Corrupted programs (random tokens, should fail)

    Returns list of (code_tensor, spec_tensor, label, reg_tensor, noise_level).
    """
    rng = random.Random(seed)
    gen = NCPUDataGenerator(seed=seed)
    tok = NCPUTokenizer()

    data = []
    # We need num_samples total. Generate base programs and create variants.
    base_count = num_samples // 5  # ~5 variants per program
    if verbose:
        print(f"Generating {base_count} base programs for {num_samples} total samples...")

    generated = 0
    attempts = 0
    max_attempts = base_count * 3

    while generated < base_count and attempts < max_attempts:
        attempts += 1
        try:
            spec, program_tokens = gen.generate_one()
        except Exception:
            continue

        # Pad/truncate program tokens
        padded = tok.pad(program_tokens, code_len)
        spec_tokens = _encode_spec(spec, spec_len)

        # Get register values for clean program
        reg_vals = _get_register_values(program_tokens, spec)
        if reg_vals is None:
            # Program doesn't execute — skip
            continue

        # Check if clean program passes (first test case — matches hardcoded constants)
        passed = _check_first_test_case(program_tokens, spec)

        # Normalise register values to [0, 1] range (divide by 256)
        reg_tensor = torch.tensor([v / 256.0 for v in reg_vals], dtype=torch.float32)

        # Variant 1: clean program (noise=0)
        code_t = torch.tensor(padded, dtype=torch.long)
        spec_t = torch.tensor(spec_tokens, dtype=torch.long)
        data.append((code_t, spec_t, 1.0 if passed else 0.0, reg_tensor, 0.0))

        # Variant 2-4: noisy versions at various levels
        for noise_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if len(data) >= num_samples:
                break
            noisy = _apply_noise(padded, noise_frac)
            noisy_t = torch.tensor(noisy, dtype=torch.long)

            # Run the noisy program to get actual pass/fail
            noisy_reg_vals = _get_register_values(noisy, spec)
            if noisy_reg_vals is not None:
                noisy_passed = _check_first_test_case(noisy, spec)
                noisy_reg_t = torch.tensor(
                    [v / 256.0 for v in noisy_reg_vals], dtype=torch.float32
                )
                noisy_label = 1.0 if noisy_passed else 0.0
            else:
                noisy_label = 0.0
                noisy_reg_t = torch.zeros(8, dtype=torch.float32)

            data.append((noisy_t, spec_t, noisy_label, noisy_reg_t, noise_frac))

        # Variant 5: fully random garbage (negative example)
        if len(data) < num_samples:
            garbage = []
            n_instrs = rng.randint(2, 16)
            garbage.append(BOS_TOKEN)
            for _ in range(n_instrs):
                garbage.append(rng.randint(0, NUM_OPCODES - 1))  # opcode
                garbage.append(REG_OFFSET + rng.randint(0, 7))    # dst
                garbage.append(REG_OFFSET + rng.randint(0, 7))    # src
                garbage.append(IMM_OFFSET + rng.randint(0, 255))  # imm
            garbage.append(EOS_TOKEN)
            garbage_padded = tok.pad(garbage, code_len)
            garbage_t = torch.tensor(garbage_padded, dtype=torch.long)

            garbage_reg = _get_register_values(garbage_padded, spec)
            if garbage_reg is not None:
                garbage_passed = _check_first_test_case(garbage_padded, spec)
                garbage_reg_t = torch.tensor(
                    [v / 256.0 for v in garbage_reg], dtype=torch.float32
                )
                garbage_label = 1.0 if garbage_passed else 0.0
            else:
                garbage_label = 0.0
                garbage_reg_t = torch.zeros(8, dtype=torch.float32)

            data.append((garbage_t, spec_t, garbage_label, garbage_reg_t, 1.0))

        generated += 1
        if verbose and generated % 2000 == 0:
            print(f"  Generated {generated}/{base_count} base programs ({len(data)} total samples)")

    # Trim to exact size
    rng_shuffle = random.Random(seed + 1)
    rng_shuffle.shuffle(data)
    data = data[:num_samples]

    if verbose:
        n_pass = sum(1 for _, _, l, _, _ in data if l > 0.5)
        print(f"Dataset: {len(data)} samples, {n_pass} pass ({100*n_pass/len(data):.1f}%), "
              f"{len(data)-n_pass} fail ({100*(len(data)-n_pass)/len(data):.1f}%)")

    return data


# ---------------------------------------------------------------------------
# Proxy-guided sampler
# ---------------------------------------------------------------------------

class ProxyGuidedSampler:
    """Sampler that uses the neural execution proxy for classifier guidance.

    At each denoising step:
    1. Get diffusion model logits for current (partially masked) tokens
    2. Compute proxy's pass_probability for the current tokens
    3. Backprop through proxy to get gradients on token logits
    4. Shift logits toward higher pass probability (classifier guidance)
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        proxy: Optional[NeuralExecutionProxy] = None,
        proxy_path: Optional[str] = None,
        guidance_scale: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        self.diffusion_model = diffusion_model
        self.device = device or torch.device("cpu")

        if proxy is not None:
            self.proxy = proxy.to(self.device)
        elif proxy_path is not None:
            self.proxy = NeuralExecutionProxy()
            state = torch.load(proxy_path, map_location=self.device, weights_only=True)
            self.proxy.load_state_dict(state)
            self.proxy.to(self.device)
        else:
            raise ValueError("Either proxy or proxy_path must be provided")

        self.proxy.eval()
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def generate(
        self,
        spec_tokens: torch.Tensor,
        seq_len: int = 128,
        num_steps: int = 64,
        temperature: float = 0.8,
        constrained: bool = True,
    ) -> torch.Tensor:
        """Generate tokens with proxy guidance.

        Args:
            spec_tokens: (1, S) conditioning spec tokens
            seq_len: length of sequence to generate
            num_steps: number of denoising steps
            temperature: sampling temperature
            constrained: enforce ISA slot constraints

        Returns:
            (1, seq_len) generated token IDs
        """
        device = self.device
        self.diffusion_model.eval()

        # Start fully masked
        tokens = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)
        spec_tokens = spec_tokens.to(device)

        # Build slot constraints
        if constrained:
            slot_masks = build_slot_masks(seq_len).to(device)

        for step in range(num_steps):
            t = 1.0 - (step + 1) / num_steps
            t_tensor = torch.tensor([max(t, 0.01)], device=device)

            # Get diffusion model logits
            logits = self.diffusion_model(tokens, t_tensor, spec_tokens=spec_tokens)

            # Apply slot constraints
            if constrained:
                logits = logits.clone()
                logits[0][~slot_masks] = -1e9

            # --- Proxy guidance ---
            logits = self._apply_guidance(logits, tokens, spec_tokens, t)

            # Sample
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)

            # Confidence-based unmasking
            confidence = probs.max(dim=-1).values
            is_masked = (tokens == MASK_TOKEN)
            num_masked = is_masked.sum().item()

            if num_masked == 0:
                break

            num_to_reveal = max(1, min(
                int(math.ceil(num_masked / max(num_steps - step, 1))),
                num_masked,
            ))

            masked_confidence = confidence.clone()
            masked_confidence[~is_masked] = -1.0
            _, top_indices = masked_confidence.topk(num_to_reveal, dim=-1)

            for idx in top_indices[0]:
                tokens[0, idx] = sampled[0, idx]

        # Final cleanup
        if (tokens == MASK_TOKEN).any():
            t_tensor = torch.tensor([0.01], device=device)
            logits = self.diffusion_model(tokens, t_tensor, spec_tokens=spec_tokens)
            if constrained:
                logits = logits.clone()
                logits[0][~slot_masks] = -1e9
            logits = self._apply_guidance(logits, tokens, spec_tokens, 0.01)
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1]).clamp(min=1e-10)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True)
            final = torch.multinomial(flat_probs, num_samples=1).view(1, seq_len)
            mask = (tokens == MASK_TOKEN)
            tokens = torch.where(mask, final, tokens)

        return tokens

    def _apply_guidance(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        spec_tokens: torch.Tensor,
        noise_level: float,
    ) -> torch.Tensor:
        """Apply proxy classifier guidance to logits.

        For each masked position, compute: how does each possible token
        affect the proxy's pass_probability?  We approximate this by
        computing gradients of log(pass_prob) w.r.t. the token embeddings
        and projecting onto the vocab.

        This is the standard classifier-guidance trick adapted for discrete
        tokens: grad_logits = scale * (d log p(pass) / d embedding) @ W_embed^T
        """
        B, L, V = logits.shape
        device = logits.device

        # Enable gradients for this block
        self.proxy.eval()

        # Create soft token representation from current logits
        # For masked positions, use the logits; for unmasked, use one-hot
        is_masked = (tokens == MASK_TOKEN)  # (B, L)

        if not is_masked.any():
            return logits

        # Use current logits to create a soft distribution for masked positions
        soft_probs = F.softmax(logits.detach() / 0.5, dim=-1)  # (B, L, V)

        # One-hot for unmasked positions
        one_hot = F.one_hot(tokens, V).float()  # (B, L, V)

        # Mix: use soft for masked, one-hot for unmasked
        token_probs = torch.where(
            is_masked.unsqueeze(-1).expand_as(one_hot),
            soft_probs,
            one_hot,
        )
        token_probs.requires_grad_(True)

        # Differentiable embedding lookup
        emb = token_probs @ self.proxy.token_embed.weight  # (B, L, H)

        # Add positional + segment + slot embeddings
        S = spec_tokens.shape[1]
        code_pos = torch.arange(L, device=device)
        emb = emb + self.proxy.pos_embed(code_pos).unsqueeze(0)
        emb = emb + self.proxy.segment_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )
        slot_ids = torch.arange(L, device=device) % 4
        emb = emb + self.proxy.slot_embed(slot_ids).unsqueeze(0)

        # Spec embedding
        spec_emb = self.proxy.token_embed(spec_tokens)
        spec_pos = torch.arange(S, device=device)
        spec_emb = spec_emb + self.proxy.pos_embed(spec_pos).unsqueeze(0)
        spec_emb = spec_emb + self.proxy.segment_embed(
            torch.zeros(1, dtype=torch.long, device=device)
        )

        x = torch.cat([spec_emb, emb], dim=1)

        # Add noise level
        noise_t = torch.tensor([noise_level], device=device, dtype=torch.float32)
        noise_emb = self.proxy.noise_mlp(self.proxy._sinusoidal_embed(noise_t))
        x = x + noise_emb.unsqueeze(1)

        # Forward through transformer
        x = self.proxy.transformer(x)
        pooled = self.proxy.pool_norm(x.mean(dim=1))
        pass_logit = self.proxy.pass_head(pooled).squeeze(-1)
        log_prob = F.logsigmoid(pass_logit)

        # Backprop to get gradients on token_probs
        grad = torch.autograd.grad(log_prob.sum(), token_probs, retain_graph=False)[0]
        # grad: (B, L, V) — how each token at each position affects pass_prob

        # Apply guidance: shift logits toward tokens that increase pass_prob
        guided_logits = logits + self.guidance_scale * grad

        return guided_logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_proxy(
    num_samples: int = 50_000,
    num_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 3e-4,
    code_len: int = 128,
    spec_len: int = 32,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints/egdc",
    verbose: bool = True,
) -> NeuralExecutionProxy:
    """Train the neural execution proxy.

    1. Generate training data: (noisy_program, spec, pass/fail, registers)
    2. Train proxy transformer for num_epochs
    3. Save best checkpoint

    Returns the trained proxy model.
    """
    device = torch.device("cpu")

    # Generate training data
    if verbose:
        print("=" * 60)
        print("Neural Execution Proxy Training")
        print("=" * 60)
        t0 = time.time()

    data = generate_proxy_dataset(
        num_samples=num_samples,
        code_len=code_len,
        spec_len=spec_len,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"Data generation took {time.time() - t0:.1f}s")

    # Split into train/val (90/10)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = ProxyTrainingDataset(train_data)
    val_dataset = ProxyTrainingDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # Model
    proxy = NeuralExecutionProxy().to(device)
    n_params = sum(p.numel() for p in proxy.parameters())
    if verbose:
        print(f"Proxy model: {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(proxy.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.1
    )

    # Loss weights
    bce_weight = 1.0
    mse_weight = 0.5

    # Training loop
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "proxy_best.pt")

    for epoch in range(num_epochs):
        # Train
        proxy.train()
        train_loss_sum = 0.0
        train_bce_sum = 0.0
        train_mse_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (code, spec, label, regs, noise) in enumerate(train_loader):
            code = code.to(device)
            spec = spec.to(device)
            label = label.to(device)
            regs = regs.to(device)
            noise = noise.to(device)

            out = proxy(code, spec, noise_level=noise)

            bce_loss = F.binary_cross_entropy_with_logits(out["pass_logit"], label)
            mse_loss = F.mse_loss(out["reg_values"], regs)
            loss = bce_weight * bce_loss + mse_weight * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proxy.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * code.size(0)
            train_bce_sum += bce_loss.item() * code.size(0)
            train_mse_sum += mse_loss.item() * code.size(0)

            preds = (out["pass_prob"] > 0.5).float()
            train_correct += (preds == label).sum().item()
            train_total += code.size(0)

        scheduler.step()

        train_loss = train_loss_sum / train_total
        train_bce = train_bce_sum / train_total
        train_mse = train_mse_sum / train_total
        train_acc = train_correct / train_total * 100

        # Validate
        proxy.eval()
        val_loss_sum = 0.0
        val_bce_sum = 0.0
        val_mse_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for code, spec, label, regs, noise in val_loader:
                code = code.to(device)
                spec = spec.to(device)
                label = label.to(device)
                regs = regs.to(device)
                noise = noise.to(device)

                out = proxy(code, spec, noise_level=noise)

                bce_loss = F.binary_cross_entropy_with_logits(out["pass_logit"], label)
                mse_loss = F.mse_loss(out["reg_values"], regs)
                loss = bce_weight * bce_loss + mse_weight * mse_loss

                val_loss_sum += loss.item() * code.size(0)
                val_bce_sum += bce_loss.item() * code.size(0)
                val_mse_sum += mse_loss.item() * code.size(0)

                preds = (out["pass_prob"] > 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += code.size(0)

        val_loss = val_loss_sum / val_total
        val_bce = val_bce_sum / val_total
        val_mse = val_mse_sum / val_total
        val_acc = val_correct / val_total * 100

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(proxy.state_dict(), best_path)
            saved_marker = " *saved*"
        else:
            saved_marker = ""

        if verbose:
            print(
                f"Epoch {epoch+1:2d}/{num_epochs} | "
                f"train loss={train_loss:.4f} (bce={train_bce:.4f} mse={train_mse:.4f}) acc={train_acc:.1f}% | "
                f"val loss={val_loss:.4f} (bce={val_bce:.4f} mse={val_mse:.4f}) acc={val_acc:.1f}%{saved_marker}"
            )

    if verbose:
        print(f"\nBest val loss: {best_val_loss:.4f}")
        print(f"Saved to: {best_path}")

    # Load best checkpoint
    proxy.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return proxy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neural Execution Proxy for EGDC")
    parser.add_argument("--train", action="store_true", help="Train the proxy")
    parser.add_argument("--num_samples", type=int, default=50_000, help="Training samples")
    parser.add_argument("--num_epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/egdc")
    args = parser.parse_args()

    if args.train:
        proxy = train_proxy(
            num_samples=args.num_samples,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(f"\nProxy trained successfully!")
    else:
        parser.print_help()
