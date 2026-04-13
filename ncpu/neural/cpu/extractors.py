"""
Neural bit-level extractors for instruction field extraction.

Classes:
    NeuralMovzExtractor - MOVZ/MOVK 16-bit immediate + 2-bit hw
    NeuralBranchExtractor - 26-bit unconditional branch offset
    NeuralBranch19Extractor - 19-bit conditional branch offset
    NeuralLoopDetector - Trained loop pattern detector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NeuralMovzExtractor(nn.Module):
    """Neural network to extract MOVZ/MOVK fields from instruction bits (18 bits)."""

    def __init__(self, d_model=128):
        super().__init__()
        self.bit_embed = nn.Embedding(2, d_model // 2)
        self.pos_embed = nn.Embedding(32, d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.imm16_head = nn.Sequential(
            nn.Linear(d_model * 32, d_model), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_model, 16)
        )
        self.hw_head = nn.Sequential(
            nn.Linear(d_model * 32, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, bits):
        batch = bits.shape[0]
        bit_idx = (bits > 0.5).long()
        pos_idx = torch.arange(32, device=bits.device).unsqueeze(0).expand(batch, -1)

        bit_emb = self.bit_embed(bit_idx)
        pos_emb = self.pos_embed(pos_idx)

        x = torch.cat([bit_emb, pos_emb], dim=-1)
        x = self.encoder(x)
        x_flat = x.reshape(batch, -1)

        return self.imm16_head(x_flat), self.hw_head(x_flat)


class NeuralBranchExtractor(nn.Module):
    """Neural network to extract branch offset from instruction bits (26 bits)."""

    def __init__(self, d_model=128):
        super().__init__()
        self.bit_embed = nn.Embedding(2, d_model // 2)
        self.pos_embed = nn.Embedding(32, d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.offset_head = nn.Sequential(
            nn.Linear(d_model * 32, d_model * 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_model * 2, 26)
        )

    def forward(self, bits):
        batch = bits.shape[0]
        bit_idx = (bits > 0.5).long()
        pos_idx = torch.arange(32, device=bits.device).unsqueeze(0).expand(batch, -1)

        bit_emb = self.bit_embed(bit_idx)
        pos_emb = self.pos_embed(pos_idx)

        x = torch.cat([bit_emb, pos_emb], dim=-1)
        x = self.encoder(x)
        x_flat = x.reshape(batch, -1)

        return self.offset_head(x_flat)


class NeuralBranch19Extractor(nn.Module):
    """Neural extractor for 19-bit branch offsets (B.cond, CBZ/CBNZ)."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.bit_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(32, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.offset_head = nn.Linear(d_model, 19)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        x = self.bit_embed(bits.unsqueeze(-1))
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.offset_head(x)


# ════════════════════════════════════════════════════════════════════════════════
# HELPER: UNSIGNED TO SIGNED 64-BIT CONVERSION
# PyTorch uses signed int64, but ARM64 registers are unsigned 64-bit.
# This converts values >= 2^63 to their signed equivalent.
# ════════════════════════════════════════════════════════════════════════════════

def _u64_to_s64(val: int) -> int:
    """Convert unsigned 64-bit value to signed for torch.int64 storage."""
    val = val & 0xFFFFFFFFFFFFFFFF  # Ensure 64-bit
    if val >= 0x8000000000000000:
        return val - 0x10000000000000000  # Convert to signed
    return val


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL LOOP DETECTOR
# ════════════════════════════════════════════════════════════════════════════════

class NeuralLoopDetector(nn.Module):
    """
    Fast Neural Loop Detector - TRAINED for 100% type / 91% register accuracy!

    Key insight: Counter register has "loop-like" value (10-100000).
    Uses opcodes (bits 21-31) + register value patterns.

    Trained weights: loop_detector_fast.pt (19K params)
    """

    def __init__(self, max_body_len: int = 32):
        super().__init__()
        self.max_body_len = max_body_len

        # ═══════════════════════════════════════════════════════════════
        # INSTRUCTION ENCODER - Focus on opcodes (bits 21-31)
        # ═══════════════════════════════════════════════════════════════
        self.opcode_embed = nn.Sequential(
            nn.Linear(11 * max_body_len, 64),  # 11 bits per instruction
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ═══════════════════════════════════════════════════════════════
        # REGISTER ANALYZER - Which registers look like counters?
        # ═══════════════════════════════════════════════════════════════
        self.reg_analyzer = nn.Sequential(
            nn.Linear(32, 64),  # 32 "counter likelihood" scores
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ═══════════════════════════════════════════════════════════════
        # OUTPUT HEADS
        # ═══════════════════════════════════════════════════════════════
        self.type_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        self.counter_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.iter_head = nn.Sequential(
            nn.Linear(64 + 1, 32),  # + selected counter value
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def compute_counter_likelihood(self, reg_values: torch.Tensor) -> torch.Tensor:
        """Score each register on how counter-like its value is."""
        vals = reg_values.float()
        min_good, max_good = 10, 100000

        in_range = (vals >= min_good) & (vals <= max_good)
        score = in_range.float()
        score = score - 0.5 * (vals > max_good).float()
        score = score - 1.0 * (vals <= 0).float()

        return score

    def forward(
        self,
        body_bits: torch.Tensor,  # [body_len, 32]
        reg_values: torch.Tensor,  # [32]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: loop_type_logits, counter_probs, iterations - ALL ON GPU!"""
        body_len = body_bits.shape[0]

        # Extract opcodes (bits 21-31)
        opcodes = body_bits[:, 21:32]  # [body_len, 11]

        # Pad to max_body_len
        if body_len < self.max_body_len:
            padding = torch.zeros(self.max_body_len - body_len, 11, device=body_bits.device)
            opcodes = torch.cat([opcodes, padding], dim=0)

        opcode_flat = opcodes.flatten()
        opcode_features = self.opcode_embed(opcode_flat)

        # Register analysis
        counter_likelihood = self.compute_counter_likelihood(reg_values)
        reg_features = self.reg_analyzer(counter_likelihood)

        # Combine
        combined = torch.cat([opcode_features, reg_features], dim=-1)

        # Predictions
        type_logits = self.type_head(combined)

        counter_logits = self.counter_head(combined)
        counter_logits = counter_logits + counter_likelihood * 2  # Bias toward good values
        counter_probs = F.softmax(counter_logits, dim=-1)

        best_counter = torch.argmax(counter_probs)
        counter_val = reg_values[best_counter].float()
        iter_input = torch.cat([combined, counter_val.unsqueeze(0) / 10000], dim=-1)
        log_iters = self.iter_head(iter_input)
        iterations = torch.pow(10.0, log_iters.clamp(1, 5))

        return type_logits, counter_probs, iterations
