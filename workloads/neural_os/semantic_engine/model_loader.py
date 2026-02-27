#!/usr/bin/env python3
"""
MODEL LOADER: Load and use trained models from checkpoints

Provides unified interface to:
1. Load trained synthesis models
2. Run inference
3. Integrate with Singularity Core
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import os


@dataclass
class ModelConfig:
    """Configuration for loaded models."""
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_actions: int = 6
    num_layers: int = 4
    dropout: float = 0.1


class TransformerEncoder(nn.Module):
    """Transformer-based program encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(256, config.embedding_dim)
        self.pos_embed = nn.Embedding(512, config.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.layer_norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        pos_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(pos_ids)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.layer_norm(self.output_proj(x))
        return x


class SynthesisPolicy(nn.Module):
    """Policy network for synthesis actions."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.state_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, input_emb: torch.Tensor, target_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state = torch.cat([input_emb, target_emb], dim=-1)
        features = self.state_encoder(state)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


class SynthesisModel:
    """
    Unified interface for trained synthesis models.
    """

    # Operation mapping (must match training)
    OPERATIONS = {
        0: ('identity', lambda x: x),
        1: ('double', lambda x: 2 * x),
        2: ('square', lambda x: x * x),
        3: ('negate', lambda x: -x),
        4: ('add_ten', lambda x: x + 10),
        5: ('times_five', lambda x: 5 * x),
    }

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.config = ModelConfig()

        # Initialize models
        self.encoder = TransformerEncoder(self.config).to(self.device)
        self.policy = SynthesisPolicy(self.config).to(self.device)

        # Load checkpoint
        self.load_checkpoint(checkpoint_path)

        # Set to eval mode
        self.encoder.eval()
        self.policy.eval()

    def load_checkpoint(self, path: str):
        """Load model weights from checkpoint."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Handle different checkpoint formats
        if 'encoder_state' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.policy.load_state_dict(checkpoint['policy_state'])
        elif 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.policy.load_state_dict(checkpoint['policy'])
        else:
            print(f"Warning: Unknown checkpoint format. Keys: {checkpoint.keys()}")

        print(f"✅ Loaded checkpoint from {path}")

    def tokenize(self, value: Any, max_len: int = 32) -> torch.Tensor:
        """Convert value to tokens."""
        s = str(value)
        tokens = [ord(c) % 256 for c in s[:max_len]]
        tokens = tokens + [0] * (max_len - len(tokens))
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def predict_operation(self, input_val: int, output_val: int) -> Tuple[str, float]:
        """
        Predict which operation transforms input to output.
        Returns (operation_name, confidence).
        """
        with torch.no_grad():
            input_tokens = self.tokenize(input_val)
            output_tokens = self.tokenize(output_val)

            input_emb = self.encoder(input_tokens)
            output_emb = self.encoder(output_tokens)

            logits, _ = self.policy(input_emb, output_emb)
            probs = F.softmax(logits, dim=-1)

            pred_idx = logits.argmax(dim=-1).item()
            confidence = probs[0, pred_idx].item()

            op_name = self.OPERATIONS[pred_idx][0]
            return op_name, confidence

    def synthesize(self, input_val: int, target_output: int) -> Optional[str]:
        """
        Synthesize the operation that transforms input to target_output.
        Returns operation name if found, None otherwise.
        """
        op_name, confidence = self.predict_operation(input_val, target_output)

        # Verify the operation
        op_fn = self.OPERATIONS[list(self.OPERATIONS.keys())[
            [v[0] for v in self.OPERATIONS.values()].index(op_name)
        ]][1]

        try:
            actual_output = op_fn(input_val)
            if actual_output == target_output:
                return op_name
        except:
            pass

        return None

    def batch_predict(self, pairs: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
        """Batch prediction for multiple (input, output) pairs."""
        results = []
        for input_val, output_val in pairs:
            results.append(self.predict_operation(input_val, output_val))
        return results


def load_best_model(base_path: str = None) -> SynthesisModel:
    """Load the best trained model."""
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))

    checkpoint_path = os.path.join(base_path, 'checkpoints', 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No best_model.pt found at {checkpoint_path}")

    return SynthesisModel(checkpoint_path)


# =============================================================================
# MAIN: Test the model loader
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL LOADER: Testing trained synthesis model")
    print("=" * 60)

    # Load model
    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to download checkpoints first!")
        exit(1)

    # Test predictions
    print("\n[Testing Predictions]")
    test_cases = [
        (5, 5),    # identity
        (5, 10),   # double
        (5, 25),   # square
        (5, -5),   # negate
        (5, 15),   # add_ten
        (5, 25),   # times_five (same as square for 5)
        (3, 9),    # square
        (4, 8),    # double
        (7, 17),   # add_ten
    ]

    correct = 0
    for input_val, output_val in test_cases:
        op_name, conf = model.predict_operation(input_val, output_val)
        # Verify
        expected_op = None
        for op_id, (name, fn) in model.OPERATIONS.items():
            try:
                if fn(input_val) == output_val:
                    expected_op = name
                    break
            except:
                pass

        is_correct = op_name == expected_op
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"  {input_val} → {output_val}: predicted={op_name} (conf={conf:.2f}) {status}")

    print(f"\nAccuracy: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")

    # Test synthesis
    print("\n[Testing Synthesis]")
    synth_cases = [
        (3, 6),   # double
        (4, 16),  # square
        (7, -7),  # negate
    ]

    for input_val, target in synth_cases:
        result = model.synthesize(input_val, target)
        print(f"  synthesize({input_val} → {target}): {result}")

    print("\n✅ Model loader ready for integration")
