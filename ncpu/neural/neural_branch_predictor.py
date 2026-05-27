"""Neural Branch Predictor — predicts branch outcomes for speculative execution.

Learns branch direction (taken/not-taken) from instruction context and flag
history, enabling the vectorizer to speculatively collapse loops even before
the branch is fully resolved.

Architecture (~8K params):
  Input: branch instruction features + recent flag history
    - branch condition code (4 bits → Embedding(16, 8))
    - flag values at branch point [N, Z, C, V] → 4 floats
    - recent flag history [4 × 4] → 16 floats (last 4 flag states)
    - branch direction (forward/backward) → 1 float
    - loop counter heuristic (counter value normalized) → 1 float
  Encoder: Linear(30→32) → ReLU → Linear(32→1) → sigmoid
  Output: P(taken) — probability the branch is taken

For the vectorizer: if P(taken) > 0.9 for a backward branch, the loop
likely continues → trigger early vectorization without waiting for SYNC.
"""

import torch
import torch.nn as nn


class NeuralBranchPredictor(nn.Module):
    """Predicts branch outcome probability from instruction context."""

    def __init__(self):
        super().__init__()
        self.cond_embed = nn.Embedding(16, 8)  # 16 ARM64 condition codes
        self.predictor = nn.Sequential(
            nn.Linear(30, 32),  # 8 (cond) + 4 (flags) + 16 (flag history) + 1 (dir) + 1 (counter)
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Ring buffer for flag history (last 4 states)
        self._flag_history = None

    def update_flag_history(self, flags, device):
        """Record current flag state into history ring buffer."""
        if self._flag_history is None:
            self._flag_history = torch.zeros(4, 4, device=device)
        # Shift history: drop oldest, append newest
        self._flag_history = torch.cat([
            self._flag_history[1:],
            flags[:4].unsqueeze(0)
        ], dim=0)

    def forward(self, cond_code, flags, is_backward, counter_hint=0.0):
        """Predict P(branch taken).

        Args:
            cond_code: int or scalar tensor (0-15)
            flags: [4] float tensor (N, Z, C, V)
            is_backward: bool — True for backward branches (likely loops)
            counter_hint: float — normalized counter value (0-1, where 0=counter exhausted)

        Returns:
            p_taken: scalar float — probability branch is taken
        """
        device = flags.device
        if self._flag_history is None:
            self._flag_history = torch.zeros(4, 4, device=device)

        cond_t = torch.tensor([cond_code], device=device, dtype=torch.long).clamp(0, 15)
        cond_emb = self.cond_embed(cond_t).squeeze(0)  # [8]

        features = torch.cat([
            cond_emb,                                           # [8]
            flags[:4],                                          # [4]
            self._flag_history.flatten(),                       # [16]
            torch.tensor([1.0 if is_backward else 0.0], device=device),  # [1]
            torch.tensor([counter_hint], device=device),        # [1]
        ])  # [30]

        return self.predictor(features).squeeze()
