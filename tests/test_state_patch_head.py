"""Tests for the learned latent-state patch head."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.state_patch_head import (
    StatePatchHead,
    StatePatchHeadConfig,
    load_state_patch_head,
)


class TestStatePatchHead(unittest.TestCase):
    def test_transform_returns_configured_output_width(self):
        head = StatePatchHead(StatePatchHeadConfig(input_dim=8, hidden_dim=16, output_dim=4))
        projection = head.transform([0.1, -0.2, 0.3], device="cpu")
        self.assertEqual(len(projection), 4)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "patch_head.pt"
            head = StatePatchHead(StatePatchHeadConfig(input_dim=8, hidden_dim=16, output_dim=4))
            torch.save(
                {
                    "config": head.config.to_dict(),
                    "state_dict": head.state_dict(),
                },
                path,
            )
            loaded = load_state_patch_head(path=path, device="cpu")
            projection = loaded.transform([0.4, 0.2], device="cpu")
            self.assertEqual(len(projection), 4)


if __name__ == "__main__":
    unittest.main()
