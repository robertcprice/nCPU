"""Unit tests for the Qwen NPCoT training harness (shape, not full run)."""

from __future__ import annotations

import unittest
from pathlib import Path

from ncpu.self_optimizing.npcot_qwen_training import (
    QwenNPCoTTrainingConfig,
    parse_cli,
)


class TestParseCli(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_cli([])
        self.assertEqual(cfg.model, "Qwen/Qwen3.5-4B")
        self.assertEqual(cfg.target_layers, [-2, -1])
        self.assertEqual(cfg.dataset, "mbpp")

    def test_target_layers_parsing(self):
        cfg = parse_cli(["--target-layers", "-4,-3,-2,-1"])
        self.assertEqual(cfg.target_layers, [-4, -3, -2, -1])

    def test_gsm8k_dataset(self):
        cfg = parse_cli(["--dataset", "gsm8k"])
        self.assertEqual(cfg.dataset, "gsm8k")

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(SystemExit):
            parse_cli(["--dataset", "bogus"])

    def test_out_paths(self):
        cfg = parse_cli([
            "--out-checkpoint", "/tmp/ckpt.pt",
            "--out-library", "/tmp/lib.json",
        ])
        self.assertEqual(cfg.out_checkpoint, Path("/tmp/ckpt.pt"))
        self.assertEqual(cfg.out_library, Path("/tmp/lib.json"))


if __name__ == "__main__":
    unittest.main()
