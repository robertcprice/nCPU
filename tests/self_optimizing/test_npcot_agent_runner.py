"""Tests for npcot_agent_runner (shape, not full run)."""

from __future__ import annotations

import unittest
from pathlib import Path

from ncpu.self_optimizing.npcot_agent_runner import (
    AgentConfig,
    parse_cli,
)


class TestParseCli(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_cli([
            "--library", "/tmp/lib.json",
            "--coprocessor-checkpoint", "/tmp/ckpt.pt",
        ])
        self.assertEqual(cfg.model, "Qwen/Qwen3.5-1.5B")
        self.assertEqual(cfg.target_layers, [-2, -1])
        self.assertTrue(cfg.continual_growth)
        self.assertEqual(cfg.max_retries, 2)

    def test_no_continual_growth(self):
        cfg = parse_cli([
            "--library", "/tmp/lib.json",
            "--coprocessor-checkpoint", "/tmp/ckpt.pt",
            "--no-continual-growth",
        ])
        self.assertFalse(cfg.continual_growth)

    def test_custom_retries(self):
        cfg = parse_cli([
            "--library", "/tmp/lib.json",
            "--coprocessor-checkpoint", "/tmp/ckpt.pt",
            "--max-retries", "10",
        ])
        self.assertEqual(cfg.max_retries, 10)

    def test_target_layers(self):
        cfg = parse_cli([
            "--library", "/tmp/lib.json",
            "--coprocessor-checkpoint", "/tmp/ckpt.pt",
            "--target-layers", "-3,-2,-1",
        ])
        self.assertEqual(cfg.target_layers, [-3, -2, -1])

    def test_retry_defaults_include_baseline(self):
        cfg = AgentConfig()
        self.assertIn(0.0, cfg.retry_gates)


if __name__ == "__main__":
    unittest.main()
