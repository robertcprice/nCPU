"""Tests for latent descriptor dataset preparation and training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing import (
    build_latent_descriptor_training_examples,
    load_trajectory,
    train_latent_descriptor_head,
)


def _write_latent_descriptor_trajectory(path: Path) -> None:
    events = [
        {
            "event": "workspace_initialized",
            "task_name": "fibonacci",
            "category": "coding",
            "status": "running",
            "max_generation_attempts": 3,
            "latent_state": {},
        },
        {
            "event": "workspace_step",
            "task_name": "fibonacci",
            "status": "running",
            "step_index": 1,
            "action": "descriptor_update",
            "success": True,
            "error": None,
            "metadata": {
                "kind": "verify_failure_descriptor",
                "adaptation_descriptor": {
                    "signal_projection": [0.5, -0.2, 0.3],
                    "update_kind": "verify_failure_descriptor",
                    "source": "latent_state",
                },
            },
            "latent_state": {
                "hidden_plan": "Use recursion.",
                "last_failure_summary": "expected 8",
                "verification_failures": 1,
                "confidence": 0.15,
                "failure_patterns": ["expected 8"],
                "recent_actions": ["write", "verify"],
            },
            "prompt": "descriptor",
            "response_text": "applied",
        },
        {
            "event": "workspace_committed",
            "task_name": "fibonacci",
            "status": "committed",
            "committed_verified": True,
            "generation_attempts": 2,
            "last_error": None,
            "latent_state": {},
            "committed_output": "def fib(n): return n",
        },
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


class TestLatentDescriptorTraining(unittest.TestCase):
    def test_build_latent_descriptor_training_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            _write_latent_descriptor_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)

            examples = build_latent_descriptor_training_examples(trajectory)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].update_kind, "verify_failure_descriptor")
        self.assertEqual(examples[0].target_vector[:3], [0.5, -0.2, 0.3])

    def test_train_latent_descriptor_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            _write_latent_descriptor_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)
            examples = build_latent_descriptor_training_examples(trajectory)

            train_path = Path(tmpdir) / "latent_descriptor_train.jsonl"
            val_path = Path(tmpdir) / "latent_descriptor_val.jsonl"
            train_path.write_text(
                "\n".join(json.dumps(example.to_dict()) for example in examples) + "\n",
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(json.dumps(example.to_dict()) for example in examples) + "\n",
                encoding="utf-8",
            )

            checkpoint_path = Path(tmpdir) / "latent_descriptor_head.pt"
            metrics = train_latent_descriptor_head(
                train_path=train_path,
                val_path=val_path,
                output_path=checkpoint_path,
                epochs=2,
                batch_size=1,
                learning_rate=1e-3,
                device="cpu",
            )
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertEqual(metrics["train_examples"], 1)
        self.assertGreaterEqual(metrics["train_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
