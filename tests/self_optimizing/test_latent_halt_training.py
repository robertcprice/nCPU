"""Tests for latent halt head dataset preparation and training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
    TrajectoryLogger,
    build_latent_halt_training_examples,
    load_trajectory,
    train_latent_halt_head,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestLatentHaltTraining(unittest.TestCase):
    def _write_trajectory(self, path: Path) -> None:
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion with correct base cases."
            if "repairing a hidden candidate" in prompt.lower():
                return (
                    "def fib(n):\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fib(n - 1) + fib(n - 2)\n"
                )
            return "def fib(n):\n    return 0\n"

        controller = BufferedInternalController(
            llm_provider=fake_provider,
            config=InternalControllerConfig(max_generation_attempts=3),
            trajectory_logger=TrajectoryLogger(str(path)),
        )
        task = InternalDeliberationTask(
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )
        workspace = controller.run_task(task)
        self.assertEqual(workspace.status, "committed")

    def test_build_latent_halt_training_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)

            examples = build_latent_halt_training_examples(trajectory)

        self.assertEqual([example.target_action for example in examples], ["continue", "commit"])
        self.assertEqual(examples[0].allowed_actions, ["continue", "fail"])
        self.assertEqual(examples[1].allowed_actions, ["commit", "continue"])
        self.assertTrue(examples[0].feature_summary["remaining_actions"])

    def test_train_latent_halt_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)
            examples = build_latent_halt_training_examples(trajectory)

            train_path = Path(tmpdir) / "latent_halt_train.jsonl"
            val_path = Path(tmpdir) / "latent_halt_val.jsonl"
            train_path.write_text(
                "\n".join(json.dumps(example.to_dict()) for example in examples) + "\n",
                encoding="utf-8",
            )
            val_path.write_text(
                "\n".join(json.dumps(example.to_dict()) for example in examples[:1]) + "\n",
                encoding="utf-8",
            )

            checkpoint_path = Path(tmpdir) / "latent_halt_head.pt"
            metrics = train_latent_halt_head(
                train_path=train_path,
                val_path=val_path,
                output_path=checkpoint_path,
                epochs=2,
                batch_size=2,
                learning_rate=1e-3,
                device="cpu",
            )
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertEqual(metrics["train_examples"], len(examples))
        self.assertGreaterEqual(metrics["train_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
