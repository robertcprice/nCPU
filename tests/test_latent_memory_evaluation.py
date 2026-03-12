"""Tests for latent-memory evaluation against a zero baseline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing.latent_memory_evaluation import evaluate_latent_memory_head
from ncpu.self_optimizing.latent_memory_head import LatentMemoryHeadConfig
from ncpu.self_optimizing.latent_memory_training import (
    build_latent_memory_training_bundle,
    train_latent_memory_head,
)
from ncpu.self_optimizing import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
    TrajectoryLogger,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestLatentMemoryEvaluation(unittest.TestCase):
    def _write_successful_trajectory(self, path: Path, *, task_name: str) -> None:
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use recursion."
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
            name=task_name,
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )
        workspace = controller.run_task(task)
        self.assertEqual(workspace.status, "committed")

    def test_evaluation_report_contains_baseline_and_model_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trajectories"
            root.mkdir()
            self._write_successful_trajectory(root / "fib_a.jsonl", task_name="fib_a")
            self._write_successful_trajectory(root / "fib_b.jsonl", task_name="fib_b")
            prepared = Path(tmpdir) / "prepared"
            bundle = build_latent_memory_training_bundle(root, prepared, val_ratio=0.5)
            checkpoint = Path(tmpdir) / "latent_memory_head.pt"
            train_latent_memory_head(
                train_path=bundle["train_path"],
                val_path=bundle["val_path"],
                output_path=checkpoint,
                config=LatentMemoryHeadConfig(),
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                device="cpu",
            )

            report = evaluate_latent_memory_head(
                train_path=bundle["train_path"],
                val_path=bundle["val_path"],
                checkpoint_path=checkpoint,
                output_path=Path(tmpdir) / "eval.json",
                device="cpu",
            )

            self.assertIn("train", report["splits"])
            self.assertIn("val", report["splits"])
            self.assertIn("baseline_zero", report["splits"]["train"])
            self.assertIn("model", report["splits"]["train"])
            self.assertIn("improvement", report["splits"]["val"])
            self.assertTrue((Path(tmpdir) / "eval.json").exists())


if __name__ == "__main__":
    unittest.main()
