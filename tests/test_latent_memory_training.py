"""Tests for latent memory training bundle creation and training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
    TrajectoryLogger,
)
from ncpu.self_optimizing.latent_memory_head import LatentMemoryHeadConfig
from ncpu.self_optimizing.latent_memory_training import (
    build_latent_memory_training_bundle,
    build_latent_memory_training_examples,
    train_latent_memory_head,
)
from ncpu.self_optimizing.trajectory_dataset import load_trajectory


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestLatentMemoryTraining(unittest.TestCase):
    def _write_successful_trajectory(self, path: Path) -> None:
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
            name="fibonacci",
            prompt="Write fib(n) in Python.",
            test_cases=FIB_TESTS,
        )
        workspace = controller.run_task(task)
        self.assertEqual(workspace.status, "committed")
        self.assertTrue(workspace.committed_verified)

    def test_build_examples_from_successful_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_successful_trajectory(trajectory_path)
            trajectory = load_trajectory(trajectory_path)

            examples = build_latent_memory_training_examples(trajectory)

            self.assertGreaterEqual(len(examples), 1)
            self.assertIn(examples[0].event_kind, {"think", "write", "verify", "patch", "commit"})
            self.assertEqual(len(examples[0].target_vector), 16)

    def test_failed_trajectory_still_produces_memory_examples_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "failed.jsonl"

            def fake_provider(_prompt: str) -> str:
                return "def fib(n):\n    return 0\n"

            controller = BufferedInternalController(
                llm_provider=fake_provider,
                config=InternalControllerConfig(max_generation_attempts=1, plan_before_generate=False),
                trajectory_logger=TrajectoryLogger(str(trajectory_path)),
            )
            task = InternalDeliberationTask(
                name="fibonacci",
                prompt="Write fib(n) in Python.",
                test_cases=FIB_TESTS,
            )
            workspace = controller.run_task(task)
            self.assertEqual(workspace.status, "failed")

            trajectory = load_trajectory(trajectory_path)
            examples = build_latent_memory_training_examples(trajectory)

            self.assertGreaterEqual(len(examples), 1)
            self.assertIn(examples[-1].event_kind, {"write", "verify", "fail"})

    def test_bundle_and_train_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "trajectories"
            root.mkdir()
            self._write_successful_trajectory(root / "fib.jsonl")
            self._write_successful_trajectory(root / "trib.jsonl")
            output_dir = Path(tmpdir) / "bundle"

            bundle = build_latent_memory_training_bundle(root, output_dir, val_ratio=0.5)
            self.assertTrue(Path(bundle["train_path"]).exists())
            self.assertTrue(Path(bundle["val_path"]).exists())

            output_path = Path(tmpdir) / "latent_memory_head.pt"
            metrics = train_latent_memory_head(
                train_path=bundle["train_path"],
                val_path=bundle["val_path"],
                output_path=output_path,
                config=LatentMemoryHeadConfig(),
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                device="cpu",
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(metrics["trained"])
            checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertIn("state_dict", checkpoint)


if __name__ == "__main__":
    unittest.main()
