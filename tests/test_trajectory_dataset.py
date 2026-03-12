"""Tests for preparing hidden SOME trajectories as training data."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing import (
    BufferedInternalController,
    InternalControllerConfig,
    InternalDeliberationTask,
    TrajectoryLogger,
    build_distillation_dataset,
    build_distillation_examples,
    load_trajectory,
    summarize_distillation_dataset,
    write_distillation_dataset,
)


FIB_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 6}, "expected": 8},
]


class TestTrajectoryDataset(unittest.TestCase):
    """Validate training-data extraction from hidden trajectories."""

    def _write_successful_repair_trajectory(self, path: Path) -> None:
        def fake_provider(prompt: str) -> str:
            if "Think privately" in prompt:
                return "Use the normal Fibonacci base cases."
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

    def test_build_distillation_examples_filters_failed_attempts_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_successful_repair_trajectory(trajectory_path)

            trajectory = load_trajectory(trajectory_path)
            examples = build_distillation_examples(trajectory)

            self.assertEqual([example.example_type for example in examples], ["think", "patch", "commit"])
            self.assertTrue(all(example.success_label for example in examples))
            self.assertEqual(examples[1].generation_attempt, 2)
            self.assertEqual(examples[1].messages[0]["role"], "user")
            self.assertIn("repairing a hidden candidate", examples[1].prompt.lower())
            self.assertIsNone(examples[-1].generation_attempt)

    def test_build_distillation_examples_can_include_failed_attempts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory_path = Path(tmpdir) / "trajectory.jsonl"
            self._write_successful_repair_trajectory(trajectory_path)

            trajectory = load_trajectory(trajectory_path)
            examples = build_distillation_examples(trajectory, include_failed_steps=True)

            self.assertEqual([example.example_type for example in examples], ["think", "write", "patch", "commit"])
            failed_write = next(example for example in examples if example.example_type == "write")
            self.assertFalse(failed_write.success_label)
            self.assertIn("expected", failed_write.verification_error or "")

    def test_dataset_builder_skips_non_trajectory_jsonl_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory_path = root / "trajectory.jsonl"
            self._write_successful_repair_trajectory(trajectory_path)
            (root / "not_a_trajectory.jsonl").write_text('{"event":"task_start"}\n', encoding="utf-8")

            dataset = build_distillation_dataset(root)
            summary = summarize_distillation_dataset(dataset)
            output_path = root / "dataset.jsonl"
            written_summary = write_distillation_dataset(dataset, output_path)

            self.assertEqual(summary["num_examples"], 3)
            self.assertEqual(summary["example_types"], {"think": 1, "patch": 1, "commit": 1})
            self.assertEqual(written_summary, summary)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["task_name"], "fibonacci")

    def test_cli_exports_training_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectory_path = root / "trajectory.jsonl"
            self._write_successful_repair_trajectory(trajectory_path)

            output_path = root / "prepared.jsonl"
            summary_path = root / "prepared.summary.json"
            script_path = Path(__file__).resolve().parents[1] / "ncpu" / "self_optimizing" / "prepare_internal_training_data.py"

            result = subprocess.run(
                [
                    "python3",
                    str(script_path),
                    "--trajectory-root",
                    str(root),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Wrote 3 training examples", result.stdout)
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_examples"], 3)


if __name__ == "__main__":
    unittest.main()
