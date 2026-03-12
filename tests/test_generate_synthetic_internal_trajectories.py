"""Tests for synthetic hidden SOME trajectory generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ncpu.self_optimizing.generate_synthetic_internal_trajectories import (
    generate_synthetic_internal_trajectories,
)
from ncpu.self_optimizing.trajectory_dataset import iter_trajectories


class TestGenerateSyntheticInternalTrajectories(unittest.TestCase):
    def test_generator_writes_verified_trajectories_with_memory_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = generate_synthetic_internal_trajectories(
                output_dir=tmpdir,
                num_trajectories=6,
            )

            self.assertEqual(summary["num_trajectories"], 6)
            trajectories = list(iter_trajectories(tmpdir))
            self.assertEqual(len(trajectories), 6)
            self.assertTrue(all(trajectory.committed_verified for trajectory in trajectories))

            sample_path = Path(summary["paths"][0])
            sample_text = sample_path.read_text(encoding="utf-8")
            self.assertIn("memory_vector", sample_text)
            self.assertIn("memory_updates", sample_text)


if __name__ == "__main__":
    unittest.main()
