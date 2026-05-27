"""Formal exit-code contract for npcot_compliance CLI (N5b).

Exit codes:
    0 — aggregate_risk in {safe, warn}, CLI succeeded.
    2 — library file not found / unreadable.
    3 — aggregate_risk = high (policy gate should block deployment).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.cli.npcot_compliance", *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestComplianceCliExitCodes(unittest.TestCase):
    def test_exit_0_on_safe_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safe.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            lib.save(path)
            result = _run_cli("--markdown", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Aggregate risk", result.stdout)

    def test_exit_0_on_warn_library(self):
        # Warn-level aggregate (naive product) should still exit 0 — it's
        # a warning, not a deployment block.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "warn.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(1, 0, 1, 0, 0.0),
                task_name="naive_product",
            )
            lib.save(path)
            result = _run_cli(
                "--max-length", "8",
                "--input-lower", "-3", "--input-upper", "3",
                str(path),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_exit_3_on_high_risk_library(self):
        # Force a high-risk aggregate by setting a very small overflow
        # threshold — any reasonable library overflows it.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "high.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            lib.save(path)
            result = _run_cli(
                "--overflow-threshold", "0.01",
                "--max-length", "100",
                "--input-lower", "-100", "--input-upper", "100",
                str(path),
            )
            self.assertEqual(result.returncode, 3, result.stderr)

    def test_exit_2_on_missing_file(self):
        result = _run_cli("/tmp/definitely-not-there-xyzzy-abc123.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("library file not found", result.stderr)

    def test_json_mode_still_respects_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            lib.save(path)
            result = _run_cli("--json", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
