"""Audit and skill-explorer tests (NV1 + NV4)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)


def _populate(library: ArrayProgramLibrary) -> None:
    library.record(
        torch.tensor([1.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
        convergence_gap=0.02,
        cached_at_step=3,
    )
    library.record(
        torch.tensor([0.0, 1.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.5),
        task_name="max",
        convergence_gap=0.1,
    )
    # Warm hits.
    library.lookup(torch.tensor([1.0, 0.0, 0.0]))
    library.lookup(torch.tensor([1.0, 0.0, 0.0]))


class TestAuditReport(unittest.TestCase):
    def test_report_fields(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85, max_entries=32)
        )
        _populate(library)
        report = library.audit_report()
        self.assertEqual(report["summary"]["entry_count"], 2)
        self.assertEqual(report["summary"]["total_hits"], 2)
        self.assertEqual(report["summary"]["unique_program_shapes"], 2)
        self.assertAlmostEqual(
            report["summary"]["avg_convergence_gap"], 0.06, places=5
        )
        self.assertAlmostEqual(
            report["summary"]["max_convergence_gap"], 0.1, places=5
        )
        sum_entry = next(e for e in report["entries"] if e["task_name"] == "sum")
        self.assertEqual(sum_entry["cached_at_step"], 3)
        self.assertEqual(sum_entry["signature_dim"], 3)
        self.assertIn("program_text", sum_entry["program"])

    def test_empty_library_report(self):
        library = ArrayProgramLibrary()
        report = library.audit_report()
        self.assertEqual(report["summary"]["entry_count"], 0)
        self.assertIsNone(report["summary"]["avg_convergence_gap"])
        self.assertEqual(report["entries"], [])

    def test_markdown_output_contains_program_blocks(self):
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        _populate(library)
        md = library.audit_markdown()
        self.assertIn("# Array Program Library — Audit Report", md)
        self.assertIn("## Summary", md)
        self.assertIn("### Skill 1: `sum`", md)
        self.assertIn("### Skill 2: `max`", md)
        self.assertIn("```rust", md)
        self.assertIn("fn array_thought", md)

    def test_markdown_empty_library(self):
        library = ArrayProgramLibrary()
        md = library.audit_markdown()
        self.assertIn("(library is empty)", md)


class TestSkillExplorerCLI(unittest.TestCase):
    def _write_library(self, tmp: Path) -> Path:
        library = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.85)
        )
        _populate(library)
        path = tmp / "library.json"
        library.save(path)
        return path

    def test_cli_runs_text_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library_path = self._write_library(tmp_path)
            # Invoke the CLI as a module via the same interpreter.
            result = subprocess.run(
                [sys.executable, "-m", "scripts.cli.npcot_skill_explorer", str(library_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NPCoT Array Program Library", result.stdout)
            self.assertIn("task=sum", result.stdout)
            self.assertIn("task=max", result.stdout)
            self.assertIn("fn array_thought", result.stdout)

    def test_cli_markdown_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library_path = self._write_library(tmp_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.cli.npcot_skill_explorer",
                    "--markdown",
                    str(library_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("### Skill 1:", result.stdout)
            self.assertIn("```rust", result.stdout)

    def test_cli_json_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library_path = self._write_library(tmp_path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.cli.npcot_skill_explorer",
                    "--json",
                    str(library_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["summary"]["entry_count"], 2)

    def test_cli_missing_file_returns_nonzero(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.cli.npcot_skill_explorer",
                "/tmp/definitely-not-there-xyzzy.json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("library file not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
