"""Tests for the NPCoT coprocessor sweep runner (N1-next).

These tests verify the script's *shape* — argparse, dry-run, compliance
attachment — without requiring a real model load or GPU. They run on CPU
in under a second.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.coprocessor.run_npcot_sweep import (
    SweepConfig,
    attach_compliance_report,
    parse_cli,
    run_dry,
)
from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)


class TestParseCli(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_cli([])
        self.assertEqual(cfg.model, "Qwen/Qwen3.5-1.5B")
        self.assertEqual(cfg.epochs, 2)
        self.assertEqual(cfg.target_layers, [-2, -1])
        self.assertFalse(cfg.dry_run)

    def test_custom_target_layers(self):
        cfg = parse_cli(["--target-layers", "2,4,6"])
        self.assertEqual(cfg.target_layers, [2, 4, 6])

    def test_dry_run_flag(self):
        cfg = parse_cli(["--dry-run"])
        self.assertTrue(cfg.dry_run)

    def test_single_layer(self):
        cfg = parse_cli(["--target-layers", "-1"])
        self.assertEqual(cfg.target_layers, [-1])


class TestDryRun(unittest.TestCase):
    def test_dry_run_without_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = Path(tmp) / "nonexistent.json"
            cfg = SweepConfig(
                model="fake/model",
                library_path=str(lib_path),
                output_json=str(Path(tmp) / "run.json"),
                dry_run=True,
            )
            report = run_dry(cfg)
            self.assertEqual(report["mode"], "dry_run")
            self.assertEqual(report["library_summary"]["entry_count"], 0)

    def test_dry_run_with_existing_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib_path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            lib.save(lib_path)
            cfg = SweepConfig(
                model="fake/model",
                library_path=str(lib_path),
                output_json=str(Path(tmp) / "run.json"),
                dry_run=True,
            )
            report = run_dry(cfg)
            self.assertEqual(report["library_summary"]["entry_count"], 1)

    def test_dry_run_resolves_home_tilde(self):
        cfg = SweepConfig(
            model="fake/model",
            library_path="~/does-not-exist.json",
            dry_run=True,
        )
        report = run_dry(cfg)
        self.assertIn("library_path_resolved", report)
        self.assertFalse("~" in report["library_path_resolved"])


class TestAttachCompliance(unittest.TestCase):
    def test_attach_compliance_to_report(self):
        lib = ArrayProgramLibrary()
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        run_report = {"mode": "dry_run"}
        attached = attach_compliance_report(run_report, lib)
        self.assertIn("compliance", attached)
        self.assertEqual(
            attached["compliance"]["aggregate"]["entry_count"], 1
        )

    def test_report_remains_json_serializable(self):
        lib = ArrayProgramLibrary()
        lib.record(
            torch.tensor([1.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="x",
        )
        run_report = {"mode": "dry_run", "timestamp": 123.0}
        attached = attach_compliance_report(run_report, lib)
        json.dumps(attached)


if __name__ == "__main__":
    unittest.main()
