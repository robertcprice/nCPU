"""MBPP runner tests (BENCH-2)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.mbpp_runner import (
    MBPPConfig,
    _check_mbpp,
    _mbpp_prompt,
    parse_cli,
    run_dry,
)


class TestParseCli(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_cli([])
        self.assertEqual(cfg.max_problems, 100)
        self.assertTrue(cfg.use_npcot)

    def test_no_library_flag(self):
        cfg = parse_cli(["--no-library"])
        self.assertFalse(cfg.use_npcot)


class TestDryRun(unittest.TestCase):
    def test_baseline_mode(self):
        cfg = MBPPConfig(use_npcot=False, dry_run=True)
        report = run_dry(cfg)
        self.assertTrue(report["all_ok"])

    def test_missing_library_fails(self):
        cfg = MBPPConfig(use_npcot=True, library_path=None, dry_run=True)
        report = run_dry(cfg)
        self.assertFalse(report["all_ok"])

    def test_valid_library_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="x",
            )
            lib.save(path)
            cfg = MBPPConfig(use_npcot=True, library_path=path, dry_run=True)
            self.assertTrue(run_dry(cfg)["all_ok"])


class TestMBPPPrompt(unittest.TestCase):
    def test_prompt_includes_tests(self):
        problem = {
            "task_id": "mbpp/1",
            "text": "Write a function to find the minimum cost.",
            "code": "",
            "test_list": ["assert min_cost(1) == 1", "assert min_cost(2) == 2"],
            "test_setup_code": "",
        }
        prompt = _mbpp_prompt(problem)
        self.assertIn("min_cost", prompt)
        self.assertIn("assert", prompt)
        self.assertIn("[BEGIN]", prompt)


class TestCheckMBPP(unittest.TestCase):
    def test_correct_solution_passes(self):
        problem = {
            "task_id": "mbpp/test",
            "test_list": [
                "assert add(1, 2) == 3",
                "assert add(5, 7) == 12",
            ],
            "test_setup_code": "",
        }
        passed, err = _check_mbpp(problem, "def add(a, b):\n    return a + b\n")
        self.assertTrue(passed, msg=err)

    def test_wrong_solution_fails(self):
        problem = {
            "task_id": "mbpp/test",
            "test_list": ["assert add(1, 2) == 3"],
            "test_setup_code": "",
        }
        passed, _ = _check_mbpp(problem, "def add(a, b):\n    return a - b\n")
        self.assertFalse(passed)

    def test_setup_code_is_executed(self):
        # Problems that need imports get them via test_setup_code.
        problem = {
            "task_id": "mbpp/import",
            "test_list": ["assert ceil_sqrt(10) == 4"],
            "test_setup_code": "import math",
        }
        solution = "def ceil_sqrt(x):\n    return math.ceil(math.sqrt(x))\n"
        passed, err = _check_mbpp(problem, solution)
        self.assertTrue(passed, msg=err)


if __name__ == "__main__":
    unittest.main()
