"""HumanEval runner tests (BENCH-1) — dry-run + code extraction + subprocess check."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.humaneval_runner import (
    HumanEvalConfig,
    _check_solution,
    _extract_code,
    parse_cli,
    run_dry,
)


class TestParseCli(unittest.TestCase):
    def test_defaults(self):
        cfg = parse_cli([])
        self.assertEqual(cfg.model, "Qwen/Qwen3.5-1.5B")
        self.assertEqual(cfg.max_problems, 164)
        self.assertTrue(cfg.use_npcot)

    def test_no_library_flag(self):
        cfg = parse_cli(["--no-library"])
        self.assertFalse(cfg.use_npcot)

    def test_custom_target_layers(self):
        cfg = parse_cli(["--target-layers", "5,10,15"])
        self.assertEqual(cfg.target_layers, [5, 10, 15])


class TestDryRun(unittest.TestCase):
    def test_baseline_mode_passes(self):
        cfg = HumanEvalConfig(model="fake/model", use_npcot=False, dry_run=True)
        report = run_dry(cfg)
        self.assertTrue(report["all_ok"])
        self.assertEqual(report["library_entries"], 0)

    def test_npcot_without_library_path_fails(self):
        cfg = HumanEvalConfig(
            model="fake/model", library_path=None, use_npcot=True, dry_run=True
        )
        report = run_dry(cfg)
        self.assertFalse(report["all_ok"])

    def test_npcot_with_missing_library_fails(self):
        cfg = HumanEvalConfig(
            model="fake/model",
            library_path=Path("/tmp/definitely-not-there.json"),
            use_npcot=True,
            dry_run=True,
        )
        report = run_dry(cfg)
        self.assertFalse(report["all_ok"])

    def test_npcot_with_valid_library_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            lib.record(
                torch.tensor([1.0, 0.0, 0.0]),
                DiscreteArrayProgram(0, 0, 0, 0, 0.0),
                task_name="sum",
            )
            lib.save(path)
            cfg = HumanEvalConfig(
                model="fake/model",
                library_path=path,
                use_npcot=True,
                dry_run=True,
            )
            report = run_dry(cfg)
            self.assertTrue(report["all_ok"])
            self.assertEqual(report["library_entries"], 1)


class TestExtractCode(unittest.TestCase):
    def test_extracts_first_fenced_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    return 42\n```\nEnd."
        self.assertEqual(_extract_code(text, ""), "def foo():\n    return 42")

    def test_strips_prompt_prefix(self):
        prompt = "def foo():\n    "
        generated = prompt + "return 42\n\nextra text"
        extracted = _extract_code(generated, prompt)
        self.assertIn("return 42", extracted)

    def test_falls_through_to_raw(self):
        text = "no code fence here, just code: return 42"
        self.assertIn("return 42", _extract_code(text, ""))


class TestCheckSolution(unittest.TestCase):
    def test_correct_solution_passes(self):
        problem = {
            "task_id": "test/0",
            "prompt": "def add(a, b):\n    ",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(5, 7) == 12\n",
            "entry_point": "add",
            "canonical_solution": "return a + b\n",
        }
        solution = problem["prompt"] + "return a + b"
        passed, err = _check_solution(problem, solution)
        self.assertTrue(passed, msg=err)

    def test_wrong_solution_fails(self):
        problem = {
            "task_id": "test/1",
            "prompt": "def add(a, b):\n    ",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
            "canonical_solution": "return a + b\n",
        }
        solution = problem["prompt"] + "return a - b"
        passed, err = _check_solution(problem, solution)
        self.assertFalse(passed)

    def test_timeout_is_detected(self):
        problem = {
            "task_id": "test/2",
            "prompt": "def wait_forever():\n    ",
            "test": "def check(candidate):\n    candidate()\n",
            "entry_point": "wait_forever",
            "canonical_solution": "while True: pass\n",
        }
        solution = problem["prompt"] + "while True: pass"
        passed, err = _check_solution(problem, solution, timeout_s=0.5)
        self.assertFalse(passed)
        self.assertIn("timeout", err)


if __name__ == "__main__":
    unittest.main()
