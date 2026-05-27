"""Tests for run_livecodebench.py — all helpers work without GPU."""

import json
import sys

import pytest


def _import():
    from ncpu.self_optimizing.run_livecodebench import (
        _build_prompt,
        _extract_entry_point,
        _is_stdin_problem,
        _norm_date,
        _outputs_match,
        _parse_test_cases,
        check_lcb_solution,
        extract_lcb_code,
    )
    return (
        _build_prompt,
        _extract_entry_point,
        _is_stdin_problem,
        _norm_date,
        _outputs_match,
        _parse_test_cases,
        check_lcb_solution,
        extract_lcb_code,
    )


# ---------------------------------------------------------------------------
# _norm_date
# ---------------------------------------------------------------------------

class TestNormDate:
    def test_strips_time(self):
        _, _, _, nd, _, _, _, _ = _import()
        assert nd("2023-05-13 00:00:00") == "2023-05-13"

    def test_already_date_only(self):
        _, _, _, nd, _, _, _, _ = _import()
        assert nd("2024-09-01") == "2024-09-01"

    def test_empty(self):
        _, _, _, nd, _, _, _, _ = _import()
        assert nd("") == ""


# ---------------------------------------------------------------------------
# _parse_test_cases
# ---------------------------------------------------------------------------

class TestParseTestCases:
    def test_basic_json(self):
        _, _, _, _, _, parse, _, _ = _import()
        row = {
            "public_test_cases": json.dumps([
                {"input": "3 5\n", "output": "8\n", "testtype": "stdin"},
            ]),
            "private_test_cases": json.dumps([
                {"input": "0 0\n", "output": "0\n", "testtype": "stdin"},
            ]),
        }
        cases = parse(row)
        assert len(cases) == 2
        assert cases[0]["testtype"] == "stdin"

    def test_empty_strings(self):
        _, _, _, _, _, parse, _, _ = _import()
        cases = parse({"public_test_cases": "", "private_test_cases": ""})
        assert cases == []

    def test_missing_fields(self):
        _, _, _, _, _, parse, _, _ = _import()
        cases = parse({})
        assert cases == []

    def test_invalid_json(self):
        _, _, _, _, _, parse, _, _ = _import()
        cases = parse({"public_test_cases": "not json"})
        assert cases == []


# ---------------------------------------------------------------------------
# _is_stdin_problem
# ---------------------------------------------------------------------------

class TestIsStdinProblem:
    def test_stdin_type(self):
        _, _, is_stdin, _, _, _, _, _ = _import()
        cases = [{"input": "1\n", "output": "1\n", "testtype": "stdin"}]
        assert is_stdin(cases) is True

    def test_non_stdin(self):
        _, _, is_stdin, _, _, _, _, _ = _import()
        cases = [{"input": [1, 2], "output": 3, "testtype": "functional"}]
        assert is_stdin(cases) is False

    def test_empty_cases(self):
        _, _, is_stdin, _, _, _, _, _ = _import()
        assert is_stdin([]) is True


# ---------------------------------------------------------------------------
# _extract_entry_point
# ---------------------------------------------------------------------------

class TestExtractEntryPoint:
    def test_class_solution_method(self):
        _, ep, _, _, _, _, _, _ = _import()
        assert ep("class Solution:\n    def twoSum(self, nums, target):\n") == "twoSum"

    def test_empty_string(self):
        _, ep, _, _, _, _, _, _ = _import()
        assert ep("") == "solve"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_codegen_stdin(self):
        bp, _, _, _, _, _, _, _ = _import()
        row = {
            "question_content": "Add two numbers.",
            "starter_code": None,
        }
        prompt = bp(row, "codegeneration", True)
        assert "Add two numbers." in prompt
        assert "stdin" in prompt

    def test_codegen_with_starter(self):
        bp, _, _, _, _, _, _, _ = _import()
        row = {
            "question_content": "Do X.",
            "starter_code": "class Solution:\n    def x(self, a):",
        }
        prompt = bp(row, "codegeneration", False)
        assert "class Solution" in prompt

    def test_selfrepair_stdin(self):
        bp, _, _, _, _, _, _, _ = _import()
        row = {"question_content": "Do X.", "starter_code": None}
        prompt = bp(row, "selfrepair", True)
        assert "bug" in prompt.lower()


# ---------------------------------------------------------------------------
# extract_lcb_code
# ---------------------------------------------------------------------------

class TestExtractLcbCode:
    def test_fenced_code(self):
        _, _, _, _, _, _, _, extract = _import()
        out = '```python\nn = int(input())\nprint(n * 2)\n```'
        code = extract(out)
        assert "n = int(input())" in code
        assert "```" not in code

    def test_bare_code(self):
        _, _, _, _, _, _, _, extract = _import()
        out = "n = int(input())\nprint(n * 2)"
        code = extract(out)
        assert "n = int(input())" in code

    def test_empty_input(self):
        _, _, _, _, _, _, _, extract = _import()
        assert extract("") == ""


# ---------------------------------------------------------------------------
# _outputs_match
# ---------------------------------------------------------------------------

class TestOutputsMatch:
    def test_exact_match(self):
        _, _, _, _, om, _, _, _ = _import()
        assert om("8\n", "8\n") is True

    def test_whitespace_normalized(self):
        _, _, _, _, om, _, _, _ = _import()
        assert om("8\n", "8") is True

    def test_numeric_close(self):
        _, _, _, _, om, _, _, _ = _import()
        assert om("3.14159\n", "3.14159\n") is True

    def test_mismatch(self):
        _, _, _, _, om, _, _, _ = _import()
        assert om("7\n", "8\n") is False

    def test_multiline_match(self):
        _, _, _, _, om, _, _, _ = _import()
        assert om("1\n2\n3\n", "1\n2\n3\n") is True


# ---------------------------------------------------------------------------
# check_lcb_solution
# ---------------------------------------------------------------------------

class TestCheckLcbSolution:
    def _stdin_problem(self, test_cases):
        return {
            "task_id": "test_1",
            "entry_point": "",
            "is_stdin": True,
            "test_cases": test_cases,
        }

    def _class_problem(self, test_cases, entry_point="add"):
        return {
            "task_id": "test_2",
            "entry_point": entry_point,
            "is_stdin": False,
            "test_cases": test_cases,
        }

    def test_stdin_correct(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._stdin_problem([
            {"input": "3 5\n", "output": "8\n", "testtype": "stdin"},
            {"input": "0 0\n", "output": "0\n", "testtype": "stdin"},
        ])
        code = "a, b = map(int, input().split())\nprint(a + b)"
        passed, err = check(problem, code)
        assert passed

    def test_stdin_wrong(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._stdin_problem([
            {"input": "3 5\n", "output": "8\n", "testtype": "stdin"},
        ])
        code = "print(0)"
        passed, err = check(problem, code)
        assert not passed

    def test_stdin_timeout(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._stdin_problem([
            {"input": "1\n", "output": "1\n", "testtype": "stdin"},
        ])
        code = "import time; time.sleep(60)"
        passed, err = check(problem, code, timeout_s=0.5)
        assert not passed
        assert "timeout" in err

    def test_class_solution_passes(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._class_problem([
            {"input": [1, 2], "output": 3, "testtype": "functional"},
        ])
        code = "class Solution:\n    def add(self, a, b):\n        return a + b"
        passed, err = check(problem, code)
        assert passed

    def test_class_solution_fails(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._class_problem([
            {"input": [1, 2], "output": 3, "testtype": "functional"},
        ])
        code = "class Solution:\n    def add(self, a, b):\n        return a - b"
        passed, err = check(problem, code)
        assert not passed

    def test_no_test_cases(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = {"task_id": "t", "entry_point": "", "is_stdin": True,
                    "test_cases": []}
        passed, err = check(problem, "pass")
        assert not passed
        assert "no test cases" in err

    def test_syntax_error_fails(self):
        _, _, _, _, _, _, check, _ = _import()
        problem = self._stdin_problem([
            {"input": "1\n", "output": "1\n", "testtype": "stdin"},
        ])
        passed, err = check(problem, "this is not valid python {")
        assert not passed


# ---------------------------------------------------------------------------
# Dry-run and config
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_no_library(self):
        from ncpu.self_optimizing.run_livecodebench import LiveCodeBenchConfig
        cfg = LiveCodeBenchConfig(dry_run=True, library_path=None)
        assert cfg.scenario == "codegeneration"
        assert cfg.start_date == ""
        assert cfg.model == "Qwen/Qwen3.5-4B"


class TestParseCli:
    def test_defaults(self):
        from ncpu.self_optimizing.run_livecodebench import parse_cli
        cfg = parse_cli([])
        assert cfg.model == "Qwen/Qwen3.5-4B"
        assert cfg.scenario == "codegeneration"
        assert cfg.start_date == ""
        assert cfg.release_version == "release_v6"

    def test_selfrepair_flag(self):
        from ncpu.self_optimizing.run_livecodebench import parse_cli
        cfg = parse_cli(["--scenario", "selfrepair"])
        assert cfg.scenario == "selfrepair"

    def test_date_filters(self):
        from ncpu.self_optimizing.run_livecodebench import parse_cli
        cfg = parse_cli(["--start-date", "2025-01-01", "--end-date", "2025-06-01"])
        assert cfg.start_date == "2025-01-01"
        assert cfg.end_date == "2025-06-01"

    def test_difficulty_lowercase(self):
        from ncpu.self_optimizing.run_livecodebench import parse_cli
        cfg = parse_cli(["--difficulty", "hard"])
        assert cfg.difficulty == "hard"
