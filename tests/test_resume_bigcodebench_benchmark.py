"""Regression tests for BigCodeBench resume metadata handling."""

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def import_resume_bigcodebench_module():
    """Import the resume runner with lightweight BigCodeBench stubs."""
    bigcodebench_module = types.ModuleType("bigcodebench")
    data_module = types.ModuleType("bigcodebench.data")
    data_utils_module = types.ModuleType("bigcodebench.data.utils")
    eval_module = types.ModuleType("bigcodebench.eval")
    gen_module = types.ModuleType("bigcodebench.gen")
    gen_util_module = types.ModuleType("bigcodebench.gen.util")
    sanitize_module = types.ModuleType("bigcodebench.sanitize")

    data_module.get_bigcodebench = lambda subset="hard": {}
    data_module.get_bigcodebench_hash = lambda subset="hard": "stub-hash"
    data_utils_module.CACHE_DIR = tempfile.gettempdir()
    eval_module.PASS = "pass"
    eval_module.untrusted_check = lambda *args, **kwargs: ("pass", {})
    gen_util_module.trusted_check = lambda *args, **kwargs: {"task_id": "stub", "time": 1.0}
    sanitize_module.sanitize = lambda raw_output, entrypoint=None: raw_output

    bigcodebench_module.data = data_module
    bigcodebench_module.eval = eval_module
    bigcodebench_module.gen = gen_module
    bigcodebench_module.sanitize = sanitize_module
    data_module.utils = data_utils_module
    gen_module.util = gen_util_module

    stub_modules = {
        "bigcodebench": bigcodebench_module,
        "bigcodebench.data": data_module,
        "bigcodebench.data.utils": data_utils_module,
        "bigcodebench.eval": eval_module,
        "bigcodebench.gen": gen_module,
        "bigcodebench.gen.util": gen_util_module,
        "bigcodebench.sanitize": sanitize_module,
    }

    with patch.dict(sys.modules, stub_modules):
        sys.modules.pop("ncpu.self_optimizing.run_bigcodebench_benchmark", None)
        sys.modules.pop("ncpu.self_optimizing.resume_bigcodebench_benchmark", None)
        return importlib.import_module("ncpu.self_optimizing.resume_bigcodebench_benchmark")


class TestResumeBigCodeBenchBenchmark(unittest.TestCase):
    """Verify resumed BigCodeBench runs keep canonical task identifiers."""

    def test_load_checkpoint_maps_placeholder_names_by_task_index(self):
        module = import_resume_bigcodebench_module()
        task_order = ["BigCodeBench/13", "BigCodeBench/15", "BigCodeBench/21"]

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "checkpoint.jsonl"
            record = {
                "event": "task_complete",
                "approach": "standard",
                "task_index": 2,
                "task_name": "task_2",
                "task_result": {
                    "name": "task_2",
                    "category": "bigcodebench-hard-instruct",
                    "success": True,
                    "attempts": 1,
                    "elapsed_seconds": 1.0,
                    "final_error": None,
                    "attempt_details": [],
                },
            }
            progress_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            checkpoint = module.load_checkpoint(str(progress_path), task_order)

        self.assertEqual(len(checkpoint["standard"]), 1)
        self.assertEqual(checkpoint["standard"][0]["name"], "BigCodeBench/15")

    def test_progress_callback_rewrites_placeholder_names(self):
        module = import_resume_bigcodebench_module()
        task_order = [f"BigCodeBench/{index}" for index in range(1, 149)]

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.jsonl"
            callback = module.make_resuming_progress_callback(
                model="qwen3.5:9b",
                subset="hard",
                split="instruct",
                progress_path=str(progress_path),
                approach="some",
                index_offset=33,
                total_tasks=148,
                initial_passes=15,
                task_order=task_order,
            )

            callback(
                {
                    "event": "task_complete",
                    "task_index": 1,
                    "task_total": 115,
                    "task_name": "task_1",
                    "success": True,
                    "attempts": 2,
                    "elapsed_seconds": 12.5,
                    "task_result": {
                        "name": "task_1",
                        "success": True,
                        "attempts": 2,
                        "elapsed_seconds": 12.5,
                        "final_error": None,
                        "attempt_details": [],
                    },
                }
            )

            written = json.loads(progress_path.read_text(encoding="utf-8").strip())

        self.assertEqual(written["task_index"], 34)
        self.assertEqual(written["task_name"], "BigCodeBench/34")
        self.assertEqual(written["task_result"]["name"], "BigCodeBench/34")
        self.assertAlmostEqual(written["running_success_rate"], 16 / 34)

    def test_normalize_progress_file_repairs_placeholder_rows_in_place(self):
        module = import_resume_bigcodebench_module()
        task_order = [f"BigCodeBench/{index}" for index in range(1, 6)]

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "progress.jsonl"
            rows = [
                {
                    "event": "task_start",
                    "approach": "some",
                    "task_index": 4,
                    "task_name": "task_4",
                },
                {
                    "event": "task_complete",
                    "approach": "some",
                    "task_index": 4,
                    "task_name": "task_4",
                    "task_result": {
                        "name": "task_4",
                        "success": False,
                        "attempts": 3,
                        "elapsed_seconds": 20.0,
                        "final_error": "invalid syntax",
                        "attempt_details": [],
                    },
                },
            ]
            progress_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            module.normalize_progress_file(str(progress_path), task_order)
            repaired = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(repaired[0]["task_name"], "BigCodeBench/4")
        self.assertEqual(repaired[1]["task_name"], "BigCodeBench/4")
        self.assertEqual(repaired[1]["task_result"]["name"], "BigCodeBench/4")

    def test_load_checkpoint_ignores_invalid_infrastructure_tail_failures(self):
        module = import_resume_bigcodebench_module()
        task_order = ["BigCodeBench/13", "BigCodeBench/15"]

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = Path(tmpdir) / "checkpoint.jsonl"
            rows = [
                {
                    "event": "task_complete",
                    "approach": "some",
                    "task_index": 1,
                    "task_name": "BigCodeBench/13",
                    "task_result": {
                        "name": "BigCodeBench/13",
                        "category": "bigcodebench-hard-instruct",
                        "success": True,
                        "attempts": 1,
                        "elapsed_seconds": 4.0,
                        "final_error": None,
                        "attempt_details": [],
                    },
                },
                {
                    "event": "task_complete",
                    "approach": "some",
                    "task_index": 2,
                    "task_name": "BigCodeBench/15",
                    "task_result": {
                        "name": "BigCodeBench/15",
                        "category": "bigcodebench-hard-instruct",
                        "success": False,
                        "attempts": 0,
                        "elapsed_seconds": 0.0,
                        "final_error": (
                            "buffered controller exception: ConnectionError: HTTPConnectionPool("
                            "host='localhost', port=11434): Max retries exceeded"
                        ),
                        "attempt_details": [],
                    },
                },
            ]
            progress_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            checkpoint = module.load_checkpoint(str(progress_path), task_order)

        self.assertEqual([task["name"] for task in checkpoint["some"]], ["BigCodeBench/13"])


if __name__ == "__main__":
    unittest.main()
