"""Tests for the autoresearch → NPCoT-library compounding loop (distiller side).

Covers:

* :func:`translate_to_5tuple` — finds the right discrete program for known
  array-reduction behaviors, refuses inexpressible / wrong-shape behaviors.
* :func:`execute_program_pure` — the pure-Python executor mirror matches the
  torch ``DiscreteArrayProgram.execute`` on random programs/inputs.
* :func:`distill_solved` — offline pass writes pending entries to the sidecar
  and never corrupts the library JSON; ``ArrayProgramLibrary`` round-trips.
* ``cli distill`` subcommand smoke test.
* Documents why pending entries live in a sidecar: ``ArrayProgramLibrary.load``
  crashes on ``signature: null`` entries.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.distiller import (
    PENDING_DISTILL_NAME,
    PROGRAM_SPACE_SIZE,
    _PROBE_ARRAYS,
    append_solved,
    distill_solved,
    execute_program_pure,
    io_pair_int_array,
    translate_to_5tuple,
)
from ncpu.autoresearch.types import IoPair, SolvedItem, WorkItem


def _solved(source: str, task_id: str = "h/0", solver: str = "llm_resample") -> SolvedItem:
    return SolvedItem(
        task_id=task_id,
        source_benchmark="humaneval",
        solver=solver,
        program_python=source,
    )


def _pairs(fn, arrays) -> list[IoPair]:
    return [IoPair(args=[list(a)], kwargs={}, expected=fn(list(a))) for a in arrays]


def _work_item(task_id: str, prompt: str, entry_point: str, io_pairs) -> WorkItem:
    return WorkItem(
        task_id=task_id,
        source_benchmark="humaneval",
        prompt=prompt,
        entry_point=entry_point,
        test_source="",
        io_pairs=io_pairs,
    )


class TestTranslateTo5Tuple(unittest.TestCase):
    def test_program_space_size(self):
        # init(3) x transform(6) x reduce(4) x post_scale(3) discrete shapes.
        self.assertEqual(PROGRAM_SPACE_SIZE, 216)

    def test_probe_set_coverage(self):
        self.assertGreaterEqual(len(_PROBE_ARRAYS), 30)
        self.assertTrue(any(len(p) == 1 for p in _PROBE_ARRAYS))           # single-element
        self.assertTrue(any(min(p) < 0 for p in _PROBE_ARRAYS))            # negatives
        self.assertTrue(any(0 in p for p in _PROBE_ARRAYS))                # zeros
        self.assertTrue(any(len(set(p)) < len(p) for p in _PROBE_ARRAYS))  # duplicates
        self.assertTrue(any(list(p) == sorted(p) and len(p) > 2 for p in _PROBE_ARRAYS))
        self.assertTrue(any(list(p) != sorted(p) for p in _PROBE_ARRAYS))  # unsorted

    def test_translates_sum(self):
        five = translate_to_5tuple(
            _solved("def total(arr):\n    return sum(arr)\n"),
            _pairs(sum, [[1, 2, 3], [0, -4]]),
            entry_point="total",
        )
        self.assertEqual(five, {
            "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
            "post_scale_idx": 0, "offset": 0.0,
        })

    def test_translates_max(self):
        five = translate_to_5tuple(
            _solved("def biggest(arr):\n    return max(arr)\n"),
            _pairs(max, [[1, 9, 3], [-5, -2]]),
            entry_point="biggest",
        )
        # init "-large" (idx 2), transform x, reduce max, post acc.
        self.assertEqual(five, {
            "init_idx": 2, "transform_idx": 0, "reduce_idx": 2,
            "post_scale_idx": 0, "offset": 0.0,
        })

    def test_translates_count_positives(self):
        behavior = lambda a: sum(1 for x in a if x > 0)  # noqa: E731
        five = translate_to_5tuple(
            _solved("def cp(arr):\n    return sum(1 for x in arr if x > 0)\n"),
            _pairs(behavior, [[1, -2, 3], [0, 0]]),
            entry_point="cp",
        )
        # transform 1{x>0} (idx 4), reduce +.
        self.assertEqual(five, {
            "init_idx": 0, "transform_idx": 4, "reduce_idx": 0,
            "post_scale_idx": 0, "offset": 0.0,
        })

    def test_translates_mean_with_post_scale(self):
        behavior = lambda a: sum(a) / len(a)  # noqa: E731
        five = translate_to_5tuple(
            _solved("def mean(arr):\n    return sum(arr) / len(arr)\n"),
            _pairs(behavior, [[1, 2, 3], [4]]),
            entry_point="mean",
        )
        self.assertEqual(five["post_scale_idx"], 1)  # acc/len

    def test_translates_length_plus_constant_offset(self):
        behavior = lambda a: len(a) + 3  # noqa: E731
        five = translate_to_5tuple(
            _solved("def f(arr):\n    return len(arr) + 3\n"),
            _pairs(behavior, [[1, 2], [0]]),
            entry_point="f",
        )
        self.assertEqual(five["transform_idx"], 3)  # constant-1 transform
        self.assertEqual(five["offset"], 3.0)

    def test_translates_body_style_with_prompt(self):
        # HumanEval-style: program_python is a continuation of the prompt.
        prompt = (
            "from typing import List\n\n\n"
            "def add_all(arr: List[int]) -> int:\n"
            '    """Return the sum of all elements."""\n'
        )
        five = translate_to_5tuple(
            _solved("    return sum(arr)\n"),
            _pairs(sum, [[1, 2, 3]]),
            prompt=prompt,
            entry_point="add_all",
        )
        self.assertIsNotNone(five)
        self.assertEqual(five["reduce_idx"], 0)

    def test_refuses_string_behavior(self):
        five = translate_to_5tuple(
            _solved("def rev(s):\n    return s[::-1]\n"),
            [IoPair(args=["abc"], kwargs={}, expected="cba")],
            entry_point="rev",
        )
        self.assertIsNone(five)

    def test_refuses_inexpressible_min(self):
        # min(arr) is NOT expressible: init choices are {0, 1, -20}, so a
        # min-reduce can never reproduce min over all-positive probe arrays.
        five = translate_to_5tuple(
            _solved("def smallest(arr):\n    return min(arr)\n"),
            _pairs(min, [[3, 1, 2]]),
            entry_point="smallest",
        )
        self.assertIsNone(five)

    def test_refuses_multi_arg_and_kwargs(self):
        five = translate_to_5tuple(
            _solved("def f(a, b):\n    return a + b\n"),
            [IoPair(args=[1, 2], kwargs={}, expected=3)],
            entry_point="f",
        )
        self.assertIsNone(five)
        five = translate_to_5tuple(
            _solved("def f(arr, k=0):\n    return sum(arr) + k\n"),
            [IoPair(args=[[1]], kwargs={"k": 2}, expected=3)],
            entry_point="f",
        )
        self.assertIsNone(five)

    def test_refuses_disallowed_import(self):
        five = translate_to_5tuple(
            _solved("import os\ndef f(arr):\n    return sum(arr)\n"),
            _pairs(sum, [[1, 2]]),
            entry_point="f",
        )
        self.assertIsNone(five)

    def test_allows_typing_import(self):
        five = translate_to_5tuple(
            _solved(
                "from typing import List\n"
                "def f(arr: List[int]) -> int:\n    return sum(arr)\n"
            ),
            _pairs(sum, [[1, 2]]),
            entry_point="f",
        )
        self.assertIsNotNone(five)

    def test_refuses_callable_io_pair_mismatch(self):
        # Recorded expected disagrees with the callable → wrong entry point
        # or stale pair; refuse rather than distill a lie.
        five = translate_to_5tuple(
            _solved("def f(arr):\n    return sum(arr)\n"),
            [IoPair(args=[[1, 2]], kwargs={}, expected=99)],
            entry_point="f",
        )
        self.assertIsNone(five)

    def test_io_pair_int_array_gate(self):
        self.assertEqual(io_pair_int_array(IoPair(args=[[1, 2]], kwargs={}, expected=3)), [1, 2])
        self.assertIsNone(io_pair_int_array(IoPair(args=["ab"], kwargs={}, expected=0)))
        self.assertIsNone(io_pair_int_array(IoPair(args=[[1.5]], kwargs={}, expected=0)))
        self.assertIsNone(io_pair_int_array(IoPair(args=[[True]], kwargs={}, expected=0)))
        self.assertIsNone(io_pair_int_array(IoPair(args=[[1]], kwargs={"k": 1}, expected=0)))


class TestPureExecutorMirrorsTorch(unittest.TestCase):
    """Cross-check execute_program_pure against the torch executor."""

    def test_constants_match_head_module(self):
        from ncpu.self_optimizing.array_executable_thought_head import (
            _ELEM_TRANSFORMS,
            _INIT_CHOICES,
            _LOG_EPS,
            _NEG_LARGE,
            _POST_SCALES,
            _REDUCE_OPS,
        )
        from ncpu.autoresearch import distiller

        self.assertEqual(distiller._N_INIT, len(_INIT_CHOICES))
        self.assertEqual(distiller._N_TRANSFORM, len(_ELEM_TRANSFORMS))
        self.assertEqual(distiller._N_REDUCE, len(_REDUCE_OPS))
        self.assertEqual(distiller._N_POST_SCALE, len(_POST_SCALES))
        self.assertEqual(distiller._PURE_LOG_EPS, _LOG_EPS)
        self.assertEqual(distiller._PURE_INIT_VALUES[2], _NEG_LARGE)

    def test_mirror_matches_torch_executor_on_random_programs(self):
        import math

        import torch

        from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram

        rng = random.Random(1234)
        checked = 0
        for _ in range(80):
            program = {
                "init_idx": rng.randrange(3),
                "transform_idx": rng.randrange(6),
                "reduce_idx": rng.randrange(4),
                "post_scale_idx": rng.randrange(3),
                "offset": rng.choice([0.0, 1.5, -2.0, 0.25]),
            }
            length = rng.randint(1, 8)
            values = [rng.randint(-9, 9) for _ in range(length)]

            torch_program = DiscreteArrayProgram.from_dict(program)
            arrays = torch.zeros(1, 8)
            arrays[0, :length] = torch.tensor(values, dtype=torch.float32)
            lengths = torch.tensor([length])
            got_torch = float(torch_program.execute(arrays, lengths).item())

            got_pure = execute_program_pure(program, values)
            self.assertTrue(
                math.isclose(got_torch, got_pure, rel_tol=1e-3, abs_tol=1e-3),
                msg=f"mismatch for {program} on {values}: "
                    f"torch={got_torch} pure={got_pure}",
            )
            checked += 1
        self.assertEqual(checked, 80)

    def test_translated_program_executes_on_torch(self):
        import torch

        from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram

        five = translate_to_5tuple(
            _solved("def total(arr):\n    return sum(arr)\n"),
            _pairs(sum, [[1, 2, 3]]),
            entry_point="total",
        )
        program = DiscreteArrayProgram.from_dict(five)
        arrays = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3])
        self.assertEqual(float(program.execute(arrays, lengths).item()), 6.0)


class TestDistillSolved(unittest.TestCase):
    def _items_and_work(self):
        sum_item = _solved("def total(arr):\n    return sum(arr)\n", task_id="h/1")
        rev_item = _solved("def rev(s):\n    return s[::-1]\n", task_id="h/2")
        work = {
            "h/1": _work_item(
                "h/1", "", "total", _pairs(sum, [[1, 2, 3], [0, -4]])
            ),
            "h/2": _work_item(
                "h/2", "", "rev",
                [IoPair(args=["abc"], kwargs={}, expected="cba")],
            ),
        }
        return [sum_item, rev_item], work

    def test_distill_writes_pending_sidecar_and_preserves_library(self):
        import torch

        from ncpu.self_optimizing.array_program_library import (
            ArrayProgramLibrary,
            DiscreteArrayProgram,
        )

        with tempfile.TemporaryDirectory() as td:
            library_path = Path(td) / "library.json"
            library = ArrayProgramLibrary()
            torch.manual_seed(7)
            library.record(
                torch.randn(8),
                DiscreteArrayProgram(
                    init_idx=0, transform_idx=0, reduce_idx=0,
                    post_scale_idx=0, offset=0.0,
                ),
                task_name="preexisting",
            )
            library.save(library_path)

            items, work = self._items_and_work()
            summary = distill_solved(items, library_path, work_items=work)

            self.assertEqual(summary["total_items"], 2)
            self.assertEqual(summary["translated"], 1)
            self.assertEqual(summary["refused"], 1)
            self.assertEqual(summary["translated_task_ids"], ["h/1"])
            self.assertEqual(summary["library_entries"], 1)

            # Sidecar present, entry well-formed and loadable as a program.
            pending_path = Path(summary["pending_path"])
            self.assertEqual(pending_path.name, PENDING_DISTILL_NAME)
            self.assertEqual(pending_path.parent, library_path.parent)
            payload = json.loads(pending_path.read_text())
            entry = payload["pending"]["h/1"]
            self.assertIsNone(entry["signature"])
            self.assertTrue(entry["pending_signature"])
            program = DiscreteArrayProgram.from_dict(entry["program"])
            self.assertEqual(program.key(), (0, 0, 0, 0))

            # The library JSON itself was never touched: still loads, still
            # exactly the pre-existing entry.
            reloaded = ArrayProgramLibrary.load(library_path)
            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded.entries[0].task_name, "preexisting")

    def test_distill_merge_dedupes_by_task_id(self):
        with tempfile.TemporaryDirectory() as td:
            library_path = Path(td) / "library.json"  # does not exist
            items, work = self._items_and_work()
            first = distill_solved(items, library_path, work_items=work)
            second = distill_solved(items, library_path, work_items=work)
            self.assertEqual(first["pending_total"], 1)
            self.assertEqual(second["pending_total"], 1)
            self.assertIsNone(second["library_entries"])

    def test_distill_reuses_existing_program_5tuple(self):
        with tempfile.TemporaryDirectory() as td:
            library_path = Path(td) / "library.json"
            item = _solved("not even python {", task_id="h/9")
            item.program_5tuple = {
                "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                "post_scale_idx": 0, "offset": 0.0,
            }
            summary = distill_solved([item], library_path)
            self.assertEqual(summary["translated"], 1)

    def test_library_load_rejects_null_signature(self):
        """Documents the sidecar decision: ArrayProgramLibrary does NOT
        tolerate ``signature: null`` entries, so signature-less offline
        distillation cannot write into the library JSON."""
        from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "library.json"
            path.write_text(json.dumps({
                "config": {},
                "entries": [{
                    "signature": None,
                    "program": {
                        "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                        "post_scale_idx": 0, "offset": 0.0,
                    },
                }],
            }))
            with self.assertRaises(TypeError):
                ArrayProgramLibrary.load(path)


class TestCliDistill(unittest.TestCase):
    def test_distill_subcommand_smoke(self):
        from ncpu.autoresearch import cli

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            solved_path = td_path / "solved_programs.jsonl"
            queue_path = td_path / "humaneval_queue.jsonl"
            library_path = td_path / "library.json"

            append_solved(
                _solved("def total(arr):\n    return sum(arr)\n", task_id="h/1"),
                out_path=solved_path,
            )
            work = _work_item("h/1", "", "total", _pairs(sum, [[1, 2, 3]]))
            queue_path.write_text(json.dumps(work.to_dict()) + "\n")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = cli.main([
                    "distill",
                    "--solved", str(solved_path),
                    "--library", str(library_path),
                    "--queue", str(queue_path),
                ])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["translated"], 1)
            self.assertTrue((td_path / PENDING_DISTILL_NAME).exists())

    def test_distill_subcommand_missing_solved(self):
        from ncpu.autoresearch import cli

        with tempfile.TemporaryDirectory() as td:
            rc = cli.main([
                "distill",
                "--solved", str(Path(td) / "nope.jsonl"),
                "--library", str(Path(td) / "library.json"),
            ])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
