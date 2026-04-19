"""Tests for autoresearch.miner — I/O extraction + JSONL emit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ncpu.autoresearch.miner import extract_io_pairs, load_queue
from ncpu.autoresearch.types import IoPair, WorkItem


SIMPLE_TEST = """
def check(candidate):
    assert candidate(1) == 1
    assert candidate(2) == 4
    assert candidate(3) == 9
    assert candidate([1, 2, 3]) == 6
"""


FLIPPED_TEST = """
def check(candidate):
    assert 42 == candidate(6, 7)
    assert 'x' == candidate('x')
"""


COMPLEX_TEST = """
def check(candidate):
    xs = [1, 2, 3]
    assert candidate(xs) == 6          # skipped: non-literal arg
    assert candidate(4, 5) == 9        # kept
    assert candidate([], 'y') == 'y'   # kept (literal empty list + str)
"""


class TestExtractIoPairs(unittest.TestCase):
    def test_plain_asserts(self):
        pairs = extract_io_pairs(SIMPLE_TEST, entry_point="square")
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs[0].args, [1])
        self.assertEqual(pairs[0].expected, 1)
        self.assertEqual(pairs[3].args, [[1, 2, 3]])
        self.assertEqual(pairs[3].expected, 6)

    def test_flipped_lhs_rhs(self):
        pairs = extract_io_pairs(FLIPPED_TEST, entry_point="add")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].args, [6, 7])
        self.assertEqual(pairs[0].expected, 42)

    def test_non_literal_args_skipped(self):
        pairs = extract_io_pairs(COMPLEX_TEST, entry_point="fn")
        # First assert uses xs variable → skipped.
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].args, [4, 5])
        self.assertEqual(pairs[1].args, [[], "y"])

    def test_handles_syntax_error(self):
        self.assertEqual(extract_io_pairs("not python :(", "x"), [])

    def test_handles_entry_point_direct(self):
        src = "def check(): assert my_fn(3) == 5"
        pairs = extract_io_pairs(src, entry_point="my_fn")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].args, [3])
        self.assertEqual(pairs[0].expected, 5)


class TestQueueRoundtrip(unittest.TestCase):
    def test_load_respects_priority(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "q.jsonl"
            items = [
                WorkItem(task_id="a", source_benchmark="humaneval", prompt="",
                         entry_point="a", test_source="", io_pairs=[], priority=0.5),
                WorkItem(task_id="b", source_benchmark="humaneval", prompt="",
                         entry_point="b", test_source="", io_pairs=[], priority=2.5),
                WorkItem(task_id="c", source_benchmark="humaneval", prompt="",
                         entry_point="c", test_source="", io_pairs=[], priority=1.5),
            ]
            with open(path, "w") as fh:
                for it in items:
                    fh.write(json.dumps(it.to_dict()) + "\n")
            loaded = load_queue(path)
            self.assertEqual([it.task_id for it in loaded], ["b", "c", "a"])

    def test_roundtrip_preserves_non_json_literals(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "q.jsonl"
            item = WorkItem(
                task_id="complex",
                source_benchmark="mbpp",
                prompt="def f(x):\n    pass\n",
                entry_point="f",
                test_source="assert f(1j) == (2+3j)\n",
                io_pairs=[IoPair(
                    args=[(1, 2), 1j],
                    kwargs={"z": (-2 + 0j)},
                    expected=(2 + 3j),
                )],
                priority=1.0,
            )
            with open(path, "w") as fh:
                fh.write(json.dumps(item.to_dict()) + "\n")
            loaded = load_queue(path)
            pair = loaded[0].io_pairs[0]
            self.assertEqual(pair.args[0], (1, 2))
            self.assertEqual(pair.args[1], 1j)
            self.assertEqual(pair.kwargs["z"], (-2 + 0j))
            self.assertEqual(pair.expected, (2 + 3j))


if __name__ == "__main__":
    unittest.main()
