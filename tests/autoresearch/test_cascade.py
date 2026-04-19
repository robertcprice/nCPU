"""Tests for solver cascade + template solver."""

from __future__ import annotations

import unittest

from ncpu.autoresearch.cascade import CascadeConfig, run_cascade
from ncpu.autoresearch.solvers import _parse_params, template_match
from ncpu.autoresearch.types import IoPair, WorkItem


SUM_PROMPT = """def total(xs):
    \"\"\"Return the sum of xs.\"\"\"
"""

SUM_TEST = """
def check(candidate):
    assert candidate([]) == 0
    assert candidate([1, 2, 3]) == 6
    assert candidate([-1, -2, -3]) == -6
"""


ADD_PROMPT = """def add(a, b):
    \"\"\"Return a+b.\"\"\"
"""

ADD_TEST = """
def check(candidate):
    assert candidate(1, 2) == 3
    assert candidate(10, -5) == 5
"""


MBPP_ADD_TEST = """
assert add(1, 2) == 3
assert add(10, -5) == 5
"""


class TestParseParams(unittest.TestCase):
    def test_two_params(self):
        self.assertEqual(_parse_params(ADD_PROMPT, "add"), ["a", "b"])

    def test_typed_params(self):
        src = "def f(x: int, y: list) -> int:\n    pass\n"
        self.assertEqual(_parse_params(src, "f"), ["x", "y"])


class TestTemplateMatch(unittest.TestCase):
    def _item(self, prompt, test, entry, pairs):
        return WorkItem(
            task_id="t/0",
            source_benchmark="humaneval",
            prompt=prompt,
            entry_point=entry,
            test_source=test,
            io_pairs=pairs,
            priority=1.0,
        )

    def test_sum_solves(self):
        pairs = [
            IoPair(args=[[]], kwargs={}, expected=0),
            IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),
        ]
        item = self._item(SUM_PROMPT, SUM_TEST, "total", pairs)
        body = template_match(item, budget_seconds=2.0)
        self.assertIsNotNone(body)
        self.assertIn("sum(xs)", body)

    def test_add_solves(self):
        pairs = [
            IoPair(args=[1, 2], kwargs={}, expected=3),
            IoPair(args=[10, -5], kwargs={}, expected=5),
        ]
        item = self._item(ADD_PROMPT, ADD_TEST, "add", pairs)
        body = template_match(item, budget_seconds=2.0)
        self.assertIsNotNone(body)
        self.assertIn("a + b", body)

    def test_unmatched_returns_none(self):
        pairs = [
            IoPair(args=["hello"], kwargs={}, expected="HELLO"),
        ]
        item = self._item("def up(s): pass\n", "", "up", pairs)
        self.assertIsNone(template_match(item, budget_seconds=1.0))


class TestCascade(unittest.TestCase):
    def test_first_passing_solver_wins(self):
        # Craft an item that template_match solves cleanly.
        pairs = [
            IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),
            IoPair(args=[[]], kwargs={}, expected=0),
        ]
        item = WorkItem(
            task_id="hv/sum", source_benchmark="humaneval",
            prompt=SUM_PROMPT, entry_point="total",
            test_source=SUM_TEST, io_pairs=pairs, priority=1.0,
        )
        cfg = CascadeConfig(solver_names=["template_match"])
        r = run_cascade(item, config=cfg)
        self.assertTrue(r.solved)
        self.assertEqual(r.solver, "template_match")
        self.assertIsNotNone(r.solved_item)
        self.assertIn("sum(xs)", r.solved_item.program_python)

    def test_no_solver_returns_not_solved(self):
        item = WorkItem(
            task_id="hv/weird", source_benchmark="humaneval",
            prompt=SUM_PROMPT, entry_point="total",
            test_source=SUM_TEST,
            io_pairs=[IoPair(args=[[1, 2, 3]], kwargs={}, expected=42)],
            priority=1.0,
        )
        cfg = CascadeConfig(solver_names=["template_match"])
        r = run_cascade(item, config=cfg)
        self.assertFalse(r.solved)

    def test_custom_solver_via_extra(self):
        hits = {"called": False}
        def always_sum(item, *, budget_seconds):
            hits["called"] = True
            return "    return sum(xs)\n"
        pairs = [IoPair(args=[[1, 2]], kwargs={}, expected=3)]
        item = WorkItem(
            task_id="hv/x", source_benchmark="humaneval",
            prompt=SUM_PROMPT, entry_point="total",
            test_source=SUM_TEST, io_pairs=pairs, priority=1.0,
        )
        cfg = CascadeConfig(
            solver_names=["custom"],
            extra_solvers={"custom": always_sum},
        )
        r = run_cascade(item, config=cfg)
        self.assertTrue(hits["called"])
        self.assertTrue(r.solved)

    def test_mbpp_style_top_level_asserts_verify(self):
        pairs = [
            IoPair(args=[1, 2], kwargs={}, expected=3),
            IoPair(args=[10, -5], kwargs={}, expected=5),
        ]
        item = WorkItem(
            task_id="mbpp/add",
            source_benchmark="mbpp",
            prompt=ADD_PROMPT,
            entry_point="add",
            test_source=MBPP_ADD_TEST,
            io_pairs=pairs,
            priority=1.0,
        )
        cfg = CascadeConfig(solver_names=["template_match"])
        r = run_cascade(item, config=cfg)
        self.assertTrue(r.solved)
        self.assertEqual(r.solver, "template_match")


if __name__ == "__main__":
    unittest.main()
