"""Tests for user-prompt → WorkItem extraction."""

from __future__ import annotations

import unittest

from ncpu.autoresearch.prompt_parser import (
    build_work_item,
    extract_entry_point,
    extract_from_prompt,
)


ARROW_PROMPT = """
Write a function `reverse` that reverses a string.

reverse("hello") -> "olleh"
reverse("") -> ""
reverse("a") -> "a"

def reverse(s):
    pass
"""


DOCTEST_PROMPT = """
def capitalize_words(s):
    \"\"\"Capitalize every word.

    >>> capitalize_words("hello world")
    'Hello World'
    >>> capitalize_words("")
    ''
    \"\"\"
    pass
"""


ASSERT_PROMPT = """
Write a function `add` that returns the sum of two numbers. Example:

```python
assert add(1, 2) == 3
assert add(-1, 1) == 0
```
"""


RETURNS_PROSE_PROMPT = """
Please implement `shout(s)`. shout("hi") returns "HI". shout("ok ok") returns "OK OK".

def shout(s):
    pass
"""


NO_IO_PROMPT = """
Write me a function to parse dates. It should handle multiple formats.

def parse_date(s):
    pass
"""


class TestExtractEntryPoint(unittest.TestCase):
    def test_picks_last_def(self):
        src = "def foo(x):\n    pass\ndef bar(y):\n    pass\n"
        self.assertEqual(extract_entry_point(src), "bar")

    def test_none_when_no_def(self):
        self.assertIsNone(extract_entry_point("hi"))


class TestArrow(unittest.TestCase):
    def test_pulls_three_pairs(self):
        r = extract_from_prompt(ARROW_PROMPT)
        self.assertEqual(r.entry_point, "reverse")
        self.assertEqual(len(r.io_pairs), 3)
        self.assertEqual(r.io_pairs[0].args, ["hello"])
        self.assertEqual(r.io_pairs[0].expected, "olleh")
        self.assertGreater(r.sources.get("arrow", 0), 0)


class TestDoctest(unittest.TestCase):
    def test_doctest_lines(self):
        r = extract_from_prompt(DOCTEST_PROMPT)
        self.assertEqual(r.entry_point, "capitalize_words")
        self.assertEqual(len(r.io_pairs), 2)
        self.assertEqual(r.io_pairs[0].args, ["hello world"])
        self.assertEqual(r.io_pairs[0].expected, "Hello World")


class TestAssertInFence(unittest.TestCase):
    def test_assert_in_code_block(self):
        r = extract_from_prompt(ASSERT_PROMPT, entry_point="add")
        self.assertEqual(len(r.io_pairs), 2)
        self.assertEqual(r.io_pairs[0].args, [1, 2])
        self.assertEqual(r.io_pairs[0].expected, 3)


class TestReturnsProse(unittest.TestCase):
    def test_returns_shout(self):
        r = extract_from_prompt(RETURNS_PROSE_PROMPT)
        self.assertEqual(r.entry_point, "shout")
        self.assertEqual(len(r.io_pairs), 2)


class TestNoIO(unittest.TestCase):
    def test_returns_empty_pairs(self):
        r = extract_from_prompt(NO_IO_PROMPT)
        self.assertEqual(r.entry_point, "parse_date")
        self.assertEqual(len(r.io_pairs), 0)
        self.assertFalse(r.ok())


class TestDedupe(unittest.TestCase):
    def test_same_pair_not_duplicated(self):
        # Both arrow and returns-prose mention the same I/O pair.
        src = """
        def shout(s):
            pass

        shout("hi") -> "HI"
        shout("hi") returns "HI"
        """
        r = extract_from_prompt(src)
        self.assertEqual(len(r.io_pairs), 1)


class TestBuildWorkItem(unittest.TestCase):
    def test_full_work_item(self):
        wi = build_work_item(ARROW_PROMPT, task_id="user/reverse")
        self.assertIsNotNone(wi)
        self.assertEqual(wi.entry_point, "reverse")
        self.assertEqual(wi.source_benchmark, "user")
        self.assertEqual(len(wi.io_pairs), 3)
        self.assertIn("def check(candidate):", wi.test_source)
        self.assertIn("assert candidate('hello') == 'olleh'", wi.test_source)

    def test_no_entry_point_returns_none(self):
        wi = build_work_item("Some prose with no def.", task_id="x/0")
        self.assertIsNone(wi)

    def test_extra_io_pairs_appended(self):
        from ncpu.autoresearch.types import IoPair
        extra = [IoPair(args=["abc"], kwargs={}, expected="cba")]
        wi = build_work_item(ARROW_PROMPT, extra_io_pairs=extra)
        self.assertEqual(len(wi.io_pairs), 4)


class TestIntegrationWithCascade(unittest.TestCase):
    """Parser produces a WorkItem the cascade can actually solve."""

    def test_cascade_solves_reverse_via_parser(self):
        from ncpu.autoresearch.cascade import CascadeConfig, run_cascade

        wi = build_work_item(ARROW_PROMPT, task_id="user/reverse")
        self.assertIsNotNone(wi)
        # template_match doesn't have reverse-string; we expect cascade
        # to report unsolved. Testing the integration shape itself.
        cfg = CascadeConfig(solver_names=["template_match"])
        r = run_cascade(wi, config=cfg)
        self.assertFalse(r.solved)  # template library doesn't cover strings

    def test_cascade_solves_sum_via_parser(self):
        """A 2-arg sum fits the template library, so parser → cascade solves it."""
        from ncpu.autoresearch.cascade import CascadeConfig, run_cascade

        prompt = """
        def add(a, b):
            \"\"\"Return a+b.\"\"\"
        add(1, 2) -> 3
        add(10, -5) -> 5
        """
        wi = build_work_item(prompt, task_id="user/add")
        self.assertIsNotNone(wi)
        self.assertEqual(wi.entry_point, "add")
        cfg = CascadeConfig(solver_names=["template_match"])
        r = run_cascade(wi, config=cfg)
        self.assertTrue(r.solved, f"cascade did not solve; error={r.error}")


if __name__ == "__main__":
    unittest.main()
