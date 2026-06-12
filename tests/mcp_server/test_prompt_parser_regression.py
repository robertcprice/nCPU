"""Regression: prompt_parser must accept list-valued arguments.

Before the fix, ``extract_from_prompt``'s ``_accept`` built its dedupe
key as ``tuple(p.args)`` — ``_freeze`` was applied to ``expected`` but
not to ``args``, so a list-valued arg produced an unhashable tuple and
``key in seen`` raised ``TypeError``.
"""

from __future__ import annotations

from ncpu.autoresearch.prompt_parser import build_work_item, extract_from_prompt
from ncpu.autoresearch.types import IoPair


def test_list_args_arrow_extraction():
    report = extract_from_prompt(
        "Write a function sum_list. sum_list([1,2,3]) -> 6"
    )
    assert len(report.io_pairs) == 1
    assert report.io_pairs[0].args == [[1, 2, 3]]
    assert report.io_pairs[0].expected == 6


def test_list_args_dedupe_still_works():
    # The same pair restated twice must dedupe to one, not crash.
    report = extract_from_prompt(
        "sum_list([1,2,3]) -> 6\n"
        "sum_list([1,2,3]) -> 6\n"
        "sum_list([4,5]) -> 9\n"
    )
    assert len(report.io_pairs) == 2
    assert [p.expected for p in report.io_pairs] == [6, 9]


def test_list_args_with_dict_and_nested_values():
    report = extract_from_prompt(
        "lookup({'a': 1}, [['x', 2]]) -> 3"
    )
    assert len(report.io_pairs) == 1
    assert report.io_pairs[0].args == [{"a": 1}, [["x", 2]]]


def test_build_work_item_dedupes_list_args_extras():
    item = build_work_item(
        "def sum_list(xs):\n    pass\n\nsum_list([1,2,3]) -> 6",
        extra_io_pairs=[
            IoPair(args=[[1, 2, 3]], kwargs={}, expected=6),  # duplicate
            IoPair(args=[[7]], kwargs={}, expected=7),  # new
        ],
    )
    assert item is not None
    assert len(item.io_pairs) == 2
