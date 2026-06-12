"""Unit tests for the Python mirror of nsynth's solved-cache fingerprint.

The string layouts here are pinned to ``nsynth/src/solved_cache.rs``
(``fingerprint_value`` / ``examples_fingerprint`` / ``encode_code``).
The end-to-end cross-check against the Rust side lives in
``test_mcp_server.py::test_consult_library_hit_after_synthesis``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ncpu.mcp_server.fingerprint import (
    decode_code,
    examples_fingerprint,
    fingerprint_value,
    lookup_solved,
)


def test_fingerprint_values():
    assert fingerprint_value(42) == "i:42"
    assert fingerprint_value(-7) == "i:-7"
    assert fingerprint_value("hi") == "s:hi"
    assert fingerprint_value("a|b~c") == "s:a\\|b\\~c"
    assert fingerprint_value([1, 2, 3]) == "a:[1,2,3]"
    assert fingerprint_value([]) == "a:[]"
    assert fingerprint_value((3, 4)) == "p:(3,4)"


def test_fingerprint_rejects_bools_and_unknowns():
    with pytest.raises(ValueError):
        fingerprint_value(True)
    with pytest.raises(ValueError):
        fingerprint_value(1.5)


def test_examples_fingerprint_matches_rust_layout():
    # Layout observed in a real bank written by the Rust binary.
    examples = [
        {"inputs": [1, 2], "expected": 3},
        {"inputs": [5, 7], "expected": 12},
        {"inputs": [0, 0], "expected": 0},
    ]
    assert examples_fingerprint(examples) == "i:1|i:2~3;;i:5|i:7~12;;i:0|i:0~0"


def test_examples_fingerprint_mixed_types():
    examples = [{"inputs": [[1, 2], "x", 5], "expected": -1}]
    assert examples_fingerprint(examples) == "a:[1,2]|s:x|i:5~-1"


def test_decode_code_roundtrip():
    assert decode_code("fn f() {\\n\\treturn 1;\\n}") == "fn f() {\n\treturn 1;\n}"
    assert decode_code("a\\\\b") == "a\\b"
    assert decode_code("plain") == "plain"


def test_lookup_solved_old_and_new_record_formats(tmp_path: Path):
    bank = tmp_path / "solved.json"
    bank.write_text(
        "i:1~2\told_method\tfn old() {\\n}\n"
        "i:3~4\tnew_method\t5\t1700000000\tfn new() {\\n}\n",
        encoding="utf-8",
    )
    old = lookup_solved(bank, "i:1~2")
    assert old is not None
    assert old["method"] == "old_method"
    assert old["code"] == "fn old() {\n}"
    assert old["success_count"] == 0

    new = lookup_solved(bank, "i:3~4")
    assert new is not None
    assert new["method"] == "new_method"
    assert new["success_count"] == 5
    assert new["last_used_at"] == 1700000000

    assert lookup_solved(bank, "i:9~9") is None
    assert lookup_solved(tmp_path / "missing.json", "i:1~2") is None
    assert lookup_solved(None, "i:1~2") is None
