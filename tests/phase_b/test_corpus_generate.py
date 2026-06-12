"""Tests for the Phase B corpus generator (ncpu/phase_b).

Covers the schema/store round-trip, the egdc-factory generator (every
emitted Mog record must be verified in-process against its own test
cases), the autoresearch joiner, the stats summary, and the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncpu.phase_b.corpus import (
    CorpusRecord,
    read_corpus,
    summarize,
    write_corpus,
)
from ncpu.phase_b.generate import (
    autoresearch_records,
    egdc_records,
    main,
)

REPO = Path(__file__).resolve().parents[2]


def _has_egdc() -> bool:
    try:
        import egdc.mog.benchmark  # noqa: F401
        import egdc.mog.lang.interpreter  # noqa: F401

        return True
    except ImportError:
        return False


egdc_required = pytest.mark.skipif(
    not _has_egdc(), reason="egdc.mog package not present"
)


# ---------------------------------------------------------------------------
# schema + store
# ---------------------------------------------------------------------------

def _rec(**kw) -> CorpusRecord:
    base = dict(
        source="egdc_factory",
        task_id="add_two_v0",
        nl_prompt="Return the sum.",
        entry_point="add_two",
        signature="fn add_two(a: i64, b: i64) -> i64",
        program_lang="mog",
        program_src="fn add_two(a: i64, b: i64) -> i64 { return a + b; }",
        io_pairs=[{"inputs": [1, 2], "expected": 3}],
        category="arithmetic",
        verified=True,
    )
    base.update(kw)
    return CorpusRecord(**base)


def test_record_roundtrip(tmp_path: Path):
    recs = [_rec(), _rec(task_id="x", verified=False)]
    out = tmp_path / "c.jsonl"
    n = write_corpus(recs, out)
    assert n == 2
    back = read_corpus(out)
    assert len(back) == 2
    assert back[0].to_dict() == recs[0].to_dict()


def test_from_dict_ignores_unknown_keys():
    rec = CorpusRecord.from_dict({**_rec().to_dict(), "stray": 99})
    assert rec.task_id == "add_two_v0"


def test_summarize_counts():
    stats = summarize([_rec(), _rec(program_lang="python", source="autoresearch_humaneval", verified=False)])
    assert stats.total == 2
    assert stats.verified == 1
    assert stats.by_lang == {"mog": 1, "python": 1}
    assert stats.total_io_pairs == 2


# ---------------------------------------------------------------------------
# egdc-factory generator
# ---------------------------------------------------------------------------

@egdc_required
def test_egdc_records_are_all_verified():
    recs = list(egdc_records(variants=2, seed=7))
    assert len(recs) >= 100  # 63 factories x 2 variants
    # every shipped reference must reproduce its own test cases in-process
    assert all(r.verified for r in recs), [
        r.task_id for r in recs if not r.verified
    ]


@egdc_required
def test_egdc_records_have_rich_program_shapes():
    recs = list(egdc_records(variants=1, seed=1))
    cats = {r.category for r in recs}
    # the whole point: shapes the array->scalar fold cannot express
    assert "arrays" in cats
    assert "strings" in cats
    # all are Mog programs with concrete io
    assert all(r.program_lang == "mog" for r in recs)
    assert all(r.io_pairs and "inputs" in r.io_pairs[0] for r in recs)
    # programs define a function (some start with a `struct` decl first)
    assert all("fn " in r.program_src for r in recs)


# ---------------------------------------------------------------------------
# autoresearch joiner
# ---------------------------------------------------------------------------

def test_autoresearch_join(tmp_path: Path):
    solved = tmp_path / "solved.jsonl"
    queue = tmp_path / "queue.jsonl"
    solved.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/1",
                "entry_point": "f",
                "program_python": "    return n + 1\n",
                "verifier_passed": True,
                "source_benchmark": "humaneval",
            }
        )
        + "\n"
    )
    queue.write_text(
        json.dumps(
            {
                "task_id": "HumanEval/1",
                "entry_point": "f",
                "prompt": 'def f(n):\n    """Add one to n."""\n',
                "io_pairs": [{"args": [4], "kwargs": {}, "expected_repr": "5"}],
            }
        )
        + "\n"
    )
    recs = list(autoresearch_records(solved_path=solved, queue_path=queue))
    assert len(recs) == 1
    r = recs[0]
    assert r.program_lang == "python"
    assert r.nl_prompt == "Add one to n."
    assert r.io_pairs == [{"inputs": [4], "expected": 5}]
    assert r.verified is True
    assert "def f(n):" in r.program_src and "return n + 1" in r.program_src


def test_autoresearch_missing_files_yields_nothing(tmp_path: Path):
    recs = list(
        autoresearch_records(
            solved_path=tmp_path / "nope.jsonl", queue_path=tmp_path / "nope2.jsonl"
        )
    )
    assert recs == []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@egdc_required
def test_cli_egdc_only_verified(tmp_path: Path, capsys):
    out = tmp_path / "corpus.jsonl"
    rc = main(["--source", "egdc", "--variants", "1", "--out", str(out), "--only-verified"])
    assert rc == 0
    recs = read_corpus(out)
    assert len(recs) >= 60
    assert all(r.verified for r in recs)
    report = json.loads(capsys.readouterr().out)
    assert report["written"] == len(recs)
    assert report["by_lang"]["mog"] == len(recs)
