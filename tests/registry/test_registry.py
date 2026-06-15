"""Verified-skill registry tests (ROADMAP Rung 4 DoD).

Covers: executor pinned outputs (exact mirror of kernels/npcot_wasm),
trustless POST /skills verification gate, library.json format switching
(v1 → format 2 lift), fingerprint dedupe, and the --verify-all trust
sweep including detection of a directly-corrupted sqlite row.
"""

from __future__ import annotations

import json
import math
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.registry import executor
from tools.registry.executor import DiscreteProgram, ProgramV2, ProgramV3

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Executor pinned-output unit tests
# ---------------------------------------------------------------------------


def test_v1_sum():
    p = DiscreteProgram(init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=0, offset=0.0)
    assert executor.execute_program(p, [1.0, 2.0, 3.0], 3) == 6.0


def test_v1_sum_ignores_padding_beyond_length():
    p = DiscreteProgram(init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=0, offset=0.0)
    assert executor.execute_program(p, [1.0, 2.0, 3.0, 99.0, 99.0], 3) == 6.0


def test_v1_max_with_neg_large_init():
    p = DiscreteProgram(init_idx=2, transform_idx=0, reduce_idx=2, post_scale_idx=0, offset=0.0)
    assert executor.execute_program(p, [3.0, -7.0, 1.0], 3) == 3.0
    # All-negative array: init NEG_LARGE=-20 stays below every element.
    assert executor.execute_program(p, [-5.0, -3.0], 2) == -3.0


def test_v1_mean_divides_by_raw_length_param():
    # Canonical quirk: v1 mean divides by the *length argument*, not the
    # effective iterated length (Rust: `(length as f32).max(1.0)`).
    p = DiscreteProgram(init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=1, offset=0.0)
    assert executor.execute_program(p, [1.0, 2.0, 3.0], 3) == 2.0
    assert executor.execute_program(p, [1.0, 2.0], 4) == 0.75  # sum=3, /4


def test_v1_log_sum_exp_is_abs_product():
    # transform=ln|x|, reduce=+, post=exp → |product|
    p = DiscreteProgram(init_idx=0, transform_idx=5, reduce_idx=0, post_scale_idx=2, offset=0.0)
    got = executor.execute_program(p, [2.0, -3.0, 4.0], 3)
    assert abs(got - 24.0) < 0.01


def test_v1_offset_applied_after_post_scale():
    p = DiscreteProgram(init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=1, offset=10.0)
    assert executor.execute_program(p, [2.0, 4.0], 2) == 13.0  # mean 3 + 10


def test_v2_dot_product():
    # arity=2, combine=product of fields, reduce=+ → dot product.
    # (2,3),(4,0.5) → 2*3 + 4*0.5 = 8
    p = ProgramV2(arity=2, combine_idx=3, guard_idx=0, guard_threshold=0.0,
                  init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=0, offset=0.0)
    assert executor.execute_program_v2(p, [2.0, 3.0, 4.0, 0.5], 2) == 8.0


def test_v2_guarded_mean_of_positives():
    # guard v>0 excludes negatives; mean divides by INCLUDED count (2), not 4.
    p = ProgramV2(arity=1, combine_idx=0, guard_idx=1, guard_threshold=0.0,
                  init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=1, offset=0.0)
    assert executor.execute_program_v2(p, [4.0, -2.0, 6.0, -8.0], 4) == 5.0


def test_v2_max_with_neg_large_init():
    p = ProgramV2(arity=1, combine_idx=0, guard_idx=0, guard_threshold=0.0,
                  init_idx=2, transform_idx=0, reduce_idx=2, post_scale_idx=0, offset=0.0)
    assert executor.execute_program_v2(p, [3.0, -7.0, 1.0], 3) == 3.0


def test_v2_from_v1_is_exact():
    v1 = DiscreteProgram(init_idx=0, transform_idx=1, reduce_idx=0, post_scale_idx=1, offset=0.5)
    lifted = ProgramV2.from_v1(v1)
    assert lifted.is_v1()
    data = [1.0, -2.0, 3.0]
    assert executor.execute_program_v2(lifted, data, 3) == executor.execute_program(v1, data, 3)


def test_v3_running_counter_replay():
    p = ProgramV3(
        arity=1, combine_idx=0, guard_idx=0, guard_threshold=0.0,
        reset_guard_idx=0, reset_threshold=0.0, state_init_idx=0,
        update_transform_idx=3, update_reduce_idx=0, post_scale_idx=0,
        output_idx=0, offset=0.0,
    )
    assert executor.execute_program_v3(p, [5.0, -2.0, 0.0, 9.0], 4) == [1.0, 2.0, 3.0, 4.0]


def test_v3_running_max_with_reset():
    p = ProgramV3(
        arity=1, combine_idx=0, guard_idx=0, guard_threshold=0.0,
        reset_guard_idx=2, reset_threshold=-8.0, state_init_idx=2,
        update_transform_idx=0, update_reduce_idx=2, post_scale_idx=0,
        output_idx=0, offset=0.0,
    )
    assert executor.execute_program_v3(p, [5.0, 2.0, -10.0, 1.0, 0.0], 5) == [5.0, 5.0, -10.0, 1.0, 1.0]


def test_v3_lift_matches_v2_final_output():
    v2 = ProgramV2(arity=2, combine_idx=3, guard_idx=0, guard_threshold=0.0,
                   init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=0, offset=0.25)
    data = [2.0, 3.0, 4.0, 0.5]
    assert executor.execute_program_v3_final(ProgramV3.from_v2(v2), data, 2) == executor.execute_program_v2(v2, data, 2)


def test_verify_accepts_v3_trace_examples():
    p = ProgramV3(
        arity=1, combine_idx=0, guard_idx=0, guard_threshold=0.0,
        reset_guard_idx=0, reset_threshold=0.0, state_init_idx=0,
        update_transform_idx=3, update_reduce_idx=0, post_scale_idx=0,
        output_idx=0, offset=0.0,
    )
    examples = [{"data": [1.0, 2.0, 3.0], "n_points": 3, "targets": [1.0, 2.0, 3.0]}]
    assert executor.verify_program(p, examples).ok


def test_verify_accepts_within_relative_tolerance():
    p = DiscreteProgram(init_idx=0, transform_idx=0, reduce_idx=0, post_scale_idx=0, offset=0.0)
    examples = [{"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0}]
    assert executor.verify_program(p, examples).ok
    # target 6 → accept = 1e-3 * max(1, 6) = 6e-3; an error of 0.01 fails.
    examples_off = [{"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.01}]
    result = executor.verify_program(p, examples_off)
    assert not result.ok
    assert result.first_failure["example_index"] == 0


# ---------------------------------------------------------------------------
# Server fixtures
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(port: int, method: str, path: str, body=None):
    """Returns (status, parsed_json). Does not raise on 4xx/5xx."""
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


@pytest.fixture()
def registry(tmp_path):
    """Spawn a registry server subprocess on a free port with a tmp db."""
    port = _free_port()
    db = tmp_path / "registry.sqlite"
    proc = subprocess.Popen(
        [sys.executable, "-m", "tools.registry.server", "--port", str(port), "--db", str(db)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 15.0
        while True:
            try:
                status, _ = _request(port, "GET", "/health")
                if status == 200:
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if proc.poll() is not None:
                out, err = proc.communicate()
                raise RuntimeError(f"server died: {out.decode()!r} {err.decode()!r}")
            if time.time() > deadline:
                raise RuntimeError("server did not become ready in 15s")
            time.sleep(0.05)
        yield port, db
    finally:
        proc.terminate()
        proc.wait(timeout=10)


SUM_SKILL = {
    "name": "sum",
    "author": "alice",
    "examples": [
        {"data": [1.0, 2.0, 3.0], "n_points": 3, "target": 6.0},
        {"data": [10.0, -4.0], "n_points": 2, "target": 6.0},
        {"data": [5.0], "n_points": 1, "target": 5.0},
    ],
    "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                "post_scale_idx": 0, "offset": 0.0},
}

DOT_SKILL = {
    "name": "dot_product",
    "author": "bob",
    "examples": [
        {"data": [2.0, 3.0, 4.0, 0.5], "n_points": 2, "target": 8.0},
        {"data": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "n_points": 3, "target": 3.0},
    ],
    "program_v2": {"arity": 2, "combine_idx": 3, "guard_idx": 0, "guard_threshold": 0.0,
                   "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                   "post_scale_idx": 0, "offset": 0.0},
}

COUNTER_SKILL = {
    "name": "running_counter",
    "author": "dana",
    "examples": [
        {"data": [5.0, -2.0, 0.0, 9.0], "n_points": 4, "targets": [1.0, 2.0, 3.0, 4.0]},
        {"data": [0.5, 0.5, -3.0, 100.0, 0.0, 7.0], "n_points": 6,
         "targets": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    ],
    "program_v3": {"arity": 1, "combine_idx": 0, "guard_idx": 0, "guard_threshold": 0.0,
                   "reset_guard_idx": 0, "reset_threshold": 0.0, "state_init_idx": 0,
                   "update_transform_idx": 3, "update_reduce_idx": 0,
                   "post_scale_idx": 0, "output_idx": 0, "offset": 0.0},
}


# ---------------------------------------------------------------------------
# 2. Valid v1 submission → accepted, listed, in library.json
# ---------------------------------------------------------------------------


def test_submit_valid_v1_skill(registry):
    port, _db = registry
    status, body = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200, body
    assert body["accepted"] is True
    assert body["duplicate"] is False
    assert len(body["fingerprint"]) == 64
    skill_id = body["skill_id"]

    status, listing = _request(port, "GET", "/skills")
    assert status == 200
    assert listing["count"] == 1
    entry = listing["skills"][0]
    assert entry["name"] == "sum"
    assert entry["author"] == "alice"
    assert entry["created_at"]

    status, record = _request(port, "GET", f"/skills/{skill_id}")
    assert status == 200
    assert record["program"]["transform_idx"] == 0
    assert len(record["examples"]) == 3

    status, lib = _request(port, "GET", "/library.json")
    assert status == 200
    # Pure-v1 registry → v1 format: no "format" key, entries carry "program".
    assert "format" not in lib
    assert lib["config"]["similarity_threshold"] == 0.85
    assert len(lib["entries"]) == 1
    e = lib["entries"][0]
    assert e["task_name"] == "sum"
    assert set(e["program"]) == set(executor.V1_FIELDS)
    sig = e["signature"]
    assert len(sig) == 8
    assert abs(math.sqrt(sum(v * v for v in sig)) - 1.0) < 1e-9  # unit vector
    # Deterministic: same skill_id → same signature.
    status2, lib2 = _request(port, "GET", "/library.json")
    assert lib2["entries"][0]["signature"] == sig


# ---------------------------------------------------------------------------
# 3. Subtly wrong program → 422 with verification error
# ---------------------------------------------------------------------------


def test_submit_wrong_program_rejected(registry):
    port, _db = registry
    wrong = json.loads(json.dumps(SUM_SKILL))
    wrong["name"] = "sum_but_actually_sum_of_squares"
    wrong["program"]["transform_idx"] = 1  # x*x — claims sum, computes sum of squares
    status, body = _request(port, "POST", "/skills", wrong)
    assert status == 422, body
    assert body["accepted"] is False
    assert body["max_err"] == pytest.approx(110.0)  # worst example: 10²+(-4)²=116 vs 6
    ff = body["first_failure"]
    assert ff["example_index"] == 0
    assert ff["expected"] == 6.0
    assert ff["got"] == pytest.approx(14.0)  # 1+4+9
    # Nothing entered the registry.
    _, listing = _request(port, "GET", "/skills")
    assert listing["count"] == 0


# ---------------------------------------------------------------------------
# 4. Valid v2 submission → accepted; library lifts to format 2
# ---------------------------------------------------------------------------


def test_submit_v2_skill_lifts_library_to_format_2(registry):
    port, _db = registry
    status, body = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200 and body["accepted"]
    status, body = _request(port, "POST", "/skills", DOT_SKILL)
    assert status == 200, body
    assert body["accepted"] is True

    status, lib = _request(port, "GET", "/library.json")
    assert status == 200
    assert lib["format"] == 2
    assert len(lib["entries"]) == 2
    for e in lib["entries"]:
        assert "program" not in e
        assert set(e["program_v2"]) == set(executor.V2_FIELDS)
    # The lifted v1 entry is the exact v1 special case.
    lifted = next(e for e in lib["entries"] if e["task_name"] == "sum")["program_v2"]
    assert (lifted["arity"], lifted["combine_idx"], lifted["guard_idx"]) == (1, 0, 0)
    # Lifted program still reproduces the v1 examples under the v2 engine.
    p = executor.program_from_dict(lifted, 2)
    assert executor.execute_program_v2(p, [1.0, 2.0, 3.0], 3) == 6.0


def test_submit_v3_skill_lifts_library_to_format_3(registry):
    port, _db = registry
    status, body = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200 and body["accepted"]
    status, body = _request(port, "POST", "/skills", COUNTER_SKILL)
    assert status == 200, body
    assert body["accepted"] is True

    status, lib = _request(port, "GET", "/library.json")
    assert status == 200
    assert lib["format"] == 3
    assert len(lib["entries"]) == 2
    for e in lib["entries"]:
        assert "program" not in e
        assert "program_v2" not in e
        assert set(e["program_v3"]) == set(executor.V3_FIELDS)
    lifted = next(e for e in lib["entries"] if e["task_name"] == "sum")["program_v3"]
    assert (lifted["arity"], lifted["combine_idx"], lifted["guard_idx"]) == (1, 0, 0)
    p = executor.program_from_dict(lifted, 3)
    assert executor.execute_program_v3_final(p, [1.0, 2.0, 3.0], 3) == 6.0


def test_submit_wrong_v3_program_rejected(registry):
    port, _db = registry
    wrong = json.loads(json.dumps(COUNTER_SKILL))
    wrong["program_v3"]["update_transform_idx"] = 0  # identity update instead of const-1 counter
    status, body = _request(port, "POST", "/skills", wrong)
    assert status == 422, body
    assert body["accepted"] is False
    assert body["first_failure"]["example_index"] == 0
    _, listing = _request(port, "GET", "/skills")
    assert listing["count"] == 0


def test_submit_wrong_v2_program_rejected(registry):
    port, _db = registry
    wrong = json.loads(json.dumps(DOT_SKILL))
    wrong["program_v2"]["combine_idx"] = 2  # Σfields instead of Πfields
    status, body = _request(port, "POST", "/skills", wrong)
    assert status == 422, body
    assert body["accepted"] is False
    assert body["first_failure"]["example_index"] == 0


# ---------------------------------------------------------------------------
# 5. Duplicate resubmission → flagged, count unchanged.
#    Same examples + different program → both kept (alternative solutions).
# ---------------------------------------------------------------------------


def test_duplicate_resubmission_deduped(registry):
    port, _db = registry
    status, first = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200 and first["duplicate"] is False
    status, second = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200, second
    assert second["accepted"] is True
    assert second["duplicate"] is True
    assert second["fingerprint"] == first["fingerprint"]
    assert second["skill_id"] == first["skill_id"]
    _, listing = _request(port, "GET", "/skills")
    assert listing["count"] == 1


def test_alternative_solution_same_fingerprint_kept(registry):
    port, _db = registry
    status, first = _request(port, "POST", "/skills", SUM_SKILL)
    assert status == 200
    # Sum of |x| is a different program that also satisfies these examples?
    # No — example 2 has a negative. Use sum with transform=abs on
    # all-positive variant: instead, submit the SAME examples solved by a
    # different (still-correct) program: identity sum via reduce default
    # cannot differ... use offset-fitted equivalent: transform=abs fails on
    # [10,-4]. A genuinely different correct program: v2 lift of sum.
    alt = {
        "name": "sum_v2_form",
        "author": "carol",
        "examples": SUM_SKILL["examples"],
        "program_v2": {"arity": 1, "combine_idx": 0, "guard_idx": 0, "guard_threshold": 0.0,
                       "init_idx": 0, "transform_idx": 0, "reduce_idx": 0,
                       "post_scale_idx": 0, "offset": 0.0},
    }
    status, second = _request(port, "POST", "/skills", alt)
    assert status == 200, second
    assert second["duplicate"] is False
    assert second["fingerprint"] == first["fingerprint"]  # same example set
    assert second["skill_id"] != first["skill_id"]
    _, listing = _request(port, "GET", "/skills")
    assert listing["count"] == 2  # both alternative solutions kept


# ---------------------------------------------------------------------------
# 6. --verify-all: green on a clean db, nonzero after direct corruption
# ---------------------------------------------------------------------------


def _run_verify_all(db) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tools.registry.server", "--db", str(db), "--verify-all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_verify_all_clean_then_corrupted(registry):
    port, db = registry
    for skill in (SUM_SKILL, DOT_SKILL, COUNTER_SKILL):
        status, body = _request(port, "POST", "/skills", skill)
        assert status == 200 and body["accepted"]

    result = _run_verify_all(db)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3/3 skills verified OK" in result.stdout

    # Corrupt the stored program directly in sqlite — simulating db
    # tampering that bypassed the POST verification gate.
    conn = sqlite3.connect(db)
    corrupted = json.dumps({"init_idx": 0, "transform_idx": 1, "reduce_idx": 0,
                            "post_scale_idx": 0, "offset": 0.0},
                           sort_keys=True, separators=(",", ":"))
    conn.execute("UPDATE skills SET program_json = ? WHERE name = 'sum'", (corrupted,))
    conn.commit()
    conn.close()

    result = _run_verify_all(db)
    assert result.returncode != 0
    assert "FAIL" in result.stdout
    assert "name='sum'" in result.stdout  # names the corrupted skill


# ---------------------------------------------------------------------------
# Malformed submissions → 400, never stored
# ---------------------------------------------------------------------------


def test_malformed_submissions_rejected_400(registry):
    port, _db = registry
    cases = [
        {},  # everything missing
        {**SUM_SKILL, "examples": []},  # empty examples
        {k: v for k, v in SUM_SKILL.items() if k != "program"},  # no program
        {**SUM_SKILL, "program_v2": DOT_SKILL["program_v2"]},  # both programs
        {**SUM_SKILL, "program_v3": COUNTER_SKILL["program_v3"]},  # v1 + v3
    ]
    for case in cases:
        status, body = _request(port, "POST", "/skills", case)
        assert status == 400, (case, body)
        assert body["accepted"] is False
    # Raw NaN in the body: Python's json.loads tolerates it, but example
    # validation requires finite numbers → 400.
    url = f"http://127.0.0.1:{port}/skills"
    raw = ('{"name": "x", "author": "y", '
           '"examples": [{"data": [NaN], "n_points": 1, "target": 1.0}], '
           '"program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0, '
           '"post_scale_idx": 0, "offset": 0.0}}').encode()
    req = urllib.request.Request(url, data=raw, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 400
    _, listing = _request(port, "GET", "/skills")
    assert listing["count"] == 0
