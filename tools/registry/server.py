"""Verified-skill registry server — crates.io for synthesized programs.

Stdlib-only (http.server + sqlite3), in the style of
``ncpu/self_optimizing/npcot_server.py``: zero third-party dependencies.

Trustless contribution model: POST /skills re-executes the submitted
program against its claimed examples with the pure-Python mirror of the
canonical NPCoT executor (``tools/registry/executor.py``, ported from
``kernels/npcot_wasm/src/lib.rs``). A skill is accepted iff the max
absolute error across ALL examples is <= 1e-3 relative to
max(1, max|target|). Wrong or spam code physically cannot enter.

Usage::

    python3 -m tools.registry.server --port 8430 --db registry.sqlite

    # Trust sweep (CI guarantee): re-verify every stored skill.
    python3 -m tools.registry.server --db registry.sqlite --verify-all

Endpoints:
    POST /skills        submit {name, author, examples, program|program_v2|program_v3}
    GET  /skills        list (author attribution + created_at)
    GET  /skills/<id>   full record
    GET  /library.json  whole registry as a loadable NPCoT library
    GET  /health        liveness + skill count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):  # direct execution: python3 tools/registry/server.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.registry import executor  # type: ignore
else:
    from . import executor

DEFAULT_DB = "./registry.sqlite"
DEFAULT_PORT = 8430
SIGNATURE_DIM = 8
SIMILARITY_THRESHOLD = 0.85

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    author        TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    format        INTEGER NOT NULL,
    program_json  TEXT NOT NULL,
    examples_json TEXT NOT NULL,
    max_err       REAL NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skills_fingerprint ON skills (fingerprint);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Canonicalization + fingerprinting
# ---------------------------------------------------------------------------


def canonicalize_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize examples to {data, n_points, target} or {data, n_points, targets}."""
    out = []
    for ex in examples:
        record = {
            "data": [float(v) for v in ex["data"]],
            "n_points": int(ex["n_points"]),
        }
        if "targets" in ex:
            record["targets"] = [float(v) for v in ex["targets"]]
        else:
            record["target"] = float(ex["target"])
        out.append(record)
    return out


def examples_fingerprint(examples: List[Dict[str, Any]]) -> str:
    """sha256 over a sorted, stable serialization of the canonical examples.

    Order-insensitive: the same example set always maps to the same
    fingerprint, so alternative orderings of a submission dedupe together.
    """
    canon = sorted(
        json.dumps(ex, sort_keys=True, separators=(",", ":"))
        for ex in canonicalize_examples(examples)
    )
    return hashlib.sha256("\n".join(canon).encode("utf-8")).hexdigest()


def canonical_program_json(program_dict: Dict[str, Any]) -> str:
    return json.dumps(program_dict, sort_keys=True, separators=(",", ":"))


def skill_signature(skill_id: int, dim: int = SIGNATURE_DIM) -> List[float]:
    """Deterministic unit vector derived from the skill id (seeded by
    sha256, independent of Python hash randomization or RNG versions),
    so /library.json is directly consultable by every NPCoT runtime."""
    digest = hashlib.sha256(f"ncpu-registry-skill-{skill_id}".encode("utf-8")).digest()
    vals = []
    for k in range(dim):
        chunk = digest[(k * 4) % len(digest) : (k * 4) % len(digest) + 4]
        u = int.from_bytes(chunk, "big")
        vals.append((u / 2.0**32) * 2.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in vals))
    if norm < 1e-12:  # unreachable in practice; keep the vector valid anyway
        vals[0] = 1.0
        norm = 1.0
    return [v / norm for v in vals]


# ---------------------------------------------------------------------------
# Request handling (standalone functions, embeddable without our server)
# ---------------------------------------------------------------------------


def _validate_examples(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("examples must be a non-empty list")
    for i, ex in enumerate(raw):
        if not isinstance(ex, dict):
            raise ValueError(f"examples[{i}] must be an object")
        for key in ("data", "n_points"):
            if key not in ex:
                raise ValueError(f"examples[{i}] missing field: {key}")
        if ("target" in ex) == ("targets" in ex):
            raise ValueError(f"examples[{i}] must contain exactly one of 'target' or 'targets'")
        if not isinstance(ex["data"], list):
            raise ValueError(f"examples[{i}].data must be a list")
        for v in ex["data"]:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"examples[{i}].data must contain finite numbers")
        if isinstance(ex["n_points"], bool) or not isinstance(ex["n_points"], int) or ex["n_points"] < 0:
            raise ValueError(f"examples[{i}].n_points must be a non-negative integer")
        if "targets" in ex:
            targets = ex["targets"]
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"examples[{i}].targets must be a non-empty list")
            for step, t in enumerate(targets):
                if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
                    raise ValueError(f"examples[{i}].targets[{step}] must be finite numbers")
            if len(targets) != ex["n_points"]:
                raise ValueError(f"examples[{i}].targets length must match n_points")
        else:
            t = ex["target"]
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
                raise ValueError(f"examples[{i}].target must be a finite number")
    return canonicalize_examples(raw)


def _parse_submission(body: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]], int, executor.Program]:
    for key in ("name", "author", "examples"):
        if key not in body:
            raise ValueError(f"missing required field: {key}")
    name = str(body["name"]).strip()
    author = str(body["author"]).strip()
    if not name or not author:
        raise ValueError("name and author must be non-empty")
    examples = _validate_examples(body["examples"])
    has_v1 = "program" in body
    has_v2 = "program_v2" in body
    has_v3 = "program_v3" in body
    if has_v1 + has_v2 + has_v3 != 1:
        raise ValueError("exactly one of 'program', 'program_v2', or 'program_v3' is required")
    if has_v1:
        version, payload = 1, body["program"]
    elif has_v2:
        version, payload = 2, body["program_v2"]
    else:
        version, payload = 3, body["program_v3"]
    if not isinstance(payload, dict):
        raise ValueError("program must be an object")
    program = executor.program_from_dict(payload, version)
    return name, author, examples, version, program


def handle_submission(body: Dict[str, Any], db_path: str) -> Tuple[int, Dict[str, Any]]:
    """Verify-then-insert. Returns ``(http_status, response_body)``.

    * 400 — malformed submission (shape errors, non-finite numbers).
    * 422 — verification FAILED: the program does not reproduce its own
      examples. The error report names the first failing example.
    * 200 — accepted (or exact duplicate, flagged as such).
    """
    try:
        name, author, examples, version, program = _parse_submission(body)
    except (ValueError, TypeError) as exc:
        return 400, {"accepted": False, "error": str(exc)}

    result = executor.verify_program(program, examples)
    if not result.ok:
        max_err = result.max_err if math.isfinite(result.max_err) else None
        response = {
            "accepted": False,
            "max_err": max_err,
            "first_failure": result.first_failure,
            "error": "verification failed: program does not reproduce its examples",
        }
        # Capture point for the cascade loop: every rejected submission
        # becomes a WorkItem via ``ncpu.autoresearch.sources.registry``
        # when NCPU_REGISTRY_MISSES_PATH is set. The autoresearch driver
        # can then run the cascade on the rejected problem and POST
        # the recovered program back to the registry, closing the loop
        # without ever letting unverified code into the store.
        misses_path = os.environ.get("NCPU_REGISTRY_MISSES_PATH")
        if misses_path:
            try:
                with open(misses_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "name": name,
                        "author": author,
                        "examples": examples,
                        "error": response["error"],
                        "first_failure": response["first_failure"],
                        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }, sort_keys=True) + "\n")
            except OSError:
                # Capture is best-effort: a missing misses file must not
                # change the registry's HTTP response.
                pass
        return 422, response

    fingerprint = examples_fingerprint(examples)
    program_json = canonical_program_json(executor.program_to_dict(program))
    examples_json = json.dumps(examples, sort_keys=True, separators=(",", ":"))

    conn = connect(db_path)
    try:
        # Dedupe: same fingerprint + identical program → duplicate.
        # Same fingerprint + DIFFERENT program → keep both (alternative
        # verified solutions to the same example set).
        row = conn.execute(
            "SELECT id FROM skills WHERE fingerprint = ? AND program_json = ? AND format = ?",
            (fingerprint, program_json, version),
        ).fetchone()
        if row is not None:
            return 200, {
                "accepted": True,
                "duplicate": True,
                "fingerprint": fingerprint,
                "skill_id": row["id"],
            }
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = conn.execute(
            "INSERT INTO skills (name, author, fingerprint, format, program_json,"
            " examples_json, max_err, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, author, fingerprint, version, program_json, examples_json,
             result.max_err, created_at),
        )
        conn.commit()
        return 200, {
            "accepted": True,
            "duplicate": False,
            "fingerprint": fingerprint,
            "skill_id": cur.lastrowid,
            "max_err": result.max_err,
        }
    finally:
        conn.close()


def list_skills(db_path: str) -> Dict[str, Any]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, author, fingerprint, format, max_err, created_at"
            " FROM skills ORDER BY id"
        ).fetchall()
        return {"skills": [dict(r) for r in rows], "count": len(rows)}
    finally:
        conn.close()


def get_skill(db_path: str, skill_id: int) -> Optional[Dict[str, Any]]:
    conn = connect(db_path)
    try:
        row = conn.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        program = json.loads(record.pop("program_json"))
        if record["format"] == 3:
            program_key = "program_v3"
        elif record["format"] == 2:
            program_key = "program_v2"
        else:
            program_key = "program"
        record[program_key] = program
        record["examples"] = json.loads(record.pop("examples_json"))
        return record
    finally:
        conn.close()


def build_library(db_path: str) -> Dict[str, Any]:
    """Render the whole registry as a loadable NPCoT library.

    Mirrors ``library_to_json`` in kernels/npcot_wasm/src/lib.rs: pure-v1
    registries emit the v1 format (no "format" key, entries carry
    "program") so every existing runtime loads them; any v2 entry
    switches the whole library to ``"format": 2`` with every entry lifted
    to ``program_v2`` (the v1→v2 lift is exact). Old v1 loaders fail
    closed on format 2 — they can never silently mis-execute a guard.
    """
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, format, program_json, created_at FROM skills ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    needs_v3 = any(r["format"] == 3 for r in rows)
    needs_v2 = (not needs_v3) and any(r["format"] >= 2 for r in rows)
    entries = []
    for r in rows:
        program = json.loads(r["program_json"])
        entry: Dict[str, Any] = {
            "signature": skill_signature(r["id"]),
            "hit_count": 0,
            "task_name": r["name"],
            "skill_id": r["id"],
            "cached_at_step": None,
            "convergence_gap": None,
        }
        if needs_v3:
            if r["format"] == 1:
                p = executor.ProgramV3.from_v1(executor.program_from_dict(program, 1))
            elif r["format"] == 2:
                p = executor.ProgramV3.from_v2(executor.program_from_dict(program, 2))
            else:
                p = executor.program_from_dict(program, 3)
            entry["program_v3"] = executor.program_to_dict(p)
        elif needs_v2:
            if r["format"] == 1:
                p = executor.ProgramV2.from_v1(executor.program_from_dict(program, 1))
            elif r["format"] == 2:
                p = executor.program_from_dict(program, 2)
            else:
                p3 = executor.program_from_dict(program, 3)
                p = p3.to_v2()
                if p is None:
                    raise ValueError(f"skill {r['id']} requires format 3 export")
            entry["program_v2"] = executor.program_to_dict(p)
        else:
            entry["program"] = program
        entries.append(entry)

    library: Dict[str, Any] = {}
    if needs_v3:
        library["format"] = 3
    elif needs_v2:
        library["format"] = 2
    library["config"] = {
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "max_entries": max(len(entries), 16),
        "normalize_epsilon": 1e-08,
    }
    library["entries"] = entries
    return library


def verify_all(db_path: str) -> List[Dict[str, Any]]:
    """Re-verify every stored skill. Returns the list of failures (empty
    when the registry is fully trustworthy)."""
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, format, program_json, examples_json FROM skills ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    failures = []
    for r in rows:
        try:
            program = executor.program_from_dict(json.loads(r["program_json"]), r["format"])
            examples = json.loads(r["examples_json"])
            result = executor.verify_program(program, examples)
            ok, max_err, first_failure = result.ok, result.max_err, result.first_failure
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            ok, max_err, first_failure = False, math.inf, {"error": str(exc)}
        if not ok:
            failures.append(
                {
                    "skill_id": r["id"],
                    "name": r["name"],
                    "max_err": max_err if math.isfinite(max_err) else None,
                    "first_failure": first_failure,
                }
            )
    return failures


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


class RegistryServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: Tuple[str, int], db_path: str):
        super().__init__(addr, RegistryHandler)
        self.db_path = db_path


class RegistryHandler(BaseHTTPRequestHandler):
    server: RegistryServer

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        db = self.server.db_path
        if path == "/health":
            self._send_json(200, {"status": "ok", "skills": list_skills(db)["count"]})
        elif path == "/skills":
            self._send_json(200, list_skills(db))
        elif path.startswith("/skills/"):
            tail = path[len("/skills/"):]
            if not tail.isdigit():
                self._send_json(404, {"error": "skill id must be an integer"})
                return
            record = get_skill(db, int(tail))
            if record is None:
                self._send_json(404, {"error": f"no skill with id {tail}"})
            else:
                self._send_json(200, record)
        elif path == "/library.json":
            self._send_json(200, build_library(db))
        else:
            self._send_json(404, {"error": f"unknown path: {path}"})

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/skills":
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"accepted": False, "error": f"malformed JSON body: {exc}"})
            return
        status, response = handle_submission(body, self.server.db_path)
        self._send_json(status, response)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Verified-skill registry server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--db", type=str, default=DEFAULT_DB)
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="re-verify every stored skill and exit nonzero on any failure",
    )
    args = parser.parse_args(argv)

    if args.verify_all:
        failures = verify_all(args.db)
        total = list_skills(args.db)["count"]
        if failures:
            print(f"VERIFY-ALL: {len(failures)}/{total} skills FAILED re-verification:")
            for f in failures:
                print(
                    f"  FAIL skill_id={f['skill_id']} name={f['name']!r}"
                    f" max_err={f['max_err']} first_failure={f['first_failure']}"
                )
            return 1
        print(f"VERIFY-ALL: {total}/{total} skills verified OK")
        return 0

    connect(args.db).close()  # create schema up front
    server = RegistryServer(("127.0.0.1", args.port), args.db)
    print(f"registry serving on http://127.0.0.1:{args.port} (db: {args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
