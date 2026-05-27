"""NPCoT HTTP server tests (NV1b)."""

from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    ArrayProgramLibraryConfig,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.npcot_server import (
    handle_consult_request,
    start_server,
)


def _make_library() -> ArrayProgramLibrary:
    lib = ArrayProgramLibrary()
    lib.record(
        torch.tensor([1.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    lib.record(
        torch.tensor([0.0, 1.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.0),
        task_name="max",
    )
    return lib


class TestHandleConsultRequest(unittest.TestCase):
    def test_hit_returns_result(self):
        lib = _make_library()
        status, payload = handle_consult_request(
            {"hidden": [1.0, 0.0, 0.0], "array": [1.0, 2.0, 3.0], "length": 3},
            lib,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["hit"])
        self.assertAlmostEqual(payload["result"], 6.0, places=4)
        self.assertEqual(payload["task_name"], "sum")
        self.assertIn("elapsed_us", payload)

    def test_miss_returns_200_hit_false(self):
        lib = ArrayProgramLibrary(
            ArrayProgramLibraryConfig(similarity_threshold=0.99)
        )
        lib.record(
            torch.tensor([1.0, 0.0, 0.0]),
            DiscreteArrayProgram(0, 0, 0, 0, 0.0),
            task_name="sum",
        )
        status, payload = handle_consult_request(
            {"hidden": [0.0, 1.0, 0.0], "array": [1.0, 2.0, 3.0], "length": 3},
            lib,
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["hit"])
        self.assertIsNone(payload["result"])

    def test_missing_field_returns_400(self):
        lib = _make_library()
        status, payload = handle_consult_request(
            {"hidden": [1.0, 0.0, 0.0], "array": [1.0, 2.0, 3.0]},
            lib,
        )
        self.assertEqual(status, 400)
        self.assertIn("missing required field", payload["error"])

    def test_malformed_input_returns_400(self):
        lib = _make_library()
        status, payload = handle_consult_request(
            {"hidden": "not-a-list", "array": [1.0, 2.0], "length": 2},
            lib,
        )
        self.assertEqual(status, 400)
        self.assertIn("malformed", payload["error"])


class TestLiveServer(unittest.TestCase):
    def setUp(self):
        self.library = _make_library()
        self.server = start_server(
            self.library, host="127.0.0.1", port=0
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(self._url(path))
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return resp.status, body
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self._url(path),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return resp.status, body
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_endpoint(self):
        status, payload = self._get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["library_entries"], 2)

    def test_audit_endpoint(self):
        status, payload = self._get("/audit")
        self.assertEqual(status, 200)
        self.assertIn("entries", payload)
        self.assertEqual(payload["summary"]["entry_count"], 2)

    def test_fingerprint_endpoint(self):
        status, payload = self._get("/fingerprint")
        self.assertEqual(status, 200)
        self.assertTrue(payload["fingerprint"].startswith("npcot1:"))

    def test_consult_endpoint_hit(self):
        status, payload = self._post(
            "/consult",
            {"hidden": [1.0, 0.0, 0.0], "array": [1.0, 2.0, 3.0], "length": 3},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["hit"])
        self.assertAlmostEqual(payload["result"], 6.0, places=4)

    def test_consult_endpoint_bad_input(self):
        status, payload = self._post("/consult", {"bad": "input"})
        self.assertEqual(status, 400)

    def test_unknown_path_returns_404(self):
        status, payload = self._get("/nonexistent")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
