"""Library signing/verification tests (NV2b)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from ncpu.self_optimizing.array_program_library import (
    ArrayProgramLibrary,
    DiscreteArrayProgram,
)
from ncpu.self_optimizing.library_signing import (
    sign_library,
    verify_library_signature,
)


def _populate(library: ArrayProgramLibrary) -> None:
    library.record(
        torch.tensor([1.0, 0.0, 0.0]),
        DiscreteArrayProgram(0, 0, 0, 0, 0.0),
        task_name="sum",
    )
    library.record(
        torch.tensor([0.0, 1.0, 0.0]),
        DiscreteArrayProgram(2, 0, 2, 0, 0.0),
        task_name="max",
    )


class TestSignAndVerify(unittest.TestCase):
    SECRET = b"test-secret-key-npcot-42"

    def test_sign_and_verify_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            payload = sign_library(path, self.SECRET)
            self.assertEqual(payload["algorithm"], "hmac-sha256")
            sig_path = Path(tmp) / "lib.json.sig"
            self.assertTrue(sig_path.exists())
            result = verify_library_signature(path, self.SECRET)
            self.assertTrue(result.valid)
            self.assertEqual(result.reason, "ok")

    def test_tampered_entry_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            sign_library(path, self.SECRET)

            # Tamper: add a new entry.
            mutated = ArrayProgramLibrary.load(path)
            mutated.record(
                torch.tensor([0.0, 0.0, 1.0]),
                DiscreteArrayProgram(0, 4, 0, 0, 0.0),
                task_name="count",
            )
            mutated.save(path)

            result = verify_library_signature(path, self.SECRET)
            self.assertFalse(result.valid)
            self.assertEqual(result.reason, "digest mismatch")

    def test_hit_count_change_does_not_invalidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            sign_library(path, self.SECRET)

            # Simulate usage — hit count rises, but content is unchanged.
            used = ArrayProgramLibrary.load(path)
            used.lookup(torch.tensor([1.0, 0.0, 0.0]))
            used.lookup(torch.tensor([1.0, 0.0, 0.0]))
            used.save(path)

            result = verify_library_signature(path, self.SECRET)
            self.assertTrue(result.valid, msg=result.reason)

    def test_wrong_secret_fails_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            sign_library(path, self.SECRET)
            result = verify_library_signature(path, b"different-secret")
            self.assertFalse(result.valid)

    def test_missing_signature_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            # No signature yet.
            result = verify_library_signature(path, self.SECRET)
            self.assertFalse(result.valid)
            self.assertIn("not found", result.reason)

    def test_custom_sig_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            sig = Path(tmp) / "custom.sig"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            sign_library(path, self.SECRET, sig_path=sig)
            self.assertTrue(sig.exists())
            result = verify_library_signature(path, self.SECRET, sig_path=sig)
            self.assertTrue(result.valid)

    def test_signature_sidecar_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.json"
            lib = ArrayProgramLibrary()
            _populate(lib)
            lib.save(path)
            sign_library(path, self.SECRET)
            sig_path = Path(tmp) / "lib.json.sig"
            payload = json.loads(sig_path.read_text())
            self.assertEqual(payload["algorithm"], "hmac-sha256")
            self.assertIn("digest", payload)


if __name__ == "__main__":
    unittest.main()
