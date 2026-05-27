"""HMAC-based tamper-detection for library JSON files (NV2b).

A shipped `ArrayProgramLibrary` is a distribution artifact — it's the
record of what skills an LLM's reasoning crystallized into, and the basis
for compliance signoff. Any modification between the signing party and the
deployment environment invalidates the audit trail.

This module provides:

* `sign_library(path, secret, sig_path=None)` — compute an HMAC-SHA256 of
  the library's *canonical content* (entries sorted by fingerprint) and
  write a matching `.sig` file. The signature ignores fields that change
  during normal use (`hit_count`, `cached_at_step`) so a deployed library
  can accumulate hit statistics without invalidating its origin signature.
* `verify_library_signature(path, secret, sig_path=None)` — recompute the
  HMAC and check it against the sidecar. Returns `(bool, detail)`.

Keys are simple byte strings — the module doesn't impose a PKI. That's a
deliberate decision: the compliance story is "the library content hasn't
changed since signing," not "we trust a specific party." A full public-key
signature chain sits one layer on top of this primitive.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary


@dataclass
class SignatureVerification:
    """Outcome of a `verify_library_signature` call."""

    valid: bool
    reason: str
    expected_digest: Optional[str] = None
    computed_digest: Optional[str] = None


def _canonical_bytes(library: ArrayProgramLibrary) -> bytes:
    """Produce a canonical byte representation stable under hit-count changes."""
    entries = []
    for entry in library.entries:
        entries.append(
            {
                "signature": [float(v) for v in entry.signature],
                "program": entry.program.to_dict(),
                "task_name": entry.task_name,
                "convergence_gap": (
                    float(entry.convergence_gap)
                    if entry.convergence_gap is not None
                    else None
                ),
                # Deliberately exclude hit_count and cached_at_step —
                # those are deployment-time mutable.
            }
        )
    # Sort entries so order doesn't matter for the signature.
    entries.sort(key=lambda e: json.dumps(e["signature"], sort_keys=True))
    canonical = {
        "schema_version": "npcot/1.0",
        "config": library.config.to_dict(),
        "entries": entries,
    }
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sign_library(
    library_or_path: ArrayProgramLibrary | str | Path,
    secret: bytes,
    *,
    sig_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Sign a library; writes `.sig` next to it unless `sig_path` given."""
    if isinstance(library_or_path, (str, Path)):
        library = ArrayProgramLibrary.load(library_or_path)
        library_path = Path(library_or_path)
    else:
        library = library_or_path
        library_path = None

    canonical = _canonical_bytes(library)
    mac = hmac.new(secret, canonical, hashlib.sha256)
    digest = mac.hexdigest()
    payload = {
        "algorithm": "hmac-sha256",
        "schema_version": "npcot/1.0",
        "digest": digest,
        "byte_length": len(canonical),
    }

    resolved_sig_path = sig_path
    if resolved_sig_path is None and library_path is not None:
        resolved_sig_path = library_path.with_suffix(
            library_path.suffix + ".sig"
        )
    if resolved_sig_path is not None:
        resolved_sig_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload


def verify_library_signature(
    library_or_path: ArrayProgramLibrary | str | Path,
    secret: bytes,
    *,
    sig_path: Optional[Path] = None,
) -> SignatureVerification:
    """Verify a library's HMAC signature sidecar."""
    if isinstance(library_or_path, (str, Path)):
        library_path = Path(library_or_path)
        library = ArrayProgramLibrary.load(library_path)
    else:
        library_path = None
        library = library_or_path

    resolved_sig_path = sig_path
    if resolved_sig_path is None and library_path is not None:
        resolved_sig_path = library_path.with_suffix(
            library_path.suffix + ".sig"
        )
    if resolved_sig_path is None or not resolved_sig_path.exists():
        return SignatureVerification(
            valid=False,
            reason=f"signature file not found: {resolved_sig_path}",
        )
    try:
        payload = json.loads(resolved_sig_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SignatureVerification(
            valid=False,
            reason=f"cannot parse signature file: {exc}",
        )

    if payload.get("algorithm") != "hmac-sha256":
        return SignatureVerification(
            valid=False,
            reason=f"unsupported algorithm: {payload.get('algorithm')!r}",
        )
    expected_digest = payload.get("digest", "")
    canonical = _canonical_bytes(library)
    computed_digest = hmac.new(
        secret, canonical, hashlib.sha256
    ).hexdigest()
    if hmac.compare_digest(expected_digest, computed_digest):
        return SignatureVerification(
            valid=True,
            reason="ok",
            expected_digest=expected_digest,
            computed_digest=computed_digest,
        )
    return SignatureVerification(
        valid=False,
        reason="digest mismatch",
        expected_digest=expected_digest,
        computed_digest=computed_digest,
    )


__all__ = [
    "SignatureVerification",
    "sign_library",
    "verify_library_signature",
]
