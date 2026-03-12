"""Sandbox-style verification helpers for buffered internal inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ncpu.self_optimizing.code_verifier import CodeVerifier


def _serialize_verification(verification: Any) -> Optional[dict[str, Any]]:
    if verification is None:
        return None
    if isinstance(verification, bool):
        return {"success": verification}
    if isinstance(verification, tuple):
        return {
            "success": bool(verification[0]) if verification else False,
            "error": verification[1] if len(verification) > 1 else None,
        }
    if hasattr(verification, "__dict__"):
        details: dict[str, Any] = {}
        for key, value in verification.__dict__.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                details[key] = value
            elif isinstance(value, (list, dict)):
                details[key] = value
            else:
                details[key] = repr(value)
        return details
    if isinstance(verification, dict):
        return verification
    return {"value": repr(verification)}


def _normalize_verification(verification: Any) -> tuple[bool, Optional[str], Optional[dict[str, Any]]]:
    if isinstance(verification, bool):
        return verification, (None if verification else "verification returned false"), {"success": verification}
    if isinstance(verification, tuple):
        success = bool(verification[0]) if verification else False
        error = verification[1] if len(verification) > 1 else None
        return success, error, _serialize_verification(verification)
    if hasattr(verification, "success"):
        success = bool(getattr(verification, "success"))
        error = getattr(verification, "error", None)
        return success, error, _serialize_verification(verification)
    raise TypeError(
        "Verifier must return a bool, (success, error) tuple, or an object with a 'success' attribute"
    )


def _summarize_failure(error: Optional[str], verification: Optional[dict[str, Any]]) -> str:
    if error:
        return error
    if not verification:
        return "verification failed"

    failed_tests = [item for item in verification.get("test_results", []) if not item.get("passed", False)]
    if failed_tests:
        first = failed_tests[0]
        if "error" in first:
            return f"test {first.get('test', '?')} raised {first['error']}"
        return (
            f"test {first.get('test', '?')} expected {first.get('expected')!r} "
            f"but got {first.get('actual')!r}"
        )

    return verification.get("error") or "verification failed"


@dataclass
class SandboxActionResult:
    """Outcome of hidden verification for a single candidate."""

    success: bool
    candidate_text: str
    normalized_candidate: str
    error: Optional[str]
    verification: Optional[dict[str, Any]]

    def failure_summary(self) -> str:
        return _summarize_failure(self.error, self.verification)

    def to_feedback_payload(self) -> dict[str, Any]:
        return {
            "response_text": self.candidate_text,
            "error": self.error,
            "verification": self.verification,
        }


class SandboxActionRunner:
    """Runs hidden verification actions before the controller commits output."""

    def __init__(self, code_verifier: Optional[CodeVerifier] = None):
        self.code_verifier = code_verifier or CodeVerifier()

    def verify_candidate(
        self,
        candidate_text: str,
        *,
        verifier: Optional[Callable[[str], Any]] = None,
        test_cases: Optional[list[dict[str, Any]]] = None,
    ) -> SandboxActionResult:
        normalized_candidate = candidate_text

        if test_cases is not None:
            normalized_candidate = self.code_verifier.extract_code(candidate_text)
            verification_raw = self.code_verifier.verify(candidate_text, test_cases=test_cases)
            verification = _serialize_verification(verification_raw)
            error = verification_raw.error if hasattr(verification_raw, "error") else None
            return SandboxActionResult(
                success=bool(getattr(verification_raw, "success", False)),
                candidate_text=candidate_text,
                normalized_candidate=normalized_candidate,
                error=_summarize_failure(error, verification),
                verification=verification,
            )

        if verifier is None:
            stripped = candidate_text.strip()
            success = bool(stripped)
            return SandboxActionResult(
                success=success,
                candidate_text=candidate_text,
                normalized_candidate=stripped,
                error=None if success else "empty candidate",
                verification={"success": success},
            )

        verification_raw = verifier(candidate_text)
        success, error, verification = _normalize_verification(verification_raw)
        return SandboxActionResult(
            success=success,
            candidate_text=candidate_text,
            normalized_candidate=candidate_text.strip(),
            error=_summarize_failure(error, verification),
            verification=verification,
        )
