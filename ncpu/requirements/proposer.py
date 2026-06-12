"""Proposers — English → RequirementsIR.

A proposer is the *untrusted* front-end of the pipeline. Its job is to turn
messy natural language into a structured :class:`RequirementsIR`, including
synthesizing concrete I/O examples when the prose gives none — that is the
hard part for "complex" prompts, and exactly where an LLM is strong and a
deterministic parser is not.

Two implementations:

* :class:`LLMProposer` — Claude (``claude-opus-4-8``) via a single forced
  tool call whose schema is the IR. The model reads the English and emits a
  precise contract: signature, ≥4 concrete examples, invariants, edge cases,
  and a reference implementation. Untrusted: everything it returns is a
  proposal the synthesizer/verifier then checks.
* :class:`DeterministicProposer` — the existing no-LLM prompt parser
  (asserts / doctests / arrow / "returns"). Works only when the English
  already contains explicit examples. Used as an offline fallback and as the
  honest baseline the LLM is measured against.

Both satisfy the same :class:`Proposer` protocol, so the future bottom-up
Phase-B encoder drops into the identical seam without touching the pipeline.
"""

from __future__ import annotations

import json
from typing import Optional, Protocol, runtime_checkable

from ncpu.requirements.ir import (
    REQUIREMENTS_TOOL_SCHEMA,
    IoExample,
    ParamSpec,
    RequirementsIR,
)


@runtime_checkable
class Proposer(Protocol):
    def propose(self, english: str) -> RequirementsIR: ...


class ProposerError(RuntimeError):
    """Raised when a proposer cannot produce an IR (no key, API error, …)."""


# ---------------------------------------------------------------------------
# LLM proposer
# ---------------------------------------------------------------------------

_SYSTEM = """\
You translate a natural-language software request into a precise, verifiable \
function contract. You are an UNTRUSTED proposer: a downstream synthesizer \
will build a program from your I/O examples and verify it, so your examples \
must be exactly correct.

Rules:
- Choose a single clear entry_point (a valid identifier) and a precise \
one-sentence description of what the function computes.
- Give params positionally with concrete types. Prefer i64, [i64], or string \
when the data is integer / integer-list / text; otherwise use the most precise \
type and the synthesizer will do its best.
- Provide AT LEAST 4 concrete io_examples — real values, not placeholders — \
covering normal cases AND the edge cases you list. inputs is positional to \
params. Every example must be correct under your own description; the program \
will be checked against them.
- List the invariants (properties that always hold) and edge_cases in plain \
language.
- Provide a correct Python reference_impl of the function.
Call the emit_requirements tool exactly once with the full contract."""

_TOOL = {
    "name": "emit_requirements",
    "description": "Emit the structured requirements contract for the request.",
    "input_schema": REQUIREMENTS_TOOL_SCHEMA,
}


class LLMProposer:
    """English → RequirementsIR via Claude, using a forced tool call.

    The forced tool (``tool_choice`` = the emit_requirements tool) guarantees
    a structured payload matching :data:`REQUIREMENTS_TOOL_SCHEMA`, which
    deserializes straight into a :class:`RequirementsIR`.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-4-8",
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        client: object = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client
        self._api_key = api_key

    def _resolve_key(self) -> Optional[str]:
        if self._api_key:
            return self._api_key
        import os

        # Default to the SDK's own ANTHROPIC_API_KEY, but fall back to
        # ANTHROPIC_TOKEN (used by some Claude Code / CI environments).
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_TOKEN")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProposerError(
                "anthropic SDK not installed — pip install anthropic"
            ) from exc
        key = self._resolve_key()
        if not key:
            raise ProposerError(
                "no Anthropic API key — set ANTHROPIC_API_KEY (or ANTHROPIC_TOKEN), "
                "or use the deterministic proposer"
            )
        try:
            if key.startswith("sk-ant-oat"):
                # Claude Code OAuth token: Bearer auth + oauth beta header,
                # not the x-api-key header a standard API key uses.
                self._client = anthropic.Anthropic(
                    auth_token=key,
                    default_headers={"anthropic-beta": "oauth-2025-04-20"},
                )
            else:
                self._client = anthropic.Anthropic(api_key=key)
        except Exception as exc:  # noqa: BLE001 — missing key etc.
            raise ProposerError(f"could not init Anthropic client: {exc}") from exc
        return self._client

    def propose(self, english: str) -> RequirementsIR:
        if not english or not english.strip():
            raise ProposerError("empty request")
        client = self._get_client()
        try:
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM,
                tools=[_TOOL],
                tool_choice={"type": "tool", "name": "emit_requirements"},
                messages=[{"role": "user", "content": english.strip()}],
            )
        except Exception as exc:  # noqa: BLE001
            raise ProposerError(f"Anthropic request failed: {exc}") from exc

        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "emit_requirements":
                payload = block.input
                if isinstance(payload, str):  # defensive: some stacks stringify
                    payload = json.loads(payload)
                return RequirementsIR.from_dict(payload)
        raise ProposerError("model did not return an emit_requirements tool call")


# ---------------------------------------------------------------------------
# Deterministic proposer (no LLM) — explicit-examples only
# ---------------------------------------------------------------------------

class DeterministicProposer:
    """No-LLM fallback: parse explicit examples out of the prompt.

    Reuses the autoresearch prompt parser. It only succeeds when the English
    already contains asserts / doctests / arrow notation / "returns" prose —
    it cannot invent examples from a pure prose spec. That limitation is the
    whole reason the LLM proposer exists; this one is the honest baseline."""

    def propose(self, english: str) -> RequirementsIR:
        from ncpu.autoresearch.prompt_parser import extract_from_prompt

        report = extract_from_prompt(english)
        if report.entry_point is None:
            raise ProposerError(
                "no entry point — deterministic proposer needs a def stub or "
                "examples that name the function"
            )
        io = [
            IoExample(inputs=list(p.args), expected=p.expected)
            for p in report.io_pairs
            if not p.kwargs
        ]
        # infer param names/types from the first example
        params: list[ParamSpec] = []
        if io:
            for i, v in enumerate(io[0].inputs):
                t = (
                    "[i64]" if isinstance(v, list)
                    else "string" if isinstance(v, str)
                    else "i64"
                )
                params.append(ParamSpec(name=chr(ord("a") + i), type=t))
        return RequirementsIR(
            entry_point=report.entry_point,
            description=f"Parsed from explicit examples ({len(io)} pairs).",
            params=params,
            return_type="i64",
            io_examples=io,
            invariants=[],
            edge_cases=[],
            reference_impl=None,
        )
