"""Requirements IR — the structured contract between English and synthesis.

The real pipeline is::

    complex English  →  [proposer]  →  RequirementsIR  →  [nCPU synth+verify]  →  verified program

The IR is the interface seam. A *proposer* (today an LLM, tomorrow the
bottom-up Phase-B encoder) turns messy English into this structured
contract; the nCPU synthesizer turns the contract's I/O examples into a
program it *verifies* before returning. The proposer is untrusted — it can
hallucinate examples or a wrong signature — so everything downstream treats
the IR as a proposal to be checked, never as ground truth. The trust lives
in the verifier, exactly as elsewhere in nCPU.

The IR deliberately carries more than I/O pairs: a signature, natural-
language invariants, edge cases, and an optional reference implementation.
The synthesizer only consumes the typed I/O today, but the richer fields
are what let a human (or a future checker) audit whether the synthesized
program actually matches the intent, and they are the supervision targets
for training the bottom-up proposer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# nsynth's native parameter types — the synthesizer can only consume these.
_NSYNTH_TYPES = {"i64", "[i64]", "string"}


@dataclass
class ParamSpec:
    name: str
    type: str  # "i64" | "[i64]" | "string" | a free-form type the synth ignores

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "type": self.type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParamSpec":
        return cls(name=str(d["name"]), type=str(d.get("type", "i64")))


@dataclass
class IoExample:
    inputs: list[Any]
    expected: Any

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "expected": self.expected}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IoExample":
        return cls(inputs=list(d["inputs"]), expected=d["expected"])


@dataclass
class RequirementsIR:
    """A structured, verifiable function contract extracted from English."""

    entry_point: str
    description: str
    params: list[ParamSpec] = field(default_factory=list)
    return_type: str = "i64"
    io_examples: list[IoExample] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)  # NL properties
    edge_cases: list[str] = field(default_factory=list)  # NL edge descriptions
    reference_impl: Optional[str] = None  # untrusted proposer code
    reference_lang: str = "python"

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "return_type": self.return_type,
            "io_examples": [e.to_dict() for e in self.io_examples],
            "invariants": list(self.invariants),
            "edge_cases": list(self.edge_cases),
            "reference_impl": self.reference_impl,
            "reference_lang": self.reference_lang,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RequirementsIR":
        return cls(
            entry_point=str(d["entry_point"]),
            description=str(d.get("description", "")),
            params=[ParamSpec.from_dict(p) for p in d.get("params", [])],
            return_type=str(d.get("return_type", "i64")),
            io_examples=[IoExample.from_dict(e) for e in d.get("io_examples", [])],
            invariants=list(d.get("invariants", [])),
            edge_cases=list(d.get("edge_cases", [])),
            reference_impl=d.get("reference_impl"),
            reference_lang=str(d.get("reference_lang", "python")),
        )

    # ---- validation -------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of problems; empty means structurally usable."""
        problems: list[str] = []
        if not self.entry_point or not self.entry_point.isidentifier():
            problems.append(f"entry_point not a valid identifier: {self.entry_point!r}")
        if not self.io_examples:
            problems.append("no io_examples — nothing to synthesize or verify against")
        for i, ex in enumerate(self.io_examples):
            if not isinstance(ex.inputs, list) or not ex.inputs:
                problems.append(f"io_examples[{i}].inputs must be a non-empty list")
        return problems

    def synth_supported(self) -> bool:
        """True when every example's values fit nsynth's type system
        (i64 / [i64] / string in, integer out) so the synthesizer can try it."""
        for ex in self.io_examples:
            if isinstance(ex.expected, bool) or not isinstance(ex.expected, int):
                return False
            for v in ex.inputs:
                if isinstance(v, bool):
                    return False
                if isinstance(v, int) or isinstance(v, str):
                    continue
                if isinstance(v, list) and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in v
                ):
                    continue
                return False
        return bool(self.io_examples)

    def signature_str(self) -> str:
        """nsynth-style signature string built from params + return_type."""
        parts = []
        for p in self.params:
            t = p.type if p.type in _NSYNTH_TYPES else "i64"
            parts.append(f"{p.name}: {t}")
        ret = self.return_type if self.return_type in _NSYNTH_TYPES else "i64"
        return f"fn {self.entry_point}({', '.join(parts)}) -> {ret}"


# JSON Schema for the proposer's forced-tool output. Kept in lockstep with
# RequirementsIR.from_dict so a validated tool call deserializes cleanly.
REQUIREMENTS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entry_point": {
            "type": "string",
            "description": "A valid identifier naming the function to build.",
        },
        "description": {
            "type": "string",
            "description": "One-sentence precise statement of what the function computes.",
        },
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "description": "i64, [i64], string, or a precise type if richer.",
                    },
                },
                "required": ["name", "type"],
            },
        },
        "return_type": {"type": "string"},
        "io_examples": {
            "type": "array",
            "description": (
                "At least 4 concrete input/output examples covering normal and "
                "edge behavior. inputs is a list positional to params. Use real "
                "values, not placeholders."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "inputs": {"type": "array"},
                    "expected": {},
                },
                "required": ["inputs", "expected"],
            },
        },
        "invariants": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Natural-language properties that must always hold.",
        },
        "edge_cases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Natural-language edge cases the implementation must handle.",
        },
        "reference_impl": {
            "type": "string",
            "description": "A correct reference implementation (Python) of the function.",
        },
    },
    "required": ["entry_point", "description", "params", "return_type", "io_examples"],
}
