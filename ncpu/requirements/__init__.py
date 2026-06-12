"""English → verified program — the requirements pipeline.

A proposer (LLM today, bottom-up encoder later) turns complex English into a
structured RequirementsIR; the nCPU synthesizer turns its I/O examples into a
program it verifies, with a held-out generalization check and an optional
cross-check against the proposer's reference. The trust lives in the verifier.
"""

from ncpu.requirements.ir import IoExample, ParamSpec, RequirementsIR
from ncpu.requirements.pipeline import ResolvedRequirement, resolve
from ncpu.requirements.proposer import (
    DeterministicProposer,
    LLMProposer,
    Proposer,
    ProposerError,
)

__all__ = [
    "RequirementsIR",
    "ParamSpec",
    "IoExample",
    "ResolvedRequirement",
    "resolve",
    "Proposer",
    "ProposerError",
    "LLMProposer",
    "DeterministicProposer",
]
