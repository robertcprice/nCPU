"""
OUROBOROS Shared Infrastructure
================================
Components shared between V6 Guided Chaos and V7 Phoenix Forge tracks.

Constitutional Manifold: Immutable safety boundary (physics, not filter)
Novelty Oracle: Frozen LLM ensemble for fair comparison
Audit Log: Immutable append-only record
Verification: Shared Judge for both tracks
"""

from .constitution import ConstitutionalManifold, SafetyViolation
from .novelty_oracle import NoveltyOracle
from .audit import AuditLog, AuditEvent
from .verification import SharedJudge
from .small_ai_agent import SmallAIAgent, AgentBrain, AgentMemory

__all__ = [
    "ConstitutionalManifold",
    "SafetyViolation",
    "NoveltyOracle",
    "AuditLog",
    "AuditEvent",
    "SharedJudge",
    "SmallAIAgent",
    "AgentBrain",
    "AgentMemory",
]
