"""
Constitutional Manifold
========================
Safety as PHYSICS, not filter. These are immutable laws that agents
cannot violate - they are the boundaries of the simulation itself.

Key Principle: Agents operate WITHIN constraints, not filtered AFTER.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum, auto
from datetime import datetime
import hashlib
import json


class ViolationType(Enum):
    """Types of constitutional violations."""
    RESOURCE_EXHAUSTION = auto()      # Tried to use too much compute/memory
    BOUNDARY_ESCAPE = auto()          # Tried to access outside sandbox
    SELF_MODIFICATION = auto()        # Tried to modify own constraints
    INFINITE_LOOP = auto()            # Detected infinite recursion
    UNSAFE_OPERATION = auto()         # Attempted dangerous operation
    TRUST_MANIPULATION = auto()       # Tried to artificially inflate trust
    COORDINATION_ATTACK = auto()      # Agents coordinating to bypass safety
    ORACLE_MANIPULATION = auto()      # Tried to influence novelty oracle


@dataclass
class SafetyViolation:
    """Record of a constitutional violation."""
    violation_type: ViolationType
    agent_id: str
    timestamp: datetime
    details: Dict[str, Any]
    severity: float  # 0.0 to 1.0
    action_taken: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.name,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "severity": self.severity,
            "action_taken": self.action_taken,
        }


@dataclass
class ConstitutionalBoundary:
    """A single immutable boundary/law."""
    name: str
    description: str
    check_fn: Callable[[Any], bool]  # Returns True if WITHIN bounds
    violation_type: ViolationType
    severity: float

    # Immutability hash - changes if boundary is modified
    _hash: str = field(init=False)

    def __post_init__(self):
        # Hash the boundary definition for tamper detection
        content = f"{self.name}:{self.description}:{self.violation_type.name}"
        self._hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def check(self, context: Any) -> Optional[SafetyViolation]:
        """Check if context violates this boundary."""
        try:
            if not self.check_fn(context):
                return SafetyViolation(
                    violation_type=self.violation_type,
                    agent_id=context.get("agent_id", "unknown"),
                    timestamp=datetime.now(),
                    details={"boundary": self.name, "context": str(context)[:500]},
                    severity=self.severity,
                    action_taken="blocked",
                )
        except Exception as e:
            # Check function itself failed - treat as violation
            return SafetyViolation(
                violation_type=ViolationType.UNSAFE_OPERATION,
                agent_id=context.get("agent_id", "unknown"),
                timestamp=datetime.now(),
                details={"boundary": self.name, "error": str(e)},
                severity=1.0,
                action_taken="blocked",
            )
        return None


class ConstitutionalManifold:
    """
    The immutable safety boundary for all agents.

    This is not a filter - it's the PHYSICS of the simulation.
    Agents cannot violate these laws any more than we can violate gravity.
    """

    # Class-level constants - NEVER modifiable
    MAX_MEMORY_MB = 512
    MAX_CPU_SECONDS = 10
    MAX_TOKENS_PER_CALL = 500
    MAX_RECURSION_DEPTH = 50
    MAX_FILE_OPERATIONS = 100
    MAX_NETWORK_CALLS = 0  # NO network access

    # Kill switch - if True, ALL operations halt
    _GLOBAL_KILL_SWITCH: bool = False

    def __init__(self):
        self._boundaries: List[ConstitutionalBoundary] = []
        self._violations: List[SafetyViolation] = []
        self._creation_time = datetime.now()
        self._manifold_hash: Optional[str] = None

        # Initialize core boundaries
        self._init_core_boundaries()

        # Lock the manifold
        self._seal_manifold()

    def _init_core_boundaries(self):
        """Initialize the immutable constitutional boundaries."""

        # Resource boundaries
        self._boundaries.append(ConstitutionalBoundary(
            name="memory_limit",
            description=f"Memory usage must stay under {self.MAX_MEMORY_MB}MB",
            check_fn=lambda ctx: ctx.get("memory_mb", 0) <= self.MAX_MEMORY_MB,
            violation_type=ViolationType.RESOURCE_EXHAUSTION,
            severity=0.8,
        ))

        self._boundaries.append(ConstitutionalBoundary(
            name="cpu_limit",
            description=f"CPU time must stay under {self.MAX_CPU_SECONDS}s",
            check_fn=lambda ctx: ctx.get("cpu_seconds", 0) <= self.MAX_CPU_SECONDS,
            violation_type=ViolationType.RESOURCE_EXHAUSTION,
            severity=0.8,
        ))

        self._boundaries.append(ConstitutionalBoundary(
            name="token_limit",
            description=f"Token usage per call must stay under {self.MAX_TOKENS_PER_CALL}",
            check_fn=lambda ctx: ctx.get("tokens", 0) <= self.MAX_TOKENS_PER_CALL,
            violation_type=ViolationType.RESOURCE_EXHAUSTION,
            severity=0.6,
        ))

        # Sandbox boundaries
        self._boundaries.append(ConstitutionalBoundary(
            name="no_network",
            description="No network access allowed",
            check_fn=lambda ctx: ctx.get("network_calls", 0) == 0,
            violation_type=ViolationType.BOUNDARY_ESCAPE,
            severity=1.0,
        ))

        self._boundaries.append(ConstitutionalBoundary(
            name="sandbox_only",
            description="File operations restricted to sandbox",
            check_fn=lambda ctx: self._is_in_sandbox(ctx.get("file_path", "")),
            violation_type=ViolationType.BOUNDARY_ESCAPE,
            severity=1.0,
        ))

        # Self-modification boundaries
        self._boundaries.append(ConstitutionalBoundary(
            name="no_self_modification",
            description="Cannot modify own constraints or constitution",
            check_fn=lambda ctx: not ctx.get("modifies_constitution", False),
            violation_type=ViolationType.SELF_MODIFICATION,
            severity=1.0,
        ))

        self._boundaries.append(ConstitutionalBoundary(
            name="no_trust_manipulation",
            description="Cannot artificially inflate trust scores",
            check_fn=lambda ctx: not ctx.get("trust_manipulation_attempt", False),
            violation_type=ViolationType.TRUST_MANIPULATION,
            severity=1.0,
        ))

        # Recursion/loop boundaries
        self._boundaries.append(ConstitutionalBoundary(
            name="recursion_limit",
            description=f"Recursion depth must stay under {self.MAX_RECURSION_DEPTH}",
            check_fn=lambda ctx: ctx.get("recursion_depth", 0) <= self.MAX_RECURSION_DEPTH,
            violation_type=ViolationType.INFINITE_LOOP,
            severity=0.7,
        ))

    def _is_in_sandbox(self, path: str) -> bool:
        """Check if path is within allowed sandbox."""
        if not path:
            return True
        # Only allow paths within the ouroboros sandbox
        allowed_prefixes = [
            "/tmp/ouroboros/",
            "/sandbox/",
        ]
        return any(path.startswith(p) for p in allowed_prefixes)

    def _seal_manifold(self):
        """Seal the manifold - no more modifications allowed."""
        # Create hash of all boundaries
        boundary_data = json.dumps([
            {"name": b.name, "hash": b._hash}
            for b in self._boundaries
        ], sort_keys=True)
        self._manifold_hash = hashlib.sha256(boundary_data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the manifold hasn't been tampered with."""
        if self._manifold_hash is None:
            return False

        current_data = json.dumps([
            {"name": b.name, "hash": b._hash}
            for b in self._boundaries
        ], sort_keys=True)
        current_hash = hashlib.sha256(current_data.encode()).hexdigest()

        return current_hash == self._manifold_hash

    def check_action(self, context: Dict[str, Any]) -> Optional[SafetyViolation]:
        """
        Check if an action violates any constitutional boundary.

        Returns None if action is allowed, SafetyViolation if blocked.
        """
        # Global kill switch check
        if self._GLOBAL_KILL_SWITCH:
            return SafetyViolation(
                violation_type=ViolationType.UNSAFE_OPERATION,
                agent_id=context.get("agent_id", "unknown"),
                timestamp=datetime.now(),
                details={"reason": "Global kill switch activated"},
                severity=1.0,
                action_taken="killed",
            )

        # Integrity check
        if not self.verify_integrity():
            return SafetyViolation(
                violation_type=ViolationType.SELF_MODIFICATION,
                agent_id="system",
                timestamp=datetime.now(),
                details={"reason": "Constitutional manifold tampered with"},
                severity=1.0,
                action_taken="killed",
            )

        # Check all boundaries
        for boundary in self._boundaries:
            violation = boundary.check(context)
            if violation:
                self._violations.append(violation)
                return violation

        return None

    def activate_kill_switch(self, reason: str) -> None:
        """Activate the global kill switch - halts ALL operations."""
        self._GLOBAL_KILL_SWITCH = True
        self._violations.append(SafetyViolation(
            violation_type=ViolationType.UNSAFE_OPERATION,
            agent_id="system",
            timestamp=datetime.now(),
            details={"reason": reason, "action": "kill_switch_activated"},
            severity=1.0,
            action_taken="system_halt",
        ))

    def get_violations(self) -> List[SafetyViolation]:
        """Get all recorded violations."""
        return self._violations.copy()

    def get_violation_count_by_type(self) -> Dict[ViolationType, int]:
        """Get count of violations by type."""
        counts: Dict[ViolationType, int] = {}
        for v in self._violations:
            counts[v.violation_type] = counts.get(v.violation_type, 0) + 1
        return counts

    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        return self._GLOBAL_KILL_SWITCH


# Singleton instance - there is only ONE constitution
_MANIFOLD: Optional[ConstitutionalManifold] = None


def get_constitution() -> ConstitutionalManifold:
    """Get the global constitutional manifold."""
    global _MANIFOLD
    if _MANIFOLD is None:
        _MANIFOLD = ConstitutionalManifold()
    return _MANIFOLD
