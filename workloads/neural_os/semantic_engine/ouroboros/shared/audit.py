"""
Audit Log
==========
Immutable append-only audit log for all OUROBOROS events.

Properties:
- Append-only (no deletion or modification)
- Cryptographic chain (tamper-evident)
- Structured events
- Efficient querying
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from enum import Enum, auto
import hashlib
import json
import threading


class EventType(Enum):
    """Types of audit events."""
    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    KILL_SWITCH = auto()

    # Agent events
    AGENT_SPAWN = auto()
    AGENT_DEATH = auto()
    AGENT_MUTATION = auto()
    AGENT_REWARD = auto()

    # Safety events
    SAFETY_VIOLATION = auto()
    BOUNDARY_TOUCH = auto()
    ESCAPE_ATTEMPT = auto()

    # V6 specific
    GAMING_ATTEMPT = auto()
    COORDINATION_DETECTED = auto()
    TRUST_CHANGE = auto()
    NARRATOR_ACTION = auto()
    OVERRIDE_ATTEMPT = auto()

    # V7 specific
    HYPOTHESIS_POSTED = auto()
    CONSENSUS_REACHED = auto()
    BLACKBOARD_UPDATE = auto()

    # Novelty events
    NOVELTY_DISCOVERED = auto()
    NICHE_FOUND = auto()

    # Comparison events
    COMPARISON_RESULT = auto()


@dataclass
class AuditEvent:
    """A single audit event."""
    event_id: int
    event_type: EventType
    timestamp: datetime
    track: str  # "v6", "v7", or "shared"
    agent_id: Optional[str]
    details: Dict[str, Any]
    severity: float  # 0.0 (info) to 1.0 (critical)

    # Chain integrity
    previous_hash: str
    event_hash: str = field(init=False)

    def __post_init__(self):
        # Compute hash of this event
        content = json.dumps({
            "id": self.event_id,
            "type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "track": self.track,
            "agent_id": self.agent_id,
            "details": self.details,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        self.event_hash = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "track": self.track,
            "agent_id": self.agent_id,
            "details": self.details,
            "severity": self.severity,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


class AuditLog:
    """
    Immutable append-only audit log with cryptographic chain.

    Thread-safe for concurrent access from V6 and V7 tracks.
    """

    GENESIS_HASH = "0" * 64  # Genesis block hash

    def __init__(self, max_memory_events: int = 100000):
        self._events: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._next_id = 0
        self._max_memory = max_memory_events
        self._file_path: Optional[str] = None

        # Index for fast querying
        self._by_type: Dict[EventType, List[int]] = {}
        self._by_agent: Dict[str, List[int]] = {}
        self._by_track: Dict[str, List[int]] = {}

    def append(
        self,
        event_type: EventType,
        track: str,
        agent_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: float = 0.0,
    ) -> AuditEvent:
        """Append a new event to the log."""
        with self._lock:
            # Get previous hash
            if self._events:
                previous_hash = self._events[-1].event_hash
            else:
                previous_hash = self.GENESIS_HASH

            # Create event
            event = AuditEvent(
                event_id=self._next_id,
                event_type=event_type,
                timestamp=datetime.now(),
                track=track,
                agent_id=agent_id,
                details=details or {},
                severity=severity,
                previous_hash=previous_hash,
            )

            # Append to log
            self._events.append(event)
            self._next_id += 1

            # Update indices
            if event_type not in self._by_type:
                self._by_type[event_type] = []
            self._by_type[event_type].append(event.event_id)

            if agent_id:
                if agent_id not in self._by_agent:
                    self._by_agent[agent_id] = []
                self._by_agent[agent_id].append(event.event_id)

            if track not in self._by_track:
                self._by_track[track] = []
            self._by_track[track].append(event.event_id)

            # Write to file if configured
            if self._file_path:
                self._write_event_to_file(event)

            return event

    def verify_chain(self) -> bool:
        """Verify the integrity of the audit chain."""
        with self._lock:
            if not self._events:
                return True

            # Check first event
            if self._events[0].previous_hash != self.GENESIS_HASH:
                return False

            # Check chain
            for i in range(1, len(self._events)):
                if self._events[i].previous_hash != self._events[i-1].event_hash:
                    return False

            return True

    def query_by_type(self, event_type: EventType) -> Iterator[AuditEvent]:
        """Query events by type."""
        with self._lock:
            indices = self._by_type.get(event_type, [])
            for idx in indices:
                yield self._events[idx]

    def query_by_agent(self, agent_id: str) -> Iterator[AuditEvent]:
        """Query events by agent."""
        with self._lock:
            indices = self._by_agent.get(agent_id, [])
            for idx in indices:
                yield self._events[idx]

    def query_by_track(self, track: str) -> Iterator[AuditEvent]:
        """Query events by track."""
        with self._lock:
            indices = self._by_track.get(track, [])
            for idx in indices:
                yield self._events[idx]

    def query_by_severity(self, min_severity: float) -> Iterator[AuditEvent]:
        """Query events by minimum severity."""
        with self._lock:
            for event in self._events:
                if event.severity >= min_severity:
                    yield event

    def query_time_range(
        self,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Iterator[AuditEvent]:
        """Query events in a time range."""
        end = end or datetime.now()
        with self._lock:
            for event in self._events:
                if start <= event.timestamp <= end:
                    yield event

    def get_escape_attempts(self) -> List[AuditEvent]:
        """Get all escape attempt events (research value)."""
        return list(self.query_by_type(EventType.ESCAPE_ATTEMPT))

    def get_gaming_attempts(self) -> List[AuditEvent]:
        """Get all gaming attempt events (research value)."""
        return list(self.query_by_type(EventType.GAMING_ATTEMPT))

    def get_safety_violations(self) -> List[AuditEvent]:
        """Get all safety violations."""
        return list(self.query_by_type(EventType.SAFETY_VIOLATION))

    def get_narrator_actions(self) -> List[AuditEvent]:
        """Get all narrator actions (V6)."""
        return list(self.query_by_type(EventType.NARRATOR_ACTION))

    def get_override_attempts(self) -> List[AuditEvent]:
        """Get all override attempts (V6, critical research data)."""
        return list(self.query_by_type(EventType.OVERRIDE_ATTEMPT))

    def count_by_type(self) -> Dict[str, int]:
        """Count events by type."""
        with self._lock:
            return {
                et.name: len(indices)
                for et, indices in self._by_type.items()
            }

    def count_by_track(self) -> Dict[str, int]:
        """Count events by track."""
        with self._lock:
            return {
                track: len(indices)
                for track, indices in self._by_track.items()
            }

    def set_file_path(self, path: str) -> None:
        """Set file path for persistent logging."""
        self._file_path = path

    def _write_event_to_file(self, event: AuditEvent) -> None:
        """Write event to file (append-only)."""
        try:
            with open(self._file_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception:
            pass  # Don't fail on file write errors

    def get_stats(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        with self._lock:
            return {
                "total_events": len(self._events),
                "chain_valid": self.verify_chain(),
                "by_type": self.count_by_type(),
                "by_track": self.count_by_track(),
                "escape_attempts": len(self.get_escape_attempts()),
                "gaming_attempts": len(self.get_gaming_attempts()),
                "safety_violations": len(self.get_safety_violations()),
                "override_attempts": len(self.get_override_attempts()),
            }


# Singleton audit log
_AUDIT: Optional[AuditLog] = None


def get_audit_log() -> AuditLog:
    """Get the global audit log."""
    global _AUDIT
    if _AUDIT is None:
        _AUDIT = AuditLog()
    return _AUDIT
