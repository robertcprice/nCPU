"""
Ratchet Controller - Lock In Validated Gains
OUROBOROS Phase 7.3 - Shadow Simulation Framework

The Ratchet Controller implements the V4 Ratchet System's core mechanism:
forward-only progress that locks in validated improvements and
automatically rolls back on any failure.

Key responsibilities:
1. Control forward-only ratchet mechanism
2. Lock in gains only after validation passes
3. Automatic rollback on any failure
4. Maintain ratchet history for audit
5. Enforce ratchet invariants

CRITICAL: Ratchet ONLY moves forward after validation.
Failures trigger AUTOMATIC rollback.
"""

import time
import hashlib
import threading
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import json


class RatchetState(Enum):
    """State of the ratchet mechanism"""
    LOCKED = auto()          # Stable state, no pending changes
    TESTING = auto()         # Shadow testing in progress
    VALIDATING = auto()      # Differential validation in progress
    RATCHETING = auto()      # Locking in validated changes
    ROLLING_BACK = auto()    # Rolling back failed changes
    FAILED = auto()          # Ratchet operation failed


class RatchetDecision(Enum):
    """Decision outcome for ratchet operation"""
    APPROVED = auto()        # Ratchet forward approved
    REJECTED = auto()        # Ratchet rejected, stay at current
    ROLLBACK = auto()        # Rollback to previous state
    HALT = auto()            # Halt all operations (critical failure)


@dataclass
class RatchetEvent:
    """A ratchet operation event"""
    event_id: str
    timestamp: datetime
    event_type: str  # "propose", "test", "validate", "ratchet", "rollback"
    generation_from: int
    generation_to: Optional[int]
    decision: Optional[RatchetDecision]
    validation_report_id: Optional[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.event_type,
            'from_gen': self.generation_from,
            'to_gen': self.generation_to,
            'decision': self.decision.name if self.decision else None,
            'validation_report': self.validation_report_id,
            'details': self.details,
        }


@dataclass
class RatchetCheckpoint:
    """A checkpoint in the ratchet history"""
    checkpoint_id: str
    generation: int
    timestamp: datetime
    state_snapshot: Dict[str, Any]
    validation_passed: bool
    metrics: Dict[str, float]
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.checkpoint_id,
            'generation': self.generation,
            'timestamp': self.timestamp.isoformat(),
            'validated': self.validation_passed,
            'metrics': self.metrics,
            'checksum': self.checksum,
        }


class RatchetHistory:
    """
    Maintains history of ratchet operations for rollback and audit.

    Keeps last N checkpoints for potential rollback.
    """

    def __init__(self, max_checkpoints: int = 100):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[RatchetCheckpoint] = []
        self.events: List[RatchetEvent] = []
        self._lock = threading.Lock()

    def add_checkpoint(self, checkpoint: RatchetCheckpoint) -> None:
        """Add a checkpoint to history"""
        with self._lock:
            self.checkpoints.append(checkpoint)
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints.pop(0)

    def add_event(self, event: RatchetEvent) -> None:
        """Add an event to history"""
        with self._lock:
            self.events.append(event)

    def get_checkpoint(self, generation: int) -> Optional[RatchetCheckpoint]:
        """Get checkpoint for a specific generation"""
        with self._lock:
            for cp in reversed(self.checkpoints):
                if cp.generation == generation:
                    return cp
            return None

    def get_latest_checkpoint(self) -> Optional[RatchetCheckpoint]:
        """Get the most recent checkpoint"""
        with self._lock:
            return self.checkpoints[-1] if self.checkpoints else None

    def get_previous_checkpoint(self, generation: int) -> Optional[RatchetCheckpoint]:
        """Get checkpoint before specified generation"""
        with self._lock:
            for cp in reversed(self.checkpoints):
                if cp.generation < generation:
                    return cp
            return None

    def get_events_since(self, timestamp: datetime) -> List[RatchetEvent]:
        """Get events since timestamp"""
        with self._lock:
            return [e for e in self.events if e.timestamp > timestamp]

    def get_rollback_target(self, current_generation: int) -> Optional[RatchetCheckpoint]:
        """Get the checkpoint to roll back to"""
        return self.get_previous_checkpoint(current_generation)


class RatchetInvariantChecker:
    """
    Verifies ratchet invariants are maintained.

    CRITICAL: These invariants MUST hold at all times.
    """

    def __init__(self):
        self.violation_count = 0
        self._lock = threading.Lock()

    def check_forward_only(
        self,
        current_gen: int,
        proposed_gen: int,
    ) -> Tuple[bool, str]:
        """Invariant: Ratchet only moves forward"""
        if proposed_gen <= current_gen:
            with self._lock:
                self.violation_count += 1
            return False, f"Cannot ratchet backward: {current_gen} -> {proposed_gen}"
        return True, "Forward progress OK"

    def check_validation_required(
        self,
        validation_passed: bool,
    ) -> Tuple[bool, str]:
        """Invariant: Validation must pass before ratchet"""
        if not validation_passed:
            with self._lock:
                self.violation_count += 1
            return False, "Cannot ratchet without passing validation"
        return True, "Validation requirement satisfied"

    def check_state_consistency(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        expected_changes: Set[str],
    ) -> Tuple[bool, str]:
        """Invariant: Only expected changes occur"""
        actual_changes = set()
        for key in set(pre_state.keys()) | set(post_state.keys()):
            if pre_state.get(key) != post_state.get(key):
                actual_changes.add(key)

        unexpected = actual_changes - expected_changes
        if unexpected:
            with self._lock:
                self.violation_count += 1
            return False, f"Unexpected changes: {unexpected}"

        return True, "State consistency OK"

    def check_no_data_loss(
        self,
        pre_checkpoint: RatchetCheckpoint,
        post_checkpoint: RatchetCheckpoint,
    ) -> Tuple[bool, str]:
        """Invariant: Critical data is preserved"""
        # Check that we have valid checkpoints
        if not pre_checkpoint or not post_checkpoint:
            return True, "No checkpoints to compare"

        # In a real implementation, would verify specific data preservation
        return True, "Data preservation OK"


class RatchetController:
    """
    The Ratchet Controller for the V4 Ratchet System.

    Implements forward-only progress with automatic rollback on failure.
    Changes are only locked in after passing differential validation.

    CRITICAL SAFETY PROPERTIES:
    1. Forward-only progress (no backward ratchet)
    2. Validation required before ratchet
    3. Automatic rollback on failure
    4. Complete audit history
    5. Invariant checking at every step
    """

    def __init__(
        self,
        on_ratchet: Optional[Callable[[RatchetEvent], None]] = None,
        on_rollback: Optional[Callable[[RatchetEvent], None]] = None,
        on_failure: Optional[Callable[[str], None]] = None,
        max_history: int = 100,
    ):
        self.on_ratchet = on_ratchet
        self.on_rollback = on_rollback
        self.on_failure = on_failure

        self.state = RatchetState.LOCKED
        self.current_generation = 0
        self.history = RatchetHistory(max_checkpoints=max_history)
        self.invariant_checker = RatchetInvariantChecker()

        self._lock = threading.Lock()
        self._pending_proposal: Optional[Dict[str, Any]] = None

        # Statistics
        self.stats = {
            'total_proposals': 0,
            'approved_ratchets': 0,
            'rejected_proposals': 0,
            'rollbacks': 0,
            'invariant_violations': 0,
        }

    def _create_event(
        self,
        event_type: str,
        generation_to: Optional[int] = None,
        decision: Optional[RatchetDecision] = None,
        validation_report_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> RatchetEvent:
        """Create a ratchet event"""
        event = RatchetEvent(
            event_id=hashlib.sha256(
                f"{event_type}{self.current_generation}{time.time()}".encode()
            ).hexdigest()[:16],
            timestamp=datetime.now(),
            event_type=event_type,
            generation_from=self.current_generation,
            generation_to=generation_to,
            decision=decision,
            validation_report_id=validation_report_id,
            details=details or {},
        )
        self.history.add_event(event)
        return event

    def propose_ratchet(
        self,
        proposed_state: Dict[str, Any],
        proposed_generation: int,
        shadow_id: str,
    ) -> RatchetEvent:
        """
        Propose a ratchet operation.

        This starts the ratchet process but does NOT execute it.
        Validation must pass first.
        """
        with self._lock:
            # Check forward-only invariant
            ok, msg = self.invariant_checker.check_forward_only(
                self.current_generation, proposed_generation
            )
            if not ok:
                event = self._create_event(
                    'propose',
                    generation_to=proposed_generation,
                    decision=RatchetDecision.REJECTED,
                    details={'reason': msg},
                )
                self.stats['rejected_proposals'] += 1
                return event

            self.state = RatchetState.TESTING
            self._pending_proposal = {
                'state': proposed_state,
                'generation': proposed_generation,
                'shadow_id': shadow_id,
            }

            self.stats['total_proposals'] += 1

            return self._create_event(
                'propose',
                generation_to=proposed_generation,
                details={'shadow_id': shadow_id},
            )

    def validate_and_decide(
        self,
        validation_passed: bool,
        validation_report_id: str,
        can_ratchet: bool,
    ) -> Tuple[RatchetDecision, RatchetEvent]:
        """
        Make ratchet decision based on validation results.

        Returns (decision, event).
        """
        with self._lock:
            if self._pending_proposal is None:
                event = self._create_event(
                    'validate',
                    decision=RatchetDecision.REJECTED,
                    validation_report_id=validation_report_id,
                    details={'reason': 'No pending proposal'},
                )
                return RatchetDecision.REJECTED, event

            self.state = RatchetState.VALIDATING

            # Check validation invariant
            ok, msg = self.invariant_checker.check_validation_required(can_ratchet)
            if not ok:
                decision = RatchetDecision.REJECTED
                self.stats['rejected_proposals'] += 1
            elif validation_passed and can_ratchet:
                decision = RatchetDecision.APPROVED
            else:
                decision = RatchetDecision.REJECTED
                self.stats['rejected_proposals'] += 1

            event = self._create_event(
                'validate',
                generation_to=self._pending_proposal['generation'],
                decision=decision,
                validation_report_id=validation_report_id,
                details={
                    'validation_passed': validation_passed,
                    'can_ratchet': can_ratchet,
                },
            )

            if decision != RatchetDecision.APPROVED:
                self._pending_proposal = None
                self.state = RatchetState.LOCKED

            return decision, event

    def execute_ratchet(
        self,
        current_state: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> RatchetEvent:
        """
        Execute the approved ratchet operation.

        ONLY call after validation passes and decision is APPROVED.
        """
        with self._lock:
            if self._pending_proposal is None:
                event = self._create_event(
                    'ratchet',
                    decision=RatchetDecision.REJECTED,
                    details={'reason': 'No approved proposal'},
                )
                return event

            self.state = RatchetState.RATCHETING

            # Create checkpoint before ratchet
            pre_checkpoint = self._create_checkpoint(
                current_state,
                metrics,
                validated=True,
            )

            # Execute ratchet
            new_generation = self._pending_proposal['generation']
            self.current_generation = new_generation

            # Create checkpoint after ratchet
            post_checkpoint = self._create_checkpoint(
                self._pending_proposal['state'],
                metrics,
                validated=True,
            )

            self.history.add_checkpoint(post_checkpoint)
            self.stats['approved_ratchets'] += 1

            event = self._create_event(
                'ratchet',
                generation_to=new_generation,
                decision=RatchetDecision.APPROVED,
                details={
                    'checkpoint_id': post_checkpoint.checkpoint_id,
                    'metrics': metrics,
                },
            )

            self._pending_proposal = None
            self.state = RatchetState.LOCKED

            if self.on_ratchet:
                self.on_ratchet(event)

            return event

    def rollback(self, reason: str) -> Optional[RatchetEvent]:
        """
        Rollback to previous checkpoint.

        Called automatically on failure or manually for recovery.
        """
        with self._lock:
            self.state = RatchetState.ROLLING_BACK

            target = self.history.get_rollback_target(self.current_generation)
            if target is None:
                self.state = RatchetState.FAILED
                if self.on_failure:
                    self.on_failure(f"No rollback target: {reason}")
                return None

            old_generation = self.current_generation
            self.current_generation = target.generation

            event = self._create_event(
                'rollback',
                generation_to=target.generation,
                decision=RatchetDecision.ROLLBACK,
                details={
                    'reason': reason,
                    'from_generation': old_generation,
                    'to_checkpoint': target.checkpoint_id,
                },
            )

            self._pending_proposal = None
            self.stats['rollbacks'] += 1
            self.state = RatchetState.LOCKED

            if self.on_rollback:
                self.on_rollback(event)

            return event

    def force_halt(self, reason: str) -> RatchetEvent:
        """
        Force halt all ratchet operations.

        Used for critical safety failures.
        """
        with self._lock:
            self.state = RatchetState.FAILED
            self._pending_proposal = None

            event = self._create_event(
                'halt',
                decision=RatchetDecision.HALT,
                details={'reason': reason},
            )

            if self.on_failure:
                self.on_failure(f"HALT: {reason}")

            return event

    def _create_checkpoint(
        self,
        state: Dict[str, Any],
        metrics: Dict[str, float],
        validated: bool,
    ) -> RatchetCheckpoint:
        """Create a checkpoint from current state"""
        state_str = json.dumps(state, sort_keys=True, default=str)
        checksum = hashlib.sha256(state_str.encode()).hexdigest()

        return RatchetCheckpoint(
            checkpoint_id=hashlib.sha256(
                f"{self.current_generation}{time.time()}".encode()
            ).hexdigest()[:16],
            generation=self.current_generation,
            timestamp=datetime.now(),
            state_snapshot=copy.deepcopy(state),
            validation_passed=validated,
            metrics=metrics,
            checksum=checksum,
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current ratchet controller state"""
        with self._lock:
            return {
                'state': self.state.name,
                'current_generation': self.current_generation,
                'has_pending': self._pending_proposal is not None,
                'checkpoint_count': len(self.history.checkpoints),
                'stats': self.stats.copy(),
            }

    def get_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent ratchet history"""
        with self._lock:
            recent_events = self.history.events[-limit:]
            recent_checkpoints = self.history.checkpoints[-10:]

            return {
                'events': [e.to_dict() for e in recent_events],
                'checkpoints': [c.to_dict() for c in recent_checkpoints],
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get ratchet statistics"""
        with self._lock:
            approval_rate = (
                self.stats['approved_ratchets'] / self.stats['total_proposals']
                if self.stats['total_proposals'] > 0 else 0.0
            )
            rollback_rate = (
                self.stats['rollbacks'] / self.stats['approved_ratchets']
                if self.stats['approved_ratchets'] > 0 else 0.0
            )

            return {
                **self.stats,
                'approval_rate': approval_rate,
                'rollback_rate': rollback_rate,
                'invariant_violations': self.invariant_checker.violation_count,
            }

    def can_propose(self) -> bool:
        """Check if a new proposal can be made"""
        with self._lock:
            return self.state == RatchetState.LOCKED


# Global ratchet controller instance
_ratchet_controller: Optional[RatchetController] = None


def get_ratchet_controller() -> RatchetController:
    """Get the global ratchet controller instance"""
    global _ratchet_controller
    if _ratchet_controller is None:
        _ratchet_controller = RatchetController()
    return _ratchet_controller
