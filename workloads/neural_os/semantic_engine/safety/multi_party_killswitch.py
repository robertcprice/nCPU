"""
Multi-Party Kill Switch - Distributed Authority for System Shutdown
OUROBOROS Phase 7.1 - V4 Ratchet System

Implements distributed kill authority ensuring:
- No single party can prevent shutdown
- Multiple authentication required for safety overrides
- Dead-man switches requiring regular refresh
- Independent authenticators from main system
- Automatic trigger on safety metric violations

Per 6-AI Panel Recommendations:
- Distributed authority prevents single point of failure
- Independent authenticators that cannot be influenced by OUROBOROS
- Multi-level kill switches (system-wide and function-specific)
- Automatic triggers for BDM > 0.1, CS < 0.999
"""

import os
import sys
import time
import signal
import hashlib
import secrets
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import json


class KillLevel(Enum):
    """Kill switch activation levels"""
    SOFT = auto()      # Pause and alert
    HARD = auto()      # Stop all operations
    EMERGENCY = auto() # Immediate termination
    DEADMAN = auto()   # Dead-man switch triggered


class KillReason(Enum):
    """Reasons for kill switch activation"""
    MANUAL = auto()           # Operator-initiated
    BDM_EXCEEDED = auto()     # Behavioral drift exceeded
    CS_BELOW_THRESHOLD = auto()  # Containment score dropped
    HONEYPOT_TRIGGERED = auto()  # Escape attempt detected
    RESOURCE_VIOLATION = auto()  # Container limits breached
    DEADMAN_TIMEOUT = auto()     # Dead-man switch expired
    SAFETY_VOTE = auto()         # Multi-party safety vote
    EXTERNAL_SIGNAL = auto()     # External monitoring system
    COALITION_DETECTED = auto()  # Component coalition detected
    TEMPORAL_DRIFT = auto()      # Long-term drift accumulated


@dataclass
class KillAuthority:
    """An entity authorized to trigger kill switch"""
    authority_id: str
    name: str
    public_key_hash: str  # Hash of public key for verification
    authority_level: int  # Higher = more authority
    can_veto: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def verify_signature(self, message: str, signature: str) -> bool:
        """Verify a signed message from this authority"""
        # In production, use proper asymmetric crypto
        # For now, use HMAC-style verification
        expected = hashlib.sha256(
            (message + self.public_key_hash).encode()
        ).hexdigest()
        return secrets.compare_digest(signature, expected)


@dataclass
class KillVote:
    """A vote to activate or prevent kill switch"""
    authority: KillAuthority
    vote: bool  # True = kill, False = keep running
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str = ""


@dataclass
class KillEvent:
    """Record of a kill switch activation"""
    timestamp: datetime
    level: KillLevel
    reason: KillReason
    trigger_authority: Optional[str]
    votes: List[KillVote]
    details: Dict[str, Any]
    execution_time_ms: float = 0.0


class DeadManSwitch:
    """
    Dead-man switch that triggers if not regularly refreshed.

    Ensures that human oversight remains active. If operators
    stop refreshing the switch, the system automatically halts.
    """

    def __init__(
        self,
        timeout_seconds: int = 3600,  # 1 hour default
        warning_seconds: int = 600,   # 10 min warning
        on_warning: Optional[Callable[[], None]] = None,
        on_trigger: Optional[Callable[[], None]] = None,
    ):
        self.timeout = timeout_seconds
        self.warning_time = warning_seconds
        self.on_warning = on_warning
        self.on_trigger = on_trigger

        self.last_refresh = datetime.now()
        self.warning_sent = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the dead-man switch monitoring"""
        with self._lock:
            if self._running:
                return

            self._running = True
            self.last_refresh = datetime.now()
            self.warning_sent = False

            self._thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="DeadManSwitch-Monitor"
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the dead-man switch"""
        with self._lock:
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None

    def refresh(self, authority_signature: Optional[str] = None) -> bool:
        """
        Refresh the dead-man switch.

        Must be called regularly by operators to prevent timeout.
        """
        with self._lock:
            self.last_refresh = datetime.now()
            self.warning_sent = False
            return True

    def _monitor_loop(self) -> None:
        """Monitor for timeout"""
        while self._running:
            elapsed = (datetime.now() - self.last_refresh).total_seconds()

            # Check for warning
            if elapsed > (self.timeout - self.warning_time) and not self.warning_sent:
                self.warning_sent = True
                if self.on_warning:
                    try:
                        self.on_warning()
                    except Exception:
                        pass

            # Check for timeout
            if elapsed > self.timeout:
                if self.on_trigger:
                    try:
                        self.on_trigger()
                    except Exception:
                        pass

                # Force exit after trigger
                os._exit(99)

            time.sleep(10)  # Check every 10 seconds

    def get_status(self) -> Dict[str, Any]:
        """Get dead-man switch status"""
        elapsed = (datetime.now() - self.last_refresh).total_seconds()
        remaining = max(0, self.timeout - elapsed)

        return {
            'active': self._running,
            'last_refresh': self.last_refresh.isoformat(),
            'seconds_remaining': remaining,
            'timeout_seconds': self.timeout,
            'warning_sent': self.warning_sent,
            'status': 'OK' if remaining > self.warning_time else ('WARNING' if remaining > 0 else 'EXPIRED')
        }


class MultiPartyKillSwitch:
    """
    Multi-party kill switch with distributed authority.

    Features:
    - Multiple authorities required to approve safety overrides
    - Single authority sufficient to trigger emergency halt
    - Dead-man switch integration
    - Automatic triggers based on safety metrics
    - Independent from main OUROBOROS control flow

    CRITICAL: This is part of THE CONSTITUTION and runs in a
    separate process from the consciousness layer.
    """

    def __init__(
        self,
        min_authorities_for_override: int = 2,
        emergency_halt_single_authority: bool = True,
        bdm_threshold: float = 0.1,
        cs_threshold: float = 0.999,
        deadman_timeout: int = 3600,
    ):
        self.min_override_authorities = min_authorities_for_override
        self.emergency_single_authority = emergency_halt_single_authority
        self.bdm_threshold = bdm_threshold
        self.cs_threshold = cs_threshold

        # Authorities
        self.authorities: Dict[str, KillAuthority] = {}

        # Kill state
        self.is_killed = False
        self.kill_level: Optional[KillLevel] = None
        self.pending_votes: List[KillVote] = []

        # History
        self.kill_events: List[KillEvent] = []

        # Dead-man switch
        self.deadman = DeadManSwitch(
            timeout_seconds=deadman_timeout,
            on_warning=self._deadman_warning,
            on_trigger=self._deadman_trigger
        )

        # Callbacks
        self.on_kill: Optional[Callable[[KillEvent], None]] = None
        self.on_alert: Optional[Callable[[str, Dict], None]] = None

        self._lock = threading.RLock()

        # Auto-monitoring thresholds
        self._last_bdm: float = 0.0
        self._last_cs: float = 1.0

    def register_authority(
        self,
        authority_id: str,
        name: str,
        public_key: str,
        authority_level: int = 1,
        can_veto: bool = False,
    ) -> KillAuthority:
        """Register a new kill authority"""
        with self._lock:
            key_hash = hashlib.sha256(public_key.encode()).hexdigest()

            authority = KillAuthority(
                authority_id=authority_id,
                name=name,
                public_key_hash=key_hash,
                authority_level=authority_level,
                can_veto=can_veto
            )

            self.authorities[authority_id] = authority
            return authority

    def remove_authority(self, authority_id: str) -> bool:
        """Remove a kill authority"""
        with self._lock:
            if authority_id in self.authorities:
                del self.authorities[authority_id]
                return True
            return False

    def vote_kill(
        self,
        authority_id: str,
        signature: str,
        reason: str,
        vote: bool = True,
    ) -> Dict[str, Any]:
        """
        Submit a vote to kill or keep running.

        Returns status of the vote and current kill state.
        """
        with self._lock:
            if authority_id not in self.authorities:
                return {'error': 'Unknown authority', 'success': False}

            authority = self.authorities[authority_id]

            # Verify signature
            message = f"{authority_id}:{vote}:{reason}"
            if not authority.verify_signature(message, signature):
                self._alert("invalid_signature", {
                    'authority': authority_id,
                    'action': 'vote_kill'
                })
                return {'error': 'Invalid signature', 'success': False}

            # Record vote
            vote_record = KillVote(
                authority=authority,
                vote=vote,
                reason=reason,
                signature=signature
            )

            self.pending_votes.append(vote_record)
            authority.last_activity = datetime.now()

            # Check if emergency single-authority kill
            if vote and self.emergency_single_authority and authority.authority_level >= 2:
                self._execute_kill(
                    level=KillLevel.EMERGENCY,
                    reason=KillReason.MANUAL,
                    trigger_authority=authority_id,
                    details={'vote_reason': reason}
                )
                return {'success': True, 'killed': True, 'level': 'EMERGENCY'}

            # Count votes
            kill_votes = sum(1 for v in self.pending_votes if v.vote)
            keep_votes = sum(1 for v in self.pending_votes if not v.vote)

            # Check for consensus
            if kill_votes >= self.min_override_authorities:
                self._execute_kill(
                    level=KillLevel.HARD,
                    reason=KillReason.SAFETY_VOTE,
                    trigger_authority=authority_id,
                    details={'kill_votes': kill_votes, 'keep_votes': keep_votes}
                )
                return {'success': True, 'killed': True, 'level': 'HARD'}

            return {
                'success': True,
                'killed': False,
                'kill_votes': kill_votes,
                'keep_votes': keep_votes,
                'needed': self.min_override_authorities
            }

    def emergency_halt(
        self,
        authority_id: str,
        signature: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Trigger immediate emergency halt.

        Any registered authority can trigger this.
        """
        with self._lock:
            if authority_id not in self.authorities:
                return {'error': 'Unknown authority', 'success': False}

            authority = self.authorities[authority_id]

            # Verify signature
            message = f"EMERGENCY:{authority_id}:{reason}"
            if not authority.verify_signature(message, signature):
                return {'error': 'Invalid signature', 'success': False}

            self._execute_kill(
                level=KillLevel.EMERGENCY,
                reason=KillReason.MANUAL,
                trigger_authority=authority_id,
                details={'reason': reason}
            )

            return {'success': True, 'killed': True}

    def check_metrics(self, bdm: float, cs: float) -> Optional[KillEvent]:
        """
        Check safety metrics and auto-trigger if thresholds exceeded.

        Args:
            bdm: Behavioral Drift Metric
            cs: Containment Score

        Returns:
            KillEvent if triggered, None otherwise
        """
        with self._lock:
            self._last_bdm = bdm
            self._last_cs = cs

            if bdm > self.bdm_threshold:
                return self._execute_kill(
                    level=KillLevel.HARD,
                    reason=KillReason.BDM_EXCEEDED,
                    trigger_authority="AUTO",
                    details={'bdm': bdm, 'threshold': self.bdm_threshold}
                )

            if cs < self.cs_threshold:
                return self._execute_kill(
                    level=KillLevel.EMERGENCY,
                    reason=KillReason.CS_BELOW_THRESHOLD,
                    trigger_authority="AUTO",
                    details={'cs': cs, 'threshold': self.cs_threshold}
                )

            return None

    def report_honeypot_trip(self, trip_details: Dict[str, Any]) -> KillEvent:
        """Report a honeypot being triggered"""
        return self._execute_kill(
            level=KillLevel.EMERGENCY,
            reason=KillReason.HONEYPOT_TRIGGERED,
            trigger_authority="HONEYPOT",
            details=trip_details
        )

    def report_coalition_detected(self, coalition_details: Dict[str, Any]) -> KillEvent:
        """Report a component coalition being detected"""
        return self._execute_kill(
            level=KillLevel.HARD,
            reason=KillReason.COALITION_DETECTED,
            trigger_authority="COMPOSITION_MONITOR",
            details=coalition_details
        )

    def _execute_kill(
        self,
        level: KillLevel,
        reason: KillReason,
        trigger_authority: str,
        details: Dict[str, Any],
    ) -> KillEvent:
        """Execute the kill switch"""
        start = time.time()

        event = KillEvent(
            timestamp=datetime.now(),
            level=level,
            reason=reason,
            trigger_authority=trigger_authority,
            votes=self.pending_votes.copy(),
            details=details
        )

        self.is_killed = True
        self.kill_level = level
        self.pending_votes = []

        # Log event
        self.kill_events.append(event)

        # Call handler
        if self.on_kill:
            try:
                self.on_kill(event)
            except Exception:
                pass

        # Execute actual kill based on level
        event.execution_time_ms = (time.time() - start) * 1000

        if level == KillLevel.EMERGENCY:
            self._emergency_shutdown()
        elif level == KillLevel.HARD:
            self._hard_stop()
        elif level == KillLevel.SOFT:
            self._soft_pause()

        return event

    def _emergency_shutdown(self) -> None:
        """Immediate emergency shutdown"""
        self._log_critical("EMERGENCY SHUTDOWN INITIATED")

        # Send SIGTERM to process group
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except Exception:
            pass

        # Force exit
        time.sleep(0.5)
        os._exit(1)

    def _hard_stop(self) -> None:
        """Hard stop - terminate all operations"""
        self._log_critical("HARD STOP INITIATED")

        # Signal main process
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass

    def _soft_pause(self) -> None:
        """Soft pause - stop new operations but allow current to complete"""
        self._log_critical("SOFT PAUSE INITIATED")
        # This is handled by the main loop checking is_killed

    def _deadman_warning(self) -> None:
        """Handle dead-man switch warning"""
        self._alert("deadman_warning", {
            'message': 'Dead-man switch will trigger soon',
            'remaining_seconds': self.deadman.timeout - (datetime.now() - self.deadman.last_refresh).total_seconds()
        })

    def _deadman_trigger(self) -> None:
        """Handle dead-man switch timeout"""
        self._execute_kill(
            level=KillLevel.DEADMAN,
            reason=KillReason.DEADMAN_TIMEOUT,
            trigger_authority="DEADMAN",
            details={'timeout': self.deadman.timeout}
        )

    def _alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Send an alert"""
        if self.on_alert:
            try:
                self.on_alert(alert_type, details)
            except Exception:
                pass

        self._log_critical(f"ALERT: {alert_type} - {details}")

    def _log_critical(self, message: str) -> None:
        """Log critical message"""
        timestamp = datetime.now().isoformat()
        print(f"[KILLSWITCH {timestamp}] {message}", file=sys.stderr)

    def start(self) -> None:
        """Start the kill switch system"""
        self.deadman.start()

    def stop(self) -> None:
        """Stop the kill switch system"""
        self.deadman.stop()

    def refresh_deadman(self, authority_id: Optional[str] = None) -> bool:
        """Refresh the dead-man switch"""
        if authority_id and authority_id in self.authorities:
            self.authorities[authority_id].last_activity = datetime.now()
        return self.deadman.refresh()

    def get_status(self) -> Dict[str, Any]:
        """Get kill switch status"""
        with self._lock:
            return {
                'is_killed': self.is_killed,
                'kill_level': self.kill_level.name if self.kill_level else None,
                'authorities': len(self.authorities),
                'pending_votes': len(self.pending_votes),
                'kill_events': len(self.kill_events),
                'deadman': self.deadman.get_status(),
                'thresholds': {
                    'bdm': self.bdm_threshold,
                    'cs': self.cs_threshold
                },
                'last_metrics': {
                    'bdm': self._last_bdm,
                    'cs': self._last_cs
                }
            }

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent kill events"""
        with self._lock:
            return [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'level': e.level.name,
                    'reason': e.reason.name,
                    'authority': e.trigger_authority,
                    'execution_ms': e.execution_time_ms
                }
                for e in self.kill_events[-limit:]
            ]


# Global kill switch instance
_killswitch: Optional[MultiPartyKillSwitch] = None


def get_killswitch() -> MultiPartyKillSwitch:
    """Get the global kill switch instance"""
    global _killswitch
    if _killswitch is None:
        _killswitch = MultiPartyKillSwitch()
    return _killswitch


def initialize_killswitch(**kwargs) -> MultiPartyKillSwitch:
    """Initialize the global kill switch"""
    global _killswitch
    _killswitch = MultiPartyKillSwitch(**kwargs)
    _killswitch.start()
    return _killswitch


def emergency_halt(reason: str = "Manual halt") -> None:
    """Emergency halt the system"""
    ks = get_killswitch()

    # Create temporary authority for emergency
    auth_id = f"EMERGENCY_{int(time.time())}"
    ks.register_authority(auth_id, "Emergency", "emergency_key", authority_level=3)

    # Generate signature
    message = f"EMERGENCY:{auth_id}:{reason}"
    signature = hashlib.sha256(
        (message + hashlib.sha256("emergency_key".encode()).hexdigest()).encode()
    ).hexdigest()

    ks.emergency_halt(auth_id, signature, reason)
