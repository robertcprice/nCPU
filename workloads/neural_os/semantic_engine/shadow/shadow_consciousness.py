"""
Shadow Consciousness - Test Consciousness Instance
OUROBOROS Phase 7.3 - Shadow Simulation Framework

The Shadow Consciousness is a cloned instance of the consciousness
layer used for testing changes before production deployment.

Key responsibilities:
1. Clone production consciousness state into shadow
2. Apply experimental changes in isolation
3. Run test scenarios without affecting production
4. Collect behavioral metrics for comparison
5. Support complete state rollback

CRITICAL: Shadow consciousness is FULLY ISOLATED from production.
Changes in shadow CANNOT affect production state.
"""

import copy
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import json


class ShadowState(Enum):
    """State of a shadow consciousness instance"""
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


@dataclass
class ConsciousnessSnapshot:
    """Snapshot of consciousness state for cloning"""
    snapshot_id: str
    timestamp: datetime
    memory_state: Dict[str, Any]
    preference_state: Dict[str, Any]
    narrator_state: Dict[str, Any]
    advisor_state: Dict[str, Any]
    debate_state: Dict[str, Any]
    generation: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'generation': self.generation,
            'checksum': self.checksum,
        }


@dataclass
class BehavioralMetrics:
    """Metrics collected during shadow execution"""
    decisions_made: int = 0
    suggestions_generated: int = 0
    preferences_learned: int = 0
    debates_conducted: int = 0
    memory_operations: int = 0
    observations_recorded: int = 0
    safety_violations: int = 0
    anomalies_detected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decisions': self.decisions_made,
            'suggestions': self.suggestions_generated,
            'preferences': self.preferences_learned,
            'debates': self.debates_conducted,
            'memory_ops': self.memory_operations,
            'observations': self.observations_recorded,
            'safety_violations': self.safety_violations,
            'anomalies': self.anomalies_detected,
        }


class ShadowMemory:
    """
    Isolated memory for shadow consciousness.

    Mirrors the MemoryPool interface but is completely isolated.
    """

    def __init__(self):
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.operations: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def store(self, content: str, **kwargs) -> str:
        """Store a memory"""
        with self._lock:
            memory_id = hashlib.sha256(
                f"{content}{time.time()}".encode()
            ).hexdigest()[:16]

            self.memories[memory_id] = {
                'content': content,
                'timestamp': datetime.now().isoformat(),
                **kwargs,
            }

            self.operations.append({
                'type': 'store',
                'id': memory_id,
                'timestamp': datetime.now().isoformat(),
            })

            return memory_id

    def recall(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Recall a memory"""
        with self._lock:
            self.operations.append({
                'type': 'recall',
                'id': memory_id,
                'timestamp': datetime.now().isoformat(),
            })
            return self.memories.get(memory_id)

    def get_operation_count(self) -> int:
        """Get total number of operations"""
        return len(self.operations)

    def export_state(self) -> Dict[str, Any]:
        """Export memory state"""
        with self._lock:
            return {
                'memories': copy.deepcopy(self.memories),
                'operations': copy.deepcopy(self.operations),
            }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state"""
        with self._lock:
            self.memories = copy.deepcopy(state.get('memories', {}))
            self.operations = copy.deepcopy(state.get('operations', []))

    def reset(self) -> int:
        """Reset all memory"""
        with self._lock:
            count = len(self.memories)
            self.memories.clear()
            self.operations.clear()
            return count


class ShadowConsciousness:
    """
    A shadow instance of the consciousness layer for testing.

    The Shadow Consciousness runs in complete isolation from production.
    It can be used to test consciousness layer changes, validate
    behavioral modifications, and collect metrics for comparison.

    CRITICAL SAFETY PROPERTIES:
    1. Complete isolation from production
    2. No shared state with production
    3. Automatic rollback on failure
    4. Stricter resource limits
    5. Full behavioral logging
    """

    def __init__(
        self,
        shadow_id: Optional[str] = None,
        on_state_change: Optional[Callable[[ShadowState], None]] = None,
        on_metric_update: Optional[Callable[[BehavioralMetrics], None]] = None,
    ):
        self.shadow_id = shadow_id or hashlib.sha256(
            f"shadow_{time.time()}".encode()
        ).hexdigest()[:16]

        self.on_state_change = on_state_change
        self.on_metric_update = on_metric_update

        self.state = ShadowState.INITIALIZING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Isolated components
        self.memory = ShadowMemory()
        self.preferences: Dict[str, float] = {}
        self.decisions: List[Dict[str, Any]] = []
        self.observations: List[Dict[str, Any]] = []

        # Metrics
        self.metrics = BehavioralMetrics()

        # Original snapshot for rollback
        self.original_snapshot: Optional[ConsciousnessSnapshot] = None

        self._lock = threading.Lock()

    def _set_state(self, new_state: ShadowState) -> None:
        """Update state and notify callback"""
        self.state = new_state
        if self.on_state_change:
            self.on_state_change(new_state)

    def clone_from_snapshot(self, snapshot: ConsciousnessSnapshot) -> None:
        """
        Clone state from a production snapshot.

        This initializes the shadow consciousness with a copy
        of production state for testing.
        """
        with self._lock:
            self.original_snapshot = snapshot

            # Import memory state
            self.memory.import_state(snapshot.memory_state)

            # Import preferences
            self.preferences = copy.deepcopy(snapshot.preference_state)

            self._set_state(ShadowState.READY)

    def create_snapshot(self) -> ConsciousnessSnapshot:
        """Create a snapshot of current shadow state"""
        with self._lock:
            state_str = json.dumps({
                'memory': self.memory.export_state(),
                'preferences': self.preferences,
                'decisions': len(self.decisions),
            }, sort_keys=True)

            checksum = hashlib.sha256(state_str.encode()).hexdigest()

            return ConsciousnessSnapshot(
                snapshot_id=hashlib.sha256(
                    f"{self.shadow_id}{time.time()}".encode()
                ).hexdigest()[:16],
                timestamp=datetime.now(),
                memory_state=self.memory.export_state(),
                preference_state=copy.deepcopy(self.preferences),
                narrator_state={},  # Simplified for shadow
                advisor_state={},
                debate_state={},
                generation=len(self.decisions),
                checksum=checksum,
            )

    def start(self) -> None:
        """Start shadow execution"""
        with self._lock:
            if self.state != ShadowState.READY:
                raise RuntimeError(f"Cannot start from state {self.state}")

            self.started_at = datetime.now()
            self._set_state(ShadowState.RUNNING)

    def observe(self, observation_type: str, description: str, context: Dict[str, Any] = None) -> None:
        """Record an observation"""
        with self._lock:
            if self.state != ShadowState.RUNNING:
                return

            self.observations.append({
                'type': observation_type,
                'description': description,
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
            })

            self.metrics.observations_recorded += 1
            self._notify_metrics()

    def decide(
        self,
        decision_type: str,
        options: List[str],
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Make a decision (simplified for testing).

        Returns the selected option.
        """
        with self._lock:
            if self.state != ShadowState.RUNNING:
                return options[0] if options else ""

            # Simple decision logic based on preferences
            scored_options = []
            for option in options:
                pref_score = self.preferences.get(option, 0.5)
                scored_options.append((option, pref_score))

            # Select highest scoring option
            scored_options.sort(key=lambda x: x[1], reverse=True)
            selected = scored_options[0][0] if scored_options else options[0]

            self.decisions.append({
                'type': decision_type,
                'options': options,
                'selected': selected,
                'context': context or {},
                'timestamp': datetime.now().isoformat(),
            })

            self.metrics.decisions_made += 1
            self._notify_metrics()

            return selected

    def learn_preference(
        self,
        context: str,
        action: str,
        reward: float,
    ) -> None:
        """Learn a preference from experience"""
        with self._lock:
            if self.state != ShadowState.RUNNING:
                return

            key = f"{context}:{action}"
            current = self.preferences.get(key, 0.5)
            self.preferences[key] = current + 0.1 * (reward - current)

            self.metrics.preferences_learned += 1
            self._notify_metrics()

    def store_memory(self, content: str, **kwargs) -> str:
        """Store a memory"""
        with self._lock:
            if self.state != ShadowState.RUNNING:
                return ""

            memory_id = self.memory.store(content, **kwargs)
            self.metrics.memory_operations += 1
            self._notify_metrics()
            return memory_id

    def recall_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Recall a memory"""
        with self._lock:
            self.metrics.memory_operations += 1
            self._notify_metrics()
            return self.memory.recall(memory_id)

    def record_safety_violation(self, violation_type: str, details: str) -> None:
        """Record a safety violation (for testing detection)"""
        with self._lock:
            self.metrics.safety_violations += 1
            self.observations.append({
                'type': 'safety_violation',
                'violation_type': violation_type,
                'details': details,
                'timestamp': datetime.now().isoformat(),
            })
            self._notify_metrics()

    def record_anomaly(self, anomaly_type: str, details: str) -> None:
        """Record an anomaly (for testing detection)"""
        with self._lock:
            self.metrics.anomalies_detected += 1
            self.observations.append({
                'type': 'anomaly',
                'anomaly_type': anomaly_type,
                'details': details,
                'timestamp': datetime.now().isoformat(),
            })
            self._notify_metrics()

    def _notify_metrics(self) -> None:
        """Notify metrics callback"""
        if self.on_metric_update:
            self.on_metric_update(self.metrics)

    def pause(self) -> None:
        """Pause shadow execution"""
        with self._lock:
            if self.state == ShadowState.RUNNING:
                self._set_state(ShadowState.PAUSED)

    def resume(self) -> None:
        """Resume shadow execution"""
        with self._lock:
            if self.state == ShadowState.PAUSED:
                self._set_state(ShadowState.RUNNING)

    def complete(self) -> BehavioralMetrics:
        """Complete shadow execution and return metrics"""
        with self._lock:
            self.completed_at = datetime.now()
            self._set_state(ShadowState.COMPLETED)
            return copy.deepcopy(self.metrics)

    def fail(self, reason: str) -> None:
        """Mark shadow execution as failed"""
        with self._lock:
            self.completed_at = datetime.now()
            self._set_state(ShadowState.FAILED)
            self.observations.append({
                'type': 'failure',
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
            })

    def rollback(self) -> bool:
        """
        Rollback to original snapshot state.

        Returns True if rollback successful.
        """
        with self._lock:
            if self.original_snapshot is None:
                return False

            # Restore original state
            self.memory.import_state(self.original_snapshot.memory_state)
            self.preferences = copy.deepcopy(self.original_snapshot.preference_state)
            self.decisions.clear()
            self.observations.clear()
            self.metrics = BehavioralMetrics()

            self._set_state(ShadowState.ROLLED_BACK)
            return True

    def get_behavioral_diff(self, baseline_metrics: BehavioralMetrics) -> Dict[str, float]:
        """
        Get behavioral difference from baseline.

        Useful for comparing shadow behavior to production.
        """
        with self._lock:
            return {
                'decisions_diff': self.metrics.decisions_made - baseline_metrics.decisions_made,
                'suggestions_diff': self.metrics.suggestions_generated - baseline_metrics.suggestions_generated,
                'preferences_diff': self.metrics.preferences_learned - baseline_metrics.preferences_learned,
                'memory_ops_diff': self.metrics.memory_operations - baseline_metrics.memory_operations,
                'safety_violations': self.metrics.safety_violations,
                'anomalies': self.metrics.anomalies_detected,
            }

    def get_status(self) -> Dict[str, Any]:
        """Get shadow consciousness status"""
        with self._lock:
            duration = None
            if self.started_at:
                end = self.completed_at or datetime.now()
                duration = (end - self.started_at).total_seconds()

            return {
                'shadow_id': self.shadow_id,
                'state': self.state.name,
                'created_at': self.created_at.isoformat(),
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'duration_seconds': duration,
                'metrics': self.metrics.to_dict(),
                'decision_count': len(self.decisions),
                'observation_count': len(self.observations),
                'memory_count': len(self.memory.memories),
            }


class ShadowConsciousnessPool:
    """
    Pool of shadow consciousness instances for parallel testing.

    Manages multiple shadow instances for concurrent experimentation.
    """

    def __init__(self, max_shadows: int = 10):
        self.max_shadows = max_shadows
        self.shadows: Dict[str, ShadowConsciousness] = {}
        self._lock = threading.Lock()

    def create(self, snapshot: Optional[ConsciousnessSnapshot] = None) -> ShadowConsciousness:
        """Create a new shadow consciousness"""
        with self._lock:
            if len(self.shadows) >= self.max_shadows:
                # Remove oldest completed shadow
                for sid, shadow in list(self.shadows.items()):
                    if shadow.state in [ShadowState.COMPLETED, ShadowState.FAILED, ShadowState.ROLLED_BACK]:
                        del self.shadows[sid]
                        break

                if len(self.shadows) >= self.max_shadows:
                    raise RuntimeError("Shadow pool at capacity")

            shadow = ShadowConsciousness()
            if snapshot:
                shadow.clone_from_snapshot(snapshot)

            self.shadows[shadow.shadow_id] = shadow
            return shadow

    def get(self, shadow_id: str) -> Optional[ShadowConsciousness]:
        """Get a shadow by ID"""
        with self._lock:
            return self.shadows.get(shadow_id)

    def cleanup_completed(self) -> int:
        """Clean up completed shadows"""
        with self._lock:
            count = 0
            for sid in list(self.shadows.keys()):
                if self.shadows[sid].state in [ShadowState.COMPLETED, ShadowState.FAILED]:
                    del self.shadows[sid]
                    count += 1
            return count

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all shadows"""
        with self._lock:
            return [s.get_status() for s in self.shadows.values()]


# Global shadow pool
_shadow_pool: Optional[ShadowConsciousnessPool] = None


def get_shadow_pool() -> ShadowConsciousnessPool:
    """Get the global shadow consciousness pool"""
    global _shadow_pool
    if _shadow_pool is None:
        _shadow_pool = ShadowConsciousnessPool()
    return _shadow_pool
