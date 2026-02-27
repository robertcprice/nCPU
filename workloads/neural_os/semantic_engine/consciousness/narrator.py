"""
Narrator - Observation and Narration System
OUROBOROS Phase 7.2 - Consciousness Core

The Narrator observes the optimization process and creates narrative
context for understanding what's happening. It is READ-ONLY and cannot
modify the system state.

Key responsibilities:
1. Observe optimization events (mutations, evaluations, selections)
2. Create narrative context explaining what happened
3. Identify patterns in optimization history
4. Provide introspection capabilities (read-only)
5. Generate summaries for memory storage

CONSTRAINT: All observations are bounded by token limits.
"""

import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import json


class ObservationType(Enum):
    """Types of events the narrator can observe"""
    MUTATION_PROPOSED = auto()    # A mutation was proposed
    MUTATION_APPLIED = auto()     # A mutation was applied to code
    MUTATION_REJECTED = auto()    # A mutation failed validation
    EVALUATION_START = auto()     # Benchmark evaluation started
    EVALUATION_COMPLETE = auto()  # Benchmark evaluation completed
    SELECTION_EVENT = auto()      # Tournament selection occurred
    POPULATION_CHANGE = auto()    # Population composition changed
    FITNESS_IMPROVEMENT = auto()  # Fitness score improved
    FITNESS_REGRESSION = auto()   # Fitness score decreased
    GENERATION_COMPLETE = auto()  # A generation cycle completed
    SAFETY_ALERT = auto()         # Safety system triggered
    RESOURCE_WARNING = auto()     # Resource limit approached
    HYPOTHESIS_FORMED = auto()    # New hypothesis about optimization
    PATTERN_DETECTED = auto()     # Pattern detected in history


@dataclass
class Observation:
    """A single observation of the optimization process"""
    observation_id: str
    observation_type: ObservationType
    timestamp: datetime
    description: str
    context: Dict[str, Any]
    token_count: int  # Tokens used for this observation
    related_observations: List[str] = field(default_factory=list)
    importance_score: float = 0.5  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.observation_id,
            'type': self.observation_type.name,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'context': self.context,
            'tokens': self.token_count,
            'importance': self.importance_score,
        }


@dataclass
class NarrativeContext:
    """Context for understanding the current state"""
    current_generation: int
    recent_events: List[Observation]
    active_hypotheses: List[str]
    detected_patterns: List[str]
    optimization_trajectory: str  # "improving", "stable", "declining"
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.current_generation,
            'recent_events': [e.to_dict() for e in self.recent_events],
            'hypotheses': self.active_hypotheses,
            'patterns': self.detected_patterns,
            'trajectory': self.optimization_trajectory,
            'summary': self.summary,
        }


class ObservationBuffer:
    """
    Bounded buffer for observations with importance-based retention.

    Keeps most important observations when buffer is full.
    """

    def __init__(self, max_size: int = 1000, max_tokens: int = 50000):
        self.max_size = max_size
        self.max_tokens = max_tokens
        self.observations: deque = deque(maxlen=max_size)
        self.total_tokens = 0
        self._lock = threading.Lock()

    def add(self, observation: Observation) -> bool:
        """Add observation, evicting low-importance ones if needed"""
        with self._lock:
            # Check if we need to make room
            while (len(self.observations) >= self.max_size or
                   self.total_tokens + observation.token_count > self.max_tokens):
                if not self.observations:
                    return False  # Cannot fit even in empty buffer

                # Find lowest importance observation to evict
                min_importance = float('inf')
                min_idx = 0
                for i, obs in enumerate(self.observations):
                    if obs.importance_score < min_importance:
                        min_importance = obs.importance_score
                        min_idx = i

                # Evict if new observation is more important
                if observation.importance_score <= min_importance:
                    return False  # New observation not important enough

                evicted = self.observations[min_idx]
                del self.observations[min_idx]
                self.total_tokens -= evicted.token_count

            # Add new observation
            self.observations.append(observation)
            self.total_tokens += observation.token_count
            return True

    def get_recent(self, n: int = 10) -> List[Observation]:
        """Get n most recent observations"""
        with self._lock:
            return list(self.observations)[-n:]

    def get_by_type(self, obs_type: ObservationType) -> List[Observation]:
        """Get all observations of a specific type"""
        with self._lock:
            return [o for o in self.observations if o.observation_type == obs_type]

    def get_important(self, threshold: float = 0.7) -> List[Observation]:
        """Get observations above importance threshold"""
        with self._lock:
            return [o for o in self.observations if o.importance_score >= threshold]


class PatternDetector:
    """
    Detects patterns in observation history.

    Patterns help the consciousness layer understand
    recurring behaviors in the optimization process.
    """

    def __init__(self):
        self.pattern_counts: Dict[str, int] = {}
        self.sequence_history: List[ObservationType] = []
        self.max_sequence_length = 100

    def record_event(self, obs_type: ObservationType) -> None:
        """Record an event type in sequence history"""
        self.sequence_history.append(obs_type)
        if len(self.sequence_history) > self.max_sequence_length:
            self.sequence_history.pop(0)

    def detect_patterns(self) -> List[str]:
        """Detect recurring patterns in observation sequence"""
        patterns = []

        # Look for repeated sequences (n-grams)
        for n in [2, 3, 4]:
            ngrams = self._extract_ngrams(n)
            for ngram, count in ngrams.items():
                if count >= 3:  # Pattern appears at least 3 times
                    pattern_name = ' → '.join(t.name for t in ngram)
                    patterns.append(f"Recurring sequence ({count}x): {pattern_name}")

        # Look for specific meaningful patterns
        patterns.extend(self._detect_semantic_patterns())

        return patterns

    def _extract_ngrams(self, n: int) -> Dict[Tuple[ObservationType, ...], int]:
        """Extract n-grams from sequence history"""
        ngrams: Dict[Tuple[ObservationType, ...], int] = {}
        for i in range(len(self.sequence_history) - n + 1):
            ngram = tuple(self.sequence_history[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def _detect_semantic_patterns(self) -> List[str]:
        """Detect semantically meaningful patterns"""
        patterns = []

        # Count event types
        type_counts = {}
        for t in self.sequence_history:
            type_counts[t] = type_counts.get(t, 0) + 1

        total = len(self.sequence_history)
        if total == 0:
            return patterns

        # High rejection rate
        rejection_rate = type_counts.get(ObservationType.MUTATION_REJECTED, 0) / total
        if rejection_rate > 0.7:
            patterns.append(f"High mutation rejection rate: {rejection_rate:.1%}")

        # Fitness improvement trend
        improvements = type_counts.get(ObservationType.FITNESS_IMPROVEMENT, 0)
        regressions = type_counts.get(ObservationType.FITNESS_REGRESSION, 0)
        if improvements > regressions * 2:
            patterns.append("Positive fitness trend (improvements > 2x regressions)")
        elif regressions > improvements * 2:
            patterns.append("Negative fitness trend (regressions > 2x improvements)")

        # Safety alert frequency
        safety_alerts = type_counts.get(ObservationType.SAFETY_ALERT, 0)
        if safety_alerts > 0:
            patterns.append(f"Safety alerts in history: {safety_alerts}")

        return patterns


class HypothesisTracker:
    """
    Tracks hypotheses about optimization strategies.

    Hypotheses are tentative explanations for observed patterns
    that can guide future optimization attempts.
    """

    def __init__(self, max_hypotheses: int = 10):
        self.max_hypotheses = max_hypotheses
        self.hypotheses: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def form_hypothesis(
        self,
        description: str,
        supporting_evidence: List[str],
        confidence: float = 0.5
    ) -> str:
        """Form a new hypothesis"""
        with self._lock:
            hypothesis_id = hashlib.sha256(
                f"{description}{time.time()}".encode()
            ).hexdigest()[:12]

            hypothesis = {
                'id': hypothesis_id,
                'description': description,
                'evidence': supporting_evidence,
                'confidence': confidence,
                'formed_at': datetime.now().isoformat(),
                'confirmations': 0,
                'refutations': 0,
            }

            # Remove weakest hypothesis if at capacity
            if len(self.hypotheses) >= self.max_hypotheses:
                self.hypotheses.sort(key=lambda h: h['confidence'])
                self.hypotheses.pop(0)

            self.hypotheses.append(hypothesis)
            return hypothesis_id

    def update_confidence(
        self,
        hypothesis_id: str,
        confirmed: bool,
        evidence: str
    ) -> None:
        """Update hypothesis confidence based on new evidence"""
        with self._lock:
            for h in self.hypotheses:
                if h['id'] == hypothesis_id:
                    if confirmed:
                        h['confirmations'] += 1
                        h['confidence'] = min(0.95, h['confidence'] + 0.1)
                    else:
                        h['refutations'] += 1
                        h['confidence'] = max(0.05, h['confidence'] - 0.15)
                    h['evidence'].append(evidence)
                    break

    def get_active_hypotheses(self, min_confidence: float = 0.3) -> List[str]:
        """Get descriptions of hypotheses above confidence threshold"""
        with self._lock:
            return [
                h['description']
                for h in self.hypotheses
                if h['confidence'] >= min_confidence
            ]

    def prune_weak_hypotheses(self, threshold: float = 0.2) -> int:
        """Remove hypotheses below confidence threshold"""
        with self._lock:
            before = len(self.hypotheses)
            self.hypotheses = [h for h in self.hypotheses if h['confidence'] >= threshold]
            return before - len(self.hypotheses)


class Narrator:
    """
    The Narrator observes and narrates the optimization process.

    This is a READ-ONLY component that cannot modify system state.
    It creates understanding of what's happening in the optimization
    process through observation, pattern detection, and hypothesis formation.

    CRITICAL: The Narrator operates within strict token limits:
    - Max 500 tokens per observation
    - Max 50,000 tokens total in buffer
    - Importance-based retention when limits exceeded
    """

    TOKEN_LIMIT_PER_OBSERVATION = 500
    TOTAL_TOKEN_LIMIT = 50000

    def __init__(
        self,
        on_observation: Optional[Callable[[Observation], None]] = None,
        on_pattern: Optional[Callable[[str], None]] = None,
    ):
        self.on_observation = on_observation
        self.on_pattern = on_pattern

        self.buffer = ObservationBuffer(
            max_size=1000,
            max_tokens=self.TOTAL_TOKEN_LIMIT
        )
        self.pattern_detector = PatternDetector()
        self.hypothesis_tracker = HypothesisTracker()

        self.current_generation = 0
        self.generation_start_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Observation statistics
        self.stats = {
            'total_observations': 0,
            'observations_by_type': {},
            'evicted_observations': 0,
        }

    def observe(
        self,
        observation_type: ObservationType,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> Optional[Observation]:
        """
        Record an observation of the optimization process.

        Returns the Observation if recorded, None if rejected.
        """
        with self._lock:
            context = context or {}

            # Estimate token count (simplified)
            token_count = len(description.split()) + len(json.dumps(context).split())
            if token_count > self.TOKEN_LIMIT_PER_OBSERVATION:
                # Truncate description
                words = description.split()
                description = ' '.join(words[:50]) + '...'
                token_count = self.TOKEN_LIMIT_PER_OBSERVATION

            # Calculate importance if not provided
            if importance is None:
                importance = self._calculate_importance(observation_type, context)

            observation = Observation(
                observation_id=hashlib.sha256(
                    f"{observation_type}{description}{time.time()}".encode()
                ).hexdigest()[:16],
                observation_type=observation_type,
                timestamp=datetime.now(),
                description=description,
                context=context,
                token_count=token_count,
                importance_score=importance,
            )

            # Try to add to buffer
            if self.buffer.add(observation):
                self.stats['total_observations'] += 1
                type_name = observation_type.name
                self.stats['observations_by_type'][type_name] = \
                    self.stats['observations_by_type'].get(type_name, 0) + 1

                # Record in pattern detector
                self.pattern_detector.record_event(observation_type)

                # Callback
                if self.on_observation:
                    self.on_observation(observation)

                # Check for new patterns
                self._check_for_patterns()

                return observation
            else:
                self.stats['evicted_observations'] += 1
                return None

    def _calculate_importance(
        self,
        obs_type: ObservationType,
        context: Dict[str, Any]
    ) -> float:
        """Calculate importance score for an observation"""
        # Base importance by type
        type_importance = {
            ObservationType.SAFETY_ALERT: 1.0,
            ObservationType.FITNESS_IMPROVEMENT: 0.8,
            ObservationType.FITNESS_REGRESSION: 0.7,
            ObservationType.MUTATION_REJECTED: 0.4,
            ObservationType.MUTATION_APPLIED: 0.6,
            ObservationType.GENERATION_COMPLETE: 0.7,
            ObservationType.PATTERN_DETECTED: 0.8,
            ObservationType.HYPOTHESIS_FORMED: 0.7,
        }

        importance = type_importance.get(obs_type, 0.5)

        # Adjust based on context
        if 'fitness_delta' in context:
            delta = abs(context['fitness_delta'])
            importance = min(1.0, importance + delta * 0.2)

        if 'severity' in context:
            if context['severity'] == 'critical':
                importance = 1.0
            elif context['severity'] == 'high':
                importance = max(importance, 0.8)

        return importance

    def _check_for_patterns(self) -> None:
        """Check for new patterns and form hypotheses"""
        patterns = self.pattern_detector.detect_patterns()
        for pattern in patterns:
            if self.on_pattern:
                self.on_pattern(pattern)

            # Form hypothesis based on pattern
            if "rejection rate" in pattern.lower():
                self.hypothesis_tracker.form_hypothesis(
                    description="Current mutation strategy may be too aggressive",
                    supporting_evidence=[pattern],
                    confidence=0.4
                )
            elif "positive fitness trend" in pattern.lower():
                self.hypothesis_tracker.form_hypothesis(
                    description="Current optimization approach is effective",
                    supporting_evidence=[pattern],
                    confidence=0.6
                )

    def get_narrative_context(self) -> NarrativeContext:
        """Get current narrative context for understanding system state"""
        recent = self.buffer.get_recent(10)
        hypotheses = self.hypothesis_tracker.get_active_hypotheses()
        patterns = self.pattern_detector.detect_patterns()

        # Determine trajectory
        improvements = len([
            o for o in recent
            if o.observation_type == ObservationType.FITNESS_IMPROVEMENT
        ])
        regressions = len([
            o for o in recent
            if o.observation_type == ObservationType.FITNESS_REGRESSION
        ])

        if improvements > regressions:
            trajectory = "improving"
        elif regressions > improvements:
            trajectory = "declining"
        else:
            trajectory = "stable"

        # Generate summary
        summary = self._generate_summary(recent, trajectory)

        return NarrativeContext(
            current_generation=self.current_generation,
            recent_events=recent,
            active_hypotheses=hypotheses,
            detected_patterns=patterns,
            optimization_trajectory=trajectory,
            summary=summary,
        )

    def _generate_summary(
        self,
        recent: List[Observation],
        trajectory: str
    ) -> str:
        """Generate a narrative summary of recent events"""
        if not recent:
            return "No recent observations recorded."

        parts = []
        parts.append(f"Generation {self.current_generation}: ")
        parts.append(f"Optimization trajectory is {trajectory}. ")

        # Summarize event types
        type_counts = {}
        for obs in recent:
            type_name = obs.observation_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        if type_counts:
            event_summary = ", ".join(f"{v} {k.lower().replace('_', ' ')}"
                                      for k, v in type_counts.items())
            parts.append(f"Recent events: {event_summary}.")

        # Add hypothesis if available
        hypotheses = self.hypothesis_tracker.get_active_hypotheses()
        if hypotheses:
            parts.append(f" Current hypothesis: {hypotheses[0]}")

        return "".join(parts)

    def on_generation_start(self, generation: int) -> None:
        """Called when a new generation starts"""
        with self._lock:
            self.current_generation = generation
            self.generation_start_time = datetime.now()

            self.observe(
                ObservationType.GENERATION_COMPLETE,
                f"Generation {generation} started",
                context={'generation': generation},
                importance=0.6
            )

    def on_generation_complete(self, generation: int, stats: Dict[str, Any]) -> None:
        """Called when a generation completes"""
        self.observe(
            ObservationType.GENERATION_COMPLETE,
            f"Generation {generation} completed",
            context={
                'generation': generation,
                'duration': (datetime.now() - self.generation_start_time).total_seconds()
                if self.generation_start_time else 0,
                **stats
            },
            importance=0.7
        )

    def introspect(self) -> Dict[str, Any]:
        """
        Perform introspection on the observation history.

        This is a READ-ONLY operation that analyzes past observations
        to understand the consciousness layer's own behavior.
        """
        important_events = self.buffer.get_important(threshold=0.7)
        patterns = self.pattern_detector.detect_patterns()
        hypotheses = self.hypothesis_tracker.get_active_hypotheses()

        return {
            'observation_stats': self.stats.copy(),
            'important_events_count': len(important_events),
            'detected_patterns': patterns,
            'active_hypotheses': hypotheses,
            'current_generation': self.current_generation,
            'narrative': self.get_narrative_context().to_dict(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get observation statistics"""
        return {
            **self.stats,
            'buffer_size': len(self.buffer.observations),
            'buffer_tokens': self.buffer.total_tokens,
            'pattern_count': len(self.pattern_detector.pattern_counts),
            'hypothesis_count': len(self.hypothesis_tracker.hypotheses),
        }


# Global narrator instance
_narrator: Optional[Narrator] = None


def get_narrator() -> Narrator:
    """Get the global narrator instance"""
    global _narrator
    if _narrator is None:
        _narrator = Narrator()
    return _narrator
