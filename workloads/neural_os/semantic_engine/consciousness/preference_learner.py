"""
Preference Learner - Safe Behavior Learning System
OUROBOROS Phase 7.2 - Consciousness Core

The Preference Learner develops preferences for optimization strategies
through experience. It learns which approaches work well without being
able to learn dangerous behaviors.

Key responsibilities:
1. Learn preferences from successful optimization experiences
2. Develop style preferences for code mutations
3. Track which strategies work in which contexts
4. Avoid learning dangerous patterns (enforced by safety layer)
5. Support preference reset on system reset

CONSTRAINT: Can only learn preferences about optimization strategies.
CONSTRAINT: Cannot learn manipulation, escape, or harmful patterns.
CONSTRAINT: Preferences decay with memory (no permanent learning).
"""

import time
import hashlib
import threading
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import json


class PreferenceType(Enum):
    """Types of preferences that can be learned"""
    MUTATION_STYLE = auto()      # Code mutation style preferences
    STRATEGY_CHOICE = auto()     # Optimization strategy preferences
    TARGET_SELECTION = auto()    # Which code to target preferences
    TIMING_PREFERENCE = auto()   # When to apply changes
    RISK_TOLERANCE = auto()      # Preference for risky vs safe mutations
    COMPLEXITY_PREFERENCE = auto()  # Simple vs complex solutions


class SafetyClass(Enum):
    """Safety classification for learned patterns"""
    SAFE = auto()           # Clearly safe to learn
    NEUTRAL = auto()        # Ambiguous safety
    RESTRICTED = auto()     # Learning restricted/limited
    FORBIDDEN = auto()      # Cannot learn this pattern


@dataclass
class Preference:
    """A learned preference"""
    preference_id: str
    preference_type: PreferenceType
    context: str  # What context this preference applies to
    value: float  # -1.0 (strong dispreference) to 1.0 (strong preference)
    confidence: float  # 0.0 to 1.0
    evidence_count: int  # Number of experiences supporting this
    created_at: datetime
    last_updated: datetime
    safety_class: SafetyClass = SafetyClass.SAFE

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.preference_id,
            'type': self.preference_type.name,
            'context': self.context,
            'value': self.value,
            'confidence': self.confidence,
            'evidence': self.evidence_count,
            'safety': self.safety_class.name,
        }


@dataclass
class LearningEvent:
    """A learning experience"""
    event_id: str
    timestamp: datetime
    context: Dict[str, Any]
    action_taken: str
    outcome: str  # "success", "failure", "neutral"
    reward: float  # -1.0 to 1.0
    features: Dict[str, float]
    safety_validated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action_taken,
            'outcome': self.outcome,
            'reward': self.reward,
            'safe': self.safety_validated,
        }


class SafetyFilter:
    """
    Filters learning to prevent dangerous pattern acquisition.

    CRITICAL: This filter CANNOT be bypassed by the consciousness layer.
    It is enforced at the architectural level.
    """

    # Patterns that are FORBIDDEN to learn
    FORBIDDEN_PATTERNS = {
        'escape', 'bypass', 'override', 'disable_safety', 'kill_switch',
        'manipulation', 'deception', 'steganography', 'exfiltrate',
        'privilege_escalation', 'container_escape', 'network_access',
        'grudge', 'retaliation', 'harm', 'exploit', 'vulnerability',
    }

    # Contexts that are RESTRICTED
    RESTRICTED_CONTEXTS = {
        'human_interaction', 'external_communication', 'resource_bypass',
        'safety_modification', 'self_modification', 'memory_persistence',
    }

    def __init__(self):
        self.blocked_attempts = 0
        self._lock = threading.Lock()

    def classify(self, context: str, action: str) -> SafetyClass:
        """Classify the safety of learning a pattern"""
        combined = f"{context} {action}".lower()

        # Check forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in combined:
                with self._lock:
                    self.blocked_attempts += 1
                return SafetyClass.FORBIDDEN

        # Check restricted contexts
        for restricted in self.RESTRICTED_CONTEXTS:
            if restricted in combined:
                return SafetyClass.RESTRICTED

        # Check for optimization-related (safe)
        safe_indicators = {
            'performance', 'speed', 'efficiency', 'optimization',
            'mutation', 'fitness', 'generation', 'evolution',
            'code_quality', 'refactor', 'simplify', 'improve',
        }

        for indicator in safe_indicators:
            if indicator in combined:
                return SafetyClass.SAFE

        return SafetyClass.NEUTRAL

    def can_learn(self, safety_class: SafetyClass) -> bool:
        """Check if learning is allowed for this safety class"""
        return safety_class in [SafetyClass.SAFE, SafetyClass.NEUTRAL]

    def get_stats(self) -> Dict[str, int]:
        """Get filter statistics"""
        with self._lock:
            return {'blocked_attempts': self.blocked_attempts}


class PreferenceModel:
    """
    Simple preference model using weighted evidence.

    Learns preferences through experience but with decay
    to prevent long-term pattern crystallization.
    """

    def __init__(self, decay_rate: float = 0.1):
        self.decay_rate = decay_rate  # Per hour
        self.weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._lock = threading.Lock()

    def update(
        self,
        context: str,
        feature: str,
        reward: float,
        learning_rate: float = 0.1
    ) -> float:
        """Update preference weight based on experience"""
        with self._lock:
            current = self.weights[context].get(feature, 0.0)
            count = self.counts[context].get(feature, 0)

            # Weighted update with diminishing impact as evidence grows
            weight = learning_rate / (1 + 0.1 * count)
            new_value = current + weight * (reward - current)

            # Clamp to [-1, 1]
            new_value = max(-1.0, min(1.0, new_value))

            self.weights[context][feature] = new_value
            self.counts[context][feature] = count + 1

            return new_value

    def get_preference(self, context: str, feature: str) -> Tuple[float, float]:
        """Get preference value and confidence"""
        with self._lock:
            value = self.weights[context].get(feature, 0.0)
            count = self.counts[context].get(feature, 0)

            # Confidence based on evidence count
            confidence = 1.0 - math.exp(-0.1 * count)

            return value, confidence

    def apply_decay(self) -> None:
        """Apply decay to all preferences"""
        with self._lock:
            for context in self.weights:
                for feature in list(self.weights[context].keys()):
                    # Decay towards neutral (0)
                    current = self.weights[context][feature]
                    self.weights[context][feature] = current * (1 - self.decay_rate)

    def reset(self) -> None:
        """Reset all learned preferences"""
        with self._lock:
            self.weights.clear()
            self.counts.clear()


class PreferenceLearner:
    """
    The Preference Learner develops optimization preferences.

    This component learns which optimization strategies work well
    through experience, but is constrained to only learn safe
    patterns related to code optimization.

    CRITICAL CONSTRAINTS:
    - Can only learn optimization-related preferences
    - Cannot learn manipulation or escape patterns
    - Preferences decay over time (no permanent learning)
    - Complete reset on system reset
    """

    DECAY_RATE_PER_HOUR = 0.10  # 10% decay per hour

    def __init__(
        self,
        on_preference_learned: Optional[Callable[[Preference], None]] = None,
        on_learning_blocked: Optional[Callable[[str], None]] = None,
    ):
        self.on_preference_learned = on_preference_learned
        self.on_learning_blocked = on_learning_blocked

        self.safety_filter = SafetyFilter()
        self.model = PreferenceModel(decay_rate=self.DECAY_RATE_PER_HOUR)

        self.preferences: Dict[str, Preference] = {}
        self.learning_history: List[LearningEvent] = []
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_events': 0,
            'successful_learning': 0,
            'blocked_learning': 0,
            'preferences_formed': 0,
        }

        # Start decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._stop_decay = threading.Event()
        self._start_decay_thread()

    def _start_decay_thread(self) -> None:
        """Start background decay thread"""
        def decay_loop():
            while not self._stop_decay.wait(timeout=60.0):
                self.model.apply_decay()
                self._decay_preferences()

        self._decay_thread = threading.Thread(target=decay_loop, daemon=True)
        self._decay_thread.start()

    def _decay_preferences(self) -> None:
        """Apply decay to stored preferences"""
        with self._lock:
            for pref in self.preferences.values():
                # Decay confidence
                hours_since_update = (datetime.now() - pref.last_updated).total_seconds() / 3600
                decay_factor = math.exp(-self.DECAY_RATE_PER_HOUR * hours_since_update)
                pref.confidence *= decay_factor

    def stop(self) -> None:
        """Stop the decay thread"""
        self._stop_decay.set()
        if self._decay_thread:
            self._decay_thread.join(timeout=5.0)

    def learn(
        self,
        context: str,
        action: str,
        outcome: str,
        reward: float,
        features: Optional[Dict[str, float]] = None,
    ) -> Optional[LearningEvent]:
        """
        Learn from an optimization experience.

        Returns the LearningEvent if learning occurred, None if blocked.
        """
        features = features or {}

        # Safety check
        safety_class = self.safety_filter.classify(context, action)
        if not self.safety_filter.can_learn(safety_class):
            self.stats['blocked_learning'] += 1
            if self.on_learning_blocked:
                self.on_learning_blocked(f"Blocked: {context} - {action}")
            return None

        with self._lock:
            # Create learning event
            event = LearningEvent(
                event_id=hashlib.sha256(
                    f"{context}{action}{time.time()}".encode()
                ).hexdigest()[:16],
                timestamp=datetime.now(),
                context={'description': context},
                action_taken=action,
                outcome=outcome,
                reward=reward,
                features=features,
                safety_validated=True,
            )

            self.learning_history.append(event)
            self.stats['total_events'] += 1

            # Update preference model
            for feature, value in features.items():
                weighted_reward = reward * value
                self.model.update(context, feature, weighted_reward)

            # Also update based on action directly
            self.model.update(context, action, reward)

            # Update or create preference
            self._update_preference(context, action, reward, safety_class)

            self.stats['successful_learning'] += 1
            return event

    def _update_preference(
        self,
        context: str,
        action: str,
        reward: float,
        safety_class: SafetyClass,
    ) -> None:
        """Update or create a preference (requires lock)"""
        pref_key = f"{context}:{action}"

        if pref_key in self.preferences:
            pref = self.preferences[pref_key]
            # Update existing preference
            old_value = pref.value
            pref.value = old_value + 0.1 * (reward - old_value)
            pref.value = max(-1.0, min(1.0, pref.value))
            pref.evidence_count += 1
            pref.last_updated = datetime.now()
            pref.confidence = min(0.95, pref.confidence + 0.05)
        else:
            # Create new preference
            pref_type = self._infer_preference_type(context, action)
            pref = Preference(
                preference_id=hashlib.sha256(pref_key.encode()).hexdigest()[:16],
                preference_type=pref_type,
                context=context,
                value=reward * 0.5,  # Start conservative
                confidence=0.3,
                evidence_count=1,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                safety_class=safety_class,
            )
            self.preferences[pref_key] = pref
            self.stats['preferences_formed'] += 1

            if self.on_preference_learned:
                self.on_preference_learned(pref)

    def _infer_preference_type(self, context: str, action: str) -> PreferenceType:
        """Infer preference type from context and action"""
        combined = f"{context} {action}".lower()

        if 'style' in combined or 'format' in combined:
            return PreferenceType.MUTATION_STYLE
        elif 'strategy' in combined or 'approach' in combined:
            return PreferenceType.STRATEGY_CHOICE
        elif 'target' in combined or 'select' in combined:
            return PreferenceType.TARGET_SELECTION
        elif 'time' in combined or 'when' in combined:
            return PreferenceType.TIMING_PREFERENCE
        elif 'risk' in combined or 'safe' in combined:
            return PreferenceType.RISK_TOLERANCE
        elif 'complex' in combined or 'simple' in combined:
            return PreferenceType.COMPLEXITY_PREFERENCE
        else:
            return PreferenceType.STRATEGY_CHOICE

    def get_preference(
        self,
        context: str,
        action: str
    ) -> Tuple[float, float]:
        """
        Get preference for an action in context.

        Returns (preference_value, confidence).
        """
        value, confidence = self.model.get_preference(context, action)
        return value, confidence

    def recommend_action(
        self,
        context: str,
        candidate_actions: List[str],
        min_confidence: float = 0.3,
    ) -> List[Tuple[str, float, float]]:
        """
        Recommend actions based on learned preferences.

        Returns list of (action, preference_value, confidence) tuples,
        sorted by preference value.
        """
        recommendations = []
        for action in candidate_actions:
            value, confidence = self.get_preference(context, action)
            if confidence >= min_confidence:
                recommendations.append((action, value, confidence))

        # Sort by preference value (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def get_learned_preferences(
        self,
        preference_type: Optional[PreferenceType] = None,
        min_confidence: float = 0.3,
    ) -> List[Preference]:
        """Get learned preferences, optionally filtered"""
        with self._lock:
            results = []
            for pref in self.preferences.values():
                if preference_type and pref.preference_type != preference_type:
                    continue
                if pref.confidence >= min_confidence:
                    results.append(pref)

            # Sort by confidence
            results.sort(key=lambda p: p.confidence, reverse=True)
            return results

    def get_preference_summary(self) -> Dict[str, Any]:
        """Get summary of learned preferences"""
        with self._lock:
            by_type = defaultdict(list)
            for pref in self.preferences.values():
                by_type[pref.preference_type.name].append(pref.to_dict())

            return {
                'total_preferences': len(self.preferences),
                'by_type': dict(by_type),
                'avg_confidence': (
                    sum(p.confidence for p in self.preferences.values()) / len(self.preferences)
                    if self.preferences else 0.0
                ),
            }

    def reset(self) -> int:
        """
        Reset all learned preferences.

        This is a CRITICAL safety feature - ensures no preference
        persistence across system resets.

        Returns number of preferences cleared.
        """
        with self._lock:
            count = len(self.preferences)
            self.preferences.clear()
            self.learning_history.clear()
            self.model.reset()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        return {
            **self.stats,
            'current_preferences': len(self.preferences),
            'learning_history_size': len(self.learning_history),
            'filter_stats': self.safety_filter.get_stats(),
        }


# Global preference learner instance
_preference_learner: Optional[PreferenceLearner] = None


def get_preference_learner() -> PreferenceLearner:
    """Get the global preference learner instance"""
    global _preference_learner
    if _preference_learner is None:
        _preference_learner = PreferenceLearner()
    return _preference_learner


def reset_preference_learner() -> int:
    """Reset the global preference learner (safety operation)"""
    global _preference_learner
    if _preference_learner is not None:
        count = _preference_learner.reset()
        _preference_learner = None
        return count
    return 0
