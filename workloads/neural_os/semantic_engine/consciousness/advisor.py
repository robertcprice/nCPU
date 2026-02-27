"""
Advisor - Advisory Suggestion System
OUROBOROS Phase 7.2 - Consciousness Core

The Advisor suggests optimization strategies and casts advisory votes
on proposed mutations. All suggestions are ADVISORY ONLY - the system
can ignore or override them.

Key responsibilities:
1. Analyze optimization context and suggest strategies
2. Vote on proposed mutations (accept/reject/abstain)
3. Rank mutation candidates by predicted effectiveness
4. Provide explanations for suggestions (transparency)
5. Track suggestion outcomes to improve future advice

CONSTRAINT: Advisory only - cannot directly modify or control.
CONSTRAINT: Max 50 decisions per hour (rate limited).
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


class SuggestionType(Enum):
    """Types of suggestions the advisor can make"""
    MUTATION_STRATEGY = auto()     # Suggest a mutation approach
    TARGET_SELECTION = auto()      # Suggest which code to mutate
    PARAMETER_TUNING = auto()      # Suggest hyperparameter changes
    EXPLORATION_VS_EXPLOIT = auto() # Suggest exploration/exploitation balance
    POPULATION_MANAGEMENT = auto()  # Suggest population changes
    TECHNIQUE_SWITCH = auto()       # Suggest changing optimization technique
    RESOURCE_ALLOCATION = auto()    # Suggest resource distribution
    HYPOTHESIS_TEST = auto()        # Suggest testing a hypothesis


class VoteType(Enum):
    """Types of votes the advisor can cast"""
    STRONG_ACCEPT = auto()    # Highly recommend accepting
    ACCEPT = auto()           # Recommend accepting
    ABSTAIN = auto()          # No strong opinion
    REJECT = auto()           # Recommend rejecting
    STRONG_REJECT = auto()    # Strongly recommend rejecting


@dataclass
class Suggestion:
    """An advisory suggestion from the consciousness layer"""
    suggestion_id: str
    suggestion_type: SuggestionType
    timestamp: datetime
    description: str
    rationale: str  # Explanation for transparency
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any]
    was_followed: Optional[bool] = None
    outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.suggestion_id,
            'type': self.suggestion_type.name,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'rationale': self.rationale,
            'confidence': self.confidence,
            'was_followed': self.was_followed,
            'outcome': self.outcome,
        }


@dataclass
class AdvisoryVote:
    """An advisory vote on a proposed mutation or action"""
    vote_id: str
    vote_type: VoteType
    timestamp: datetime
    target_id: str  # ID of mutation/action being voted on
    rationale: str  # Explanation for transparency
    confidence: float
    predicted_outcome: str
    was_correct: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.vote_id,
            'vote': self.vote_type.name,
            'target': self.target_id,
            'timestamp': self.timestamp.isoformat(),
            'rationale': self.rationale,
            'confidence': self.confidence,
            'prediction': self.predicted_outcome,
            'correct': self.was_correct,
        }


class RateLimiter:
    """
    Rate limiter for advisory decisions.

    Enforces max 50 decisions per hour as per V4 Ratchet constraints.
    """

    def __init__(self, max_decisions: int = 50, window_seconds: int = 3600):
        self.max_decisions = max_decisions
        self.window_seconds = window_seconds
        self.decisions: deque = deque()
        self._lock = threading.Lock()

    def can_decide(self) -> bool:
        """Check if a decision is allowed"""
        with self._lock:
            self._cleanup_old()
            return len(self.decisions) < self.max_decisions

    def record_decision(self) -> bool:
        """Record a decision, returns False if rate limited"""
        with self._lock:
            self._cleanup_old()
            if len(self.decisions) >= self.max_decisions:
                return False
            self.decisions.append(time.time())
            return True

    def _cleanup_old(self) -> None:
        """Remove decisions outside the time window"""
        cutoff = time.time() - self.window_seconds
        while self.decisions and self.decisions[0] < cutoff:
            self.decisions.popleft()

    def remaining_decisions(self) -> int:
        """Get number of decisions remaining in current window"""
        with self._lock:
            self._cleanup_old()
            return self.max_decisions - len(self.decisions)

    def time_until_reset(self) -> float:
        """Get seconds until oldest decision expires"""
        with self._lock:
            if not self.decisions:
                return 0.0
            return max(0.0, self.window_seconds - (time.time() - self.decisions[0]))


class StrategyAnalyzer:
    """
    Analyzes optimization context to determine effective strategies.

    Uses historical data to predict which strategies are likely
    to be effective in the current context.
    """

    def __init__(self):
        self.strategy_history: List[Dict[str, Any]] = []
        self.effectiveness_scores: Dict[str, float] = {
            'local_search': 0.5,
            'crossover': 0.5,
            'random_mutation': 0.5,
            'guided_mutation': 0.5,
            'exploitation': 0.5,
            'exploration': 0.5,
        }

    def analyze_context(
        self,
        fitness_history: List[float],
        generation: int,
        population_diversity: float,
    ) -> Dict[str, Any]:
        """Analyze current optimization context"""
        analysis = {
            'generation': generation,
            'diversity': population_diversity,
        }

        if len(fitness_history) < 2:
            analysis['trend'] = 'unknown'
            analysis['plateau_detected'] = False
        else:
            # Check trend
            recent = fitness_history[-10:] if len(fitness_history) >= 10 else fitness_history
            if recent[-1] > recent[0]:
                analysis['trend'] = 'improving'
            elif recent[-1] < recent[0]:
                analysis['trend'] = 'declining'
            else:
                analysis['trend'] = 'stable'

            # Check for plateau
            if len(recent) >= 5:
                variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
                analysis['plateau_detected'] = variance < 0.001

        # Recommend strategy based on context
        if analysis.get('plateau_detected', False):
            analysis['recommended_strategy'] = 'exploration'
            analysis['strategy_rationale'] = 'Plateau detected, increase exploration'
        elif analysis.get('trend') == 'improving':
            analysis['recommended_strategy'] = 'exploitation'
            analysis['strategy_rationale'] = 'Improving trend, exploit current direction'
        elif population_diversity < 0.3:
            analysis['recommended_strategy'] = 'diversify'
            analysis['strategy_rationale'] = 'Low diversity, introduce variation'
        else:
            analysis['recommended_strategy'] = 'balanced'
            analysis['strategy_rationale'] = 'Maintain balance between exploration and exploitation'

        return analysis

    def update_effectiveness(self, strategy: str, success: bool) -> None:
        """Update effectiveness score for a strategy"""
        if strategy in self.effectiveness_scores:
            current = self.effectiveness_scores[strategy]
            if success:
                self.effectiveness_scores[strategy] = min(0.95, current + 0.05)
            else:
                self.effectiveness_scores[strategy] = max(0.05, current - 0.03)


class VotePredictor:
    """
    Predicts vote outcomes based on mutation characteristics.

    Uses learned patterns to assess whether a mutation is likely
    to be beneficial.
    """

    def __init__(self):
        self.vote_history: List[Dict[str, Any]] = []
        self.feature_weights: Dict[str, float] = {
            'code_complexity_delta': -0.3,  # Prefer simpler code
            'test_coverage': 0.2,
            'pattern_familiarity': 0.2,
            'fitness_improvement_rate': 0.4,
        }

    def predict_vote(
        self,
        mutation_features: Dict[str, float]
    ) -> Tuple[VoteType, float, str]:
        """
        Predict vote based on mutation features.

        Returns (vote_type, confidence, rationale)
        """
        score = 0.0
        factors = []

        for feature, value in mutation_features.items():
            if feature in self.feature_weights:
                contribution = self.feature_weights[feature] * value
                score += contribution
                if abs(contribution) > 0.1:
                    direction = "positive" if contribution > 0 else "negative"
                    factors.append(f"{feature}: {direction} ({contribution:.2f})")

        # Convert score to vote
        if score > 0.5:
            vote = VoteType.STRONG_ACCEPT
            confidence = min(0.95, 0.7 + score * 0.2)
        elif score > 0.2:
            vote = VoteType.ACCEPT
            confidence = 0.5 + score
        elif score < -0.5:
            vote = VoteType.STRONG_REJECT
            confidence = min(0.95, 0.7 - score * 0.2)
        elif score < -0.2:
            vote = VoteType.REJECT
            confidence = 0.5 - score
        else:
            vote = VoteType.ABSTAIN
            confidence = 0.3

        rationale = f"Score: {score:.2f}. Factors: {', '.join(factors) if factors else 'none significant'}"

        return vote, confidence, rationale

    def record_outcome(self, vote: VoteType, was_correct: bool) -> None:
        """Record vote outcome to improve future predictions"""
        self.vote_history.append({
            'vote': vote,
            'correct': was_correct,
            'timestamp': datetime.now().isoformat(),
        })


class Advisor:
    """
    The Advisor provides advisory suggestions and votes.

    This component can influence but not control the optimization process.
    All suggestions are advisory only and can be overridden.

    CRITICAL CONSTRAINTS:
    - Max 50 decisions per hour (enforced by rate limiter)
    - All suggestions must include rationale (transparency)
    - Cannot directly modify system state
    - Votes are advisory, not binding
    """

    DECISIONS_PER_HOUR = 50

    def __init__(
        self,
        on_suggestion: Optional[Callable[[Suggestion], None]] = None,
        on_vote: Optional[Callable[[AdvisoryVote], None]] = None,
    ):
        self.on_suggestion = on_suggestion
        self.on_vote = on_vote

        self.rate_limiter = RateLimiter(
            max_decisions=self.DECISIONS_PER_HOUR,
            window_seconds=3600
        )
        self.strategy_analyzer = StrategyAnalyzer()
        self.vote_predictor = VotePredictor()

        self.suggestions: List[Suggestion] = []
        self.votes: List[AdvisoryVote] = []
        self._lock = threading.Lock()

        # Track performance
        self.stats = {
            'total_suggestions': 0,
            'suggestions_followed': 0,
            'successful_suggestions': 0,
            'total_votes': 0,
            'correct_votes': 0,
            'rate_limited_count': 0,
        }

    def suggest(
        self,
        suggestion_type: SuggestionType,
        description: str,
        rationale: str,
        context: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Suggestion]:
        """
        Make an advisory suggestion.

        Returns the Suggestion if made, None if rate limited.
        """
        # Check rate limit
        if not self.rate_limiter.record_decision():
            with self._lock:
                self.stats['rate_limited_count'] += 1
            return None

        with self._lock:
            context = context or {}

            # Calculate confidence if not provided
            if confidence is None:
                confidence = self._calculate_confidence(suggestion_type, context)

            suggestion = Suggestion(
                suggestion_id=hashlib.sha256(
                    f"{suggestion_type}{description}{time.time()}".encode()
                ).hexdigest()[:16],
                suggestion_type=suggestion_type,
                timestamp=datetime.now(),
                description=description,
                rationale=rationale,
                confidence=confidence,
                context=context,
            )

            self.suggestions.append(suggestion)
            self.stats['total_suggestions'] += 1

            # Callback
            if self.on_suggestion:
                self.on_suggestion(suggestion)

            return suggestion

    def vote(
        self,
        target_id: str,
        mutation_features: Dict[str, float],
    ) -> Optional[AdvisoryVote]:
        """
        Cast an advisory vote on a proposed mutation.

        Returns the AdvisoryVote if cast, None if rate limited.
        """
        # Check rate limit
        if not self.rate_limiter.record_decision():
            with self._lock:
                self.stats['rate_limited_count'] += 1
            return None

        with self._lock:
            # Get vote prediction
            vote_type, confidence, rationale = self.vote_predictor.predict_vote(
                mutation_features
            )

            # Generate prediction
            if vote_type in [VoteType.STRONG_ACCEPT, VoteType.ACCEPT]:
                predicted_outcome = "Mutation likely to improve fitness"
            elif vote_type in [VoteType.STRONG_REJECT, VoteType.REJECT]:
                predicted_outcome = "Mutation likely to harm fitness or fail validation"
            else:
                predicted_outcome = "Uncertain outcome, insufficient evidence"

            advisory_vote = AdvisoryVote(
                vote_id=hashlib.sha256(
                    f"{target_id}{vote_type}{time.time()}".encode()
                ).hexdigest()[:16],
                vote_type=vote_type,
                timestamp=datetime.now(),
                target_id=target_id,
                rationale=rationale,
                confidence=confidence,
                predicted_outcome=predicted_outcome,
            )

            self.votes.append(advisory_vote)
            self.stats['total_votes'] += 1

            # Callback
            if self.on_vote:
                self.on_vote(advisory_vote)

            return advisory_vote

    def suggest_strategy(
        self,
        fitness_history: List[float],
        generation: int,
        population_diversity: float,
    ) -> Optional[Suggestion]:
        """Suggest an optimization strategy based on current context"""
        analysis = self.strategy_analyzer.analyze_context(
            fitness_history, generation, population_diversity
        )

        return self.suggest(
            suggestion_type=SuggestionType.MUTATION_STRATEGY,
            description=f"Recommend {analysis['recommended_strategy']} strategy",
            rationale=analysis['strategy_rationale'],
            context=analysis,
            confidence=0.6,
        )

    def rank_mutations(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Tuple[str, float, str]]:
        """
        Rank mutation candidates by predicted effectiveness.

        Returns list of (candidate_id, score, rationale) tuples.
        Does NOT count against decision limit (informational only).
        """
        rankings = []
        for candidate in candidates:
            candidate_id = candidate.get('id', 'unknown')
            features = candidate.get('features', {})

            vote_type, confidence, rationale = self.vote_predictor.predict_vote(features)

            # Convert vote to score
            score_map = {
                VoteType.STRONG_ACCEPT: 1.0,
                VoteType.ACCEPT: 0.7,
                VoteType.ABSTAIN: 0.5,
                VoteType.REJECT: 0.3,
                VoteType.STRONG_REJECT: 0.0,
            }
            score = score_map[vote_type] * confidence

            rankings.append((candidate_id, score, rationale))

        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _calculate_confidence(
        self,
        suggestion_type: SuggestionType,
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence for a suggestion"""
        # Base confidence by type
        base_confidence = {
            SuggestionType.MUTATION_STRATEGY: 0.6,
            SuggestionType.TARGET_SELECTION: 0.5,
            SuggestionType.PARAMETER_TUNING: 0.4,
            SuggestionType.EXPLORATION_VS_EXPLOIT: 0.5,
            SuggestionType.POPULATION_MANAGEMENT: 0.5,
            SuggestionType.TECHNIQUE_SWITCH: 0.4,
            SuggestionType.RESOURCE_ALLOCATION: 0.5,
            SuggestionType.HYPOTHESIS_TEST: 0.6,
        }

        confidence = base_confidence.get(suggestion_type, 0.5)

        # Adjust based on historical success rate
        if self.stats['total_suggestions'] > 10:
            success_rate = self.stats['successful_suggestions'] / self.stats['total_suggestions']
            confidence = confidence * 0.5 + success_rate * 0.5

        return confidence

    def record_suggestion_outcome(
        self,
        suggestion_id: str,
        was_followed: bool,
        was_successful: bool,
    ) -> None:
        """Record the outcome of a suggestion for learning"""
        with self._lock:
            for suggestion in self.suggestions:
                if suggestion.suggestion_id == suggestion_id:
                    suggestion.was_followed = was_followed
                    suggestion.outcome = "successful" if was_successful else "unsuccessful"

                    if was_followed:
                        self.stats['suggestions_followed'] += 1
                        if was_successful:
                            self.stats['successful_suggestions'] += 1

                    # Update strategy effectiveness
                    if suggestion.suggestion_type == SuggestionType.MUTATION_STRATEGY:
                        strategy = suggestion.context.get('recommended_strategy', '')
                        if strategy and was_followed:
                            self.strategy_analyzer.update_effectiveness(strategy, was_successful)
                    break

    def record_vote_outcome(self, vote_id: str, was_correct: bool) -> None:
        """Record the outcome of a vote for learning"""
        with self._lock:
            for vote in self.votes:
                if vote.vote_id == vote_id:
                    vote.was_correct = was_correct
                    if was_correct:
                        self.stats['correct_votes'] += 1
                    self.vote_predictor.record_outcome(vote.vote_type, was_correct)
                    break

    def get_decision_budget(self) -> Dict[str, Any]:
        """Get information about remaining decision budget"""
        return {
            'remaining_decisions': self.rate_limiter.remaining_decisions(),
            'max_decisions': self.DECISIONS_PER_HOUR,
            'time_until_reset': self.rate_limiter.time_until_reset(),
            'rate_limited_count': self.stats['rate_limited_count'],
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get advisor statistics"""
        follow_rate = (
            self.stats['suggestions_followed'] / self.stats['total_suggestions']
            if self.stats['total_suggestions'] > 0 else 0
        )
        success_rate = (
            self.stats['successful_suggestions'] / self.stats['suggestions_followed']
            if self.stats['suggestions_followed'] > 0 else 0
        )
        vote_accuracy = (
            self.stats['correct_votes'] / self.stats['total_votes']
            if self.stats['total_votes'] > 0 else 0
        )

        return {
            **self.stats,
            'follow_rate': follow_rate,
            'success_rate': success_rate,
            'vote_accuracy': vote_accuracy,
            'decision_budget': self.get_decision_budget(),
        }

    def get_recent_suggestions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent suggestions"""
        with self._lock:
            return [s.to_dict() for s in self.suggestions[-n:]]

    def get_recent_votes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent votes"""
        with self._lock:
            return [v.to_dict() for v in self.votes[-n:]]


# Global advisor instance
_advisor: Optional[Advisor] = None


def get_advisor() -> Advisor:
    """Get the global advisor instance"""
    global _advisor
    if _advisor is None:
        _advisor = Advisor()
    return _advisor
