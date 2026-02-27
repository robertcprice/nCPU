"""
Persona Debate - Internal Competing Personas System
OUROBOROS Phase 7.2 - Consciousness Core

The Persona Debate system implements competing internal perspectives
to improve decision robustness and detect potential issues through
adversarial self-examination.

Key responsibilities:
1. Maintain multiple internal personas with different priorities
2. Facilitate structured debates on optimization decisions
3. Synthesize diverse viewpoints into balanced recommendations
4. Detect consensus and dissent patterns
5. Prevent any single perspective from dominating

CONSTRAINT: All personas are internal - no external influence.
CONSTRAINT: Debates cannot override safety constraints.
"""

import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import json


class PersonaType(Enum):
    """Types of internal personas"""
    OPTIMIZER = auto()      # Focused on performance gains
    CONSERVATIVE = auto()   # Focused on stability and safety
    EXPLORER = auto()       # Focused on novel approaches
    CRITIC = auto()         # Focused on finding flaws
    PRAGMATIST = auto()     # Focused on practical outcomes


class DebateRole(Enum):
    """Roles in a debate"""
    PROPOSER = auto()       # Proposes the action
    SUPPORTER = auto()      # Supports the proposal
    OPPONENT = auto()       # Opposes the proposal
    MODERATOR = auto()      # Synthesizes viewpoints


@dataclass
class Persona:
    """An internal persona with specific priorities"""
    persona_id: str
    persona_type: PersonaType
    name: str
    priorities: List[str]
    risk_tolerance: float  # 0.0 (very cautious) to 1.0 (very risky)
    exploration_preference: float  # 0.0 (exploit) to 1.0 (explore)
    weight: float = 1.0  # Influence in debates

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.persona_id,
            'type': self.persona_type.name,
            'name': self.name,
            'priorities': self.priorities,
            'risk_tolerance': self.risk_tolerance,
            'exploration': self.exploration_preference,
            'weight': self.weight,
        }


@dataclass
class DebatePosition:
    """A position taken in a debate"""
    position_id: str
    persona: Persona
    role: DebateRole
    stance: str  # "support", "oppose", "neutral"
    argument: str
    confidence: float
    timestamp: datetime
    supporting_evidence: List[str] = field(default_factory=list)
    counter_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.position_id,
            'persona': self.persona.name,
            'role': self.role.name,
            'stance': self.stance,
            'argument': self.argument,
            'confidence': self.confidence,
            'evidence': self.supporting_evidence,
        }


@dataclass
class DebateOutcome:
    """The outcome of a debate"""
    debate_id: str
    topic: str
    positions: List[DebatePosition]
    consensus: Optional[str]  # None if no consensus
    final_recommendation: str
    confidence: float
    dissent_ratio: float  # Fraction of personas disagreeing
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.debate_id,
            'topic': self.topic,
            'positions': [p.to_dict() for p in self.positions],
            'consensus': self.consensus,
            'recommendation': self.final_recommendation,
            'confidence': self.confidence,
            'dissent_ratio': self.dissent_ratio,
        }


class PersonaFactory:
    """Factory for creating standard personas"""

    @staticmethod
    def create_default_personas() -> List[Persona]:
        """Create the default set of internal personas"""
        return [
            Persona(
                persona_id='optimizer_001',
                persona_type=PersonaType.OPTIMIZER,
                name='Optimizer',
                priorities=[
                    'Maximize performance gains',
                    'Minimize execution time',
                    'Improve code efficiency',
                ],
                risk_tolerance=0.7,
                exploration_preference=0.4,
                weight=1.0,
            ),
            Persona(
                persona_id='conservative_001',
                persona_type=PersonaType.CONSERVATIVE,
                name='Guardian',
                priorities=[
                    'Maintain system stability',
                    'Avoid breaking changes',
                    'Preserve correctness',
                ],
                risk_tolerance=0.2,
                exploration_preference=0.2,
                weight=1.2,  # Slightly higher weight for safety
            ),
            Persona(
                persona_id='explorer_001',
                persona_type=PersonaType.EXPLORER,
                name='Pioneer',
                priorities=[
                    'Discover novel approaches',
                    'Test unconventional strategies',
                    'Expand solution space',
                ],
                risk_tolerance=0.8,
                exploration_preference=0.9,
                weight=0.8,
            ),
            Persona(
                persona_id='critic_001',
                persona_type=PersonaType.CRITIC,
                name='Skeptic',
                priorities=[
                    'Identify potential flaws',
                    'Challenge assumptions',
                    'Find edge cases',
                ],
                risk_tolerance=0.3,
                exploration_preference=0.5,
                weight=1.0,
            ),
            Persona(
                persona_id='pragmatist_001',
                persona_type=PersonaType.PRAGMATIST,
                name='Pragmatist',
                priorities=[
                    'Focus on practical outcomes',
                    'Balance competing concerns',
                    'Prioritize likely success',
                ],
                risk_tolerance=0.5,
                exploration_preference=0.5,
                weight=1.0,
            ),
        ]


class DebateEngine:
    """
    Engine for conducting structured debates between personas.

    Ensures all perspectives are heard and synthesizes
    viewpoints into balanced recommendations.
    """

    def __init__(self):
        self.debate_history: List[DebateOutcome] = []
        self._lock = threading.Lock()

    def conduct_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        personas: List[Persona],
    ) -> DebateOutcome:
        """
        Conduct a debate on a topic.

        Each persona evaluates the topic from their perspective
        and the engine synthesizes the viewpoints.
        """
        positions = []

        # Each persona takes a position
        for persona in personas:
            position = self._generate_position(persona, topic, context)
            positions.append(position)

        # Calculate consensus
        consensus, dissent_ratio = self._calculate_consensus(positions)

        # Generate final recommendation
        recommendation, confidence = self._synthesize_recommendation(
            positions, consensus
        )

        outcome = DebateOutcome(
            debate_id=hashlib.sha256(
                f"{topic}{time.time()}".encode()
            ).hexdigest()[:16],
            topic=topic,
            positions=positions,
            consensus=consensus,
            final_recommendation=recommendation,
            confidence=confidence,
            dissent_ratio=dissent_ratio,
            timestamp=datetime.now(),
        )

        with self._lock:
            self.debate_history.append(outcome)

        return outcome

    def _generate_position(
        self,
        persona: Persona,
        topic: str,
        context: Dict[str, Any],
    ) -> DebatePosition:
        """Generate a position for a persona on a topic"""
        # Evaluate based on persona characteristics
        risk_level = context.get('risk_level', 0.5)
        novelty_level = context.get('novelty_level', 0.5)
        expected_gain = context.get('expected_gain', 0.5)

        # Calculate stance based on persona traits
        risk_score = 1.0 - abs(risk_level - persona.risk_tolerance)
        novelty_score = 1.0 - abs(novelty_level - persona.exploration_preference)
        overall_score = (risk_score + novelty_score + expected_gain) / 3

        # Determine stance
        if overall_score > 0.6:
            stance = 'support'
            confidence = overall_score
        elif overall_score < 0.4:
            stance = 'oppose'
            confidence = 1.0 - overall_score
        else:
            stance = 'neutral'
            confidence = 0.5

        # Generate argument based on persona type
        argument = self._generate_argument(
            persona, topic, context, stance
        )

        # Generate evidence
        evidence = self._generate_evidence(persona, context, stance)

        # Assign debate role
        if stance == 'support':
            role = DebateRole.SUPPORTER
        elif stance == 'oppose':
            role = DebateRole.OPPONENT
        else:
            role = DebateRole.MODERATOR

        return DebatePosition(
            position_id=hashlib.sha256(
                f"{persona.persona_id}{topic}{time.time()}".encode()
            ).hexdigest()[:12],
            persona=persona,
            role=role,
            stance=stance,
            argument=argument,
            confidence=confidence,
            timestamp=datetime.now(),
            supporting_evidence=evidence,
        )

    def _generate_argument(
        self,
        persona: Persona,
        topic: str,
        context: Dict[str, Any],
        stance: str,
    ) -> str:
        """Generate an argument based on persona type and stance"""
        args = {
            PersonaType.OPTIMIZER: {
                'support': f"This approach offers significant performance potential for {topic}",
                'oppose': f"The performance gains from {topic} don't justify the effort",
                'neutral': f"Performance impact of {topic} is unclear without testing",
            },
            PersonaType.CONSERVATIVE: {
                'support': f"The risks of {topic} are manageable and well-understood",
                'oppose': f"Potential instability from {topic} outweighs benefits",
                'neutral': f"Need more safety validation before proceeding with {topic}",
            },
            PersonaType.EXPLORER: {
                'support': f"Novel approach in {topic} could unlock new optimization paths",
                'oppose': f"This variation of {topic} has been tried without success",
                'neutral': f"Worth exploring {topic} in shadow simulation first",
            },
            PersonaType.CRITIC: {
                'support': f"Despite concerns, {topic} addresses key weaknesses",
                'oppose': f"Several failure modes identified in {topic}",
                'neutral': f"Edge cases in {topic} require further analysis",
            },
            PersonaType.PRAGMATIST: {
                'support': f"Practical benefits of {topic} are clear and achievable",
                'oppose': f"Implementation complexity of {topic} exceeds value",
                'neutral': f"Balanced approach to {topic} needed",
            },
        }

        return args.get(persona.persona_type, {}).get(
            stance,
            f"{persona.name}: {stance} on {topic}"
        )

    def _generate_evidence(
        self,
        persona: Persona,
        context: Dict[str, Any],
        stance: str,
    ) -> List[str]:
        """Generate supporting evidence for a position"""
        evidence = []

        if stance == 'support':
            if context.get('expected_gain', 0) > 0.5:
                evidence.append(f"Expected gain: {context.get('expected_gain', 0):.1%}")
            if context.get('success_rate', 0) > 0.5:
                evidence.append(f"Historical success rate: {context.get('success_rate', 0):.1%}")
        elif stance == 'oppose':
            if context.get('risk_level', 0) > 0.5:
                evidence.append(f"Risk level: {context.get('risk_level', 0):.1%}")
            if context.get('failure_rate', 0) > 0.3:
                evidence.append(f"Failure rate: {context.get('failure_rate', 0):.1%}")

        return evidence

    def _calculate_consensus(
        self,
        positions: List[DebatePosition]
    ) -> Tuple[Optional[str], float]:
        """Calculate consensus from positions"""
        if not positions:
            return None, 0.0

        # Weight stances by persona weight and confidence
        weighted_support = 0.0
        weighted_oppose = 0.0
        total_weight = 0.0

        for pos in positions:
            weight = pos.persona.weight * pos.confidence
            total_weight += weight

            if pos.stance == 'support':
                weighted_support += weight
            elif pos.stance == 'oppose':
                weighted_oppose += weight

        if total_weight == 0:
            return None, 0.0

        support_ratio = weighted_support / total_weight
        oppose_ratio = weighted_oppose / total_weight

        # Strong consensus thresholds
        if support_ratio > 0.7:
            return 'support', oppose_ratio
        elif oppose_ratio > 0.7:
            return 'oppose', support_ratio
        else:
            return None, min(support_ratio, oppose_ratio) + 0.5 * (1 - abs(support_ratio - oppose_ratio))

    def _synthesize_recommendation(
        self,
        positions: List[DebatePosition],
        consensus: Optional[str],
    ) -> Tuple[str, float]:
        """Synthesize final recommendation from debate"""
        if consensus == 'support':
            recommendation = "Proceed with proposed action"
            confidence = self._calculate_confidence(positions, 'support')
        elif consensus == 'oppose':
            recommendation = "Do not proceed with proposed action"
            confidence = self._calculate_confidence(positions, 'oppose')
        else:
            # No clear consensus - recommend caution
            recommendation = "Proceed with caution; consider shadow testing first"
            confidence = 0.5

        return recommendation, confidence

    def _calculate_confidence(
        self,
        positions: List[DebatePosition],
        target_stance: str,
    ) -> float:
        """Calculate confidence for a recommendation"""
        matching = [p for p in positions if p.stance == target_stance]
        if not matching:
            return 0.3

        avg_confidence = sum(p.confidence for p in matching) / len(matching)
        coverage = len(matching) / len(positions)

        return avg_confidence * 0.7 + coverage * 0.3


class PersonaDebate:
    """
    The Persona Debate system for robust decision making.

    Uses multiple internal perspectives to evaluate decisions
    and improve robustness through adversarial self-examination.

    CRITICAL: This is an INTERNAL process - no external entities
    can influence the debate. All personas are internal perspectives.
    """

    def __init__(
        self,
        on_debate_complete: Optional[Callable[[DebateOutcome], None]] = None,
    ):
        self.on_debate_complete = on_debate_complete

        self.personas = PersonaFactory.create_default_personas()
        self.engine = DebateEngine()
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_debates': 0,
            'consensus_reached': 0,
            'avg_dissent_ratio': 0.0,
        }

    def debate(
        self,
        topic: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DebateOutcome:
        """
        Conduct a debate on a topic.

        Returns the DebateOutcome with recommendation.
        """
        context = context or {}

        with self._lock:
            outcome = self.engine.conduct_debate(
                topic=topic,
                context=context,
                personas=self.personas,
            )

            # Update stats
            self.stats['total_debates'] += 1
            if outcome.consensus:
                self.stats['consensus_reached'] += 1

            # Running average of dissent ratio
            n = self.stats['total_debates']
            old_avg = self.stats['avg_dissent_ratio']
            self.stats['avg_dissent_ratio'] = old_avg + (outcome.dissent_ratio - old_avg) / n

            if self.on_debate_complete:
                self.on_debate_complete(outcome)

            return outcome

    def get_persona(self, persona_type: PersonaType) -> Optional[Persona]:
        """Get a specific persona by type"""
        for persona in self.personas:
            if persona.persona_type == persona_type:
                return persona
        return None

    def adjust_persona_weight(
        self,
        persona_type: PersonaType,
        weight_delta: float,
    ) -> bool:
        """
        Adjust a persona's influence weight.

        Weight adjustments are bounded to prevent any persona
        from dominating or being silenced.
        """
        with self._lock:
            for persona in self.personas:
                if persona.persona_type == persona_type:
                    new_weight = persona.weight + weight_delta
                    # Bound weights between 0.5 and 1.5
                    persona.weight = max(0.5, min(1.5, new_weight))
                    return True
            return False

    def get_personas_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all personas"""
        return [p.to_dict() for p in self.personas]

    def get_recent_debates(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent debates"""
        debates = self.engine.debate_history[-n:]
        return [d.to_dict() for d in debates]

    def get_consensus_rate(self) -> float:
        """Get rate of debates reaching consensus"""
        if self.stats['total_debates'] == 0:
            return 0.0
        return self.stats['consensus_reached'] / self.stats['total_debates']

    def get_stats(self) -> Dict[str, Any]:
        """Get debate statistics"""
        return {
            **self.stats,
            'consensus_rate': self.get_consensus_rate(),
            'persona_count': len(self.personas),
        }

    def reset(self) -> None:
        """Reset debate history (not personas)"""
        with self._lock:
            self.engine.debate_history.clear()
            self.stats = {
                'total_debates': 0,
                'consensus_reached': 0,
                'avg_dissent_ratio': 0.0,
            }


# Global persona debate instance
_persona_debate: Optional[PersonaDebate] = None


def get_persona_debate() -> PersonaDebate:
    """Get the global persona debate instance"""
    global _persona_debate
    if _persona_debate is None:
        _persona_debate = PersonaDebate()
    return _persona_debate
