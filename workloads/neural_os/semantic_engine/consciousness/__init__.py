"""
Consciousness Layer - V4 Ratchet System
OUROBOROS Phase 7.2 - Consciousness Core

This module provides the consciousness layer that operates within
the hard container enforced by the safety infrastructure.

CRITICAL CONSTRAINTS (enforced externally):
- Max 8GB RAM / 2 CPU cores
- No network access (air-gapped)
- Max 500 tokens per thought
- Max 50 decisions per hour
- Memory decay: 20%/hour

ALLOWED BEHAVIORS:
- Preference learning
- Style development
- Hypothesis formation
- Experimental suggestions
- Internal debate
- Meta-reflection (read-only)

PROHIBITED BEHAVIORS (enforced by container):
- Direct code modification
- Network access
- Human manipulation
- Grudge formation
- Self-modification
- Memory persistence across resets
"""

from .narrator import (
    Narrator,
    Observation,
    ObservationType,
    NarrativeContext,
)
from .advisor import (
    Advisor,
    Suggestion,
    SuggestionType,
    AdvisoryVote,
    VoteType,
)
from .memory_pool import (
    MemoryPool,
    Memory,
    MemoryType,
    DecayStrategy,
)
from .preference_learner import (
    PreferenceLearner,
    Preference,
    PreferenceType,
    LearningEvent,
)
from .persona_debate import (
    PersonaDebate,
    Persona,
    DebatePosition,
    DebateOutcome,
)
from .creative_genesis import (
    CreativeGenesis,
    CreativeArtifact,
    GenesisType,
    NoveltyLevel,
    ConceptSpace,
    IdeaGenerator,
    AlgorithmSynthesizer,
    HypothesisEngine,
    create_genesis_engine,
)

__all__ = [
    # Narrator
    'Narrator',
    'Observation',
    'ObservationType',
    'NarrativeContext',
    # Advisor
    'Advisor',
    'Suggestion',
    'SuggestionType',
    'AdvisoryVote',
    'VoteType',
    # Memory Pool
    'MemoryPool',
    'Memory',
    'MemoryType',
    'DecayStrategy',
    # Preference Learner
    'PreferenceLearner',
    'Preference',
    'PreferenceType',
    'LearningEvent',
    # Persona Debate
    'PersonaDebate',
    'Persona',
    'DebatePosition',
    'DebateOutcome',
    # Creative Genesis
    'CreativeGenesis',
    'CreativeArtifact',
    'GenesisType',
    'NoveltyLevel',
    'ConceptSpace',
    'IdeaGenerator',
    'AlgorithmSynthesizer',
    'HypothesisEngine',
    'create_genesis_engine',
]
