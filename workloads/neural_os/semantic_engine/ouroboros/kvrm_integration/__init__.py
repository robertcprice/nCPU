"""
OUROBOROS-KVRM Integration
===========================
Connects OUROBOROS agents to the KVRM ecosystem.

OUROBOROS agents ARE KVRMs - they use the sense-think-act cycle:
- SENSE: Read problem state and other agents' outputs from SharedKVMemory
- THINK: Use LLM brain (Ollama) to reason about solutions
- ACT: Write solutions, hypotheses, observations to SharedKVMemory

The Meta-Narrator is a special KVRM that observes all agents and can
request overrides (with human approval).

Panel Recommendations Implemented:
- Human approval for OVERRIDE (Claude recommendation)
- Emergence detection (ChatGPT recommendation)
- Cross-generation meta-learning (all panels)
- Hybrid mode switching (Grok recommendation)

This connects directly to:
- /KVRM/kvrm-ecosystem/core/kvrm_base.py
- /KVRM/kvrm-ecosystem/core/digital_organism.py
- /KVRM/kvrm-ecosystem/core/shared_memory.py
"""

from .agent_kvrm import AgentKVRM, AgentConfig, OllamaLLM
from .narrator_kvrm import NarratorKVRM, NarratorConfig, TrustLevel
from .ouroboros_organism import OuroborosOrganism, OuroborosConfig, EmergenceSignal
from .meta_learner import MetaLearner, LearningSignal

__all__ = [
    # Core Agents
    "AgentKVRM",
    "AgentConfig",
    "OllamaLLM",
    # Narrator
    "NarratorKVRM",
    "NarratorConfig",
    "TrustLevel",
    # Orchestrator
    "OuroborosOrganism",
    "OuroborosConfig",
    "EmergenceSignal",
    # Meta-Learning
    "MetaLearner",
    "LearningSignal",
]
