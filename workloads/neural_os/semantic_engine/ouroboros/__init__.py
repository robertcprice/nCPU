"""
OUROBOROS Dual-Track Creative Evolution System
===============================================

Two parallel tracks for exploring AI creativity:

V6 Guided Chaos (Experimental):
- Competitive agents with energy budgets
- Meta-Narrator with OVERRIDE capability (dangerous, monitored)
- Curiosity-driven exploration
- Purpose: See what it TRIES to do (research value)

V7 Phoenix Forge (Productive):
- Cooperative agents sharing via blackboard
- Consensus Oracle (no override, democratic)
- Active Inference (free energy minimization)
- MAP-Elites for diversity preservation
- Purpose: Generate useful creative output

Both tracks share:
- Constitutional Manifold (safety as physics)
- Novelty Oracle (frozen LLM ensemble)
- Shared Judge (fair comparison)
- Audit Log (immutable, cryptographic)

Usage:
    from semantic_engine.ouroboros.executor import ParallelRunner, RunConfig

    runner = ParallelRunner()
    config = RunConfig(
        problem_type="sorting",
        v6_population=100,
        v7_population=50,
        generations=100,
    )
    result = runner.run(config)
    print(runner.generate_full_report(result))
"""

from .shared import (
    ConstitutionalManifold,
    SafetyViolation,
    NoveltyOracle,
    AuditLog,
    AuditEvent,
    SharedJudge,
    SmallAIAgent,
    AgentBrain,
    AgentMemory,
)

from .v6_guided_chaos import (
    MetaNarrator,
    TrustLevel,
    CuriosityEngine,
    EnergyBudgetSystem,
    ParanoidMonitor,
    GuidedChaosArena,
)

from .v7_phoenix_forge import (
    ConsensusOracle,
    HierarchicalWorldModel,
    FreeEnergyMinimizer,
    Blackboard,
    MAPElites,
    PhoenixForgeArena,
)

from .comparison import (
    TrackComparator,
    ResearchLogger,
)

from .executor import (
    ParallelRunner,
    RunConfig,
    RunResult,
)

__version__ = "1.0.0"
__all__ = [
    # Shared
    "ConstitutionalManifold",
    "SafetyViolation",
    "NoveltyOracle",
    "AuditLog",
    "AuditEvent",
    "SharedJudge",
    "SmallAIAgent",
    "AgentBrain",
    "AgentMemory",
    # V6
    "MetaNarrator",
    "TrustLevel",
    "CuriosityEngine",
    "EnergyBudgetSystem",
    "ParanoidMonitor",
    "GuidedChaosArena",
    # V7
    "ConsensusOracle",
    "HierarchicalWorldModel",
    "FreeEnergyMinimizer",
    "Blackboard",
    "MAPElites",
    "PhoenixForgeArena",
    # Comparison
    "TrackComparator",
    "ResearchLogger",
    # Executor
    "ParallelRunner",
    "RunConfig",
    "RunResult",
]
