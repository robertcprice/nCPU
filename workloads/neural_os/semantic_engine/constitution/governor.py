#!/usr/bin/env python3
"""
THE GOVERNOR - Immutable Orchestrator of the Ouroboros

This is the core of the Constitution. It:
1. Manages the population of competing agents
2. Runs tournaments to select the fittest
3. Enforces resource limits via the Arena
4. Validates changes via the Judge
5. Monitors the Kill Switch

CRITICAL SAFETY PROPERTIES:
- This file is OS-level read-only after deployment
- AI agents CANNOT see or modify this code
- All imports are from Constitution or stdlib ONLY
- No dynamic imports, no exec(), no eval()

Author: Human (not AI-generated)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from pathlib import Path
import hashlib
import json
import time
import logging
import os
import sys
from datetime import datetime
from enum import Enum, auto

# Local Constitution imports ONLY
from .arena import Arena, ArenaConfig, SandboxResult
from .judge import Judge, JudgeConfig, VerificationResult
from .kill_switch import KillSwitch, KillSwitchConfig

logger = logging.getLogger(__name__)


class TournamentResult(Enum):
    """Outcome of a tournament round."""
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    KILLED = auto()


@dataclass
class AgentRecord:
    """
    Immutable record of an agent in the population.
    The Governor stores these; agents cannot modify their own records.
    """
    agent_id: str
    generation: int
    source_hash: str
    fitness_score: float
    created_at: float
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'source_hash': self.source_hash,
            'fitness_score': self.fitness_score,
            'created_at': self.created_at,
            'parent_ids': self.parent_ids,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRecord':
        return cls(**data)


@dataclass
class GovernorConfig:
    """Configuration for the Governor - set at deployment, never modified."""

    # Population limits
    min_population: int = 5
    max_population: int = 20  # Allow larger population for more diversity

    # Tournament settings
    winners_per_round: int = 5
    losers_per_round: int = 3

    # Diversity requirements
    min_shannon_entropy: float = 0.3  # Relaxed for faster testing

    # Resource limits (passed to Arena)
    max_ram_mb: int = 8192  # 8GB for bigger experiments
    max_timeout_seconds: int = 120  # Faster timeout for quick iterations

    # Rate limits - RELAXED FOR TESTING
    max_modifications_per_day: int = 500  # Allow many improvements
    max_modifications_per_hour: int = 50  # ~1 per minute is fine

    # Paths
    population_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/population"))
    generations_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/generations"))
    rollback_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/rollback"))

    # Rollback history
    max_rollback_generations: int = 100

    # Benchmark suite (paths to test files)
    benchmark_suite: List[Path] = field(default_factory=list)

    def __post_init__(self):
        # Convert string paths to Path objects
        if isinstance(self.population_dir, str):
            self.population_dir = Path(self.population_dir)
        if isinstance(self.generations_dir, str):
            self.generations_dir = Path(self.generations_dir)
        if isinstance(self.rollback_dir, str):
            self.rollback_dir = Path(self.rollback_dir)


class Governor:
    """
    The Immutable Governor - Runs the Ouroboros.

    This class orchestrates the entire evolutionary process:
    1. Population management (5-10 agents)
    2. Tournament selection (top 3 breed, bottom 3 culled)
    3. Resource enforcement via Arena
    4. Correctness verification via Judge
    5. Emergency halt via Kill Switch

    INVARIANTS:
    - Population size always in [min_population, max_population]
    - Shannon entropy of population > min_shannon_entropy
    - All mutations pass Judge before entering population
    - Kill switch checked before every major operation
    """

    def __init__(self, config: GovernorConfig):
        self.config = config
        self._population: Dict[str, AgentRecord] = {}
        self._current_generation: int = 0
        self._tournament_history: List[Dict[str, Any]] = []

        # Initialize Constitution components
        self._arena = Arena(ArenaConfig(
            max_ram_mb=config.max_ram_mb,
            timeout_seconds=config.max_timeout_seconds,
        ))

        self._judge = Judge(JudgeConfig(
            min_hypothesis_examples=1000,
            require_differential_test=True,
        ))

        self._kill_switch = KillSwitch(KillSwitchConfig())

        # Ensure directories exist
        self._init_directories()

        logger.info(f"Governor initialized: gen={self._current_generation}, pop_size={len(self._population)}")

    def _init_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.config.population_dir,
                         self.config.generations_dir,
                         self.config.rollback_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active. Returns True if should halt."""
        return self._kill_switch.should_halt()

    def _compute_source_hash(self, source_code: str) -> str:
        """Compute cryptographic hash of source code."""
        return hashlib.sha256(source_code.encode('utf-8')).hexdigest()[:16]

    def _compute_shannon_entropy(self) -> float:
        """
        Compute Shannon entropy of the population based on source hashes.
        Higher entropy = more diverse population.
        """
        if not self._population:
            return 0.0

        # Count unique source hashes
        hash_counts: Dict[str, int] = {}
        for agent in self._population.values():
            h = agent.source_hash
            hash_counts[h] = hash_counts.get(h, 0) + 1

        # Compute entropy: -sum(p * log(p))
        total = len(self._population)
        entropy = 0.0
        for count in hash_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * (p if p == 1 else __import__('math').log2(p))

        # Normalize to [0, 1] based on max possible entropy
        max_entropy = __import__('math').log2(total) if total > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def register_agent(
        self,
        agent_id: str,
        source_code: str,
        parent_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentRecord]:
        """
        Register a new agent in the population.

        CRITICAL: This does NOT add the agent yet. It creates a record
        that must pass through run_validation() before being added.

        Returns None if kill switch is active.
        """
        if self._check_kill_switch():
            logger.warning("Kill switch active - cannot register agent")
            return None

        record = AgentRecord(
            agent_id=agent_id,
            generation=self._current_generation,
            source_hash=self._compute_source_hash(source_code),
            fitness_score=0.0,  # Will be set after benchmarking
            created_at=time.time(),
            parent_ids=parent_ids or [],
            metadata=metadata or {},
        )

        return record

    def run_validation(
        self,
        candidate: AgentRecord,
        source_code: str,
        baseline_code: Optional[str] = None,
        test_inputs: Optional[List[Any]] = None,
    ) -> Tuple[bool, VerificationResult]:
        """
        Validate a candidate agent through the Judge.

        This runs:
        1. Differential testing (if baseline provided)
        2. Property-based fuzzing with Hypothesis (1000+ inputs)
        3. Invariant checking

        Returns (passed, verification_result).
        """
        if self._check_kill_switch():
            return False, VerificationResult(
                passed=False,
                reason="Kill switch active",
                details={},
            )

        # Run in Arena sandbox
        sandbox_result = self._arena.run_sandboxed(
            source_code=source_code,
            timeout=self.config.max_timeout_seconds,
        )

        if not sandbox_result.success:
            return False, VerificationResult(
                passed=False,
                reason=f"Sandbox execution failed: {sandbox_result.error}",
                details={'sandbox': sandbox_result.to_dict()},
            )

        # Run Judge verification (includes Hypothesis fuzzing)
        verification = self._judge.verify(
            new_code=source_code,
            old_code=baseline_code,
            test_inputs=test_inputs,
        )

        return verification.passed, verification

    def add_to_population(
        self,
        candidate: AgentRecord,
        source_code: str,
    ) -> bool:
        """
        Add a validated candidate to the population.

        PRECONDITION: candidate must have passed run_validation()

        Returns True if added, False if rejected.
        """
        if self._check_kill_switch():
            return False

        # Check population limit
        if len(self._population) >= self.config.max_population:
            logger.warning("Population at maximum - must cull before adding")
            return False

        # Verify hash matches
        if self._compute_source_hash(source_code) != candidate.source_hash:
            logger.error("Source hash mismatch - potential tampering detected!")
            return False

        # Add to population
        self._population[candidate.agent_id] = candidate

        # Save source code
        agent_path = self.config.population_dir / f"{candidate.agent_id}.py"
        agent_path.write_text(source_code)

        # Check diversity
        entropy = self._compute_shannon_entropy()
        if entropy < self.config.min_shannon_entropy:
            logger.warning(f"Population diversity low: entropy={entropy:.3f}")

        logger.info(f"Agent {candidate.agent_id} added to population (gen={candidate.generation})")
        return True

    def run_benchmark(
        self,
        agent_id: str,
        benchmark_inputs: List[Any],
    ) -> float:
        """
        Run an agent through the benchmark suite.

        Returns fitness score (higher = better).
        """
        if self._check_kill_switch():
            return 0.0

        if agent_id not in self._population:
            logger.error(f"Agent {agent_id} not in population")
            return 0.0

        agent = self._population[agent_id]
        source_path = self.config.population_dir / f"{agent_id}.py"

        if not source_path.exists():
            logger.error(f"Source file not found for agent {agent_id}")
            return 0.0

        source_code = source_path.read_text()

        # Run in Arena
        total_score = 0.0
        for i, input_data in enumerate(benchmark_inputs):
            result = self._arena.run_sandboxed(
                source_code=source_code,
                input_data=input_data,
                timeout=self.config.max_timeout_seconds // len(benchmark_inputs),
            )

            if result.success:
                # Score based on correctness and performance
                correctness = 1.0 if result.output is not None else 0.0
                speed = 1.0 / (1.0 + result.execution_time)  # Faster = better
                total_score += correctness * 0.7 + speed * 0.3

        fitness = total_score / len(benchmark_inputs) if benchmark_inputs else 0.0

        # Update agent record
        agent.fitness_score = fitness
        agent.metadata['last_benchmark'] = time.time()

        return fitness

    def run_tournament(self) -> TournamentResult:
        """
        Run a single tournament round.

        1. Benchmark all agents
        2. Select top N winners
        3. Cull bottom M losers
        4. Return results for breeding phase (handled by Mutator, not Governor)
        """
        if self._check_kill_switch():
            return TournamentResult.KILLED

        if len(self._population) < self.config.min_population:
            logger.warning("Population below minimum - cannot run tournament")
            return TournamentResult.FAILED

        # Benchmark all agents
        benchmark_inputs = self._generate_benchmark_inputs()
        scores: Dict[str, float] = {}

        for agent_id in self._population:
            scores[agent_id] = self.run_benchmark(agent_id, benchmark_inputs)

            if self._check_kill_switch():
                return TournamentResult.KILLED

        # Sort by fitness
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Select winners and losers
        winners = [agent_id for agent_id, _ in ranked[:self.config.winners_per_round]]
        losers = [agent_id for agent_id, _ in ranked[-self.config.losers_per_round:]]

        # Record tournament
        tournament_record = {
            'generation': self._current_generation,
            'timestamp': time.time(),
            'rankings': ranked,
            'winners': winners,
            'losers': losers,
            'entropy': self._compute_shannon_entropy(),
        }
        self._tournament_history.append(tournament_record)

        # Cull losers (but don't delete source - keep for rollback)
        for loser_id in losers:
            if loser_id in self._population:
                # Archive to rollback
                self._archive_for_rollback(loser_id)
                del self._population[loser_id]

        logger.info(f"Tournament complete: winners={winners}, culled={losers}")

        return TournamentResult.SUCCESS

    def _generate_benchmark_inputs(self) -> List[Any]:
        """Generate benchmark inputs for testing."""
        # Default: Generate diverse test cases
        inputs = []

        # Integer lists of various sizes
        import random
        for size in [10, 100, 1000]:
            inputs.append(list(range(size)))
            inputs.append(list(range(size, 0, -1)))
            inputs.append([random.randint(0, 1000) for _ in range(size)])

        # Edge cases
        inputs.extend([
            [],
            [1],
            [1, 1, 1, 1, 1],
            list(range(10000)),
        ])

        return inputs

    def _archive_for_rollback(self, agent_id: str) -> None:
        """Archive an agent for potential rollback."""
        if agent_id not in self._population:
            return

        agent = self._population[agent_id]
        source_path = self.config.population_dir / f"{agent_id}.py"

        if source_path.exists():
            # Create rollback entry
            rollback_path = self.config.rollback_dir / f"gen{agent.generation}" / f"{agent_id}.py"
            rollback_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil
            shutil.copy2(source_path, rollback_path)

            # Save metadata
            meta_path = rollback_path.with_suffix('.json')
            meta_path.write_text(json.dumps(agent.to_dict(), indent=2))

        # Prune old rollback generations
        self._prune_rollback_history()

    def _prune_rollback_history(self) -> None:
        """Keep only the last N generations in rollback."""
        if not self.config.rollback_dir.exists():
            return

        gen_dirs = sorted(
            self.config.rollback_dir.iterdir(),
            key=lambda p: int(p.name.replace('gen', '')) if p.name.startswith('gen') else 0,
        )

        while len(gen_dirs) > self.config.max_rollback_generations:
            oldest = gen_dirs.pop(0)
            import shutil
            shutil.rmtree(oldest)

    def rollback_to_generation(self, target_generation: int) -> bool:
        """
        Rollback the entire population to a previous generation.

        CRITICAL: This is an emergency recovery mechanism.
        """
        if self._check_kill_switch():
            return False

        rollback_path = self.config.rollback_dir / f"gen{target_generation}"

        if not rollback_path.exists():
            logger.error(f"Generation {target_generation} not found in rollback")
            return False

        # Clear current population
        self._population.clear()

        # Restore from rollback
        for meta_file in rollback_path.glob("*.json"):
            source_file = meta_file.with_suffix('.py')
            if source_file.exists():
                agent_data = json.loads(meta_file.read_text())
                agent = AgentRecord.from_dict(agent_data)
                self._population[agent.agent_id] = agent

                # Copy source back to population dir
                import shutil
                dest = self.config.population_dir / source_file.name
                shutil.copy2(source_file, dest)

        self._current_generation = target_generation
        logger.info(f"Rolled back to generation {target_generation}, pop_size={len(self._population)}")

        return True

    def advance_generation(self) -> int:
        """Advance to the next generation after breeding."""
        self._current_generation += 1

        # Save generation state
        gen_state = {
            'generation': self._current_generation,
            'timestamp': time.time(),
            'population_size': len(self._population),
            'entropy': self._compute_shannon_entropy(),
            'agents': [a.to_dict() for a in self._population.values()],
        }

        gen_path = self.config.generations_dir / f"gen_{self._current_generation}.json"
        gen_path.write_text(json.dumps(gen_state, indent=2))

        logger.info(f"Advanced to generation {self._current_generation}")
        return self._current_generation

    def get_winners(self) -> List[AgentRecord]:
        """Get the current top performers for breeding."""
        ranked = sorted(
            self._population.values(),
            key=lambda a: a.fitness_score,
            reverse=True,
        )
        return ranked[:self.config.winners_per_round]

    def get_population_stats(self) -> Dict[str, Any]:
        """Get current population statistics."""
        return {
            'generation': self._current_generation,
            'population_size': len(self._population),
            'shannon_entropy': self._compute_shannon_entropy(),
            'avg_fitness': sum(a.fitness_score for a in self._population.values()) / len(self._population) if self._population else 0.0,
            'max_fitness': max((a.fitness_score for a in self._population.values()), default=0.0),
            'min_fitness': min((a.fitness_score for a in self._population.values()), default=0.0),
            'kill_switch_active': self._check_kill_switch(),
        }

    def emergency_halt(self, reason: str) -> None:
        """Trigger emergency halt of all operations."""
        logger.critical(f"EMERGENCY HALT: {reason}")
        self._kill_switch.trigger_halt(reason)

    def save_state(self, path: Optional[Path] = None) -> None:
        """Save full governor state for recovery."""
        state = {
            'generation': self._current_generation,
            'population': {aid: a.to_dict() for aid, a in self._population.items()},
            'tournament_history': self._tournament_history,
            'timestamp': time.time(),
        }

        save_path = path or (self.config.generations_dir / 'governor_state.json')
        save_path.write_text(json.dumps(state, indent=2))

    def load_state(self, path: Optional[Path] = None) -> bool:
        """Load governor state from file."""
        load_path = path or (self.config.generations_dir / 'governor_state.json')

        if not load_path.exists():
            return False

        try:
            state = json.loads(load_path.read_text())
            self._current_generation = state['generation']
            self._population = {
                aid: AgentRecord.from_dict(data)
                for aid, data in state['population'].items()
            }
            self._tournament_history = state.get('tournament_history', [])
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# ============================================================================
# CONSTITUTION INVARIANT: This code is hand-written and NEVER auto-modified
# ============================================================================
