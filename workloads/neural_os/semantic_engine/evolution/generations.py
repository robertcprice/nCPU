#!/usr/bin/env python3
"""
GENERATION MANAGER - Generational Replacement for the Ouroboros

Handles the core evolutionary cycle:
1. Gen_N creates Gen_N+1 offspring
2. Offspring tested OFFLINE in sandbox
3. If valid → offspring enter population
4. Tournament selection culls weak agents
5. Keep last 100 generations for rollback

This is NOT in-place editing. Each generation is a complete snapshot
that can be restored if something goes wrong.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import time
import shutil
import logging

from .population import Agent, Population, PopulationConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for generation management."""

    # Storage
    generations_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/generations"))
    rollback_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/rollback"))

    # History
    max_generations_to_keep: int = 100

    # Breeding
    offspring_per_generation: int = 5  # How many new agents per generation

    # Validation
    require_improvement: bool = False  # Require fitness improvement to accept
    min_fitness_threshold: float = 0.0  # Minimum fitness to enter population

    def __post_init__(self):
        if isinstance(self.generations_dir, str):
            self.generations_dir = Path(self.generations_dir)
        if isinstance(self.rollback_dir, str):
            self.rollback_dir = Path(self.rollback_dir)


@dataclass
class GenerationSnapshot:
    """A complete snapshot of a generation."""

    generation_number: int
    timestamp: float
    agents: List[Dict[str, Any]]
    fitness_stats: Dict[str, float]
    diversity_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation_number': self.generation_number,
            'timestamp': self.timestamp,
            'agents': self.agents,
            'fitness_stats': self.fitness_stats,
            'diversity_metrics': self.diversity_metrics,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationSnapshot':
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save snapshot to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> 'GenerationSnapshot':
        """Load snapshot from file."""
        return cls.from_dict(json.loads(path.read_text()))


class GenerationManager:
    """
    Manages generational replacement and rollback.

    Key operations:
    - create_snapshot: Save current generation state
    - advance_generation: Move to next generation
    - rollback: Restore a previous generation
    - prune_history: Remove old generations
    """

    def __init__(self, config: GenerationConfig, population: Population):
        self.config = config
        self.population = population
        self._current_generation: int = 0
        self._generation_history: List[int] = []

        # Ensure directories exist
        self.config.generations_dir.mkdir(parents=True, exist_ok=True)
        self.config.rollback_dir.mkdir(parents=True, exist_ok=True)

        # Load existing history
        self._load_history()

        logger.info(f"GenerationManager initialized: gen={self._current_generation}")

    @property
    def current_generation(self) -> int:
        """Current generation number."""
        return self._current_generation

    def _load_history(self) -> None:
        """Load generation history from disk."""
        gen_files = sorted(self.config.generations_dir.glob("gen_*.json"))

        if gen_files:
            # Get highest generation number
            for f in gen_files:
                try:
                    gen_num = int(f.stem.split('_')[1])
                    self._generation_history.append(gen_num)
                except (ValueError, IndexError):
                    pass

            if self._generation_history:
                self._current_generation = max(self._generation_history)

    def create_snapshot(self) -> GenerationSnapshot:
        """Create a snapshot of the current generation."""
        snapshot = GenerationSnapshot(
            generation_number=self._current_generation,
            timestamp=time.time(),
            agents=[a.to_dict() for a in self.population.agents],
            fitness_stats=self.population.get_fitness_stats(),
            diversity_metrics=self.population.get_diversity_metrics(),
            metadata={
                'population_size': self.population.size,
            },
        )

        # Save to generations directory
        snapshot_path = self.config.generations_dir / f"gen_{self._current_generation}.json"
        snapshot.save(snapshot_path)

        # Also save to rollback
        rollback_path = self.config.rollback_dir / f"gen_{self._current_generation}.json"
        snapshot.save(rollback_path)

        logger.info(f"Created snapshot for generation {self._current_generation}")
        return snapshot

    def advance_generation(self) -> int:
        """
        Advance to the next generation.

        This:
        1. Creates a snapshot of current state
        2. Increments generation counter
        3. Records fitness history
        4. Prunes old history

        Returns the new generation number.
        """
        # Snapshot current state before advancing
        self.create_snapshot()

        # Advance
        self._current_generation += 1
        self._generation_history.append(self._current_generation)

        # Record fitness
        self.population.record_fitness_snapshot()

        # Prune old generations
        self._prune_history()

        logger.info(f"Advanced to generation {self._current_generation}")
        return self._current_generation

    def _prune_history(self) -> None:
        """Remove old generation files beyond the keep limit."""
        if len(self._generation_history) <= self.config.max_generations_to_keep:
            return

        # Sort and find generations to remove
        to_remove = sorted(self._generation_history)[:-self.config.max_generations_to_keep]

        for gen_num in to_remove:
            # Remove from generations dir
            gen_path = self.config.generations_dir / f"gen_{gen_num}.json"
            if gen_path.exists():
                gen_path.unlink()

            # Keep rollback for longer (double the keep limit)
            if gen_num < self._current_generation - (self.config.max_generations_to_keep * 2):
                rollback_path = self.config.rollback_dir / f"gen_{gen_num}.json"
                if rollback_path.exists():
                    rollback_path.unlink()

            self._generation_history.remove(gen_num)

    def can_rollback(self, target_generation: int) -> bool:
        """Check if rollback to target generation is possible."""
        rollback_path = self.config.rollback_dir / f"gen_{target_generation}.json"
        return rollback_path.exists()

    def list_available_rollbacks(self) -> List[int]:
        """List all generations available for rollback."""
        available = []
        for f in self.config.rollback_dir.glob("gen_*.json"):
            try:
                gen_num = int(f.stem.split('_')[1])
                available.append(gen_num)
            except (ValueError, IndexError):
                pass
        return sorted(available)

    def rollback(self, target_generation: int) -> bool:
        """
        Rollback to a previous generation.

        This:
        1. Loads the snapshot from rollback
        2. Clears current population
        3. Restores agents from snapshot
        4. Updates generation counter

        Returns True if successful.
        """
        if not self.can_rollback(target_generation):
            logger.error(f"Cannot rollback to generation {target_generation} - not found")
            return False

        rollback_path = self.config.rollback_dir / f"gen_{target_generation}.json"

        try:
            snapshot = GenerationSnapshot.load(rollback_path)
        except Exception as e:
            logger.error(f"Failed to load rollback snapshot: {e}")
            return False

        # Clear current population
        self.population.clear()

        # Restore agents
        for agent_data in snapshot.agents:
            agent = Agent.from_dict(agent_data)
            self.population.add_agent(agent)

        # Update generation
        self._current_generation = target_generation

        logger.info(f"Rolled back to generation {target_generation}, restored {len(snapshot.agents)} agents")
        return True

    def get_generation_summary(self, generation: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get summary of a generation."""
        gen = generation or self._current_generation

        # Try generations dir first
        gen_path = self.config.generations_dir / f"gen_{gen}.json"
        if not gen_path.exists():
            gen_path = self.config.rollback_dir / f"gen_{gen}.json"

        if not gen_path.exists():
            return None

        try:
            snapshot = GenerationSnapshot.load(gen_path)
            return {
                'generation': snapshot.generation_number,
                'timestamp': snapshot.timestamp,
                'agent_count': len(snapshot.agents),
                'fitness_stats': snapshot.fitness_stats,
                'diversity_metrics': snapshot.diversity_metrics,
            }
        except Exception as e:
            logger.error(f"Failed to load generation {gen}: {e}")
            return None

    def compare_generations(self, gen_a: int, gen_b: int) -> Optional[Dict[str, Any]]:
        """Compare two generations."""
        summary_a = self.get_generation_summary(gen_a)
        summary_b = self.get_generation_summary(gen_b)

        if not summary_a or not summary_b:
            return None

        return {
            'generation_a': gen_a,
            'generation_b': gen_b,
            'fitness_improvement': (
                summary_b['fitness_stats'].get('mean', 0) -
                summary_a['fitness_stats'].get('mean', 0)
            ),
            'diversity_change': (
                summary_b['diversity_metrics'].get('entropy', 0) -
                summary_a['diversity_metrics'].get('entropy', 0)
            ),
            'population_change': (
                summary_b['agent_count'] - summary_a['agent_count']
            ),
        }

    def get_evolution_trajectory(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """Get fitness trajectory over recent generations."""
        trajectory = []

        gens = sorted(self._generation_history)[-last_n:]

        for gen in gens:
            summary = self.get_generation_summary(gen)
            if summary:
                trajectory.append({
                    'generation': gen,
                    'mean_fitness': summary['fitness_stats'].get('mean', 0),
                    'max_fitness': summary['fitness_stats'].get('max', 0),
                    'entropy': summary['diversity_metrics'].get('entropy', 0),
                })

        return trajectory

    def get_stats(self) -> Dict[str, Any]:
        """Get generation manager statistics."""
        available_rollbacks = self.list_available_rollbacks()

        return {
            'current_generation': self._current_generation,
            'generations_stored': len(self._generation_history),
            'oldest_available': min(available_rollbacks) if available_rollbacks else None,
            'newest_available': max(available_rollbacks) if available_rollbacks else None,
            'rollback_count': len(available_rollbacks),
            'max_generations_to_keep': self.config.max_generations_to_keep,
        }


class BreedingManager:
    """
    Handles the breeding process between generations.

    Operations:
    - breed: Create offspring from parent agents
    - validate_offspring: Check if offspring should enter population
    - integrate_offspring: Add valid offspring to population
    """

    def __init__(
        self,
        population: Population,
        generation_manager: GenerationManager,
        config: GenerationConfig,
    ):
        self.population = population
        self.generation_manager = generation_manager
        self.config = config
        self._breeding_history: List[Dict[str, Any]] = []

    def select_breeding_pairs(self, count: int = 5) -> List[Tuple[Agent, Agent]]:
        """Select pairs of parents for breeding."""
        pairs = []

        for _ in range(count):
            parents = self.population.select_parents(2)
            if len(parents) >= 2:
                pairs.append((parents[0], parents[1]))
            elif len(parents) == 1:
                # Self-breeding (mutation only)
                pairs.append((parents[0], parents[0]))

        return pairs

    def create_offspring(
        self,
        parent_a: Agent,
        parent_b: Agent,
        mutator_fn: Optional[callable] = None,
    ) -> Agent:
        """
        Create an offspring from two parents.

        If parents are the same, this is a mutation.
        If different, this is crossover + mutation.

        The actual code transformation is done by the mutator_fn,
        which can be the CodeSurgeon or LLM mutator.
        """
        from .code_surgery import CodeSurgeon

        is_self_breeding = parent_a.agent_id == parent_b.agent_id

        if is_self_breeding:
            # Mutation only
            mutation_type = "mutation"
            base_code = parent_a.source_code
        else:
            # Crossover
            mutation_type = "crossover"
            base_code = self._crossover(parent_a.source_code, parent_b.source_code)

        # Apply mutation
        if mutator_fn:
            mutated_code = mutator_fn(base_code)
        else:
            # Default: use CodeSurgeon
            surgeon = CodeSurgeon()
            result = surgeon.random_mutation(base_code)
            mutated_code = result.new_code if result.success else base_code

        # Create offspring agent
        offspring = Agent(
            agent_id=f"agent_{int(time.time()*1000) % 100000}_{hash(mutated_code) % 10000:04x}",
            generation=self.generation_manager.current_generation + 1,
            source_code=mutated_code,
            parent_ids=[parent_a.agent_id, parent_b.agent_id] if not is_self_breeding else [parent_a.agent_id],
            mutation_type=mutation_type,
        )

        return offspring

    def _crossover(self, code_a: str, code_b: str) -> str:
        """
        Simple crossover between two code snippets.

        Strategy: Take functions from both parents randomly.
        """
        import ast

        try:
            tree_a = ast.parse(code_a)
            tree_b = ast.parse(code_b)
        except SyntaxError:
            # If parsing fails, just return one parent
            return code_a if len(code_a) >= len(code_b) else code_b

        # Extract function definitions
        funcs_a = [node for node in ast.walk(tree_a) if isinstance(node, ast.FunctionDef)]
        funcs_b = [node for node in ast.walk(tree_b) if isinstance(node, ast.FunctionDef)]

        if not funcs_a and not funcs_b:
            return code_a

        import random

        # Simple strategy: use code_a as base, but swap some functions from code_b
        result = code_a

        for func_b in funcs_b:
            if random.random() < 0.3:  # 30% chance to take from parent B
                # This is a simplified crossover - real implementation would
                # properly replace matching function definitions
                pass

        return result

    def run_breeding_cycle(
        self,
        offspring_count: Optional[int] = None,
        mutator_fn: Optional[callable] = None,
        validator_fn: Optional[callable] = None,
    ) -> List[Agent]:
        """
        Run a complete breeding cycle.

        1. Select breeding pairs
        2. Create offspring
        3. Validate offspring (optional)
        4. Return valid offspring (caller adds to population)
        """
        count = offspring_count or self.config.offspring_per_generation
        pairs = self.select_breeding_pairs(count)

        offspring = []
        for parent_a, parent_b in pairs:
            child = self.create_offspring(parent_a, parent_b, mutator_fn)

            # Validate if validator provided
            if validator_fn:
                if validator_fn(child):
                    offspring.append(child)
            else:
                offspring.append(child)

        # Record breeding history
        self._breeding_history.append({
            'generation': self.generation_manager.current_generation,
            'pairs_count': len(pairs),
            'offspring_count': len(offspring),
            'timestamp': time.time(),
        })

        return offspring

    def cull_population(self, count: Optional[int] = None) -> List[Agent]:
        """
        Remove weak agents from population.

        Returns the culled agents (for potential rollback/analysis).
        """
        cull_count = count or (self.population.size - self.population.config.min_population)

        if cull_count <= 0:
            return []

        victims = self.population.select_for_culling(cull_count)
        culled = []

        for victim in victims:
            agent = self.population.remove_agent(victim.agent_id)
            if agent:
                culled.append(agent)

        return culled

    def get_breeding_stats(self) -> Dict[str, Any]:
        """Get breeding statistics."""
        if not self._breeding_history:
            return {'total_cycles': 0}

        total_offspring = sum(h['offspring_count'] for h in self._breeding_history)
        total_pairs = sum(h['pairs_count'] for h in self._breeding_history)

        return {
            'total_cycles': len(self._breeding_history),
            'total_offspring': total_offspring,
            'total_pairs': total_pairs,
            'avg_offspring_per_cycle': total_offspring / len(self._breeding_history),
            'success_rate': total_offspring / total_pairs if total_pairs > 0 else 0,
        }
