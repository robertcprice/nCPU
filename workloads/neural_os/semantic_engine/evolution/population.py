#!/usr/bin/env python3
"""
THE POPULATION - Agent Management for the Ouroboros

Manages a population of 5-20 competing agents, each with:
- Source code (the actual implementation)
- Fitness score (from benchmarks)
- Lineage (parent agents)
- Generation number

Tournament selection:
- Top N agents breed/mutate to create offspring
- Bottom M agents are culled
- Population maintains diversity via Shannon entropy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import hashlib
import time
import json
import random
import math
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """
    A single agent in the population.

    Each agent represents a complete implementation that can be
    benchmarked and compared against other agents.
    """
    agent_id: str
    generation: int
    source_code: str
    fitness_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    parent_ids: List[str] = field(default_factory=list)
    mutation_type: str = "genesis"  # genesis, crossover, mutation, llm_mutation
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_hash(self) -> str:
        """Cryptographic hash of source code."""
        return hashlib.sha256(self.source_code.encode()).hexdigest()[:16]

    @property
    def code_length(self) -> int:
        """Length of source code in characters."""
        return len(self.source_code)

    @property
    def line_count(self) -> int:
        """Number of lines in source code."""
        return len(self.source_code.split('\n'))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'generation': self.generation,
            'source_code': self.source_code,
            'source_hash': self.source_hash,
            'fitness_score': self.fitness_score,
            'created_at': self.created_at,
            'parent_ids': self.parent_ids,
            'mutation_type': self.mutation_type,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        # Remove computed properties
        data = {k: v for k, v in data.items() if k != 'source_hash'}
        return cls(**data)

    @classmethod
    def create_genesis(cls, source_code: str, agent_id: Optional[str] = None) -> 'Agent':
        """Create a genesis agent (first generation, no parents)."""
        return cls(
            agent_id=agent_id or f"agent_{uuid.uuid4().hex[:8]}",
            generation=0,
            source_code=source_code,
            mutation_type="genesis",
        )


@dataclass
class PopulationConfig:
    """Configuration for population management."""

    min_population: int = 5
    max_population: int = 20

    # Tournament settings
    tournament_size: int = 3  # Number of agents per tournament
    selection_pressure: float = 0.7  # Probability of selecting better agent

    # Diversity
    min_shannon_entropy: float = 0.3
    diversity_bonus: float = 0.1  # Bonus fitness for unique solutions

    # Breeding
    crossover_rate: float = 0.3
    mutation_rate: float = 0.6
    llm_mutation_rate: float = 0.1  # Use LLM for mutation

    # Storage
    population_dir: Path = field(default_factory=lambda: Path("/tmp/ouroboros/population"))

    def __post_init__(self):
        if isinstance(self.population_dir, str):
            self.population_dir = Path(self.population_dir)


class Population:
    """
    Manages the population of competing agents.

    Key operations:
    - add_agent: Add a new agent (must pass Judge first)
    - remove_agent: Remove an agent (culling)
    - select_parents: Tournament selection for breeding
    - get_diverse_sample: Get agents for diversity
    - compute_entropy: Measure population diversity
    """

    def __init__(self, config: PopulationConfig):
        self.config = config
        self._agents: Dict[str, Agent] = {}
        self._fitness_history: List[Dict[str, float]] = []

        # Ensure directory exists
        self.config.population_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Population initialized: min={config.min_population}, max={config.max_population}")

    @property
    def size(self) -> int:
        """Current population size."""
        return len(self._agents)

    @property
    def agents(self) -> List[Agent]:
        """List of all agents."""
        return list(self._agents.values())

    @property
    def best_agent(self) -> Optional[Agent]:
        """Agent with highest fitness."""
        if not self._agents:
            return None
        return max(self._agents.values(), key=lambda a: a.fitness_score)

    @property
    def worst_agent(self) -> Optional[Agent]:
        """Agent with lowest fitness."""
        if not self._agents:
            return None
        return min(self._agents.values(), key=lambda a: a.fitness_score)

    def add_agent(self, agent: Agent) -> bool:
        """
        Add an agent to the population.

        Returns False if population is at maximum capacity.
        """
        if self.size >= self.config.max_population:
            logger.warning(f"Population at maximum capacity ({self.config.max_population})")
            return False

        self._agents[agent.agent_id] = agent

        # Save to disk
        self._save_agent(agent)

        logger.info(f"Added agent {agent.agent_id} (gen={agent.generation}, fitness={agent.fitness_score:.4f})")
        return True

    def remove_agent(self, agent_id: str) -> Optional[Agent]:
        """Remove an agent from the population."""
        if agent_id not in self._agents:
            return None

        agent = self._agents.pop(agent_id)

        # Remove from disk (but keep in rollback)
        agent_path = self.config.population_dir / f"{agent_id}.json"
        if agent_path.exists():
            agent_path.unlink()

        logger.info(f"Removed agent {agent_id}")
        return agent

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def update_fitness(self, agent_id: str, fitness: float) -> bool:
        """Update an agent's fitness score."""
        if agent_id not in self._agents:
            return False

        self._agents[agent_id].fitness_score = fitness
        self._save_agent(self._agents[agent_id])
        return True

    def _save_agent(self, agent: Agent) -> None:
        """Save agent to disk."""
        agent_path = self.config.population_dir / f"{agent.agent_id}.json"
        agent_path.write_text(json.dumps(agent.to_dict(), indent=2))

    def load_from_disk(self) -> int:
        """Load population from disk. Returns number loaded."""
        count = 0
        for agent_file in self.config.population_dir.glob("*.json"):
            try:
                data = json.loads(agent_file.read_text())
                agent = Agent.from_dict(data)
                self._agents[agent.agent_id] = agent
                count += 1
            except Exception as e:
                logger.error(f"Failed to load {agent_file}: {e}")

        logger.info(f"Loaded {count} agents from disk")
        return count

    def select_parents(self, count: int = 2) -> List[Agent]:
        """
        Select parents for breeding using tournament selection.

        Tournament selection:
        1. Randomly pick N agents
        2. Select the best one with probability P
        3. Otherwise select randomly from the rest
        """
        if self.size < count:
            return self.agents

        parents = []
        available = list(self._agents.values())

        for _ in range(count):
            if len(available) < self.config.tournament_size:
                break

            # Random tournament
            tournament = random.sample(available, min(self.config.tournament_size, len(available)))
            tournament.sort(key=lambda a: a.fitness_score, reverse=True)

            # Selection pressure
            if random.random() < self.config.selection_pressure:
                selected = tournament[0]
            else:
                selected = random.choice(tournament[1:]) if len(tournament) > 1 else tournament[0]

            parents.append(selected)
            available.remove(selected)

        return parents

    def select_for_culling(self, count: int = 1) -> List[Agent]:
        """
        Select agents to be removed (inverse tournament selection).

        Preferentially selects low-fitness agents while maintaining
        minimum population size.
        """
        if self.size <= self.config.min_population:
            return []

        max_cull = self.size - self.config.min_population
        count = min(count, max_cull)

        # Sort by fitness (ascending - worst first)
        ranked = sorted(self._agents.values(), key=lambda a: a.fitness_score)

        # Select from bottom half with bias toward worst
        cull_pool = ranked[:max(1, len(ranked) // 2)]
        selected = []

        for _ in range(count):
            if not cull_pool:
                break

            # Inverse tournament - select worst with higher probability
            tournament = random.sample(cull_pool, min(self.config.tournament_size, len(cull_pool)))
            tournament.sort(key=lambda a: a.fitness_score)

            if random.random() < self.config.selection_pressure:
                victim = tournament[0]  # Worst in tournament
            else:
                victim = random.choice(tournament)

            selected.append(victim)
            cull_pool.remove(victim)

        return selected

    def compute_shannon_entropy(self) -> float:
        """
        Compute Shannon entropy of the population based on source hashes.

        Higher entropy = more diverse population.
        Entropy of 0 = all agents identical.
        Entropy of 1 = all agents unique.
        """
        if not self._agents:
            return 0.0

        # Count occurrences of each unique source hash
        hash_counts: Dict[str, int] = {}
        for agent in self._agents.values():
            h = agent.source_hash
            hash_counts[h] = hash_counts.get(h, 0) + 1

        # Compute entropy: -sum(p * log2(p))
        total = len(self._agents)
        entropy = 0.0

        for count in hash_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        max_entropy = math.log2(total) if total > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_diversity_metrics(self) -> Dict[str, Any]:
        """Get detailed diversity metrics."""
        if not self._agents:
            return {'entropy': 0.0, 'unique_count': 0, 'total_count': 0}

        hashes = [a.source_hash for a in self._agents.values()]
        unique_hashes = set(hashes)

        return {
            'entropy': self.compute_shannon_entropy(),
            'unique_count': len(unique_hashes),
            'total_count': len(self._agents),
            'unique_ratio': len(unique_hashes) / len(self._agents),
            'is_diverse': self.compute_shannon_entropy() >= self.config.min_shannon_entropy,
        }

    def get_fitness_stats(self) -> Dict[str, float]:
        """Get fitness statistics."""
        if not self._agents:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}

        fitnesses = [a.fitness_score for a in self._agents.values()]
        mean = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)

        return {
            'min': min(fitnesses),
            'max': max(fitnesses),
            'mean': mean,
            'std': math.sqrt(variance),
            'count': len(fitnesses),
        }

    def rank_agents(self) -> List[Tuple[int, Agent]]:
        """Return agents ranked by fitness (1 = best)."""
        sorted_agents = sorted(self._agents.values(), key=lambda a: a.fitness_score, reverse=True)
        return [(i + 1, agent) for i, agent in enumerate(sorted_agents)]

    def get_generation_histogram(self) -> Dict[int, int]:
        """Count agents by generation."""
        histogram: Dict[int, int] = {}
        for agent in self._agents.values():
            gen = agent.generation
            histogram[gen] = histogram.get(gen, 0) + 1
        return dict(sorted(histogram.items()))

    def record_fitness_snapshot(self) -> None:
        """Record current fitness distribution for history."""
        snapshot = {a.agent_id: a.fitness_score for a in self._agents.values()}
        self._fitness_history.append(snapshot)

        # Keep last 1000 snapshots
        if len(self._fitness_history) > 1000:
            self._fitness_history = self._fitness_history[-1000:]

    def get_improvement_rate(self, window: int = 10) -> float:
        """
        Compute rate of fitness improvement over last N snapshots.

        Returns positive value if improving, negative if declining.
        """
        if len(self._fitness_history) < 2:
            return 0.0

        recent = self._fitness_history[-min(window, len(self._fitness_history)):]

        # Compute mean fitness for each snapshot
        means = []
        for snapshot in recent:
            if snapshot:
                means.append(sum(snapshot.values()) / len(snapshot))

        if len(means) < 2:
            return 0.0

        # Linear regression slope
        n = len(means)
        x_mean = (n - 1) / 2
        y_mean = sum(means) / n

        numerator = sum((i - x_mean) * (means[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def is_stagnant(self, threshold: float = 0.001, window: int = 20) -> bool:
        """Check if population improvement has stagnated."""
        rate = self.get_improvement_rate(window)
        return abs(rate) < threshold

    def clear(self) -> None:
        """Clear all agents from population."""
        self._agents.clear()
        self._fitness_history.clear()

        # Clear disk
        for f in self.config.population_dir.glob("*.json"):
            f.unlink()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        return {
            'size': self.size,
            'min_size': self.config.min_population,
            'max_size': self.config.max_population,
            'fitness': self.get_fitness_stats(),
            'diversity': self.get_diversity_metrics(),
            'generations': self.get_generation_histogram(),
            'improvement_rate': self.get_improvement_rate(),
            'is_stagnant': self.is_stagnant(),
            'best_agent_id': self.best_agent.agent_id if self.best_agent else None,
            'worst_agent_id': self.worst_agent.agent_id if self.worst_agent else None,
        }
