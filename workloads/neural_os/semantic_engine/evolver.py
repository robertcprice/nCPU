#!/usr/bin/env python3
"""
EVOLVER: Genetic Evolution of RL Policies (EvoRL)

Grok's recommendation:
"EvoRL: Genetic algos evolve RL policies; reward = semantic novelty + compression ratio"

This module adds genetic algorithms to evolve better synthesis policies,
combining the exploration power of evolution with the optimization power of RL.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
import random
import copy
import numpy as np
from sympy import Symbol, Expr, Add, Mul, Integer
from collections import defaultdict


@dataclass
class Gene:
    """A single gene encoding a synthesis behavior."""
    action_weights: List[float]  # Weights for action selection
    temperature: float  # Exploration temperature
    composition_preference: float  # Prefer composition vs application
    mdl_weight: float  # How much to weight compression
    novelty_weight: float  # How much to weight novelty
    mutation_rate: float  # Self-adaptive mutation rate

    def mutate(self) -> 'Gene':
        """Mutate this gene."""
        new_weights = [
            w + random.gauss(0, self.mutation_rate)
            for w in self.action_weights
        ]
        # Normalize weights
        total = sum(abs(w) for w in new_weights)
        if total > 0:
            new_weights = [w / total for w in new_weights]

        return Gene(
            action_weights=new_weights,
            temperature=max(0.1, self.temperature + random.gauss(0, self.mutation_rate)),
            composition_preference=max(0, min(1, self.composition_preference + random.gauss(0, self.mutation_rate))),
            mdl_weight=max(0, self.mdl_weight + random.gauss(0, self.mutation_rate)),
            novelty_weight=max(0, self.novelty_weight + random.gauss(0, self.mutation_rate)),
            mutation_rate=max(0.01, min(0.5, self.mutation_rate + random.gauss(0, 0.01)))
        )

    @classmethod
    def random(cls, num_actions: int = 32) -> 'Gene':
        """Create a random gene."""
        weights = [random.random() for _ in range(num_actions)]
        total = sum(weights)
        weights = [w / total for w in weights]
        return cls(
            action_weights=weights,
            temperature=random.uniform(0.1, 2.0),
            composition_preference=random.random(),
            mdl_weight=random.random(),
            novelty_weight=random.random(),
            mutation_rate=random.uniform(0.01, 0.2)
        )


@dataclass
class Chromosome:
    """A chromosome encoding a complete synthesis strategy."""
    genes: List[Gene]
    fitness: float = 0.0
    novelty_score: float = 0.0
    compression_score: float = 0.0
    discoveries: List[str] = field(default_factory=list)

    def crossover(self, other: 'Chromosome') -> 'Chromosome':
        """Crossover with another chromosome."""
        # Single-point crossover
        point = random.randint(1, len(self.genes) - 1)
        new_genes = self.genes[:point] + other.genes[point:]
        return Chromosome(genes=[g.mutate() for g in new_genes])

    def mutate(self) -> 'Chromosome':
        """Mutate this chromosome."""
        return Chromosome(genes=[g.mutate() for g in self.genes])

    @classmethod
    def random(cls, num_genes: int = 5, num_actions: int = 32) -> 'Chromosome':
        """Create a random chromosome."""
        return cls(genes=[Gene.random(num_actions) for _ in range(num_genes)])


class NoveltyArchive:
    """Archive of novel behaviors for novelty search."""

    def __init__(self, max_size: int = 1000):
        self.behaviors: List[Tuple[str, float]] = []
        self.max_size = max_size
        self.behavior_hashes: set = set()

    def add(self, behavior_desc: str, fitness: float):
        """Add a behavior to the archive."""
        h = hash(behavior_desc)
        if h not in self.behavior_hashes:
            self.behaviors.append((behavior_desc, fitness))
            self.behavior_hashes.add(h)
            if len(self.behaviors) > self.max_size:
                # Remove oldest
                old_desc, _ = self.behaviors.pop(0)
                self.behavior_hashes.discard(hash(old_desc))

    def novelty_score(self, behavior_desc: str, k: int = 15) -> float:
        """Compute novelty of a behavior (distance to k-nearest neighbors)."""
        if len(self.behaviors) == 0:
            return 1.0

        # Use string edit distance as distance metric
        distances = []
        for archived_desc, _ in self.behaviors:
            dist = self._edit_distance(behavior_desc, archived_desc)
            distances.append(dist)

        distances.sort()
        k_nearest = distances[:min(k, len(distances))]
        return sum(k_nearest) / len(k_nearest) if k_nearest else 1.0

    def _edit_distance(self, s1: str, s2: str) -> float:
        """Simple edit distance (normalized)."""
        if len(s1) == 0 and len(s2) == 0:
            return 0.0
        # Simplified: use set difference
        set1, set2 = set(s1.split()), set(s2.split())
        union = set1 | set2
        intersection = set1 & set2
        if len(union) == 0:
            return 0.0
        return 1.0 - len(intersection) / len(union)


class EvoRL:
    """
    Evolutionary Reinforcement Learning for Synthesis Policies.

    Combines:
    - Genetic algorithms for exploration
    - Novelty search for diversity
    - MDL compression for quality
    - RL for fine-tuning
    """

    def __init__(
        self,
        population_size: int = 50,
        num_genes: int = 5,
        num_actions: int = 32,
        elite_fraction: float = 0.1,
        tournament_size: int = 3
    ):
        self.population_size = population_size
        self.num_genes = num_genes
        self.num_actions = num_actions
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size

        # Initialize population
        self.population = [
            Chromosome.random(num_genes, num_actions)
            for _ in range(population_size)
        ]

        # Novelty archive
        self.archive = NoveltyArchive()

        # Statistics
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.discoveries: List[str] = []

    def evaluate_chromosome(
        self,
        chromosome: Chromosome,
        synthesis_fn: Callable[[Chromosome, Expr], Optional[Expr]],
        test_cases: List[Tuple[Expr, Expr]]
    ) -> float:
        """Evaluate a chromosome on synthesis tasks."""
        total_score = 0.0
        behaviors = []

        for input_expr, target_expr in test_cases:
            result = synthesis_fn(chromosome, input_expr)

            if result is not None:
                # Correctness check
                correct = result == target_expr

                # Compression score (shorter is better)
                compression = 1.0 / (1 + len(str(result)))

                # Create behavior description
                behavior = f"{input_expr} -> {result}"
                behaviors.append(behavior)

                # Combined score
                score = (1.0 if correct else 0.0) + 0.3 * compression
                total_score += score

        # Novelty score
        behavior_desc = " | ".join(behaviors)
        novelty = self.archive.novelty_score(behavior_desc)

        # Combined fitness: correctness + novelty + compression
        chromosome.fitness = total_score / len(test_cases)
        chromosome.novelty_score = novelty

        # Add to archive if novel enough
        if novelty > 0.3:
            self.archive.add(behavior_desc, chromosome.fitness)

        # Combined with novelty bonus
        return chromosome.fitness + 0.2 * novelty

    def select_tournament(self) -> Chromosome:
        """Tournament selection."""
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda c: c.fitness + 0.2 * c.novelty_score)

    def evolve_generation(
        self,
        synthesis_fn: Callable[[Chromosome, Expr], Optional[Expr]],
        test_cases: List[Tuple[Expr, Expr]]
    ):
        """Evolve one generation."""
        # Evaluate all chromosomes
        for chromosome in self.population:
            self.evaluate_chromosome(chromosome, synthesis_fn, test_cases)

        # Sort by combined score
        self.population.sort(key=lambda c: c.fitness + 0.2 * c.novelty_score, reverse=True)

        # Track statistics
        best_fitness = self.population[0].fitness
        avg_fitness = sum(c.fitness for c in self.population) / len(self.population)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Track diversity (average pairwise distance in novelty archive)
        if len(self.archive.behaviors) > 1:
            diversity = len(self.archive.behaviors) / self.archive.max_size
        else:
            diversity = 1.0
        self.diversity_history.append(diversity)

        # Elitism: keep top performers
        num_elite = max(1, int(self.population_size * self.elite_fraction))
        new_population = self.population[:num_elite]

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < 0.7:
                # Crossover
                parent1 = self.select_tournament()
                parent2 = self.select_tournament()
                child = parent1.crossover(parent2)
            else:
                # Mutation only
                parent = self.select_tournament()
                child = parent.mutate()

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return best_fitness, avg_fitness, diversity

    def get_best_policy(self) -> Chromosome:
        """Get the best chromosome (policy)."""
        return max(self.population, key=lambda c: c.fitness + 0.2 * c.novelty_score)

    def chromosome_to_action_probs(self, chromosome: Chromosome) -> np.ndarray:
        """Convert chromosome to action probabilities."""
        # Average the gene weights
        probs = np.zeros(self.num_actions)
        for gene in chromosome.genes:
            probs += np.array(gene.action_weights)
        probs /= len(chromosome.genes)

        # Apply temperature from first gene
        temp = chromosome.genes[0].temperature
        probs = np.exp(probs / temp)
        probs /= probs.sum()

        return probs


class SynthesisPolicyEvolver:
    """
    High-level interface for evolving synthesis policies.
    Integrates with the Meta-Cognitive Orchestrator.
    """

    def __init__(self, num_actions: int = 32):
        self.evolver = EvoRL(
            population_size=50,
            num_genes=5,
            num_actions=num_actions
        )

        # Define synthesis operations
        self.operations = {
            'identity': lambda x: x,
            'double': lambda x: 2 * x,
            'square': lambda x: x * x,
            'add_one': lambda x: x + 1,
            'negate': lambda x: -x,
        }

        # Best discovered policies
        self.elite_policies: List[Chromosome] = []

    def synthesize(self, chromosome: Chromosome, input_expr: Expr) -> Optional[Expr]:
        """Attempt synthesis using chromosome's policy."""
        action_probs = self.evolver.chromosome_to_action_probs(chromosome)

        # Sample operations based on weights
        op_names = list(self.operations.keys())

        try:
            # Apply weighted combination of operations
            result = input_expr
            for i, (op_name, op_fn) in enumerate(self.operations.items()):
                if i < len(action_probs):
                    weight = action_probs[i % len(action_probs)]
                    if random.random() < weight:
                        result = op_fn(result)
            return result
        except Exception:
            return None

    def evolve(self, generations: int = 100) -> Dict[str, Any]:
        """Run evolution for specified generations."""
        # Create test cases
        x = Symbol('x')
        test_cases = [
            (x, x * x),  # square
            (x, 2 * x),  # double
            (x, x + 1),  # add_one
            (Integer(3), Integer(6)),  # double 3
            (Integer(2), Integer(4)),  # square 2
        ]

        results = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity': [],
            'final_best': None
        }

        for gen in range(generations):
            best, avg, div = self.evolver.evolve_generation(
                self.synthesize,
                test_cases
            )

            results['generations'].append(gen)
            results['best_fitness'].append(best)
            results['avg_fitness'].append(avg)
            results['diversity'].append(div)

            if gen % 10 == 0:
                print(f"  Gen {gen}: best={best:.3f}, avg={avg:.3f}, diversity={div:.3f}")

        results['final_best'] = self.evolver.get_best_policy()

        # Store elite policies
        self.elite_policies = sorted(
            self.evolver.population,
            key=lambda c: c.fitness,
            reverse=True
        )[:5]

        return results


# =============================================================================
# INTEGRATION WITH MCO
# =============================================================================

def get_evolved_policy_weights(evolver: SynthesisPolicyEvolver) -> np.ndarray:
    """Get action weights from the best evolved policy."""
    if evolver.elite_policies:
        return evolver.evolver.chromosome_to_action_probs(evolver.elite_policies[0])
    return None


def merge_evolved_with_rl(
    evolved_weights: np.ndarray,
    rl_weights: np.ndarray,
    evolution_factor: float = 0.3
) -> np.ndarray:
    """Merge evolved policy weights with RL policy weights."""
    return (1 - evolution_factor) * rl_weights + evolution_factor * evolved_weights


# =============================================================================
# MAIN: Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EVOLVER: Genetic Evolution of RL Policies")
    print("=" * 60)

    # Create evolver
    evolver = SynthesisPolicyEvolver(num_actions=32)

    print("\n[1] Running evolution (100 generations)...")
    results = evolver.evolve(generations=100)

    print("\n[2] Evolution Results:")
    print(f"  Final best fitness: {results['best_fitness'][-1]:.3f}")
    print(f"  Final avg fitness: {results['avg_fitness'][-1]:.3f}")
    print(f"  Final diversity: {results['diversity'][-1]:.3f}")

    print("\n[3] Best Policy Genes:")
    best = results['final_best']
    for i, gene in enumerate(best.genes):
        print(f"  Gene {i}: temp={gene.temperature:.2f}, "
              f"mdl_w={gene.mdl_weight:.2f}, nov_w={gene.novelty_weight:.2f}")

    print("\n[4] Action Probabilities from Best Policy:")
    probs = evolver.evolver.chromosome_to_action_probs(best)
    top_actions = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
    for idx, prob in top_actions:
        print(f"  Action {idx}: {prob:.3f}")

    print("\n[5] Elite Policies:")
    for i, elite in enumerate(evolver.elite_policies):
        print(f"  Elite {i}: fitness={elite.fitness:.3f}, novelty={elite.novelty_score:.3f}")

    print("\n✅ EvoRL ready for integration with MCO")
