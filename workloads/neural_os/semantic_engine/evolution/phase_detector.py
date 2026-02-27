#!/usr/bin/env python3
"""
PHASE DETECTOR - Population Dynamics Analysis for Ouroboros

Detects the evolutionary phase of the population and recommends strategies:
- DISORDERED: Random mutations, no structure → Need exploitation
- ORDERED: Converged to local optimum → Need exploration
- CRITICAL: Edge of chaos, maximum adaptability → Maintain balance
- GLASSY: Trapped in near-optimal states → Need perturbation

Based on DeepSeek's hybrid review recommendations on emergent computation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EvolutionaryPhase(Enum):
    """The four phases of evolutionary dynamics."""
    DISORDERED = "disordered"   # Random, chaotic - need more exploitation
    ORDERED = "ordered"         # Converged, stuck - need more exploration
    CRITICAL = "critical"       # Edge of chaos - optimal state, maintain
    GLASSY = "glassy"           # Trapped in many near-optima - need perturbation
    UNKNOWN = "unknown"         # Insufficient data


@dataclass
class PhaseMetrics:
    """Metrics used to determine evolutionary phase."""
    diversity_entropy: float = 0.0          # Shannon entropy of population
    fitness_correlation: float = 0.0        # Correlation between fitness values
    innovation_rate: float = 0.0            # Rate of new unique solutions
    fitness_improvement_rate: float = 0.0   # Recent fitness improvement rate
    stagnation_generations: int = 0         # Generations without improvement
    population_spread: float = 0.0          # Std dev of fitness
    unique_solution_ratio: float = 0.0      # Unique solutions / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            'diversity_entropy': self.diversity_entropy,
            'fitness_correlation': self.fitness_correlation,
            'innovation_rate': self.innovation_rate,
            'fitness_improvement_rate': self.fitness_improvement_rate,
            'stagnation_generations': self.stagnation_generations,
            'population_spread': self.population_spread,
            'unique_solution_ratio': self.unique_solution_ratio,
        }


@dataclass
class PhaseStrategy:
    """Recommended strategy for the current phase."""
    phase: EvolutionaryPhase
    confidence: float

    # Strategy parameters
    exploration_rate: float = 0.5           # 0.0 = pure exploitation, 1.0 = pure exploration
    mutation_intensity: float = 0.5         # How aggressive mutations should be
    llm_usage_rate: float = 0.5             # How often to use LLM vs AST
    tournament_pressure: float = 0.5        # Selection pressure (0 = weak, 1 = strong)
    shock_probability: float = 0.0          # Probability of random shock/perturbation

    # Actions
    inject_diversity: bool = False          # Should we inject new random agents?
    focus_on_best: bool = False             # Should we clone and mutate best only?
    increase_population: bool = False       # Should we grow the population?
    prune_similar: bool = False             # Should we remove similar agents?

    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase': self.phase.value,
            'confidence': self.confidence,
            'exploration_rate': self.exploration_rate,
            'mutation_intensity': self.mutation_intensity,
            'llm_usage_rate': self.llm_usage_rate,
            'tournament_pressure': self.tournament_pressure,
            'shock_probability': self.shock_probability,
            'inject_diversity': self.inject_diversity,
            'focus_on_best': self.focus_on_best,
            'increase_population': self.increase_population,
            'prune_similar': self.prune_similar,
            'message': self.message,
        }


@dataclass
class PhaseDetectorConfig:
    """Configuration for phase detection."""

    # Thresholds for phase detection
    disordered_entropy_threshold: float = 0.85      # High entropy = disordered
    ordered_entropy_threshold: float = 0.25         # Low entropy = ordered
    stagnation_threshold: int = 5                   # Generations without improvement
    glassy_threshold: float = 0.1                   # Low improvement but high diversity

    # History tracking
    fitness_history_size: int = 20                  # Generations to track
    innovation_window: int = 10                     # Window for innovation rate

    # Adaptive parameters
    adaptive_thresholds: bool = True                # Adjust thresholds over time


class PhaseDetector:
    """
    Detects the evolutionary phase of a population.

    Uses multiple metrics to determine if the population is:
    - DISORDERED: High diversity, low fitness correlation, random exploration
    - ORDERED: Low diversity, high fitness correlation, converged
    - CRITICAL: Balanced state, maximum adaptability
    - GLASSY: High diversity but stuck in local optima
    """

    def __init__(self, config: PhaseDetectorConfig = None):
        self.config = config or PhaseDetectorConfig()

        # History tracking
        self._fitness_history: deque = deque(maxlen=self.config.fitness_history_size)
        self._best_fitness_history: deque = deque(maxlen=self.config.fitness_history_size)
        self._diversity_history: deque = deque(maxlen=self.config.fitness_history_size)
        self._innovation_history: deque = deque(maxlen=self.config.innovation_window)
        self._unique_solutions_seen: set = set()

        # Current state
        self._current_phase: EvolutionaryPhase = EvolutionaryPhase.UNKNOWN
        self._current_metrics: PhaseMetrics = PhaseMetrics()
        self._generations_analyzed: int = 0

        logger.info("PhaseDetector initialized")

    def update(
        self,
        fitness_values: List[float],
        solution_hashes: List[str],
        best_fitness: float,
    ) -> None:
        """
        Update the detector with new generation data.

        Args:
            fitness_values: Fitness scores of all agents
            solution_hashes: Unique hashes of each solution (for diversity)
            best_fitness: Best fitness in this generation
        """
        self._generations_analyzed += 1

        # Track fitness history
        if fitness_values:
            mean_fitness = sum(fitness_values) / len(fitness_values)
            self._fitness_history.append(mean_fitness)
            self._best_fitness_history.append(best_fitness)

        # Track diversity
        unique_count = len(set(solution_hashes))
        total_count = len(solution_hashes)
        diversity = unique_count / max(total_count, 1)
        self._diversity_history.append(diversity)

        # Track innovation (new solutions)
        new_solutions = sum(1 for h in solution_hashes if h not in self._unique_solutions_seen)
        self._innovation_history.append(new_solutions)
        self._unique_solutions_seen.update(solution_hashes)

    def compute_metrics(self, fitness_values: List[float], solution_hashes: List[str]) -> PhaseMetrics:
        """Compute all phase detection metrics."""
        metrics = PhaseMetrics()

        if not fitness_values:
            return metrics

        n = len(fitness_values)

        # 1. Diversity entropy (Shannon entropy normalized to 0-1)
        if solution_hashes:
            unique = len(set(solution_hashes))
            metrics.unique_solution_ratio = unique / n
            # Shannon entropy of solution distribution
            if unique > 1:
                # Simple entropy based on unique ratio
                p = metrics.unique_solution_ratio
                if 0 < p < 1:
                    metrics.diversity_entropy = -p * math.log2(p) - (1-p) * math.log2(1-p)
                else:
                    metrics.diversity_entropy = 0.0 if p == 0 else 1.0
            else:
                metrics.diversity_entropy = 0.0

        # 2. Fitness correlation (how similar are fitness values?)
        mean_fitness = sum(fitness_values) / n
        if n > 1:
            variance = sum((f - mean_fitness) ** 2 for f in fitness_values) / (n - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 0
            metrics.population_spread = std_dev

            # Normalized correlation (0 = all same, 1 = high variance)
            max_possible_spread = (max(fitness_values) - min(fitness_values)) / 2
            if max_possible_spread > 0:
                metrics.fitness_correlation = 1 - min(std_dev / max_possible_spread, 1.0)
            else:
                metrics.fitness_correlation = 1.0  # All same = high correlation
        else:
            metrics.fitness_correlation = 1.0
            metrics.population_spread = 0.0

        # 3. Innovation rate (from history)
        if self._innovation_history:
            metrics.innovation_rate = sum(self._innovation_history) / len(self._innovation_history)

        # 4. Fitness improvement rate
        if len(self._best_fitness_history) >= 2:
            recent = list(self._best_fitness_history)[-5:]
            if len(recent) >= 2:
                improvements = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                metrics.fitness_improvement_rate = sum(improvements) / len(improvements)

        # 5. Stagnation detection
        if len(self._best_fitness_history) >= 2:
            best_ever = max(self._best_fitness_history)
            stagnant = 0
            for bf in reversed(list(self._best_fitness_history)):
                if bf >= best_ever * 0.999:  # Within 0.1% of best
                    stagnant += 1
                else:
                    break
            metrics.stagnation_generations = stagnant

        self._current_metrics = metrics
        return metrics

    def detect_phase(
        self,
        fitness_values: List[float],
        solution_hashes: List[str],
    ) -> Tuple[EvolutionaryPhase, float]:
        """
        Detect the current evolutionary phase.

        Returns:
            Tuple of (phase, confidence)
        """
        metrics = self.compute_metrics(fitness_values, solution_hashes)

        # Need some data to make a decision
        if self._generations_analyzed < 3:
            return EvolutionaryPhase.UNKNOWN, 0.0

        # Decision logic based on metrics

        # DISORDERED: High diversity, low correlation, high innovation
        if (metrics.diversity_entropy > self.config.disordered_entropy_threshold and
            metrics.fitness_correlation < 0.5 and
            metrics.innovation_rate > 0.5):
            confidence = min(metrics.diversity_entropy, 1.0)
            self._current_phase = EvolutionaryPhase.DISORDERED
            return EvolutionaryPhase.DISORDERED, confidence

        # ORDERED: Low diversity, high correlation, low innovation
        if (metrics.diversity_entropy < self.config.ordered_entropy_threshold and
            metrics.fitness_correlation > 0.7):
            confidence = 1.0 - metrics.diversity_entropy
            self._current_phase = EvolutionaryPhase.ORDERED
            return EvolutionaryPhase.ORDERED, confidence

        # GLASSY: High diversity but stagnant
        if (metrics.stagnation_generations >= self.config.stagnation_threshold and
            metrics.diversity_entropy > 0.4 and
            metrics.fitness_improvement_rate < self.config.glassy_threshold):
            confidence = min(metrics.stagnation_generations / 10.0, 1.0)
            self._current_phase = EvolutionaryPhase.GLASSY
            return EvolutionaryPhase.GLASSY, confidence

        # CRITICAL: Balanced state
        if (0.3 <= metrics.diversity_entropy <= 0.7 and
            0.3 <= metrics.fitness_correlation <= 0.7 and
            metrics.fitness_improvement_rate >= 0):
            confidence = 1.0 - abs(0.5 - metrics.diversity_entropy) * 2
            self._current_phase = EvolutionaryPhase.CRITICAL
            return EvolutionaryPhase.CRITICAL, confidence

        # Default to CRITICAL with low confidence if unclear
        self._current_phase = EvolutionaryPhase.CRITICAL
        return EvolutionaryPhase.CRITICAL, 0.3

    def get_strategy(
        self,
        fitness_values: List[float],
        solution_hashes: List[str],
    ) -> PhaseStrategy:
        """
        Get recommended strategy based on current phase.

        Returns strategy parameters optimized for the detected phase.
        """
        phase, confidence = self.detect_phase(fitness_values, solution_hashes)
        metrics = self._current_metrics

        if phase == EvolutionaryPhase.DISORDERED:
            # Need more exploitation - focus on best solutions
            return PhaseStrategy(
                phase=phase,
                confidence=confidence,
                exploration_rate=0.3,
                mutation_intensity=0.3,
                llm_usage_rate=0.2,
                tournament_pressure=0.8,
                shock_probability=0.0,
                inject_diversity=False,
                focus_on_best=True,
                increase_population=False,
                prune_similar=False,
                message="DISORDERED phase: High chaos, focusing on exploitation to find structure"
            )

        elif phase == EvolutionaryPhase.ORDERED:
            # Need more exploration - inject diversity
            return PhaseStrategy(
                phase=phase,
                confidence=confidence,
                exploration_rate=0.8,
                mutation_intensity=0.7,
                llm_usage_rate=0.6,
                tournament_pressure=0.3,
                shock_probability=0.2,
                inject_diversity=True,
                focus_on_best=False,
                increase_population=True,
                prune_similar=False,
                message="ORDERED phase: Converged, injecting diversity to escape local optima"
            )

        elif phase == EvolutionaryPhase.GLASSY:
            # Need perturbation - apply shocks
            return PhaseStrategy(
                phase=phase,
                confidence=confidence,
                exploration_rate=0.6,
                mutation_intensity=0.9,
                llm_usage_rate=0.8,
                tournament_pressure=0.4,
                shock_probability=0.4,
                inject_diversity=True,
                focus_on_best=False,
                increase_population=False,
                prune_similar=True,
                message="GLASSY phase: Stuck in local optima, applying perturbation shocks"
            )

        elif phase == EvolutionaryPhase.CRITICAL:
            # Maintain balance - optimal state
            return PhaseStrategy(
                phase=phase,
                confidence=confidence,
                exploration_rate=0.5,
                mutation_intensity=0.5,
                llm_usage_rate=0.5,
                tournament_pressure=0.5,
                shock_probability=0.05,
                inject_diversity=False,
                focus_on_best=False,
                increase_population=False,
                prune_similar=False,
                message="CRITICAL phase: Edge of chaos - optimal state, maintaining balance"
            )

        else:  # UNKNOWN
            return PhaseStrategy(
                phase=phase,
                confidence=0.0,
                exploration_rate=0.5,
                mutation_intensity=0.5,
                llm_usage_rate=0.5,
                tournament_pressure=0.5,
                shock_probability=0.0,
                message="UNKNOWN phase: Gathering data, using balanced defaults"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status."""
        return {
            'generations_analyzed': self._generations_analyzed,
            'current_phase': self._current_phase.value,
            'current_metrics': self._current_metrics.to_dict(),
            'fitness_history_length': len(self._fitness_history),
            'unique_solutions_seen': len(self._unique_solutions_seen),
        }

    def reset(self) -> None:
        """Reset the detector state."""
        self._fitness_history.clear()
        self._best_fitness_history.clear()
        self._diversity_history.clear()
        self._innovation_history.clear()
        self._unique_solutions_seen.clear()
        self._current_phase = EvolutionaryPhase.UNKNOWN
        self._current_metrics = PhaseMetrics()
        self._generations_analyzed = 0
        logger.info("PhaseDetector reset")


# Convenience functions
def analyze_population_phase(
    fitness_values: List[float],
    solution_hashes: List[str],
) -> Dict[str, Any]:
    """
    One-shot population phase analysis.

    For quick analysis without tracking history.
    """
    detector = PhaseDetector()
    # Simulate some history for better detection
    for _ in range(3):
        detector.update(fitness_values, solution_hashes, max(fitness_values) if fitness_values else 0)

    strategy = detector.get_strategy(fitness_values, solution_hashes)

    return {
        'phase': strategy.phase.value,
        'confidence': strategy.confidence,
        'strategy': strategy.to_dict(),
        'metrics': detector._current_metrics.to_dict(),
    }
