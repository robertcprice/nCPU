"""
Python wrapper for GPU optimizer.
"""

from typing import Dict, List, Any
from ncpu_metal import (
    GpuGeneticOptimizer,
    GpuNeuralScheduler,
    GaConfig,
    benchmark_genetic,
)


def run_genetic_optimizer(
    fitness_fn,
    weight_dim: int = 20,
    population_size: int = 20,
    generations: int = 30,
) -> Dict[str, Any]:
    """Run genetic algorithm optimization."""
    config = GaConfig(
        population_size=population_size,
        elite_size=4,
        mutation_rate=0.1,
        crossover_rate=0.7,
        generations=generations,
        tournament_size=3,
    )

    optimizer = GpuGeneticOptimizer(config)
    optimizer.initialize(weight_dim)

    # Python fitness function needs to be wrapped
    result = optimizer.run(fitness_fn)

    return {
        "best_weights": result.best_weights,
        "best_fitness": result.best_fitness,
        "history": result.history,
        "generations": result.generations,
    }


def create_neural_scheduler() -> GpuNeuralScheduler:
    """Create a neural scheduler."""
    return GpuNeuralScheduler()


def benchmark_optimizer(weight_dim: int = 20) -> Dict[str, Any]:
    """Benchmark genetic optimizer."""
    # Simple quadratic fitness function - minimum at x=1
    def fitness(weights):
        return -sum((w - 1.0) ** 2 for w in weights)

    result = benchmark_genetic(fitness, weight_dim, 30)

    return {
        "best_weights": result.best_weights,
        "best_fitness": result.best_fitness,
        "generations": result.generations,
    }
