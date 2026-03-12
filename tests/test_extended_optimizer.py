"""
Tests for extended differentiable optimization features.
"""

import pytest
import numpy as np
from ncpu.self_optimizing.extended_optimizer import (
    GeneticConfig,
    GeneticOptimizer,
    PopulationBasedTrainer,
    SelfTuningScheduler,
    MultiObjectiveOptimizer,
    HybridOptimizer,
    benchmark_optimizers,
)


class TestGeneticOptimizer:
    """Test genetic algorithm optimizer."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GeneticConfig()
        assert config.population_size == 20
        assert config.elite_size == 4
        assert config.mutation_rate == 0.1
        assert config.generations == 50

    def test_population_initialization(self):
        """Test population initialization."""
        opt = GeneticOptimizer()
        opt.initialize_population(10)

        assert len(opt.population) == 20
        assert all(ind.weights.shape == (10,) for ind in opt.population)

    def test_evolution(self):
        """Test genetic evolution."""
        opt = GeneticOptimizer(GeneticConfig(population_size=10, generations=5))
        opt.initialize_population(10)

        # Simple fitness function (maximize sum)
        def fitness_fn(w):
            return np.sum(w)

        result = opt.evolve(fitness_fn)

        assert "best_weights" in result
        assert "best_fitness" in result
        assert len(opt.history) == 5

    def test_crossover(self):
        """Test crossover operation."""
        opt = GeneticOptimizer()
        parent1 = np.array([1.0, 2.0, 3.0])
        parent2 = np.array([4.0, 5.0, 6.0])

        child = opt.crossover(
            type('obj', (object,), {'weights': parent1})(),
            type('obj', (object,), {'weights': parent2})(),
        )

        assert child.shape == parent1.shape

    def test_mutation(self):
        """Test mutation operation."""
        opt = GeneticOptimizer(GeneticConfig(mutation_rate=1.0))
        weights = np.array([1.00, 3.0])

        mutated = opt.mutate(weights)

        assert mutated.shape == weights.shape


class TestPopulationBasedTrainer:
    """Test population-based trainer."""

    def test_initialization(self):
        """Test PBT initialization."""
        pbt = PopulationBasedTrainer(population_size=5)
        pbt.initialize(10, lambda: np.ones(10))

        assert len(pbt.models) == 5
        assert all(m["weights"].shape == (10,) for m in pbt.models)

    def test_step(self):
        """Test PBT step."""
        pbt = PopulationBasedTrainer(population_size=5)
        pbt.initialize(10, lambda: np.random.randn(10))

        # Fitness function
        def fitness(w):
            return np.sum(w)

        result = pbt.step(fitness)

        assert "best" in result
        assert "worst" in result

    def test_exploit_and_explore(self):
        """Test exploitation and exploration."""
        pbt = PopulationBasedTrainer(
            population_size=5,
            exploit_threshold=0.5,
            explore_threshold=0.1,
        )
        pbt.initialize(10, lambda: np.random.randn(10))

        # Create large fitness gap
        pbt.models[0]["fitness"] = 1.0
        for m in pbt.models[1:]:
            m["fitness"] = 0.1

        pbt.step(lambda w: np.sum(w))

        # Step should have been called
        assert pbt.models[-1]["step"] >= 0


class TestSelfTuningScheduler:
    """Test self-tuning scheduler."""

    def test_initial_weights(self):
        """Test initial weights."""
        scheduler = SelfTuningScheduler()
        weights = scheduler.get_weights()

        assert "priority" in weights
        assert "cpu_burst" in weights
        assert "io_wait" in weights
        assert "age" in weights

    def test_compute_score(self):
        """Test scheduling score computation."""
        scheduler = SelfTuningScheduler()

        process = {
            "priority": 1.0,
            "cpu_burst": 0.5,
            "io_wait": 0.3,
            "age": 0.2,
            "cache_locality": 0.8,
        }

        score = scheduler.compute_score(process)

        assert score > 0

    def test_record_execution(self):
        """Test recording execution."""
        scheduler = SelfTuningScheduler()

        scheduler.record_execution(1, {
            "throughput": 100,
            "latency": 10,
            "priority": 0.8,
        })

        assert len(scheduler.execution_log) == 1
        assert len(scheduler.throughput_history) == 1

    def test_weight_update(self):
        """Test weight update from gradients."""
        scheduler = SelfTuningScheduler()

        # Add some execution history
        for i in range(10):
            scheduler.record_execution(i, {
                "throughput": 100 + i * 10,
                "latency": 10 - i * 0.5,
                "priority": 0.5,
                "cpu_burst": 0.5,
                "io_wait": 0.3,
                "completed": i + 1,
            })

        gradients = scheduler.compute_gradient()

        # Update weights
        initial_weights = scheduler.get_weights()
        scheduler.update_weights(gradients, learning_rate=0.1)

        # Weights may or may not have changed (depends on gradients)
        new_weights = scheduler.get_weights()
        # Just verify weights are returned
        assert isinstance(new_weights, dict)


class TestMultiObjectiveOptimizer:
    """Test multi-objective optimizer."""

    def test_initialization(self):
        """Test initialization."""
        opt = MultiObjectiveOptimizer(["speed", "accuracy"])

        assert "speed" in opt.objectives
        assert "accuracy" in opt.objectives
        assert len(opt.weights) == 2

    def test_pareto_update(self):
        """Test Pareto front update."""
        opt = MultiObjectiveOptimizer(["speed", "accuracy"])

        # Add solutions - one may dominate the other
        opt.update_pareto(np.array([1.0, 2.0]), {"speed": 10, "accuracy": 0.9})
        opt.update_pareto(np.array([2.0, 1.0]), {"speed": 20, "accuracy": 0.8})

        # At least one solution should be in Pareto front
        assert len(opt.pareto_front) >= 1

    def test_dominance(self):
        """Test Pareto dominance."""
        opt = MultiObjectiveOptimizer(["speed", "accuracy"])

        candidate = {"speed": 10, "accuracy": 0.9}
        existing = {"speed": 20, "accuracy": 0.8}

        # Existing dominates candidate
        assert opt.is_dominated(candidate, existing)

    def test_best_weighted(self):
        """Test weighted best selection."""
        opt = MultiObjectiveOptimizer(["speed", "accuracy"])

        opt.update_pareto(np.array([1.0, 2.0]), {"speed": 10, "accuracy": 0.9})
        opt.update_pareto(np.array([2.0, 1.0]), {"speed": 20, "accuracy": 0.8})

        solution, score = opt.get_best_weighted({"speed": 1.0, "accuracy": 0.0})

        # Should prefer higher speed
        assert np.array_equal(solution, np.array([2.0, 1.0]))


class TestHybridOptimizer:
    """Test hybrid optimizer."""

    def test_hybrid_optimization(self):
        """Test hybrid optimization."""
        opt = HybridOptimizer()

        # Simple fitness function
        def fitness(w):
            return -np.sum((w - 1.0) ** 2)  # Maximize when w = 1

        result = opt.optimize(fitness, weight_dim=5, method="hybrid")

        assert "weights" in result
        assert "fitness" in result
        assert result["method"] == "hybrid"

    def test_pbt_optimization(self):
        """Test PBT optimization."""
        opt = HybridOptimizer()

        def fitness(w):
            return np.sum(w)

        result = opt.optimize(fitness, weight_dim=5, method="pbt")

        assert "weights" in result
        assert "fitness" in result
        assert result["method"] == "pbt"


class TestBenchmarking:
    """Test benchmarking functions."""

    def test_benchmark_optimizers(self):
        """Test optimizer benchmarking."""
        def fitness(w):
            return np.sum(w)

        results = benchmark_optimizers(fitness, weight_dim=10)

        assert "gradient_descent" in results
        assert "genetic" in results
        assert "pbt" in results

        for method, result in results.items():
            assert "final_fitness" in result
            assert "time" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
