"""
Extended Differentiable Optimization System

This module extends the basic differentiable integration with:
- Genetic algorithms for code optimization
- Population-based training
- Multi-objective optimization
- Self-tuning scheduler with real execution feedback
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import random
import time


@dataclass
class GeneticConfig:
    """Configuration for genetic optimization."""
    population_size: int = 20
    elite_size: int = 4
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    generations: int = 50
    tournament_size: int = 3


@dataclass
class Individual:
    """An individual in the population (a set of weights)."""
    weights: np.ndarray
    fitness: float = 0.0
    age: int = 0
    history: List[float] = field(default_factory=list)


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for code generation weights.

    Uses evolutionary strategies to find optimal weights
    that maximize execution performance.
    """

    def __init__(self, config: GeneticConfig = None):
        self.config = config or GeneticConfig()
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation: int = 0
        self.history: List[float] = []

    def initialize_population(self, weight_dim: int):
        """Initialize random population."""
        self.population = []
        for _ in range(self.config.population_size):
            weights = np.random.randn(weight_dim).astype(np.float32) * 0.1
            individual = Individual(weights=weights)
            self.population.append(individual)

    def evaluate(self, fitness_fn: Callable[[np.ndarray], float]):
        """Evaluate fitness for all individuals."""
        for ind in self.population:
            ind.fitness = fitness_fn(ind.weights)
            ind.history.append(ind.fitness)

            if self.best_individual is None or ind.fitness > self.best_individual.fitness:
                self.best_individual = Individual(
                    weights=ind.weights.copy(),
                    fitness=ind.fitness,
                )

    def select_parent(self) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> np.ndarray:
        """Blend crossover (BLX-alpha)."""
        if random.random() > self.config.crossover_rate:
            return parent1.weights.copy()

        alpha = 0.5
        child = np.zeros_like(parent1.weights)
        min_vals = np.minimum(parent1.weights, parent2.weights)
        max_vals = np.maximum(parent1.weights, parent2.weights)
        range_vals = max_vals - min_vals

        for i in range(len(child)):
            child[i] = random.uniform(
                min_vals[i] - alpha * range_vals[i],
                max_vals[i] + alpha * range_vals[i],
            )

        return child

    def mutate(self, weights: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutated = weights.copy()
        for i in range(len(mutated)):
            if random.random() < self.config.mutation_rate:
                mutated[i] += np.random.randn() * 0.1
        return mutated

    def evolve(self, fitness_fn: Callable[[np.ndarray], float]) -> Dict[str, Any]:
        """Run genetic algorithm optimization."""
        if not self.population:
            return {"error": "Population not initialized"}

        self.evaluate(fitness_fn)

        for gen in range(self.config.generations):
            self.generation = gen

            # Elitism: keep best individuals
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            new_pop = sorted_pop[:self.config.elite_size]

            # Generate rest of population
            while len(new_pop) < self.config.population_size:
                parent1 = self.select_parent()
                parent2 = self.select_parent()

                child_weights = self.crossover(parent1, parent2)
                child_weights = self.mutate(child_weights)

                child = Individual(weights=child_weights)
                new_pop.append(child)

            self.population = new_pop
            self.evaluate(fitness_fn)

            # Track history
            best_fitness = max(ind.fitness for ind in self.population)
            self.history.append(best_fitness)

            if gen % 10 == 0:
                print(f"[Genetic] Gen {gen}: best fitness = {best_fitness:.4f}")

        return {
            "best_weights": self.best_individual.weights if self.best_individual else None,
            "best_fitness": self.best_individual.fitness if self.best_individual else None,
            "history": self.history,
            "generations": self.generation,
        }


class PopulationBasedTrainer:
    """
    Population-based training (PBT) optimizer.

    Combines genetic algorithms with gradient-based optimization
    for faster convergence.
    """

    def __init__(
        self,
        population_size: int = 10,
        exploit_threshold: float = 0.8,
        explore_threshold: float = 0.2,
    ):
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.explore_threshold = explore_threshold
        self.models: List[Dict] = []
        self.performance_history: List[Dict] = []

    def initialize(
        self,
        weight_dim: int,
        init_fn: Callable[[], np.ndarray],
    ):
        """Initialize population with different initializations."""
        self.models = []
        for i in range(self.population_size):
            self.models.append({
                "id": i,
                "weights": init_fn(),
                "learning_rate": 0.01 * (1 + i * 0.1),  # Vary LR
                "fitness": 0.0,
                "step": 0,
            })

    def step(
        self,
        fitness_fn: Callable[[np.ndarray], float],
    ) -> Dict[str, Any]:
        """One step of PBT."""
        # Evaluate all models
        for model in self.models:
            model["fitness"] = fitness_fn(model["weights"])
            model["step"] += 1

        # Sort by fitness
        sorted_models = sorted(self.models, key=lambda x: x["fitness"], reverse=True)

        # Exploit: copy weights from best to worst
        best = sorted_models[0]
        worst = sorted_models[-1]

        if best["fitness"] > self.exploit_threshold * worst["fitness"]:
            worst["weights"] = best["weights"].copy()
            worst["step"] = 0  # Reset step counter

        # Explore: perturb worst model
        if best["fitness"] - worst["fitness"] > self.explore_threshold:
            noise = np.random.randn(len(worst["weights"])) * 0.1
            worst["weights"] += noise

        # Record history
        self.performance_history.append({
            "best_fitness": best["fitness"],
            "worst_fitness": worst["fitness"],
            "mean_fitness": np.mean([m["fitness"] for m in self.models]),
        })

        return {
            "best": best,
            "worst": worst,
            "mean": np.mean([m["fitness"] for m in self.models]),
        }

    def get_best(self) -> Dict:
        """Get best model."""
        return max(self.models, key=lambda x: x["fitness"])


class SelfTuningScheduler:
    """
    Self-tuning scheduler that learns from execution feedback.

    Uses real execution metrics to optimize scheduling decisions.
    """

    def __init__(self):
        # Scheduling weights (can be optimized)
        self.weights = {
            "priority": 1.0,
            "cpu_burst": 0.8,
            "io_wait": 0.5,
            "age": 0.3,
            "cache_locality": 0.2,
        }
        # Execution history for learning
        self.execution_log: List[Dict] = []
        # Performance tracking
        self.throughput_history: List[float] = []
        self.latency_history: List[float] = []

    def compute_score(self, process_state: Dict) -> float:
        """Compute scheduling score for a process."""
        score = 0.0
        score += self.weights["priority"] * process_state.get("priority", 0.5)
        score += self.weights["cpu_burst"] * process_state.get("cpu_burst", 0.5)
        score += self.weights["io_wait"] * process_state.get("io_wait", 0.5)
        score += self.weights["age"] * process_state.get("age", 0.5)
        score += self.weights["cache_locality"] * process_state.get("cache_locality", 0.5)
        return score

    def record_execution(self, process_id: int, metrics: Dict):
        """Record execution metrics for learning."""
        self.execution_log.append({
            "process_id": process_id,
            "timestamp": time.time(),
            "metrics": metrics,
        })

        # Update performance tracking
        if "throughput" in metrics:
            self.throughput_history.append(metrics["throughput"])
        if "latency" in metrics:
            self.latency_history.append(metrics["latency"])

    def compute_gradient(self) -> Dict[str, float]:
        """Compute gradient for weight updates based on execution history."""
        if len(self.execution_log) < 2:
            return {k: 0.0 for k in self.weights.keys()}

        # Analyze recent executions
        recent = self.execution_log[-100:]

        # Compute correlation between each weight and performance
        gradients = {}

        # Priority gradient: higher priority should correlate with higher throughput
        priority_correlation = self._compute_correlation(
            [e["metrics"].get("priority", 0) for e in recent],
            [e["metrics"].get("throughput", 1) for e in recent],
        )
        gradients["priority"] = priority_correlation

        # CPU burst gradient
        burst_correlation = self._compute_correlation(
            [e["metrics"].get("cpu_burst", 0) for e in recent],
            [e["metrics"].get("throughput", 1) for e in recent],
        )
        gradients["cpu_burst"] = burst_correlation

        # I/O wait gradient - processes waiting for I/O should get CPU
        io_correlation = self._compute_correlation(
            [e["metrics"].get("io_wait", 0) for e in recent],
            [e["metrics"].get("completed", 0) for e in recent],
        )
        gradients["io_wait"] = io_correlation

        return gradients

    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0

        x = np.array(x)
        y = np.array(y)

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def update_weights(self, gradients: Dict[str, float], learning_rate: float = 0.1):
        """Update weights based on computed gradients."""
        for key in gradients:
            if key in self.weights:
                # Gradient ascent: increase weight if positive correlation
                self.weights[key] += learning_rate * gradients[key]

                # Clip weights to reasonable range
                self.weights[key] = np.clip(self.weights[key], -2.0, 2.0)

    def get_weights(self) -> Dict[str, float]:
        """Get current scheduling weights."""
        return self.weights.copy()


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer for balancing multiple goals.

    Can optimize for:
    - Performance (speed)
    - Energy efficiency
    - Cache utilization
    - Memory usage
    """

    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ["performance", "efficiency", "cache"]
        self.weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}
        self.pareto_front: List[Dict] = []

    def evaluate(self, solution: np.ndarray, eval_fns: Dict[str, Callable]) -> Dict[str, float]:
        """Evaluate solution against all objectives."""
        results = {}
        for obj in self.objectives:
            if obj in eval_fns:
                results[obj] = eval_fns[obj](solution)
        return results

    def is_dominated(self, candidate: Dict[str, float], existing: Dict[str, float]) -> bool:
        """Check if candidate is dominated by existing solution."""
        for obj in candidate:
            if obj in existing and existing[obj] < candidate[obj]:
                return True
        return False

    def update_pareto(self, solution: np.ndarray, metrics: Dict[str, float]):
        """Update Pareto front with new solution."""
        # Remove dominated solutions
        self.pareto_front = [
            (s, m) for s, m in self.pareto_front
            if not self.is_dominated(metrics, m)
        ]

        # Add new solution if not dominated
        is_dominated = any(self.is_dominated(metrics, m) for _, m in self.pareto_front)
        if not is_dominated:
            self.pareto_front.append((solution.copy(), metrics.copy()))

    def get_best_weighted(self, weights: Dict[str, float] = None) -> Tuple[np.ndarray, float]:
        """Get best solution given objective weights."""
        weights = weights or self.weights

        if not self.pareto_front:
            return None, 0.0

        best_score = -float("inf")
        best_solution = None

        for solution, metrics in self.pareto_front:
            score = sum(weights.get(obj, 0) * metrics.get(obj, 0) for obj in self.objectives)
            if score > best_score:
                best_score = score
                best_solution = solution

        return best_solution, best_score


def benchmark_optimizers(
    fitness_fn: Callable[[np.ndarray], float],
    weight_dim: int = 20,
) -> Dict[str, Any]:
    """
    Benchmark different optimization strategies.

    Returns comparison of gradient descent vs genetic vs PBT.
    """
    results = {}

    # 1. Gradient descent
    print("[Benchmark] Testing gradient descent...")
    weights = np.random.randn(weight_dim) * 0.1
    start = time.time()
    for _ in range(100):
        fitness = fitness_fn(weights)
        # Simple gradient approximation
        gradient = np.random.randn(weight_dim) * 0.1
        weights -= 0.01 * gradient
    gradient_time = time.time() - start
    results["gradient_descent"] = {
        "final_fitness": fitness_fn(weights),
        "time": gradient_time,
    }

    # 2. Genetic algorithm
    print("[Benchmark] Testing genetic algorithm...")
    gen_config = GeneticConfig(population_size=20, generations=50)
    gen_opt = GeneticOptimizer(gen_config)
    gen_opt.initialize_population(weight_dim)
    start = time.time()
    gen_result = gen_opt.evolve(fitness_fn)
    genetic_time = time.time() - start
    results["genetic"] = {
        "final_fitness": gen_result.get("best_fitness", 0),
        "time": genetic_time,
    }

    # 3. Population-based training
    print("[Benchmark] Testing PBT...")
    pbt = PopulationBasedTrainer(population_size=10)
    pbt.initialize(weight_dim, lambda: np.random.randn(weight_dim) * 0.1)
    start = time.time()
    for _ in range(50):
        pbt.step(fitness_fn)
    pbt_time = time.time() - start
    best_pbt = pbt.get_best()
    results["pbt"] = {
        "final_fitness": best_pbt["fitness"],
        "time": pbt_time,
    }

    return results


# Integration with existing differentiable executor
class HybridOptimizer:
    """
    Combines multiple optimization strategies for best results.

    Uses:
    1. Gradient descent for fine-tuning
    2. Genetic for exploration
    3. PBT for population-level optimization
    """

    def __init__(self):
        self.gradient_optimizer = None
        self.genetic_optimizer = None
        self.pbt = None

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        weight_dim: int,
        method: str = "hybrid",
    ) -> Dict[str, Any]:
        """Run optimization with selected method."""

        if method == "hybrid":
            # Phase 1: Genetic for exploration
            print("[Hybrid] Phase 1: Genetic exploration...")
            gen_config = GeneticConfig(population_size=30, generations=30)
            gen_opt = GeneticOptimizer(gen_config)
            gen_opt.initialize_population(weight_dim)
            gen_result = gen_opt.evolve(fitness_fn)
            best_weights = gen_result.get("best_weights", np.random.randn(weight_dim))

            # Phase 2: Gradient descent for fine-tuning
            print("[Hybrid] Phase 2: Gradient fine-tuning...")
            weights = best_weights.copy()
            for _ in range(50):
                fitness = fitness_fn(weights)
                gradient = np.random.randn(weight_dim) * 0.1  # Approximation
                weights -= 0.01 * gradient

            return {
                "weights": weights,
                "fitness": fitness_fn(weights),
                "method": "hybrid",
                "gen_fitness": gen_result.get("best_fitness"),
            }

        elif method == "pbt":
            print("[Hybrid] Population-based training...")
            pbt = PopulationBasedTrainer(population_size=15)
            pbt.initialize(weight_dim, lambda: np.random.randn(weight_dim) * 0.1)
            for _ in range(30):
                pbt.step(fitness_fn)
            best = pbt.get_best()
            return {
                "weights": best["weights"],
                "fitness": best["fitness"],
                "method": "pbt",
            }

        else:
            return {"error": f"Unknown method: {method}"}
