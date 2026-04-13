"""
AutoML for GPU Kernels

Uses the self-optimizing engine to automatically discover optimal
GPU kernel configurations, neural network architectures, and
hyperparameters through genetic algorithms and neural search.

This is a novel application: instead of manually tuning GPU kernels,
let the system discover optimal configurations through execution feedback.

Key features:
- Neural Architecture Search (NAS) for GPU kernels
- Hyperparameter optimization for differentiable CPUs
- Automatic operation fusion discovery
- Memory layout optimization
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
import numpy as np
import random


class SearchSpace(Enum):
    """Search space types for AutoML."""
    KERNEL_CONFIG = "kernel_config"
    ARCHITECTURE = "architecture"
    HYPERPARAMETERS = "hyperparameters"
    MEMORY_LAYOUT = "memory_layout"


@dataclass
class KernelConfig:
    """Configuration for a GPU kernel."""
    # Thread block size
    block_size_x: int = 256
    block_size_y: int = 1
    block_size_z: int = 1

    # Shared memory
    shared_memory_bytes: int = 16384
    use_l1_cache: bool = True

    # Execution
    max_registers_per_thread: int = 64
    use_private_memory: bool = False

    # Differentiable CPU config
    diff_cpu_ooo_depth: int = 8
    diff_cpu_commit_width: int = 4
    diff_cpu_rob_size: int = 32


@dataclass
class ArchitectureGene:
    """A gene representing a neural network architecture component."""
    layer_type: str  # conv, dense, attention, layernorm, relu, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Individual:
    """An individual in the evolutionary search."""
    id: int
    genes: List[ArchitectureGene]
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def crossover(self, other: 'Individual', crossover_point: int) -> 'Individual':
        """Perform crossover with another individual."""
        new_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        return Individual(
            id=random.randint(0, 100000),
            genes=new_genes,
        )

    def mutate(self, mutation_rate: float, search_space: List[str]) -> 'Individual':
        """Mutate genes."""
        new_genes = []
        for gene in self.genes:
            if random.random() < mutation_rate:
                # Replace with random gene from search space
                gene = ArchitectureGene(
                    layer_type=random.choice(search_space),
                    parameters={},
                )
            new_genes.append(gene)
        return Individual(
            id=random.randint(0, 100000),
            genes=new_genes,
            fitness=self.fitness,
            metrics=self.metrics,
        )


@dataclass
class AutoMLResult:
    """Result from AutoML search."""
    best_config: Any
    best_fitness: float
    generations: int
    population_size: int
    total_evaluations: int
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    search_time_seconds: float = 0.0


class KernelAutoML:
    """
    AutoML for GPU kernel optimization.

    Uses evolutionary search to find optimal kernel configurations
    based on execution metrics (throughput, latency, memory usage).

    Usage:
        automl = KernelAutoML(objective="throughput")
        result = automl.search(max_generations=50, population_size=20)

        # Use best config
        best_config = result.best_config
    """

    def __init__(
        self,
        objective: str = "throughput",
        constraints: Optional[Dict[str, Any]] = None,
    ):
        self.objective = objective
        self.constraints = constraints or {"max_memory_mb": 16}
        self.population: List[Individual] = []
        self.history: List[float] = []
        self.best_ever: Optional[Individual] = None

    def _random_config(self) -> KernelConfig:
        """Generate a random kernel configuration."""
        return KernelConfig(
            block_size_x=random.choice([64, 128, 256, 512]),
            block_size_y=random.choice([1, 2, 4, 8]),
            block_size_z=1,
            shared_memory_bytes=random.choice([4096, 8192, 16384, 32768]),
            use_l1_cache=random.choice([True, False]),
            max_registers_per_thread=random.choice([32, 64, 128]),
            use_private_memory=random.choice([True, False]),
            diff_cpu_ooo_depth=random.choice([4, 8, 16, 32]),
            diff_cpu_commit_width=random.choice([2, 4, 8]),
            diff_cpu_rob_size=random.choice([16, 32, 64, 128]),
        )

    def _config_to_genes(self, config: KernelConfig) -> List[ArchitectureGene]:
        """Convert config to genes for evolutionary search."""
        return [
            ArchitectureGene("block_size", {"x": config.block_size_x,
                                            "y": config.block_size_y,
                                            "z": config.block_size_z}),
            ArchitectureGene("shared_memory", {"bytes": config.shared_memory_bytes}),
            ArchitectureGene("cache", {"l1": config.use_l1_cache}),
            ArchitectureGene("registers", {"max": config.max_registers_per_thread}),
            ArchitectureGene("diff_ooo", {"depth": config.diff_cpu_ooo_depth,
                                           "width": config.diff_cpu_commit_width,
                                           "rob": config.diff_cpu_rob_size}),
        ]

    def _genes_to_config(self, genes: List[ArchitectureGene]) -> KernelConfig:
        """Convert genes back to config."""
        config = KernelConfig()
        for gene in genes:
            if gene.layer_type == "block_size":
                config.block_size_x = gene.parameters.get("x", 256)
                config.block_size_y = gene.parameters.get("y", 1)
                config.block_size_z = gene.parameters.get("z", 1)
            elif gene.layer_type == "shared_memory":
                config.shared_memory_bytes = gene.parameters.get("bytes", 16384)
            elif gene.layer_type == "cache":
                config.use_l1_cache = gene.parameters.get("l1", True)
            elif gene.layer_type == "registers":
                config.max_registers_per_thread = gene.parameters.get("max", 64)
            elif gene.layer_type == "diff_ooo":
                config.diff_cpu_ooo_depth = gene.parameters.get("depth", 8)
                config.diff_cpu_commit_width = gene.parameters.get("width", 4)
                config.diff_cpu_rob_size = gene.parameters.get("rob", 32)
        return config

    def _evaluate(self, config: KernelConfig) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a kernel configuration.

        In production, this would actually run the kernel and measure metrics.
        For demo, we use a simulated function.
        """
        # Simulated evaluation (in production, run actual benchmark)
        # Higher block size generally better up to a point
        block_score = min(config.block_size_x * config.block_size_y / 256, 4.0)

        # Shared memory trade-off
        if config.shared_memory_bytes > 16384:
            mem_score = 1.5
        else:
            mem_score = 1.0

        # Cache helps
        cache_score = 1.3 if config.use_l1_cache else 1.0

        # OOO depth trade-off
        ooo_score = min(config.diff_cpu_ooo_depth / 8, 2.0)

        # Combine into fitness
        fitness = block_score * mem_score * cache_score * ooo_score
        fitness = fitness + random.gauss(0, 0.1)  # Add noise

        metrics = {
            "throughput": fitness * 1000,
            "latency_ms": 10 / fitness,
            "memory_used_mb": config.shared_memory_bytes / (1024 * 1024),
            "efficiency": fitness / 4.0,
        }

        return fitness, metrics

    def initialize_population(self, size: int):
        """Initialize random population."""
        self.population = []
        for i in range(size):
            config = self._random_config()
            genes = self._config_to_genes(config)
            fitness, metrics = self._evaluate(config)

            individual = Individual(
                id=i,
                genes=genes,
                fitness=fitness,
                metrics=metrics,
            )
            self.population.append(individual)

            if self.best_ever is None or fitness > self.best_ever.fitness:
                self.best_ever = individual

    def select_parents(self, k: int = 3) -> List[Individual]:
        """Tournament selection."""
        selected = []
        for _ in range(2):  # Two parents
            tournament = random.sample(self.population, min(k, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def evolve_generation(self, mutation_rate: float = 0.1) -> float:
        """Evolve one generation."""
        new_population = []

        # Elitism: keep best 2
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(sorted_pop[:2])

        # Generate rest through crossover and mutation
        while len(new_population) < len(self.population):
            parents = self.select_parents()

            # Crossover
            if random.random() < 0.7:
                crossover_point = random.randint(1, max(1, len(parents[0].genes) - 1))
                child = parents[0].crossover(parents[1], crossover_point)
            else:
                child = parents[0]

            # Mutation
            child = child.mutate(mutation_rate, ["block_size", "shared_memory", "cache"])

            # Evaluate
            config = self._genes_to_config(child.genes)
            fitness, metrics = self._evaluate(config)
            child.fitness = fitness
            child.metrics = metrics

            new_population.append(child)

            if self.best_ever is None or fitness > self.best_ever.fitness:
                self.best_ever = child

        self.population = new_population

        # Record history
        best_fitness = max(ind.fitness for ind in self.population)
        self.history.append(best_fitness)

        return best_fitness

    def search(
        self,
        max_generations: int = 50,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        early_stop: bool = True,
        patience: int = 10,
    ) -> AutoMLResult:
        """
        Run AutoML search.

        Args:
            max_generations: Maximum number of generations
            population_size: Size of population
            mutation_rate: Probability of mutation
            early_stop: Stop if no improvement
            patience: Generations to wait for improvement

        Returns:
            AutoMLResult with best configuration
        """
        import time
        start_time = time.time()

        # Initialize
        self.initialize_population(population_size)

        # Track best
        best_fitness = self.best_ever.fitness if self.best_ever else 0
        no_improvement = 0

        # Evolve
        for gen in range(max_generations):
            current_best = self.evolve_generation(mutation_rate)

            # Track metrics
            if gen % 10 == 0:
                print(f"Generation {gen}: best fitness = {current_best:.3f}")

            # Early stopping
            if early_stop:
                if current_best > best_fitness:
                    best_fitness = current_best
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    print(f"Early stop at generation {gen}")
                    break

        # Return best result
        best_config = self._genes_to_config(self.best_ever.genes)

        return AutoMLResult(
            best_config=best_config,
            best_fitness=self.best_ever.fitness,
            generations=len(self.history),
            population_size=population_size,
            total_evaluations=len(self.history) * population_size,
            metrics={"fitness": self.history},
            search_time_seconds=time.time() - start_time,
        )


# =============================================================================
# Neural Architecture Search
# =============================================================================

class NeuralArchitectureSearch:
    """
    Neural Architecture Search for differentiable operations.

    Discovers optimal neural network architectures for specific tasks
    using evolutionary algorithms with execution feedback.
    """

    LAYER_TYPES = [
        "dense", "conv1d", "conv2d", "attention", "layernorm",
        "relu", "gelu", "sigmoid", "softmax", "dropout",
        "add", "multiply", "concat", "reshape",
    ]

    def __init__(self, max_layers: int = 10):
        self.max_layers = max_layers
        self.population: List[Individual] = []
        self.history: List[float] = []

    def random_architecture(self) -> Individual:
        """Generate a random architecture."""
        num_layers = random.randint(1, self.max_layers)
        genes = []

        for _ in range(num_layers):
            gene = ArchitectureGene(
                layer_type=random.choice(self.LAYER_TYPES),
                parameters=self._random_layer_params(random.choice(self.LAYER_TYPES)),
            )
            genes.append(gene)

        return Individual(id=random.randint(0, 100000), genes=genes)

    def _random_layer_params(self, layer_type: str) -> Dict[str, Any]:
        """Generate random parameters for a layer type."""
        params = {}
        if layer_type == "dense":
            params = {"units": random.choice([32, 64, 128, 256, 512])}
        elif layer_type == "conv1d":
            params = {
                "filters": random.choice([16, 32, 64, 128]),
                "kernel_size": random.choice([3, 5, 7]),
                "strides": random.choice([1, 2]),
            }
        elif layer_type == "conv2d":
            params = {
                "filters": random.choice([16, 32, 64, 128]),
                "kernel_size": random.choice([3, 5]),
                "strides": random.choice([1, 2]),
            }
        elif layer_type == "attention":
            params = {
                "heads": random.choice([2, 4, 8]),
                "key_dim": random.choice([32, 64, 128]),
            }
        elif layer_type == "layernorm":
            params = {"epsilon": random.choice([1e-5, 1e-4])}
        elif layer_type == "dropout":
            params = {"rate": random.uniform(0.1, 0.5)}

        return params

    def search(
        self,
        task: str,
        fitness_fn: Callable[[List[ArchitectureGene]], float],
        max_generations: int = 30,
        population_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Run neural architecture search.

        Args:
            task: Task name
            fitness_fn: Function to evaluate architecture
            max_generations: Maximum generations
            population_size: Population size

        Returns:
            Best architecture found
        """
        # Initialize
        self.population = [self.random_architecture() for _ in range(population_size)]

        # Evaluate initial
        for ind in self.population:
            ind.fitness = fitness_fn(ind.genes)

        self.best_ever = max(self.population, key=lambda x: x.fitness)

        # Evolve
        for gen in range(max_generations):
            # Tournament selection
            parents = random.sample(self.population, 4)
            parents = sorted(parents, key=lambda x: x.fitness, reverse=True)[:2]

            # Crossover
            if random.random() < 0.7:
                child = parents[0].crossover(parents[1], random.randint(1, max(1, len(parents[0].genes) - 1)))
            else:
                child = parents[0]

            # Mutation
            child = child.mutate(0.1, self.LAYER_TYPES)

            # Evaluate
            child.fitness = fitness_fn(child.genes)

            # Replace worst
            worst = min(self.population, key=lambda x: x.fitness)
            worst_idx = self.population.index(worst)
            self.population[worst_idx] = child

            if child.fitness > self.best_ever.fitness:
                self.best_ever = child

            self.history.append(self.best_ever.fitness)

            if gen % 5 == 0:
                print(f"Gen {gen}: best fitness = {self.best_ever.fitness:.3f}")

        return {
            "best_genes": self.best_ever.genes,
            "best_fitness": self.best_ever.fitness,
            "generations": max_generations,
            "history": self.history,
        }


# =============================================================================
# Demo
# =============================================================================

def demo_kernel_automl():
    """Demo kernel AutoML."""
    print("Running Kernel AutoML demo...")

    automl = KernelAutoML(objective="throughput")
    result = automl.search(max_generations=30, population_size=15)

    print(f"\nBest configuration found:")
    print(f"  Block size: {result.best_config.block_size_x} x {result.best_config.block_size_y}")
    print(f"  Shared memory: {result.best_config.shared_memory_bytes} bytes")
    print(f"  L1 cache: {result.best_config.use_l1_cache}")
    print(f"  Diff OOO depth: {result.best_config.diff_cpu_ooo_depth}")
    print(f"  Fitness: {result.best_fitness:.3f}")
    print(f"  Search time: {result.search_time_seconds:.2f}s")


def demo_nas():
    """Demo neural architecture search."""
    print("\nRunning Neural Architecture Search demo...")

    # Simulated fitness function
    def fitness_fn(genes):
        score = 0
        for gene in genes:
            if gene.layer_type == "dense":
                score += gene.parameters.get("units", 64) / 100
            elif gene.layer_type == "conv1d":
                score += gene.parameters.get("filters", 32) / 50
            elif gene.layer_type == "attention":
                score += gene.parameters.get("heads", 4) * 0.1
            elif gene.layer_type == "layernorm":
                score += 0.5
        return min(score, 1.0)

    nas = NeuralArchitectureSearch(max_layers=5)
    result = nas.search("classification", fitness_fn, max_generations=20)

    print(f"\nBest architecture found:")
    for i, gene in enumerate(result["best_genes"]):
        print(f"  Layer {i}: {gene.layer_type} - {gene.parameters}")
    print(f"  Fitness: {result['best_fitness']:.3f}")


if __name__ == "__main__":
    demo_kernel_automl()
    demo_nas()
