#!/usr/bin/env python3
"""
OUROBOROS - Self-Evolving Code Optimization Engine

Named after the ancient symbol of the serpent eating its own tail,
Ouroboros represents cyclical self-improvement and eternal return.

This is the main entry point for the self-evolving system.
It orchestrates all components:
- Constitution (Governor, Arena, Judge, Kill Switch)
- Evolution (Population, Generations, Code Surgery)
- LLM Mutation (Ollama integration, fallback)

Usage:
    python ouroboros.py --target-dir /path/to/code
    python ouroboros.py --demo  # Run with demo targets

Ouroboros will:
1. Discover code to optimize
2. Create an initial population
3. Run evolutionary cycles:
   - Tournament selection
   - Breeding (crossover + mutation)
   - Validation (Judge with Hypothesis)
   - Generational replacement
4. Continue until stopped or target achieved
"""

import argparse
import asyncio
import logging
import time
import signal
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Constitution imports
from semantic_engine.constitution import (
    Governor, GovernorConfig,
    Arena, ArenaConfig,
    Judge, JudgeConfig,
    KillSwitch, KillSwitchConfig,
)

# Evolution imports
from semantic_engine.evolution import (
    Agent, Population, PopulationConfig,
    GenerationManager, GenerationConfig,
    CodeSurgeon,
)
from semantic_engine.evolution.generations import BreedingManager

# LLM imports
from llm import (
    OllamaMutator, OllamaConfig, MutationRequest,
    ModelSelector, ComplexityLevel,
    FallbackMutator,
)
from llm.ollama_mutator import OptimizationGoal, SyncOllamaMutator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('Ouroboros')


@dataclass
class OuroborosConfig:
    """Configuration for the Ouroboros self-evolving engine."""

    # Directories
    work_dir: Path = Path("/tmp/ouroboros")
    target_dir: Optional[Path] = None

    # Population settings
    initial_population_size: int = 5
    max_population_size: int = 15

    # Evolution settings
    generations_per_cycle: int = 10
    offspring_per_generation: int = 3

    # LLM settings
    use_llm: bool = True
    llm_mutation_rate: float = 0.5  # 50% LLM, 50% AST

    # Stopping criteria
    max_generations: int = 1000
    target_fitness: float = 0.95
    stagnation_threshold: int = 50  # Stop if no improvement for N generations

    # Safety
    enable_kill_switch: bool = True

    def __post_init__(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        if self.target_dir and isinstance(self.target_dir, str):
            self.target_dir = Path(self.target_dir)


# Demo target functions to optimize
DEMO_TARGETS = {
    "bubble_sort": '''
def bubble_sort(arr):
    """Simple bubble sort - O(n²)."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
''',

    "fibonacci": '''
def fibonacci(n):
    """Naive recursive fibonacci - O(2^n)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',

    "find_duplicates": '''
def find_duplicates(arr):
    """Find duplicates in array - O(n²)."""
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
''',

    "is_prime": '''
def is_prime(n):
    """Check if n is prime - O(n)."""
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
''',

    "matrix_multiply": '''
def matrix_multiply(A, B):
    """Matrix multiplication - O(n³)."""
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result
''',
}


class Ouroboros:
    """
    OUROBOROS - Self-Evolving Code Optimization System.

    Named after the ancient symbol of the serpent eating its own tail,
    Ouroboros represents cyclical self-improvement and eternal return.

    This is the main orchestrator that runs the evolutionary loop:
    1. Initialize population with target code
    2. Run tournaments to select fittest agents
    3. Breed offspring using LLM + AST mutations
    4. Validate offspring through Judge (Hypothesis fuzzing)
    5. Replace weak agents with validated offspring
    6. Repeat until target fitness or max generations
    """

    def __init__(self, config: OuroborosConfig):
        self.config = config
        self._running = False
        self._start_time: Optional[float] = None

        # Initialize work directories
        self.config.work_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Constitution components
        logger.info("Initializing Constitution...")
        self.kill_switch = KillSwitch(KillSwitchConfig(
            kill_file_path=self.config.work_dir / "kill_switch",
            heartbeat_file_path=self.config.work_dir / "heartbeat",
        ))

        self.arena = Arena(ArenaConfig(
            sandbox_dir=self.config.work_dir / "sandbox",
        ))

        self.judge = Judge(JudgeConfig())

        self.governor = Governor(GovernorConfig(
            min_population=3,
            max_population=self.config.max_population_size,
            population_dir=self.config.work_dir / "population",
            generations_dir=self.config.work_dir / "generations",
            rollback_dir=self.config.work_dir / "rollback",
        ))

        # Initialize Evolution components
        logger.info("Initializing Evolution...")
        self.population = Population(PopulationConfig(
            min_population=3,
            max_population=self.config.max_population_size,
            population_dir=self.config.work_dir / "pop_data",
        ))

        self.gen_manager = GenerationManager(
            GenerationConfig(
                generations_dir=self.config.work_dir / "gen_data",
                rollback_dir=self.config.work_dir / "rb_data",
                offspring_per_generation=self.config.offspring_per_generation,
            ),
            self.population,
        )

        self.breeding_manager = BreedingManager(
            self.population,
            self.gen_manager,
            GenerationConfig(),
        )

        self.code_surgeon = CodeSurgeon()

        # Initialize LLM components
        logger.info("Initializing LLM...")
        self.model_selector = ModelSelector()
        self.fallback_mutator = FallbackMutator()

        if self.config.use_llm:
            self.llm_mutator = SyncOllamaMutator(OllamaConfig())
            self._llm_available = self.llm_mutator.is_available()
            if self._llm_available:
                logger.info("✅ Ollama LLM available")
            else:
                logger.warning("⚠️ Ollama not available, using fallback mutations")
        else:
            self.llm_mutator = None
            self._llm_available = False

        # Statistics
        self._stats = {
            'generations': 0,
            'mutations': 0,
            'llm_mutations': 0,
            'ast_mutations': 0,
            'validations': 0,
            'accepts': 0,
            'rejects': 0,
            'best_fitness': 0.0,
        }

    def _check_halt(self) -> bool:
        """Check if we should halt."""
        return self.kill_switch.should_halt()

    def _mutate_code(self, code: str) -> str:
        """
        Apply mutation to code using LLM or fallback.

        Strategy:
        - If LLM available and random < llm_mutation_rate: Use LLM
        - Otherwise: Use AST-based fallback
        """
        import random

        self._stats['mutations'] += 1

        # Decide mutation strategy
        use_llm = (
            self._llm_available and
            random.random() < self.config.llm_mutation_rate
        )

        if use_llm:
            self._stats['llm_mutations'] += 1

            # Select optimization goal
            goals = [
                OptimizationGoal.SPEED,
                OptimizationGoal.ALGORITHMIC,
                OptimizationGoal.SIMPLICITY,
            ]

            request = MutationRequest(
                source_code=code,
                goal=random.choice(goals),
                model=self.model_selector.select_model(code),
            )

            response = self.llm_mutator.mutate(request)

            if response.success:
                return response.mutated_code
            else:
                # Fall back to AST
                logger.debug(f"LLM mutation failed: {response.error}")

        # AST-based mutation
        self._stats['ast_mutations'] += 1
        result = self.code_surgeon.random_mutation(code)

        if result.success:
            return result.new_code

        # Try fallback mutator
        fb_result = self.fallback_mutator.mutate(code)
        return fb_result.mutated_code if fb_result.success else code

    def _validate_code(self, new_code: str, old_code: str) -> bool:
        """Validate mutated code through the Judge."""
        self._stats['validations'] += 1

        result = self.judge.verify(new_code, old_code)

        if result.passed:
            self._stats['accepts'] += 1
            return True
        else:
            self._stats['rejects'] += 1
            logger.debug(f"Validation failed: {result.reason}")
            return False

    def _benchmark_agent(self, agent: Agent) -> float:
        """Run benchmarks on an agent and return fitness score."""
        import random

        # Generate test inputs based on code type
        test_inputs = []

        code_lower = agent.source_code.lower()

        if 'sort' in code_lower:
            # Sorting benchmarks
            for size in [10, 100, 500]:
                test_inputs.append([random.randint(0, 1000) for _ in range(size)])
                test_inputs.append(list(range(size)))  # Already sorted
                test_inputs.append(list(range(size, 0, -1)))  # Reverse sorted

        elif 'fibonacci' in code_lower or 'fib' in code_lower:
            # Fibonacci benchmarks
            test_inputs = list(range(0, 25))

        elif 'prime' in code_lower:
            # Prime benchmarks
            test_inputs = list(range(2, 1000, 7))

        elif 'duplicate' in code_lower:
            # Duplicate finding benchmarks
            for size in [50, 100, 200]:
                arr = [random.randint(0, size // 2) for _ in range(size)]
                test_inputs.append(arr)

        elif 'matrix' in code_lower:
            # Matrix benchmarks
            for size in [5, 10, 20]:
                A = [[random.randint(0, 10) for _ in range(size)] for _ in range(size)]
                B = [[random.randint(0, 10) for _ in range(size)] for _ in range(size)]
                test_inputs.append((A, B))

        else:
            # Generic list benchmarks
            test_inputs = [
                [],
                [1],
                list(range(100)),
                [random.randint(0, 1000) for _ in range(100)],
            ]

        # Run benchmarks
        total_time = 0.0
        successful = 0

        for input_data in test_inputs[:10]:  # Limit to 10 inputs
            result = self.arena.run_sandboxed(
                agent.source_code,
                input_data,
                timeout=5,  # 5 second timeout per input
            )

            if result.success:
                successful += 1
                total_time += result.execution_time

        if successful == 0:
            return 0.0

        # Fitness = correctness * speed_factor
        correctness = successful / min(len(test_inputs), 10)
        avg_time = total_time / successful
        speed_factor = 1.0 / (1.0 + avg_time * 10)  # Faster = higher

        return correctness * 0.7 + speed_factor * 0.3

    def initialize_population(self, targets: Dict[str, str]) -> None:
        """Initialize population with target code."""
        logger.info(f"Initializing population with {len(targets)} targets...")

        for name, code in targets.items():
            agent = Agent.create_genesis(code, agent_id=f"genesis_{name}")

            # Benchmark
            fitness = self._benchmark_agent(agent)
            agent.fitness_score = fitness
            agent.metadata['target_name'] = name

            self.population.add_agent(agent)
            logger.info(f"  Added {name}: fitness={fitness:.4f}")

        logger.info(f"Population initialized: {self.population.size} agents")

    def run_generation(self) -> Dict[str, Any]:
        """Run a single generation cycle."""
        if self._check_halt():
            return {'halted': True}

        gen_start = time.time()
        self._stats['generations'] += 1
        gen_num = self.gen_manager.current_generation

        logger.info(f"=== Generation {gen_num} ===")

        # 1. Select parents
        parents = self.breeding_manager.select_breeding_pairs(
            self.config.offspring_per_generation
        )

        # 2. Create offspring
        offspring = []
        for parent_a, parent_b in parents:
            if self._check_halt():
                break

            # Mutate parent code
            base_code = parent_a.source_code
            mutated_code = self._mutate_code(base_code)

            # Validate
            if mutated_code != base_code and self._validate_code(mutated_code, base_code):
                child = Agent(
                    agent_id=f"gen{gen_num}_{len(offspring)}_{hash(mutated_code) % 10000:04x}",
                    generation=gen_num,
                    source_code=mutated_code,
                    parent_ids=[parent_a.agent_id, parent_b.agent_id],
                    mutation_type="llm" if self._llm_available else "ast",
                )

                # Benchmark
                child.fitness_score = self._benchmark_agent(child)
                offspring.append(child)

        # 3. Add offspring to population
        added = 0
        for child in offspring:
            if self.population.size < self.config.max_population_size:
                self.population.add_agent(child)
                added += 1

        # 4. Cull weak agents if over limit
        if self.population.size > self.config.max_population_size:
            culled = self.breeding_manager.cull_population(
                self.population.size - self.config.max_population_size
            )
            logger.info(f"  Culled {len(culled)} weak agents")

        # 5. Advance generation
        self.gen_manager.advance_generation()

        # Update stats
        best = self.population.best_agent
        if best:
            self._stats['best_fitness'] = max(self._stats['best_fitness'], best.fitness_score)

        gen_time = time.time() - gen_start

        result = {
            'generation': gen_num,
            'offspring_created': len(offspring),
            'offspring_added': added,
            'population_size': self.population.size,
            'best_fitness': best.fitness_score if best else 0,
            'entropy': self.population.compute_shannon_entropy(),
            'time': gen_time,
        }

        logger.info(
            f"  Offspring: {len(offspring)}, Pop: {self.population.size}, "
            f"Best: {result['best_fitness']:.4f}, Time: {gen_time:.2f}s"
        )

        return result

    def run(self, max_generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the evolutionary loop.

        Returns statistics and final state.
        """
        max_gen = max_generations or self.config.max_generations
        self._running = True
        self._start_time = time.time()

        # Start heartbeat
        self.kill_switch.start_heartbeat()

        logger.info("=" * 60)
        logger.info("🔥 GENETIC FORGE STARTING")
        logger.info("=" * 60)
        logger.info(f"Max generations: {max_gen}")
        logger.info(f"Population size: {self.population.size}")
        logger.info(f"LLM available: {self._llm_available}")
        logger.info("=" * 60)

        generations_without_improvement = 0
        last_best_fitness = 0.0

        try:
            while self._running and self._stats['generations'] < max_gen:
                # Check halt
                if self._check_halt():
                    logger.warning("Kill switch triggered!")
                    break

                # Run generation
                result = self.run_generation()

                if result.get('halted'):
                    break

                # Check improvement
                current_best = result.get('best_fitness', 0)
                if current_best > last_best_fitness + 0.001:
                    generations_without_improvement = 0
                    last_best_fitness = current_best
                else:
                    generations_without_improvement += 1

                # Check stagnation
                if generations_without_improvement >= self.config.stagnation_threshold:
                    logger.info(f"Stagnation detected after {generations_without_improvement} generations")
                    break

                # Check target fitness
                if current_best >= self.config.target_fitness:
                    logger.info(f"🎉 Target fitness {self.config.target_fitness} achieved!")
                    break

                # Pulse heartbeat
                self.kill_switch.pulse()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self._running = False
            self.kill_switch.stop_heartbeat()

        # Final stats
        elapsed = time.time() - self._start_time
        self._stats['elapsed_time'] = elapsed

        logger.info("=" * 60)
        logger.info("🏁 GENETIC FORGE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generations: {self._stats['generations']}")
        logger.info(f"Mutations: {self._stats['mutations']} (LLM: {self._stats['llm_mutations']}, AST: {self._stats['ast_mutations']})")
        logger.info(f"Validations: {self._stats['validations']} (Accept: {self._stats['accepts']}, Reject: {self._stats['rejects']})")
        logger.info(f"Best fitness: {self._stats['best_fitness']:.4f}")
        logger.info(f"Elapsed time: {elapsed:.2f}s")
        logger.info("=" * 60)

        # Return final state
        return {
            'stats': self._stats,
            'population': self.population.get_stats(),
            'best_agent': self.population.best_agent.to_dict() if self.population.best_agent else None,
            'generations': self.gen_manager.get_stats(),
        }

    def stop(self) -> None:
        """Stop the evolutionary loop."""
        self._running = False
        logger.info("Stop requested")


def main():
    parser = argparse.ArgumentParser(description="OUROBOROS - Self-Evolving Code Optimization")
    parser.add_argument('--target-dir', type=Path, help="Directory containing code to optimize")
    parser.add_argument('--demo', action='store_true', help="Run with demo targets")
    parser.add_argument('--max-generations', type=int, default=50, help="Maximum generations to run")
    parser.add_argument('--population-size', type=int, default=10, help="Maximum population size")
    parser.add_argument('--no-llm', action='store_true', help="Disable LLM mutations")
    parser.add_argument('--work-dir', type=Path, default="/tmp/ouroboros", help="Working directory")

    args = parser.parse_args()

    # Create config
    config = OuroborosConfig(
        work_dir=args.work_dir,
        target_dir=args.target_dir,
        max_population_size=args.population_size,
        max_generations=args.max_generations,
        use_llm=not args.no_llm,
    )

    # Create the Ouroboros engine
    engine = Ouroboros(config)

    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Signal received, stopping...")
        engine.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize population
    if args.demo:
        engine.initialize_population(DEMO_TARGETS)
    elif args.target_dir:
        # Load from directory
        targets = {}
        for py_file in args.target_dir.glob("*.py"):
            targets[py_file.stem] = py_file.read_text()
        if not targets:
            logger.error(f"No Python files found in {args.target_dir}")
            sys.exit(1)
        engine.initialize_population(targets)
    else:
        logger.error("Must specify --demo or --target-dir")
        sys.exit(1)

    # Run the Ouroboros cycle
    results = engine.run()

    # Print best agent code
    if results.get('best_agent'):
        print("\n" + "=" * 60)
        print("BEST AGENT CODE:")
        print("=" * 60)
        print(results['best_agent']['source_code'])


if __name__ == "__main__":
    main()
