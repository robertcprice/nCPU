"""
Experiment Tracking for SOME

Integrates with Weights & Biases (wandb) for experiment tracking,
visualization, and collaboration. Also supports MLflow for self-hosted setups.

Usage:
    from ncpu.self_optimizing.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(
        project="some-experiments",
        entity="my-team",
    )

    # Track a generation cycle
    tracker.log_generation(
        generation=1,
        task="sort_array",
        parameters={"algorithm": "quicksort"},
        metrics={"execution_time_ms": 5.2, "accuracy": 1.0},
    )

    # Track optimization progress
    tracker.log_optimization(
        iteration=50,
        fitness=0.95,
        best_weights=[0.1, 0.2, ...],
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import json
import time
from pathlib import Path
from datetime import datetime


class TrackingBackend(Enum):
    """Available experiment tracking backends."""
    WANDB = "wandb"
    MLFLOW = "mlflow"
    LOCAL = "local"  # JSON files
    NONE = "none"


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    project: str = "some-experiments"
    entity: Optional[str] = None  # For wandb
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    backend: TrackingBackend = TrackingBackend.LOCAL
    log_frequency: int = 1  # Log every N generations


@dataclass
class GenerationRecord:
    """Record of a single code generation."""
    generation: int
    task: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    execution_time_ms: float
    success: bool
    feedback_type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationRecord:
    """Record of optimization progress."""
    iteration: int
    fitness: float
    best_weights: List[float]
    population_diversity: float
    timestamp: float = field(default_factory=time.time)


class ExperimentTracker:
    """
    Experiment tracker for SOME.

    Supports multiple backends:
    - Weights & Biases (wandb)
    - MLflow
    - Local JSON files
    - No-op (disabled)
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.backend = self.config.backend
        self._run = None
        self._mlflow_run = None
        self._local_dir: Optional[Path] = None
        self._generation_count = 0
        self._optimization_count = 0

        self._init_backend()

    def _init_backend(self):
        """Initialize the tracking backend."""
        if self.backend == TrackingBackend.WANDB:
            self._init_wandb()
        elif self.backend == TrackingBackend.MLFLOW:
            self._init_mlflow()
        elif self.backend == TrackingBackend.LOCAL:
            self._init_local()

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self._run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=self.config.name,
                tags=self.config.tags,
                notes=self.config.notes,
            )
        except ImportError:
            print("Warning: wandb not installed. Falling back to local tracking.")
            self.backend = TrackingBackend.LOCAL
            self._init_local()

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow
            mlflow.set_experiment(self.config.project)
            self._mlflow_run = mlflow.start_run(
                run_name=self.config.name,
                tags={tag: tag for tag in self.config.tags},
            )
        except ImportError:
            print("Warning: mlflow not installed. Falling back to local tracking.")
            self.backend = TrackingBackend.LOCAL
            self._init_local()

    def _init_local(self):
        """Initialize local JSON file tracking."""
        base_dir = Path.home() / ".ncpu" / "experiments"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create experiment directory
        exp_name = self.config.name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._local_dir = base_dir / f"{self.config.project}_{exp_name}"
        self._local_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self._local_dir / "config.json"
        config_path.write_text(json.dumps({
            "project": self.config.project,
            "name": exp_name,
            "tags": self.config.tags,
            "notes": self.config.notes,
            "backend": self.backend.value,
        }))

    def log_generation(self, generation: int, task: str, parameters: Dict[str, Any],
                      metrics: Dict[str, float], execution_time_ms: float,
                      success: bool, feedback_type: str = "success"):
        """Log a code generation event."""
        self._generation_count += 1

        # Skip if not logging frequency
        if self._generation_count % self.config.log_frequency != 0:
            return

        record = GenerationRecord(
            generation=generation,
            task=task,
            parameters=parameters,
            metrics=metrics,
            execution_time_ms=execution_time_ms,
            success=success,
            feedback_type=feedback_type,
        )

        if self.backend == TrackingBackend.WANDB and self._run:
            self._run.log({
                "generation": generation,
                "task": task,
                "execution_time_ms": execution_time_ms,
                "success": success,
                **{f"metrics/{k}": v for k, v in metrics.items()},
                **{f"params/{k}": v for k, v in parameters.items()},
            })

        elif self.backend == TrackingBackend.MLFLOW and self._mlflow_run:
            import mlflow
            mlflow.log_metrics({
                f"generation_{k}": v for k, v in metrics.items()
            }, step=generation)

        elif self.backend == TrackingBackend.LOCAL:
            self._log_local("generations", record)

    def log_optimization(self, iteration: int, fitness: float, best_weights: List[float],
                        population_diversity: float = 0.0):
        """Log optimization progress."""
        self._optimization_count += 1

        # Skip if not logging frequency
        if self._optimization_count % self.config.log_frequency != 0:
            return

        record = OptimizationRecord(
            iteration=iteration,
            fitness=fitness,
            best_weights=best_weights,
            population_diversity=population_diversity,
        )

        if self.backend == TrackingBackend.WANDB and self._run:
            self._run.log({
                "iteration": iteration,
                "fitness": fitness,
                "population_diversity": population_diversity,
                "best_weight_0": best_weights[0] if best_weights else 0,
            })

        elif self.backend == TrackingBackend.MLFLOW and self._mlflow_run:
            import mlflow
            mlflow.log_metrics({
                "fitness": fitness,
                "diversity": population_diversity,
            }, step=iteration)

        elif self.backend == TrackingBackend.LOCAL:
            self._log_local("optimization", record)

    def log_execution(self, execution_time_ms: float, memory_bytes: int,
                     cpu_cycles: int, cache_hits: int, cache_misses: int):
        """Log execution metrics."""
        if self.backend == TrackingBackend.WANDB and self._run:
            self._run.log({
                "execution_time_ms": execution_time_ms,
                "memory_bytes": memory_bytes,
                "cpu_cycles": cpu_cycles,
                "cache_hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
            })

        elif self.backend == TrackingBackend.MLFLOW and self._mlflow_run:
            import mlflow
            mlflow.log_metrics({
                "execution_time_ms": execution_time_ms,
                "memory_bytes": memory_bytes,
            })

    def log_feedback(self, feedback_type: str, improvement_direction: str,
                    correctness_score: float):
        """Log feedback signal from execution."""
        if self.backend == TrackingBackend.WANDB and self._run:
            self._run.log({
                "feedback_type": feedback_type,
                "improvement_direction": improvement_direction,
                "correctness_score": correctness_score,
            })

    def _log_local(self, category: str, record: Any):
        """Log to local JSON file."""
        if self._local_dir is None:
            return

        # Create category file
        file_path = self._local_dir / f"{category}.jsonl"

        # Append record
        with open(file_path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    def log_table(self, name: str, data: List[Dict[str, Any]]):
        """Log a table of data (e.g., ablation study results)."""
        if self.backend == TrackingBackend.WANDB and self._run:
            import wandb
            table = wandb.Table(data=data)
            self._run.log({name: table})

        elif self.backend == TrackingBackend.MLFLOW and self._mlflow_run:
            import mlflow
            # MLflow doesn't have direct table support, log as artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(data, f)
                mlflow.log_artifact(f.name, artifact_path=name)

    def finish(self):
        """Finish the experiment."""
        if self.backend == TrackingBackend.WANDB and self._run:
            self._run.finish()

        elif self.backend == TrackingBackend.MLFLOW and self._mlflow_run:
            import mlflow
            mlflow.end_run()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracker(backend: str = "local", **kwargs) -> ExperimentTracker:
    """Create an experiment tracker with the specified backend."""
    config = ExperimentConfig(
        backend=TrackingBackend(backend),
        **kwargs,
    )
    return ExperimentTracker(config)


# =============================================================================
# Demo Usage
# =============================================================================

def demo():
    """Demo showing how to use the experiment tracker."""
    # Local tracking (no external dependencies)
    tracker = ExperimentTracker(
        config=ExperimentConfig(
            project="demo",
            name="demo-run",
            backend=TrackingBackend.LOCAL,
        )
    )

    # Simulate generations
    for gen in range(10):
        tracker.log_generation(
            generation=gen,
            task="sort_array",
            parameters={"algorithm": "quicksort", "pivot": "median"},
            metrics={"accuracy": 0.95 + gen * 0.005, "throughput": 1000 + gen * 50},
            execution_time_ms=5.0 - gen * 0.1,
            success=True,
        )

    # Simulate optimization
    import math
    for i in range(50):
        tracker.log_optimization(
            iteration=i,
            fitness=0.5 + 0.5 * (1 - math.exp(-i / 20)),
            best_weights=[0.1 * math.sin(i * 0.1), 0.2, 0.3],
            population_diversity=0.8 * math.exp(-i / 30),
        )

    tracker.finish()
    print("Demo complete. Check ~/.ncpu/experiments/ for results.")


if __name__ == "__main__":
    demo()
