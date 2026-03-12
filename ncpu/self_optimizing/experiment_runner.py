"""
Experiment Runner with SVM Weight Sharing

Runs super-autoresearch experiments with efficient weight sharing
via Shared Virtual Memory protocol.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
import time


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    preset: str = "small"
    experiment_minutes: int = 5
    warm_start: bool = True
    use_checkpointing: bool = True


@dataclass
class Experiment:
    """Represents a single experiment"""
    experiment_id: str
    config: ExperimentConfig
    code_modifications: str
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[dict] = None


@dataclass
class ModelWeights:
    """Model weights that can be shared via SVM"""
    weights_id: str
    data: np.ndarray
    metadata: dict = field(default_factory=dict)


class SOMExperimentRunner:
    """
    Runs super-autoresearch experiments with SVM weight sharing.

    Key features:
    - Persistent workers for parallel execution
    - SVM for efficient weight sharing between runs
    - Warm-start from base weights
    - Checkpoint management
    """

    def __init__(
        self,
        num_workers: int = 4,
        device_id: int = 0,
    ):
        from ncpu.self_optimizing import PersistentGpuWorkersProtocol
        from ncpu.self_optimizing.protocols import SharedVirtualMemoryProtocol

        self.num_workers = num_workers
        self.device_id = device_id

        # Initialize protocols
        self.workers = PersistentGpuWorkersProtocol(num_workers=num_workers)
        self.svm = SharedVirtualMemoryProtocol(devices=[device_id])

        # State
        self.experiments: dict[str, Experiment] = {}
        self.shared_weights: Optional[ModelWeights] = None
        self.weight_region = None

    def initialize(self):
        """Initialize worker pool"""
        self.workers.initialize()

    def shutdown(self):
        """Shutdown all components"""
        self.workers.shutdown()

    def allocate_shared_weights(self, weights: ModelWeights):
        """
        Allocate shared memory for model weights.

        This allows multiple workers to read weights directly
        without GPU-to-CPU-to-GPU copies.
        """
        self.shared_weights = weights

        # Allocate SVM region
        self.weight_region = self.svm.allocate_for_tensor(
            name=weights.weights_id,
            data=weights.data,
            devices=[self.device_id],
        )

        print(f"Allocated shared weights: {weights.weights_id} ({weights.data.nbytes} bytes)")

    def warm_start_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> ModelWeights:
        """
        Load checkpoint and prepare for warm-start experiments.

        Uses SVM to share weights between workers efficiently.
        """
        # In real impl: load actual checkpoint
        # For demo: create dummy weights
        weights = ModelWeights(
            weights_id="base_model",
            data=np.random.randn(100, 100).astype(np.float32),
            metadata={"source": checkpoint_path, "val_bpb": 2.0}
        )

        self.allocate_shared_weights(weights)

        return weights

    def submit_experiment(
        self,
        experiment: Experiment,
    ) -> str:
        """
        Submit an experiment to the worker pool.

        If warm_start is enabled, workers can read shared weights directly.
        """
        # Get weight pointer if available
        weight_ptr = None
        if self.shared_weights and experiment.config.warm_start:
            weight_ptr = self.svm.get_pointer(
                self.shared_weights.weights_id,
                self.device_id
            )

        # Submit to worker
        work_id = self.workers.submit_work(
            operation="run_experiment",
            inputs={
                "experiment_id": experiment.experiment_id,
                "config": experiment.config.__dict__,
                "modifications": experiment.code_modifications,
                "weight_ptr": weight_ptr,
            },
            outputs={},
        )

        experiment.status = "running"
        self.experiments[experiment.experiment_id] = experiment

        return work_id

    def submit_parallel_experiments(
        self,
        experiments: list[Experiment],
    ) -> list[str]:
        """
        Submit multiple experiments to run in parallel.
        """
        work_ids = []

        for exp in experiments:
            work_id = self.submit_experiment(exp)
            work_ids.append(work_id)

        return work_ids

    def wait_for_completion(
        self,
        timeout_seconds: int = 600,
    ) -> dict[str, Experiment]:
        """
        Wait for all experiments to complete.
        """
        start = time.time()

        while time.time() - start < timeout_seconds:
            # Check if all experiments completed
            all_done = all(
                exp.status in ["completed", "failed"]
                for exp in self.experiments.values()
            )

            if all_done:
                break

            time.sleep(5)  # Check every 5 seconds

        return self.experiments

    def get_best_result(self) -> Optional[Experiment]:
        """Get the best experiment result (lowest val_bpb)"""
        completed = [
            exp for exp in self.experiments.values()
            if exp.status == "completed" and exp.result
        ]

        if not completed:
            return None

        return min(completed, key=lambda e: e.result.get("val_bpb", float('inf')))

    def share_best_weights(self) -> Optional[ModelWeights]:
        """
        Get the best performing weights for sharing.

        Useful for warm-starting next round of experiments.
        """
        best = self.get_best_result()
        if not best:
            return None

        # In real impl: extract actual weights
        return ModelWeights(
            weights_id=f"best_from_{best.experiment_id}",
            data=np.random.randn(100, 100).astype(np.float32),
            metadata=best.result
        )


class CheckpointManager:
    """
    Manages training checkpoints with SVM support.

    Enables efficient checkpoint sharing between experiments.
    """

    def __init__(self, svm):
        self.svm = svm
        self.checkpoints: dict[str, ModelWeights] = {}

    def save_checkpoint(
        self,
        checkpoint_id: str,
        weights: np.ndarray,
        metadata: dict,
    ) -> ModelWeights:
        """Save a checkpoint to SVM"""
        model_weights = ModelWeights(
            weights_id=checkpoint_id,
            data=weights,
            metadata=metadata,
        )

        # Store in SVM
        self.svm.allocate_for_tensor(
            name=checkpoint_id,
            data=weights,
        )

        self.checkpoints[checkpoint_id] = model_weights

        return model_weights

    def load_checkpoint(self, checkpoint_id: str) -> Optional[ModelWeights]:
        """Load a checkpoint from SVM"""
        return self.checkpoints.get(checkpoint_id)

    def share_checkpoint(self, checkpoint_id: str, device_id: int = 0):
        """Share a checkpoint with a specific device"""
        weights = self.checkpoints.get(checkpoint_id)
        if weights:
            return self.svm.get_pointer(checkpoint_id, device_id)
        return None


def demo():
    """Demo of experiment runner"""
    print("=== SOM Experiment Runner Demo ===\n")

    # Create runner
    runner = SOMExperimentRunner(num_workers=4)
    runner.initialize()

    # Warm-start from checkpoint
    print("Loading checkpoint for warm-start...")
    weights = runner.warm_start_from_checkpoint("baseline_run")

    # Create experiments
    experiments = []
    for i in range(4):
        exp = Experiment(
            experiment_id=f"exp_{i}",
            config=ExperimentConfig(
                preset="small",
                warm_start=True,
            ),
            code_modifications=f"# Modification {i}",
        )
        experiments.append(exp)

    # Submit in parallel
    print(f"Submitting {len(experiments)} experiments...")
    runner.submit_parallel_experiments(experiments)

    # Wait for completion
    print("Waiting for experiments...")
    results = runner.wait_for_completion(timeout_seconds=10)

    # Get best
    best = runner.get_best_result()
    if best:
        print(f"Best experiment: {best.experiment_id}")
        print(f"val_bpb: {best.result.get('val_bpb', 'N/A')}")

    runner.shutdown()


if __name__ == "__main__":
    demo()
