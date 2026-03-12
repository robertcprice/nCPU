"""
Tests for new self-optimizing modules: model zoo, experiment tracker, and AutoML.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncpu.self_optimizing.model_zoo import (
    ModelZoo,
    PreTrainedModel,
    TensorDescriptor,
    OperationCategory,
    get_model_zoo,
    load_operation,
    list_operations,
)
from ncpu.self_optimizing.experiment_tracker import (
    ExperimentTracker,
    ExperimentConfig,
    TrackingBackend,
    GenerationRecord,
    OptimizationRecord,
)
from ncpu.self_optimizing.automl import (
    KernelAutoML,
    KernelConfig,
    NeuralArchitectureSearch,
    AutoMLResult,
)


class TestModelZoo:
    """Tests for the Model Zoo."""

    def test_zoo_initialization(self):
        """Test that model zoo loads with built-in models."""
        zoo = ModelZoo()
        models = zoo.list_models()
        assert len(models) > 0
        print(f"Loaded {len(models)} models")

    def test_get_model(self):
        """Test getting a model by name."""
        zoo = ModelZoo()
        model = zoo.get("quicksort_ascending")
        assert model is not None
        assert model.category == OperationCategory.SORTING
        assert model.accuracy == 1.0

    def test_list_by_category(self):
        """Test listing models by category."""
        zoo = ModelZoo()
        sorting_models = zoo.list_models(OperationCategory.SORTING)
        assert len(sorting_models) > 0
        assert all("sort" in m for m in sorting_models)

    def test_search(self):
        """Test model search."""
        zoo = ModelZoo()
        results = zoo.search("sort")
        assert len(results) > 0

    def test_descriptor_serialization(self):
        """Test tensor descriptor serialization."""
        desc = TensorDescriptor(
            operation="test",
            category=OperationCategory.SORTING,
            inputs={},
            outputs={},
            shape=(-1,),
            dtype="float32",
        )
        data = desc.to_bytes()
        restored = TensorDescriptor.from_bytes(data)
        assert restored.operation == desc.operation

    def test_global_zoo(self):
        """Test global zoo singleton."""
        zoo1 = get_model_zoo()
        zoo2 = get_model_zoo()
        assert zoo1 is zoo2

    def test_model_with_weights(self):
        """Test model with pre-trained weights."""
        zoo = ModelZoo()
        model = zoo.get("layer_norm")
        assert model.descriptor.weights is not None
        assert len(model.descriptor.weights) > 0


class TestExperimentTracker:
    """Tests for the Experiment Tracker."""

    def test_local_backend_init(self):
        """Test local backend initialization."""
        config = ExperimentConfig(
            project="test-project",
            name="test-run",
            backend=TrackingBackend.LOCAL,
        )
        tracker = ExperimentTracker(config)
        assert tracker.backend == TrackingBackend.LOCAL

    def test_log_generation(self):
        """Test logging a generation."""
        config = ExperimentConfig(
            project="test-project",
            backend=TrackingBackend.LOCAL,
            log_frequency=1,
        )
        tracker = ExperimentTracker(config)

        tracker.log_generation(
            generation=1,
            task="test_task",
            parameters={"param1": 1.0},
            metrics={"accuracy": 0.95},
            execution_time_ms=10.0,
            success=True,
        )
        tracker.finish()

    def test_log_optimization(self):
        """Test logging optimization progress."""
        config = ExperimentConfig(
            project="test-project",
            backend=TrackingBackend.LOCAL,
            log_frequency=1,
        )
        tracker = ExperimentTracker(config)

        for i in range(5):
            tracker.log_optimization(
                iteration=i,
                fitness=0.5 + i * 0.1,
                best_weights=[0.1, 0.2],
                population_diversity=0.8,
            )
        tracker.finish()

    def test_context_manager(self):
        """Test tracker as context manager."""
        with ExperimentTracker(ExperimentConfig(
            project="test-project",
            backend=TrackingBackend.LOCAL,
        )) as tracker:
            tracker.log_generation(
                generation=1,
                task="test",
                parameters={},
                metrics={},
                execution_time_ms=1.0,
                success=True,
            )


class TestKernelAutoML:
    """Tests for Kernel AutoML."""

    def test_config_defaults(self):
        """Test default kernel config."""
        config = KernelConfig()
        assert config.block_size_x == 256
        assert config.shared_memory_bytes == 16384
        assert config.diff_cpu_ooo_depth == 8

    def test_automl_initialization(self):
        """Test AutoML initialization."""
        automl = KernelAutoML(objective="throughput")
        assert automl.objective == "throughput"
        assert automl.population == []

    def test_population_init(self):
        """Test population initialization."""
        automl = KernelAutoML()
        automl.initialize_population(10)
        assert len(automl.population) == 10

    def test_evolution(self):
        """Test one generation of evolution."""
        automl = KernelAutoML()
        automl.initialize_population(10)
        best = automl.evolve_generation(mutation_rate=0.1)
        assert best > 0
        assert len(automl.history) == 1

    def test_full_search(self):
        """Test full AutoML search."""
        automl = KernelAutoML()
        result = automl.search(
            max_generations=10,
            population_size=10,
            early_stop=False,
        )

        assert isinstance(result, AutoMLResult)
        assert result.best_fitness > 0
        assert result.generations > 0
        assert result.best_config is not None

    def test_best_config(self):
        """Test that best config has expected fields."""
        automl = KernelAutoML()
        result = automl.search(max_generations=5, population_size=5)

        config = result.best_config
        assert hasattr(config, 'block_size_x')
        assert hasattr(config, 'shared_memory_bytes')
        assert hasattr(config, 'diff_cpu_ooo_depth')


class TestNeuralArchitectureSearch:
    """Tests for Neural Architecture Search."""

    def test_random_architecture(self):
        """Test random architecture generation."""
        nas = NeuralArchitectureSearch(max_layers=5)
        arch = nas.random_architecture()

        assert len(arch.genes) > 0
        assert len(arch.genes) <= 5

    def test_nas_search(self):
        """Test full NAS search."""
        nas = NeuralArchitectureSearch(max_layers=3)

        def fitness_fn(genes):
            return sum(1 for g in genes if g.layer_type == "dense")

        result = nas.search("test", fitness_fn, max_generations=5, population_size=5)

        assert result["best_fitness"] >= 0
        assert len(result["best_genes"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
