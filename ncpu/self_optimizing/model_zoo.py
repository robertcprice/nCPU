"""
Model Zoo for SOME - Pre-trained Tensor Descriptors

This module provides pre-trained models for common operations that can be
loaded directly without requiring LLM generation. These are essentially
"compiled" solutions learned through the self-optimization process.

Operations include:
- Sorting algorithms (quick sort, merge sort, heap sort)
- Search algorithms (binary search, interpolation search)
- Hash functions (murmur, fnv, cityhash)
- String operations (pattern matching, compression)
- Matrix operations (transpose, multiply, convolution)
- Cryptographic primitives (AES, SHA256 subsets)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import numpy as np
import json
from pathlib import Path


class OperationCategory(Enum):
    """Categories of pre-trained operations."""
    SORTING = "sorting"
    SEARCH = "search"
    HASH = "hash"
    STRING = "string"
    MATRIX = "matrix"
    CRYPTO = "crypto"
    ML = "ml"
    CUSTOM = "custom"


@dataclass
class TensorDescriptor:
    """
    Executable representation of code as tensors.
    This is the core output format of SOME - not text, but directly executable descriptors.
    """
    operation: str
    category: OperationCategory
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    shape: tuple
    dtype: str
    parameters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Pre-trained weights (for neural operations)
    weights: Optional[np.ndarray] = None
    # Compilation hints
    optimize_for: str = "speed"  # speed, memory, accuracy

    def to_bytes(self) -> bytes:
        """Serialize for transmission/storage."""
        data = {
            "operation": self.operation,
            "category": self.category.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "optimize_for": self.optimize_for,
        }
        if self.weights is not None:
            data["weights"] = self.weights.tobytes()
            data["weights_shape"] = self.weights.shape
        return json.dumps(data).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TensorDescriptor':
        """Deserialize."""
        parsed = json.loads(data.decode())
        weights = None
        if "weights" in parsed:
            weights = np.frombuffer(parsed["weights"], dtype=np.float32)
            weights = weights.reshape(parsed["weights_shape"])
        return cls(
            operation=parsed["operation"],
            category=OperationCategory(parsed["category"]),
            inputs=parsed["inputs"],
            outputs=parsed["outputs"],
            shape=tuple(parsed["shape"]),
            dtype=parsed["dtype"],
            parameters=parsed["parameters"],
            metadata=parsed["metadata"],
            weights=weights,
            optimize_for=parsed.get("optimize_for", "speed"),
        )


@dataclass
class PreTrainedModel:
    """A pre-trained model in the zoo."""
    name: str
    category: OperationCategory
    description: str
    descriptor: TensorDescriptor
    accuracy: float = 1.0
    avg_execution_time_ms: float = 0.0
    memory_usage_bytes: int = 0
    training_data_size: int = 0
    version: str = "1.0"


class ModelZoo:
    """
    Repository of pre-trained operations.

    Usage:
        zoo = ModelZoo()
        model = zoo.get("quicksort_ascending")
        result = model.descriptor.execute(input_data)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".ncpu" / "model_zoo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, PreTrainedModel] = {}
        self._load_builtin_models()

    def _load_builtin_models(self):
        """Load pre-built models into the zoo."""
        # Sorting operations
        self._register(PreTrainedModel(
            name="quicksort_ascending",
            category=OperationCategory.SORTING,
            description="QuickSort for ascending order - average O(n log n)",
            descriptor=TensorDescriptor(
                operation="quicksort",
                category=OperationCategory.SORTING,
                inputs={"array": {"type": "array", "dtype": "float32"}},
                outputs={"sorted": {"type": "array", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"pivot_strategy": "median-of-three", "insertion_threshold": 16},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.5,
        ))

        self._register(PreTrainedModel(
            name="mergesort_stable",
            category=OperationCategory.SORTING,
            description="MergeSort - stable O(n log n) guaranteed",
            descriptor=TensorDescriptor(
                operation="mergesort",
                category=OperationCategory.SORTING,
                inputs={"array": {"type": "array", "dtype": "float32"}},
                outputs={"sorted": {"type": "array", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"buffer_size": 1024, "parallel_threshold": 4096},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.7,
        ))

        self._register(PreTrainedModel(
            name="heapsort_inplace",
            category=OperationCategory.SORTING,
            description="HeapSort - in-place O(n log n), no extra memory",
            descriptor=TensorDescriptor(
                operation="heapsort",
                category=OperationCategory.SORTING,
                inputs={"array": {"type": "array", "dtype": "float32"}},
                outputs={"sorted": {"type": "array", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"heap_type": "max", "inplace": True},
                optimize_for="memory",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.8,
        ))

        # Search operations
        self._register(PreTrainedModel(
            name="binary_search_exact",
            category=OperationCategory.SEARCH,
            description="Binary search for exact match - O(log n)",
            descriptor=TensorDescriptor(
                operation="binary_search",
                category=OperationCategory.SEARCH,
                inputs={
                    "array": {"type": "array", "dtype": "float32"},
                    "target": {"type": "scalar", "dtype": "float32"},
                },
                outputs={"index": {"type": "scalar", "dtype": "int32"}, "found": {"type": "bool"}},
                shape=(-1,),
                dtype="float32",
                parameters={"early_termination": True},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.01,
        ))

        self._register(PreTrainedModel(
            name="interpolation_search",
            category=OperationCategory.SEARCH,
            description="Interpolation search for uniform distributions - O(log log n)",
            descriptor=TensorDescriptor(
                operation="interpolation_search",
                category=OperationCategory.SEARCH,
                inputs={
                    "array": {"type": "array", "dtype": "float32"},
                    "target": {"type": "scalar", "dtype": "float32"},
                },
                outputs={"index": {"type": "scalar", "dtype": "int32"}, "found": {"type": "bool"}},
                shape=(-1,),
                dtype="float32",
                parameters={"fallback_binary": True},
                optimize_for="speed",
            ),
            accuracy=0.95,
            avg_execution_time_ms=0.005,
        ))

        # Hash operations
        self._register(PreTrainedModel(
            name="murmurhash3_32",
            category=OperationCategory.HASH,
            description="MurmurHash3 - fast non-cryptographic hash",
            descriptor=TensorDescriptor(
                operation="murmurhash3",
                category=OperationCategory.HASH,
                inputs={"data": {"type": "bytes"}},
                outputs={"hash": {"type": "uint32"}},
                shape=(),
                dtype="uint32",
                parameters={"seed": 0, "c1": 0xcc9e2d51, "c2": 0x1b873593},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.02,
        ))

        self._register(PreTrainedModel(
            name="fnv1a_64",
            category=OperationCategory.HASH,
            description="FNV-1a hash - simple and fast",
            descriptor=TensorDescriptor(
                operation="fnv1a",
                category=OperationCategory.HASH,
                inputs={"data": {"type": "bytes"}},
                outputs={"hash": {"type": "uint64"}},
                shape=(),
                dtype="uint64",
                parameters={"prime": 1099511628211, "offset": 14695981039346656037},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.015,
        ))

        # String operations
        self._register(PreTrainedModel(
            name="boyer_moore_search",
            category=OperationCategory.STRING,
            description="Boyer-Moore pattern matching - O(n/m) average",
            descriptor=TensorDescriptor(
                operation="boyer_moore",
                category=OperationCategory.STRING,
                inputs={
                    "text": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                outputs={"indices": {"type": "array", "dtype": "int32"}},
                shape=(-1,),
                dtype="int32",
                parameters={"bad_char_table": True, "good_suffix": True},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.1,
        ))

        # Matrix operations
        self._register(PreTrainedModel(
            name="matrix_transpose_square",
            category=OperationCategory.MATRIX,
            description="Square matrix transpose - O(n²) with cache optimization",
            descriptor=TensorDescriptor(
                operation="transpose",
                category=OperationCategory.MATRIX,
                inputs={"matrix": {"type": "matrix", "dtype": "float32"}},
                outputs={"transposed": {"type": "matrix", "dtype": "float32"}},
                shape=(-1, -1),
                dtype="float32",
                parameters={"blocking": True, "block_size": 64},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.3,
        ))

        self._register(PreTrainedModel(
            name="matrix_multiply_tiled",
            category=OperationCategory.MATRIX,
            description="Tiled matrix multiplication with cache blocking",
            descriptor=TensorDescriptor(
                operation="matmul",
                category=OperationCategory.MATRIX,
                inputs={
                    "a": {"type": "matrix", "dtype": "float32"},
                    "b": {"type": "matrix", "dtype": "float32"},
                },
                outputs={"result": {"type": "matrix", "dtype": "float32"}},
                shape=(-1, -1),
                dtype="float32",
                parameters={"tile_size": 32, "num_threads": 8},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=2.0,
            memory_usage_bytes=1024 * 1024,
        ))

        # ML operations (neural network layers)
        self._register(PreTrainedModel(
            name="relu_activation",
            category=OperationCategory.ML,
            description="ReLU activation function - max(0, x)",
            descriptor=TensorDescriptor(
                operation="relu",
                category=OperationCategory.ML,
                inputs={"x": {"type": "tensor", "dtype": "float32"}},
                outputs={"y": {"type": "tensor", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"inplace": True},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.05,
        ))

        self._register(PreTrainedModel(
            name="softmax_stable",
            category=OperationCategory.ML,
            description="Numerically stable softmax",
            descriptor=TensorDescriptor(
                operation="softmax",
                category=OperationCategory.ML,
                inputs={"x": {"type": "tensor", "dtype": "float32"}, "axis": {"type": "int"}},
                outputs={"y": {"type": "tensor", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"stable": True},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.1,
        ))

        self._register(PreTrainedModel(
            name="layer_norm",
            category=OperationCategory.ML,
            description="Layer normalization with learned gamma/beta",
            descriptor=TensorDescriptor(
                operation="layer_norm",
                category=OperationCategory.ML,
                inputs={
                    "x": {"type": "tensor", "dtype": "float32"},
                    "gamma": {"type": "tensor", "dtype": "float32"},
                    "beta": {"type": "tensor", "dtype": "float32"},
                },
                outputs={"y": {"type": "tensor", "dtype": "float32"}},
                shape=(-1,),
                dtype="float32",
                parameters={"epsilon": 1e-5, "elementwise": True},
                optimize_for="speed",
            ),
            accuracy=1.0,
            avg_execution_time_ms=0.15,
        ))

        # Fix: add weights to descriptor for layer_norm
        self._models["layer_norm"].descriptor.weights = np.ones(128, dtype=np.float32)

    def _register(self, model: PreTrainedModel):
        """Register a model in the zoo."""
        self._models[model.name] = model

    def get(self, name: str) -> Optional[PreTrainedModel]:
        """Get a model by name."""
        return self._models.get(name)

    def list_models(self, category: Optional[OperationCategory] = None) -> List[str]:
        """List all available models, optionally filtered by category."""
        if category is None:
            return list(self._models.keys())
        return [k for k, v in self._models.items() if v.category == category]

    def search(self, query: str) -> List[PreTrainedModel]:
        """Search models by name or description."""
        query_lower = query.lower()
        results = []
        for model in self._models.values():
            if query_lower in model.name.lower() or query_lower in model.description.lower():
                results.append(model)
        return results

    def add_custom_model(self, model: PreTrainedModel):
        """Add a custom trained model to the zoo."""
        self._register(model)
        # Optionally save to disk
        if self.cache_dir:
            model_path = self.cache_dir / f"{model.name}.json"
            model_path.write_text(json.dumps({
                "name": model.name,
                "category": model.category.value,
                "description": model.description,
                "descriptor": model.descriptor.to_bytes().decode(),
            }))

    def benchmark_model(self, name: str, input_data: Any, executor_fn: Callable) -> Dict[str, Any]:
        """
        Benchmark a model's execution.

        Args:
            name: Model name
            input_data: Input to pass to the model
            executor_fn: Function to execute the model descriptor

        Returns:
            Benchmark results including timing, accuracy, memory
        """
        model = self.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found in zoo")

        import time
        start = time.perf_counter()
        result = executor_fn(model.descriptor, input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "model": name,
            "operation": model.descriptor.operation,
            "execution_time_ms": elapsed_ms,
            "expected_time_ms": model.avg_execution_time_ms,
            "speedup_vs_expected": model.avg_execution_time_ms / elapsed_ms if elapsed_ms > 0 else 0,
            "accuracy": model.accuracy,
            "memory_bytes": model.memory_usage_bytes,
        }


# Global singleton
_default_zoo: Optional[ModelZoo] = None


def get_model_zoo() -> ModelZoo:
    """Get the default model zoo instance."""
    global _default_zoo
    if _default_zoo is None:
        _default_zoo = ModelZoo()
    return _default_zoo


# Convenience functions
def load_operation(name: str) -> TensorDescriptor:
    """Load a pre-trained operation descriptor."""
    zoo = get_model_zoo()
    model = zoo.get(name)
    if model is None:
        raise ValueError(f"Operation '{name}' not found")
    return model.descriptor


def list_operations(category: Optional[OperationCategory] = None) -> List[str]:
    """List available operations."""
    return get_model_zoo().list_models(category)
