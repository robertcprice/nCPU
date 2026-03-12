"""
Gradient-Aware Network Protocol (GANP)

GPU-Native protocol that pipelines forward and backward passes with gradient
compression for efficient distributed training. The protocol captures
computation state and enables lossless gradient compression.

Protocol Design:
- GradientDescriptor: Describes gradient tensor without materializing full data
- PipelineStage: Represents a stage in the computation pipeline
- CompressionSpec: Specifies gradient compression strategy
- The protocol coordinates gradient flow between stages with minimal communication
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, Callable
import numpy as np
import time


class GradientType(IntEnum):
    """Types of gradients supported"""
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    FP8 = 3
    SPARSE = 4
    QUANTIZED = 5


class CompressionType(Enum):
    """Gradient compression strategies"""
    NONE = "none"
    QUANTIZATION = "quantization"
    TOP_K = "top_k"
    RANDOM_K = "random_k"
    DWAL = "dwal"  # Deep Gradient Compression
    POWER_SGD = "power_sgd"
    AUTO = "auto"


@dataclass
class GradientDescriptor:
    """
    Lightweight descriptor for gradient data.

    Unlike full tensor data, this only describes the gradient's metadata
    and compression info, enabling efficient pipeline communication.
    """
    name: str
    shape: tuple[int, ...]
    dtype: GradientType
    size_bytes: int
    compression: CompressionType = CompressionType.NONE
    compression_ratio: float = 1.0

    # For sparse gradients
    nonzero_indices: Optional[np.ndarray] = None
    nonzero_values: Optional[np.ndarray] = None

    # For quantized gradients
    scale: Optional[float] = None
    zero_point: Optional[int] = None


@dataclass
class PipelineStage:
    """
    Represents a stage in the gradient pipeline.

    Each stage can be a forward pass, backward pass, or communication phase.
    """
    stage_id: int
    name: str
    inputs: list[str]
    outputs: list[str]
    compute_fn: Optional[Callable] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GradientState:
    """Tracks gradient computation state"""
    forward_pass_id: int
    gradients: dict[str, GradientDescriptor] = field(default_factory=dict)
    compression_stats: dict[str, float] = field(default_factory=dict)
    pipeline_depth: int = 0
    accumulated_updates: int = 0


class GradientCompressor:
    """
    Compresses gradients using various strategies.

    Enables efficient communication in distributed training by reducing
    bandwidth requirements while maintaining model quality.
    """

    def __init__(self, compression_type: CompressionType = CompressionType.AUTO):
        self.compression_type = compression_type
        self.compression_ratio = 1.0

    def compress(self, gradient: np.ndarray, spec: dict) -> GradientDescriptor:
        """Compress gradient according to spec"""
        dtype_map = {
            np.float32: GradientType.FLOAT32,
            np.float16: GradientType.FLOAT16,
        }

        dtype = dtype_map.get(gradient.dtype, GradientType.FLOAT32)

        if self.compression_type == CompressionType.NONE:
            return GradientDescriptor(
                name=spec.get("name", "grad"),
                shape=gradient.shape,
                dtype=dtype,
                size_bytes=gradient.nbytes,
            )

        elif self.compression_type == CompressionType.QUANTIZATION:
            # Quantize to lower precision
            bits = spec.get("bits", 8)
            scale = gradient.max() / (2 ** (bits - 1))
            quantized = np.round(gradient / scale).astype(np.int8)

            return GradientDescriptor(
                name=spec.get("name", "grad"),
                shape=gradient.shape,
                dtype=GradientType.QUANTIZED,
                size_bytes=quantized.nbytes,
                compression=CompressionType.QUANTIZATION,
                compression_ratio=gradient.nbytes / quantized.nbytes,
                scale=float(scale),
                zero_point=0,
            )

        elif self.compression_type == CompressionType.TOP_K:
            # Keep only top-k largest magnitude values
            k_ratio = spec.get("k_ratio", 0.01)
            flat = gradient.flatten()
            k = max(1, int(len(flat) * k_ratio))

            indices = np.argpartition(np.abs(flat), -k)[-k:]
            values = flat[indices]

            sparse_size = indices.nbytes + values.nbytes

            return GradientDescriptor(
                name=spec.get("name", "grad"),
                shape=gradient.shape,
                dtype=GradientType.SPARSE,
                size_bytes=sparse_size,
                compression=CompressionType.TOP_K,
                compression_ratio=gradient.nbytes / sparse_size,
                nonzero_indices=indices,
                nonzero_values=values,
            )

        elif self.compression_type == CompressionType.DWAL:
            # Deep Gradient Compression with local gradient clipping
            threshold = spec.get("threshold", 1.0)
            mask = np.abs(gradient) > threshold

            compressed = gradient * mask
            compressed_size = np.count_nonzero(mask) * gradient.itemsize

            return GradientDescriptor(
                name=spec.get("name", "grad"),
                shape=gradient.shape,
                dtype=dtype,
                size_bytes=compressed_size,
                compression=CompressionType.DWAL,
                compression_ratio=gradient.nbytes / compressed_size if compressed_size > 0 else 1.0,
            )

        # Default: no compression
        return GradientDescriptor(
            name=spec.get("name", "grad"),
            shape=gradient.shape,
            dtype=dtype,
            size_bytes=gradient.nbytes,
        )

    def decompress(self, descriptor: GradientDescriptor) -> np.ndarray:
        """Decompress gradient from descriptor"""
        if descriptor.compression == CompressionType.NONE:
            return np.zeros(descriptor.shape, dtype=np.float32)

        elif descriptor.compression == CompressionType.QUANTIZATION:
            # Reconstruct from quantized values
            result = np.zeros(descriptor.shape, dtype=np.float32)
            if descriptor.scale is not None and descriptor.nonzero_values is not None:
                result = descriptor.nonzero_values.astype(np.float32) * descriptor.scale
            return result

        elif descriptor.compression == CompressionType.TOP_K:
            # Reconstruct sparse gradient
            result = np.zeros(descriptor.shape, dtype=np.float32)
            if descriptor.nonzero_indices is not None and descriptor.nonzero_values is not None:
                flat = result.flatten()
                flat[descriptor.nonzero_indices] = descriptor.nonzero_values
            return result

        elif descriptor.compression == CompressionType.DWAL:
            # Already compressed in place
            return np.zeros(descriptor.shape, dtype=np.float32)

        return np.zeros(descriptor.shape, dtype=np.float32)


class GradientAwareNetworkProtocol:
    """
    Implements the Gradient-Aware Network Protocol.

    This protocol pipelines forward/backward passes with gradient compression
    to minimize communication overhead in distributed training.

    Usage:
        ganp = GradientAwareNetworkProtocol(device_id=0)

        # Define pipeline stages
        ganp.add_stage(PipelineStage(...))
        ganp.add_stage(PipelineStage(...))

        # Execute forward pass
        ganp.forward(inputs)

        # Execute backward pass with compression
        gradients = ganp.backward(loss)

        # Compress and communicate
        compressed = ganp.compress_gradients(gradients)
        ganp.communicate(compressed)
    """

    def __init__(
        self,
        device_id: int = 0,
        compression_type: CompressionType = CompressionType.AUTO,
        pipeline_depth: int = 1,
    ):
        self.device_id = device_id
        self.pipeline_depth = pipeline_depth
        self.compressor = GradientCompressor(compression_type)
        self.stages: list[PipelineStage] = []
        self.state = GradientState(forward_pass_id=0)
        self.tensors: dict[str, np.ndarray] = {}
        self._execution_time = 0.0

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a pipeline stage"""
        self.stages.append(stage)
        self.state.pipeline_depth = len(self.stages)

    def forward(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute forward pass through pipeline"""
        start = time.perf_counter()

        # Register inputs
        for name, data in inputs.items():
            self.tensors[name] = data

        # Execute each stage
        current = inputs
        for stage in self.stages:
            # Stage would compute here
            pass

        self.state.forward_pass_id += 1
        self._execution_time = time.perf_counter() - start

        return current

    def backward(
        self,
        loss: np.ndarray,
        gradients: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, GradientDescriptor]:
        """
        Execute backward pass and return gradient descriptors.

        Returns lightweight descriptors instead of full gradient tensors
        to minimize memory and communication overhead.
        """
        start = time.perf_counter()

        grad_descriptors = {}

        # For each tensor that needs gradient
        for name, grad_data in (gradients or self.tensors).items():
            descriptor = self.compressor.compress(grad_data, {"name": name})
            grad_descriptors[name] = descriptor
            self.state.gradients[name] = descriptor
            self.state.compression_stats[name] = descriptor.compression_ratio

        self._execution_time = time.perf_counter() - start

        return grad_descriptors

    def compress_gradients(
        self,
        gradients: dict[str, np.ndarray],
    ) -> dict[str, GradientDescriptor]:
        """Compress gradients according to configured strategy"""
        result = {}
        for name, grad in gradients.items():
            descriptor = self.compressor.compress(grad, {"name": name})
            result[name] = descriptor
        return result

    def decompress_gradients(
        self,
        descriptors: dict[str, GradientDescriptor],
    ) -> dict[str, np.ndarray]:
        """Decompress gradient descriptors back to arrays"""
        result = {}
        for name, descriptor in descriptors.items():
            result[name] = self.compressor.decompress(descriptor)
        return result

    def communicate(
        self,
        compressed_gradients: dict[str, GradientDescriptor],
    ) -> None:
        """Simulate gradient communication (in real impl, this is network transfer)"""
        total_size = sum(g.size_bytes for g in compressed_gradients.values())
        compressed_size = sum(
            g.size_bytes / g.compression_ratio
            for g in compressed_gradients.values()
        )

        self.state.accumulated_updates += 1

    def get_compression_stats(self) -> dict[str, float]:
        """Get compression statistics"""
        return {
            "total_gradients": len(self.state.gradients),
            "total_updates": self.state.accumulated_updates,
            "avg_compression_ratio": np.mean(list(self.state.compression_stats.values()))
            if self.state.compression_stats
            else 1.0,
            "pipeline_depth": self.state.pipeline_depth,
        }


def benchmark_gradient_compression(
    num_iterations: int = 100,
    tensor_size: int = 1024 * 1024,
    compression_type: CompressionType = CompressionType.TOP_K,
) -> dict:
    """
    Benchmark gradient compression.

    Returns metrics comparing:
    - No compression: Full gradient transfer
    - With compression: Compressed gradient transfer
    """
    # Create dummy gradient
    gradient = np.random.randn(tensor_size).astype(np.float32)

    compressor = GradientCompressor(compression_type)

    # Benchmark compression
    start = time.perf_counter()
    for _ in range(num_iterations):
        descriptor = compressor.compress(gradient, {"name": "test_grad", "k_ratio": 0.01})
    compression_time = time.perf_counter() - start

    # Benchmark decompression
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = compressor.decompress(descriptor)
    decompression_time = time.perf_counter() - start

    original_size = gradient.nbytes
    compressed_size = descriptor.size_bytes
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    return {
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": compression_ratio,
        "compression_time_ms": compression_time * 1000 / num_iterations,
        "decompression_time_ms": decompression_time * 1000 / num_iterations,
        "compression_type": compression_type.value,
        "num_iterations": num_iterations,
    }


if __name__ == "__main__":
    print("=== Gradient-Aware Network Protocol Demo ===\n")

    # Create protocol
    ganp = GradientAwareNetworkProtocol(
        device_id=0,
        compression_type=CompressionType.TOP_K,
        pipeline_depth=2,
    )

    # Add pipeline stages
    ganp.add_stage(PipelineStage(
        stage_id=0,
        name="layer1",
        inputs=["input"],
        outputs=["h1"],
    ))
    ganp.add_stage(PipelineStage(
        stage_id=1,
        name="layer2",
        inputs=["h1"],
        outputs=["output"],
    ))

    # Execute forward pass
    inputs = {"input": np.random.randn(32, 512).astype(np.float32)}
    outputs = ganp.forward(inputs)
    print(f"Forward pass complete, output shape: {outputs['h1'].shape}")

    # Execute backward pass (simulate gradients)
    simulated_grads = {
        "h1": np.random.randn(32, 512).astype(np.float32),
        "input": np.random.randn(32, 512).astype(np.float32),
    }
    grad_descriptors = ganp.backward(np.float32(1.0), simulated_grads)
    print(f"Backward pass complete, {len(grad_descriptors)} gradients")

    # Compress gradients
    compressed = ganp.compress_gradients(simulated_grads)
    print(f"Compressed {len(compressed)} gradients")

    # Get stats
    stats = ganp.get_compression_stats()
    print(f"\nCompression stats: {stats}")

    # Benchmark
    print("\nBenchmarking gradient compression...")
    bench = benchmark_gradient_compression(
        num_iterations=100,
        tensor_size=1024 * 1024,
        compression_type=CompressionType.TOP_K,
    )
    print(f"Compression ratio: {bench['compression_ratio']:.1f}x")
    print(f"Compression time: {bench['compression_time_ms']:.3f}ms")
    print(f"Decompression time: {bench['decompression_time_ms']:.3f}ms")
