"""Tests for Gradient-Aware Network Protocol"""

import unittest
import numpy as np
from ncpu.os.gpu.protocols.gradient_aware_network import (
    GradientType,
    CompressionType,
    GradientDescriptor,
    PipelineStage,
    GradientCompressor,
    GradientAwareNetworkProtocol,
    benchmark_gradient_compression,
)


class TestGradientType(unittest.TestCase):
    """Test GradientType enum"""

    def test_all_types_defined(self):
        """Test all gradient types are defined"""
        self.assertEqual(GradientType.FLOAT32, 0)
        self.assertEqual(GradientType.FLOAT16, 1)
        self.assertEqual(GradientType.BFLOAT16, 2)
        self.assertEqual(GradientType.SPARSE, 4)


class TestCompressionType(unittest.TestCase):
    """Test CompressionType enum"""

    def test_all_compression_types(self):
        """Test compression types"""
        self.assertEqual(CompressionType.NONE.value, "none")
        self.assertEqual(CompressionType.QUANTIZATION.value, "quantization")
        self.assertEqual(CompressionType.TOP_K.value, "top_k")


class TestGradientDescriptor(unittest.TestCase):
    """Test GradientDescriptor"""

    def test_create_descriptor(self):
        """Test creating gradient descriptor"""
        desc = GradientDescriptor(
            name="weights",
            shape=(1024, 512),
            dtype=GradientType.FLOAT32,
            size_bytes=2097152,
        )
        self.assertEqual(desc.name, "weights")
        self.assertEqual(desc.shape, (1024, 512))
        self.assertEqual(desc.compression, CompressionType.NONE)

    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        desc = GradientDescriptor(
            name="test",
            shape=(100,),
            dtype=GradientType.FLOAT32,
            size_bytes=400,
            compression_ratio=4.0,
        )
        self.assertEqual(desc.compression_ratio, 4.0)


class TestPipelineStage(unittest.TestCase):
    """Test PipelineStage"""

    def test_create_stage(self):
        """Test creating pipeline stage"""
        stage = PipelineStage(
            stage_id=0,
            name="forward",
            inputs=["x"],
            outputs=["y"],
        )
        self.assertEqual(stage.stage_id, 0)
        self.assertEqual(stage.name, "forward")


class TestGradientCompressor(unittest.TestCase):
    """Test GradientCompressor"""

    def test_create_compressor(self):
        """Test creating compressor"""
        compressor = GradientCompressor(CompressionType.TOP_K)
        self.assertEqual(compressor.compression_type, CompressionType.TOP_K)

    def test_compress_no_compression(self):
        """Test compression with NONE"""
        compressor = GradientCompressor(CompressionType.NONE)
        gradient = np.random.randn(100).astype(np.float32)

        desc = compressor.compress(gradient, {"name": "test"})

        self.assertEqual(desc.compression, CompressionType.NONE)
        self.assertEqual(desc.compression_ratio, 1.0)

    def test_compress_top_k(self):
        """Test TOP_K compression"""
        compressor = GradientCompressor(CompressionType.TOP_K)
        gradient = np.random.randn(1000).astype(np.float32)

        desc = compressor.compress(gradient, {"name": "test", "k_ratio": 0.1})

        self.assertEqual(desc.compression, CompressionType.TOP_K)
        self.assertGreater(desc.compression_ratio, 1.0)

    def test_decompress(self):
        """Test decompression"""
        compressor = GradientCompressor(CompressionType.NONE)
        gradient = np.random.randn(100).astype(np.float32)

        desc = compressor.compress(gradient, {"name": "test"})
        recovered = compressor.decompress(desc)

        self.assertIsInstance(recovered, np.ndarray)


class TestGradientAwareNetworkProtocol(unittest.TestCase):
    """Test GradientAwareNetworkProtocol"""

    def test_create_protocol(self):
        """Test creating protocol"""
        ganp = GradientAwareNetworkProtocol(device_id=0)
        self.assertEqual(ganp.device_id, 0)

    def test_add_stage(self):
        """Test adding pipeline stages"""
        ganp = GradientAwareNetworkProtocol()
        stage = PipelineStage(0, "layer1", ["x"], ["y"])
        ganp.add_stage(stage)
        self.assertEqual(len(ganp.stages), 1)

    def test_forward(self):
        """Test forward pass"""
        ganp = GradientAwareNetworkProtocol()
        inputs = {"input": np.random.randn(32, 64).astype(np.float32)}
        outputs = ganp.forward(inputs)
        self.assertEqual(ganp.state.forward_pass_id, 1)

    def test_backward(self):
        """Test backward pass"""
        ganp = GradientAwareNetworkProtocol()
        grads = {"input": np.random.randn(32, 64).astype(np.float32)}
        result = ganp.backward(np.float32(1.0), grads)
        self.assertIsInstance(result, dict)

    def test_compress_gradients(self):
        """Test gradient compression"""
        ganp = GradientAwareNetworkProtocol(compression_type=CompressionType.TOP_K)
        grads = {"grad1": np.random.randn(1000).astype(np.float32)}
        compressed = ganp.compress_gradients(grads)
        self.assertIn("grad1", compressed)

    def test_compression_stats(self):
        """Test getting compression stats"""
        ganp = GradientAwareNetworkProtocol()
        stats = ganp.get_compression_stats()
        self.assertIn("total_gradients", stats)
        self.assertIn("avg_compression_ratio", stats)


class TestBenchmark(unittest.TestCase):
    """Test benchmarking"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmark_gradient_compression(
            num_iterations=10,
            compression_type=CompressionType.TOP_K,
        )
        self.assertIn("compression_ratio", results)
        self.assertIn("compression_time_ms", results)


if __name__ == "__main__":
    unittest.main()
