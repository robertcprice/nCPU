"""Tests for Tensor Streaming Protocol"""

import unittest
import numpy as np
from ncpu.os.gpu.protocols.tensor_stream import (
    TensorDtype,
    TensorOp,
    TensorDescriptor,
    TensorStreamProtocol,
    create_descriptor,
    benchmarkTSP,
)


class TestTensorDescriptor(unittest.TestCase):
    """Test TensorDescriptor functionality"""

    def test_create_descriptor(self):
        """Test creating a basic tensor descriptor"""
        desc = TensorDescriptor(
            device_id=0,
            ptr=0x1000,
            shape=(1024, 512),
            dtype=TensorDtype.FLOAT32
        )
        self.assertEqual(desc.shape, (1024, 512))
        self.assertEqual(desc.dtype, TensorDtype.FLOAT32)
        self.assertEqual(desc.device_id, 0)

    def test_default_strides(self):
        """Test that default strides are computed correctly"""
        desc = TensorDescriptor(
            device_id=0,
            ptr=0,
            shape=(2, 3, 4),
            dtype=TensorDtype.FLOAT32
        )
        # Row-major: stride = (12, 4, 1) for shape (2, 3, 4)
        self.assertEqual(desc.stride, (12, 4, 1))

    def test_nbytes(self):
        """Test nbytes calculation"""
        desc = TensorDescriptor(
            device_id=0,
            ptr=0,
            shape=(100, 100),
            dtype=TensorDtype.FLOAT32
        )
        # 100 * 100 * 4 bytes = 40000
        self.assertEqual(desc.nbytes, 40000)

    def test_nbytes_float16(self):
        """Test nbytes for float16"""
        desc = TensorDescriptor(
            device_id=0,
            ptr=0,
            shape=(100, 100),
            dtype=TensorDtype.FLOAT16
        )
        # 100 * 100 * 2 bytes = 20000
        self.assertEqual(desc.nbytes, 20000)


class TestTensorStreamProtocol(unittest.TestCase):
    """Test TensorStreamProtocol functionality"""

    def test_register_tensor(self):
        """Test registering tensors"""
        tsp = TensorStreamProtocol(device_id=0)
        desc = TensorDescriptor(
            device_id=0,
            ptr=0x1000,
            shape=(10, 10),
            dtype=TensorDtype.FLOAT32
        )
        tsp.register_tensor("test_tensor", desc)

        self.assertIn("test_tensor", tsp.tensors)
        self.assertEqual(tsp.tensors["test_tensor"].shape, (10, 10))

    def test_queue_op(self):
        """Test queuing operations"""
        tsp = TensorStreamProtocol()

        # Register tensors
        for name in ["a", "b", "c"]:
            tsp.register_tensor(name, TensorDescriptor(
                device_id=0, ptr=0, shape=(10, 10), dtype=TensorDtype.FLOAT32
            ))

        # Queue operation
        tsp.queue_op(TensorOp.ADD, ["a", "b"], ["c"])

        self.assertEqual(len(tsp.op_queue), 1)
        op, inputs, outputs = tsp.op_queue[0]
        self.assertEqual(op, TensorOp.ADD)
        self.assertEqual(inputs, ["a", "b"])
        self.assertEqual(outputs, ["c"])

    def test_queue_op_validates_tensors(self):
        """Test that queue_op validates tensor existence"""
        tsp = TensorStreamProtocol()
        tsp.register_tensor("a", TensorDescriptor(
            device_id=0, ptr=0, shape=(10, 10), dtype=TensorDtype.FLOAT32
        ))

        # Should raise for missing tensor
        with self.assertRaises(ValueError):
            tsp.queue_op(TensorOp.ADD, ["a", "missing"], ["c"])

    def test_execute_clears_queue(self):
        """Test that execute clears the operation queue"""
        tsp = TensorStreamProtocol()

        # Register and queue
        for name in ["a", "b"]:
            tsp.register_tensor(name, TensorDescriptor(
                device_id=0, ptr=0, shape=(10, 10), dtype=TensorDtype.FLOAT32
            ))

        tsp.queue_op(TensorOp.ADD, ["a"], ["b"])
        self.assertEqual(len(tsp.op_queue), 1)

        tsp.execute()
        self.assertEqual(len(tsp.op_queue), 0)

    def test_clone_descriptor(self):
        """Test cloning descriptors"""
        tsp = TensorStreamProtocol()
        orig = TensorDescriptor(
            device_id=0,
            ptr=0x1000,
            shape=(10, 20),
            dtype=TensorDtype.FLOAT32
        )
        tsp.register_tensor("orig", orig)

        cloned = tsp.clone_descriptor("orig")

        self.assertIsNotNone(cloned)
        self.assertEqual(cloned.shape, orig.shape)
        self.assertEqual(cloned.dtype, orig.dtype)
        self.assertEqual(cloned.ptr, orig.ptr)
        # Verify it's a different object
        self.assertIsNot(cloned, orig)


class TestCreateDescriptor(unittest.TestCase):
    """Test create_descriptor factory function"""

    def test_from_numpy_float32(self):
        """Test creating descriptor from float32 numpy array"""
        arr = np.random.randn(10, 20).astype(np.float32)
        desc = create_descriptor(arr, device_id=0)

        self.assertEqual(desc.shape, (10, 20))
        self.assertEqual(desc.dtype, TensorDtype.FLOAT32)

    def test_from_numpy_float16(self):
        """Test creating descriptor from float16 numpy array"""
        arr = np.random.randn(5, 5).astype(np.float16)
        desc = create_descriptor(arr)

        self.assertEqual(desc.dtype, TensorDtype.FLOAT16)

    def test_from_numpy_int32(self):
        """Test creating descriptor from int32 numpy array"""
        arr = np.random.randint(0, 100, (3, 3), dtype=np.int32)
        desc = create_descriptor(arr)

        self.assertEqual(desc.dtype, TensorDtype.INT32)


class TestBenchmarkTSP(unittest.TestCase):
    """Test TSP benchmarking"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmarkTSP(num_ops=10, batch_size=4)

        self.assertIn("traditional_time", results)
        self.assertIn("tsp_time", results)
        self.assertIn("speedup", results)
        self.assertIn("num_ops", results)
        self.assertIn("batch_size", results)

        self.assertEqual(results["num_ops"], 10)
        self.assertEqual(results["batch_size"], 4)


class TestTensorOps(unittest.TestCase):
    """Test TensorOp enum"""

    def test_all_ops_defined(self):
        """Test that all expected operations are defined"""
        expected_ops = [
            "ADD", "MUL", "MATMUL", "SOFTMAX", "RELU", "GELU",
            "LAYERNORM", "RMSNORM", "ATTENTION", "SILU", "CLIP",
            "TRANSPOSE", "RESHAPE", "SLICE", "CONCAT"
        ]

        for op_name in expected_ops:
            self.assertTrue(hasattr(TensorOp, op_name))


class TestTensorDtype(unittest.TestCase):
    """Test TensorDtype enum"""

    def test_all_dtypes_defined(self):
        """Test that all expected dtypes are defined"""
        expected_dtypes = ["FLOAT32", "FLOAT16", "INT32", "INT8", "UINT8"]

        for dtype_name in expected_dtypes:
            self.assertTrue(hasattr(TensorDtype, dtype_name))


if __name__ == "__main__":
    unittest.main()
