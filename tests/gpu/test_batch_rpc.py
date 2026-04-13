"""Tests for Batch RPC Protocol"""

import unittest
import numpy as np
from ncpu.os.gpu.protocols.batch_rpc import (
    RpcOpcode,
    RpcArg,
    RpcRequest,
    RpcResult,
    RpcBatch,
    BatchRpcProtocol,
    benchmark_batch_rpc,
)


class TestRpcOpcodes(unittest.TestCase):
    """Test RPC opcode definitions"""

    def test_memory_ops_defined(self):
        """Test memory operation opcodes"""
        self.assertEqual(RpcOpcode.MEMORY_ALLOC, 10)
        self.assertEqual(RpcOpcode.MEMORY_FREE, 11)
        self.assertEqual(RpcOpcode.MEMORY_COPY, 12)

    def test_compute_ops_defined(self):
        """Test compute operation opcodes"""
        self.assertEqual(RpcOpcode.MATMUL, 20)
        self.assertEqual(RpcOpcode.CONV2D, 21)
        self.assertEqual(RpcOpcode.REDUCE, 22)


class TestRpcArg(unittest.TestCase):
    """Test RpcArg functionality"""

    def test_scalar_arg(self):
        """Test scalar argument"""
        arg = RpcArg(value=42)
        self.assertEqual(arg.value, 42)
        self.assertFalse(arg.is_tensor)
        self.assertFalse(arg.is_ref)

    def test_tensor_arg(self):
        """Test tensor argument"""
        arr = np.array([1, 2, 3])
        arg = RpcArg(value=arr, is_tensor=True)
        self.assertTrue(arg.is_tensor)
        self.assertIsInstance(arg.value, np.ndarray)

    def test_reference_arg(self):
        """Test reference argument"""
        arg = RpcArg(value="tensor_name", is_ref=True)
        self.assertTrue(arg.is_ref)


class TestRpcRequest(unittest.TestCase):
    """Test RpcRequest functionality"""

    def test_create_request(self):
        """Test creating RPC request"""
        req = RpcRequest(
            opcode=RpcOpcode.MATMUL,
            args=[RpcArg(value="A"), RpcArg(value="B")],
            output_ref="C",
        )
        self.assertEqual(req.opcode, RpcOpcode.MATMUL)
        self.assertEqual(req.output_ref, "C")

    def test_encode_request(self):
        """Test request encoding"""
        req = RpcRequest(
            opcode=RpcOpcode.ADD,
            args=[RpcArg(value=1), RpcArg(value=2)],
        )
        encoded = req.encode()
        self.assertIsInstance(encoded, bytes)
        self.assertGreater(len(encoded), 0)


class TestRpcBatch(unittest.TestCase):
    """Test RpcBatch functionality"""

    def test_create_batch(self):
        """Test creating a batch"""
        batch = RpcBatch(max_batch_size=10)
        self.assertEqual(len(batch), 0)
        self.assertTrue(batch.is_empty)

    def test_add_request(self):
        """Test adding requests to batch"""
        batch = RpcBatch()
        req_id = batch.add(
            RpcOpcode.MATMUL,
            ["A", "B"],
            output_ref="C",
        )
        self.assertEqual(req_id, 0)
        self.assertEqual(len(batch), 1)

    def test_add_with_dependency(self):
        """Test adding request with dependency"""
        batch = RpcBatch()
        batch.add(RpcOpcode.MATMUL, ["A", "B"], output_ref="C")
        batch.add_with_dep(
            RpcOpcode.RELU,
            ["C"],
            depends_on=0,
            output_ref="C",
        )
        self.assertEqual(len(batch), 2)

    def test_execute_clears_batch(self):
        """Test that execute clears the batch"""
        batch = RpcBatch()
        batch.add(RpcOpcode.MATMUL, ["A", "B"])
        results = batch.execute()
        self.assertEqual(len(results), 1)
        self.assertTrue(batch.is_empty)


class TestBatchRpcProtocol(unittest.TestCase):
    """Test BatchRpcProtocol functionality"""

    def test_create_protocol(self):
        """Test creating protocol"""
        protocol = BatchRpcProtocol(device_id=0)
        self.assertEqual(protocol.device_id, 0)
        self.assertEqual(protocol.pending_ops, 0)

    def test_register_tensor(self):
        """Test registering tensors"""
        protocol = BatchRpcProtocol()
        arr = np.random.randn(10, 10).astype(np.float32)
        protocol.register_tensor("test", arr)
        self.assertIn("test", protocol.tensors)

    def test_queue_matmul(self):
        """Test queuing matmul operation"""
        protocol = BatchRpcProtocol()
        protocol.register_tensor("A", np.random.randn(10, 10).astype(np.float32))
        protocol.register_tensor("B", np.random.randn(10, 10).astype(np.float32))

        req_id = protocol.queue_matmul("A", "B", "C")
        self.assertEqual(req_id, 0)
        self.assertEqual(protocol.pending_ops, 1)

    def test_execute_returns_results(self):
        """Test execute returns results"""
        protocol = BatchRpcProtocol()
        protocol.queue_matmul("A", "B", "C")
        results = protocol.execute()
        self.assertIsInstance(results, list)


class TestBenchmark(unittest.TestCase):
    """Test benchmarking function"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmark_batch_rpc(num_ops=10, batch_size=4)

        self.assertIn("traditional_time", results)
        self.assertIn("batch_rpc_time", results)
        self.assertIn("speedup", results)
        self.assertIn("num_ops", results)


if __name__ == "__main__":
    unittest.main()
