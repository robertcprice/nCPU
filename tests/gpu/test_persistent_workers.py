"""Tests for Persistent GPU Workers Protocol"""

import unittest
import numpy as np
import time
from ncpu.os.gpu.protocols.persistent_workers import (
    WorkPriority,
    WorkItem,
    WorkResult,
    PersistentWorker,
    WorkerPool,
    PersistentGpuWorkersProtocol,
    benchmark_persistent_workers,
)


class TestWorkPriority(unittest.TestCase):
    """Test WorkPriority enum"""

    def test_priority_levels(self):
        """Test priority levels"""
        self.assertEqual(WorkPriority.LOW, 0)
        self.assertEqual(WorkPriority.NORMAL, 1)
        self.assertEqual(WorkPriority.HIGH, 2)
        self.assertEqual(WorkPriority.CRITICAL, 3)


class TestWorkItem(unittest.TestCase):
    """Test WorkItem"""

    def test_create_work_item(self):
        """Test creating work item"""
        item = WorkItem(
            work_id=1,
            operation="matmul",
            inputs={"A": np.array([1, 2, 3])},
            outputs={"C": np.array([0])},
        )
        self.assertEqual(item.work_id, 1)
        self.assertEqual(item.operation, "matmul")
        self.assertEqual(item.priority, WorkPriority.NORMAL)

    def test_priority_comparison(self):
        """Test priority comparison"""
        low = WorkItem(work_id=1, operation="a", inputs={}, outputs={}, priority=WorkPriority.LOW)
        high = WorkItem(work_id=2, operation="b", inputs={}, outputs={}, priority=WorkPriority.HIGH)
        self.assertLess(low, high)


class TestWorkResult(unittest.TestCase):
    """Test WorkResult"""

    def test_create_result(self):
        """Test creating work result"""
        result = WorkResult(
            work_id=1,
            success=True,
            processing_time_ms=1.5,
        )
        self.assertEqual(result.work_id, 1)
        self.assertTrue(result.success)


class TestPersistentWorker(unittest.TestCase):
    """Test PersistentWorker"""

    def test_create_worker(self):
        """Test creating worker"""
        worker = PersistentWorker(worker_id=0)
        self.assertEqual(worker.worker_id, 0)
        self.assertFalse(worker.running)

    def test_start_stop(self):
        """Test starting and stopping worker"""
        worker = PersistentWorker(worker_id=0)
        worker.start()
        self.assertTrue(worker.running)
        worker.stop()
        self.assertFalse(worker.running)


class TestWorkerPool(unittest.TestCase):
    """Test WorkerPool"""

    def test_create_pool(self):
        """Test creating worker pool"""
        pool = WorkerPool(num_workers=2)
        self.assertEqual(pool.num_workers, 2)
        self.assertEqual(len(pool.work_queues), 2)

    def test_start_stop(self):
        """Test starting and stopping pool"""
        pool = WorkerPool(num_workers=2)
        pool.start()
        self.assertTrue(pool.running)
        pool.stop()
        self.assertFalse(pool.running)

    def test_submit_work(self):
        """Test submitting work to pool"""
        pool = WorkerPool(num_workers=2)
        pool.start()

        item = WorkItem(
            work_id=0,
            operation="test",
            inputs={},
            outputs={},
        )
        work_id = pool.submit(item)
        self.assertGreater(work_id, 0)

        pool.stop()

    def test_get_stats(self):
        """Test getting pool statistics"""
        pool = WorkerPool(num_workers=2)
        pool.start()
        stats = pool.get_stats()
        self.assertIn("num_workers", stats)
        pool.stop()


class TestPersistentGpuWorkersProtocol(unittest.TestCase):
    """Test PersistentGpuWorkersProtocol"""

    def test_create_protocol(self):
        """Test creating protocol"""
        pgwp = PersistentGpuWorkersProtocol(device_id=0, num_workers=2)
        self.assertEqual(pgwp.device_id, 0)
        self.assertEqual(pgwp.num_workers, 2)

    def test_initialize(self):
        """Test initializing protocol"""
        pgwp = PersistentGpuWorkersProtocol(num_workers=2)
        pgwp.initialize()
        self.assertTrue(pgwp.initialized)
        pgwp.shutdown()

    def test_submit_work(self):
        """Test submitting work"""
        pgwp = PersistentGpuWorkersProtocol(num_workers=2)
        pgwp.initialize()

        work_id = pgwp.submit_work(
            operation="test",
            inputs={},
            outputs={},
        )
        self.assertGreater(work_id, 0)

        pgwp.shutdown()

    def test_submit_batch(self):
        """Test submitting batch"""
        pgwp = PersistentGpuWorkersProtocol(num_workers=2)
        pgwp.initialize()

        operations = [
            {"operation": "op1", "inputs": {}, "outputs": {}},
            {"operation": "op2", "inputs": {}, "outputs": {}},
        ]
        work_ids = pgwp.submit_batch(operations)
        self.assertEqual(len(work_ids), 2)

        pgwp.shutdown()

    def test_get_stats(self):
        """Test getting stats"""
        pgwp = PersistentGpuWorkersProtocol(num_workers=2)
        pgwp.initialize()
        stats = pgwp.get_stats()
        self.assertIn("num_workers", stats)
        pgwp.shutdown()


class TestBenchmark(unittest.TestCase):
    """Test benchmarking"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmark_persistent_workers(num_operations=10, num_workers=2)
        self.assertIn("traditional_time", results)
        self.assertIn("speedup", results)


if __name__ == "__main__":
    unittest.main()
