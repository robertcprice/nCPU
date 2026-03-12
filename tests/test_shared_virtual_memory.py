"""Tests for Shared Virtual Memory Protocol"""

import unittest
import numpy as np
from ncpu.os.gpu.protocols.shared_virtual_memory import (
    SvmFlags,
    TransferDirection,
    SvmRegion,
    SvmPointer,
    DmaTransfer,
    SvmAllocator,
    DmaEngine,
    SharedVirtualMemoryProtocol,
    benchmark_svm_transfer,
)


class TestSvmFlags(unittest.TestCase):
    """Test SVM flags"""

    def test_flags_defined(self):
        """Test flag values"""
        self.assertEqual(SvmFlags.READ, 0x1)
        self.assertEqual(SvmFlags.WRITE, 0x2)
        self.assertEqual(SvmFlags.READ_WRITE, 0x3)
        self.assertEqual(SvmFlags.ATOMIC, 0x4)


class TestTransferDirection(unittest.TestCase):
    """Test transfer direction"""

    def test_directions_defined(self):
        """Test direction values"""
        self.assertEqual(TransferDirection.GPU_TO_GPU, 0)
        self.assertEqual(TransferDirection.GPU_TO_HOST, 1)
        self.assertEqual(TransferDirection.HOST_TO_GPU, 2)


class TestSvmRegion(unittest.TestCase):
    """Test SvmRegion"""

    def test_create_region(self):
        """Test creating SVM region"""
        region = SvmRegion(
            region_id=0,
            virtual_addr=0x10000000,
            size_bytes=4096,
            device_ids=[0, 1],
        )
        self.assertEqual(region.region_id, 0)
        self.assertEqual(region.size_bytes, 4096)
        self.assertTrue(region.is_allocated)

    def test_nbytes_property(self):
        """Test nbytes property"""
        region = SvmRegion(0, 0, 8192, [0])
        self.assertEqual(region.nbytes, 8192)


class TestSvmPointer(unittest.TestCase):
    """Test SvmPointer"""

    def test_create_pointer(self):
        """Test creating pointer"""
        ptr = SvmPointer(
            addr=0x10000000,
            region_id=0,
            device_id=0,
        )
        self.assertEqual(ptr.addr, 0x10000000)
        self.assertEqual(ptr.region_id, 0)


class TestDmaTransfer(unittest.TestCase):
    """Test DmaTransfer"""

    def test_create_transfer(self):
        """Test creating DMA transfer"""
        src = SvmPointer(0x1000, 0, 0)
        dst = SvmPointer(0x2000, 0, 1)

        transfer = DmaTransfer(
            transfer_id=0,
            src=src,
            dst=dst,
            size_bytes=1024,
        )
        self.assertEqual(transfer.size_bytes, 1024)


class TestSvmAllocator(unittest.TestCase):
    """Test SvmAllocator"""

    def test_create_allocator(self):
        """Test creating allocator"""
        allocator = SvmAllocator(devices=[0, 1])
        self.assertEqual(len(allocator.devices), 2)

    def test_allocate(self):
        """Test allocating memory"""
        allocator = SvmAllocator(devices=[0, 1])
        region = allocator.allocate(4096, [0, 1])
        self.assertIsInstance(region, SvmRegion)
        self.assertTrue(region.is_allocated)

    def test_deallocate(self):
        """Test deallocating memory"""
        allocator = SvmAllocator(devices=[0, 1])
        region = allocator.allocate(4096, [0, 1])
        allocator.deallocate(region)
        self.assertEqual(len(allocator.regions), 0)


class TestDmaEngine(unittest.TestCase):
    """Test DmaEngine"""

    def test_create_engine(self):
        """Test creating DMA engine"""
        allocator = SvmAllocator([0, 1])
        engine = DmaEngine(allocator)
        self.assertIsNotNone(engine)

    def test_submit_transfer(self):
        """Test submitting transfer"""
        allocator = SvmAllocator([0, 1])
        engine = DmaEngine(allocator)

        src = SvmPointer(0x1000, 0, 0, size=1024)
        dst = SvmPointer(0x2000, 0, 1, size=1024)

        transfer_id = engine.submit_transfer(src, dst, 1024)
        self.assertGreaterEqual(transfer_id, 0)


class TestSharedVirtualMemoryProtocol(unittest.TestCase):
    """Test SharedVirtualMemoryProtocol"""

    def test_create_protocol(self):
        """Test creating protocol"""
        svmp = SharedVirtualMemoryProtocol(devices=[0, 1])
        self.assertEqual(len(svmp.devices), 2)

    def test_allocate(self):
        """Test allocating memory"""
        svmp = SharedVirtualMemoryProtocol([0, 1])
        region = svmp.allocate(4096, [0, 1])
        self.assertIsInstance(region, SvmRegion)

    def test_allocate_for_tensor(self):
        """Test allocating for tensor"""
        svmp = SharedVirtualMemoryProtocol([0, 1])
        data = np.random.randn(10, 10).astype(np.float32)
        region = svmp.allocate_for_tensor("weights", data, [0, 1])
        self.assertIn("weights", svmp.tensors)
        self.assertIn("weights", svmp.pointers)

    def test_get_pointer(self):
        """Test getting pointer"""
        svmp = SharedVirtualMemoryProtocol([0, 1])
        data = np.random.randn(10, 10).astype(np.float32)
        svmp.allocate_for_tensor("weights", data, [0, 1])
        ptr = svmp.get_pointer("weights", 0)
        self.assertIsNotNone(ptr)

    def test_transfer(self):
        """Test transfer"""
        svmp = SharedVirtualMemoryProtocol([0, 1])
        data = np.random.randn(10, 10).astype(np.float32)
        svmp.allocate_for_tensor("weights", data, [0, 1])

        src = svmp.get_pointer("weights", 0)
        dst = SvmPointer(src.addr, src.region_id, 1, size=src.size)

        transfer_id = svmp.transfer(src, dst)
        self.assertGreaterEqual(transfer_id, 0)


class TestBenchmark(unittest.TestCase):
    """Test benchmarking"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmark_svm_transfer(tensor_size_mb=1.0, num_transfers=10)
        self.assertIn("traditional_time", results)
        self.assertIn("speedup", results)


if __name__ == "__main__":
    unittest.main()
