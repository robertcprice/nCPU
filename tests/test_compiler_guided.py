"""Tests for Compiler-Guided Protocol"""

import unittest
import numpy as np
from ncpu.os.gpu.protocols.compiler_guided import (
    StateType,
    CompressionType,
    RegisterState,
    MemoryRegion,
    StateCapture,
    StateDiff,
    MigrationUnit,
    StateReplayer,
    CompilerGuidedProtocol,
    benchmark_state_capture,
)


class TestStateType(unittest.TestCase):
    """Test StateType enum"""

    def test_types_defined(self):
        """Test state types"""
        self.assertEqual(StateType.REGISTER.value, "register")
        self.assertEqual(StateType.MEMORY.value, "memory")
        self.assertEqual(StateType.STACK.value, "stack")


class TestCompressionType(unittest.TestCase):
    """Test CompressionType"""

    def test_compression_types(self):
        """Test compression types"""
        self.assertEqual(CompressionType.NONE, 0)
        self.assertEqual(CompressionType.ZSTD, 1)


class TestRegisterState(unittest.TestCase):
    """Test RegisterState"""

    def test_create_registers(self):
        """Test creating register state"""
        regs = RegisterState()
        self.assertEqual(regs.pc, 0)
        self.assertEqual(regs.sp, 0)

    def test_set_registers(self):
        """Test setting register values"""
        regs = RegisterState(pc=0x1000, sp=0x8000, x0=42)
        self.assertEqual(regs.pc, 0x1000)
        self.assertEqual(regs.x0, 42)

    def test_to_from_bytes(self):
        """Test serialization"""
        regs = RegisterState(pc=0x1000, sp=0x8000, x0=42)
        data = regs.to_bytes()
        self.assertIsInstance(data, bytes)

        regs2 = RegisterState.from_bytes(data)
        self.assertEqual(regs.pc, regs2.pc)
        self.assertEqual(regs.sp, regs2.sp)

    def test_hash(self):
        """Test hash"""
        regs = RegisterState(pc=0x1000)
        h = regs.hash()
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 16)


class TestMemoryRegion(unittest.TestCase):
    """Test MemoryRegion"""

    def test_create_region(self):
        """Test creating memory region"""
        region = MemoryRegion(
            start=0x1000,
            size=4096,
            data=b'\x00' * 4096,
        )
        self.assertEqual(region.start, 0x1000)
        self.assertEqual(region.size, 4096)

    def test_to_from_bytes(self):
        """Test serialization"""
        region = MemoryRegion(start=0x1000, size=1024, data=b'test')
        data = region.to_bytes()
        self.assertIsInstance(data, bytes)

        region2 = MemoryRegion.from_bytes(data)
        self.assertEqual(region.start, region2.start)
        self.assertEqual(region.size, region2.size)


class TestStateCapture(unittest.TestCase):
    """Test StateCapture"""

    def test_create_capture(self):
        """Test creating state capture"""
        regs = RegisterState()
        capture = StateCapture(
            capture_id=1,
            timestamp_us=1000000,
            pc=0x1000,
            registers=regs,
        )
        self.assertEqual(capture.capture_id, 1)
        self.assertEqual(capture.pc, 0x1000)

    def test_total_size(self):
        """Test total size calculation"""
        regs = RegisterState()
        capture = StateCapture(
            capture_id=1,
            timestamp_us=0,
            pc=0,
            registers=regs,
        )
        size = capture.total_size()
        self.assertGreater(size, 0)

    def test_to_bytes(self):
        """Test serialization"""
        regs = RegisterState()
        capture = StateCapture(
            capture_id=1,
            timestamp_us=1000000,
            pc=0x1000,
            registers=regs,
        )
        data = capture.to_bytes()
        self.assertIsInstance(data, bytes)


class TestStateDiff(unittest.TestCase):
    """Test StateDiff"""

    def test_create_diff(self):
        """Test creating state diff"""
        diff = StateDiff(
            from_capture_id=1,
            to_capture_id=2,
            timestamp_us=2000000,
        )
        self.assertEqual(diff.from_capture_id, 1)

    def test_changed_registers(self):
        """Test changed registers"""
        diff = StateDiff(0, 1, 0)
        diff.changed_regs = {"x0": 42, "x1": 100}
        self.assertEqual(len(diff.changed_regs), 2)

    def test_total_size(self):
        """Test size calculation"""
        diff = StateDiff(0, 1, 0)
        size = diff.total_size()
        self.assertGreater(size, 0)


class TestMigrationUnit(unittest.TestCase):
    """Test MigrationUnit"""

    def test_create_unit(self):
        """Test creating migration unit"""
        unit = MigrationUnit(
            unit_id="test",
            elf_hash="abc123",
            entry_point=0x1000,
            initial_sp=0x8000,
        )
        self.assertEqual(unit.unit_id, "test")
        self.assertEqual(unit.elf_hash, "abc123")

    def test_add_capture(self):
        """Test adding captures"""
        unit = MigrationUnit(unit_id="test", elf_hash="", entry_point=0, initial_sp=0)
        regs = RegisterState()
        capture = StateCapture(1, 0, 0, regs)
        unit.add_capture(capture)
        self.assertEqual(len(unit.captures), 1)

    def test_get_latest_capture(self):
        """Test getting latest capture"""
        unit = MigrationUnit(unit_id="test", elf_hash="", entry_point=0, initial_sp=0)
        regs = RegisterState()
        unit.add_capture(StateCapture(1, 0, 0, regs))
        unit.add_capture(StateCapture(2, 0, 0, regs))
        latest = unit.get_latest_capture()
        self.assertEqual(latest.capture_id, 2)


class TestStateReplayer(unittest.TestCase):
    """Test StateReplayer"""

    def test_create_replayer(self):
        """Test creating replayer"""
        replayer = StateReplayer()
        self.assertEqual(len(replayer.captures), 0)

    def test_add_capture(self):
        """Test adding captures"""
        replayer = StateReplayer()
        regs = RegisterState()
        capture = StateCapture(1, 0, 0, regs)
        replayer.add_capture(capture)
        self.assertIn(1, replayer.captures)

    def test_replay_to(self):
        """Test replaying to capture"""
        replayer = StateReplayer()
        regs = RegisterState()
        capture = StateCapture(1, 0, 0, regs)
        replayer.add_capture(capture)
        result = replayer.replay_to(1)
        self.assertIsNotNone(result)


class TestCompilerGuidedProtocol(unittest.TestCase):
    """Test CompilerGuidedProtocol"""

    def test_create_protocol(self):
        """Test creating protocol"""
        cgp = CompilerGuidedProtocol("/fake/path.elf")
        self.assertEqual(cgp.elf_path, "/fake/path.elf")

    def test_capture(self):
        """Test capturing state"""
        cgp = CompilerGuidedProtocol("/fake/path.elf")
        capture_id = cgp.capture("test")
        self.assertGreater(capture_id, 0)

    def test_capture_multiple(self):
        """Test multiple captures"""
        cgp = CompilerGuidedProtocol("/fake/path.elf")
        id1 = cgp.capture("start")
        id2 = cgp.capture("end")
        self.assertNotEqual(id1, id2)

    def test_replay_to(self):
        """Test replay"""
        cgp = CompilerGuidedProtocol("/fake/path.elf")
        capture_id = cgp.capture("test")
        state = cgp.replay_to(capture_id)
        self.assertIsNotNone(state)

    def test_create_migration_unit(self):
        """Test creating migration unit"""
        cgp = CompilerGuidedProtocol("/fake/path.elf")
        cgp.capture("test")
        unit = cgp.create_migration_unit()
        self.assertIsInstance(unit, MigrationUnit)


class TestBenchmark(unittest.TestCase):
    """Test benchmarking"""

    def test_benchmark_returns_dict(self):
        """Test benchmark returns expected dictionary"""
        results = benchmark_state_capture(num_captures=10)
        self.assertIn("full_capture_time", results)
        self.assertIn("diff_capture_time", results)


if __name__ == "__main__":
    unittest.main()
