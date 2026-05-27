"""Native Rust/Metal backend tests for DiscreteArrayProgram (NV3+)."""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import (
    DiscreteArrayProgram,
    get_native_backend,
    reset_native_backend_cache,
)


class TestNativeBackendLoads(unittest.TestCase):
    def setUp(self):
        reset_native_backend_cache()

    def test_can_load_or_return_none(self):
        # Loading must not crash — may legitimately return None on systems
        # without a compiled ncpu_metal.abi3.so.
        module = get_native_backend()
        if module is not None:
            self.assertTrue(hasattr(module, "npcot_execute_cpu"))
            self.assertTrue(hasattr(module, "NpcotGpuExecutor"))

    def test_cache_is_memoized(self):
        module_a = get_native_backend()
        module_b = get_native_backend()
        self.assertIs(module_a, module_b)


class TestDiscreteProgramNativeExecution(unittest.TestCase):
    def setUp(self):
        reset_native_backend_cache()
        self._has_native = get_native_backend() is not None

    def test_auto_backend_matches_python_sum(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0, 2.0])
        python_out = program.execute(arrays, lengths)
        native_out = program.execute_native(arrays, lengths, backend="auto")
        self.assertTrue(torch.allclose(python_out, native_out, atol=1e-5))

    def test_auto_backend_matches_python_max(self):
        program = DiscreteArrayProgram(2, 0, 2, 0, 0.0)
        arrays = torch.tensor(
            [
                [-1.0, -3.0, -2.0, 99.0, 99.0],
                [5.0, 0.0, -4.0, 99.0, 99.0],
            ]
        )
        lengths = torch.tensor([3.0, 3.0])
        python_out = program.execute(arrays, lengths)
        native_out = program.execute_native(arrays, lengths)
        self.assertTrue(torch.allclose(python_out, native_out, atol=1e-5))

    def test_auto_backend_matches_python_count_positive(self):
        program = DiscreteArrayProgram(0, 4, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, -2.0, 3.0, 0.0, 0.0], [-1.0, -2.0, -3.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0, 3.0])
        python_out = program.execute(arrays, lengths)
        native_out = program.execute_native(arrays, lengths)
        self.assertTrue(torch.allclose(python_out, native_out, atol=1e-5))

    def test_mean_with_offset_round_trips(self):
        program = DiscreteArrayProgram(0, 0, 0, 1, -1.5)
        arrays = torch.tensor([[2.0, 4.0, 6.0, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        python_out = program.execute(arrays, lengths)
        native_out = program.execute_native(arrays, lengths)
        self.assertTrue(torch.allclose(python_out, native_out, atol=1e-5))

    def test_rust_cpu_backend_explicit(self):
        if not self._has_native:
            self.skipTest("native backend not available")
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, 2.0, 3.0]])
        lengths = torch.tensor([3.0])
        result = program.execute_native(arrays, lengths, backend="rust_cpu")
        self.assertTrue(torch.allclose(result, torch.tensor([6.0]), atol=1e-5))

    def test_metal_backend_explicit(self):
        if not self._has_native:
            self.skipTest("native backend not available")
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        lengths = torch.tensor([4.0])
        try:
            result = program.execute_native(arrays, lengths, backend="metal")
        except RuntimeError:
            self.skipTest("metal device not available")
        self.assertTrue(torch.allclose(result, torch.tensor([10.0]), atol=1e-5))

    def test_python_backend_bypasses_native(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, 2.0, 3.0]])
        lengths = torch.tensor([3.0])
        out = program.execute_native(arrays, lengths, backend="python")
        self.assertTrue(torch.allclose(out, torch.tensor([6.0]), atol=1e-5))

    def test_unavailable_backend_raises(self):
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor([[1.0, 2.0, 3.0]])
        lengths = torch.tensor([3.0])
        # When native backend IS available, unknown string is ValueError.
        # When native backend is NOT available, it's RuntimeError. Either
        # way, something must be raised — verifies we don't silently ignore.
        with self.assertRaises((ValueError, RuntimeError)):
            program.execute_native(arrays, lengths, backend="cuda")


if __name__ == "__main__":
    unittest.main()
