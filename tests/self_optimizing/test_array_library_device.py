"""Device-portability tests for discrete program execution (NV3).

Discrete `DiscreteArrayProgram.execute` must run on whichever device its
inputs live on — CPU always, MPS when available on macOS, CUDA when available.
The execution is pure tensor ops (arithmetic + `torch.where`), so portability
should be free, but we verify it and lock in any breakage via regression.
"""

from __future__ import annotations

import unittest

import torch

from ncpu.self_optimizing.array_program_library import DiscreteArrayProgram


class TestDeviceExecution(unittest.TestCase):
    def _run_sum_program_on(self, device: str) -> None:
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.0]], device=device
        )
        lengths = torch.tensor([3.0], device=device)
        result = program.execute(arrays, lengths)
        self.assertEqual(str(result.device).split(":")[0], device)
        self.assertTrue(
            torch.allclose(result.cpu(), torch.tensor([6.0]), atol=1e-4)
        )

    def test_cpu(self):
        self._run_sum_program_on("cpu")

    def test_mps_if_available(self):
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            self.skipTest("MPS not available")
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.0]], device="mps"
        )
        lengths = torch.tensor([3.0], device="mps")
        result = program.execute(arrays, lengths)
        self.assertEqual(result.device.type, "mps")
        self.assertTrue(
            torch.allclose(result.cpu(), torch.tensor([6.0]), atol=1e-4)
        )

    def test_cuda_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        program = DiscreteArrayProgram(0, 0, 0, 0, 0.0)
        arrays = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.0]], device="cuda"
        )
        lengths = torch.tensor([3.0], device="cuda")
        result = program.execute(arrays, lengths)
        self.assertEqual(result.device.type, "cuda")
        self.assertTrue(
            torch.allclose(result.cpu(), torch.tensor([6.0]), atol=1e-4)
        )

    def test_max_program_cpu_matches_reference(self):
        program = DiscreteArrayProgram(2, 0, 2, 0, 0.0)
        arrays = torch.tensor(
            [
                [-5.0, -2.0, -10.0, 0.0, 0.0],
                [1.0, 3.0, 2.0, 0.0, 0.0],
            ]
        )
        lengths = torch.tensor([3.0, 3.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([-2.0, 3.0])))

    def test_fractional_arrays_work(self):
        # Library fast-path must work on non-integer inputs too.
        program = DiscreteArrayProgram(0, 2, 0, 0, 0.0)  # sum of |x|
        arrays = torch.tensor([[1.5, -2.25, 0.75, 0.0, 0.0]])
        lengths = torch.tensor([3.0])
        result = program.execute(arrays, lengths)
        self.assertTrue(torch.allclose(result, torch.tensor([4.5]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
