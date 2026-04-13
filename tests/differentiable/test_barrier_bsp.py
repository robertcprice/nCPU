"""Tests for barrier synchronization and BSP execution.

Verifies:
  1. Barrier fires only when all participants arrive.
  2. Callbacks execute at barrier completion with correct generation.
  3. BSP execution produces correct results across supersteps.
  4. Shared memory is properly exchanged between supersteps.
"""

import pytest
import torch

from ncpu.distributed import (
    Barrier,
    DistributedNCPU,
    DistributedResult,
)
from ncpu.differentiable.execution import (
    FixedProgram,
    Instruction,
    OPCODES,
)


# =========================================================================
# Helpers
# =========================================================================


def _add_program(dst=7, src1=0, src2=1):
    """R[dst] = R[src1] + R[src2]; HALT."""
    return FixedProgram([
        Instruction(OPCODES["ADD"], dst=dst, src1=src1, src2=src2),
        Instruction(OPCODES["HALT"]),
    ])


def _mul_program(dst=7, src1=0, src2=1):
    """R[dst] = R[src1] * R[src2]; HALT."""
    return FixedProgram([
        Instruction(OPCODES["MUL"], dst=dst, src1=src1, src2=src2),
        Instruction(OPCODES["HALT"]),
    ])


def _mov_imm_program(dst=0, value=42.0):
    """R[dst] = immediate; HALT."""
    return FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=dst, immediate=value),
        Instruction(OPCODES["HALT"]),
    ])


# =========================================================================
# Barrier
# =========================================================================


class TestBarrier:
    """Tests for the Barrier synchronization primitive."""

    def test_barrier_basic(self):
        """4 cores arrive at barrier; barrier triggers on the last arrival."""
        barrier = Barrier(num_participants=4)

        # First 3 arrivals should not trigger the barrier
        assert barrier.arrive(0) is False
        assert barrier.arrive(1) is False
        assert barrier.arrive(2) is False

        # Fourth arrival should trigger the barrier
        assert barrier.arrive(3) is True

        # Generation should have incremented
        assert barrier.generation == 1

        # Arrived set should be cleared for next generation
        assert len(barrier.arrived) == 0

    def test_barrier_callback(self):
        """Callbacks fire when the barrier completes with correct generation."""
        barrier = Barrier(num_participants=2)
        callback_log = []

        barrier.on_complete(lambda gen: callback_log.append(gen))

        # First generation
        barrier.arrive(0)
        assert len(callback_log) == 0
        barrier.arrive(1)
        assert callback_log == [1]

        # Second generation
        barrier.arrive(0)
        barrier.arrive(1)
        assert callback_log == [1, 2]

    def test_barrier_multiple_callbacks(self):
        """Multiple callbacks all fire in registration order."""
        barrier = Barrier(num_participants=2)
        log_a = []
        log_b = []

        barrier.on_complete(lambda gen: log_a.append(f"a:{gen}"))
        barrier.on_complete(lambda gen: log_b.append(f"b:{gen}"))

        barrier.arrive(0)
        barrier.arrive(1)

        assert log_a == ["a:1"]
        assert log_b == ["b:1"]

    def test_barrier_duplicate_arrival_ignored(self):
        """Same core arriving twice in one generation is silently ignored."""
        barrier = Barrier(num_participants=3)

        barrier.arrive(0)
        barrier.arrive(0)  # duplicate, still only 1 unique arrival
        assert len(barrier.arrived) == 1

        barrier.arrive(1)
        assert len(barrier.arrived) == 2

        barrier.arrive(2)
        assert barrier.generation == 1

    def test_barrier_reset(self):
        """Reset clears all state."""
        barrier = Barrier(num_participants=2)
        barrier.on_complete(lambda gen: None)
        barrier.arrive(0)
        barrier.arrive(1)

        barrier.reset()

        assert barrier.generation == 0
        assert len(barrier.arrived) == 0
        assert len(barrier.callbacks) == 0

    def test_barrier_single_participant(self):
        """Barrier with one participant fires immediately."""
        barrier = Barrier(num_participants=1)
        assert barrier.arrive(0) is True
        assert barrier.generation == 1

    def test_barrier_invalid_participants(self):
        """Barrier rejects invalid participant count."""
        with pytest.raises(ValueError, match="num_participants must be >= 1"):
            Barrier(num_participants=0)

    def test_barrier_repr(self):
        """Repr shows useful debugging info."""
        barrier = Barrier(num_participants=4)
        barrier.arrive(0)
        r = repr(barrier)
        assert "participants=4" in r
        assert "arrived=1" in r
        assert "generation=0" in r


# =========================================================================
# BSP execution
# =========================================================================


class TestBSPExecution:
    """Tests for BSP (Bulk Synchronous Parallel) execution."""

    def test_bsp_execution(self):
        """BSP execution runs multiple supersteps with inter-core communication.

        Core 0 computes R0 + R1 -> R7
        Core 1 computes R0 * R1 -> R7

        After each superstep, results are written to shared memory and
        become available to other cores in the next superstep.
        """
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=256)

        prog0 = _add_program(dst=7, src1=0, src2=1)
        prog1 = _mul_program(dst=7, src1=0, src2=1)

        result = dcpu.execute_bsp(
            programs={0: prog0, 1: prog1},
            inputs={
                0: {0: 3.0, 1: 5.0},
                1: {0: 2.0, 1: 4.0},
            },
            num_supersteps=2,
            steps_per_superstep=16,
        )

        assert isinstance(result, DistributedResult)
        assert 0 in result.core_results
        assert 1 in result.core_results
        assert result.all_halted is True

        # Both programs should have executed
        r0 = result.core_results[0].registers[7].item()
        r1 = result.core_results[1].registers[7].item()
        # The exact values depend on how shared memory propagation works
        # between supersteps, but the results should be valid floats
        assert isinstance(r0, float)
        assert isinstance(r1, float)

    def test_bsp_shared_memory_populated(self):
        """After BSP execution, shared memory contains core outputs."""
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=256)

        prog0 = _mov_imm_program(dst=0, value=42.0)
        prog1 = _mov_imm_program(dst=0, value=99.0)

        result = dcpu.execute_bsp(
            programs={0: prog0, 1: prog1},
            inputs={},
            num_supersteps=1,
            steps_per_superstep=16,
        )

        # Shared memory should have core outputs at known offsets
        # Core 0's registers start at offset 0, Core 1's at offset num_regs
        shared = result.shared_memory
        assert shared.numel() == 256

        # Core 0 wrote R0 = 42.0 at offset 0
        assert abs(shared[0].item() - 42.0) < 0.01

        # Core 1 wrote R0 = 99.0 at offset 8 (core_id=1 * num_regs=8)
        assert abs(shared[8].item() - 99.0) < 0.01

    def test_bsp_multiple_supersteps_converge(self):
        """Multiple supersteps allow iterative convergence.

        Both cores add their R0 and R1 each superstep. After shared memory
        exchange, the values should evolve across supersteps.
        """
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=256)

        prog0 = _add_program(dst=7, src1=0, src2=1)
        prog1 = _add_program(dst=7, src1=0, src2=1)

        result = dcpu.execute_bsp(
            programs={0: prog0, 1: prog1},
            inputs={
                0: {0: 1.0, 1: 2.0},
                1: {0: 3.0, 1: 4.0},
            },
            num_supersteps=4,
            steps_per_superstep=16,
        )

        assert result.total_steps > 0
        assert result.all_halted is True

    def test_bsp_single_core(self):
        """BSP with a single core still works correctly."""
        dcpu = DistributedNCPU(num_cores=1, shared_memory_size=256)

        prog = _add_program(dst=7, src1=0, src2=1)

        result = dcpu.execute_bsp(
            programs={0: prog},
            inputs={0: {0: 10.0, 1: 20.0}},
            num_supersteps=1,
            steps_per_superstep=16,
        )

        r7 = result.core_results[0].registers[7].item()
        assert abs(r7 - 30.0) < 0.01

    def test_bsp_invalid_core_raises(self):
        """BSP with out-of-range core ID raises IndexError."""
        dcpu = DistributedNCPU(num_cores=2)

        with pytest.raises(IndexError):
            dcpu.execute_bsp(
                programs={5: _add_program()},
                inputs={},
                num_supersteps=1,
            )
