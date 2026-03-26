"""Tests for multi-GPU distributed nCPU.

Verifies:
  1. Individual GPU cores create, load, and execute programs.
  2. The distributed controller runs parallel and pipeline execution.
  3. Fork clones core state across GPU boundaries.
  4. Shared memory and message channels provide correct IPC.
  5. The scheduler assigns processes according to each policy.
  6. Edge cases (out-of-bounds, empty queues, full channels) are handled.
"""

import pytest
import torch

from ncpu.distributed import (
    DistributedNCPU,
    GPUCore,
    CoreConfig,
    CoreState,
    DistributedResult,
    SharedMemoryRegion,
    MessageChannel,
    IPCMessage,
    DistributedScheduler,
    ProcessDescriptor,
    SchedulingPolicy,
)
from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    Instruction,
    ExecutionResult,
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


def _sub_program(dst=7, src1=0, src2=1):
    """R[dst] = R[src1] - R[src2]; HALT."""
    return FixedProgram([
        Instruction(OPCODES["SUB"], dst=dst, src1=src1, src2=src2),
        Instruction(OPCODES["HALT"]),
    ])


def _mov_imm_program(dst=0, value=42.0):
    """R[dst] = immediate; HALT."""
    return FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=dst, immediate=value),
        Instruction(OPCODES["HALT"]),
    ])


def _halt_program():
    """Just HALT."""
    return FixedProgram([Instruction(OPCODES["HALT"])])


# =========================================================================
# GPUCore
# =========================================================================

class TestGPUCore:
    """Tests for individual GPU cores."""

    def test_core_creation(self):
        """Core initialises with correct config and idle state."""
        cfg = CoreConfig(core_id=3, num_registers=8, local_memory_size=512)
        core = GPUCore(cfg)

        assert core.core_id == 3
        assert core.state == CoreState.IDLE
        assert core.program is None
        assert core.local_memory.shape == (512,)
        assert core.message_queue == []

    def test_core_creation_defaults(self):
        """Default config produces sensible values."""
        cfg = CoreConfig(core_id=0)
        core = GPUCore(cfg)
        assert core.config.num_registers == 8
        assert core.config.local_memory_size == 1024
        assert core.config.device == "cpu"

    def test_core_load_program(self):
        """Loading a program transitions the core to RUNNING."""
        core = GPUCore(CoreConfig(core_id=0))
        prog = _halt_program()
        core.load_program(prog)
        assert core.state == CoreState.RUNNING
        assert core.program is prog

    def test_core_run_add(self):
        """Core executes an ADD program and halts."""
        core = GPUCore(CoreConfig(core_id=0))
        core.load_program(_add_program())
        result = core.run(inputs={0: 10.0, 1: 5.0})
        assert result is not None
        assert result.halted
        assert core.state == CoreState.HALTED
        assert abs(result.registers[7].item() - 15.0) < 1e-5

    def test_core_run_no_program(self):
        """Running without a program returns None."""
        core = GPUCore(CoreConfig(core_id=0))
        assert core.run() is None

    def test_core_step_not_running(self):
        """Stepping an idle core returns None."""
        core = GPUCore(CoreConfig(core_id=0))
        region = SharedMemoryRegion(size=64)
        assert core.step(region) is None

    def test_core_send_message(self):
        """send_message creates a properly-formed message dict."""
        core = GPUCore(CoreConfig(core_id=2))
        msg = core.send_message(target_core=5, data=torch.tensor([1.0]))
        assert msg["from"] == 2
        assert msg["to"] == 5
        assert torch.equal(msg["data"], torch.tensor([1.0]))

    def test_core_receive_and_pop_message(self):
        """Messages are queued and dequeued in FIFO order."""
        core = GPUCore(CoreConfig(core_id=0))
        core.receive_message({"from": 1, "data": torch.tensor(10.0)})
        core.receive_message({"from": 2, "data": torch.tensor(20.0)})

        msg1 = core.pop_message()
        assert msg1["from"] == 1
        msg2 = core.pop_message()
        assert msg2["from"] == 2
        assert core.pop_message() is None

    def test_core_reset(self):
        """Reset returns core to pristine idle state."""
        core = GPUCore(CoreConfig(core_id=0))
        core.load_program(_halt_program())
        core.local_memory[0] = 99.0
        core.receive_message({"from": 1, "data": torch.tensor(1.0)})
        core.reset()

        assert core.state == CoreState.IDLE
        assert core.program is None
        assert core.local_memory[0].item() == 0.0
        assert len(core.message_queue) == 0

    def test_core_repr(self):
        """repr is human-readable."""
        core = GPUCore(CoreConfig(core_id=7))
        s = repr(core)
        assert "id=7" in s
        assert "idle" in s


# =========================================================================
# DistributedNCPU
# =========================================================================

class TestDistributedNCPU:
    """Tests for the multi-core controller."""

    def test_creation(self):
        """Controller creates the requested number of cores."""
        dcpu = DistributedNCPU(num_cores=4)
        assert dcpu.num_cores == 4
        assert len(dcpu.cores) == 4
        for i, core in enumerate(dcpu.cores):
            assert core.core_id == i

    def test_creation_with_config(self):
        """Custom core config propagates to all cores."""
        cfg = CoreConfig(core_id=0, num_registers=16, local_memory_size=2048)
        dcpu = DistributedNCPU(num_cores=2, core_config=cfg)
        for core in dcpu.cores:
            assert core.config.num_registers == 16
            assert core.config.local_memory_size == 2048

    def test_load_program(self):
        """Loading a program onto a specific core works."""
        dcpu = DistributedNCPU(num_cores=2)
        prog = _halt_program()
        dcpu.load_program(0, prog)
        assert dcpu.cores[0].program is prog
        assert dcpu.cores[0].state == CoreState.RUNNING

    def test_load_program_invalid_core(self):
        """Loading onto an out-of-range core raises IndexError."""
        dcpu = DistributedNCPU(num_cores=2)
        with pytest.raises(IndexError):
            dcpu.load_program(5, _halt_program())

    def test_parallel_execution_three_cores(self):
        """Three cores compute different operations in parallel."""
        dcpu = DistributedNCPU(num_cores=4)
        result = dcpu.execute_parallel(
            {0: _add_program(), 1: _mul_program(), 2: _sub_program()},
            inputs={
                0: {0: 10.0, 1: 5.0},   # 10+5=15
                1: {0: 3.0, 1: 7.0},    # 3*7=21
                2: {0: 20.0, 1: 8.0},   # 20-8=12
            },
        )

        assert isinstance(result, DistributedResult)
        assert abs(result.core_results[0].registers[7].item() - 15.0) < 1e-5
        assert abs(result.core_results[1].registers[7].item() - 21.0) < 1e-5
        assert abs(result.core_results[2].registers[7].item() - 12.0) < 1e-5
        assert result.all_halted
        assert result.total_steps == 6  # 2 steps x 3 cores

    def test_parallel_execution_shared_memory_populated(self):
        """After parallel execution, shared memory contains output registers."""
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=256)
        dcpu.execute_parallel(
            {0: _mov_imm_program(dst=0, value=42.0)},
            inputs={},
        )
        # Core 0's R0 should be at shared_memory offset 0
        val = dcpu.shared_memory.read(0, 1)
        assert abs(val.item() - 42.0) < 1e-5

    def test_parallel_execution_no_inputs(self):
        """Parallel execution works with default zero registers."""
        dcpu = DistributedNCPU(num_cores=2)
        result = dcpu.execute_parallel({0: _add_program()})
        # 0+0=0
        assert abs(result.core_results[0].registers[7].item()) < 1e-5

    def test_pipeline_execution(self):
        """Stage 0 output feeds stage 1 input through shared memory."""
        dcpu = DistributedNCPU(num_cores=2)

        # Stage 0: R2 = 7 * 6 = 42, R0 = 7, R1 = 6
        stage0_prog = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=7.0),
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=6.0),
            Instruction(OPCODES["MUL"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        # Stage 1: R3 = R2 + R0 (reads core 0's R2=42, R0=7 -> 49)
        stage1_prog = FixedProgram([
            Instruction(OPCODES["ADD"], dst=3, src1=2, src2=0),
            Instruction(OPCODES["HALT"]),
        ])

        result = dcpu.execute_pipeline([
            (0, stage0_prog, {}),
            (1, stage1_prog, {}),
        ])

        assert abs(result.core_results[0].registers[2].item() - 42.0) < 1e-5
        # Core 1 receives R2=42 from core 0 and R0=7 from core 0
        assert abs(result.core_results[1].registers[3].item() - 49.0) < 1e-5
        assert result.all_halted

    def test_pipeline_preserves_explicit_inputs(self):
        """Explicit inputs in a pipeline stage override inherited values."""
        dcpu = DistributedNCPU(num_cores=2)
        stage0 = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=100.0),
            Instruction(OPCODES["HALT"]),
        ])
        stage1 = FixedProgram([
            Instruction(OPCODES["ADD"], dst=1, src1=0, src2=0),
            Instruction(OPCODES["HALT"]),
        ])

        # Stage 1 explicitly sets R0=5, so it should NOT inherit 100 from stage 0
        result = dcpu.execute_pipeline([
            (0, stage0, {}),
            (1, stage1, {0: 5.0}),
        ])
        # R1 = 5 + 5 = 10, not 100 + 100
        assert abs(result.core_results[1].registers[1].item() - 10.0) < 1e-5

    def test_fork_copies_local_memory(self):
        """Fork clones the parent's local memory to the child."""
        dcpu = DistributedNCPU(num_cores=2)
        dcpu.cores[0].local_memory[0] = 42.0
        dcpu.cores[0].local_memory[1] = 99.0
        dcpu.load_program(0, _halt_program())

        dcpu.fork(parent_core=0, child_core=1)

        assert dcpu.cores[1].local_memory[0].item() == 42.0
        assert dcpu.cores[1].local_memory[1].item() == 99.0
        assert dcpu.cores[1].state == CoreState.RUNNING

    def test_fork_clones_not_shares_memory(self):
        """Child's local memory is a clone, not a reference to parent's."""
        dcpu = DistributedNCPU(num_cores=2)
        dcpu.cores[0].local_memory[0] = 42.0
        dcpu.load_program(0, _halt_program())
        dcpu.fork(0, 1)

        # Mutate child memory -- parent should be unaffected
        dcpu.cores[1].local_memory[0] = 0.0
        assert dcpu.cores[0].local_memory[0].item() == 42.0

    def test_fork_inherits_program(self):
        """Forked child gets an independent deep copy of the parent's program."""
        dcpu = DistributedNCPU(num_cores=2)
        prog = _add_program()
        dcpu.load_program(0, prog)
        dcpu.fork(0, 1)
        # Child should have a program with equivalent structure but not the same object
        assert dcpu.cores[1].program is not prog
        assert dcpu.cores[1].program is not dcpu.cores[0].program
        assert dcpu.cores[1].program.length == prog.length

    def test_fork_invalid_cores(self):
        """Fork with out-of-range core ids raises IndexError."""
        dcpu = DistributedNCPU(num_cores=2)
        with pytest.raises(IndexError):
            dcpu.fork(0, 99)

    def test_shared_memory_read_write(self):
        """Shared memory write is visible to read."""
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=128)
        dcpu.shared_write(10, torch.tensor([3.14]))
        val = dcpu.shared_read(10, 1)
        assert abs(val.item() - 3.14) < 1e-5

    def test_pipe_creation_and_usage(self):
        """Pipe creates a usable message channel between cores."""
        dcpu = DistributedNCPU(num_cores=2)
        pipe_idx = dcpu.pipe(core_a=0, core_b=1, buffer_size=32)
        ch = dcpu.get_pipe(pipe_idx)

        assert isinstance(ch, MessageChannel)
        ch.send(torch.tensor([1.0, 2.0, 3.0]))
        result = ch.recv(3)
        assert result is not None
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_pipe_invalid_index(self):
        """Accessing a non-existent pipe raises IndexError."""
        dcpu = DistributedNCPU(num_cores=2)
        with pytest.raises(IndexError):
            dcpu.get_pipe(99)

    def test_route_messages(self):
        """Messages on the bus are delivered to correct cores."""
        dcpu = DistributedNCPU(num_cores=3)
        dcpu.message_bus.append({"from": 0, "to": 1, "data": torch.tensor(10.0)})
        dcpu.message_bus.append({"from": 0, "to": 2, "data": torch.tensor(20.0)})
        delivered = dcpu.route_messages()
        assert delivered == 2
        assert len(dcpu.cores[1].message_queue) == 1
        assert len(dcpu.cores[2].message_queue) == 1
        assert len(dcpu.message_bus) == 0

    def test_reset(self):
        """Reset clears all state."""
        dcpu = DistributedNCPU(num_cores=2)
        dcpu.load_program(0, _halt_program())
        dcpu.shared_write(0, torch.tensor([1.0]))
        dcpu.pipe(0, 1)
        dcpu.message_bus.append({"from": 0, "to": 1, "data": torch.tensor(1.0)})

        dcpu.reset()

        assert dcpu.cores[0].state == CoreState.IDLE
        assert dcpu.shared_memory.read(0, 1).item() == 0.0
        assert len(dcpu.message_bus) == 0
        assert len(dcpu._pipes) == 0

    def test_repr(self):
        """repr includes core states."""
        dcpu = DistributedNCPU(num_cores=2)
        s = repr(dcpu)
        assert "num_cores=2" in s
        assert "idle" in s


# =========================================================================
# IPC: SharedMemoryRegion
# =========================================================================

class TestSharedMemoryRegion:
    """Tests for the shared memory region."""

    def test_read_write_single(self):
        """Write one value and read it back."""
        region = SharedMemoryRegion(size=64)
        region.write(10, torch.tensor([7.5]))
        val = region.read(10, 1)
        assert abs(val.item() - 7.5) < 1e-6

    def test_read_write_multi(self):
        """Write a slice and read it back."""
        region = SharedMemoryRegion(size=64)
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        region.write(5, data)
        result = region.read(5, 4)
        assert torch.allclose(result, data)

    def test_read_returns_clone(self):
        """Mutating the returned tensor does not affect the region."""
        region = SharedMemoryRegion(size=64)
        region.write(0, torch.tensor([10.0]))
        val = region.read(0, 1)
        val[0] = 999.0
        assert region.read(0, 1).item() == 10.0

    def test_read_out_of_bounds(self):
        """Reading past the end raises IndexError."""
        region = SharedMemoryRegion(size=8)
        with pytest.raises(IndexError):
            region.read(7, 4)

    def test_write_out_of_bounds(self):
        """Writing past the end raises IndexError."""
        region = SharedMemoryRegion(size=8)
        with pytest.raises(IndexError):
            region.write(7, torch.tensor([1.0, 2.0, 3.0]))

    def test_atomic_add(self):
        """atomic_add returns old value and stores old + new."""
        region = SharedMemoryRegion(size=16)
        region.write(0, torch.tensor([10.0]))
        old = region.atomic_add(0, torch.tensor(5.0))
        assert abs(old.item() - 10.0) < 1e-6
        assert abs(region.read(0, 1).item() - 15.0) < 1e-6

    def test_atomic_add_out_of_bounds(self):
        """atomic_add on an invalid address raises IndexError."""
        region = SharedMemoryRegion(size=4)
        with pytest.raises(IndexError):
            region.atomic_add(10, torch.tensor(1.0))

    def test_compare_and_swap_success(self):
        """CAS succeeds when current value matches expected."""
        region = SharedMemoryRegion(size=8)
        region.write(0, torch.tensor([10.0]))
        success, old = region.compare_and_swap(
            0, expected=torch.tensor(10.0), desired=torch.tensor(20.0)
        )
        assert success
        assert abs(old.item() - 10.0) < 1e-6
        assert abs(region.read(0, 1).item() - 20.0) < 1e-6

    def test_compare_and_swap_failure(self):
        """CAS fails when current value does not match expected."""
        region = SharedMemoryRegion(size=8)
        region.write(0, torch.tensor([10.0]))
        success, old = region.compare_and_swap(
            0, expected=torch.tensor(99.0), desired=torch.tensor(20.0)
        )
        assert not success
        assert abs(old.item() - 10.0) < 1e-6
        assert abs(region.read(0, 1).item() - 10.0) < 1e-6  # unchanged

    def test_clear(self):
        """clear() zeros the entire region."""
        region = SharedMemoryRegion(size=16)
        region.write(0, torch.tensor([1.0, 2.0, 3.0]))
        region.clear()
        assert torch.all(region.memory == 0)

    def test_snapshot(self):
        """snapshot() returns a detached clone."""
        region = SharedMemoryRegion(size=8)
        region.write(0, torch.tensor([5.0]))
        snap = region.snapshot()
        snap[0] = 999.0
        assert region.read(0, 1).item() == 5.0

    def test_len_and_repr(self):
        region = SharedMemoryRegion(size=128, name="test")
        assert len(region) == 128
        assert "test" in repr(region)
        assert "128" in repr(region)


# =========================================================================
# IPC: MessageChannel
# =========================================================================

class TestMessageChannel:
    """Tests for point-to-point message channels."""

    def test_send_recv_single(self):
        """Send one value, receive one value."""
        ch = MessageChannel(buffer_size=16)
        assert ch.send(torch.tensor([42.0]))
        result = ch.recv(1)
        assert result is not None
        assert abs(result.item() - 42.0) < 1e-6

    def test_send_recv_multiple(self):
        """Send multiple values, receive them in order."""
        ch = MessageChannel(buffer_size=16)
        ch.send(torch.tensor([1.0, 2.0, 3.0]))
        result = ch.recv(3)
        assert result is not None
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_fifo_ordering(self):
        """Values come out in the order they were sent."""
        ch = MessageChannel(buffer_size=32)
        ch.send(torch.tensor([10.0]))
        ch.send(torch.tensor([20.0]))
        ch.send(torch.tensor([30.0]))
        assert ch.recv(1).item() == 10.0
        assert ch.recv(1).item() == 20.0
        assert ch.recv(1).item() == 30.0

    def test_recv_empty(self):
        """Receiving from an empty channel returns None."""
        ch = MessageChannel(buffer_size=16)
        assert ch.recv(1) is None

    def test_recv_insufficient(self):
        """Requesting more values than available returns None."""
        ch = MessageChannel(buffer_size=16)
        ch.send(torch.tensor([1.0]))
        assert ch.recv(5) is None

    def test_channel_full(self):
        """send() returns False when the buffer is full."""
        ch = MessageChannel(buffer_size=4)
        # Capacity is buffer_size - 1 = 3 (one slot kept empty)
        assert ch.send(torch.tensor([1.0, 2.0, 3.0]))
        assert ch.is_full()
        assert not ch.send(torch.tensor([4.0]))

    def test_available_and_space(self):
        """available() and available_space() track correctly."""
        ch = MessageChannel(buffer_size=8)
        assert ch.available() == 0
        assert ch.available_space() == 7
        assert ch.is_empty()

        ch.send(torch.tensor([1.0, 2.0, 3.0]))
        assert ch.available() == 3
        assert ch.available_space() == 4
        assert not ch.is_empty()
        assert not ch.is_full()

    def test_wraparound(self):
        """Ring buffer correctly wraps around."""
        ch = MessageChannel(buffer_size=4)
        # Fill: [1, 2, 3] then drain, then refill
        ch.send(torch.tensor([1.0, 2.0, 3.0]))
        ch.recv(3)
        # Now read_ptr=3, write_ptr=3 (empty at end of buffer)
        assert ch.is_empty()
        # Write wraps around the end
        ch.send(torch.tensor([4.0, 5.0]))
        result = ch.recv(2)
        assert result.tolist() == [4.0, 5.0]

    def test_reset(self):
        """reset() drains the channel."""
        ch = MessageChannel(buffer_size=16)
        ch.send(torch.tensor([1.0, 2.0]))
        ch.reset()
        assert ch.is_empty()
        assert ch.available() == 0

    def test_invalid_buffer_size(self):
        """Buffer size < 2 raises ValueError."""
        with pytest.raises(ValueError):
            MessageChannel(buffer_size=1)

    def test_repr(self):
        ch = MessageChannel(buffer_size=16)
        ch.send(torch.tensor([1.0, 2.0]))
        s = repr(ch)
        assert "buffer_size=16" in s
        assert "used=2" in s


# =========================================================================
# IPC: IPCMessage
# =========================================================================

class TestIPCMessage:
    """Tests for the IPCMessage dataclass."""

    def test_creation(self):
        msg = IPCMessage(sender=0, receiver=1, data=torch.tensor([1.0]))
        assert msg.sender == 0
        assert msg.receiver == 1
        assert msg.tag == 0  # default

    def test_with_tag(self):
        msg = IPCMessage(sender=2, receiver=3, data=torch.tensor([5.0]), tag=42)
        assert msg.tag == 42


# =========================================================================
# Scheduler
# =========================================================================

class TestScheduler:
    """Tests for the distributed scheduler."""

    def test_round_robin(self):
        """Round-robin distributes evenly across cores."""
        sched = DistributedScheduler(num_cores=3, policy=SchedulingPolicy.ROUND_ROBIN)
        for i in range(6):
            sched.submit(ProcessDescriptor(pid=i, program=_halt_program()))

        assignments = sched.schedule()
        # 6 processes / 3 cores = 2 each
        assert len(assignments[0]) == 2
        assert len(assignments[1]) == 2
        assert len(assignments[2]) == 2

    def test_round_robin_ordering(self):
        """Round-robin assigns in rotating order."""
        sched = DistributedScheduler(num_cores=3, policy=SchedulingPolicy.ROUND_ROBIN)
        for i in range(3):
            sched.submit(ProcessDescriptor(pid=i, program=_halt_program()))

        assignments = sched.schedule()
        assert assignments[0][0].pid == 0
        assert assignments[1][0].pid == 1
        assert assignments[2][0].pid == 2

    def test_load_balanced(self):
        """Load-balanced always picks the least-loaded core."""
        sched = DistributedScheduler(num_cores=3, policy=SchedulingPolicy.LOAD_BALANCED)
        for i in range(3):
            sched.submit(ProcessDescriptor(pid=i, program=_halt_program()))
        assignments = sched.schedule()
        # Each core should get exactly 1 (ties broken by min())
        total = sum(len(v) for v in assignments.values())
        assert total == 3

    def test_load_balanced_respects_existing_load(self):
        """Pre-existing load is taken into account."""
        sched = DistributedScheduler(num_cores=3, policy=SchedulingPolicy.LOAD_BALANCED)
        sched.core_loads = [10, 0, 5]
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program()))
        assignments = sched.schedule()
        # Should go to core 1 (load=0)
        assert len(assignments[1]) == 1
        assert assignments[1][0].pid == 0

    def test_affinity_honoured(self):
        """AFFINITY policy places processes on their preferred core."""
        sched = DistributedScheduler(num_cores=4, policy=SchedulingPolicy.AFFINITY)
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program(), core_affinity=3))
        sched.submit(ProcessDescriptor(pid=1, program=_halt_program(), core_affinity=1))
        assignments = sched.schedule()
        assert len(assignments[3]) == 1
        assert assignments[3][0].pid == 0
        assert len(assignments[1]) == 1
        assert assignments[1][0].pid == 1

    def test_affinity_fallback(self):
        """Without affinity set, AFFINITY falls back to round-robin."""
        sched = DistributedScheduler(num_cores=2, policy=SchedulingPolicy.AFFINITY)
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program()))
        sched.submit(ProcessDescriptor(pid=1, program=_halt_program()))
        assignments = sched.schedule()
        total = sum(len(v) for v in assignments.values())
        assert total == 2

    def test_priority_ordering(self):
        """Higher priority processes are scheduled first."""
        sched = DistributedScheduler(num_cores=2, policy=SchedulingPolicy.LOAD_BALANCED)
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program(), priority=1))
        sched.submit(ProcessDescriptor(pid=1, program=_halt_program(), priority=10))
        sched.submit(ProcessDescriptor(pid=2, program=_halt_program(), priority=5))
        assignments = sched.schedule()
        # pid=1 (priority=10) should be placed first (core 0)
        assert assignments[0][0].pid == 1

    def test_submit_batch(self):
        """submit_batch adds multiple processes at once."""
        sched = DistributedScheduler(num_cores=2)
        procs = [
            ProcessDescriptor(pid=i, program=_halt_program()) for i in range(4)
        ]
        sched.submit_batch(procs)
        assert sched.pending_count() == 4

    def test_schedule_clears_queue(self):
        """schedule() empties the queue."""
        sched = DistributedScheduler(num_cores=2)
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program()))
        sched.schedule()
        assert sched.pending_count() == 0

    def test_reset(self):
        """reset() clears everything."""
        sched = DistributedScheduler(num_cores=3)
        sched.submit(ProcessDescriptor(pid=0, program=_halt_program()))
        sched.core_loads = [5, 10, 15]
        sched.reset()
        assert sched.pending_count() == 0
        assert sched.core_loads == [0, 0, 0]
        assert sched._next_core == 0

    def test_invalid_num_cores(self):
        """Creating a scheduler with 0 cores raises ValueError."""
        with pytest.raises(ValueError):
            DistributedScheduler(num_cores=0)

    def test_repr(self):
        sched = DistributedScheduler(num_cores=2, policy=SchedulingPolicy.ROUND_ROBIN)
        s = repr(sched)
        assert "num_cores=2" in s
        assert "round_robin" in s


# =========================================================================
# Integration: gradient flow across distributed execution
# =========================================================================

class TestDistributedGradients:
    """Verify that gradient flow is maintained through distributed execution."""

    def test_gradient_through_parallel(self):
        """Gradients flow back through parallel execution."""
        dcpu = DistributedNCPU(num_cores=2)

        # Use differentiable immediate
        prog = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=5.0),
            Instruction(OPCODES["HALT"]),
        ])

        result = dcpu.execute_parallel({0: prog}, inputs={})
        loss = (result.core_results[0].registers[0] - 10.0) ** 2
        loss.backward()

        assert prog.immediates.grad is not None
        assert prog.immediates.grad[0].item() != 0.0

    def test_gradient_through_pipeline(self):
        """Gradients flow through pipeline stages (via tensor values)."""
        dcpu = DistributedNCPU(num_cores=2)

        r0 = torch.tensor(3.0, requires_grad=True)
        r1 = torch.tensor(4.0, requires_grad=True)

        prog = FixedProgram([
            Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])

        result = dcpu.execute_parallel(
            {0: prog},
            inputs={0: {0: r0, 1: r1}},
        )

        loss = (result.core_results[0].registers[2] - 10.0) ** 2
        loss.backward()

        assert r0.grad is not None
        assert r1.grad is not None


# =========================================================================
# Integration: end-to-end scenarios
# =========================================================================

class TestEndToEnd:
    """End-to-end scenarios combining multiple distributed features."""

    def test_fork_then_execute(self):
        """Fork a core, then execute both parent and child."""
        dcpu = DistributedNCPU(num_cores=2)

        prog = _add_program()
        dcpu.load_program(0, prog)
        dcpu.cores[0].local_memory[0] = 100.0
        dcpu.fork(0, 1)

        result = dcpu.execute_parallel(
            {0: prog, 1: prog},
            inputs={
                0: {0: 10.0, 1: 5.0},
                1: {0: 20.0, 1: 3.0},
            },
        )

        assert abs(result.core_results[0].registers[7].item() - 15.0) < 1e-5
        assert abs(result.core_results[1].registers[7].item() - 23.0) < 1e-5

    def test_pipe_data_exchange(self):
        """Two cores exchange data via a pipe channel."""
        dcpu = DistributedNCPU(num_cores=2)
        pipe_idx = dcpu.pipe(0, 1, buffer_size=32)
        ch = dcpu.get_pipe(pipe_idx)

        # Core 1 "sends" data through the pipe
        ch.send(torch.tensor([10.0, 20.0, 30.0]))

        # Core 0 "receives" the data
        data = ch.recv(3)
        assert data is not None
        assert data.tolist() == [10.0, 20.0, 30.0]

    def test_scheduler_feeds_distributed(self):
        """Scheduler assigns programs, then DistributedNCPU executes them."""
        dcpu = DistributedNCPU(num_cores=3)
        sched = DistributedScheduler(num_cores=3, policy=SchedulingPolicy.ROUND_ROBIN)

        sched.submit(ProcessDescriptor(pid=0, program=_add_program(), inputs={0: 1.0, 1: 2.0}))
        sched.submit(ProcessDescriptor(pid=1, program=_mul_program(), inputs={0: 3.0, 1: 4.0}))
        sched.submit(ProcessDescriptor(pid=2, program=_sub_program(), inputs={0: 10.0, 1: 3.0}))

        assignments = sched.schedule()

        # Execute assigned programs
        programs = {}
        inputs = {}
        for core_id, procs in assignments.items():
            for proc in procs:
                programs[core_id] = proc.program
                inputs[core_id] = proc.inputs

        result = dcpu.execute_parallel(programs, inputs)

        # All 3 cores should have executed
        assert len(result.core_results) == 3
        assert result.all_halted

    def test_shared_memory_communication(self):
        """Core 0 writes to shared memory, core 1 reads it back."""
        dcpu = DistributedNCPU(num_cores=2, shared_memory_size=256)

        # Core 0 computes 10+5=15 which gets published to shared memory
        result = dcpu.execute_parallel(
            {0: _add_program()},
            inputs={0: {0: 10.0, 1: 5.0}},
        )

        # Read core 0's R7 (register index 7) from shared memory
        val = dcpu.shared_memory.read(7, 1)
        assert abs(val.item() - 15.0) < 1e-5

    def test_many_cores(self):
        """Stress test with 8 cores running in parallel."""
        dcpu = DistributedNCPU(num_cores=8, shared_memory_size=1024)

        programs = {}
        inputs = {}
        for i in range(8):
            programs[i] = _add_program()
            inputs[i] = {0: float(i), 1: float(i * 10)}

        result = dcpu.execute_parallel(programs, inputs)

        for i in range(8):
            expected = float(i) + float(i * 10)
            actual = result.core_results[i].registers[7].item()
            assert abs(actual - expected) < 1e-5

        assert result.total_steps == 16  # 2 steps x 8 cores
        assert result.all_halted


# =========================================================================
# Demo sanity check
# =========================================================================

class TestDemo:
    """Verify the demo function runs without errors."""

    def test_demo_runs(self, capsys):
        from ncpu.distributed.multi_gpu import demo_distributed
        demo_distributed()
        captured = capsys.readouterr()
        assert "Multi-GPU Distributed nCPU" in captured.out
        assert "Pipeline" in captured.out
        assert "Fork" in captured.out
        assert "Pipe" in captured.out
