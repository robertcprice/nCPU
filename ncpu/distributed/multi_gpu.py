"""Multi-GPU distributed nCPU execution.

Each GPU is a core.  Multiple cores form a distributed neural computer.
The cores communicate via shared memory and message channels.

On Apple Silicon, "multi-GPU" leverages the unified memory architecture:
each core runs a separate DifferentiableEngine, but all share the same
physical memory (Metal's StorageModeShared).  On discrete GPUs, each core
would reside on a different CUDA device with inter-device transfers via
NCCL or CUDA IPC.

Execution models:
  - **Parallel**: every core runs an independent program simultaneously.
  - **Pipeline**: core N's output feeds core N+1's input (dataflow).
  - **Fork**: clone a running core's state onto an idle core (like UNIX fork).
  - **Pipe**: allocate a shared-memory FIFO between two cores.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    ExecutionResult,
    FixedProgram,
    Instruction,
    OPCODES,
    SoftProgram,
)

from .ipc import Barrier, MessageChannel, SharedMemoryRegion
from .scheduler import DistributedScheduler, ProcessDescriptor


# ---------------------------------------------------------------------------
# Core state machine
# ---------------------------------------------------------------------------

class CoreState(Enum):
    """Lifecycle states for a single GPU core."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"  # blocked on IPC / synchronisation
    HALTED = "halted"


# ---------------------------------------------------------------------------
# Per-core configuration
# ---------------------------------------------------------------------------

@dataclass
class CoreConfig:
    """Configuration for a single GPU core.

    Attributes:
        core_id: Unique integer identifier for this core.
        num_registers: Width of the register file.
        local_memory_size: Words of core-private memory.
        device: PyTorch device string ("cpu", "mps", "cuda:0", ...).
    """

    core_id: int
    num_registers: int = 8
    local_memory_size: int = 1024
    device: str = "cpu"


@dataclass
class DeviceInfo:
    """Description of one execution device visible to the host."""

    name: str
    kind: str
    index: Optional[int]
    available: bool = True


@dataclass
class DeviceAssignment:
    """Maps a core to the execution device it should use."""

    core_id: int
    requested_device: str
    assigned_device: str
    reason: str


# ---------------------------------------------------------------------------
# Distributed execution result
# ---------------------------------------------------------------------------

@dataclass
class DistributedResult:
    """Result of distributed execution across multiple cores.

    Attributes:
        core_results: Mapping from core id to its ExecutionResult.
        shared_memory: Snapshot of the shared memory region after execution.
        messages_exchanged: Total number of messages delivered.
        total_steps: Sum of steps executed across all cores.
        all_halted: True if every active core reached HALT.
    """

    core_results: dict[int, ExecutionResult]
    shared_memory: torch.Tensor
    messages_exchanged: int
    total_steps: int
    all_halted: bool
    device_assignments: dict[int, str] = field(default_factory=dict)


@dataclass
class ScheduledExecutionResult:
    """Result of scheduling processes then executing them on the controller."""

    process_results: dict[int, ExecutionResult]
    process_to_core: dict[int, int]
    core_to_pids: dict[int, list[int]]
    device_assignments: dict[int, str]
    scheduled_count: int


# ---------------------------------------------------------------------------
# A single GPU core
# ---------------------------------------------------------------------------

class GPUCore(nn.Module):
    """A single core in the distributed nCPU.

    Each core has:
      - Its own DifferentiableEngine (registers, PC, flags).
      - Local memory (private to this core).
      - Access to shared memory (visible to all cores).
      - A message queue for inter-core communication.
    """

    def __init__(self, config: CoreConfig) -> None:
        super().__init__()
        self.config = config
        self.core_id = config.core_id
        self.engine = DifferentiableEngine(num_registers=config.num_registers)

        # Core-local memory: private scratchpad that is not visible to
        # other cores (analogous to L1 cache).
        self.local_memory = torch.zeros(config.local_memory_size)

        self.state: CoreState = CoreState.IDLE
        self.program: Optional[FixedProgram] = None
        self.message_queue: list[dict] = []

    # -- program management -------------------------------------------------

    def load_program(self, program: FixedProgram) -> None:
        """Load a program onto this core and mark it runnable."""
        self.program = program
        self.state = CoreState.RUNNING

    # -- execution ----------------------------------------------------------

    def step(
        self,
        shared_memory: SharedMemoryRegion,
        inputs: Optional[dict[int, float]] = None,
    ) -> Optional[ExecutionResult]:
        """Execute one step on this core.

        Returns the execution result or None if the core is not running.
        """
        if self.state != CoreState.RUNNING or self.program is None:
            return None

        inputs = inputs or {}
        result = self.engine.execute_fixed(self.program, inputs, max_steps=1)
        if result.halted:
            self.state = CoreState.HALTED
        return result

    def run(
        self,
        inputs: Optional[dict[int, float]] = None,
        max_steps: int = 64,
    ) -> Optional[ExecutionResult]:
        """Run the loaded program to completion (up to *max_steps*).

        Returns the execution result or None if no program is loaded.
        """
        if self.program is None:
            return None
        self.state = CoreState.RUNNING
        inputs = inputs or {}
        result = self.engine.execute_fixed(self.program, inputs, max_steps=max_steps)
        if result.halted:
            self.state = CoreState.HALTED
        return result

    # -- messaging ----------------------------------------------------------

    def send_message(self, target_core: int, data: torch.Tensor) -> dict:
        """Create a message destined for *target_core*."""
        return {"from": self.core_id, "to": target_core, "data": data}

    def receive_message(self, message: dict) -> None:
        """Enqueue an incoming message."""
        self.message_queue.append(message)

    def pop_message(self) -> Optional[dict]:
        """Dequeue the oldest message, or None if empty."""
        if self.message_queue:
            return self.message_queue.pop(0)
        return None

    # -- state management ---------------------------------------------------

    def reset(self) -> None:
        """Reset the core to its initial idle state."""
        self.state = CoreState.IDLE
        self.program = None
        self.local_memory.zero_()
        self.message_queue.clear()

    def __repr__(self) -> str:
        return (
            f"GPUCore(id={self.core_id}, state={self.state.value}, "
            f"has_program={self.program is not None})"
        )


# ---------------------------------------------------------------------------
# Distributed nCPU: the multi-core controller
# ---------------------------------------------------------------------------

class DistributedNCPU(nn.Module):
    """Multi-GPU distributed neural computer.

    Manages multiple GPUCores with shared memory and message passing.
    Supports:
      - Parallel execution across cores
      - Shared memory for inter-core data exchange
      - Message passing for synchronisation
      - Fork: spawn a new process on another core
      - Pipe: create a communication channel between cores
      - Wait: block until another core completes
      - Pipeline: staged dataflow execution across cores
      - Device-aware dispatch planning across CPU / MPS / CUDA backends

    On Apple Silicon, "multi-GPU" uses the unified memory architecture:
    each core runs a separate DifferentiableEngine, but all share the
    same physical memory (Metal's StorageModeShared).
    """

    def __init__(
        self,
        num_cores: int = 4,
        shared_memory_size: int = 4096,
        core_config: Optional[CoreConfig] = None,
        devices: Optional[list[str]] = None,
        device_strategy: str = "round_robin",
    ) -> None:
        super().__init__()
        self.num_cores = num_cores
        self.device_strategy = device_strategy
        self.available_devices = self.discover_devices()
        self.device_assignments: list[DeviceAssignment] = []

        # Shared memory visible to all cores
        self.shared_memory = SharedMemoryRegion(
            size=shared_memory_size, name="global"
        )

        requested_devices = devices or self._expand_requested_devices(
            core_config.device if core_config else "cpu"
        )

        # Build individual cores
        self.cores = nn.ModuleList()
        for i in range(num_cores):
            assigned_device, reason = self._assign_device_for_core(
                i, requested_devices
            )
            cfg = CoreConfig(
                core_id=i,
                num_registers=(
                    core_config.num_registers if core_config else 8
                ),
                local_memory_size=(
                    core_config.local_memory_size if core_config else 1024
                ),
                device=assigned_device,
            )
            self.device_assignments.append(
                DeviceAssignment(
                    core_id=i,
                    requested_device=requested_devices[i % len(requested_devices)],
                    assigned_device=assigned_device,
                    reason=reason,
                )
            )
            self.cores.append(GPUCore(cfg))

        # Message bus: collects all in-flight messages for routing
        self.message_bus: list[dict] = []

        # Pipe registry: tracks active pipes between cores
        self._pipes: list[MessageChannel] = []
        self._pipe_endpoints: list[tuple[int, int]] = []  # (reader, writer)

    # -- device discovery / assignment -------------------------------------

    @staticmethod
    def discover_devices() -> list[DeviceInfo]:
        """Discover execution backends visible to PyTorch."""
        devices = [DeviceInfo(name="cpu", kind="cpu", index=None)]
        if torch.backends.mps.is_available():
            devices.append(DeviceInfo(name="mps", kind="mps", index=None))
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                devices.append(DeviceInfo(name=f"cuda:{idx}", kind="cuda", index=idx))
        return devices

    def _expand_requested_devices(self, requested: str) -> list[str]:
        if requested == "auto":
            gpu_devices = [d.name for d in self.available_devices if d.kind in {"cuda", "mps"}]
            return gpu_devices or ["cpu"]
        return [requested]

    def _assign_device_for_core(self, core_id: int, requested_devices: list[str]) -> tuple[str, str]:
        if not requested_devices:
            return "cpu", "no devices requested; fell back to cpu"

        if self.device_strategy == "mirror":
            requested = requested_devices[0]
        else:
            requested = requested_devices[core_id % len(requested_devices)]

        available_names = {d.name for d in self.available_devices}
        if requested in available_names:
            return requested, f"requested device '{requested}' is available"

        if requested == "auto":
            fallback = self._expand_requested_devices("auto")[0]
            return fallback, "auto-selected first available accelerator or cpu"

        return "cpu", f"requested device '{requested}' unavailable; fell back to cpu"

    def get_device_map(self) -> dict[int, str]:
        """Return the assigned device for each core."""
        return {assignment.core_id: assignment.assigned_device for assignment in self.device_assignments}

    def get_device_assignment_report(self) -> list[dict[str, str | int]]:
        """Human-readable report for dispatch decisions."""
        return [
            {
                "core_id": assignment.core_id,
                "requested_device": assignment.requested_device,
                "assigned_device": assignment.assigned_device,
                "reason": assignment.reason,
            }
            for assignment in self.device_assignments
        ]

    def rebalance_devices(self, devices: list[str], strategy: str = "round_robin") -> None:
        """Reassign core devices without rebuilding the controller."""
        self.device_strategy = strategy
        self.device_assignments = []
        for i, core in enumerate(self.cores):
            assigned_device, reason = self._assign_device_for_core(i, devices)
            core.config.device = assigned_device
            self.device_assignments.append(
                DeviceAssignment(
                    core_id=i,
                    requested_device=devices[i % len(devices)],
                    assigned_device=assigned_device,
                    reason=reason,
                )
            )

    # -- program loading ----------------------------------------------------

    def load_program(self, core_id: int, program: FixedProgram) -> None:
        """Load a program onto a specific core."""
        self._validate_core(core_id)
        self.cores[core_id].load_program(program)

    # -- fork / pipe / wait -------------------------------------------------

    def fork(self, parent_core: int, child_core: int) -> None:
        """Fork: copy parent core's state to child core.

        Analogous to UNIX fork() but across GPU cores.  The child receives
        a *clone* of the parent's local memory and inherits its program.
        Both cores continue running independently from that point.
        """
        self._validate_core(parent_core)
        self._validate_core(child_core)
        parent = self.cores[parent_core]
        child = self.cores[child_core]

        if parent.program is not None:
            child.program = copy.deepcopy(parent.program)  # deep copy to isolate gradients
        child.local_memory = parent.local_memory.clone()
        child.state = CoreState.RUNNING

    def pipe(self, core_a: int, core_b: int, buffer_size: int = 256) -> int:
        """Create a pipe between two cores.

        Returns the pipe index which can be used to reference the channel
        via ``get_pipe()``.

        Args:
            core_a: Reader core id.
            core_b: Writer core id.
            buffer_size: Ring-buffer capacity in words.

        Returns:
            Integer pipe index.
        """
        self._validate_core(core_a)
        self._validate_core(core_b)
        channel = MessageChannel(buffer_size=buffer_size)
        pipe_idx = len(self._pipes)
        self._pipes.append(channel)
        self._pipe_endpoints.append((core_a, core_b))
        return pipe_idx

    def get_pipe(self, pipe_idx: int) -> MessageChannel:
        """Retrieve the MessageChannel for an existing pipe."""
        if pipe_idx < 0 or pipe_idx >= len(self._pipes):
            raise IndexError(f"pipe index {pipe_idx} out of range")
        return self._pipes[pipe_idx]

    def wait(self, core_id: int) -> Optional[ExecutionResult]:
        """Wait for a core to reach the HALTED state.

        In this synchronous implementation the core must already have been
        executed.  Returns None if the core never ran.
        """
        self._validate_core(core_id)
        core = self.cores[core_id]
        if core.state == CoreState.HALTED:
            # Re-execute to obtain the result (engines are stateless between
            # calls, so we replay).
            return None  # caller should read from core_results
        return None

    # -- shared memory convenience ------------------------------------------

    def shared_write(self, address: int, value: torch.Tensor) -> None:
        """Write a value to the global shared memory."""
        self.shared_memory.write(address, value)

    def shared_read(self, address: int, length: int = 1) -> torch.Tensor:
        """Read from the global shared memory."""
        return self.shared_memory.read(address, length)

    # -- message routing ----------------------------------------------------

    def route_messages(self) -> int:
        """Deliver all messages on the bus to their target cores.

        Returns the number of messages delivered.
        """
        delivered = 0
        for msg in self.message_bus:
            target = msg["to"]
            if 0 <= target < self.num_cores:
                self.cores[target].receive_message(msg)
                delivered += 1
        self.message_bus.clear()
        return delivered

    # -- scheduler integration ---------------------------------------------

    def execute_scheduled(
        self,
        processes: list[ProcessDescriptor],
        scheduler: Optional[DistributedScheduler] = None,
        max_steps: int = 64,
    ) -> ScheduledExecutionResult:
        """Schedule processes across cores, then execute them.

        If multiple processes land on the same core, they are executed
        sequentially in assignment order and returned per pid.
        """
        scheduler = scheduler or DistributedScheduler(
            num_cores=self.num_cores,
            core_devices=[self.get_device_map()[i] for i in range(self.num_cores)],
        )
        scheduler.set_core_devices([self.get_device_map()[i] for i in range(self.num_cores)])
        scheduler.submit_batch(processes)
        assignments = scheduler.schedule()

        process_results: dict[int, ExecutionResult] = {}
        process_to_core: dict[int, int] = {}
        core_to_pids: dict[int, list[int]] = {cid: [] for cid in range(self.num_cores)}

        for core_id, procs in assignments.items():
            core = self.cores[core_id]
            for proc in procs:
                core.load_program(proc.program)
                result = core.run(inputs=proc.inputs, max_steps=max_steps)
                if result is None:
                    continue
                process_results[proc.pid] = result
                process_to_core[proc.pid] = core_id
                core_to_pids[core_id].append(proc.pid)
                if result.registers is not None:
                    num_regs = result.registers.numel()
                    base = core_id * num_regs
                    if base + num_regs <= self.shared_memory.size:
                        self.shared_memory.write(base, result.registers.detach())
                scheduler.complete(core_id)

        return ScheduledExecutionResult(
            process_results=process_results,
            process_to_core=process_to_core,
            core_to_pids=core_to_pids,
            device_assignments=self.get_device_map(),
            scheduled_count=len(process_results),
        )

    # -- parallel execution -------------------------------------------------

    def execute_parallel(
        self,
        programs: dict[int, FixedProgram],
        inputs: Optional[dict[int, dict[int, float]]] = None,
        max_steps: int = 64,
    ) -> DistributedResult:
        """Execute programs in parallel across multiple cores.

        Each core runs independently.  Shared memory and messages provide
        inter-core communication.  After execution, each core's output
        registers (R0..R7) are written into the shared memory region at
        offset ``core_id * num_regs`` for cross-core visibility.

        Args:
            programs: ``{core_id: FixedProgram}`` mapping.
            inputs: ``{core_id: {reg_index: value}}`` initial register state.
            max_steps: Maximum execution steps per core.

        Returns:
            DistributedResult with per-core results and shared state.
        """
        inputs = inputs or {}

        # Load and execute each core
        core_results: dict[int, ExecutionResult] = {}
        for core_id, prog in programs.items():
            self._validate_core(core_id)
            core = self.cores[core_id]
            core.load_program(prog)
            core_inputs = inputs.get(core_id, {})
            result = core.engine.execute_fixed(prog, core_inputs, max_steps=max_steps)
            core_results[core_id] = result

            if result.halted:
                core.state = CoreState.HALTED

            # Publish output registers to shared memory
            if result.registers is not None:
                num_regs = result.registers.numel()
                base = core_id * num_regs
                if base + num_regs <= self.shared_memory.size:
                    self.shared_memory.write(base, result.registers.detach())

        # Deliver any queued messages
        delivered = self.route_messages()

        all_halted = all(
            self.cores[cid].state == CoreState.HALTED
            or core_results[cid].halted
            for cid in programs
        )

        return DistributedResult(
            core_results=core_results,
            shared_memory=self.shared_memory.snapshot(),
            messages_exchanged=delivered,
            total_steps=sum(r.steps_executed for r in core_results.values()),
            all_halted=all_halted,
            device_assignments=self.get_device_map(),
        )

    # -- pipeline execution -------------------------------------------------

    def execute_pipeline(
        self,
        stages: list[tuple[int, FixedProgram, dict]],
        max_steps: int = 64,
    ) -> DistributedResult:
        """Execute a pipeline across cores (stage N output feeds stage N+1).

        This is a dataflow execution model: each core processes one stage
        of a computation pipeline, passing results through shared memory.

        Args:
            stages: ``[(core_id, program, initial_inputs), ...]`` ordered
                from first to last pipeline stage.
            max_steps: Maximum execution steps per stage.

        Returns:
            DistributedResult with all stage results.
        """
        results: dict[int, ExecutionResult] = {}

        for i, (core_id, prog, initial_inputs) in enumerate(stages):
            self._validate_core(core_id)
            core = self.cores[core_id]
            core.load_program(prog)

            # After the first stage, feed the previous stage's output
            # registers as this stage's inputs.
            stage_inputs = dict(initial_inputs)  # copy so we don't mutate
            if i > 0:
                prev_core_id = stages[i - 1][0]
                prev_result = results[prev_core_id]
                num_regs = min(
                    prev_result.registers.numel(),
                    core.config.num_registers,
                )
                for reg_idx in range(num_regs):
                    val = prev_result.registers[reg_idx]
                    # Only inject if the caller didn't explicitly set it
                    if reg_idx not in stage_inputs:
                        stage_inputs[reg_idx] = val

            result = core.engine.execute_fixed(
                prog, stage_inputs, max_steps=max_steps
            )
            results[core_id] = result

            if result.halted:
                core.state = CoreState.HALTED

            # Write to shared memory
            if result.registers is not None:
                num_regs = result.registers.numel()
                base = core_id * num_regs
                if base + num_regs <= self.shared_memory.size:
                    self.shared_memory.write(base, result.registers.detach())

        return DistributedResult(
            core_results=results,
            shared_memory=self.shared_memory.snapshot(),
            messages_exchanged=0,
            total_steps=sum(r.steps_executed for r in results.values()),
            all_halted=all(r.halted for r in results.values()),
            device_assignments=self.get_device_map(),
        )

    # -- BSP (Bulk Synchronous Parallel) execution ---------------------------

    def execute_bsp(
        self,
        programs: dict[int, FixedProgram],
        inputs: dict[int, dict[int, float]],
        num_supersteps: int = 4,
        steps_per_superstep: int = 16,
    ) -> DistributedResult:
        """Execute in BSP (Bulk Synchronous Parallel) mode.

        Each superstep:
        1. All cores execute independently for *steps_per_superstep*.
        2. Barrier synchronization ensures every core finishes before
           the next phase.
        3. Core outputs are written to shared memory so that all cores
           can observe the results.
        4. The next superstep reads the updated shared memory, enabling
           iterative convergence algorithms.

        This is the standard model for parallel graph algorithms,
        iterative solvers, and MapReduce-style computations.

        Args:
            programs: ``{core_id: FixedProgram}`` mapping.
            inputs: ``{core_id: {reg_index: value}}`` initial register
                state.  Updated with shared memory values between
                supersteps.
            num_supersteps: Number of BSP supersteps to execute.
            steps_per_superstep: Maximum execution steps per core per
                superstep.

        Returns:
            DistributedResult with final per-core results and shared
            memory state after all supersteps.
        """
        core_ids = sorted(programs.keys())
        for cid in core_ids:
            self._validate_core(cid)

        barrier = Barrier(num_participants=len(core_ids))

        # Mutable copy of inputs so we can inject shared-memory values
        # between supersteps.
        current_inputs: dict[int, dict[int, float]] = {
            cid: dict(inputs.get(cid, {})) for cid in core_ids
        }

        core_results: dict[int, ExecutionResult] = {}

        for _superstep in range(num_supersteps):
            # ---- 1. Compute: each core runs independently ----------------
            for cid in core_ids:
                core = self.cores[cid]
                core.load_program(programs[cid])
                result = core.engine.execute_fixed(
                    programs[cid],
                    current_inputs[cid],
                    max_steps=steps_per_superstep,
                )
                core_results[cid] = result

                if result.halted:
                    core.state = CoreState.HALTED

            # ---- 2. Barrier: all cores synchronize -----------------------
            for cid in core_ids:
                barrier.arrive(cid)

            # ---- 3. Communicate: publish registers to shared memory ------
            for cid in core_ids:
                result = core_results[cid]
                if result.registers is not None:
                    num_regs = result.registers.numel()
                    base = cid * num_regs
                    if base + num_regs <= self.shared_memory.size:
                        self.shared_memory.write(
                            base, result.registers.detach()
                        )

            # ---- 4. Prepare next superstep: read shared memory -----------
            # Each core's inputs for the next superstep are the outputs of
            # every core's registers from shared memory.  We only inject
            # values that the caller did not explicitly set.
            for cid in core_ids:
                result = core_results[cid]
                if result.registers is not None:
                    num_regs = result.registers.numel()
                    for other_cid in core_ids:
                        if other_cid == cid:
                            continue
                        base = other_cid * num_regs
                        if base + num_regs <= self.shared_memory.size:
                            other_regs = self.shared_memory.read(
                                base, num_regs
                            )
                            for ri in range(num_regs):
                                if ri not in inputs.get(cid, {}):
                                    current_inputs[cid][ri] = (
                                        other_regs[ri].item()
                                    )

        # Deliver any queued messages
        delivered = self.route_messages()

        all_halted = all(
            self.cores[cid].state == CoreState.HALTED
            or core_results[cid].halted
            for cid in core_ids
        )

        return DistributedResult(
            core_results=core_results,
            shared_memory=self.shared_memory.snapshot(),
            messages_exchanged=delivered,
            total_steps=sum(r.steps_executed for r in core_results.values()),
            all_halted=all_halted,
            device_assignments=self.get_device_map(),
        )

    # -- utilities ----------------------------------------------------------

    def _validate_core(self, core_id: int) -> None:
        if core_id < 0 or core_id >= self.num_cores:
            raise IndexError(
                f"core_id {core_id} out of range [0, {self.num_cores})"
            )

    def reset(self) -> None:
        """Reset all cores and shared state."""
        for core in self.cores:
            core.reset()
        self.shared_memory.clear()
        self.message_bus.clear()
        self._pipes.clear()
        self._pipe_endpoints.clear()

    def __repr__(self) -> str:
        states = [c.state.value for c in self.cores]
        devices = self.get_device_map()
        return (
            f"DistributedNCPU(num_cores={self.num_cores}, "
            f"states={states}, devices={devices})"
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_distributed() -> None:
    """Demo: parallel and pipeline computation across 4 GPU cores."""
    print("=" * 60)
    print("Multi-GPU Distributed nCPU")
    print("=" * 60)

    dcpu = DistributedNCPU(num_cores=4)

    # -- Parallel execution -------------------------------------------------
    # Core 0: compute a + b
    # Core 1: compute c * d
    # Core 2: compute e - f
    prog0 = FixedProgram([
        Instruction(OPCODES["ADD"], dst=7, src1=0, src2=1),
        Instruction(OPCODES["HALT"]),
    ])
    prog1 = FixedProgram([
        Instruction(OPCODES["MUL"], dst=7, src1=0, src2=1),
        Instruction(OPCODES["HALT"]),
    ])
    prog2 = FixedProgram([
        Instruction(OPCODES["SUB"], dst=7, src1=0, src2=1),
        Instruction(OPCODES["HALT"]),
    ])

    result = dcpu.execute_parallel(
        {0: prog0, 1: prog1, 2: prog2},
        inputs={
            0: {0: 10.0, 1: 5.0},   # 10+5 = 15
            1: {0: 3.0, 1: 7.0},    # 3*7  = 21
            2: {0: 20.0, 1: 8.0},   # 20-8 = 12
        },
    )

    print(f"\nParallel execution across {dcpu.num_cores} cores:")
    for core_id, res in sorted(result.core_results.items()):
        print(f"  Core {core_id}: R7 = {res.registers[7].item():.1f}")
    print(f"Total steps: {result.total_steps}")
    print(f"All halted:  {result.all_halted}")

    # -- Pipeline execution -------------------------------------------------
    print("\n--- Pipeline Mode ---")
    dcpu.reset()

    pipe_result = dcpu.execute_pipeline([
        (0, FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=7.0),
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=6.0),
            Instruction(OPCODES["MUL"], dst=2, src1=0, src2=1),   # R2 = 42
            Instruction(OPCODES["HALT"]),
        ]), {}),
        (1, FixedProgram([
            Instruction(OPCODES["ADD"], dst=3, src1=2, src2=0),   # reads core 0 R2
            Instruction(OPCODES["HALT"]),
        ]), {}),
    ])

    print("Pipeline: Core 0 -> Core 1")
    r2 = pipe_result.core_results[0].registers[2].item()
    r3 = pipe_result.core_results[1].registers[3].item()
    print(f"  Core 0 R2 = {r2:.1f}  (7 * 6)")
    print(f"  Core 1 R3 = {r3:.1f}  (42 + 7, reads core 0's output)")

    # -- Fork demo ----------------------------------------------------------
    print("\n--- Fork ---")
    dcpu.reset()
    dcpu.load_program(0, FixedProgram([
        Instruction(OPCODES["MOV_IMM"], dst=0, immediate=99.0),
        Instruction(OPCODES["HALT"]),
    ]))
    dcpu.cores[0].local_memory[0] = 42.0
    dcpu.fork(parent_core=0, child_core=1)
    print(f"  Forked core 0 -> core 1")
    print(f"  Core 1 local_memory[0] = {dcpu.cores[1].local_memory[0].item():.1f}")
    print(f"  Core 1 state = {dcpu.cores[1].state.value}")

    # -- Pipe demo ----------------------------------------------------------
    print("\n--- Pipe ---")
    pipe_idx = dcpu.pipe(core_a=0, core_b=1, buffer_size=64)
    ch = dcpu.get_pipe(pipe_idx)
    ch.send(torch.tensor([3.14, 2.72]))
    received = ch.recv(2)
    print(f"  Pipe {pipe_idx}: sent [3.14, 2.72], received {received.tolist()}")

    print("\nDone.")


if __name__ == "__main__":
    demo_distributed()
