"""Distributed process scheduler for multi-GPU nCPU.

Assigns processes to GPU cores according to a configurable scheduling policy.
This mirrors an OS-level scheduler but operates at the granularity of entire
DifferentiableEngine programs rather than individual instructions.

Supported policies:
  - ROUND_ROBIN: rotate through cores sequentially (fairness-first).
  - LOAD_BALANCED: pick the least-loaded core (throughput-first).
  - AFFINITY: honour per-process core preferences, falling back to
    round-robin when no affinity is specified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------

class SchedulingPolicy(Enum):
    """Scheduling strategy for distributing processes across cores."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    AFFINITY = "affinity"


# ---------------------------------------------------------------------------
# Process descriptor
# ---------------------------------------------------------------------------

@dataclass
class ProcessDescriptor:
    """Describes a process to be scheduled on the distributed nCPU.

    Attributes:
        pid: Unique process identifier.
        program: The FixedProgram or SoftProgram to execute.
        inputs: Initial register values ``{reg_index: value}``.
        core_affinity: Preferred core id (optional).  Only honoured by the
            AFFINITY policy.
        priority: Higher values run first when the scheduler must break ties.
    """

    pid: int
    program: object  # FixedProgram | SoftProgram
    inputs: dict = field(default_factory=dict)
    core_affinity: Optional[int] = None
    priority: int = 0


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class DistributedScheduler:
    """Schedule processes across GPU cores.

    Maintains an internal queue of submitted processes and, on each call to
    ``schedule()``, assigns every queued process to a core according to the
    active policy.  The returned mapping ``{core_id: [processes]}`` can be
    fed directly to ``DistributedNCPU.execute_parallel``.
    """

    def __init__(
        self,
        num_cores: int,
        policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN,
    ) -> None:
        if num_cores < 1:
            raise ValueError("num_cores must be >= 1")
        self.num_cores = num_cores
        self.policy = policy
        self.core_loads: list[int] = [0] * num_cores
        self._next_core: int = 0
        self.process_queue: list[ProcessDescriptor] = []

    # -- submission ---------------------------------------------------------

    def submit(self, process: ProcessDescriptor) -> None:
        """Add a process to the scheduling queue."""
        self.process_queue.append(process)

    def submit_batch(self, processes: list[ProcessDescriptor]) -> None:
        """Add multiple processes to the scheduling queue."""
        self.process_queue.extend(processes)

    # -- scheduling ---------------------------------------------------------

    def schedule(self) -> dict[int, list[ProcessDescriptor]]:
        """Assign all queued processes to cores.

        Returns ``{core_id: [ProcessDescriptor, ...]}``.  The queue is
        cleared after scheduling.
        """
        assignments: dict[int, list[ProcessDescriptor]] = {
            i: [] for i in range(self.num_cores)
        }

        # Sort by descending priority so high-priority jobs are placed first
        # (matters for LOAD_BALANCED where order affects which core is
        # least-loaded at the time of assignment).
        sorted_queue = sorted(
            self.process_queue, key=lambda p: -p.priority
        )

        for proc in sorted_queue:
            core = self._pick_core(proc)
            assignments[core].append(proc)
            self.core_loads[core] += 1

        self.process_queue.clear()
        return assignments

    def _pick_core(self, proc: ProcessDescriptor) -> int:
        """Select a core for *proc* based on the active policy."""
        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            core = self._next_core
            self._next_core = (self._next_core + 1) % self.num_cores
            return core

        if self.policy == SchedulingPolicy.LOAD_BALANCED:
            return min(range(self.num_cores), key=lambda i: self.core_loads[i])

        if self.policy == SchedulingPolicy.AFFINITY:
            if (
                proc.core_affinity is not None
                and 0 <= proc.core_affinity < self.num_cores
            ):
                return proc.core_affinity
            # Fallback to round-robin when no affinity specified
            core = self._next_core
            self._next_core = (self._next_core + 1) % self.num_cores
            return core

        # Unreachable unless someone adds a new policy without a handler
        raise ValueError(f"Unknown scheduling policy: {self.policy}")

    # -- completion ---------------------------------------------------------

    def complete(self, core_id: int) -> None:
        """Notify the scheduler that a process on *core_id* has finished.

        Decrements the load counter so that LOAD_BALANCED scheduling can
        correctly identify under-utilised cores.
        """
        if core_id < 0 or core_id >= self.num_cores:
            raise IndexError(
                f"core_id {core_id} out of range [0, {self.num_cores})"
            )
        if self.core_loads[core_id] > 0:
            self.core_loads[core_id] -= 1

    # -- introspection / reset ----------------------------------------------

    def pending_count(self) -> int:
        """Number of processes waiting to be scheduled."""
        return len(self.process_queue)

    def reset(self) -> None:
        """Reset scheduler state (loads, pointer, queue)."""
        self.core_loads = [0] * self.num_cores
        self._next_core = 0
        self.process_queue.clear()

    def __repr__(self) -> str:
        return (
            f"DistributedScheduler(num_cores={self.num_cores}, "
            f"policy={self.policy.value}, "
            f"queued={len(self.process_queue)}, "
            f"loads={self.core_loads})"
        )
