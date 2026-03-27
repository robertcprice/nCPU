"""Multi-GPU distributed nCPU.

Extends nCPU from a single GPU to multiple GPUs acting as multiple cores
of a single neural computer. Each GPU is one core with its own registers,
PC, and local memory. Inter-GPU communication uses shared memory regions
and message passing, analogous to a multi-core CPU's cache coherence protocol.

Architecture:
  - Each GPU core runs an independent DifferentiableEngine
  - Shared memory regions enable inter-core communication
  - A distributed scheduler manages core allocation
  - Fork/pipe/wait extend across GPU boundaries
"""

from .multi_gpu import (
    DistributedNCPU,
    GPUCore,
    CoreConfig,
    CoreState,
    DistributedResult,
    DeviceInfo,
    DeviceAssignment,
)
from .ipc import (
    Barrier,
    SharedMemoryRegion,
    MessageChannel,
    IPCMessage,
)
from .scheduler import (
    DistributedScheduler,
    ProcessDescriptor,
    SchedulingPolicy,
)

__all__ = [
    "Barrier",
    "DistributedNCPU",
    "GPUCore",
    "CoreConfig",
    "CoreState",
    "DistributedResult",
    "DeviceInfo",
    "DeviceAssignment",
    "SharedMemoryRegion",
    "MessageChannel",
    "IPCMessage",
    "DistributedScheduler",
    "ProcessDescriptor",
    "SchedulingPolicy",
]
