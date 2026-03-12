"""GPU-Native Protocols Package"""

from ncpu.os.gpu.protocols.tensor_stream import (
    TensorDtype,
    TensorOp,
    TensorDescriptor,
    TensorStreamProtocol,
    create_descriptor,
    benchmarkTSP,
)

from ncpu.os.gpu.protocols.batch_rpc import (
    RpcOpcode,
    RpcArg,
    RpcRequest,
    RpcResult,
    RpcBatch,
    BatchRpcProtocol,
    benchmark_batch_rpc,
)

from ncpu.os.gpu.protocols.gradient_aware_network import (
    GradientType,
    CompressionType,
    GradientDescriptor,
    PipelineStage,
    GradientCompressor,
    GradientAwareNetworkProtocol,
    benchmark_gradient_compression,
)

from ncpu.os.gpu.protocols.persistent_workers import (
    WorkPriority,
    WorkItem,
    WorkResult,
    PersistentWorker,
    WorkerPool,
    PersistentGpuWorkersProtocol,
    benchmark_persistent_workers,
)

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

__all__ = [
    # Tensor Streaming Protocol
    "TensorDtype",
    "TensorOp",
    "TensorDescriptor",
    "TensorStreamProtocol",
    "create_descriptor",
    "benchmarkTSP",
    # Batch RPC
    "RpcOpcode",
    "RpcArg",
    "RpcRequest",
    "RpcResult",
    "RpcBatch",
    "BatchRpcProtocol",
    "benchmark_batch_rpc",
    # Gradient-Aware Network
    "GradientType",
    "CompressionType",
    "GradientDescriptor",
    "PipelineStage",
    "GradientCompressor",
    "GradientAwareNetworkProtocol",
    "benchmark_gradient_compression",
    # Persistent Workers
    "WorkPriority",
    "WorkItem",
    "WorkResult",
    "PersistentWorker",
    "WorkerPool",
    "PersistentGpuWorkersProtocol",
    "benchmark_persistent_workers",
    # Shared Virtual Memory
    "SvmFlags",
    "TransferDirection",
    "SvmRegion",
    "SvmPointer",
    "DmaTransfer",
    "SvmAllocator",
    "DmaEngine",
    "SharedVirtualMemoryProtocol",
    "benchmark_svm_transfer",
    # Compiler-Guided
    "StateType",
    "RegisterState",
    "MemoryRegion",
    "StateCapture",
    "StateDiff",
    "MigrationUnit",
    "StateReplayer",
    "CompilerGuidedProtocol",
    "benchmark_state_capture",
]
