# GPU-Native Protocols

The nCPU project implements several innovative GPU-native protocols that go beyond traditional CPU protocols. These protocols leverage GPU-specific capabilities like tensor operations, persistent workers, and direct memory access.

## Overview

| Protocol | Purpose | Key Innovation |
|----------|---------|----------------|
| Tensor Streaming Protocol (TSP) | Zero-copy tensor operations | Descriptors only, no data movement |
| Batch RPC Protocol | Bundle multiple RPC calls | Single fused kernel launch |
| Gradient-Aware Network Protocol | Distributed training pipelines | Gradient compression & pipelining |
| Persistent GPU Workers | Zero kernel launch overhead | Resident worker threads |
| Shared Virtual Memory (SVM) | GPU-to-GPU DMA | Zero-copy inter-GPU communication |
| Compiler-Guided Protocol | State capture & replay | ELF-based state migration |

---

## 1. Tensor Streaming Protocol (TSP)

### Concept

TSP sends **tensor descriptors** (metadata about tensors) instead of serializing actual data. The GPU executes operations directly from device memory using the descriptors.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    TensorDtype, TensorOp, TensorDescriptor, TensorStreamProtocol
)

# Create descriptor (no data movement!)
desc = TensorDescriptor(
    device_id=0,
    ptr=0x1000,  # GPU pointer
    shape=(1024, 512),
    dtype=TensorDtype.FLOAT32
)

# Use protocol
tsp = TensorStreamProtocol()
tsp.register_tensor("input", desc)
tsp.queue_op(TensorOp.MATMUL, ["input", "weights"], ["output"])
tsp.execute()  # Single fused kernel
```

### Supported Operations

- **Compute**: ADD, MUL, MATMUL, CONV2D, REDUCE, SCAN, SORT
- **Activation**: RELU, GELU, SILU, SOFTMAX
- **Tensor**: SLICE, CONCAT, RESHAPE, PAD
- **Control**: BARRIER, CONDITIONAL, SPAWN

---

## 2. Batch RPC Protocol

### Concept

Bundle multiple RPC calls into a single GPU kernel launch, eliminating per-call overhead. All requests execute as a fused operation.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    RpcOpcode, RpcBatch, BatchRpcProtocol
)

# Create protocol
protocol = BatchRpcProtocol(device_id=0)

# Queue multiple operations (all execute as ONE kernel)
protocol.queue_matmul("A", "B", "C")
protocol.queue_relu("C")
protocol.queue_softmax("C", "output")

# Execute as single fused kernel
results = protocol.execute()
```

### Benefits

- **1 kernel launch** for N operations
- **Fused execution** - no intermediate memory
- **Batch optimization** - GPU can fuse operations

---

## 3. Gradient-Aware Network Protocol

### Concept

Pipelines forward/backward passes with gradient compression for efficient distributed training. Minimizes communication overhead through compression.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    GradientType, CompressionType, GradientAwareNetworkProtocol
)

# Create protocol with compression
ganp = GradientAwareNetworkProtocol(
    compression_type=CompressionType.TOP_K
)

# Execute forward pass
outputs = ganp.forward(inputs)

# Backward pass returns descriptors (not full tensors!)
grad_descriptors = ganp.backward(loss)

# Compress for communication
compressed = ganp.compress_gradients(grad_descriptors)
```

### Compression Strategies

- **QUANTIZATION**: Lower precision (e.g., 8-bit)
- **TOP_K**: Keep only k largest magnitude values
- **DWAL**: Deep Gradient Compression with local clipping
- **POWER_SGD**: Low-rank decomposition

---

## 4. Persistent GPU Workers Protocol

### Concept

Maintains persistent GPU worker threads that remain resident, processing work items without kernel launch overhead.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    WorkPriority, WorkItem, PersistentGpuWorkersProtocol
)

# Create and start worker pool
pgwp = PersistentGpuWorkersProtocol(num_workers=4)
pgwp.initialize()

# Submit work (zero launch overhead!)
for i in range(100):
    pgwp.submit_work(
        operation="matmul",
        inputs={"A": A, "B": B},
        outputs={"C": C},
    )

# Workers process in background
# ...
pgwp.shutdown()
```

### Benefits

- **Zero kernel launch overhead** after initial pool creation
- **Background processing** - async work submission
- **Load balancing** across workers
- **Priority queue** support

---

## 5. Shared Virtual Memory Protocol (SVM)

### Concept

Enables direct GPU-to-GPU memory access via Shared Virtual Memory, eliminating CPU-based memory copies.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    SvmFlags, SvmRegion, SharedVirtualMemoryProtocol
)

# Create protocol
svmp = SharedVirtualMemoryProtocol(devices=[0, 1])

# Allocate shared memory
weights = np.random.randn(1024, 1024).astype(np.float32)
region = svmp.allocate_for_tensor("weights", weights, devices=[0, 1])

# Direct GPU-to-GPU transfer
svmp.transfer_tensor("weights", src_device=0, dst_device=1)
```

### Benefits

- **Zero-copy** GPU-to-GPU communication
- **DMA-based** transfers
- **Hardware support** for memory coherence
- **Unified address space** across GPUs

---

## 6. Compiler-Guided Protocol

### Concept

Compiler embeds metadata in ELF for state capture, deterministic replay, reverse execution, and migration.

### Key Classes

```python
from ncpu.os.gpu.protocols import (
    RegisterState, StateCapture, MigrationUnit,
    CompilerGuidedProtocol
)

# Create protocol with ELF
cgp = CompilerGuidedProtocol("/path/to/elf")

# Capture state at checkpoint
id1 = cgp.capture("checkpoint_1")

# ... make changes ...

# Capture another state
id2 = cgp.capture("checkpoint_2")

# Replay to earlier state
cgp.replay_to(id1)

# Create migration unit for checkpointing
unit = cgp.create_migration_unit()
```

### Features

- **State capture**: Full register + memory snapshots
- **State diff**: Only changed portions
- **Replay**: Deterministic execution replay
- **Migration**: Portable checkpoint bundles

---

## Running Tests

```bash
# Run all protocol tests
python -m pytest tests/test_tensor_stream.py \
    tests/test_batch_rpc.py \
    tests/test_gradient_aware_network.py \
    tests/test_persistent_workers.py \
    tests/test_shared_virtual_memory.py \
    tests/test_compiler_guided.py -v
```

## Benchmarking

Each protocol includes a benchmark function:

```python
from ncpu.os.gpu.protocols import (
    benchmarkTSP,
    benchmark_batch_rpc,
    benchmark_gradient_compression,
    benchmark_persistent_workers,
    benchmark_svm_transfer,
    benchmark_state_capture,
)

results = benchmarkTSP(num_ops=1000, batch_size=32)
print(f"Speedup: {results['speedup']:.1f}x")
```
