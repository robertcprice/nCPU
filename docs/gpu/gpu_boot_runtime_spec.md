# GPU Boot Runtime Spec

Status: Draft v0.1

## Intent

This spec defines a GPU-first boot and runtime model for nCPU.

The target is not "Python mediates Linux syscalls faster." The target is:

- one Rust host launch
- a persistent GPU runtime
- no Python in the execution path
- closed-world execution with no host servicing in the default mode
- a Linux-compatible persona for existing BusyBox and Alpine-style userland
- a native persona that stops over-imitating Linux and maps work into GPU-friendly primitives

This spec is grounded in the current repo state:

- `kernels/rust_metal/src/full_arm64.rs` provides the primary Rust/Metal ARM64 executor
- `ncpu/os/gpu/elf_loader.py` currently owns ELF loading and Linux stack setup
- `ncpu/os/gpu/filesystem.py` currently owns the in-memory filesystem and pipe model
- `ncpu/os/gpu/alpine.py` currently builds the Alpine-style rootfs snapshot in Python
- `kernels/rust_metal/src/pure_gpu.rs` already points toward persistent one-launch GPU execution

## Goals

1. Boot directly into GPU execution after one Rust launch step.
2. Run a useful Linux-ish userland without per-syscall CPU mediation.
3. Keep the system deterministic and inspectable.
4. Add a native ABI that is more efficient than Linux compatibility mode.
5. Treat host interaction as optional device bridges, not the core runtime.

## Non-Goals

1. Reproducing all Linux kernel internals exactly.
2. Claiming that a Metal shader can directly perform host file, socket, or clock operations without a host bridge.
3. Locking the future architecture to ARM64 emulation forever.
4. Preserving Python as part of the runtime path.

## Operating Modes

### Mode A: Closed-World GPU Boot

This is the primary target.

- Rust allocates unified memory, loads a boot image, submits the persistent kernel, and stops participating.
- The GPU mounts a read-only rootfs snapshot plus a copy-on-write overlay.
- No host bridge queues are serviced after launch.
- Console output is written to GPU-resident logs.
- Time, randomness, and networking are synthetic unless pre-seeded in the image.

### Mode B: Bridged GPU Runtime

This is the compatibility and integration target.

- Same GPU runtime as Mode A.
- Optional bridge queues are enabled for host console, clock, random, filesystem sync, and networking.
- Rust services bridge queues when enabled.
- This is still Rust-only at the host layer and never uses Python.

### Mode C: Native Persona

This is the long-term performance target.

- Programs do not need to pretend to be Linux processes.
- Tasks can use native hypercalls and GPU-oriented objects directly.
- Hot paths can bypass ARM64 decode entirely.

## Design Principles

1. Determinism first.
2. Persistent kernel, not launch-per-syscall.
3. Kernel objects in memory, not host callbacks.
4. Linux compatibility is a persona, not the core ontology.
5. Host interaction is explicit and capability-gated.
6. The fastest path is native and shader-oriented.

## Target Architecture

```text
Rust Host Launcher
    |
    | one-time setup only
    v
Unified Memory Boot Image
    |
    v
Persistent Metal Kernel
    |
    +-- Stage0 GPU boot ROM
    +-- Stage1 microkernel runtime
    +-- Linux persona executor (ARM64 compatibility)
    +-- Native persona executor (hypercall ABI)
    +-- VFS / pipes / event queues / task scheduler
    +-- Optional host bridge queues
```

## Boot Model

### Host Responsibilities

The Rust host launcher is allowed to:

- allocate unified memory
- place the boot image in memory
- compile or load the persistent Metal pipeline
- submit the initial command buffer
- optionally service bridge queues if the chosen mode enables them

The Rust host launcher is not allowed to:

- emulate guest syscalls as the normal execution model
- own the guest filesystem
- own the guest process model
- own ELF stack setup after the boot image is built

### GPU Responsibilities

The GPU runtime owns:

- scheduler
- task tables
- address-space metadata
- VFS tables
- fd tables
- pipes and message rings
- timers and event queues
- Linux persona syscall translation
- native hypercall dispatch
- debug and trace buffers

## Boot Image Format

The boot image is a single binary blob placed in unified memory before launch.

### Header

The image begins with a fixed-size header:

```c
struct BootImageHeader {
    uint32_t magic;              // "NCBT"
    uint16_t version;            // format version
    uint16_t header_size;        // bytes
    uint64_t flags;              // image-wide flags
    uint64_t image_size;         // full byte size
    uint64_t checksum64;         // whole-image checksum
    uint32_t memory_profile;     // 16M, 64M, 256M, ...
    uint32_t region_count;       // number of region descriptors
    uint32_t task_count;         // number of boot task descriptors
    uint32_t root_task_index;    // initial task
    uint64_t region_table_offset;
    uint64_t task_table_offset;
    uint64_t symbol_table_offset;
    uint64_t rootfs_offset;
    uint64_t rootfs_size;
    uint64_t boot_args_offset;
    uint64_t boot_args_size;
    uint64_t reserved[3];
};
```

### Region Descriptors

Each image region is described by:

```c
struct BootRegionDesc {
    uint32_t kind;         // kernel, rootfs, elf_text, elf_data, native_module, config
    uint32_t flags;        // R/W/X, compressed, copy_on_write, preload, debug_visible
    uint64_t guest_base;   // target GPU address
    uint64_t mem_size;     // in-memory size
    uint64_t file_offset;  // offset inside boot image
    uint64_t file_size;    // stored size
    uint64_t align;        // placement alignment
    uint64_t checksum64;   // region checksum
    uint64_t reserved;
};
```

### Task Descriptors

Boot tasks are declared explicitly:

```c
struct BootTaskDesc {
    uint32_t persona;          // stage0, linux, native, service
    uint32_t flags;            // privileged, bridged, trace_on_boot, etc.
    uint64_t entry_pc;         // ARM64 PC or native module entry id
    uint64_t stack_top;        // initial stack top
    uint64_t arg_ptr;          // pointer to argv/boot args
    uint64_t env_ptr;          // pointer to env block
    uint32_t addr_space_id;    // address-space seed
    uint32_t fd_table_id;      // fd table seed
    uint32_t cwd_node_id;      // root cwd node
    uint32_t capability_mask;  // allowed bridge and debug capabilities
    uint64_t reserved;
};
```

### Boot Image Region Kinds

At minimum the image supports:

- kernel code
- kernel data
- rootfs snapshot
- ARM64 ELF text/data segments
- native module blobs
- debug manifest
- symbol table
- boot arguments and environment

## Standard Memory Map

The standard profile uses 256 MiB of unified memory.

```text
0x0000_0000 - 0x0000_FFFF   Guard / reserved
0x0001_0000 - 0x0001_FFFF   Stage0 GPU boot ROM
0x0002_0000 - 0x0002_FFFF   Boot image header + descriptor tables
0x0003_0000 - 0x0003_FFFF   Optional bridge queue descriptors
0x0004_0000 - 0x0007_FFFF   Kernel code scratch / bootstrap work area
0x0008_0000 - 0x000F_FFFF   Kernel superblock + object arena headers
0x0010_0000 - 0x003F_FFFF   Kernel heap / allocators / scheduler state
0x0040_0000 - 0x013F_FFFF   Read-only rootfs snapshot
0x0140_0000 - 0x017F_FFFF   Copy-on-write overlay data + metadata
0x0180_0000 - 0x01BF_FFFF   Pipes, message rings, event queues
0x01C0_0000 - 0x01FF_FFFF   Task table, address spaces, fd tables
0x0200_0000 - 0x02FF_FFFF   ARM64 basic-block cache and micro-op cache
0x0300_0000 - 0x03FF_FFFF   Initial ELF/native module image space
0x0400_0000 - 0x0BFF_FFFF   User heaps, stacks, anon mappings, page arenas
0x0C00_0000 - 0x0C1F_FFFF   Debug control block + trace buffers
0x0C20_0000 - 0x0C3F_FFFF   Snapshot, replay, profiler buffers
0x0C40_0000 - 0x0FFF_FFFF   Future expansion and device windows
```

### Reduced Bring-Up Profile

A 16 MiB or 64 MiB profile may be used during bring-up, but it must preserve the same ordering:

1. stage0 and descriptor area
2. kernel object arenas
3. rootfs snapshot
4. task and cache regions
5. debug and trace regions

## Boot Sequence

### Stage H0: Rust Host Launch

1. Build or load a boot image.
2. Allocate unified memory using the selected profile.
3. Copy the boot image into the descriptor and region area.
4. Seed bridge queue descriptors if bridged mode is enabled.
5. Launch the persistent Metal kernel once.

### Stage G0: GPU Boot ROM

1. Validate boot image magic, version, and checksum.
2. Install the standard memory map.
3. Create the kernel superblock.
4. Materialize region descriptors.
5. Create the root address space and first task.
6. Mount the rootfs snapshot.
7. Transfer control to stage1.

### Stage G1: GPU Microkernel Bring-Up

1. Initialize scheduler state.
2. Initialize task, address-space, fd, and VFS arenas.
3. Initialize debug buffers.
4. Initialize event queues and pipe rings.
5. Enable optional bridge queues if capabilities permit.
6. Launch the root task.

### Stage G2: Init Task

The root task may be one of:

- a Linux persona task running a static ARM64 `init` or BusyBox shell
- a native persona task that mounts services and launches further work

## Kernel Object Model

All kernel state lives in preallocated arenas. Objects use integer ids, not host pointers.

### Kernel Superblock

Global runtime state:

- memory profile
- arena bases and capacities
- current epoch / generation
- scheduler state
- bridge capability mask
- root task id
- rootfs root node id
- debug mode flags

### Task Control Block

Each task has:

- task id
- persona kind
- run state
- priority and cycle budget
- address-space id
- fd table id
- cwd node id
- wait object id
- event queue id
- capability mask
- last stop reason
- accounting counters

### ARM64 CPU State

For Linux persona tasks:

- X0-X30 and SP
- PC
- NZCV
- SIMD state only if the compatibility tier needs it
- cached basic-block pointer or block id

### Address Space

Address spaces are metadata, not full host VM objects:

- region descriptors
- page ownership / arena mapping
- copy-on-write generation ids
- stack and heap bounds
- executable region ids

### VFS Node

Each node stores:

- node id
- type: file, dir, symlink, device, pipe
- name hash
- parent id
- content extent or child table id
- stat metadata
- flags: read-only, synthetic, bridged

### FD Table

Each process-visible fd table entry stores:

- fd number
- node or pipe id
- cursor / offset
- flags
- rights mask

### Pipe Ring

Pipes and IPC queues are GPU-native ring buffers:

- capacity
- read index
- write index
- waiter masks or waiter ids
- closed-read and closed-write flags

### Event Queue

Used for:

- timer wakeups
- bridge completions
- pipe readability / writability
- task exit / wait notifications
- debug triggers

### Bridge Queue

Bridge queues are only used when the mode enables them:

- queue header
- request ring
- completion ring
- bridge type
- capability id

## Scheduler and Execution Engine

### Baseline Scheduler

The first scheduler is deterministic and conservative:

- single logical execution lane
- round-robin or fixed-priority task selection
- cycle-budget timeslicing
- explicit wait queues for blocked tasks

This preserves the current deterministic debugging advantages.

### Personas

#### Linux Persona

- Executes ARM64 guest code.
- Uses Linux syscall numbers and Linux userspace calling conventions.
- Syscalls translate into kernel object operations.

#### Native Persona

- Executes either ARM64 tasks using reserved hypercalls or future native modules.
- Talks directly to the microkernel ABI.
- Avoids the Linux syscall translation layer.

### Performance Path

Linux compatibility is not the end state. The performance path is:

1. ARM64 fetch/decode/execute for compatibility
2. basic-block cache
3. predecoded micro-op cache
4. hot loop promotion into specialized shader paths
5. native tasks that bypass ARM64 decode entirely

## Linux Persona Boundary

The Linux persona is a compatibility layer, not a promise to implement every kernel feature exactly.

### Tier 0: Closed-World BusyBox / Alpine Bring-Up

Required first:

- `openat`, `close`, `read`, `write`, `lseek`
- `fstat`, `newfstatat`, `getdents64`, `readlinkat`, `faccessat`
- `brk`, `mmap`, `munmap`, `mprotect`
- `uname`, `clock_gettime`, `clock_getres`, `getrandom`
- `dup3`, `pipe2`, `chdir`, `getcwd`
- `exit`, `exit_group`
- musl init calls such as `set_tid_address`, `set_robust_list`, `rt_sigaction`, `rt_sigprocmask`

### Tier 1: Process and Shell Expansion

- `clone` or a controlled `fork` approximation
- `wait4`
- `execve`
- signal delivery subset
- futex wait/wake mapped to GPU wait queues

### Tier 2: Optional Bridge-Backed Linux Surface

- sockets
- host-backed block devices
- real wall clock
- true host randomness
- external filesystem sync

### Linux Compatibility Rule

The compatibility rule is:

"Implement the guest-visible behavior needed by supported userland, then map it to GPU-native objects."

This is better than blindly copying Linux internals.

## Native Hypercall ABI

The native ABI is the long-term efficient interface.

### Trap Encoding

- `SVC #0x0000` is reserved for Linux persona syscalls.
- `SVC #0x4E43` is reserved for native nCPU hypercalls.

### Register Convention

- `x16` = service group
- `x8`  = operation id
- `x0`..`x5` = arguments
- `x6` = flags
- `x7` = capability token
- return values in `x0` and `x1`
- negative `x0` values encode errors

### Service Groups

```text
0  TASK       spawn, exit, wait, yield, query
1  MEMORY     alloc, free, map, protect, share
2  VFS        open node, read, write, stat, enumerate
3  IPC        pipe, queue, send, recv, subscribe
4  TIMER      sleep, poll, deadline, monotonic
5  CONSOLE    log, read ring, flush
6  BRIDGE     host device request, completion query
7  DEBUG      trace, snapshot, watch, profile
8  ACCEL      native compute service entry points
```

### Native ABI Rule

Every feature added for Linux compatibility should be considered for a native hypercall that exposes the same capability more directly.

## Optional Host Bridges

Bridges are devices, not the runtime core.

### Required Properties

- explicit enable bit in the boot image and task capability mask
- queue-based request/completion protocol
- no ad hoc syscall fallback path
- deterministic stubs when disabled

### Bridge Types

1. Console bridge
2. Clock bridge
3. Random bridge
4. Network bridge
5. Filesystem sync bridge
6. Debugger export bridge

## Performance-First Decisions

The runtime should stop mirroring CPU and Linux habits when a GPU-native equivalent is better.

### Required Optimizations

1. Read-only rootfs + copy-on-write overlay
2. Arena allocators over scattered heap metadata
3. GPU-resident pipe and event rings
4. Basic-block cache keyed by `(pc, page_generation)`
5. Predecoded micro-op cache for hot regions
6. Hot loop promotion into specialized kernels where safe
7. Native service tasks for common dataflow-heavy operations

### Future Native Module Direction

After Linux persona bring-up works, nCPU should support native modules that:

- package GPU-oriented kernels or bytecode
- use the native hypercall ABI directly
- bypass Linux ABI friction
- expose capabilities that have no clean Linux equivalent

## Debugging and Determinism

The current GPU debugging advantage should be preserved.

Required:

- persistent trace buffer
- task and event snapshots
- deterministic replay in closed-world mode
- bridge event logging in bridged mode
- capability-gated inspection APIs

## Implementation Phases

### Phase 0: Spec and Format

- write the architecture spec
- define boot image structs and constants in Rust
- define the standard memory map and object ids

### Phase 1: Rust-Only Bring-Up

- replace Python ELF loading with Rust boot-image building
- replace Python runtime loops with a Rust launcher
- keep current ARM64 executor while removing Python from the path

### Phase 2: GPU Microkernel Core

- implement stage0 boot ROM
- implement kernel superblock and arenas
- implement scheduler, tasks, events, pipes, and VFS

### Phase 3: Linux Persona Bring-Up

- translate the current BusyBox/Alpine syscall subset into GPU kernel objects
- boot BusyBox or a shell as the root task
- support closed-world Alpine-style userland

### Phase 4: Native Persona

- implement the native hypercall ABI
- add one native demo program that does not use Linux syscalls
- measure the gap against Linux persona execution

### Phase 5: Optimization

- basic-block cache
- micro-op cache
- hot loop promotion
- bridge devices

## Success Criteria

The architecture is succeeding when all of the following are true:

1. A Rust launcher can boot a GPU image with no Python involvement.
2. The GPU can enter a shell or init task after a single host launch.
3. Closed-world mode runs without per-syscall CPU servicing.
4. BusyBox / Alpine-style userland works through the Linux persona.
5. At least one native task outperforms the Linux persona because it bypasses the compatibility layer.
6. The debug and trace model remains deterministic and first-class.
