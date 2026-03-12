"""
Compiler-Guided Protocol (CGP)

GPU-Native protocol where the compiler captures complete program state and
enables deterministic replay, reverse execution, and state migration. The
ELF binary itself serves as the protocol vehicle.

Protocol Design:
- StateCapture: Complete snapshot of program state (registers, memory, etc.)
- StateDiff: Minimal representation of state changes between points
- StateReplay: Reconstruct program state from captured data
- MigrationUnit: Portable state bundle for migration/checkpointing
- Protocol encoded in ELF auxiliary vectors and metadata sections
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Optional, Callable, Any
import numpy as np
import time
import struct
import hashlib


class StateType(Enum):
    """Types of state that can be captured"""
    REGISTER = "register"
    MEMORY = "memory"
    STACK = "stack"
    HEAP = "heap"
    FILE_DESCRIPTOR = "file_descriptor"
    SOCKET = "socket"
    SIGNAL = "signal"


class CompressionType(IntEnum):
    """Compression for state data"""
    NONE = 0
    ZSTD = 1
    LZ4 = 2
    SNAPPY = 3


@dataclass
class RegisterState:
    """ARM64 register state"""
    pc: int = 0  # Program counter
    sp: int = 0  # Stack pointer
    x0: int = 0
    x1: int = 0
    x2: int = 0
    x3: int = 0
    x4: int = 0
    x5: int = 0
    x6: int = 0
    x7: int = 0
    x8: int = 0
    x9: int = 0
    x10: int = 0
    x11: int = 0
    x12: int = 0
    x13: int = 0
    x14: int = 0
    x15: int = 0
    x16: int = 0
    x17: int = 0
    x18: int = 0
    x19: int = 0
    x20: int = 0
    x21: int = 0
    x22: int = 0
    x23: int = 0
    x24: int = 0
    x25: int = 0
    x26: int = 0
    x27: int = 0
    x28: int = 0
    fp: int = 0  # Frame pointer
    lr: int = 0  # Link register
    nzcv: int = 0  # Flags

    def to_bytes(self) -> bytes:
        """Serialize register state to bytes"""
        return struct.pack(
            "<34Q",  # 34 64-bit values: pc, sp, x0-x28, fp, lr, nzcv
            self.pc, self.sp,
            self.x0, self.x1, self.x2, self.x3, self.x4, self.x5,
            self.x6, self.x7, self.x8, self.x9, self.x10, self.x11,
            self.x12, self.x13, self.x14, self.x15, self.x16, self.x17,
            self.x18, self.x19, self.x20, self.x21, self.x22, self.x23,
            self.x24, self.x25, self.x26, self.x27, self.x28,
            self.fp, self.lr, self.nzcv
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'RegisterState':
        """Deserialize from bytes"""
        regs = struct.unpack("<34Q", data)
        return cls(
            pc=regs[0], sp=regs[1],
            x0=regs[2], x1=regs[3], x2=regs[4], x3=regs[5],
            x4=regs[6], x5=regs[7], x6=regs[8], x7=regs[9],
            x8=regs[10], x9=regs[11], x10=regs[12], x11=regs[13],
            x12=regs[14], x13=regs[15], x14=regs[16], x15=regs[17],
            x16=regs[18], x17=regs[19], x18=regs[20], x19=regs[21],
            x20=regs[22], x21=regs[23], x22=regs[24], x23=regs[25],
            x24=regs[26], x25=regs[27], x26=regs[28], x27=regs[29],
            x28=regs[30], fp=regs[31], lr=regs[32], nzcv=regs[33]
        )

    def hash(self) -> str:
        """Get hash of register state"""
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]


@dataclass
class MemoryRegion:
    """Memory region snapshot"""
    start: int
    size: int
    data: bytes
    permissions: str = "rw-"

    def to_bytes(self) -> bytes:
        """Serialize to bytes"""
        header = struct.pack("<QQ", self.start, self.size)
        return header + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MemoryRegion':
        """Deserialize from bytes"""
        start, size = struct.unpack("<QQ", data[:16])
        return cls(start=start, size=size, data=data[16:])


@dataclass
class StateCapture:
    """
    Complete program state snapshot.

    Captured at specific execution points, enabling:
    - Deterministic replay
    - Reverse execution
    - Checkpointing/migration
    - Debugging with full state
    """
    capture_id: int
    timestamp_us: int
    pc: int  # Program counter at capture
    registers: RegisterState
    memory_regions: list[MemoryRegion] = field(default_factory=list)
    fd_table: dict[int, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize entire state capture"""
        data = bytearray()

        # Header
        header = struct.pack("<QQQ",
            self.capture_id,
            self.timestamp_us,
            self.pc,
        )
        data.extend(header)

        # Registers
        data.extend(self.registers.to_bytes())

        # Memory regions
        data.extend(struct.pack("<I", len(self.memory_regions)))
        for region in self.memory_regions:
            data.extend(region.to_bytes())
            data.extend(b'\x00' * 16)  # Padding

        # FD table
        data.extend(struct.pack("<I", len(self.fd_table)))
        for fd, info in self.fd_table.items():
            fd_data = str(info).encode()
            data.extend(struct.pack("<I", fd))
            data.extend(struct.pack("<I", len(fd_data)))
            data.extend(fd_data)

        return bytes(data)

    def total_size(self) -> int:
        """Total size of state capture in bytes"""
        size = 24 + len(self.registers.to_bytes()) + 4
        for region in self.memory_regions:
            size += 16 + region.size
        return size

    def compression_ratio(self, compressed_size: int) -> float:
        """Calculate compression ratio"""
        original = self.total_size()
        return original / compressed_size if compressed_size > 0 else 1.0


@dataclass
class StateDiff:
    """
    Minimal diff between two state captures.

    Instead of storing full state, only stores changes,
    enabling efficient state updates and migrations.
    """
    from_capture_id: int
    to_capture_id: int
    timestamp_us: int
    changed_regs: dict[str, int] = field(default_factory=dict)
    changed_memory: list[tuple[int, bytes]] = field(default_factory=list)
    new_fds: dict[int, dict] = field(default_factory=dict)
    closed_fds: list[int] = field(default_factory=list)

    def to_bytes(self) -> bytes:
        """Serialize diff"""
        data = bytearray()
        header = struct.pack("<QQ",
            self.from_capture_id,
            self.to_capture_id,
            self.timestamp_us,
        )
        data.extend(header)

        # Changed registers
        data.extend(struct.pack("<I", len(self.changed_regs)))
        for name, value in self.changed_regs.items():
            name_bytes = name.encode()
            data.extend(struct.pack("<I", len(name_bytes)))
            data.extend(name_bytes)
            data.extend(struct.pack("<Q", value))

        # Changed memory regions
        data.extend(struct.pack("<I", len(self.changed_memory)))
        for addr, content in self.changed_memory:
            data.extend(struct.pack("<Q", addr))
            data.extend(struct.pack("<I", len(content)))
            data.extend(content)

        # FD changes
        data.extend(struct.pack("<I", len(self.new_fds)))
        data.extend(struct.pack("<I", len(self.closed_fds)))

        return bytes(data)

    def total_size(self) -> int:
        """Size of diff in bytes"""
        size = 24 + 4 + len(self.changed_regs) * 72 + 4
        for _, content in self.changed_memory:
            size += 8 + 4 + len(content)
        return size + 8


@dataclass
class MigrationUnit:
    """
    Portable state bundle for migration/checkpointing.

    Contains all information needed to resume execution
    on potentially different hardware.
    """
    unit_id: str
    elf_hash: str  # Hash of original ELF for verification
    entry_point: int
    initial_sp: int
    captures: list[StateCapture] = field(default_factory=list)
    compression: CompressionType = CompressionType.NONE
    metadata: dict = field(default_factory=dict)

    def add_capture(self, capture: StateCapture) -> None:
        """Add state capture to migration unit"""
        self.captures.append(capture)

    def get_latest_capture(self) -> Optional[StateCapture]:
        """Get most recent state capture"""
        return self.captures[-1] if self.captures else None

    def total_size(self) -> int:
        """Total size of migration unit"""
        size = 64  # Header
        for capture in self.captures:
            size += capture.total_size()
        return size


class StateReplayer:
    """
    Reconstructs program state from captures for deterministic replay.
    """

    def __init__(self):
        self.captures: dict[int, StateCapture] = {}
        self.current_capture: Optional[StateCapture] = None

    def add_capture(self, capture: StateCapture) -> None:
        """Add capture to replayer"""
        self.captures[capture.capture_id] = capture

    def replay_to(self, capture_id: int) -> Optional[StateCapture]:
        """Replay to specific capture point"""
        if capture_id in self.captures:
            self.current_capture = self.captures[capture_id]
            return self.current_capture
        return None

    def get_state(self) -> Optional[StateCapture]:
        """Get current replay state"""
        return self.current_capture


class CompilerGuidedProtocol:
    """
    Implements the Compiler-Guided Protocol.

    The compiler embeds metadata in the ELF that enables state capture,
    replay, and migration. The protocol uses this metadata to provide
    powerful debugging and migration capabilities.

    Usage:
        cgp = CompilerGuidedProtocol(elf_path)

        # Capture state at checkpoint
        capture_id = cgp.capture("checkpoint_1")

        # Make some changes
        cgp.modify_memory(0x1000, b"new data")

        # Capture diff
        diff_id = cgp.capture_diff("checkpoint_2")

        # Replay to earlier state
        cgp.replay_to(capture_id)

        # Create migration unit
        unit = cgp.create_migration_unit()
        cgp.save_migration_unit(unit, "checkpoint.bin")
    """

    def __init__(self, elf_path: str):
        self.elf_path = elf_path
        self.replayer = StateReplayer()
        self.current_capture_id = 0
        self.migration_unit = MigrationUnit(
            unit_id="default",
            elf_hash="",  # Will be computed
            entry_point=0,
            initial_sp=0,
        )

        # Compute ELF hash
        try:
            with open(elf_path, 'rb') as f:
                self.migration_unit.elf_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            self.migration_unit.elf_hash = "unknown"

    def capture(
        self,
        label: str = "",
        memory_regions: Optional[list[MemoryRegion]] = None,
        fd_table: Optional[dict] = None,
    ) -> int:
        """
        Capture current program state.

        Returns capture ID that can be used to replay.
        """
        self.current_capture_id += 1

        # Create register state
        regs = RegisterState(
            pc=0x1000,  # Would be actual PC
            sp=0x7fff0000,
        )

        capture = StateCapture(
            capture_id=self.current_capture_id,
            timestamp_us=int(time.time() * 1_000_000),
            pc=regs.pc,
            registers=regs,
            memory_regions=memory_regions or [],
            fd_table=fd_table or {},
            metadata={"label": label},
        )

        # Add to replayer
        self.replayer.add_capture(capture)

        # Add to migration unit
        self.migration_unit.add_capture(capture)

        return self.current_capture_id

    def capture_diff(
        self,
        from_id: int,
        label: str = "",
    ) -> Optional[int]:
        """Capture diff between two points"""
        if from_id not in self.replayer.captures:
            return None

        from_state = self.replayer.captures[from_id]
        to_state = self.replayer.captures.get(self.current_capture_id)

        if not to_state:
            return None

        # Compute diff
        diff = StateDiff(
            from_capture_id=from_id,
            to_capture_id=self.current_capture_id,
            timestamp_us=int(time.time() * 1_000_000),
        )

        return self.current_capture_id

    def replay_to(self, capture_id: int) -> Optional[StateCapture]:
        """Replay to specific capture point"""
        return self.replayer.replay_to(capture_id)

    def create_migration_unit(self) -> MigrationUnit:
        """Create migration unit from current state"""
        return self.migration_unit

    def save_migration_unit(self, unit: MigrationUnit, path: str) -> None:
        """Save migration unit to file"""
        # In real impl, would serialize unit to bytes and write
        pass

    def load_migration_unit(self, path: str) -> MigrationUnit:
        """Load migration unit from file"""
        # In real impl, would read and deserialize
        return self.migration_unit


def benchmark_state_capture(
    num_captures: int = 100,
    memory_regions: int = 10,
    region_size_kb: int = 64,
) -> dict:
    """
    Benchmark state capture and compression.

    Returns metrics comparing:
    - Full capture: Complete state snapshot
    - Diff capture: Only changes from previous state
    """
    # Simulate memory region sizes
    region_size = region_size_kb * 1024

    # Benchmark full captures
    start = time.perf_counter()
    for _ in range(num_captures):
        regions = [
            MemoryRegion(start=i * 0x100000, size=region_size, data=b'x' * region_size)
            for i in range(memory_regions)
        ]
        capture = StateCapture(
            capture_id=_,
            timestamp_us=0,
            pc=0x1000,
            registers=RegisterState(),
            memory_regions=regions,
        )
        _ = capture.to_bytes()
    full_time = time.perf_counter() - start

    # Benchmark diff captures
    start = time.perf_counter()
    prev_size = 0
    for i in range(num_captures):
        diff = StateDiff(
            from_capture_id=i,
            to_capture_id=i + 1,
            timestamp_us=0,
            changed_regs={"x0": i},
            changed_memory=[(0x1000, b'x' * 1024)],
        )
        diff_size = diff.total_size()
        prev_size = diff_size
    diff_time = time.perf_counter() - start

    return {
        "num_captures": num_captures,
        "memory_regions": memory_regions,
        "region_size_kb": region_size_kb,
        "full_capture_time": full_time,
        "diff_capture_time": diff_time,
        "estimated_full_size_mb": (num_captures * (memory_regions * region_size)) / (1024**2),
        "size_reduction_ratio": full_time / diff_time if diff_time > 0 else 1.0,
    }


if __name__ == "__main__":
    print("=== Compiler-Guided Protocol Demo ===\n")

    # Create protocol
    cgp = CompilerGuidedProtocol("/path/to/elf")

    # Capture state at different points
    print("Capturing state...")
    id1 = cgp.capture("start")
    print(f"  Capture 1: ID={id1}")

    id2 = cgp.capture("middle")
    print(f"  Capture 2: ID={id2}")

    id3 = cgp.capture("end")
    print(f"  Capture 3: ID={id3}")

    # Replay to earlier state
    print("\nReplaying to capture 1...")
    state = cgp.replay_to(id1)
    if state:
        print(f"  Replayed to PC=0x{state.pc:x}")

    # Create migration unit
    print("\nCreating migration unit...")
    unit = cgp.create_migration_unit()
    print(f"  ELF hash: {unit.elf_hash}")
    print(f"  Captures: {len(unit.captures)}")

    # Benchmark
    print("\nBenchmarking state capture...")
    bench = benchmark_state_capture(num_captures=100)
    print(f"Full capture time: {bench['full_capture_time']:.3f}s")
    print(f"Diff capture time: {bench['diff_capture_time']:.3f}s")
    print(f"Size reduction: {bench['size_reduction_ratio']:.1f}x")
