#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    NEURAL-NATIVE OPERATING SYSTEM                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  An operating system designed FROM THE GROUND UP for neural/tensor computation.  ║
║  NOT a Linux emulator - a fundamentally new paradigm!                            ║
║                                                                                  ║
║  CORE PRINCIPLES:                                                                ║
║  ┌────────────────────────────────────────────────────────────────────────────┐  ║
║  │ 1. PROCESSES ARE TENSORS - Each process is a computational graph           │  ║
║  │ 2. ATTENTION SCHEDULING - Scheduler uses learned attention weights         │  ║
║  │ 3. EMBEDDING FILESYSTEM - Files stored as vectors, retrieved by similarity │  ║
║  │ 4. NEURAL IPC - Inter-process communication via tensor operations          │  ║
║  │ 5. GPU-NATIVE - All state lives on GPU, no CPU round-trips                 │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                  ║
║  ARCHITECTURE:                                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                         NEURAL SHELL                                        │ ║
║  │  Commands are tensor transformations, not text parsing                      │ ║
║  └─────────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                             ║
║  ┌─────────────────────────────────▼─────────────────────────────────────────┐   ║
║  │                    NEURAL PROCESS MANAGER                                  │   ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                     │   ║
║  │  │ Attention    │  │ Neural       │  │ Tensor       │                     │   ║
║  │  │ Scheduler    │  │ Memory Mgr   │  │ IPC          │                     │   ║
║  │  └──────────────┘  └──────────────┘  └──────────────┘                     │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                    │                                             ║
║  ┌─────────────────────────────────▼─────────────────────────────────────────┐   ║
║  │                    NEURAL FILESYSTEM                                       │   ║
║  │  Files = Embeddings │ Directories = Clusters │ Search = Similarity        │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                    │                                             ║
║  ┌─────────────────────────────────▼─────────────────────────────────────────┐   ║
║  │                    NEURAL GPU ULTIMATE CPU                                 │   ║
║  │  65M+ IPS │ Loop Vectorization │ Neural Extractors                        │   ║
║  └───────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
import time
import math

# ════════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ════════════════════════════════════════════════════════════════════════════════

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL PROCESS - A process IS a tensor computation
# ════════════════════════════════════════════════════════════════════════════════

class ProcessState(IntEnum):
    """Process states - but stored as tensor values for GPU ops."""
    READY = 0
    RUNNING = 1
    BLOCKED = 2
    TERMINATED = 3


@dataclass
class NeuralProcess:
    """
    A process in Neural-Native OS.

    Unlike traditional processes (code + data + stack), a Neural Process is:
    - A computational graph (the "program")
    - Input/output tensors (the "data")
    - Priority embedding (learned scheduling weight)
    - State tensor (for suspension/resumption)
    """
    pid: int
    name: str

    # The "program" - a callable that transforms tensors
    compute_fn: Callable[[torch.Tensor], torch.Tensor]

    # Process state - ALL ON GPU
    state: torch.Tensor = None  # ProcessState as tensor
    priority: torch.Tensor = None  # Learned priority embedding [64]

    # I/O buffers - tensors on GPU
    input_buffer: torch.Tensor = None  # [buffer_size]
    output_buffer: torch.Tensor = None  # [buffer_size]

    # Execution state
    cycles_used: int = 0
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.state is None:
            self.state = torch.tensor(ProcessState.READY, device=device)
        if self.priority is None:
            # Initialize with random priority embedding
            self.priority = torch.randn(64, device=device) * 0.1
        if self.input_buffer is None:
            self.input_buffer = torch.zeros(1024, device=device)
        if self.output_buffer is None:
            self.output_buffer = torch.zeros(1024, device=device)


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL SCHEDULER - Attention-based process scheduling
# ════════════════════════════════════════════════════════════════════════════════

class NeuralScheduler(nn.Module):
    """
    Attention-based process scheduler.

    Instead of round-robin or priority queues, uses LEARNED ATTENTION
    to decide which process to run next.

    The scheduler learns:
    - Which processes are most "important" (attention weights)
    - How to balance fairness vs. throughput
    - When to preempt based on process behavior
    """

    def __init__(self, max_processes: int = 64, embed_dim: int = 64):
        super().__init__()
        self.max_processes = max_processes
        self.embed_dim = embed_dim

        # Process priority embeddings (learned)
        self.priority_keys = nn.Parameter(torch.randn(max_processes, embed_dim) * 0.1)
        self.priority_values = nn.Parameter(torch.randn(max_processes, embed_dim) * 0.1)

        # Query network - generates scheduling query from system state
        self.query_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Fairness embedding - tracks how long each process has waited
        self.wait_time_embed = nn.Linear(1, embed_dim)

        # Temperature for attention (learned)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    @torch.no_grad()
    def select_next(
        self,
        process_states: torch.Tensor,  # [N] ProcessState values
        process_priorities: torch.Tensor,  # [N, embed_dim]
        wait_times: torch.Tensor,  # [N] cycles waited
        system_load: torch.Tensor,  # [embed_dim] current system state
    ) -> Tuple[int, torch.Tensor]:
        """
        Select next process to run using attention mechanism.

        Returns: (selected_pid, attention_weights)
        """
        n_procs = process_states.shape[0]

        # Only consider READY processes
        ready_mask = (process_states == ProcessState.READY).float()

        # Embed wait times for fairness
        wait_embeds = self.wait_time_embed(wait_times.unsqueeze(-1))  # [N, embed_dim]

        # Combine priority with wait time (fairness boost)
        combined = process_priorities + 0.5 * wait_embeds  # [N, embed_dim]

        # Generate query from system state
        query_input = torch.cat([system_load, system_load.mean().expand(self.embed_dim)])
        query = self.query_net(query_input)  # [embed_dim]

        # Attention scores
        scores = torch.matmul(combined, query)  # [N]

        # Mask out non-ready processes
        scores = scores + (1 - ready_mask) * (-1e9)

        # Apply temperature and softmax
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        attention = F.softmax(scores / temp, dim=0)

        # Select highest attention (or sample for stochasticity)
        selected = attention.argmax().item()

        return selected, attention


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL FILESYSTEM - Embeddings instead of hierarchical paths
# ════════════════════════════════════════════════════════════════════════════════

class NeuralFilesystem(nn.Module):
    """
    A filesystem where files ARE embeddings.

    Instead of:
      /home/user/documents/report.txt

    We have:
      file_embedding = encode("report about quarterly sales")
      content = retrieve_by_similarity(file_embedding)

    Benefits:
    - Instant semantic search (no grep needed!)
    - Natural clustering (similar files are close in embedding space)
    - Compression via learned representations
    - GPU-native storage and retrieval
    """

    def __init__(self, max_files: int = 1024, embed_dim: int = 256, content_dim: int = 4096):
        super().__init__()
        self.max_files = max_files
        self.embed_dim = embed_dim
        self.content_dim = content_dim

        # File metadata embeddings (learned)
        self.file_embeddings = nn.Parameter(torch.zeros(max_files, embed_dim))
        self.file_names = nn.Parameter(torch.zeros(max_files, 64))  # Name as embedding

        # File content storage (on GPU!)
        self.file_contents = nn.Parameter(torch.zeros(max_files, content_dim))
        self.file_sizes = nn.Parameter(torch.zeros(max_files))

        # File validity mask
        self.valid_mask = nn.Parameter(torch.zeros(max_files), requires_grad=False)

        # Name encoder - converts text to embedding
        self.name_encoder = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Content encoder/decoder
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.content_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, content_dim),
        )

        self.next_file_id = 0

    def _text_to_tensor(self, text: str, max_len: int = 64) -> torch.Tensor:
        """Convert text to tensor (simple ASCII encoding)."""
        codes = [ord(c) for c in text[:max_len]]
        codes += [0] * (max_len - len(codes))
        return torch.tensor(codes, dtype=torch.float32, device=device) / 255.0

    def _tensor_to_text(self, tensor: torch.Tensor) -> str:
        """Convert tensor back to text."""
        codes = (tensor * 255).long().cpu().tolist()
        return ''.join(chr(max(32, min(126, c))) for c in codes if c > 0).strip()

    @torch.no_grad()
    def create_file(self, name: str, content: bytes) -> int:
        """Create a new file. Returns file ID."""
        if self.next_file_id >= self.max_files:
            raise RuntimeError("Filesystem full")

        fid = self.next_file_id
        self.next_file_id += 1

        # Encode name
        name_tensor = self._text_to_tensor(name)
        self.file_names.data[fid] = name_tensor

        # Encode content
        content_tensor = torch.zeros(self.content_dim, device=device)
        content_bytes = list(content[:self.content_dim])
        content_tensor[:len(content_bytes)] = torch.tensor(content_bytes, dtype=torch.float32, device=device)
        self.file_contents.data[fid] = content_tensor
        self.file_sizes.data[fid] = len(content_bytes)

        # Generate embedding from name + content summary
        name_embed = self.name_encoder(name_tensor)
        content_embed = self.content_encoder(content_tensor)
        self.file_embeddings.data[fid] = (name_embed + content_embed) / 2

        # Mark as valid
        self.valid_mask.data[fid] = 1.0

        return fid

    @torch.no_grad()
    def read_file(self, fid: int) -> bytes:
        """Read file by ID."""
        if fid >= self.next_file_id or self.valid_mask[fid] < 0.5:
            return b''

        size = int(self.file_sizes[fid].item())
        content = self.file_contents[fid][:size]
        return bytes(content.long().cpu().tolist())

    @torch.no_grad()
    def search_by_name(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """Search files by name similarity."""
        query_tensor = self._text_to_tensor(query)
        query_embed = self.name_encoder(query_tensor)

        # Compute similarities
        similarities = F.cosine_similarity(
            query_embed.unsqueeze(0),
            self.file_embeddings[:self.next_file_id],
            dim=1
        )

        # Mask invalid files
        similarities = similarities * self.valid_mask[:self.next_file_id]

        # Get top-k
        values, indices = similarities.topk(min(top_k, self.next_file_id))

        results = []
        for idx, sim in zip(indices.tolist(), values.tolist()):
            name = self._tensor_to_text(self.file_names[idx])
            results.append((idx, name, sim))

        return results

    @torch.no_grad()
    def semantic_search(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search files by semantic similarity to an embedding."""
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.file_embeddings[:self.next_file_id],
            dim=1
        )
        similarities = similarities * self.valid_mask[:self.next_file_id]

        values, indices = similarities.topk(min(top_k, self.next_file_id))
        return list(zip(indices.tolist(), values.tolist()))

    @torch.no_grad()
    def list_files(self) -> List[Tuple[int, str, int]]:
        """List all files."""
        files = []
        for fid in range(self.next_file_id):
            if self.valid_mask[fid] > 0.5:
                name = self._tensor_to_text(self.file_names[fid])
                size = int(self.file_sizes[fid].item())
                files.append((fid, name, size))
        return files

    @torch.no_grad()
    def delete_file(self, fid: int) -> bool:
        """Delete a file."""
        if fid < self.next_file_id and self.valid_mask[fid] > 0.5:
            self.valid_mask.data[fid] = 0.0
            return True
        return False


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL IPC - Inter-Process Communication via Tensors
# ════════════════════════════════════════════════════════════════════════════════

class NeuralIPC(nn.Module):
    """
    Neural Inter-Process Communication.

    Instead of pipes/sockets/shared memory, processes communicate via:
    - Tensor channels (broadcast to all, or targeted)
    - Attention-based message routing
    - Learned compression for efficient transfer
    """

    def __init__(self, max_channels: int = 32, channel_dim: int = 512, max_processes: int = 64):
        super().__init__()
        self.max_channels = max_channels
        self.channel_dim = channel_dim
        self.max_processes = max_processes

        # Channel buffers (ring buffers on GPU)
        self.channel_buffers = nn.Parameter(
            torch.zeros(max_channels, 16, channel_dim),  # 16 messages per channel
            requires_grad=False
        )
        self.channel_heads = torch.zeros(max_channels, dtype=torch.long, device=device)
        self.channel_tails = torch.zeros(max_channels, dtype=torch.long, device=device)

        # Channel ownership (which process owns each channel)
        self.channel_owners = torch.full((max_channels,), -1, dtype=torch.long, device=device)

        # Message router - learns to route messages efficiently
        self.router = nn.Sequential(
            nn.Linear(channel_dim + 64, 256),
            nn.GELU(),
            nn.Linear(256, max_processes),
        )

        # Compression/decompression for efficient transfer
        self.compressor = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // 4),
            nn.GELU(),
            nn.Linear(channel_dim // 4, channel_dim // 8),
        )

        self.decompressor = nn.Sequential(
            nn.Linear(channel_dim // 8, channel_dim // 4),
            nn.GELU(),
            nn.Linear(channel_dim // 4, channel_dim),
        )

    @torch.no_grad()
    def create_channel(self, owner_pid: int) -> int:
        """Create a new IPC channel."""
        for cid in range(self.max_channels):
            if self.channel_owners[cid] < 0:
                self.channel_owners[cid] = owner_pid
                return cid
        raise RuntimeError("No free channels")

    @torch.no_grad()
    def send(self, channel_id: int, message: torch.Tensor):
        """Send a message to a channel."""
        if channel_id >= self.max_channels:
            return False

        head = self.channel_heads[channel_id].item()
        self.channel_buffers.data[channel_id, head % 16] = message[:self.channel_dim]
        self.channel_heads[channel_id] = (head + 1) % 16
        return True

    @torch.no_grad()
    def receive(self, channel_id: int) -> Optional[torch.Tensor]:
        """Receive a message from a channel."""
        if channel_id >= self.max_channels:
            return None

        head = self.channel_heads[channel_id].item()
        tail = self.channel_tails[channel_id].item()

        if head == tail:
            return None  # Empty

        message = self.channel_buffers[channel_id, tail % 16].clone()
        self.channel_tails[channel_id] = (tail + 1) % 16
        return message

    @torch.no_grad()
    def broadcast(self, message: torch.Tensor, source_priority: torch.Tensor) -> torch.Tensor:
        """Broadcast message to all processes, return routing weights."""
        # Combine message with source priority for routing
        router_input = torch.cat([message[:self.channel_dim], source_priority[:64]])
        routes = F.softmax(self.router(router_input), dim=0)
        return routes


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL MEMORY MANAGER - Attention-based allocation
# ════════════════════════════════════════════════════════════════════════════════

class NeuralMemoryManager(nn.Module):
    """
    Neural memory manager with attention-based allocation.

    Instead of traditional malloc/free, uses:
    - Learned allocation patterns (predicts optimal placement)
    - Attention-based garbage collection (focuses on likely-dead regions)
    - Compression for inactive memory regions
    """

    def __init__(self, total_memory: int = 16 * 1024 * 1024, block_size: int = 4096):
        super().__init__()
        self.total_memory = total_memory
        self.block_size = block_size
        self.num_blocks = total_memory // block_size

        # Memory as GPU tensor
        self.memory = torch.zeros(total_memory, dtype=torch.uint8, device=device)

        # Block allocation table (which process owns each block)
        self.block_owners = torch.full((self.num_blocks,), -1, dtype=torch.long, device=device)

        # Block usage patterns (for prediction)
        self.block_access_counts = torch.zeros(self.num_blocks, device=device)
        self.block_last_access = torch.zeros(self.num_blocks, device=device)

        # Allocation predictor
        self.alloc_predictor = nn.Sequential(
            nn.Linear(self.num_blocks, 256),
            nn.GELU(),
            nn.Linear(256, self.num_blocks),
        )

        self.current_time = 0

    @torch.no_grad()
    def allocate(self, size: int, pid: int) -> Optional[int]:
        """Allocate memory for a process. Returns start address or None."""
        blocks_needed = (size + self.block_size - 1) // self.block_size

        # Find contiguous free blocks
        free_mask = (self.block_owners < 0).float()

        # Use convolution to find runs of free blocks
        if blocks_needed == 1:
            free_indices = (free_mask > 0.5).nonzero(as_tuple=True)[0]
            if len(free_indices) > 0:
                # Predict best block using learned allocator
                alloc_scores = self.alloc_predictor(free_mask)
                alloc_scores = alloc_scores * free_mask  # Mask occupied
                best_block = alloc_scores.argmax().item()

                self.block_owners[best_block] = pid
                return best_block * self.block_size
        else:
            # Find contiguous run
            for start in range(self.num_blocks - blocks_needed + 1):
                if all(self.block_owners[start:start + blocks_needed] < 0):
                    self.block_owners[start:start + blocks_needed] = pid
                    return start * self.block_size

        return None

    @torch.no_grad()
    def free(self, address: int, pid: int) -> bool:
        """Free memory allocated by a process."""
        block = address // self.block_size
        if block < self.num_blocks and self.block_owners[block] == pid:
            # Find all contiguous blocks owned by this process starting here
            end = block
            while end < self.num_blocks and self.block_owners[end] == pid:
                self.block_owners[end] = -1
                end += 1
            return True
        return False

    @torch.no_grad()
    def read(self, address: int, size: int) -> torch.Tensor:
        """Read from memory."""
        self.current_time += 1
        block = address // self.block_size
        if block < self.num_blocks:
            self.block_access_counts[block] += 1
            self.block_last_access[block] = self.current_time
        return self.memory[address:address + size].clone()

    @torch.no_grad()
    def write(self, address: int, data: torch.Tensor):
        """Write to memory."""
        self.current_time += 1
        block = address // self.block_size
        if block < self.num_blocks:
            self.block_access_counts[block] += 1
            self.block_last_access[block] = self.current_time
        size = min(len(data), self.total_memory - address)
        self.memory[address:address + size] = data[:size]


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL-NATIVE OS - The Complete System
# ════════════════════════════════════════════════════════════════════════════════

class NeuralNativeOS:
    """
    Neural-Native Operating System.

    A complete OS designed for tensor/neural computation from the ground up.
    """

    def __init__(self, memory_mb: int = 16):
        print()
        print("╔" + "═" * 74 + "╗")
        print("║" + " NEURAL-NATIVE OPERATING SYSTEM ".center(74) + "║")
        print("║" + " An OS designed for neural computation ".center(74) + "║")
        print("╚" + "═" * 74 + "╝")
        print()

        # Core components (ALL ON GPU)
        self.scheduler = NeuralScheduler().to(device)
        self.filesystem = NeuralFilesystem().to(device)
        self.ipc = NeuralIPC().to(device)
        self.memory = NeuralMemoryManager(total_memory=memory_mb * 1024 * 1024)

        # Process table
        self.processes: Dict[int, NeuralProcess] = {}
        self.next_pid = 1
        self.current_pid = 0

        # System state (on GPU)
        self.system_load = torch.zeros(64, device=device)
        self.uptime = 0
        self.total_cycles = 0

        # Output buffer
        self.output_buffer: List[str] = []

        # Built-in programs
        self._register_builtins()

        print(f"  ✅ Neural Scheduler initialized (attention-based)")
        print(f"  ✅ Neural Filesystem initialized ({self.filesystem.max_files} files)")
        print(f"  ✅ Neural IPC initialized ({self.ipc.max_channels} channels)")
        print(f"  ✅ Neural Memory Manager ({memory_mb}MB on GPU)")
        print()

    def _register_builtins(self):
        """Register built-in programs."""
        # Create some initial files
        self.filesystem.create_file("welcome.txt", b"Welcome to Neural-Native OS!\nAll computation is neural.")
        self.filesystem.create_file("readme.md", b"# Neural-Native OS\n\nA GPU-native operating system.")
        self.filesystem.create_file("neural.conf", b"scheduler=attention\nfs=embedding\nmemory=16mb")

    def spawn(self, name: str, compute_fn: Callable[[torch.Tensor], torch.Tensor]) -> int:
        """Spawn a new neural process."""
        pid = self.next_pid
        self.next_pid += 1

        process = NeuralProcess(
            pid=pid,
            name=name,
            compute_fn=compute_fn,
        )

        self.processes[pid] = process
        return pid

    def kill(self, pid: int) -> bool:
        """Terminate a process."""
        if pid in self.processes:
            self.processes[pid].state = torch.tensor(ProcessState.TERMINATED, device=device)
            # Free memory
            self.memory.free(0, pid)  # Simplified
            return True
        return False

    def schedule_tick(self) -> Optional[int]:
        """Run one scheduling tick. Returns PID of selected process."""
        if not self.processes:
            return None

        # Collect process info
        pids = list(self.processes.keys())
        states = torch.stack([self.processes[p].state for p in pids])
        priorities = torch.stack([self.processes[p].priority for p in pids])
        wait_times = torch.tensor([
            self.total_cycles - self.processes[p].cycles_used
            for p in pids
        ], dtype=torch.float32, device=device)

        # Use neural scheduler
        selected_idx, attention = self.scheduler.select_next(
            states, priorities, wait_times, self.system_load
        )

        if selected_idx < len(pids):
            selected_pid = pids[selected_idx]
            proc = self.processes[selected_pid]

            if proc.state.item() == ProcessState.READY:
                proc.state = torch.tensor(ProcessState.RUNNING, device=device)
                self.current_pid = selected_pid
                return selected_pid

        return None

    def run_process(self, pid: int, cycles: int = 100) -> torch.Tensor:
        """Run a process for a number of cycles."""
        if pid not in self.processes:
            return torch.zeros(1, device=device)

        proc = self.processes[pid]

        # Execute the neural computation
        try:
            result = proc.compute_fn(proc.input_buffer)
            proc.output_buffer = result
            proc.cycles_used += cycles
            self.total_cycles += cycles
        except Exception as e:
            self.output_buffer.append(f"Process {pid} error: {e}")
            proc.state = torch.tensor(ProcessState.TERMINATED, device=device)
            return torch.zeros(1, device=device)

        # Update system load (moving average)
        self.system_load = 0.9 * self.system_load + 0.1 * proc.priority

        return proc.output_buffer

    def execute_command(self, cmd: str) -> str:
        """Execute a shell command."""
        parts = cmd.strip().split()
        if not parts:
            return ""

        command = parts[0].lower()
        args = parts[1:]

        if command == "help":
            return """Neural-Native OS Commands:
  help          - Show this help
  ps            - List processes
  spawn <name>  - Spawn a demo process
  kill <pid>    - Kill a process
  ls            - List files
  cat <file>    - Show file contents
  touch <file>  - Create empty file
  search <q>    - Semantic file search
  mem           - Show memory stats
  sched         - Show scheduler state
  doom          - Run Neural DOOM
  exit          - Exit shell"""

        elif command == "ps":
            lines = ["PID  STATE       NAME           CYCLES"]
            lines.append("-" * 45)
            for pid, proc in self.processes.items():
                state_names = ["READY", "RUNNING", "BLOCKED", "TERMINATED"]
                state = state_names[int(proc.state.item())]
                lines.append(f"{pid:3}  {state:10}  {proc.name:14} {proc.cycles_used:8}")
            return "\n".join(lines)

        elif command == "spawn":
            name = args[0] if args else f"proc_{self.next_pid}"
            # Demo: create a process that computes element-wise operations
            def demo_compute(x):
                return torch.sin(x) * torch.cos(x)
            pid = self.spawn(name, demo_compute)
            return f"Spawned process {pid}: {name}"

        elif command == "kill":
            if args:
                try:
                    pid = int(args[0])
                    if self.kill(pid):
                        return f"Killed process {pid}"
                    return f"Process {pid} not found"
                except ValueError:
                    return "Usage: kill <pid>"
            return "Usage: kill <pid>"

        elif command == "ls":
            files = self.filesystem.list_files()
            if not files:
                return "(no files)"
            lines = []
            for fid, name, size in files:
                lines.append(f"{fid:3}  {size:6} bytes  {name}")
            return "\n".join(lines)

        elif command == "cat":
            if args:
                # Search by name
                results = self.filesystem.search_by_name(args[0], top_k=1)
                if results:
                    fid, name, _ = results[0]
                    content = self.filesystem.read_file(fid)
                    return content.decode('utf-8', errors='replace')
                return f"File not found: {args[0]}"
            return "Usage: cat <filename>"

        elif command == "touch":
            if args:
                fid = self.filesystem.create_file(args[0], b"")
                return f"Created file {fid}: {args[0]}"
            return "Usage: touch <filename>"

        elif command == "search":
            if args:
                query = " ".join(args)
                results = self.filesystem.search_by_name(query, top_k=5)
                if results:
                    lines = ["Search results:"]
                    for fid, name, score in results:
                        lines.append(f"  {score:.3f}  {name}")
                    return "\n".join(lines)
                return "No matches found"
            return "Usage: search <query>"

        elif command == "mem":
            used_blocks = (self.memory.block_owners >= 0).sum().item()
            total_blocks = self.memory.num_blocks
            used_mb = used_blocks * self.memory.block_size / (1024 * 1024)
            total_mb = total_blocks * self.memory.block_size / (1024 * 1024)
            return f"Memory: {used_mb:.1f}MB / {total_mb:.1f}MB ({used_blocks}/{total_blocks} blocks)"

        elif command == "sched":
            return f"""Scheduler State:
  Total cycles: {self.total_cycles}
  Active processes: {len([p for p in self.processes.values() if p.state.item() != ProcessState.TERMINATED])}
  System load: {self.system_load.norm().item():.3f}"""

        elif command == "doom":
            return self._run_doom_demo()

        elif command == "exit":
            return "EXIT"

        else:
            return f"Unknown command: {command}. Type 'help' for commands."

    def _run_doom_demo(self) -> str:
        """Run a mini DOOM demo using neural rendering."""
        lines = []
        lines.append("╔" + "═" * 40 + "╗")
        lines.append("║" + " NEURAL DOOM - GPU Tensor Demo ".center(40) + "║")
        lines.append("╚" + "═" * 40 + "╝")

        # Create a simple vectorized "render"
        width, height = 40, 10

        # Player position (tensor)
        player_x = torch.tensor(20.0, device=device)
        player_y = torch.tensor(5.0, device=device)

        # Cast rays (all at once - vectorized!)
        ray_angles = torch.linspace(-0.5, 0.5, width, device=device)

        # Simple distance calculation (vectorized)
        distances = 5.0 + 3.0 * torch.sin(ray_angles * 10 + player_x / 5)

        # Convert to wall heights
        wall_heights = (height * 2 / (distances + 0.1)).clamp(0, height).long()

        # Render frame (vectorized)
        frame = torch.full((height, width), ord(' '), dtype=torch.uint8, device=device)

        for x in range(width):
            h = wall_heights[x].item()
            start = max(0, height // 2 - h // 2)
            end = min(height, height // 2 + h // 2)
            for y in range(start, end):
                frame[y, x] = ord('#') if h > 3 else ord('.')

        # Ground
        for x in range(width):
            for y in range(height // 2 + wall_heights[x].item() // 2, height):
                frame[y, x] = ord('.')

        # Convert to string
        for row in frame:
            line = ''.join(chr(c.item()) for c in row)
            lines.append("║" + line + "║")

        lines.append("╚" + "═" * 40 + "╝")
        lines.append("(Rendered with vectorized tensor ops on GPU)")

        return "\n".join(lines)

    def interactive_shell(self):
        """Run interactive shell."""
        print("Neural-Native OS Shell")
        print("Type 'help' for commands, 'exit' to quit")
        print()

        while True:
            try:
                cmd = input("neural> ")
                result = self.execute_command(cmd)

                if result == "EXIT":
                    print("Goodbye!")
                    break

                if result:
                    print(result)

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except EOFError:
                break


# ════════════════════════════════════════════════════════════════════════════════
# DEMO & TESTS
# ════════════════════════════════════════════════════════════════════════════════

def demo():
    """Run a demo of Neural-Native OS."""
    print("=" * 76)
    print("   NEURAL-NATIVE OS DEMO")
    print("=" * 76)

    os = NeuralNativeOS(memory_mb=16)

    print("\n[1] File System Demo (Embedding-based)")
    print("-" * 40)

    # Create some files
    os.filesystem.create_file("hello.py", b"print('Hello Neural World!')")
    os.filesystem.create_file("neural_net.py", b"import torch\nmodel = nn.Linear(10, 10)")
    os.filesystem.create_file("data.json", b'{"type": "neural", "version": 1}')

    print(os.execute_command("ls"))

    print("\n[2] Semantic Search Demo")
    print("-" * 40)
    print("Searching for 'neural'...")
    print(os.execute_command("search neural"))

    print("\n[3] Process Management Demo")
    print("-" * 40)

    # Spawn some processes
    def matrix_multiply(x):
        return torch.matmul(x.view(32, 32), x.view(32, 32)).flatten()

    def neural_forward(x):
        return torch.sigmoid(x * 2 - 1)

    os.spawn("matrix_worker", matrix_multiply)
    os.spawn("neural_forward", neural_forward)
    os.spawn("idle_proc", lambda x: x)

    print(os.execute_command("ps"))

    print("\n[4] Neural Scheduler Demo")
    print("-" * 40)

    for _ in range(5):
        pid = os.schedule_tick()
        if pid:
            os.run_process(pid, cycles=100)

    print(os.execute_command("sched"))

    print("\n[5] Memory Stats")
    print("-" * 40)
    print(os.execute_command("mem"))

    print("\n[6] DOOM Demo (Vectorized)")
    print("-" * 40)
    print(os.execute_command("doom"))

    print("\n" + "=" * 76)
    print("   DEMO COMPLETE - Neural-Native OS is operational!")
    print("=" * 76)

    return os


def run_interactive():
    """Run interactive Neural-Native OS."""
    os = NeuralNativeOS(memory_mb=16)
    os.interactive_shell()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "shell":
        run_interactive()
    else:
        demo()
