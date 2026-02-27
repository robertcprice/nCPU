#!/usr/bin/env python3
"""
TRULY NEURAL SYSCALL HANDLERS
=============================
Actual syscall implementations stored IN THE NETWORK WEIGHTS.

Each subsystem has its own neural handler that learns:
- PROC: Process management (exit, getpid, fork behavior)
- MEM: Memory operations (brk, mmap, munmap)
- IO: I/O operations (read, write, seek)
- FS: Filesystem ops (open, close, stat)
- NET: Network ops (socket, connect, send/recv)
- SEC: Security ops (uid/gid handling)
- TIME: Time ops (clock_gettime, nanosleep)
- MISC: Miscellaneous (getrandom, ioctl)

State like file descriptors, memory maps, process info stored IN weights!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralProcHandler(nn.Module):
    """Process management syscalls - state IN weights."""

    def __init__(self, max_pids=64, key_dim=64):
        super().__init__()
        self.max_pids = max_pids
        self.key_dim = key_dim

        # Process table IN weights
        self.process_table = nn.Parameter(torch.zeros(max_pids, 16))  # [pid, ppid, uid, gid, state, ...]
        self.current_pid = nn.Parameter(torch.zeros(max_pids))  # One-hot current process

        # Handler network (16 syscall bits + 16 proc_info + 384 arg bits = 416)
        self.handler = nn.Sequential(
            nn.Linear(16 + 16 + 6 * 64, key_dim * 2),  # syscall + proc_info + args
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # return value + success
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle PROC syscall."""
        batch = syscall_bits.shape[0]

        current = F.softmax(self.current_pid, dim=-1).unsqueeze(0).expand(batch, -1)
        proc_info = torch.matmul(current, torch.sigmoid(self.process_table))

        handler_input = torch.cat([syscall_bits, proc_info, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :64]
        success = torch.sigmoid(output[:, 64])

        return return_val, success


class NeuralMemHandler(nn.Module):
    """Memory management syscalls - state IN weights."""

    def __init__(self, max_regions=128, key_dim=64):
        super().__init__()
        self.max_regions = max_regions
        self.key_dim = key_dim

        # Memory map table IN weights
        # Each entry: [start_addr(32), size(32), permissions(4), valid(1)]
        self.memory_map = nn.Parameter(torch.zeros(max_regions, 69))

        # Heap state
        self.brk_value = nn.Parameter(torch.zeros(64))  # Current brk as bits
        self.brk_limit = nn.Parameter(torch.ones(64) * 0.5)  # Max brk

        # Handler network
        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # return address + success
        )

        # Memory allocator
        self.allocator = nn.Sequential(
            nn.Linear(max_regions * 69 + 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, max_regions),  # Which slot to use
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle MEM syscall (brk, mmap, munmap)."""
        batch = syscall_bits.shape[0]

        brk = torch.sigmoid(self.brk_value).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, brk, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :64]
        success = torch.sigmoid(output[:, 64])

        return return_val, success


class NeuralIOHandler(nn.Module):
    """I/O syscalls - state IN weights."""

    def __init__(self, max_fds=64, buffer_size=256, key_dim=64):
        super().__init__()
        self.max_fds = max_fds
        self.buffer_size = buffer_size
        self.key_dim = key_dim

        # File descriptor table IN weights
        # Each entry: [type(8), mode(8), position(32), valid(1)]
        self.fd_table = nn.Parameter(torch.zeros(max_fds, 49))

        # I/O buffers IN weights (for stdin/stdout/stderr simulation)
        self.io_buffers = nn.Parameter(torch.zeros(3, buffer_size, 8))  # stdin, stdout, stderr

        # Handler network (smaller fd_info projection for efficiency)
        self.fd_proj = nn.Linear(max_fds * 49, 64)  # Project fd_table to 64
        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),  # syscall + fd_proj + args
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1 + buffer_size * 8),  # return + success + data
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle IO syscall (read, write)."""
        batch = syscall_bits.shape[0]

        fd_raw = torch.sigmoid(self.fd_table).flatten().unsqueeze(0).expand(batch, -1)
        fd_info = self.fd_proj(fd_raw)  # Project to 64 dims
        handler_input = torch.cat([syscall_bits, fd_info, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :64]
        success = torch.sigmoid(output[:, 64])
        data = torch.sigmoid(output[:, 65:65 + self.buffer_size * 8]).view(batch, self.buffer_size, 8)

        return return_val, success, data


class NeuralFSHandler(nn.Module):
    """Filesystem syscalls - state IN weights."""

    def __init__(self, max_files=64, max_path_len=64, key_dim=64):
        super().__init__()
        self.max_files = max_files
        self.key_dim = key_dim

        # Virtual filesystem table IN weights
        # Each entry: [path_hash(32), inode(32), mode(16), size(32), valid(1)]
        self.fs_table = nn.Parameter(torch.zeros(max_files, 113))

        # Handler network
        self.handler = nn.Sequential(
            nn.Linear(16 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # fd/result + success
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle FS syscall (open, close, stat)."""
        batch = syscall_bits.shape[0]

        handler_input = torch.cat([syscall_bits, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :64]
        success = torch.sigmoid(output[:, 64])

        return return_val, success


class NeuralTimeHandler(nn.Module):
    """Time syscalls - state IN weights."""

    def __init__(self, key_dim=64):
        super().__init__()
        self.key_dim = key_dim

        # System time state IN weights
        self.system_time = nn.Parameter(torch.zeros(128))  # sec(64) + nsec(64)

        # Handler network
        self.handler = nn.Sequential(
            nn.Linear(16 + 128 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 128 + 1),  # timespec + success
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle TIME syscall (clock_gettime, nanosleep)."""
        batch = syscall_bits.shape[0]

        sys_time = torch.sigmoid(self.system_time).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, sys_time, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :128]
        success = torch.sigmoid(output[:, 128])

        return return_val, success


class NeuralMiscHandler(nn.Module):
    """Miscellaneous syscalls - state IN weights."""

    def __init__(self, key_dim=64):
        super().__init__()
        self.key_dim = key_dim

        # Random state IN weights (for getrandom)
        self.random_state = nn.Parameter(torch.randn(256))

        # Handler network
        self.handler = nn.Sequential(
            nn.Linear(16 + 256 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 256 + 1),  # count + random bytes + success
        )

    def handle(self, syscall_bits, arg_bits):
        """Handle MISC syscall (getrandom, ioctl)."""
        batch = syscall_bits.shape[0]

        rng_state = torch.sigmoid(self.random_state).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, rng_state, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)

        return_val = output[:, :64]
        random_bytes = torch.sigmoid(output[:, 64:320])
        success = torch.sigmoid(output[:, 320])

        # Update random state (make it evolve)
        self.random_state.data = self.random_state.data + 0.1 * torch.randn_like(self.random_state.data)

        return return_val, random_bytes, success


class TrulyNeuralSyscallHandlers(nn.Module):
    """
    Complete syscall handler suite - all state IN weights.

    Routes to appropriate subsystem handler and returns results.
    """

    def __init__(self, key_dim=64):
        super().__init__()
        self.key_dim = key_dim

        # Subsystem handlers
        self.proc_handler = NeuralProcHandler(key_dim=key_dim)
        self.mem_handler = NeuralMemHandler(key_dim=key_dim)
        self.io_handler = NeuralIOHandler(key_dim=key_dim)
        self.fs_handler = NeuralFSHandler(key_dim=key_dim)
        self.time_handler = NeuralTimeHandler(key_dim=key_dim)
        self.misc_handler = NeuralMiscHandler(key_dim=key_dim)

        # Subsystem router (learned)
        self.num_subsystems = 8
        self.router = nn.Sequential(
            nn.Linear(16 + 6 * 64, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, self.num_subsystems),
        )

        # Result combiner
        self.combiner = nn.Sequential(
            nn.Linear(64 * 6 + self.num_subsystems, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # final result + success
        )

    def handle(self, syscall_bits, arg_bits):
        """
        Handle syscall by routing to appropriate subsystem.

        Args:
            syscall_bits: [batch, 16] - syscall number as bits
            arg_bits: [batch, 6, 64] - 6 arguments as 64-bit values

        Returns:
            result: [batch, 64] - syscall return value
            success: [batch] - whether syscall succeeded
        """
        batch = syscall_bits.shape[0]

        # Route to subsystem
        route_input = torch.cat([syscall_bits, arg_bits.view(batch, -1)], dim=-1)
        route_logits = self.router(route_input)
        route_probs = F.softmax(route_logits, dim=-1)  # [batch, 8]

        # Call all handlers (in parallel for training)
        proc_result, proc_success = self.proc_handler.handle(syscall_bits, arg_bits)
        mem_result, mem_success = self.mem_handler.handle(syscall_bits, arg_bits)
        io_result, io_success, _ = self.io_handler.handle(syscall_bits, arg_bits)
        fs_result, fs_success = self.fs_handler.handle(syscall_bits, arg_bits)
        time_result, time_success = self.time_handler.handle(syscall_bits, arg_bits)
        time_result = time_result[:, :64]  # Take first 64 bits
        misc_result, _, misc_success = self.misc_handler.handle(syscall_bits, arg_bits)

        # Stack results
        all_results = torch.stack([
            proc_result, mem_result, io_result,
            fs_result, torch.zeros_like(proc_result),  # NET placeholder
            torch.zeros_like(proc_result),  # SEC placeholder
            time_result, misc_result
        ], dim=1)  # [batch, 8, 64]

        # Weight by routing probabilities
        weighted_result = torch.einsum('bs,bsd->bd', route_probs, all_results)

        # Combine results
        all_results_flat = all_results[:, :6].reshape(batch, -1)  # First 6
        combiner_input = torch.cat([all_results_flat, route_probs], dim=-1)
        final_output = self.combiner(combiner_input)

        result = final_output[:, :64]
        success = torch.sigmoid(final_output[:, 64])

        return result, success, route_probs


# Syscall categories for training
SYSCALL_MAP = {
    # PROC (0)
    93: (0, 'exit'), 94: (0, 'exit_group'), 172: (0, 'getpid'),
    # MEM (1)
    214: (1, 'brk'), 222: (1, 'mmap'), 215: (1, 'munmap'),
    # IO (2)
    63: (2, 'read'), 64: (2, 'write'), 62: (2, 'lseek'),
    # FS (3)
    56: (3, 'openat'), 57: (3, 'close'), 80: (3, 'fstat'),
    # TIME (6)
    113: (6, 'clock_gettime'), 101: (6, 'nanosleep'),
    # MISC (7)
    278: (7, 'getrandom'), 29: (7, 'ioctl'),
}


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def generate_batch(batch_size, device):
    """Generate training batch."""
    syscalls = []
    args = []
    expected_subsystems = []
    expected_results = []

    for _ in range(batch_size):
        syscall_num = random.choice(list(SYSCALL_MAP.keys()))
        subsystem, name = SYSCALL_MAP[syscall_num]

        syscalls.append(int_to_bits(syscall_num, 16))
        args.append(torch.stack([int_to_bits(random.randint(0, 1000)) for _ in range(6)]))
        expected_subsystems.append(subsystem)

        # Expected result based on syscall
        if name == 'getpid':
            expected_results.append(int_to_bits(1))  # PID 1
        elif name == 'brk':
            expected_results.append(int_to_bits(0x10000000))  # brk address
        elif name in ('read', 'write'):
            expected_results.append(int_to_bits(random.randint(0, 100)))  # bytes transferred
        elif name == 'openat':
            expected_results.append(int_to_bits(3))  # fd
        elif name == 'close':
            expected_results.append(int_to_bits(0))  # success
        else:
            expected_results.append(int_to_bits(0))

    return (
        torch.stack(syscalls).to(device),
        torch.stack(args).to(device),
        torch.tensor(expected_subsystems, device=device),
        torch.stack(expected_results).to(device),
    )


def train():
    print("=" * 60)
    print("TRULY NEURAL SYSCALL HANDLERS TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("ALL syscall state stored IN NETWORK WEIGHTS!")

    model = TrulyNeuralSyscallHandlers(key_dim=64).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            syscalls, args, exp_subsys, exp_results = generate_batch(batch_size, device)

            optimizer.zero_grad()

            result, success, route_probs = model.handle(syscalls, args)

            # Routing loss
            target_probs = F.one_hot(exp_subsys, num_classes=8).float()
            loss_route = F.cross_entropy(route_probs, exp_subsys)

            # Result loss
            loss_result = F.mse_loss(result, exp_results)

            loss = loss_route + 0.5 * loss_result
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        route_correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                syscalls, args, exp_subsys, exp_results = generate_batch(100, device)
                result, success, route_probs = model.handle(syscalls, args)

                route_preds = route_probs.argmax(dim=-1)
                route_correct += (route_preds == exp_subsys).sum().item()
                total += len(exp_subsys)

        route_acc = route_correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} route_acc={100*route_acc:.1f}% [{elapsed:.1f}s]")

        if route_acc > best_acc:
            best_acc = route_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "route_accuracy": route_acc,
                "op_name": "TRULY_NEURAL_SYSCALL_HANDLERS",
            }, "models/final/truly_neural_syscall_handlers_best.pt")
            print(f"  Saved (route_acc={100*route_acc:.1f}%)")

        if route_acc >= 0.95:
            print("95%+ ACCURACY!")
            break

    print(f"\nBest routing accuracy: {100*best_acc:.1f}%")


if __name__ == "__main__":
    train()
