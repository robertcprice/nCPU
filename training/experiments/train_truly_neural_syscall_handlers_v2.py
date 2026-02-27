#!/usr/bin/env python3
"""
TRULY NEURAL SYSCALL HANDLERS V2
================================
Improved version targeting 99%+ accuracy.

Changes from V1:
1. Deeper router network for better syscall→subsystem mapping
2. More training epochs (200 vs 100)
3. Higher accuracy threshold (99% vs 95%)
4. Better learning rate schedule
5. Balanced syscall sampling
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

        self.process_table = nn.Parameter(torch.zeros(max_pids, 16))
        self.current_pid = nn.Parameter(torch.zeros(max_pids))

        self.handler = nn.Sequential(
            nn.Linear(16 + 16 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        current = F.softmax(self.current_pid, dim=-1).unsqueeze(0).expand(batch, -1)
        proc_info = torch.matmul(current, torch.sigmoid(self.process_table))
        handler_input = torch.cat([syscall_bits, proc_info, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class NeuralMemHandler(nn.Module):
    """Memory management syscalls - state IN weights."""

    def __init__(self, max_regions=128, key_dim=64):
        super().__init__()
        self.key_dim = key_dim
        self.brk_value = nn.Parameter(torch.zeros(64))

        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        brk = torch.sigmoid(self.brk_value).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, brk, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class NeuralIOHandler(nn.Module):
    """I/O syscalls - state IN weights."""

    def __init__(self, max_fds=64, buffer_size=256, key_dim=64):
        super().__init__()
        self.fd_proj = nn.Linear(max_fds * 49, 64)
        self.fd_table = nn.Parameter(torch.zeros(max_fds, 49))

        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        fd_raw = torch.sigmoid(self.fd_table).flatten().unsqueeze(0).expand(batch, -1)
        fd_info = self.fd_proj(fd_raw)
        handler_input = torch.cat([syscall_bits, fd_info, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class NeuralFSHandler(nn.Module):
    """Filesystem syscalls - state IN weights."""

    def __init__(self, key_dim=64):
        super().__init__()
        self.handler = nn.Sequential(
            nn.Linear(16 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        handler_input = torch.cat([syscall_bits, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class NeuralTimeHandler(nn.Module):
    """Time syscalls - state IN weights."""

    def __init__(self, key_dim=64):
        super().__init__()
        self.system_time = nn.Parameter(torch.zeros(64))

        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        sys_time = torch.sigmoid(self.system_time).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, sys_time, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class NeuralMiscHandler(nn.Module):
    """Miscellaneous syscalls - state IN weights."""

    def __init__(self, key_dim=64):
        super().__init__()
        self.random_state = nn.Parameter(torch.randn(64))

        self.handler = nn.Sequential(
            nn.Linear(16 + 64 + 6 * 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]
        rng = torch.sigmoid(self.random_state).unsqueeze(0).expand(batch, -1)
        handler_input = torch.cat([syscall_bits, rng, arg_bits.view(batch, -1)], dim=-1)
        output = self.handler(handler_input)
        return output[:, :64], torch.sigmoid(output[:, 64])


class TrulyNeuralSyscallHandlersV2(nn.Module):
    """
    Improved syscall handler suite - deeper router for better accuracy.
    """

    def __init__(self, key_dim=128):
        super().__init__()
        self.key_dim = key_dim
        self.num_subsystems = 8

        # Subsystem handlers
        self.proc_handler = NeuralProcHandler(key_dim=key_dim)
        self.mem_handler = NeuralMemHandler(key_dim=key_dim)
        self.io_handler = NeuralIOHandler(key_dim=key_dim)
        self.fs_handler = NeuralFSHandler(key_dim=key_dim)
        self.time_handler = NeuralTimeHandler(key_dim=key_dim)
        self.misc_handler = NeuralMiscHandler(key_dim=key_dim)

        # DEEPER router for better syscall→subsystem mapping
        self.router = nn.Sequential(
            nn.Linear(16, key_dim),  # Only use syscall bits for routing
            nn.GELU(),
            nn.Linear(key_dim, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, key_dim // 2),
            nn.GELU(),
            nn.Linear(key_dim // 2, self.num_subsystems),
        )

        # Result combiner
        self.combiner = nn.Sequential(
            nn.Linear(64 * 6 + self.num_subsystems, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),
        )

    def handle(self, syscall_bits, arg_bits):
        batch = syscall_bits.shape[0]

        # Route based on syscall number only (not args)
        route_logits = self.router(syscall_bits)
        route_probs = F.softmax(route_logits, dim=-1)

        # Call all handlers
        proc_result, _ = self.proc_handler.handle(syscall_bits, arg_bits)
        mem_result, _ = self.mem_handler.handle(syscall_bits, arg_bits)
        io_result, _ = self.io_handler.handle(syscall_bits, arg_bits)
        fs_result, _ = self.fs_handler.handle(syscall_bits, arg_bits)
        time_result, _ = self.time_handler.handle(syscall_bits, arg_bits)
        misc_result, _ = self.misc_handler.handle(syscall_bits, arg_bits)

        # Stack results
        all_results = torch.stack([
            proc_result, mem_result, io_result,
            fs_result, torch.zeros_like(proc_result),  # NET placeholder
            torch.zeros_like(proc_result),  # SEC placeholder
            time_result, misc_result
        ], dim=1)

        # Weight by routing
        weighted_result = torch.einsum('bs,bsd->bd', route_probs, all_results)

        # Combine
        all_results_flat = all_results[:, :6].reshape(batch, -1)
        combiner_input = torch.cat([all_results_flat, route_probs], dim=-1)
        final_output = self.combiner(combiner_input)

        return final_output[:, :64], torch.sigmoid(final_output[:, 64]), route_probs


# Expanded syscall map for better coverage
SYSCALL_MAP = {
    # PROC (0)
    93: (0, 'exit'), 94: (0, 'exit_group'), 172: (0, 'getpid'),
    173: (0, 'getppid'), 174: (0, 'getuid'), 175: (0, 'geteuid'),
    # MEM (1)
    214: (1, 'brk'), 222: (1, 'mmap'), 215: (1, 'munmap'),
    226: (1, 'mprotect'),
    # IO (2)
    63: (2, 'read'), 64: (2, 'write'), 62: (2, 'lseek'),
    66: (2, 'writev'), 65: (2, 'readv'),
    # FS (3)
    56: (3, 'openat'), 57: (3, 'close'), 80: (3, 'fstat'),
    79: (3, 'fstatat'), 48: (3, 'faccessat'),
    # TIME (6)
    113: (6, 'clock_gettime'), 101: (6, 'nanosleep'),
    169: (6, 'gettimeofday'),
    # MISC (7)
    278: (7, 'getrandom'), 29: (7, 'ioctl'),
    98: (7, 'futex'),
}


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def generate_batch(batch_size, device):
    """Generate training batch with balanced sampling."""
    syscalls = []
    args = []
    expected_subsystems = []
    expected_results = []

    # Group syscalls by subsystem for balanced sampling
    subsys_syscalls = {}
    for num, (subsys, name) in SYSCALL_MAP.items():
        if subsys not in subsys_syscalls:
            subsys_syscalls[subsys] = []
        subsys_syscalls[subsys].append((num, name))

    for _ in range(batch_size):
        # Balance across subsystems
        subsys = random.choice(list(subsys_syscalls.keys()))
        syscall_num, name = random.choice(subsys_syscalls[subsys])

        syscalls.append(int_to_bits(syscall_num, 16))
        args.append(torch.stack([int_to_bits(random.randint(0, 1000)) for _ in range(6)]))
        expected_subsystems.append(subsys)

        # Expected result
        if name == 'getpid':
            expected_results.append(int_to_bits(1))
        elif name == 'brk':
            expected_results.append(int_to_bits(0x10000000))
        elif name in ('read', 'write'):
            expected_results.append(int_to_bits(random.randint(0, 100)))
        elif name == 'openat':
            expected_results.append(int_to_bits(3))
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
    print("TRULY NEURAL SYSCALL HANDLERS V2 TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("Improved version targeting 99%+ accuracy!")

    model = TrulyNeuralSyscallHandlersV2(key_dim=128).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 256

    for epoch in range(200):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            syscalls, args, exp_subsys, exp_results = generate_batch(batch_size, device)

            optimizer.zero_grad()

            result, success, route_probs = model.handle(syscalls, args)

            # Routing loss (primary)
            loss_route = F.cross_entropy(route_probs, exp_subsys)

            # Result loss
            loss_result = F.mse_loss(result, exp_results)

            loss = loss_route + 0.3 * loss_result
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
                syscalls, args, exp_subsys, _ = generate_batch(200, device)
                _, _, route_probs = model.handle(syscalls, args)

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
                "op_name": "TRULY_NEURAL_SYSCALL_HANDLERS_V2",
            }, "models/final/truly_neural_syscall_handlers_v2_best.pt")
            print(f"  Saved (route_acc={100*route_acc:.1f}%)")

        if route_acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest routing accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()
    test_syscalls = [
        (93, 0, 'exit→PROC'),
        (214, 1, 'brk→MEM'),
        (63, 2, 'read→IO'),
        (56, 3, 'openat→FS'),
        (113, 6, 'clock_gettime→TIME'),
        (278, 7, 'getrandom→MISC'),
    ]

    with torch.no_grad():
        all_pass = True
        for syscall_num, expected_subsys, name in test_syscalls:
            syscall_bits = int_to_bits(syscall_num, 16).unsqueeze(0).to(device)
            arg_bits = torch.zeros(1, 6, 64, device=device)

            _, _, route_probs = model.handle(syscall_bits, arg_bits)
            pred_subsys = route_probs.argmax(dim=-1).item()

            status = "✓" if pred_subsys == expected_subsys else f"✗ got {pred_subsys}"
            if pred_subsys != expected_subsys:
                all_pass = False

            print(f"  {name}: {status}")

        if all_pass:
            print("\n✓ ALL VERIFICATION TESTS PASSED!")
        else:
            print("\n✗ Some verification tests failed")


if __name__ == "__main__":
    train()
