#!/usr/bin/env python3
"""
NEURAL SYSCALL ROUTER
======================
Routes Linux syscalls to appropriate subsystems.

Instead of a hardcoded switch-case, this neural network:
1. Takes syscall number and arguments
2. Routes to the correct subsystem (PROC, MEM, IO, FS, NET, SEC)
3. Predicts the expected response type

This is KVRM_ORCH from the micro-KVRM architecture!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


# Linux ARM64 syscall categories
SYSCALL_SUBSYSTEMS = {
    # PROC - Process management
    'PROC': [93, 94, 172, 174, 175, 176, 177, 178, 220, 221, 222, 260, 435],  # exit, getpid, getuid, clone, fork, etc.

    # MEM - Memory management
    'MEM': [214, 215, 222, 226, 227, 228, 229, 230, 231, 232, 233, 234],  # brk, mmap, mprotect, munmap, etc.

    # IO - I/O operations
    'IO': [63, 64, 65, 66, 67, 68, 62, 17, 23, 24, 25, 29],  # read, write, readv, writev, lseek, ioctl, etc.

    # FS - Filesystem
    'FS': [56, 57, 48, 49, 79, 80, 78, 34, 35, 83, 84, 85, 86, 87, 88],  # openat, close, faccessat, fstat, mkdir, etc.

    # NET - Networking
    'NET': [198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],  # socket, bind, listen, accept, etc.

    # SEC - Security/Permissions
    'SEC': [146, 147, 148, 149, 150, 151, 152, 153, 160, 161],  # setuid, setgid, setgroups, setsid, etc.

    # TIME - Time-related
    'TIME': [101, 113, 114, 115, 169, 170],  # nanosleep, clock_gettime, clock_settime, etc.

    # MISC - Miscellaneous
    'MISC': [278, 279, 280, 281, 282, 283, 284],  # getrandom, etc.
}

SUBSYSTEM_TO_IDX = {name: idx for idx, name in enumerate(SYSCALL_SUBSYSTEMS.keys())}
NUM_SUBSYSTEMS = len(SYSCALL_SUBSYSTEMS)

# Build reverse mapping
SYSCALL_TO_SUBSYSTEM = {}
for subsystem, syscalls in SYSCALL_SUBSYSTEMS.items():
    for sc in syscalls:
        SYSCALL_TO_SUBSYSTEM[sc] = subsystem


class NeuralSyscallRouter(nn.Module):
    """
    Neural syscall router (KVRM_ORCH).

    Routes syscalls to the appropriate subsystem based on:
    - Syscall number
    - Arguments (for context-dependent routing)
    """

    def __init__(self, max_syscall=512, num_args=6, d_model=128):
        super().__init__()
        self.max_syscall = max_syscall
        self.num_subsystems = NUM_SUBSYSTEMS

        # Syscall embedding
        self.syscall_embed = nn.Embedding(max_syscall, d_model)

        # Argument encoding (each arg is 64-bit, we encode as features)
        self.arg_encoder = nn.Sequential(
            nn.Linear(num_args * 64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Combined reasoning
        self.router = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, NUM_SUBSYSTEMS)
        )

        # Response type predictor
        self.response_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)  # success, error, blocked, special
        )

    def forward(self, syscall_num, arg_bits):
        """
        Args:
            syscall_num: [batch] - syscall numbers
            arg_bits: [batch, 6, 64] - 6 arguments as bits

        Returns:
            subsystem_logits: [batch, num_subsystems]
            response_logits: [batch, 4]
        """
        batch = syscall_num.shape[0]

        # Embed syscall
        sc_emb = self.syscall_embed(syscall_num.clamp(0, self.max_syscall - 1))

        # Encode arguments
        arg_flat = arg_bits.view(batch, -1)  # [batch, 6*64]
        arg_emb = self.arg_encoder(arg_flat)

        # Combine
        combined = torch.cat([sc_emb, arg_emb], dim=-1)

        # Route
        subsystem_logits = self.router(combined)
        response_logits = self.response_head(combined)

        return subsystem_logits, response_logits


class NeuralSyscallExecutor(nn.Module):
    """
    Per-subsystem syscall executor.

    Each subsystem has its own small model that handles
    the actual syscall execution.
    """

    def __init__(self, subsystem_name, d_model=64):
        super().__init__()
        self.subsystem = subsystem_name

        # Syscall-specific processing
        self.process = nn.Sequential(
            nn.Linear(64 + 6 * 64, d_model * 2),  # syscall one-hot + args
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

        # Result head (return value as bits)
        self.result_head = nn.Linear(d_model, 64)

        # Error code head
        self.error_head = nn.Linear(d_model, 16)  # Common error codes

    def forward(self, syscall_one_hot, arg_bits):
        """Execute syscall within this subsystem."""
        batch = arg_bits.shape[0]
        arg_flat = arg_bits.view(batch, -1)

        x = torch.cat([syscall_one_hot, arg_flat], dim=-1)
        h = self.process(x)

        result = torch.sigmoid(self.result_head(h))
        error = self.error_head(h)

        return result, error


def int_to_bits(val, bits=64):
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def generate_batch(batch_size, device):
    """Generate training batch of syscalls."""
    syscalls = []
    args = []
    targets = []
    responses = []

    for _ in range(batch_size):
        # Pick random subsystem, then random syscall from it
        subsystem = random.choice(list(SYSCALL_SUBSYSTEMS.keys()))
        syscall_list = SYSCALL_SUBSYSTEMS[subsystem]
        syscall = random.choice(syscall_list)

        # Random arguments
        arg_list = []
        for _ in range(6):
            if random.random() < 0.3:
                arg = 0
            elif random.random() < 0.6:
                arg = random.randint(0, 1000)
            else:
                arg = random.randint(0, 1 << 32)
            arg_list.append(int_to_bits(arg, 64))

        syscalls.append(syscall)
        args.append(torch.stack(arg_list))
        targets.append(SUBSYSTEM_TO_IDX[subsystem])

        # Random response type (mostly success)
        if random.random() < 0.8:
            responses.append(0)  # success
        elif random.random() < 0.5:
            responses.append(1)  # error
        else:
            responses.append(2)  # blocked

    return (
        torch.tensor(syscalls, device=device),
        torch.stack(args).to(device),
        torch.tensor(targets, device=device),
        torch.tensor(responses, device=device)
    )


def train():
    print("=" * 60)
    print("NEURAL SYSCALL ROUTER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Subsystems: {list(SYSCALL_SUBSYSTEMS.keys())}")

    # Create model
    model = NeuralSyscallRouter(max_syscall=512, num_args=6, d_model=128).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 512

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(100):
            syscalls, args, targets, responses = generate_batch(batch_size, device)

            optimizer.zero_grad()
            subsystem_logits, response_logits = model(syscalls, args)

            loss_subsystem = F.cross_entropy(subsystem_logits, targets)
            loss_response = F.cross_entropy(response_logits, responses)
            loss = loss_subsystem + 0.5 * loss_response

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for _ in range(20):
                syscalls, args, targets, responses = generate_batch(256, device)
                subsystem_logits, _ = model(syscalls, args)
                preds = subsystem_logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += len(targets)

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/100:.4f} acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": acc,
                "op_name": "SYSCALL_ROUTER",
                "architecture": "NeuralSyscallRouter",
                "subsystems": list(SYSCALL_SUBSYSTEMS.keys())
            }, "models/final/neural_syscall_router_best.pt")
            print(f"  Saved (acc={100*acc:.1f}%)")

        if acc >= 0.99:
            print("99%+ ACCURACY!")
            break

    print(f"\nBest accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification:")
    model.eval()

    test_syscalls = [
        (63, 'IO'),      # read
        (64, 'IO'),      # write
        (93, 'PROC'),    # exit
        (214, 'MEM'),    # brk
        (222, 'MEM'),    # mmap
        (56, 'FS'),      # openat
        (57, 'FS'),      # close
        (278, 'MISC'),   # getrandom
    ]

    subsystem_names = list(SYSCALL_SUBSYSTEMS.keys())

    with torch.no_grad():
        for sc, expected_sub in test_syscalls:
            sc_t = torch.tensor([sc], device=device)
            args_t = torch.zeros(1, 6, 64, device=device)
            logits, _ = model(sc_t, args_t)
            pred_idx = logits.argmax(dim=-1).item()
            pred_sub = subsystem_names[pred_idx]

            status = "OK" if pred_sub == expected_sub else f"GOT {pred_sub}"
            print(f"  syscall {sc} → {expected_sub}: {status}")


if __name__ == "__main__":
    train()
