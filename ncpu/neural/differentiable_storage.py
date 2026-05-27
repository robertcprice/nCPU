"""Differentiable Storage: memory and registers that support gradient flow.

This module provides soft-addressable storage for end-to-end differentiable
program execution. Instead of hard integer indices, reads and writes use
soft attention weights over all locations, enabling gradient-based learning
of WHICH register/address to access.

Key insight: in standard CPUs, register selection (e.g., "read R5") is a
discrete, non-differentiable operation. By replacing hard indexing with
soft attention, the optimizer can learn register allocation via gradients.

Three components:
  1. DifferentiableRegisterFile: soft-addressed register bank with attention
  2. DifferentiableRAM: Neural Turing Machine-style read/write memory
  3. DifferentiableStorageSystem: combined registers + RAM with a controller

These are designed to plug into ncpu/differentiable/ for program synthesis.
The existing DifferentiableEngine uses _soft_read/_soft_write internally;
this module provides a standalone, composable version with richer features.

Integration:
    from ncpu.neural.differentiable_storage import (
        DifferentiableRegisterFile,
        DifferentiableRAM,
        DifferentiableStorageSystem,
    )

    # Direct use
    regs = DifferentiableRegisterFile(n_regs=8, word_size=32)
    query = torch.randn(8, requires_grad=True)  # soft index
    value = regs.soft_read(query)  # differentiable!
    loss = (value - target).pow(2).sum()
    loss.backward()  # gradients flow to query AND register contents

    # Program synthesis with learned register allocation
    system = DifferentiableStorageSystem(n_regs=8, mem_size=64, word_size=16)
    # ... optimizer learns which registers to read/write ...
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Differentiable Register File
# ---------------------------------------------------------------------------


class DifferentiableRegisterFile(nn.Module):
    """Register file where reads and writes are differentiable.

    Uses soft attention over register indices instead of hard indexing.
    write(soft_idx, value): distributes the value across registers weighted
                            by soft_idx attention
    read(soft_idx): returns weighted sum of all registers by soft_idx

    This enables gradient-based program synthesis: the optimizer can learn
    WHICH register to read/write by adjusting the soft index logits.

    Parameters:
        n_regs: Number of registers (default 32, matching ARM64).
        word_size: Bits per register (default 32).
    """

    def __init__(self, n_regs: int = 32, word_size: int = 32):
        super().__init__()
        self.n_regs = n_regs
        self.word_size = word_size
        # Register bank: learnable initial state
        self.bank = nn.Parameter(
            torch.zeros(n_regs, word_size), requires_grad=True
        )

    def soft_read(self, query_logits: torch.Tensor) -> torch.Tensor:
        """Read with soft attention over all registers.

        Args:
            query_logits: (n_regs,) raw logits for register selection.
                The softmax converts these to attention weights.

        Returns:
            (word_size,) weighted sum of register contents.
        """
        attn = F.softmax(query_logits, dim=0)  # (n_regs,)
        return (attn.unsqueeze(1) * self.bank).sum(0)  # (word_size,)

    def soft_write(
        self,
        query_logits: torch.Tensor,
        value: torch.Tensor,
        write_strength: Optional[torch.Tensor] = None,
    ) -> None:
        """Write with soft attention, distributing value across registers.

        The write is a soft blend: each register gets a weighted mix of
        its old value and the new value, where the weight is determined
        by the attention distribution.

        Args:
            query_logits: (n_regs,) raw logits for register selection.
            value: (word_size,) value to write.
            write_strength: Optional scalar in [0,1] controlling how much
                the write actually happens (0=no write, 1=full write).
                Useful for conditional writes.
        """
        attn = F.softmax(query_logits, dim=0)  # (n_regs,)
        if write_strength is not None:
            attn = attn * write_strength

        # Soft write: blend old and new values weighted by attention
        new_bank = (
            self.bank * (1 - attn.unsqueeze(1))
            + value.unsqueeze(0) * attn.unsqueeze(1)
        )
        self.bank = nn.Parameter(new_bank, requires_grad=True)

    def hard_read(self, idx: int) -> torch.Tensor:
        """Read a specific register (hard indexing, still differentiable w.r.t. content)."""
        return self.bank[idx]

    def hard_write(self, idx: int, value: torch.Tensor) -> None:
        """Write a specific register (hard indexing)."""
        new_bank = self.bank.clone()
        new_bank[idx] = value
        self.bank = nn.Parameter(new_bank, requires_grad=True)

    def reset(self) -> None:
        """Zero all registers."""
        self.bank = nn.Parameter(
            torch.zeros_like(self.bank), requires_grad=True
        )

    @property
    def state(self) -> torch.Tensor:
        """Current register bank as a detached snapshot."""
        return self.bank.detach().clone()


# ---------------------------------------------------------------------------
# Differentiable RAM (Neural Turing Machine style)
# ---------------------------------------------------------------------------


class DifferentiableRAM(nn.Module):
    """RAM with differentiable read/write via soft addressing.

    Implements the Neural Turing Machine memory mechanism:
    - Content-based addressing: find location by matching a key
    - Location-based addressing: shift focus to adjacent locations
    - Erase + add write: controlled blending of old and new content

    All operations produce gradients for end-to-end training.

    Parameters:
        size: Number of memory words.
        word_size: Bits per word.
    """

    def __init__(self, size: int = 256, word_size: int = 32):
        super().__init__()
        self.size = size
        self.word_size = word_size
        self.memory = nn.Parameter(
            torch.zeros(size, word_size), requires_grad=True
        )

    def content_address(
        self,
        key: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Content-based addressing: find memory locations matching key.

        Computes cosine similarity between key and all memory words,
        then applies softmax with sharpness beta to produce attention weights.

        Args:
            key: (word_size,) query vector.
            beta: Sharpness parameter (higher = sharper focus).

        Returns:
            (size,) attention weights over memory locations.
        """
        # Add epsilon to avoid division by zero in cosine similarity
        similarity = F.cosine_similarity(
            key.unsqueeze(0),
            self.memory + 1e-8,
            dim=1,
        )
        return F.softmax(beta * similarity, dim=0)

    def location_address(
        self,
        prev_weights: torch.Tensor,
        shift: torch.Tensor,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Location-based addressing: shift attention relative to previous.

        Applies a convolutional shift to the previous attention distribution,
        then sharpens with gamma.

        Args:
            prev_weights: (size,) previous attention weights.
            shift: (3,) shift kernel [left, stay, right] (softmax-normalized).
            gamma: Sharpening parameter (>= 1.0).

        Returns:
            (size,) new attention weights.
        """
        # Circular convolution for shift
        shift_normalized = F.softmax(shift, dim=0)
        # Pad for circular convolution
        padded = torch.cat([prev_weights[-1:], prev_weights, prev_weights[:1]])
        shifted = F.conv1d(
            padded.unsqueeze(0).unsqueeze(0),
            shift_normalized.unsqueeze(0).unsqueeze(0),
            padding=0,
        ).squeeze()

        # Sharpening
        if gamma > 1.0:
            sharp = shifted.pow(gamma)
            sharp = sharp / (sharp.sum() + 1e-8)
            return sharp
        return shifted

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        """Read using attention weights.

        Args:
            weights: (size,) attention distribution over locations.

        Returns:
            (word_size,) weighted sum of memory contents.
        """
        return (weights.unsqueeze(1) * self.memory).sum(0)

    def write(
        self,
        weights: torch.Tensor,
        erase_vector: torch.Tensor,
        add_vector: torch.Tensor,
    ) -> None:
        """NTM-style write: erase then add.

        The write operation is decomposed into two phases:
        1. Erase: multiply each location by (1 - w_i * e) where w_i is the
           attention weight and e is the erase vector
        2. Add: add w_i * a where a is the add vector

        This decomposition allows the network to learn both what to forget
        and what to remember.

        Args:
            weights: (size,) attention distribution.
            erase_vector: (word_size,) what to erase (sigmoid-bounded [0,1]).
            add_vector: (word_size,) what to add.
        """
        # Erase
        erase = weights.unsqueeze(1) * erase_vector.unsqueeze(0)
        new_memory = self.memory * (1 - erase)
        # Add
        add = weights.unsqueeze(1) * add_vector.unsqueeze(0)
        new_memory = new_memory + add
        self.memory = nn.Parameter(new_memory, requires_grad=True)

    def simple_write(
        self,
        weights: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Simplified write: soft blend old and new values.

        Easier to use than the full NTM erase+add when you just want
        to store a value at a soft address.

        Args:
            weights: (size,) attention distribution.
            value: (word_size,) value to write.
        """
        w = weights.unsqueeze(1)
        new_memory = self.memory * (1 - w) + value.unsqueeze(0) * w
        self.memory = nn.Parameter(new_memory, requires_grad=True)

    def reset(self) -> None:
        """Zero all memory."""
        self.memory = nn.Parameter(
            torch.zeros_like(self.memory), requires_grad=True
        )


# ---------------------------------------------------------------------------
# Combined Storage System
# ---------------------------------------------------------------------------


class DifferentiableStorageSystem(nn.Module):
    """Combined register file + RAM with a learned controller.

    The controller is a small MLP that takes the current register state
    and produces addressing parameters for both the register file and RAM.
    This enables end-to-end learning of:
    - Which register to read/write
    - Which RAM address to access
    - What value to store
    - When to use registers vs RAM

    This is the bridge between ncpu/neural/differentiable_storage.py and
    the existing ncpu/differentiable/ program synthesis pipeline.

    Parameters:
        n_regs: Number of registers.
        mem_size: Number of RAM words.
        word_size: Bits per word.
        controller_hidden: Hidden size of the controller MLP.
    """

    def __init__(
        self,
        n_regs: int = 8,
        mem_size: int = 64,
        word_size: int = 16,
        controller_hidden: int = 64,
    ):
        super().__init__()
        self.registers = DifferentiableRegisterFile(n_regs, word_size)
        self.ram = DifferentiableRAM(mem_size, word_size)
        self.n_regs = n_regs
        self.mem_size = mem_size
        self.word_size = word_size

        # Controller: register state -> addressing parameters
        input_dim = n_regs * word_size
        self.controller = nn.Sequential(
            nn.Linear(input_dim, controller_hidden),
            nn.GELU(),
            nn.Linear(controller_hidden, controller_hidden),
            nn.GELU(),
        )

        # Output heads
        self.reg_read_head = nn.Linear(controller_hidden, n_regs)
        self.reg_write_head = nn.Linear(controller_hidden, n_regs)
        self.ram_key_head = nn.Linear(controller_hidden, word_size)
        self.ram_erase_head = nn.Linear(controller_hidden, word_size)
        self.ram_add_head = nn.Linear(controller_hidden, word_size)
        self.value_transform = nn.Linear(controller_hidden, word_size)

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one controller step.

        1. Read the current register state
        2. Pass through controller MLP
        3. Produce read/write addresses and values
        4. Execute the reads and writes

        Returns:
            (reg_read_value, ram_read_value): values read from registers
            and RAM in this step.
        """
        # Flatten register state as controller input
        reg_state = self.registers.bank.reshape(-1)
        hidden = self.controller(reg_state)

        # Register read
        reg_read_logits = self.reg_read_head(hidden)
        reg_value = self.registers.soft_read(reg_read_logits)

        # RAM read (content-based)
        ram_key = self.ram_key_head(hidden)
        ram_weights = self.ram.content_address(ram_key, beta=5.0)
        ram_value = self.ram.read(ram_weights)

        # Compute new value to write (transform of what we read)
        combined = self.value_transform(hidden)

        # Register write
        reg_write_logits = self.reg_write_head(hidden)
        self.registers.soft_write(reg_write_logits, combined)

        # RAM write
        ram_erase = torch.sigmoid(self.ram_erase_head(hidden))
        ram_add = self.ram_add_head(hidden)
        self.ram.write(ram_weights, ram_erase, ram_add)

        return reg_value, ram_value

    def reset(self) -> None:
        """Reset all storage to zeros."""
        self.registers.reset()
        self.ram.reset()


# ---------------------------------------------------------------------------
# Training demo: learn to copy R0 -> Rdst via gradient descent
# ---------------------------------------------------------------------------


def train_register_copy(
    n_regs: int = 8,
    word_size: int = 16,
    src_reg: int = 0,
    dst_reg: int = 5,
    n_steps: int = 300,
    lr: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Train soft register indices to learn a copy R_src -> R_dst.

    This demonstrates the core capability: the optimizer discovers which
    register to read and which to write through gradient descent alone.

    We set up:
    - A register file with a known value in R_src
    - Learnable query logits for read and write operations
    - Loss = MSE between R_dst and the target value

    The optimizer must learn:
    - read_logits that focus on R_src
    - write_logits that focus on R_dst

    All operations use pure tensor arithmetic (no nn.Parameter reassignment)
    to maintain the computation graph through the full read-write-read cycle.

    Returns dict with training metrics.
    """
    # Put a target value in R_src
    target_value = torch.randn(word_size)

    # Learnable soft indices
    read_logits = nn.Parameter(torch.randn(n_regs) * 0.1)
    write_logits = nn.Parameter(torch.randn(n_regs) * 0.1)

    optimizer = torch.optim.Adam([read_logits, write_logits], lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Initialize register bank as a plain tensor (not nn.Parameter)
        # with the target value in R_src. This stays in the computation
        # graph without needing Parameter reassignment.
        bank = torch.zeros(n_regs, word_size)
        bank[src_reg] = target_value

        # Soft read: weighted sum of register contents
        read_attn = F.softmax(read_logits, dim=0)   # (n_regs,)
        value = (read_attn.unsqueeze(1) * bank).sum(0)  # (word_size,)

        # Soft write: blend old bank with new value
        write_attn = F.softmax(write_logits, dim=0)  # (n_regs,)
        bank_after = (
            bank * (1 - write_attn.unsqueeze(1))
            + value.unsqueeze(0) * write_attn.unsqueeze(1)
        )

        # Loss: how close is R_dst to the target?
        result = bank_after[dst_reg]
        loss = F.mse_loss(result, target_value)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            ra = F.softmax(read_logits, dim=0).detach()
            wa = F.softmax(write_logits, dim=0).detach()
            rf = int(ra.argmax().item())
            wf = int(wa.argmax().item())
            print(
                f"  Step {step:4d}  loss={loss.item():.6f}  "
                f"read_focus=R{rf} ({ra[rf]:.3f})  "
                f"write_focus=R{wf} ({wa[wf]:.3f})"
            )

    # Final check
    read_attn_final = F.softmax(read_logits, dim=0).detach()
    write_attn_final = F.softmax(write_logits, dim=0).detach()
    learned_src = int(read_attn_final.argmax().item())
    learned_dst = int(write_attn_final.argmax().item())
    correct = learned_src == src_reg and learned_dst == dst_reg

    results = {
        "learned_src": learned_src,
        "learned_dst": learned_dst,
        "expected_src": src_reg,
        "expected_dst": dst_reg,
        "correct": correct,
        "final_loss": losses[-1],
        "steps": n_steps,
        "losses": losses,
        "read_attention": read_attn_final.tolist(),
        "write_attention": write_attn_final.tolist(),
    }

    if verbose:
        status = "CORRECT" if correct else "WRONG"
        print(
            f"\n  Result: learned R{learned_src}->R{learned_dst} "
            f"(expected R{src_reg}->R{dst_reg}) [{status}]"
        )
        print(f"  Final loss: {losses[-1]:.6f}")

    return results


def train_memory_copy(
    mem_size: int = 16,
    word_size: int = 8,
    n_steps: int = 300,
    lr: float = 0.02,
    verbose: bool = True,
) -> dict:
    """Train soft addressing to learn a memory copy operation.

    Write a value to one address, then learn to read it back via
    content-based addressing.

    This demonstrates the Neural Turing Machine's content-addressing:
    the optimizer learns a key that retrieves the stored value.
    """
    ram = DifferentiableRAM(size=mem_size, word_size=word_size)

    target_value = torch.randn(word_size)

    # Write to a known address (hard)
    write_addr = 7
    write_weights = torch.zeros(mem_size)
    write_weights[write_addr] = 1.0
    ram.simple_write(write_weights, target_value)

    # Learn a content-based key to retrieve this value
    key = nn.Parameter(torch.randn(word_size) * 0.1)
    beta = nn.Parameter(torch.tensor(1.0))

    optimizer = torch.optim.Adam([key, beta], lr=lr)
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Content-based read
        weights = ram.content_address(key, beta=F.softplus(beta))
        value = ram.read(weights)

        loss = F.mse_loss(value, target_value)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (step % 75 == 0 or step == n_steps - 1):
            focus_addr = int(weights.detach().argmax().item())
            focus_weight = float(weights.detach().max().item())
            print(
                f"  Step {step:4d}  loss={loss.item():.6f}  "
                f"focus=addr[{focus_addr}] ({focus_weight:.3f})  "
                f"beta={F.softplus(beta).item():.2f}"
            )

    final_weights = ram.content_address(key.detach(), beta=F.softplus(beta.detach()))
    focus_addr = int(final_weights.argmax().item())

    results = {
        "target_addr": write_addr,
        "learned_addr": focus_addr,
        "correct": focus_addr == write_addr,
        "final_loss": losses[-1],
        "steps": n_steps,
        "losses": losses,
    }

    if verbose:
        status = "CORRECT" if results["correct"] else "WRONG"
        print(
            f"\n  Result: learned addr[{focus_addr}] "
            f"(expected addr[{write_addr}]) [{status}]"
        )

    return results
