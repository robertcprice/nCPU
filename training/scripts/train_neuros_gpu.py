#!/usr/bin/env python3
"""Comprehensive GPU training script for nCPU neural assembler and compiler.

Trains three neural networks to replace classical assembler/compiler components:

1. NeuralCodeGenNet  -- Maps 6 instruction features -> 32-bit binary encoding
                        Target: 90%+ exact-match accuracy against ClassicalAssembler
2. NeuralTokenizerNet -- CNN char classifier for assembly source tokenization
3. PeepholeOptimizerNet -- Classifies 3-instruction windows for peephole opts

The key insight: the codegen's encoding is deterministic (opcode/regs/imm -> 32-bit word),
so we can exhaustively enumerate ALL valid instruction encodings to build a training set
of thousands of examples, instead of relying on ~64 examples from 7 programs.

Works on CUDA (vast.ai), MPS (Apple Silicon), or CPU.

Usage:
    # Local (Apple Silicon):
    python training/train_neuros_gpu.py

    # vast.ai GPU:
    python training/train_neuros_gpu.py --epochs-codegen 3000 --epochs-tokenizer 800

    # Quick smoke test:
    python training/train_neuros_gpu.py --quick
"""

import sys
import os
import time
import random
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ncpu.os.assembler import (
    ClassicalAssembler, NeuralAssembler, NeuralCodeGenNet, NeuralTokenizerNet,
    Opcode, MNEMONIC_MAP, REGISTERS, AsmInstruction,
    FMT_NONE, FMT_REG_IMM, FMT_REG_REG, FMT_3REG, FMT_REG, FMT_2REG,
    FMT_ADDR, FMT_SHIFT, encode_instruction_features,
)
from ncpu.os.compiler import NeuralCompiler, PeepholeOptimizerNet


# =============================================================================
# Device Selection
# =============================================================================

def select_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] CUDA: {name} ({mem:.1f} GB)")
        return dev
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS")
        return torch.device("mps")
    else:
        print("[Device] CPU (no GPU detected)")
        return torch.device("cpu")


# =============================================================================
# Data Generation: Exhaustive Instruction Enumeration
# =============================================================================

def generate_exhaustive_training_data(
    device: torch.device,
    num_immediates: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Generate ALL valid instruction encodings as training data.

    Enumerates every opcode x register-combination x representative immediates
    to produce thousands of (feature_vector, target_bits) pairs that fully
    cover the nCPU encoding space.

    The encoding is deterministic:
        word = (opcode << 24) | (rd << 21) | (rs1 << 18) | (rs2 << 15) | (imm & 0x7FFF)

    So we can generate perfect ground truth for every possible instruction.

    Returns:
        features: [N, 6] float tensor of normalized instruction features
        targets:  [N, 32] float tensor of target bit patterns
        count:    number of training examples generated
    """
    assembler = ClassicalAssembler()
    all_features = []
    all_targets = []

    # Representative immediate values spanning the 15-bit signed range
    if num_immediates >= 32768:
        # Full coverage: ALL 32768 values (-16384 to 16383)
        imm_values = list(range(-16384, 16384))
        print(f"  [ALL IMM] Using all {len(imm_values)} immediate values for full generalization")
    else:
        # Include edge cases: 0, 1, -1, powers of 2, max/min, random samples
        imm_values = sorted(set([
            0, 1, 2, 3, 4, 5, 7, 8, 10, 15, 16, 31, 32, 42, 55, 63, 64,
            100, 127, 128, 255, 256, 511, 512, 1000, 1023, 1024, 2048,
            4096, 8191, 8192, 16383,  # max 15-bit unsigned
            -1, -2, -5, -10, -128, -256, -1000, -16384,  # negatives (sign-extended)
        ] + [random.randint(0, 16383) for _ in range(num_immediates)]
          + [random.randint(-16384, -1) for _ in range(num_immediates // 4)]))

    regs = list(range(8))  # R0-R7

    def add_example(opcode: int, fmt: int, rd: int = 0, rs1: int = 0,
                    rs2: int = 0, imm: int = 0):
        instr = AsmInstruction(opcode=opcode, fmt=fmt, rd=rd, rs1=rs1,
                               rs2=rs2, imm=imm)
        word = assembler._encode(instr)
        features = encode_instruction_features(
            instr.opcode, instr.rd, instr.rs1, instr.rs2,
            instr.imm, instr.fmt,
        )
        target = _int_to_bits(word)
        all_features.append(features)
        all_targets.append(target)

    # --- FMT_NONE: NOP, HALT (2 instructions) ---
    add_example(Opcode.NOP, FMT_NONE)
    add_example(Opcode.HALT, FMT_NONE)

    # --- FMT_REG_IMM: MOV Rd, imm (8 regs x N immediates) ---
    for rd in regs:
        for imm in imm_values:
            add_example(Opcode.MOV_IMM, FMT_REG_IMM, rd=rd, imm=imm)

    # --- FMT_REG_REG: MOV Rd, Rs (8 x 8 = 64) ---
    for rd in regs:
        for rs in regs:
            add_example(Opcode.MOV_REG, FMT_REG_REG, rd=rd, rs1=rs)

    # --- FMT_3REG: ADD/SUB/MUL/DIV/AND/OR/XOR (7 opcodes x 8^3 = 3584) ---
    three_reg_ops = [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                     Opcode.AND, Opcode.OR, Opcode.XOR]
    for op in three_reg_ops:
        for rd in regs:
            for rs1 in regs:
                for rs2 in regs:
                    add_example(op, FMT_3REG, rd=rd, rs1=rs1, rs2=rs2)

    # --- FMT_SHIFT: SHL/SHR with register src2 (2 x 8^3 = 1024) ---
    for op in [Opcode.SHL, Opcode.SHR]:
        for rd in regs:
            for rs1 in regs:
                for rs2 in regs:
                    add_example(op, FMT_SHIFT, rd=rd, rs1=rs1, rs2=rs2)

    # --- FMT_SHIFT: SHL/SHR with immediate amount (2 x 8 x 8 x shifts) ---
    shift_amounts = [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 24, 31]
    for op in [Opcode.SHL, Opcode.SHR]:
        for rd in regs:
            for rs1 in regs:
                for imm in shift_amounts:
                    add_example(op, FMT_SHIFT, rd=rd, rs1=rs1, imm=imm)

    # --- FMT_REG: INC/DEC (2 x 8 = 16) ---
    for op in [Opcode.INC, Opcode.DEC]:
        for rd in regs:
            add_example(op, FMT_REG, rd=rd)

    # --- FMT_2REG: CMP Rs1, Rs2 (8 x 8 = 64) ---
    for rs1 in regs:
        for rs2 in regs:
            add_example(Opcode.CMP, FMT_2REG, rs1=rs1, rs2=rs2)

    # --- FMT_ADDR: JMP/JZ/JNZ/JS/JNS with address (5 x N addresses) ---
    jump_ops = [Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.JS, Opcode.JNS]
    jump_targets = sorted(set([0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 100,
                               255, 1000, 8191, 16383]))
    for op in jump_ops:
        for addr in jump_targets:
            add_example(op, FMT_ADDR, imm=addr)

    features_tensor = torch.stack(all_features).to(device)
    targets_tensor = torch.stack(all_targets).to(device)
    count = len(all_features)

    print(f"[Data] Generated {count:,} exhaustive training examples")
    print(f"       Breakdown: NOP/HALT=2, MOV_IMM={8*len(imm_values)}, MOV_REG=64, "
          f"3REG={7*512}, SHIFT_REG={2*512}, SHIFT_IMM={2*8*8*len(shift_amounts)}, "
          f"INC/DEC=16, CMP=64, JMP={5*len(jump_targets)}")

    return features_tensor, targets_tensor, count


def generate_synthetic_programs(n: int = 200) -> List[str]:
    """Generate random valid assembly programs for augmentation.

    Creates programs by randomly combining instructions with valid operands.
    These supplement the real programs from the programs/ directory and
    provide additional instruction sequences the model may not have seen.

    Args:
        n: Number of synthetic programs to generate

    Returns:
        List of assembly source code strings
    """
    random.seed(42)
    programs = []

    # Instruction templates
    three_reg_mnemonics = ["ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR"]
    jump_mnemonics = ["JMP", "JZ", "JNZ", "JS", "JNS"]

    for prog_idx in range(n):
        lines = []
        num_instructions = random.randint(3, 20)
        label_counter = 0
        labels_defined = []

        for i in range(num_instructions):
            choice = random.random()

            if choice < 0.20:
                # MOV Rd, imm
                rd = random.randint(0, 7)
                imm = random.choice([0, 1, -1, random.randint(-100, 1000),
                                     random.randint(0, 16383)])
                lines.append(f"    MOV R{rd}, {imm}")
            elif choice < 0.35:
                # MOV Rd, Rs
                rd = random.randint(0, 7)
                rs = random.randint(0, 7)
                lines.append(f"    MOV R{rd}, R{rs}")
            elif choice < 0.55:
                # 3-reg ALU
                mnem = random.choice(three_reg_mnemonics)
                rd = random.randint(0, 7)
                rs1 = random.randint(0, 7)
                rs2 = random.randint(0, 7)
                lines.append(f"    {mnem} R{rd}, R{rs1}, R{rs2}")
            elif choice < 0.60:
                # SHL/SHR with immediate
                mnem = random.choice(["SHL", "SHR"])
                rd = random.randint(0, 7)
                rs = random.randint(0, 7)
                amt = random.randint(0, 31)
                lines.append(f"    {mnem} R{rd}, R{rs}, {amt}")
            elif choice < 0.65:
                # INC/DEC
                mnem = random.choice(["INC", "DEC"])
                rd = random.randint(0, 7)
                lines.append(f"    {mnem} R{rd}")
            elif choice < 0.75:
                # CMP
                rs1 = random.randint(0, 7)
                rs2 = random.randint(0, 7)
                lines.append(f"    CMP R{rs1}, R{rs2}")
            elif choice < 0.82:
                # Label definition
                label_name = f"label_{prog_idx}_{label_counter}"
                label_counter += 1
                labels_defined.append(label_name)
                lines.append(f"{label_name}:")
            elif choice < 0.92:
                # Jump to existing label
                if labels_defined:
                    mnem = random.choice(jump_mnemonics)
                    target = random.choice(labels_defined)
                    lines.append(f"    {mnem} {target}")
                else:
                    # Jump to numeric address
                    mnem = random.choice(jump_mnemonics)
                    addr = random.randint(0, 20)
                    lines.append(f"    {mnem} {addr}")
            elif choice < 0.96:
                # NOP
                lines.append("    NOP")
            else:
                # Standalone MOV with hex immediate
                rd = random.randint(0, 7)
                imm = random.randint(0, 255)
                lines.append(f"    MOV R{rd}, 0x{imm:02X}")

        lines.append("    HALT")
        programs.append("\n".join(lines))

    print(f"[Data] Generated {len(programs)} synthetic programs")
    return programs


def load_real_programs(programs_dir: Path) -> List[str]:
    """Load all .asm programs from the programs directory.

    Args:
        programs_dir: Path to the programs/ directory

    Returns:
        List of assembly source code strings
    """
    programs = []
    asm_files = sorted(programs_dir.rglob("*.asm"))
    for path in asm_files:
        programs.append(path.read_text())
    print(f"[Data] Loaded {len(programs)} real programs from {programs_dir}")
    return programs


# =============================================================================
# Utility Functions
# =============================================================================

def _int_to_bits(value: int) -> torch.Tensor:
    """Convert integer to 32-bit tensor (LSB-first)."""
    bits = torch.zeros(32, dtype=torch.float32)
    for i in range(32):
        bits[i] = (value >> i) & 1
    return bits


def _bits_to_int(bits: torch.Tensor) -> int:
    """Convert 32-bit tensor (LSB-first) to integer using int64 arithmetic."""
    # Must use .long() to avoid float32 precision loss above 2^24
    result = 0
    for i in range(32):
        if bits[i] > 0.5:
            result |= (1 << i)
    return result


def count_exact_matches(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 2048,
) -> Tuple[int, int]:
    """Count exact 32-bit word matches between model output and targets.

    Processes in batches to avoid OOM on large datasets.

    Returns:
        (exact_matches, total_examples)
    """
    model.eval()
    exact = 0
    total = features.shape[0]

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_f = features[start:end]
            batch_t = targets[start:end]
            logits = model(batch_f)
            preds = (torch.sigmoid(logits) > 0.5).float()
            # A row matches exactly when ALL 32 bits match
            row_match = (preds == batch_t).all(dim=1)
            exact += row_match.sum().item()

    return int(exact), total


# =============================================================================
# Training: Neural CodeGen
# =============================================================================

def train_codegen(
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    epochs: int = 2000,
    lr: float = 3e-3,
    batch_size: int = 512,
    hidden_dim: int = 256,
    patience: int = 200,
    target_accuracy: float = 0.95,
    val_split: float = 0.1,
) -> Tuple[NeuralCodeGenNet, Dict]:
    """Train the neural code generator to produce exact 32-bit encodings.

    Strategy for reaching 90%+ exact match (up from 7.8%):
    1. Exhaustive data: thousands of examples covering full encoding space
    2. Larger network: 256-dim hidden layers (up from 128)
    3. More epochs: 2000+ with early stopping
    4. Cosine annealing LR schedule with warm restarts
    5. Curriculum learning: train on simple instructions first
    6. Gradient clipping for stability

    Args:
        features:  [N, 6] normalized instruction features
        targets:   [N, 32] target bit patterns
        device:    compute device
        epochs:    maximum training epochs
        lr:        initial learning rate
        batch_size: training batch size
        hidden_dim: MLP hidden dimension
        patience:  epochs without improvement before stopping
        target_accuracy: stop when this exact-match rate is reached
        val_split: fraction of data held out for validation

    Returns:
        (trained_model, training_stats)
    """
    print("\n" + "=" * 70)
    print("TRAINING: Neural CodeGen (6 features -> 32-bit encoding)")
    print("=" * 70)

    N = features.shape[0]
    print(f"  Dataset:    {N:,} examples")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Val split:  {val_split:.0%}")

    # --- Validation split ---
    perm = torch.randperm(N, device=device)
    val_size = int(N * val_split)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_features = features[train_idx]
    train_targets = targets[train_idx]
    val_features = features[val_idx]
    val_targets = targets[val_idx]

    print(f"  Train set:  {len(train_idx):,}")
    print(f"  Val set:    {len(val_idx):,}")

    # --- Model ---
    model = NeuralCodeGenNet(hidden_dim=hidden_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-6
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # No curriculum learning — train on full dataset from epoch 0
    # With binary features, the network can learn all instruction types simultaneously

    # --- Training loop ---
    best_val_exact = 0.0
    best_model_state = None
    eval_interval = 10
    evals_without_improve = 0
    patience_evals = max(1, patience // eval_interval)  # Convert epoch patience to eval-step patience
    history = []
    t_start = time.time()

    print(f"\n  {'Epoch':>6} | {'Loss':>10} | {'Bit Acc':>8} | {'Train EM':>9} | "
          f"{'Val EM':>8} | {'LR':>10} | {'Phase':>8} | {'Time':>6}")
    print("  " + "-" * 82)

    for epoch in range(epochs):
        model.train()

        # Train on full dataset every epoch
        epoch_features = train_features
        epoch_targets = train_targets

        # Mini-batch training
        n_train = epoch_features.shape[0]
        perm_train = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm_train[start:end]
            batch_f = epoch_features[idx]
            batch_t = epoch_targets[idx]

            optimizer.zero_grad()
            logits = model(batch_f)
            loss = loss_fn(logits, batch_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)

        # --- Evaluation (every eval_interval epochs or at key checkpoints) ---
        if epoch % eval_interval == 0 or epoch == epochs - 1 or epoch < 5:
            model.eval()
            with torch.no_grad():
                # Bit accuracy on full training set
                logits_all = model(train_features)
                preds_all = (torch.sigmoid(logits_all) > 0.5).float()
                bit_acc = (preds_all == train_targets).float().mean().item()

                # Exact match on training set
                train_exact, train_total = count_exact_matches(
                    model, train_features, train_targets)
                train_em = train_exact / max(1, train_total)

                # Exact match on validation set
                val_exact, val_total = count_exact_matches(
                    model, val_features, val_targets)
                val_em = val_exact / max(1, val_total)

            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start

            print(f"  {epoch:6d} | {avg_loss:10.6f} | {bit_acc:7.4f} | "
                  f"{train_em:8.4f} | {val_em:7.4f} | {current_lr:10.2e} | "
                  f"{'Full':>8} | {elapsed:5.0f}s")

            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "bit_acc": bit_acc,
                "train_exact_match": train_em,
                "val_exact_match": val_em,
                "lr": current_lr,
            })

            # Track best model
            if val_em > best_val_exact:
                best_val_exact = val_em
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                evals_without_improve = 0
            else:
                evals_without_improve += 1

            # Early stopping
            if best_val_exact >= target_accuracy:
                print(f"\n  [EARLY STOP] Target accuracy {target_accuracy:.0%} reached at epoch {epoch}")
                break
            if evals_without_improve >= patience_evals:
                print(f"\n  [EARLY STOP] No improvement for {patience} epochs "
                      f"({evals_without_improve} evals) at epoch {epoch}")
                break

    # --- Load best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    # --- Final evaluation ---
    train_exact, train_total = count_exact_matches(model, train_features, train_targets)
    val_exact, val_total = count_exact_matches(model, val_features, val_targets)
    full_exact, full_total = count_exact_matches(model, features, targets)

    elapsed_total = time.time() - t_start

    print(f"\n  RESULTS:")
    print(f"    Training exact match: {train_exact:,}/{train_total:,} = {train_exact/max(1,train_total):.4f}")
    print(f"    Validation exact match: {val_exact:,}/{val_total:,} = {val_exact/max(1,val_total):.4f}")
    print(f"    Full dataset exact match: {full_exact:,}/{full_total:,} = {full_exact/max(1,full_total):.4f}")
    print(f"    Total training time: {elapsed_total:.1f}s")

    stats = {
        "train_exact_match": train_exact / max(1, train_total),
        "val_exact_match": val_exact / max(1, val_total),
        "full_exact_match": full_exact / max(1, full_total),
        "train_exact": train_exact,
        "val_exact": val_exact,
        "full_exact": full_exact,
        "total_examples": full_total,
        "epochs_trained": epoch + 1,
        "training_time_s": elapsed_total,
        "hidden_dim": hidden_dim,
        "history": history,
    }

    return model, stats


# =============================================================================
# Training: Neural Tokenizer
# =============================================================================

def generate_tokenizer_training_data(
    programs: List[str],
    device: torch.device,
    max_seq_len: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Generate character-level tokenization training data.

    For each character in each program, assigns a token class:
        0 = whitespace
        1 = mnemonic character
        2 = register character
        3 = immediate/number character
        4 = label_ref character
        5 = label_def character (label followed by colon)
        6 = comma
        7 = newline

    Uses the ClassicalAssembler's parsing as ground truth.

    Args:
        programs: list of assembly source strings
        device: compute device
        max_seq_len: maximum sequence length (truncate longer programs)

    Returns:
        chars: [N, max_seq_len] int tensor of ASCII codes
        labels: [N, max_seq_len] int tensor of token class labels
        count: number of examples
    """
    all_chars = []
    all_labels = []

    # Known token sets
    mnemonics = set(MNEMONIC_MAP.keys()) | {"MOV"}
    register_names = set(REGISTERS.keys())

    for source in programs:
        lines = source.split("\n")
        char_classes = []

        for line in lines:
            # Strip comments for classification
            clean = line.split(";")[0].split("#")[0]
            upper_clean = clean.upper().strip()

            # Classify each character position
            for i, ch in enumerate(clean):
                if ch == "\n" or i == len(clean) - 1:
                    char_classes.append(7)  # newline at end of each line
                    continue
                if ch in (" ", "\t"):
                    char_classes.append(0)
                elif ch == ",":
                    char_classes.append(6)
                elif ch == ":":
                    char_classes.append(5)  # part of label definition
                else:
                    # Need context: what token does this char belong to?
                    # Parse the line to determine
                    cls = _classify_char_in_line(clean, i, mnemonics, register_names)
                    char_classes.append(cls)

            char_classes.append(7)  # newline between lines

        # Convert source to ASCII codes
        full_text = "\n".join(line.split(";")[0].split("#")[0] for line in lines)
        ascii_codes = [min(ord(c), 127) for c in full_text]

        # Truncate or pad
        if len(ascii_codes) > max_seq_len:
            ascii_codes = ascii_codes[:max_seq_len]
            char_classes = char_classes[:max_seq_len]
        else:
            pad_len = max_seq_len - len(ascii_codes)
            ascii_codes.extend([0] * pad_len)
            char_classes.extend([0] * pad_len)

        # Ensure lengths match
        min_len = min(len(ascii_codes), len(char_classes), max_seq_len)
        ascii_codes = ascii_codes[:min_len] + [0] * (max_seq_len - min_len)
        char_classes = char_classes[:min_len] + [0] * (max_seq_len - min_len)

        all_chars.append(torch.tensor(ascii_codes, dtype=torch.long))
        all_labels.append(torch.tensor(char_classes, dtype=torch.long))

    chars = torch.stack(all_chars).to(device)
    labels = torch.stack(all_labels).to(device)
    print(f"[Tokenizer] Generated {len(all_chars)} sequences, max_len={max_seq_len}")
    return chars, labels, len(all_chars)


def _classify_char_in_line(
    line: str,
    pos: int,
    mnemonics: set,
    register_names: set,
) -> int:
    """Classify a single character in context of its assembly line.

    Returns token class (0-7) based on what token the character belongs to.
    """
    upper_line = line.upper().strip()

    # Extract the "word" this character belongs to
    # Find word boundaries
    start = pos
    while start > 0 and line[start - 1] not in (" ", "\t", ",", ":"):
        start -= 1
    end = pos + 1
    while end < len(line) and line[end] not in (" ", "\t", ",", ":"):
        end += 1

    word = line[start:end].strip().upper()

    # Label definition: line ending with ':'
    stripped = line.strip()
    if stripped.endswith(":") and not stripped.startswith((" ", "\t")):
        colon_pos = stripped.index(":")
        if pos < len(line) and pos <= line.index(":") if ":" in line else False:
            return 5  # label_def

    # Check if this word is a mnemonic
    if word in mnemonics:
        return 1  # mnemonic

    # Check if this word is a register
    if word in register_names:
        return 2  # register

    # Check if this word looks like a number (decimal, hex, binary)
    clean_word = word.lstrip("-")
    if clean_word.isdigit() or clean_word.startswith("0X") or clean_word.startswith("0B"):
        return 3  # immediate

    # Check for label reference (in jump targets)
    parts = upper_line.split()
    if len(parts) >= 2 and parts[0] in ("JMP", "JZ", "JNZ", "JS", "JNS"):
        target = parts[-1].strip()
        if word == target and not target.isdigit():
            return 4  # label_ref

    # Default to whitespace if unclassifiable
    return 0


def train_tokenizer(
    chars: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 500,
    lr: float = 2e-3,
    batch_size: int = 32,
) -> Tuple[NeuralTokenizerNet, Dict]:
    """Train the CNN character-level tokenizer.

    The tokenizer classifies each character in assembly source as belonging
    to one of 8 token types. Uses 1D convolutions for local context.

    Args:
        chars:  [N, seq_len] int tensor of ASCII codes
        labels: [N, seq_len] int tensor of token class labels
        device: compute device
        epochs: training epochs
        lr: learning rate
        batch_size: batch size

    Returns:
        (trained_model, training_stats)
    """
    print("\n" + "=" * 70)
    print("TRAINING: Neural Tokenizer (char -> token type)")
    print("=" * 70)

    N = chars.shape[0]
    seq_len = chars.shape[1]
    print(f"  Dataset:  {N} sequences x {seq_len} chars")
    print(f"  Epochs:   {epochs}")

    model = NeuralTokenizerNet(vocab_size=128, embed_dim=32, num_classes=8).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Create ignore mask for padding (char code 0)
    pad_mask = (chars == 0)
    labels_masked = labels.clone()
    labels_masked[pad_mask] = -1

    best_acc = 0.0
    best_state = None
    t_start = time.time()

    print(f"\n  {'Epoch':>6} | {'Loss':>10} | {'Accuracy':>9} | {'Time':>6}")
    print("  " + "-" * 42)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_chars = chars[idx]
            batch_labels = labels_masked[idx]

            optimizer.zero_grad()
            logits = model(batch_chars)  # [B, L, C]
            # Reshape for cross-entropy: [B*L, C] vs [B*L]
            loss = loss_fn(logits.reshape(-1, 8), batch_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        # Evaluate every 25 epochs
        if epoch % 25 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(chars)
                preds = logits.argmax(dim=-1)
                valid_mask = ~pad_mask
                correct = ((preds == labels) & valid_mask).float().sum()
                total_valid = valid_mask.float().sum()
                acc = (correct / total_valid).item()

            elapsed = time.time() - t_start
            print(f"  {epoch:6d} | {avg_loss:10.6f} | {acc:8.4f} | {elapsed:5.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed_total = time.time() - t_start
    print(f"\n  RESULTS:")
    print(f"    Best accuracy: {best_acc:.4f}")
    print(f"    Training time: {elapsed_total:.1f}s")

    stats = {
        "best_accuracy": best_acc,
        "epochs_trained": epochs,
        "training_time_s": elapsed_total,
    }

    return model, stats


# =============================================================================
# Training: Peephole Optimizer
# =============================================================================

def generate_optimizer_training_data(
    programs: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Generate training data for the peephole optimizer.

    Compiles nsl programs and compares unoptimized vs optimized IR
    to create labeled 3-instruction windows.

    Labels:
        0 = no optimization
        1 = constant folding
        2 = strength reduction
        3 = dead store elimination
        4 = identity elimination

    Args:
        programs: list of nsl source code strings
        device: compute device

    Returns:
        features: [N, 15] float tensor (3 instructions x 5 features each)
        labels: [N] long tensor of optimization classes
        count: number of windows
    """
    from ncpu.os.compiler import NeuralCompiler
    from ncpu.os.language import Lexer, Parser

    compiler = NeuralCompiler(device=device)
    all_features = []
    all_labels = []

    # nsl programs for optimizer training
    nsl_programs = programs + [
        # Constant folding opportunities
        "var a = 3; var b = 5; var c = a + b; halt;",
        "var x = 10; var y = 20; var z = x * y; halt;",
        "var p = 100; var q = 4; var r = p / q; halt;",

        # Dead store opportunities
        "var x = 5; x = 10; halt;",
        "var a = 1; a = 2; a = 3; halt;",

        # Identity operations
        "var x = 5; var y = x + 0; halt;",
        "var x = 5; var y = x * 1; halt;",

        # Mixed opportunities
        "var a = 2; var b = 3; var c = a + b; var d = c * 2; halt;",
        "var x = 0; x = 1; var y = x + 0; halt;",

        # No optimization needed
        "var a = 5; var b = 10; var c = a + b; halt;",
        "var x = 1; if (x == 1) { var y = 2; } halt;",

        # Loops with various patterns
        "var i = 0; while (i < 10) { i = i + 1; } halt;",
        "var s = 0; var i = 1; while (i < 5) { s = s + i; i = i + 1; } halt;",

        # Arithmetic chains
        "var a = 1; var b = 2; var c = 3; var d = a + b + c; halt;",
        "var x = 10; var y = x - 3; var z = y * 2; halt;",
    ]

    for source in nsl_programs:
        try:
            tokens = Lexer(source).tokenize()
            ast = Parser(tokens).parse()
            ir_before, _ = compiler.ir_gen.generate(ast)
            ir_after, opt_count = compiler.optimizer.optimize(ir_before)
        except Exception:
            continue

        # Generate sliding windows from the unoptimized IR
        for i in range(len(ir_before) - 2):
            window = ir_before[i:i + 3]
            features = compiler._window_features(window)

            # Classify what happened at this position
            label = _classify_window_optimization(ir_before, ir_after, i, compiler)
            all_features.append(features)
            all_labels.append(label)

    if not all_features:
        print("[Optimizer] WARNING: No training data generated")
        return torch.zeros(1, 15, device=device), torch.zeros(1, dtype=torch.long, device=device), 0

    features = torch.stack(all_features).to(device)
    labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    count = len(all_features)
    print(f"[Optimizer] Generated {count} training windows")

    # Print class distribution
    for c in range(5):
        n = (labels == c).sum().item()
        print(f"  Class {c}: {n} ({n/max(1,count)*100:.1f}%)")

    return features, labels, count


def _classify_window_optimization(
    before: List,
    after: List,
    idx: int,
    compiler,
) -> int:
    """Classify what optimization was applied at a window position.

    Returns:
        0 = none, 1 = constant fold, 2 = strength reduction,
        3 = dead store, 4 = identity elimination
    """
    if idx >= len(before) or idx >= len(after):
        return 0

    instr_b = before[idx]
    instr_a = after[idx] if idx < len(after) else None

    # Check if instruction was removed entirely
    if instr_a is None or len(after) < len(before):
        # Something was removed - could be dead store or identity
        if instr_b.op == "mov" and idx + 1 < len(before):
            next_b = before[idx + 1]
            if next_b.op == "mov" and next_b.dest == instr_b.dest:
                return 3  # dead store
        if instr_b.op == "mov" and instr_b.dest == instr_b.src1 and not instr_b.src2:
            return 4  # identity (MOV Rx, Rx)
        return 0

    # Check if operation changed
    if instr_b.op != instr_a.op:
        if instr_a.op == "mov" and instr_a.comment and "folded" in instr_a.comment:
            return 1  # constant folding
        if instr_b.op == "mul" and instr_a.op == "shl":
            return 2  # strength reduction
        return 0

    return 0  # no change


def train_optimizer(
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int = 300,
    lr: float = 2e-3,
    batch_size: int = 64,
) -> Tuple[PeepholeOptimizerNet, Dict]:
    """Train the peephole optimization classifier.

    Trains a small MLP to predict which (if any) peephole optimization
    applies to a 3-instruction window.

    Args:
        features: [N, 15] float tensor (3 instrs x 5 features)
        labels:   [N] long tensor of optimization classes
        device:   compute device
        epochs:   training epochs
        lr:       learning rate
        batch_size: batch size

    Returns:
        (trained_model, training_stats)
    """
    print("\n" + "=" * 70)
    print("TRAINING: Peephole Optimizer (3-instr window -> optimization class)")
    print("=" * 70)

    N = features.shape[0]
    if N == 0:
        print("  WARNING: No training data, skipping optimizer training")
        model = PeepholeOptimizerNet().to(device)
        return model, {"best_accuracy": 0.0, "epochs_trained": 0}

    print(f"  Dataset: {N} windows")
    print(f"  Epochs:  {epochs}")

    model = PeepholeOptimizerNet(window_size=3, feat_per_instr=5, num_opts=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Class weights for imbalanced data (most windows have no optimization)
    class_counts = torch.zeros(5, device=device)
    for c in range(5):
        class_counts[c] = max((labels == c).sum().item(), 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 5.0  # normalize

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_acc = 0.0
    best_state = None
    t_start = time.time()

    print(f"\n  {'Epoch':>6} | {'Loss':>10} | {'Accuracy':>9} | {'Time':>6}")
    print("  " + "-" * 42)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]
            batch_f = features[idx]
            batch_l = labels[idx]

            optimizer.zero_grad()
            logits = model(batch_f)
            loss = loss_fn(logits, batch_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        if epoch % 25 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()

            elapsed = time.time() - t_start
            print(f"  {epoch:6d} | {avg_loss:10.6f} | {acc:8.4f} | {elapsed:5.0f}s")

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed_total = time.time() - t_start
    print(f"\n  RESULTS:")
    print(f"    Best accuracy: {best_acc:.4f}")
    print(f"    Training time: {elapsed_total:.1f}s")

    stats = {
        "best_accuracy": best_acc,
        "epochs_trained": epochs,
        "training_time_s": elapsed_total,
    }

    return model, stats


# =============================================================================
# Validation: End-to-End Assembly
# =============================================================================

def validate_codegen_e2e(
    model: NeuralCodeGenNet,
    programs: List[str],
    device: torch.device,
) -> Dict:
    """Validate the trained codegen against real assembly programs end-to-end.

    Assembles each program with both the classical assembler and the neural
    codegen, comparing every instruction word for exact match.

    Args:
        model: trained NeuralCodeGenNet
        programs: list of assembly source strings
        device: compute device

    Returns:
        Validation results with per-program and aggregate statistics
    """
    print("\n" + "=" * 70)
    print("VALIDATION: End-to-End Assembly (Neural vs Classical)")
    print("=" * 70)

    assembler = ClassicalAssembler()
    model.eval()

    total_correct = 0
    total_instructions = 0
    program_results = []

    for i, source in enumerate(programs):
        result = assembler.assemble(source)
        if not result.success:
            continue

        prog_correct = 0
        prog_total = len(result.instructions)

        for instr, classical_word in zip(result.instructions, result.binary):
            features = encode_instruction_features(
                instr.opcode, instr.rd, instr.rs1, instr.rs2,
                instr.imm, instr.fmt, device,
            )

            with torch.no_grad():
                logits = model(features.unsqueeze(0))
                bits = (torch.sigmoid(logits[0]) > 0.5)
                neural_word = _bits_to_int(bits)

            if neural_word == classical_word:
                prog_correct += 1
            else:
                # Report mismatch details
                source_line = instr.source if instr.source else f"op={instr.opcode}"
                print(f"    MISMATCH [{source_line}]: "
                      f"neural=0x{neural_word:08X} vs classical=0x{classical_word:08X}")

            total_correct += (neural_word == classical_word)
            total_instructions += 1

        em_rate = prog_correct / max(1, prog_total)
        name = source.split("\n")[0][:60]
        program_results.append({"name": name, "exact_match": em_rate,
                                "correct": prog_correct, "total": prog_total})
        status = "PASS" if prog_correct == prog_total else "FAIL"
        print(f"  [{status}] {name}: {prog_correct}/{prog_total} "
              f"({em_rate:.1%})")

    overall = total_correct / max(1, total_instructions)
    print(f"\n  OVERALL: {total_correct}/{total_instructions} = {overall:.4f}")

    return {
        "overall_exact_match": overall,
        "total_correct": total_correct,
        "total_instructions": total_instructions,
        "program_results": program_results,
    }


# =============================================================================
# Model Saving
# =============================================================================

def save_models(
    codegen_model: NeuralCodeGenNet,
    tokenizer_model: NeuralTokenizerNet,
    optimizer_model: PeepholeOptimizerNet,
    output_dir: Path,
    stats: Dict,
):
    """Save all trained models and training metadata.

    Args:
        codegen_model: trained codegen network
        tokenizer_model: trained tokenizer network
        optimizer_model: trained optimizer network
        output_dir: directory to save models
        stats: training statistics to save alongside models
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    codegen_path = output_dir / "assembler_codegen.pt"
    tokenizer_path = output_dir / "assembler_tokenizer.pt"
    optimizer_path = output_dir / "compiler_optimizer.pt"
    stats_path = output_dir / "training_stats.json"

    torch.save(codegen_model.state_dict(), codegen_path)
    torch.save(tokenizer_model.state_dict(), tokenizer_path)
    torch.save(optimizer_model.state_dict(), optimizer_path)

    # Save stats (convert non-serializable items)
    clean_stats = {}
    for key, val in stats.items():
        if isinstance(val, dict):
            clean_stats[key] = {}
            for k, v in val.items():
                if isinstance(v, list):
                    # Truncate long history lists for readability
                    clean_stats[key][k] = v[-10:] if len(v) > 10 else v
                elif isinstance(v, (int, float, str, bool)):
                    clean_stats[key][k] = v
        elif isinstance(val, (int, float, str, bool)):
            clean_stats[key] = val

    with open(stats_path, "w") as f:
        json.dump(clean_stats, f, indent=2, default=str)

    print(f"\n  Models saved to {output_dir}/")
    print(f"    {codegen_path.name}  ({codegen_path.stat().st_size / 1024:.1f} KB)")
    print(f"    {tokenizer_path.name}  ({tokenizer_path.stat().st_size / 1024:.1f} KB)")
    print(f"    {optimizer_path.name}  ({optimizer_path.stat().st_size / 1024:.1f} KB)")
    print(f"    {stats_path.name}")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Full training pipeline for nCPU neural assembler and compiler.

    Orchestrates:
    1. Exhaustive instruction data generation
    2. CodeGen training with curriculum learning
    3. Tokenizer training on assembly programs
    4. Peephole optimizer training on compiled nsl programs
    5. End-to-end validation against real programs
    6. Model saving
    """
    parser = argparse.ArgumentParser(
        description="Train nCPU neural assembler and compiler models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train_neuros_gpu.py                     # Full training
  python training/train_neuros_gpu.py --quick              # Quick smoke test
  python training/train_neuros_gpu.py --epochs-codegen 5000  # Extended codegen training
  python training/train_neuros_gpu.py --hidden-dim 512     # Larger model
        """,
    )
    parser.add_argument("--epochs-codegen", type=int, default=5000,
                        help="CodeGen training epochs (default: 5000)")
    parser.add_argument("--epochs-tokenizer", type=int, default=500,
                        help="Tokenizer training epochs (default: 500)")
    parser.add_argument("--epochs-optimizer", type=int, default=300,
                        help="Optimizer training epochs (default: 300)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="CodeGen hidden dimension (default: 256)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="CodeGen batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="CodeGen learning rate (default: 3e-3)")
    parser.add_argument("--num-synthetic", type=int, default=200,
                        help="Number of synthetic programs to generate (default: 200)")
    parser.add_argument("--num-immediates", type=int, default=64,
                        help="Number of random immediate values (default: 64)")
    parser.add_argument("--all-immediates", action="store_true",
                        help="Train on ALL 32768 immediate values for full generalization")
    parser.add_argument("--target-accuracy", type=float, default=1.0,
                        help="Stop codegen training at this exact-match rate (default: 1.0)")
    parser.add_argument("--patience", type=int, default=500,
                        help="Early stopping patience in epochs (default: 500)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for models (default: models/research/neuros/)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with reduced epochs")
    parser.add_argument("--codegen-only", action="store_true",
                        help="Train only the codegen model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # All-immediates mode: train on every possible 15-bit immediate value
    if args.all_immediates:
        args.num_immediates = 32768  # Will be ignored — we override below
        print("[MODE] All immediates — full 32K coverage for generalization")

    # Quick mode overrides
    if args.quick:
        args.epochs_codegen = 100
        args.epochs_tokenizer = 50
        args.epochs_optimizer = 30
        args.num_synthetic = 20
        args.num_immediates = 16
        args.patience = 50
        print("[MODE] Quick smoke test")

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Setup
    device = select_device()
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models" / "os"
    programs_dir = PROJECT_ROOT / "programs"

    print(f"\n{'='*70}")
    print(f"nCPU Neural Assembler/Compiler Training Pipeline")
    print(f"{'='*70}")
    print(f"  Project root:  {PROJECT_ROOT}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Programs dir:  {programs_dir}")
    print(f"  Device:        {device}")
    print(f"  PyTorch:       {torch.__version__}")
    if device.type == "cuda":
        print(f"  CUDA version:  {torch.version.cuda}")

    t_pipeline_start = time.time()
    all_stats = {}

    # ─── Phase 1: Data Generation ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 1: Data Generation")
    print(f"{'='*70}")

    # Load real programs
    real_programs = load_real_programs(programs_dir)

    # Generate synthetic programs
    synthetic_programs = generate_synthetic_programs(n=args.num_synthetic)
    all_programs = real_programs + synthetic_programs

    # Generate exhaustive instruction encodings
    codegen_features, codegen_targets, n_codegen = generate_exhaustive_training_data(
        device=device, num_immediates=args.num_immediates
    )

    # Also add instructions from real+synthetic programs to training data
    assembler = ClassicalAssembler()
    extra_features = []
    extra_targets = []
    for source in all_programs:
        result = assembler.assemble(source)
        if not result.success:
            continue
        for instr, word in zip(result.instructions, result.binary):
            features = encode_instruction_features(
                instr.opcode, instr.rd, instr.rs1, instr.rs2,
                instr.imm, instr.fmt, device,
            )
            target = _int_to_bits(word).to(device)
            extra_features.append(features)
            extra_targets.append(target)

    if extra_features:
        codegen_features = torch.cat([codegen_features, torch.stack(extra_features)])
        codegen_targets = torch.cat([codegen_targets, torch.stack(extra_targets)])
        print(f"[Data] Added {len(extra_features)} examples from real+synthetic programs")
        print(f"[Data] Total codegen training examples: {codegen_features.shape[0]:,}")

    # ─── Phase 2: Train CodeGen ──────────────────────────────────────────
    codegen_model, codegen_stats = train_codegen(
        features=codegen_features,
        targets=codegen_targets,
        device=device,
        epochs=args.epochs_codegen,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        target_accuracy=args.target_accuracy,
    )
    all_stats["codegen"] = codegen_stats

    # End-to-end validation
    val_stats = validate_codegen_e2e(codegen_model, real_programs, device)
    all_stats["codegen_validation"] = val_stats

    if args.codegen_only:
        # Save just codegen and exit
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(codegen_model.state_dict(), output_dir / "assembler_codegen.pt")
        print(f"\n  CodeGen model saved to {output_dir / 'assembler_codegen.pt'}")
        return

    # ─── Phase 3: Train Tokenizer ──────────────────────────────────────────
    tok_chars, tok_labels, n_tok = generate_tokenizer_training_data(
        all_programs, device=device
    )

    tokenizer_model, tok_stats = train_tokenizer(
        chars=tok_chars,
        labels=tok_labels,
        device=device,
        epochs=args.epochs_tokenizer,
    )
    all_stats["tokenizer"] = tok_stats

    # ─── Phase 4: Train Optimizer ──────────────────────────────────────────
    # Generate nsl programs for optimizer training
    nsl_programs = [
        "var a = 3; var b = 5; var c = a + b; halt;",
        "var x = 10; var y = 20; var z = x * y; halt;",
        "var x = 5; x = 10; halt;",
        "var a = 1; var b = a + 0; halt;",
        "var x = 5; var y = x + 3; var z = y - 1; halt;",
        "var i = 0; while (i < 5) { i = i + 1; } halt;",
        "var a = 2; var b = 3; if (a < b) { var c = b - a; } halt;",
        "var p = 7; var q = 3; var r = p * q; var s = r + 1; halt;",
    ]

    opt_features, opt_labels, n_opt = generate_optimizer_training_data(
        nsl_programs, device=device
    )

    optimizer_model, opt_stats = train_optimizer(
        features=opt_features,
        labels=opt_labels,
        device=device,
        epochs=args.epochs_optimizer,
    )
    all_stats["optimizer"] = opt_stats

    # ─── Phase 5: Save Everything ──────────────────────────────────────────
    save_models(codegen_model, tokenizer_model, optimizer_model, output_dir, all_stats)

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_pipeline_start

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:     {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  Device:         {device}")
    print(f"  Models saved:   {output_dir}/")
    print(f"\n  CodeGen:")
    print(f"    Full exact match:   {codegen_stats['full_exact_match']:.4f} "
          f"({codegen_stats['full_exact']}/{codegen_stats['total_examples']})")
    print(f"    Val exact match:    {codegen_stats['val_exact_match']:.4f}")
    print(f"    E2E on real progs:  {val_stats['overall_exact_match']:.4f} "
          f"({val_stats['total_correct']}/{val_stats['total_instructions']})")
    print(f"  Tokenizer:")
    print(f"    Best accuracy:      {tok_stats['best_accuracy']:.4f}")
    print(f"  Optimizer:")
    print(f"    Best accuracy:      {opt_stats['best_accuracy']:.4f}")

    target_met = codegen_stats["full_exact_match"] >= 0.90
    print(f"\n  Target (90% exact match): {'ACHIEVED' if target_met else 'NOT YET'}")

    if not target_met:
        print(f"\n  Suggestions to improve:")
        print(f"    1. Increase --epochs-codegen to 5000+")
        print(f"    2. Increase --hidden-dim to 512")
        print(f"    3. Increase --num-immediates to 128 for denser coverage")
        print(f"    4. Try --lr 1e-3 for finer convergence")
        print(f"    5. Run on CUDA GPU for faster iteration")


if __name__ == "__main__":
    main()
