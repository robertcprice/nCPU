#!/usr/bin/env python3
"""Train the neural instruction decoder — replaces the Qwen2.5-Coder LLM.

A ~50K parameter character-level CNN that classifies assembly text into
one of 22 opcodes. Trains in <60s on CPU.

The model handles the HARD part (opcode identification despite case,
whitespace, and separator variations). Operand extraction is deterministic
once the opcode format is known — exactly like a real CPU decoder.

Usage:
    python training/train_decode_neural.py
    python training/train_decode_neural.py --samples 100000 --epochs 10
"""

import sys
import time
import random
import argparse

sys.path.insert(0, ".")

import torch
import torch.nn as nn
from pathlib import Path

from ncpu.model.architectures import InstructionDecoderNet

REGISTERS = [f"R{i}" for i in range(8)]
LABEL_NAMES = [
    "loop", "start", "end", "done", "exit", "next", "skip", "continue",
    "check", "test", "begin", "finish", "main", "init", "cleanup",
]
OPCODES = InstructionDecoderNet.OPCODES
OPCODE_TO_IDX = {op: i for i, op in enumerate(OPCODES)}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation (inlined for self-contained training)
# ═══════════════════════════════════════════════════════════════════════════════

def random_case(s):
    c = random.choice(["upper", "lower", "title", "mixed"])
    if c == "upper": return s.upper()
    if c == "lower": return s.lower()
    if c == "title": return s.title()
    return "".join(ch.upper() if random.random() > 0.5 else ch.lower() for ch in s)


def random_reg():
    return random_case(random.choice(REGISTERS))


def random_imm():
    v = random.choice([
        random.randint(0, 10),
        random.randint(0, 255),
        random.randint(0, 65535),
        random.randint(-100, -1),
    ])
    if random.random() < 0.2 and v >= 0:
        return f"0x{v:x}" if random.random() > 0.5 else f"0x{v:X}"
    return str(v)


def random_sep():
    return random.choice([", ", ",", " , ", "  ,  ", " ", "  "])


def add_ws(s):
    if random.random() < 0.2: s = "  " + s
    if random.random() < 0.2: s = s + "  "
    if random.random() < 0.1: s = "\t" + s
    return s


def gen_sample():
    """Generate one (instruction_text, opcode_index) pair."""
    category = random.choices(
        ["mov_ri", "mov_rr", "arith", "cmp", "jump", "unary", "halt", "nop", "invalid"],
        weights=[20, 10, 30, 10, 15, 6, 3, 3, 3],
    )[0]

    if category == "mov_ri":
        instr = f"{random_case('MOV')} {random_reg()}{random_sep()}{random_imm()}"
        return add_ws(instr), OPCODE_TO_IDX["OP_MOV_REG_IMM"]

    if category == "mov_rr":
        instr = f"{random_case('MOV')} {random_reg()}{random_sep()}{random_reg()}"
        return add_ws(instr), OPCODE_TO_IDX["OP_MOV_REG_REG"]

    if category == "arith":
        op, key = random.choice([
            ("ADD", "OP_ADD"), ("SUB", "OP_SUB"), ("MUL", "OP_MUL"), ("DIV", "OP_DIV"),
            ("AND", "OP_AND"), ("OR", "OP_OR"), ("XOR", "OP_XOR"),
            ("SHL", "OP_SHL"), ("SHR", "OP_SHR"),
        ])
        instr = f"{random_case(op)} {random_reg()}{random_sep()}{random_reg()}{random_sep()}{random_reg()}"
        return add_ws(instr), OPCODE_TO_IDX[key]

    if category == "cmp":
        instr = f"{random_case('CMP')} {random_reg()}{random_sep()}{random_reg()}"
        return add_ws(instr), OPCODE_TO_IDX["OP_CMP"]

    if category == "jump":
        op, key = random.choice([
            ("JMP", "OP_JMP"), ("JZ", "OP_JZ"), ("JNZ", "OP_JNZ"),
            ("JS", "OP_JS"), ("JNS", "OP_JNS"),
        ])
        target = random_case(random.choice(LABEL_NAMES)) if random.random() < 0.7 else str(random.randint(0, 20))
        instr = f"{random_case(op)} {target}"
        return add_ws(instr), OPCODE_TO_IDX[key]

    if category == "unary":
        op, key = random.choice([("INC", "OP_INC"), ("DEC", "OP_DEC")])
        instr = f"{random_case(op)} {random_reg()}"
        return add_ws(instr), OPCODE_TO_IDX[key]

    if category == "halt":
        return add_ws(random_case("HALT")), OPCODE_TO_IDX["OP_HALT"]

    if category == "nop":
        return add_ws(random_case("NOP")), OPCODE_TO_IDX["OP_NOP"]

    # invalid
    invalid = random.choice([
        f"{random.choice(['FOO', 'BAR', 'MOVE', 'ADDD'])} R1, R2",
        "", "   ", "$$#@!", "123", "MO", "AD",
        f"MOV R9, 5", f"ADD R8, R1, R2",
    ])
    return add_ws(invalid), OPCODE_TO_IDX["OP_INVALID"]


def generate_dataset(n_samples):
    """Generate training data as tensors."""
    texts, labels = [], []
    for _ in range(n_samples):
        text, label = gen_sample()
        texts.append(text)
        labels.append(label)
    return texts, labels


def encode_batch(texts, max_len=64):
    """Encode a list of instruction strings to padded char tensor."""
    batch = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, text in enumerate(texts):
        chars = [min(ord(c), 127) for c in text[:max_len]]
        batch[i, :len(chars)] = torch.tensor(chars, dtype=torch.long)
    return batch


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    print("=" * 60)
    print("  Neural Instruction Decoder — Training")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    # Generate data
    print(f"\n[1] Generating {args.samples} training samples...")
    t0 = time.perf_counter()
    texts, labels = generate_dataset(args.samples)
    gen_time = time.perf_counter() - t0
    print(f"    Generated in {gen_time:.1f}s")

    # Distribution
    from collections import Counter
    dist = Counter(labels)
    for idx in sorted(dist):
        print(f"    {OPCODES[idx]:<22} {dist[idx]:>6} ({100*dist[idx]/len(labels):.1f}%)")

    # Encode
    X = encode_batch(texts)
    Y = torch.tensor(labels, dtype=torch.long)

    # Split 90/10
    n_val = max(1, len(X) // 10)
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]
    X_train, Y_train = X[n_val:], Y[n_val:]
    X_val, Y_val = X[:n_val], Y[:n_val]

    print(f"\n    Train: {len(X_train)}  Val: {len(X_val)}")

    # Model
    model = InstructionDecoderNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[2] Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size

    # Training loop
    print(f"\n[3] Training for {args.epochs} epochs (batch_size={batch_size}, lr={args.lr})...")
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        X_train, Y_train = X_train[perm], Y_train[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size].to(device)
            yb = Y_train[i:i+batch_size].to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)

        train_acc = 100 * correct / total
        avg_loss = total_loss / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_pred = val_logits.argmax(dim=1)
            val_acc = 100 * (val_pred == Y_val.to(device)).float().mean().item()

        elapsed = time.perf_counter() - t_start
        print(f"    Epoch {epoch+1:>2}/{args.epochs}  "
              f"loss={avg_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%  "
              f"[{elapsed:.1f}s]")

        if val_acc >= 99.9 and epoch >= 4:
            print(f"    Early stop: {val_acc:.1f}% validation accuracy")
            break

    train_time = time.perf_counter() - t_start
    print(f"\n    Training complete in {train_time:.1f}s")

    # Final validation accuracy per class
    print(f"\n[4] Per-class validation accuracy:")
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val.to(device))
        val_pred = val_logits.argmax(dim=1).cpu()

    for idx in range(len(OPCODES)):
        mask = Y_val == idx
        if mask.sum() == 0:
            continue
        class_acc = 100 * (val_pred[mask] == idx).float().mean().item()
        print(f"    {OPCODES[idx]:<22} {class_acc:>6.1f}% ({mask.sum().item()} samples)")

    overall_acc = 100 * (val_pred == Y_val).float().mean().item()
    print(f"\n    Overall: {overall_acc:.1f}%")

    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "decode.pt"

    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), model_path)
    size_kb = model_path.stat().st_size / 1024
    print(f"\n[5] Saved to {model_path} ({size_kb:.0f} KB)")
    print(f"    {n_params:,} parameters, {overall_acc:.1f}% accuracy")

    return overall_acc


def main():
    parser = argparse.ArgumentParser(description="Train neural instruction decoder")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="models/decode")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
