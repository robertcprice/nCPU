#!/usr/bin/env python3
"""
Dispatch network v3: Opcode-focused architecture for 100% accuracy.

Key insight: The opcode byte (bits 31:24) is the PRIMARY signal.
We one-hot encode the opcode to give each opcode its own feature vector.

This ensures the network can perfectly distinguish between opcodes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


class OpcodeEmbeddingNetwork(nn.Module):
    """
    Network using opcode embedding + additional features.

    Architecture:
    1. One-hot opcode (256) → Embedding (32) + Other features (12)
    2. Combined (44) → Hidden (32) → Output (7)
    """

    def __init__(self, opcode_size=256, embed_dim=32, hidden_size=32, output_size=7):
        super().__init__()

        # Opcode embedding (one-hot opcode → dense vector)
        self.opcode_embedding = nn.Embedding(opcode_size, embed_dim)

        # Combined features: embedding + additional features
        combined_size = embed_dim + 12  # 32 + 12 = 44

        # Hidden layers
        self.fc1 = nn.Linear(combined_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        # Initialize
        nn.init.xavier_uniform_(self.opcode_embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, opcode_idx, features):
        """
        Forward pass.

        Args:
            opcode_idx: [batch] opcode indices (0-255)
            features: [batch, 12] additional features

        Returns:
            [batch, 7] kernel logits
        """
        # Embed opcode
        opcode_embed = self.opcode_embedding(opcode_idx)  # [batch, 32]

        # Concatenate with other features
        combined = torch.cat([opcode_embed, features], dim=1)  # [batch, 44]

        # Hidden layers
        hidden = torch.relu(self.fc1(combined))
        hidden = torch.relu(self.fc2(hidden))

        # Output
        output = self.fc3(hidden)

        return output

    def predict_kernel(self, instruction, pc):
        """
        Predict which kernel to use for a single instruction.

        Args:
            instruction: int (instruction word)
            pc: int (program counter)

        Returns:
            int: kernel index (0-6)
        """
        with torch.no_grad():
            opcode = (instruction >> 24) & 0xFF
            features = extract_instruction_features(instruction, pc)

            opcode_tensor = torch.tensor([opcode], dtype=torch.long)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)

            logits = self.forward(opcode_tensor, features_tensor)
            kernel = torch.argmax(logits, dim=1).item()

        return kernel


def extract_instruction_features(instruction, pc):
    """
    Extract additional features beyond opcode.
    """
    features = np.zeros(12, dtype=np.float32)

    # Instruction category (top 4 bits of opcode)
    features[0] = ((instruction >> 28) & 0xF) / 15.0

    # Instruction structural patterns
    features[1] = ((instruction >> 0) & 0xFF) / 255.0
    features[2] = ((instruction >> 16) & 0xFF) / 255.0

    # PC-based features
    features[3] = float((pc >> 0) & 0xFF) / 255.0
    features[4] = float((pc >> 8) & 0xFF) / 255.0

    # Register fields
    features[5] = float((instruction >> 0) & 0x1F) / 31.0
    features[6] = float((instruction >> 5) & 0x1F) / 31.0

    # Size field (for load/store)
    features[7] = float((instruction >> 30) & 0x3) / 3.0

    # Instruction class
    features[8] = float((instruction >> 26) & 0x3) / 3.0

    # SF bit
    features[9] = float((instruction >> 31) & 0x1)

    # Additional structural hints
    features[10] = float((instruction >> 10) & 0xFFF) / 4095.0  # Immediate/part of offset
    features[11] = float((instruction >> 21) & 0x3) / 3.0

    return features


def create_opcode_kernel_map():
    """
    Create a comprehensive mapping of opcodes to kernels.

    This covers ALL ARM64 opcodes that we support.
    """
    opcode_to_kernel = {}

    # KERNEL_ARITHMETIC = 0
    arithmetic_opcodes = [
        0x91, 0x11, 0x10, 0x00,  # ADD
        0xD1, 0x51, 0x50, 0x40,  # SUB
        0xB1, 0x71, 0x70, 0xF0,  # ADDS/SUBS
        0xD2, 0x52,              # MOVZ
        0xF2, 0x72,              # MOVK
        0x92, 0x12,              # MOVN
        0x93, 0x13,              # SBFM (extend)
    ]
    for op in arithmetic_opcodes:
        opcode_to_kernel[op] = 0

    # KERNEL_LOGICAL = 1
    logical_opcodes = [
        0x8A, 0x0A,  # AND
        0xAA, 0x2A,  # ORR
        0xCA, 0x4A,  # EOR
        0xBA, 0x3A,  # ANDS
        0xEA,        # EON
    ]
    for op in logical_opcodes:
        opcode_to_kernel[op] = 1

    # KERNEL_LOADSTORE = 2
    loadstore_opcodes = [
        0xF9, 0x79,  # LDR/STR (64-bit)
        0xB9, 0x39,  # LDR/STR (32-bit)
        0x39, 0x38,  # LDRB/STRB
        0x79, 0x78,  # LDRH/STRH
        0xF8, 0x78,  # LDUR/STUR (unscaled) - ADDED
        0xB8, 0x38,  # LDUR/STUR (32-bit, unscaled) - ADDED
        0xA8, 0x28,  # LDP
        0xA9, 0x29,  # STP
        0xC8, 0xC0,  # LDPSW
        0x89, 0x88,  # LDP (32-bit)
    ]
    for op in loadstore_opcodes:
        opcode_to_kernel[op] = 2

    # KERNEL_BRANCH = 3
    branch_opcodes = [
        0x14,        # B (unconditional)
        0x94,        # BL
        0xD6,        # BR/BLR/RET
        0x54, 0x55,  # B.cond
        0xB4, 0xB5,  # CBZ/CBNZ
        0x34, 0x35,  # CBZ/CBNZ (32-bit)
        0x44, 0x45,  # TBZ/TBNZ
    ]
    for op in branch_opcodes:
        opcode_to_kernel[op] = 3

    # KERNEL_MULDIV = 4
    muldiv_opcodes = [
        0x9B,        # MADD/MSUB
        0x1B,        # MADD/MSUB (32-bit)
        0x9C,        # SMADDL/UMADDL
    ]
    for op in muldiv_opcodes:
        opcode_to_kernel[op] = 4

    # KERNEL_EXTEND_SHIFT = 5
    extend_shift_opcodes = [
        0x93,        # SXTW/SXTH/SXTB
        0x13,        # SXTB (32-bit)
        0x90, 0x10,  # ADRP
        0xB0,        # ADR
        0xD3,        # UBFM (LSL)
        0x53,        # EXTR (rotate)
        0xAC,        # Shift (immediate)
    ]
    for op in extend_shift_opcodes:
        opcode_to_kernel[op] = 5

    # KERNEL_SYSTEM = 6
    system_opcodes = [
        0xD4,        # HLT/SVC/BRK/etc
        0xD5,        # SYS/MSR/MRS
    ]
    for op in system_opcodes:
        opcode_to_kernel[op] = 6

    return opcode_to_kernel


def create_training_samples_from_map():
    """
    Create training samples by varying the instruction word while keeping opcode fixed.
    """
    opcode_to_kernel = create_opcode_kernel_map()
    samples = []

    for opcode, kernel in opcode_to_kernel.items():
        # Create variations for this opcode
        # Vary: register fields, immediates, offset fields
        for rd in [0, 5, 10, 20, 30]:
            for rn in [0, 5, 10, 20, 30]:
                for rm in [0, 5, 10, 20]:
                    for imm in [0, 1, 0x10, 0x100, 0x1000]:
                        # Base instruction
                        inst = (opcode << 24)

                        # Add fields based on instruction type
                        if kernel in [0, 1]:  # Arithmetic/Logical - usually 3 registers
                            inst |= (rm & 0x1F) << 16
                            inst |= (rn & 0x1F) << 5
                            inst |= (rd & 0x1F)
                        elif kernel == 2:  # Load/Store
                            inst |= (imm >> 3) << 10  # Offset
                            inst |= (rn & 0x1F) << 5
                            inst |= (rd & 0x1F)
                        elif kernel == 3:  # Branch
                            inst |= (imm >> 2)
                            if opcode in [0xB4, 0xB5]:  # CBZ/CBNZ
                                inst |= (rd & 0x1F) << 5
                        elif kernel == 4:  # MulDiv
                            inst |= (rm & 0x1F) << 16
                            inst |= (rn & 0x1F) << 5
                            inst |= (rd & 0x1F)
                        else:  # Extend/Shift/System
                            inst |= (rn & 0x1F) << 5
                            inst |= (rd & 0x1F)

                        samples.append((inst, kernel))

    print(f"Created {len(samples)} base samples")

    # Add PC variations
    expanded = []
    for inst, kernel in samples:
        expanded.append((inst, 0, kernel))
        for pc in [0x100, 0x1000, 0x10000]:
            expanded.append((inst, pc, kernel))

    print(f"Expanded to {len(expanded)} samples with PC variations")
    return expanded


def train_to_100_percent(max_epochs=500):
    """
    Train with goal of 100% accuracy using opcode embedding.
    """
    print("=" * 80)
    print("  🧠 TRAINING DISPATCH NETWORK V3 - GOAL: 100% ACCURACY")
    print("=" * 80)
    print()

    # Create training data
    print("Creating training data from opcode map...")
    samples = create_training_samples_from_map()

    X_opcodes = []
    X_features = []
    y = []

    for inst, pc, kernel in samples:
        opcode = (inst >> 24) & 0xFF
        features = extract_instruction_features(inst, pc)

        X_opcodes.append(opcode)
        X_features.append(features)
        y.append(kernel)

    X_opcodes = np.array(X_opcodes, dtype=np.int64)
    X_features = np.array(X_features, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Opcodes shape: {X_opcodes.shape}")
    print(f"Features shape: {X_features.shape}")
    print(f"Labels shape: {y.shape}")
    print()

    # Show distribution
    unique, counts = np.unique(y, return_counts=True)
    kernel_names = ["ARITHMETIC", "LOGICAL", "LOADSTORE", "BRANCH",
                    "MULDIV", "EXTEND_SHIFT", "SYSTEM"]
    print("Sample distribution:")
    for kernel, count in zip(unique, counts):
        print(f"  Kernel {kernel} ({kernel_names[kernel]}): {count} samples")
    print()

    # Convert to tensors
    X_opcode_tensor = torch.from_numpy(X_opcodes)
    X_feature_tensor = torch.from_numpy(X_features)
    y_tensor = torch.from_numpy(y)

    # Split
    indices = np.random.permutation(len(X_opcodes))
    split = int(0.8 * len(X_opcodes))

    X_train_op = X_opcode_tensor[indices[:split]]
    X_train_feat = X_feature_tensor[indices[:split]]
    y_train = y_tensor[indices[:split]]

    X_val_op = X_opcode_tensor[indices[split:]]
    X_val_feat = X_feature_tensor[indices[split:]]
    y_val = y_tensor[indices[split:]]

    # Create model
    print("Creating opcode embedding network...")
    model = OpcodeEmbeddingNetwork()
    print(f"Network: One-hot opcode(256) → Embedding(32) + Features(12) → Hidden(32) → Output(7)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()

    # Train
    print("Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val_acc = 0
    best_model = None
    patience = 30
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        outputs = model(X_train_op, X_train_feat)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_op, X_val_feat)
            val_loss = criterion(val_outputs, y_val)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val).float().mean().item()

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                train_outputs = model(X_train_op, X_train_feat)
                _, train_pred = torch.max(train_outputs, 1)
                train_acc = (train_pred == y_train).float().mean().item()

            print(f"Epoch [{epoch+1:3d}/{max_epochs}] "
                  f"Loss: {loss.item():.6f} | "
                  f"Train: {train_acc*100:5.1f}% | "
                  f"Val: {val_acc*100:5.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Check for 100% on validation
        if val_acc >= 1.0:
            print(f"\n✅ 100% validation accuracy achieved at epoch {epoch+1}!")
            break

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)

    # Final accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X_opcode_tensor, X_feature_tensor)
        _, pred = torch.max(outputs, 1)
        final_acc = (pred == y_tensor).float().mean().item()

    print()
    print(f"Final accuracy: {final_acc*100:.2f}%")
    print(f"Best validation: {best_val_acc*100:.2f}%")
    print()

    # Test on specific instructions
    print("Testing on instructions:")
    test_insts = [
        (0xD2000C80, 0, "MOVZ X0, #100"),
        (0xAA000000, 1, "ORR X0, X0, X0"),
        (0xF9400000, 2, "LDR X0, [X0]"),
        (0x14000000, 3, "B #0"),
        (0x9B000000, 4, "MADD X0, X0, X0, X0"),
        (0x9340FC00, 5, "SXTW X0, W0"),
        (0xD4400000, 6, "HLT #0"),
    ]

    for inst, expected, desc in test_insts:
        pred = model.predict_kernel(inst, 0)
        status = "✅" if pred == expected else "❌"
        print(f"  {status} {desc}: {pred} vs {expected}")
    print()

    # Save weights for Metal shader
    print("Saving weights for Metal shader...")
    print("-" * 80)

    # Extract weights in the format expected by the shader
    # Layout: embedding (256*32) + fc1_w (44*32) + fc1_b (32) + fc2_w (32*16) + fc2_b (16) + fc3_w (16*7) + fc3_b (7)

    # Embedding: [256, 32]
    embed = model.opcode_embedding.weight.data.detach().cpu().numpy().flatten()  # 8192

    # FC1: [32, 44] -> flatten to [1408]
    fc1_w = model.fc1.weight.data.detach().cpu().numpy().flatten()  # 1408
    # FC1 bias: [32]
    fc1_b = model.fc1.bias.data.detach().cpu().numpy().flatten()  # 32

    # FC2: [16, 32] -> flatten to [512]
    fc2_w = model.fc2.weight.data.detach().cpu().numpy().flatten()  # 512
    # FC2 bias: [16]
    fc2_b = model.fc2.bias.data.detach().cpu().numpy().flatten()  # 16

    # FC3: [7, 16] -> flatten to [112]
    fc3_w = model.fc3.weight.data.detach().cpu().numpy().flatten()  # 112
    # FC3 bias: [7]
    fc3_b = model.fc3.bias.data.detach().cpu().numpy().flatten()  # 7

    # Concatenate in correct order
    flat_weights = np.concatenate([
        embed,   # 8192
        fc1_w,   # 1408
        fc1_b,   # 32
        fc2_w,   # 512
        fc2_b,   # 16
        fc3_w,   # 112
        fc3_b,   # 7
    ])  # Total: 8192 + 1408 + 32 + 512 + 16 + 112 + 7 = 10279

    print(f"Total weights: {len(flat_weights)} (expected 10279)")
    print(f"Weight range: [{flat_weights.min():.3f}, {flat_weights.max():.3f}]")

    # Save
    output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "dispatch_weights_embedding_100pct.npy"
    np.save(output_path, flat_weights)
    print(f"✅ Saved to: {output_path}")
    print()

    # Save model checkpoint
    model_path = output_dir / "dispatch_embedding_model_100pct.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model checkpoint to: {model_path}")
    print()

    return model


if __name__ == "__main__":
    model = train_to_100_percent()
