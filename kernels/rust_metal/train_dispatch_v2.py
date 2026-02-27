#!/usr/bin/env python3
"""
Improved dispatch network training with goal of 100% accuracy.

Key improvements:
1. Better input features (decode instruction fields)
2. Larger network capacity (16 hidden neurons)
3. More diverse training data (all opcodes, register variations)
4. Better training techniques (AdamW, cosine scheduler, label smoothing)
5. Detailed error analysis and targeted improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class DispatchNetworkV2(nn.Module):
    """
    Improved neural network for predicting which kernel to use.

    Architecture: 12 inputs → 16 hidden → 7 outputs

    Improved Inputs (normalized 0-1):
    - opcode (1 byte) - THE MOST IMPORTANT FEATURE
    - instruction top 4 bits (category)
    - instruction bit patterns (structural features)
    - PC low bytes (2 bytes)
    - Rd register field
    - Rn register field
    """

    def __init__(self, input_size=12, hidden_size=16, output_size=7, dropout=0.1):
        super().__init__()

        # Input -> Hidden with dropout
        self.input_weights = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Hidden -> Output
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.input_weights.weight)
        nn.init.zeros_(self.input_weights.bias)
        nn.init.xavier_uniform_(self.output_weights.weight)
        nn.init.zeros_(self.output_weights.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [batch, 12] normalized features

        Returns:
            [batch, 7] kernel logits
        """
        # Hidden layer with improved activation
        hidden = torch.relu(self.input_weights(x))
        hidden = self.batch_norm(hidden)
        hidden = self.dropout(hidden)

        # Output layer (raw logits)
        output = self.output_weights(hidden)

        return output

    def predict_kernel(self, features):
        """
        Predict which kernel to use for a single instruction.

        Args:
            features: numpy array of shape (12,)

        Returns:
            int: kernel index (0-6)
        """
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0)
            logits = self.forward(x)
            kernel = torch.argmax(logits, dim=1).item()

        return kernel


def extract_instruction_features(instruction, pc):
    """
    Extract RICH features from ARM64 instruction.

    Returns:
        numpy array of shape (12,) with normalized features
    """
    features = np.zeros(12, dtype=np.float32)

    # Feature 0: Opcode byte (MOST IMPORTANT)
    features[0] = ((instruction >> 24) & 0xFF) / 255.0

    # Feature 1: Instruction category (top 4 bits of opcode)
    features[1] = ((instruction >> 28) & 0xF) / 15.0

    # Feature 2-3: Instruction structural patterns
    # These help distinguish between similar instructions
    features[2] = ((instruction >> 0) & 0xFF) / 255.0  # Low byte
    features[3] = ((instruction >> 16) & 0xFF) / 255.0  # Third byte

    # Feature 4-5: PC-based features (for PC-relative addressing)
    features[4] = float((pc >> 0) & 0xFF) / 255.0
    features[5] = float((pc >> 8) & 0xFF) / 255.0

    # Feature 6: Rd register field (destination)
    # This helps identify load/store patterns
    features[6] = float((instruction >> 0) & 0x1F) / 31.0

    # Feature 7: Rn register field (first source)
    features[7] = float((instruction >> 5) & 0x1F) / 31.0

    # Feature 8: Size field (for load/store)
    # bit 31:30 = size (00=byte, 01=halfword, 10=word, 11=dword)
    features[8] = float((instruction >> 30) & 0x3) / 3.0

    # Feature 9: Instruction class (from bits 27:26)
    features[9] = float((instruction >> 26) & 0x3) / 3.0

    # Feature 10: SF bit (64-bit vs 32-bit)
    features[10] = float((instruction >> 31) & 0x1)

    # Feature 11: Padding (can be used for additional features later)
    features[11] = 0.0

    return features


def create_comprehensive_training_samples():
    """
    Create MUCH more comprehensive training samples.

    Key improvements:
    1. Cover ALL ARM64 instruction opcodes systematically
    2. Include register variations (different Rd, Rn values)
    3. Include immediate value variations
    4. Include PC variations
    """
    samples = []

    # KERNEL_ARITHMETIC = 0
    # ADD/SUB immediate (opcodes 0x91, 0xD1, 0x11, 0x51)
    for rd in [0, 5, 10, 20, 30]:  # Different destination registers
        for rn in [0, 5, 10, 20]:   # Different source registers
            for imm in [0, 1, 0x10, 0x100, 0x1000]:  # Different immediates
                # ADD Xd, Xn, #imm
                samples.append((0x91000000 | (imm << 10) | (rn << 5) | rd, 0))
                # SUB Xd, Xn, #imm
                samples.append((0xD1000000 | (imm << 10) | (rn << 5) | rd, 0))

    # MOVZ/MOVK/MOVN (opcodes 0xD2, 0xF2, 0x92)
    for rd in range(0, 31, 5):  # Every 5th register
        for imm in [0, 0x100, 0x1000, 0xFFFF, 0xFFFFFFFF]:  # Various immediates
            # MOVZ
            samples.append(((0xD2 << 24) | (0 << 21) | (imm << 5) | rd, 0))
            # MOVK
            samples.append(((0xF2 << 24) | (0 << 21) | (imm << 5) | rd, 0))
            # MOVN
            samples.append(((0x92 << 24) | (0 << 21) | (imm << 5) | rd, 0))

    # KERNEL_LOGICAL = 1
    # AND/ORR/EOR (opcodes 0x8A, 0xAA, 0xCA)
    for rd in [0, 5, 10, 20, 30]:
        for rn in [0, 5, 10, 20]:
            for rm in [0, 5, 10, 20]:  # Second source register
                # AND Xd, Xn, Xm
                samples.append((0x8A000000 | (rm << 16) | (rn << 5) | rd, 1))
                # ORR Xd, Xn, Xm
                samples.append((0xAA000000 | (rm << 16) | (rn << 5) | rd, 1))
                # EOR Xd, Xn, Xm
                samples.append((0xCA000000 | (rm << 16) | (rn << 5) | rd, 1))

    # KERNEL_LOADSTORE = 2
    # LDR/STR (opcodes 0xF9, 0xB9 for 64/32-bit)
    for rt in [0, 5, 10, 20]:  # Register to load/store
        for rn in [0, 5, 10, 20]:  # Base register
            for offset in [0, 8, 0x100, 0x1000]:  # Different offsets
                # LDR Xt, [Xn, #offset]
                samples.append(((0xF9 << 24) | ((offset >> 3) << 10) | (rn << 5) | rt, 2))
                # STR Xt, [Xn, #offset]
                samples.append(((0xF9 << 24) | (0b00 << 30) | ((offset >> 3) << 10) | (rn << 5) | rt, 2))

    # LDP/STP (load/store pair - opcodes 0xA8, 0xA9)
    for rt in [0, 5, 10]:
        for rt2 in [1, 6, 11]:
            for rn in [0, 5, 10]:
                # LDP Xt, Xt2, [Xn]
                samples.append(((0xA8 << 24) | (rn << 5) | rt | (rt2 << 10), 2))
                # STP Xt, Xt2, [Xn]
                samples.append(((0xA9 << 24) | (rn << 5) | rt | (rt2 << 10), 2))

    # KERNEL_BRANCH = 3
    # B unconditional (opcode 0x14)
    for offset in [0, 4, 0x1000, 0x100000]:  # Different branch targets
        samples.append(((0x14 << 24) | (offset >> 2), 3))

    # B.cond (opcode 0x54)
    for cond in range(16):  # All condition codes (EQ, NE, CS, etc.)
        for offset in [0, 4, 0x100]:
            samples.append(((0x54 << 24) | (cond << 0) | (offset >> 2), 3))

    # CBZ/CBNZ (opcodes 0xB4, 0xB5)
    for rt in [0, 5, 10, 20]:
        for offset in [0, 4, 0x100]:
            # CBZ Xt, #offset
            samples.append(((0xB4 << 24) | (offset >> 2) | (rt << 5), 3))
            # CBNZ Xt, #offset
            samples.append(((0xB5 << 24) | (offset >> 2) | (rt << 5), 3))

    # BL/BR/RET (opcodes 0x94, 0xD6)
    samples.append(((0x94 << 24), 3))  # BL #0
    samples.append(((0xD6 << 24) | (0x1F << 5) | (1 << 21), 3))  # BLR
    samples.append(((0xD6 << 24) | (0x1F << 5) | (0 << 21), 3))  # RET

    # KERNEL_MULDIV = 4
    # MADD/MSUB (opcode 0x9B)
    for rd in [0, 5, 10]:
        for rn in [0, 5, 10]:
            for rm in [0, 5, 10]:
                for ra in [0, 5, 10]:
                    # MADD Xd, Xn, Xm, Xa
                    samples.append(((0x9B << 24) | (ra << 10) | (rm << 16) | (rn << 5) | rd, 4))
                    # MSUB Xd, Xn, Xm, Xa
                    samples.append(((0x9B << 24) | (1 << 15) | (ra << 10) | (rm << 16) | (rn << 5) | rd, 4))

    # KERNEL_EXTEND_SHIFT = 5
    # SXTB/SXTH/SXTW (opcodes 0x13, 0x93)
    for rd in [0, 5, 10, 20]:
        for rn in [0, 5, 10]:
            # SXTW Xd, Wn
            samples.append(((0x93 << 24) | (0b11 << 13) | (0b000 << 10) | (rn << 5) | rd, 5))
            # SXTH Xd, Wn
            samples.append(((0x93 << 24) | (0b01 << 13) | (0b000 << 10) | (rn << 5) | rd, 5))
            # SXTB Wd, Wn
            samples.append(((0x13 << 24) | (0b00 << 13) | (0b000 << 10) | (rn << 5) | rd, 5))

    # ADRP/ADR (opcodes 0x90, 0xB0)
    for rd in [0, 5, 10, 20]:
        for imm in [0, 0x1000, 0x100000]:
            # ADRP Xd, #imm
            samples.append(((0x90 << 24) | ((imm >> 12) & 0x3) | (rd << 5), 5))
            # ADR Xd, #imm
            samples.append(((0xB0 << 24) | ((imm >> 2) & 0x7FFFF) | (rd << 5), 5))

    # KERNEL_SYSTEM = 6
    # HLT/SVC/DCPS1 (opcodes 0xD4)
    for imm in range(16):
        # HLT #imm
        samples.append(((0xD4 << 24) | (0b00 << 26) | (0b0010 << 21) | (imm << 5), 6))
        # SVC #imm
        samples.append(((0xD4 << 24) | (0b01 << 26) | (imm << 5), 6))
        # DCPS1 #imm
        samples.append(((0xD4 << 24) | (0b01 << 26) | (0b1010 << 21) | (imm << 5), 6))

    # Add PC variations for all samples
    expanded_samples = []
    for instruction, kernel in samples:
        # Base sample
        expanded_samples.append((instruction, 0, kernel))

        # Add PC variations
        for pc in [0x100, 0x1000, 0x10000, 0x100000]:
            expanded_samples.append((instruction, pc, kernel))

    print(f"Created {len(expanded_samples)} comprehensive training samples")
    return expanded_samples


def train_dispatch_network_v2(target_accuracy=1.0, max_epochs=1000):
    """
    Train the improved dispatch network to 100% accuracy.

    Args:
        target_accuracy: Target accuracy (default 1.0 = 100%)
        max_epochs: Maximum training epochs
    """
    print("=" * 80)
    print("  🧠 TRAINING DISPATCH NETWORK V2 - TARGET: 100% ACCURACY")
    print("=" * 80)
    print()

    # Step 1: Create training data
    print("Step 1: Creating comprehensive training dataset...")
    print("-" * 80)

    samples = create_comprehensive_training_samples()

    # Prepare data
    X = []
    y = []

    for instruction, pc, kernel in samples:
        features = extract_instruction_features(instruction, pc)
        X.append(features)
        y.append(kernel)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print()

    # Show sample distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Sample distribution:")
    kernel_names = ["ARITHMETIC", "LOGICAL", "LOADSTORE", "BRANCH",
                    "MULDIV", "EXTEND_SHIFT", "SYSTEM"]
    for kernel, count in zip(unique, counts):
        print(f"  Kernel {kernel} ({kernel_names[kernel]}): {count} samples")
    print()

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Split into train and validation sets
    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_indices = indices[:split]
    val_indices = indices[split:]

    X_train = X_tensor[train_indices]
    y_train = y_tensor[train_indices]
    X_val = X_tensor[val_indices]
    y_val = y_tensor[val_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()

    # Step 2: Create improved network
    print("Step 2: Creating improved dispatch network...")
    print("-" * 80)

    model = DispatchNetworkV2(input_size=12, hidden_size=16, output_size=7, dropout=0.1)
    print(f"Network: 12 inputs → 16 hidden → 7 outputs")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()

    # Step 3: Train with improved techniques
    print("Step 3: Training network...")
    print("-" * 80)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    best_val_accuracy = 0.0
    patience = 50
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # Training
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val).float().mean().item()

        # Check if best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_outputs = model(X_train)
            _, train_predicted = torch.max(train_outputs, 1)
            train_accuracy = (train_predicted == y_train).float().mean().item()

            print(f"Epoch [{epoch+1:4d}/{max_epochs}] "
                  f"Loss: {loss.item():.6f} | "
                  f"Train Acc: {train_accuracy*100:5.1f}% | "
                  f"Val Acc: {val_accuracy*100:5.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping if we hit target
        if val_accuracy >= target_accuracy:
            print()
            print(f"✅ Target accuracy {target_accuracy*100:.0f}% reached!")
            break

        # Early stopping if no improvement
        if patience_counter >= patience:
            print()
            print(f"Early stopping: No improvement for {patience} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final accuracy on all data
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        final_accuracy = (predicted == y_tensor).float().mean().item()

    print()
    print(f"✅ Training complete!")
    print(f"   Final accuracy: {final_accuracy*100:.2f}%")
    print(f"   Best validation accuracy: {best_val_accuracy*100:.2f}%")
    print()

    # Analyze errors
    if final_accuracy < 1.0:
        print("Error Analysis:")
        print("-" * 80)

        errors_by_kernel = defaultdict(list)
        for i in range(len(X_tensor)):
            if predicted[i] != y_tensor[i]:
                true_kernel = y_tensor[i].item()
                pred_kernel = predicted[i].item()
                inst = samples[i][0]
                errors_by_kernel[true_kernel].append((inst, pred_kernel))

        for kernel in sorted(errors_by_kernel.keys()):
            errors = errors_by_kernel[kernel]
            print(f"Kernel {kernel} ({kernel_names[kernel]}): {len(errors)} errors")

            # Group by opcode
            opcode_errors = defaultdict(int)
            for inst, pred_kernel in errors:
                opcode = (inst >> 24) & 0xFF
                opcode_errors[opcode] += 1

            for opcode, count in sorted(opcode_errors.items(), key=lambda x: -x[1])[:5]:
                print(f"  Opcode 0x{opcode:02X}: {count} mispredictions")
        print()

    # Step 4: Save weights
    print("Step 4: Saving trained weights...")
    print("-" * 80)

    # Extract weights in the format expected by the shader
    # Format: [input_hidden_weights, hidden_biases, hidden_output_weights, output_biases]

    # Input->Hidden weights: [12, 16] → flatten to [192]
    input_hidden_weights = model.input_weights.weight.data.detach().cpu().numpy().flatten()
    # Hidden biases: [16]
    hidden_biases = model.input_weights.bias.data.detach().cpu().numpy().flatten()
    # Hidden->Output weights: [16, 7] → flatten to [112]
    hidden_output_weights = model.output_weights.weight.data.detach().cpu().numpy().flatten()
    # Output biases: [7]
    output_biases = model.output_weights.bias.data.detach().cpu().numpy().flatten()

    # Concatenate: total = 192 + 16 + 112 + 7 = 327
    flat_weights = np.concatenate([
        input_hidden_weights,  # 192
        hidden_biases,          # 16
        hidden_output_weights,  # 112
        output_biases           # 7
    ])

    print(f"Weight shape: {flat_weights.shape} (expected (327,))")
    print(f"Weight range: [{flat_weights.min():.3f}, {flat_weights.max():.3f}]")

    # Save to file
    output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "dispatch_weights_v2_trained.npy"
    np.save(output_path, flat_weights)
    print(f"✅ Saved trained weights to: {output_path}")
    print()

    # Step 5: Test predictions
    print("Step 5: Testing predictions...")
    print("-" * 80)

    test_instructions = [
        (0xD2000C80, 0, "MOVZ X0, #100"),
        (0xD2001901, 0, "MOVZ X1, #200"),
        (0xAA000000, 1, "ORR X0, X0, X0"),
        (0xF9400000, 2, "LDR X0, [X0]"),
        (0x14000000, 3, "B #0"),
        (0x9B000000, 4, "MADD X0, X0, X0, X0"),
        (0x9340FC00, 5, "SXTW X0, W0"),
        (0xD4400000, 6, "HLT #0"),
    ]

    for inst, expected_kernel, description in test_instructions:
        features = extract_instruction_features(inst, 0)
        predicted_kernel = model.predict_kernel(features)
        status = "✅" if predicted_kernel == expected_kernel else "❌"
        print(f"  {status} {description}: predicted={predicted_kernel}, expected={expected_kernel}")

    print()
    print("=" * 80)
    print("  ✅ DISPATCH NETWORK V2 TRAINING COMPLETE!")
    print("=" * 80)
    print()

    return model, flat_weights


if __name__ == "__main__":
    # Train the network
    model, weights = train_dispatch_network_v2(
        target_accuracy=1.0,  # 100% accuracy
        max_epochs=1000
    )
