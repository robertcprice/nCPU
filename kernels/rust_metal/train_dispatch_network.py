#!/usr/bin/env python3
"""
Train the dispatch neural network on execution traces.

This script collects execution traces using the working neural dispatch system,
then trains the 8→8→7 network to accurately predict which kernel to use.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the GPU module
try:
    import kvrm_metal
except ImportError:
    print("❌ kvrm_metal module not found!")
    print("   Build with: maturin develop --release")
    sys.exit(1)


class DispatchNetwork(nn.Module):
    """
    Neural network for predicting which kernel to use.

    Architecture: 8 inputs → 8 hidden → 7 outputs

    Inputs (normalized 0-1):
    - opcode (1 byte)
    - instruction bytes (4 bytes, little-endian)
    - PC low bytes (2 bytes)

    Outputs (7 kernel types):
    - KERNEL_ARITHMETIC = 0
    - KERNEL_LOGICAL = 1
    - KERNEL_LOADSTORE = 2
    - KERNEL_BRANCH = 3
    - KERNEL_MULDIV = 4
    - KERNEL_EXTEND_SHIFT = 5
    - KERNEL_SYSTEM = 6
    """

    def __init__(self, input_size=8, hidden_size=8, output_size=7):
        super().__init__()

        # Input -> Hidden
        self.input_weights = nn.Linear(input_size, hidden_size)

        # Hidden -> Output
        self.output_weights = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: [batch, 8] normalized features

        Returns:
            [batch, 7] kernel logits
        """
        # Hidden layer with tanh activation
        hidden = torch.tanh(self.input_weights(x))

        # Output layer (no activation - raw logits)
        output = self.output_weights(hidden)

        return output

    def predict_kernel(self, opcode, instruction_bytes, pc):
        """
        Predict which kernel to use for a single instruction.

        Args:
            opcode: int (0-255)
            instruction_bytes: int (0-0xFFFFFFFF)
            pc: int

        Returns:
            int: kernel index (0-6)
        """
        # Normalize inputs (ensure int operations first)
        features = torch.zeros(8, dtype=torch.float32)
        features[0] = float(opcode) / 255.0
        features[1] = float((instruction_bytes >> 0) & 0xFF) / 255.0
        features[2] = float((instruction_bytes >> 8) & 0xFF) / 255.0
        features[3] = float((instruction_bytes >> 16) & 0xFF) / 255.0
        features[4] = float((instruction_bytes >> 24) & 0xFF) / 255.0
        features[5] = float((pc >> 0) & 0xFF) / 255.0
        features[6] = float((pc >> 8) & 0xFF) / 255.0
        features[7] = 0.0  # Padding

        with torch.no_grad():
            logits = self.forward(features.unsqueeze(0))
            kernel = torch.argmax(logits, dim=1).item()

        return kernel


def create_training_samples():
    """
    Create training samples from known ARM64 instructions.

    Returns:
        List of (features, kernel_index) tuples
    """
    samples = []

    # KERNEL_ARITHMETIC = 0
    arithmetic_ops = [
        # ADD immediate
        (0x91000000, 0x91, 0),  # ADD X0, X0, #0
        (0x91000400, 0x91, 0),  # ADD X0, X1, #1
        (0x91003C00, 0x91, 0),  # ADD X0, X1, #0xF
        # SUB immediate
        (0xD1000000, 0xD1, 0),  # SUB X0, X0, #0
        (0xD1000400, 0xD1, 0),  # SUB X0, X1, #1
        # MOVZ
        (0xD2800000, 0xD2, 0),  # MOVZ X0, #0
        (0xD2800540, 0xD2, 0),  # MOVZ X0, #42
        (0xD2000C80, 0xD2, 0),  # MOVZ X0, #100
        # MOVK
        (0xF2800000, 0xF2, 0),  # MOVK X0, #0
        # MOVN
        (0x92800000, 0x92, 0),  # MOVN X0, #0
    ]

    # KERNEL_LOGICAL = 1
    logical_ops = [
        (0x8A000000, 0x8A, 1),  # AND X0, X0, X0
        (0xAA000000, 0xAA, 1),  # ORR X0, X0, X0
        (0xCA000000, 0xCA, 1),  # EOR X0, X0, X0
        (0x8A000001, 0x8A, 1),  # AND X1, X0, X0
        (0xAA000001, 0xAA, 1),  # ORR X1, X0, X0
        (0xCA000001, 0xCA, 1),  # EOR X1, X0, X0
    ]

    # KERNEL_LOADSTORE = 2
    loadstore_ops = [
        (0xF9400000, 0xF9, 2),  # LDR X0, [X0]
        (0xF9000000, 0xF9, 2),  # STR X0, [X0]
        (0xB9400000, 0xB9, 2),  # LDR W0, [X0]
        (0xB9000000, 0xB9, 2),  # STR W0, [X0]
        (0x39400000, 0x39, 2),  # LDRB W0, [X0]
        (0x39000000, 0x39, 2),  # STRB W0, [X0]
        (0xF94003E0, 0xF9, 2),  # LDR X0, [SP]
        (0xF90003E0, 0xF9, 2),  # STR X0, [SP]
        (0xA8C00000, 0xA8, 2),  # LDP X0, X1, [X0]
        (0xA9000000, 0xA9, 2),  # STP X0, X1, [X0]
    ]

    # KERNEL_BRANCH = 3
    branch_ops = [
        (0x14000000, 0x14, 3),  # B #0
        (0x14000004, 0x14, 3),  # B #1
        (0x94000000, 0x94, 3),  # BL #0
        (0xD61F0000, 0xD6, 3),  # BR X0
        (0xD63F0000, 0xD6, 3),  # BLR X0
        (0xD65F03C0, 0xD6, 3),  # RET
        (0xB4000000, 0xB4, 3),  # CBZ X0, #0
        (0xB5000000, 0xB5, 3),  # CBNZ X0, #0
        (0x54000000, 0x54, 3),  # B.EQ #0
        (0x54000001, 0x54, 3),  # B.NE #0
    ]

    # KERNEL_MULDIV = 4
    muldiv_ops = [
        (0x9B000000, 0x9B, 4),  # MADD X0, X0, X0, X0
        (0x9B007C00, 0x9B, 4),  # MADD X0, X0, X0, X31
        (0x9B008000, 0x9B, 4),  # MSUB X0, X0, X0, X0
    ]

    # KERNEL_EXTEND_SHIFT = 5
    extend_shift_ops = [
        (0x9340FC00, 0x93, 5),  # SXTW X0, W0
        (0x9340FC01, 0x93, 5),  # SXTW X1, W1
        (0x13000000, 0x13, 5),  # SXTB W0, W0
        (0x93405C00, 0x93, 5),  # SXTB X0, W0
        (0x93407C00, 0x93, 5),  # SXTH X0, W0
        (0x90000000, 0x90, 5),  # ADRP X0, #0
        (0xB0000000, 0xB0, 5),  # ADR X0, #0
    ]

    # KERNEL_SYSTEM = 6
    system_ops = [
        (0xD4400000, 0xD4, 6),  # HLT #0
        (0xD4A00001, 0xD4, 6),  # DCPS1 #0
        (0xD4000001, 0xD4, 6),  # SVC #0
    ]

    # Combine all operations
    all_ops = (
        arithmetic_ops +
        logical_ops +
        loadstore_ops +
        branch_ops +
        muldiv_ops +
        extend_shift_ops +
        system_ops
    )

    # Create samples with variations
    for instruction, opcode, kernel in all_ops:
        # Add base sample at PC=0
        samples.append((instruction, opcode, 0, kernel))

        # Add variation at different PCs (for PC-based features)
        for pc_offset in [0x100, 0x1000, 0x10000]:
            samples.append((instruction, opcode, pc_offset, kernel))

    print(f"Created {len(samples)} training samples")
    return samples


def encode_instruction_features(instruction, pc):
    """
    Encode instruction and PC into 8 normalized features.

    Returns:
        numpy array of shape (8,)
    """
    features = np.zeros(8, dtype=np.float32)

    # Normalize to 0-1 range
    features[0] = ((instruction >> 24) & 0xFF) / 255.0  # opcode byte
    features[1] = ((instruction >> 0) & 0xFF) / 255.0
    features[2] = ((instruction >> 8) & 0xFF) / 255.0
    features[3] = ((instruction >> 16) & 0xFF) / 255.0
    features[4] = ((instruction >> 24) & 0xFF) / 255.0
    features[5] = ((pc >> 0) & 0xFF) / 255.0
    features[6] = ((pc >> 8) & 0xFF) / 255.0
    features[7] = 0.0  # Padding

    return features


def train_dispatch_network(epochs=100, batch_size=32, learning_rate=0.01):
    """
    Train the dispatch network.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print("=" * 80)
    print("  🧠 TRAINING DISPATCH NETWORK")
    print("=" * 80)
    print()

    # Step 1: Create training data
    print("Step 1: Creating training dataset...")
    print("-" * 80)

    samples = create_training_samples()

    # Prepare data
    X = []  # Features
    y = []  # Kernel labels

    for instruction, opcode, pc, kernel in samples:
        features = encode_instruction_features(instruction, pc)
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
    for kernel, count in zip(unique, counts):
        print(f"  Kernel {kernel}: {count} samples")
    print()

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Step 2: Create network
    print("Step 2: Creating dispatch network...")
    print("-" * 80)

    model = DispatchNetwork(input_size=8, hidden_size=8, output_size=7)
    print(f"Network: 8 inputs → 8 hidden → 7 outputs")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print()

    # Step 3: Train
    print("Step 3: Training network...")
    print("-" * 80)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {loss.item():.6f} Accuracy: {accuracy.item()*100:5.1f}%")

    # Final accuracy
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        final_accuracy = (predicted == y_tensor).float().mean()

    print()
    print(f"✅ Training complete! Final accuracy: {final_accuracy.item()*100:.1f}%")
    print()

    # Step 4: Save weights
    print("Step 4: Saving trained weights...")
    print("-" * 80)

    # Extract weights in the format expected by the shader
    # Format: [input_hidden_weights, hidden_biases, hidden_output_weights, output_biases]

    # Input->Hidden weights: [8, 8] → flatten to [64]
    input_hidden_weights = model.input_weights.weight.data.detach().numpy().flatten()
    # Hidden biases: [8]
    hidden_biases = model.input_weights.bias.data.detach().numpy().flatten()
    # Hidden->Output weights: [8, 7] → flatten to [56]
    hidden_output_weights = model.output_weights.weight.data.detach().numpy().flatten()
    # Output biases: [7]
    output_biases = model.output_weights.bias.data.detach().numpy().flatten()

    # Concatenate: total = 64 + 8 + 56 + 7 = 135
    flat_weights = np.concatenate([
        input_hidden_weights,  # 64
        hidden_biases,          # 8
        hidden_output_weights,  # 56
        output_biases           # 7
    ])

    print(f"Weight shape: {flat_weights.shape} (expected (135,))")
    print(f"Weight range: [{flat_weights.min():.3f}, {flat_weights.max():.3f}]")

    # Save to file
    output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "dispatch_weights_trained.npy"
    np.save(output_path, flat_weights)
    print(f"✅ Saved trained weights to: {output_path}")
    print()

    # Step 5: Test a few predictions
    print("Step 5: Testing predictions...")
    print("-" * 80)

    test_instructions = [
        (0xD2000C80, 0, "MOVZ X0, #100"),
        (0xD2001901, 1, "MOVZ X1, #200"),
        (0xAA000000, 1, "ORR X0, X0, X0"),
        (0xF9400000, 2, "LDR X0, [X0]"),
        (0x14000000, 3, "B #0"),
    ]

    for inst, expected_kernel, description in test_instructions:
        predicted_kernel = model.predict_kernel(
            (inst >> 24) & 0xFF,
            inst,
            0
        )
        status = "✅" if predicted_kernel == expected_kernel else "❌"
        print(f"  {status} {description}: predicted={predicted_kernel}, expected={expected_kernel}")

    print()
    print("=" * 80)
    print("  ✅ DISPATCH NETWORK TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("Next: Test the trained weights in the neural dispatch system!")
    print("=" * 80)
    print()

    return model


if __name__ == "__main__":
    # Train the network
    model = train_dispatch_network(
        epochs=200,
        batch_size=32,
        learning_rate=0.01
    )
