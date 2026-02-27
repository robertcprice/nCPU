#!/usr/bin/env python3
"""
Test pure neural dispatch with 100% accurate embedding weights.

This script tests the embedding-based neural dispatch without fallback.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

# Load the trained model
model_path = Path(__file__).parent / "weights" / "dispatch_embedding_model_100pct.pt"

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("   Run: python3 train_dispatch_v3.py")
    sys.exit(1)

print("=" * 80)
print("  🧠 TESTING PURE NEURAL DISPATCH - 100% ACCURACY")
print("=" * 80)
print()

# Define the network (must match training)
class OpcodeEmbeddingNetwork(nn.Module):
    def __init__(self, opcode_size=256, embed_dim=32, hidden_size=32, output_size=7):
        super().__init__()
        self.opcode_embedding = nn.Embedding(opcode_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim + 12, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, opcode_idx, features):
        opcode_embed = self.opcode_embedding(opcode_idx)
        combined = torch.cat([opcode_embed, features], dim=1)
        hidden = torch.relu(self.fc1(combined))
        hidden = torch.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        return output

    def predict_kernel(self, instruction, pc):
        with torch.no_grad():
            opcode = (instruction >> 24) & 0xFF
            features = extract_instruction_features(instruction, pc)

            opcode_tensor = torch.tensor([opcode], dtype=torch.long)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)

            logits = self.forward(opcode_tensor, features_tensor)
            kernel = torch.argmax(logits, dim=1).item()

        return kernel

def extract_instruction_features(instruction, pc):
    """Extract features (must match training)"""
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

    # Immediate/part field
    features[10] = float((instruction >> 10) & 0xFFF) / 4095.0

    # Padding
    features[11] = 0.0

    return features

# Load model
print(f"Loading trained model from: {model_path}")
model = OpcodeEmbeddingNetwork()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("✅ Model loaded")
print()

# Test instructions covering all 7 kernels
test_cases = [
    # (instruction, expected_kernel, description)
    (0xD2000C80, 0, "MOVZ X0, #100 (ARITHMETIC)"),
    (0x91000000, 0, "ADD X0, X0, #0 (ARITHMETIC)"),
    (0xD1000000, 0, "SUB X0, X0, #0 (ARITHMETIC)"),

    (0xAA000000, 1, "ORR X0, X0, X0 (LOGICAL)"),
    (0x8A000000, 1, "AND X0, X0, X0 (LOGICAL)"),
    (0xCA000000, 1, "EOR X0, X0, X0 (LOGICAL)"),

    (0xF9400000, 2, "LDR X0, [X0] (LOADSTORE)"),
    (0xF9000000, 2, "STR X0, [X0] (LOADSTORE)"),
    (0xA8C00000, 2, "LDP X0, X1, [X0] (LOADSTORE)"),

    (0x14000000, 3, "B #0 (BRANCH)"),
    (0x94000000, 3, "BL #0 (BRANCH)"),
    (0xD65F03C0, 3, "RET (BRANCH)"),
    (0xB4000000, 3, "CBZ X0, #0 (BRANCH)"),

    (0x9B000000, 4, "MADD X0, X0, X0, X0 (MULDIV)"),

    (0x9340FC00, 5, "SXTW X0, W0 (EXTEND_SHIFT)"),
    (0x90000000, 5, "ADRP X0, #0 (EXTEND_SHIFT)"),
    (0xB0000000, 5, "ADR X0, #0 (EXTEND_SHIFT)"),

    (0xD4400000, 6, "HLT #0 (SYSTEM)"),
    (0xD4000001, 6, "SVC #0 (SYSTEM)"),
]

kernel_names = [
    "ARITHMETIC", "LOGICAL", "LOADSTORE", "BRANCH",
    "MULDIV", "EXTEND_SHIFT", "SYSTEM"
]

print("Testing pure neural predictions (NO FALLBACK):")
print("-" * 80)

correct = 0
total = 0

for inst, expected, description in test_cases:
    predicted = model.predict_kernel(inst, 0)
    status = "✅" if predicted == expected else "❌"

    if predicted == expected:
        correct += 1
    total += 1

    print(f"  {status} {description}")
    print(f"      Instruction: 0x{inst:08X}")
    print(f"      Predicted: {predicted} ({kernel_names[predicted]}) | Expected: {expected} ({kernel_names[expected]})")
    print()

print("-" * 80)
print(f"Results: {correct}/{total} correct ({100*correct/total:.1f}%)")
print("=" * 80)
print()

if correct == total:
    print("✅ 100% ACCURACY ACHIEVED!")
    print()
    print("The embedding-based neural dispatch achieves perfect accuracy.")
    print("This means we can REMOVE the opcode-based fallback entirely!")
    print()
    print("Next steps:")
    print("  1. Update neural_dispatch.rs to use embedding shader")
    print("  2. Remove get_kernel_from_opcode() fallback")
    print("  3. Test on real workloads (DOOM, Linux boot)")
else:
    print(f"⚠️  Accuracy: {100*correct/total:.1f}%")
    print()
    print("Some predictions are still incorrect. Need more training or better features.")

print()
print("Kernel prediction breakdown:")
for kernel_idx, kernel_name in enumerate(kernel_names):
    kernel_tests = [t for t in test_cases if t[1] == kernel_idx]
    if kernel_tests:
        kernel_correct = sum(1 for inst, exp, desc in kernel_tests if model.predict_kernel(inst, 0) == exp)
        print(f"  {kernel_name}: {kernel_correct}/{len(kernel_tests)} correct")
