#!/usr/bin/env python3
"""
TRAIN NEURAL ITERATION PREDICTOR ON REAL DATA
==============================================

Train the iteration predictor on REAL loop data with:
- ACTUAL pattern embeddings from the pattern recognizer (not random noise)
- REAL register values from actual execution
- DIVERSE loop types (MEMSET, MEMCPY, POLLING, BUBBLE_SORT)

This replaces the synthetic training with real data from actual loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path


class IterationPredictor(nn.Module):
    """
    Predict loop iterations from:
    - Loop body instructions (encoded as real embeddings)
    - Current register values (counter, limit)
    - Loop start PC (context)
    """

    def __init__(self, d_model=128):
        super().__init__()

        # Input: instruction encoding + normalized register values
        inst_encoding_dim = 128  # From pattern recognizer
        reg_dim = 2  # normalized counter + limit (scalar values)
        pc_dim = 1  # normalized PC (scalar value)

        input_dim = inst_encoding_dim + reg_dim + pc_dim  # 131 total

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.ReLU()  # Iterations can't be negative
        )

    def forward(self, inst_encoding, counter, limit, pc):
        """
        Args:
            inst_encoding: Encoded loop body (batch, d_model)
            counter: Loop counter value (batch,)
            limit: Loop limit value (batch,)
            pc: Loop start address (batch,)

        Returns:
            Predicted iterations (batch, 1)
        """
        # Normalize inputs
        counter_norm = torch.log2(torch.abs(counter) + 1) / 64  # Log scale, normalized
        limit_norm = torch.log2(torch.abs(limit) + 1) / 64
        pc_norm = (pc.float() / 0x20000) * 2 - 1  # Normalize to [-1, 1]

        # Concatenate features
        x = torch.cat([
            inst_encoding,
            counter_norm.unsqueeze(1),
            limit_norm.unsqueeze(1),
            pc_norm.unsqueeze(1)
        ], dim=1)

        # Predict iterations
        iterations = self.predictor(x)
        return iterations


class RealLoopDataset(Dataset):
    """Dataset for training on REAL loop data."""

    def __init__(self, data_file: str):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert to tensors
        inst_encoding = torch.tensor(item['inst_encoding'], dtype=torch.float32)
        counter = torch.tensor(item['counter'], dtype=torch.float32)
        limit = torch.tensor(item['limit'], dtype=torch.float32)
        pc = torch.tensor(item['pc'], dtype=torch.float32)
        iterations = torch.tensor(item['iterations'], dtype=torch.float32)

        return inst_encoding, counter, limit, pc, iterations


def train_iteration_predictor_real(data_file='models/combined_loop_data.json', epochs=100, batch_size=32):
    """Train the iteration predictor on REAL data."""

    print("="*70)
    print(" TRAINING NEURAL ITERATION PREDICTOR ON REAL DATA")
    print("="*70)
    print()

    # Check if training data exists
    data_path = Path(data_file)
    if not data_path.exists():
        print(f"❌ Training data not found at {data_file}")
        print("   Run generate_diverse_loop_data.py first!")
        return

    # Load data
    print(f"Loading training data from {data_file}...")
    dataset = RealLoopDataset(str(data_file))
    print(f"✅ Loaded {len(dataset)} training samples")
    print()

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using MPS (Apple Silicon GPU) acceleration")
    elif torch.cuda.is_available():
        print("✅ Using CUDA (NVIDIA GPU) acceleration")
    else:
        print("⚠️  Using CPU (no GPU acceleration)")

    model = IterationPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for inst_encoding, counter, limit, pc, iterations in train_loader:
            inst_encoding = inst_encoding.to(device)
            counter = counter.to(device)
            limit = limit.to(device)
            pc = pc.to(device)
            iterations = iterations.to(device).unsqueeze(1)

            optimizer.zero_grad()
            predictions = model(inst_encoding, counter, limit, pc)
            loss = criterion(predictions, iterations)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inst_encoding, counter, limit, pc, iterations in val_loader:
                inst_encoding = inst_encoding.to(device)
                counter = counter.to(device)
                limit = limit.to(device)
                pc = pc.to(device)
                iterations = iterations.to(device).unsqueeze(1)

                predictions = model(inst_encoding, counter, limit, pc)
                loss = criterion(predictions, iterations)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = Path('models/iteration_predictor_real_best.pt')
            model_path.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch
            }, model_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss={train_loss:.2f}, Val Loss={val_loss:.2f}")

    print()
    print(f"✅ Training complete! Best val loss: {best_val_loss:.2f}")
    print(f"✅ Model saved to models/iteration_predictor_real_best.pt")
    print()

    # Test the model
    print("="*70)
    print(" TESTING MODEL")
    print("="*70)
    print()

    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(val_dataset))):
            inst_encoding, counter, limit, pc, target_iterations = val_dataset[i]
            inst_encoding = inst_encoding.unsqueeze(0).to(device)
            counter = counter.unsqueeze(0).to(device)
            limit = limit.unsqueeze(0).to(device)
            pc = pc.unsqueeze(0).to(device)

            prediction = model(inst_encoding, counter, limit, pc)
            predicted = prediction[0, 0].item()

            print(f"Sample {i+1}:")
            print(f"  Actual iterations: {target_iterations.item():.0f}")
            print(f"  Predicted iterations: {predicted:.0f}")
            print(f"  Error: {abs(predicted - target_iterations.item()):.0f} ({abs(predicted - target_iterations.item())/target_iterations.item()*100:.1f}%)")
            print()

    print("="*70)
    print(" TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    train_iteration_predictor_real(epochs=100, batch_size=32)
