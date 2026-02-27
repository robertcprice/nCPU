#!/usr/bin/env python3
"""
TRAIN NEURAL ITERATION PREDICTOR
=================================

Train a neural network to predict loop iterations from:
- Loop body instructions (encoded)
- Current register values (counter, limit)
- Loop start PC (context)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json


class IterationPredictor(nn.Module):
    """
    Predict loop iterations from:
    - Loop body instructions (encoded)
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


class LoopIterationDataset(Dataset):
    """Dataset for training loop iteration predictor."""

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


def generate_training_data():
    """
    Generate synthetic training data for iteration prediction.

    In production, this would come from actual loop traces.
    """

    data = []

    # Generate synthetic loop data
    for i in range(1000):
        # Random loop parameters
        counter = np.random.randint(0, 10000)
        limit = np.random.randint(counter, counter + 5000)
        iterations = limit - counter
        pc = np.random.randint(0x10000, 0x12000)

        # Random instruction encoding (simulate pattern recognizer output)
        inst_encoding = np.random.randn(128).astype(np.float32)

        data.append({
            'inst_encoding': inst_encoding.tolist(),
            'counter': int(counter),
            'limit': int(limit),
            'pc': int(pc),
            'iterations': int(iterations)
        })

    # Save to file
    output_file = Path('models/loop_iterations_train.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Generated {len(data)} training samples to {output_file}")
    return str(output_file)


def train_iteration_predictor(epochs=50, batch_size=32):
    """Train the iteration predictor."""

    print("="*70)
    print(" TRAINING NEURAL ITERATION PREDICTOR")
    print("="*70)
    print()

    # Check if training data exists
    data_file = Path('models/loop_iterations_train.json')
    if not data_file.exists():
        print("Generating synthetic training data...")
        data_file = generate_training_data()

    # Load data
    print(f"Loading training data from {data_file}...")
    dataset = LoopIterationDataset(str(data_file))

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
    model = IterationPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Training on device: {device}")
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
            model_path = Path('models/iteration_predictor_best.pt')
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
    print(f"✅ Model saved to models/iteration_predictor_best.pt")
    print()
    print("="*70)
    print(" TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    train_iteration_predictor(epochs=50, batch_size=32)
