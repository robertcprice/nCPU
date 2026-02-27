#!/usr/bin/env python3
"""
Neural OoO Scheduler Training

Trains the differentiable neural OoO scheduler to learn optimal instruction scheduling.
Uses PyTorch for gradient computation with Metal backend for fast inference.

Architecture:
- Instruction Encoder: Opcode embeddings + feature extraction
- Dependency Predictor: Soft dependency matrix prediction
- Scheduler: Attention-based instruction selection

Training Signal:
- Minimize cycles (maximize parallelism)
- Reward correct execution ordering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
import kvrm_metal as metal


class NeuralOoOScheduler(nn.Module):
    """
    Differentiable Neural OoO Scheduler

    Learns to predict instruction dependencies and optimal execution order.
    """

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64, num_opcodes: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Instruction embedding (opcode -> vector)
        self.opcode_embed = nn.Embedding(num_opcodes, embed_dim)

        # Feature encoder for additional instruction bits
        self.feature_encoder = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.ReLU(),
        )

        # Dependency prediction network
        self.dep_query = nn.Linear(embed_dim, hidden_dim)
        self.dep_key = nn.Linear(embed_dim, hidden_dim)
        self.dep_value = nn.Linear(embed_dim, hidden_dim)

        # Scheduler attention
        self.scheduler_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learned hazard weights (RAW, WAR, WAW, FLAG, BIAS)
        self.hazard_weights = nn.Parameter(torch.tensor([5.0, 0.5, 3.0, 5.0, -2.0]))

        # Temperature for Gumbel-softmax (annealed during training)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def extract_features(self, inst: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """Extract 12-dimensional feature vector from instruction"""
        features = torch.zeros(inst.shape[0], 12, device=inst.device)

        # Instruction category (top 4 bits)
        features[:, 0] = ((inst >> 28) & 0xF).float() / 15.0
        # Structural patterns
        features[:, 1] = (inst & 0xFF).float() / 255.0
        features[:, 2] = ((inst >> 16) & 0xFF).float() / 255.0
        # PC features
        features[:, 3] = (pc & 0xFF).float() / 255.0
        features[:, 4] = ((pc >> 8) & 0xFF).float() / 255.0
        # Register fields
        features[:, 5] = (inst & 0x1F).float() / 31.0
        features[:, 6] = ((inst >> 5) & 0x1F).float() / 31.0
        # Size field
        features[:, 7] = ((inst >> 30) & 0x3).float() / 3.0
        # Instruction class
        features[:, 8] = ((inst >> 26) & 0x3).float() / 3.0
        # SF bit
        features[:, 9] = ((inst >> 31) & 0x1).float()
        # Immediate field
        features[:, 10] = ((inst >> 10) & 0xFFF).float() / 4095.0
        # Padding
        features[:, 11] = 0.0

        return features

    def encode_instructions(self, instructions: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """Encode instructions into embedding space"""
        # Extract opcodes
        opcodes = ((instructions >> 24) & 0xFF).long()

        # Get opcode embeddings
        opcode_embeds = self.opcode_embed(opcodes)  # [batch, window, embed_dim]

        # Get feature encodings
        # pc: [batch] -> [batch, window] -> [batch*window] to match flattened instructions
        batch_size, window_size = instructions.shape
        pc_expanded = pc.unsqueeze(-1).expand(batch_size, window_size).reshape(-1)
        features = self.extract_features(instructions.view(-1), pc_expanded)
        feature_embeds = self.feature_encoder(features).view(*instructions.shape, -1)

        # Combine
        return opcode_embeds + feature_embeds

    def predict_dependencies(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict soft dependency matrix using attention.

        Returns: [batch, window, window] matrix where dep[i][j] = P(j depends on i)
        """
        batch_size, window_size, _ = embeddings.shape

        # Project to query/key space
        queries = self.dep_query(embeddings)  # [batch, window, hidden]
        keys = self.dep_key(embeddings)

        # Attention scores
        scale = self.hidden_dim ** -0.5
        scores = torch.bmm(queries, keys.transpose(1, 2)) * scale  # [batch, window, window]

        # Mask: can't depend on self or future instructions
        mask = torch.triu(torch.ones(window_size, window_size, device=embeddings.device), diagonal=0)
        scores = scores.masked_fill(mask.bool().unsqueeze(0), float('-inf'))

        # Sigmoid to get dependency probabilities
        dep_matrix = torch.sigmoid(scores)
        dep_matrix = dep_matrix * (1 - mask.unsqueeze(0))  # Zero out upper triangle

        return dep_matrix

    def predict_dependencies_hybrid(self, instructions: torch.Tensor) -> torch.Tensor:
        """
        Hybrid dependency prediction using learned hazard weights.
        Faster and more interpretable than pure attention.
        """
        batch_size, window_size = instructions.shape

        # Extract register fields
        rd = (instructions & 0x1F)  # [batch, window]
        rn = ((instructions >> 5) & 0x1F)
        rm = ((instructions >> 16) & 0x1F)

        dep_matrix = torch.zeros(batch_size, window_size, window_size, device=instructions.device)

        for i in range(window_size):
            for j in range(i + 1, window_size):
                # RAW hazard: j reads what i writes
                raw = ((rd[:, i] == rn[:, j]) | (rd[:, i] == rm[:, j])) & (rd[:, i] < 31)

                # WAW hazard: both write same register
                waw = (rd[:, i] == rd[:, j]) & (rd[:, i] < 31) & (rd[:, j] < 31)

                # Score using learned weights
                score = (self.hazard_weights[0] * raw.float() +
                        self.hazard_weights[2] * waw.float() +
                        self.hazard_weights[4])  # Bias

                dep_matrix[:, i, j] = torch.sigmoid(score)

        return dep_matrix

    def schedule_execution(self, embeddings: torch.Tensor, dep_matrix: torch.Tensor,
                          executed_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute readiness scores for each instruction.

        Returns: [batch, window] readiness scores
        """
        batch_size, window_size, _ = embeddings.shape

        # Compute readiness based on dependencies
        # Ready if all dependencies are satisfied (executed)
        # Soft version: readiness = product of (1 - dep[j][i] * (1 - executed[j]))

        readiness = torch.ones(batch_size, window_size, device=embeddings.device)

        for i in range(window_size):
            for j in range(i):  # j < i (potential producers)
                # If j not executed, dependency hurts readiness
                dep_contribution = dep_matrix[:, j, i] * (1 - executed_mask[:, j])
                readiness[:, i] = readiness[:, i] * (1 - dep_contribution)

        # Already executed instructions not ready
        readiness = readiness * (1 - executed_mask)

        return readiness

    def forward(self, instructions: torch.Tensor, pc: torch.Tensor,
                use_hybrid: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            instructions: [batch, window] instruction words
            pc: [batch] or [batch, window] program counter

        Returns:
            dep_matrix: [batch, window, window] soft dependencies
            readiness: [batch, window] initial readiness scores
        """
        # Encode instructions
        embeddings = self.encode_instructions(instructions, pc)

        # Predict dependencies
        if use_hybrid:
            dep_matrix = self.predict_dependencies_hybrid(instructions)
        else:
            dep_matrix = self.predict_dependencies(embeddings)

        # Initial readiness (no instructions executed yet)
        executed_mask = torch.zeros(instructions.shape[0], instructions.shape[1],
                                   device=instructions.device)
        readiness = self.schedule_execution(embeddings, dep_matrix, executed_mask)

        return dep_matrix, readiness

    def export_weights(self) -> np.ndarray:
        """Export weights to flat array for Metal backend.

        Metal shader expects exactly 4 weights: [RAW, WAW, FLAG, BIAS]
        PyTorch hazard_weights: [RAW, WAR, WAW, FLAG, BIAS] (5 values)
        We map: [0, 2, 3, 4] -> [RAW, WAW, FLAG, BIAS]
        """
        hw = self.hazard_weights.detach().cpu().numpy()
        # Metal expects: [RAW, WAW, FLAG, BIAS]
        # PyTorch has: [RAW=0, WAR=1, WAW=2, FLAG=3, BIAS=4]
        return np.array([hw[0], hw[2], hw[3], hw[4]], dtype=np.float32)


def generate_training_data(num_samples: int = 1000, window_size: int = 8
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic training data with known dependencies.

    Returns:
        instructions: [num_samples, window_size]
        pc: [num_samples]
        labels: [num_samples, window_size, window_size] ground truth dependencies
    """
    instructions = torch.randint(0, 0xFFFFFFFF, (num_samples, window_size), dtype=torch.long)
    pc = torch.arange(num_samples) * 32

    # Generate ground truth dependencies based on register hazards
    labels = torch.zeros(num_samples, window_size, window_size)

    for b in range(num_samples):
        for i in range(window_size):
            rd_i = int(instructions[b, i] & 0x1F)
            for j in range(i + 1, window_size):
                rn_j = int((instructions[b, j] >> 5) & 0x1F)
                rm_j = int((instructions[b, j] >> 16) & 0x1F)
                rd_j = int(instructions[b, j] & 0x1F)

                # RAW hazard
                if rd_i < 31 and (rd_i == rn_j or rd_i == rm_j):
                    labels[b, i, j] = 1.0
                # WAW hazard
                if rd_i < 31 and rd_j < 31 and rd_i == rd_j:
                    labels[b, i, j] = 1.0

    return instructions, pc, labels


def train_neural_ooo(epochs: int = 100, batch_size: int = 64,
                    learning_rate: float = 1e-3, save_path: str = "neural_ooo_weights.pt"):
    """Train the Neural OoO scheduler"""
    print("=" * 60)
    print("Neural OoO Scheduler Training")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = NeuralOoOScheduler().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Generate training data
    print("\nGenerating training data...")
    train_instructions, train_pc, train_labels = generate_training_data(10000)
    train_instructions = train_instructions.to(device)
    train_pc = train_pc.to(device)
    train_labels = train_labels.to(device)

    val_instructions, val_pc, val_labels = generate_training_data(1000)
    val_instructions = val_instructions.to(device)
    val_pc = val_pc.to(device)
    val_labels = val_labels.to(device)

    print(f"Training samples: {len(train_instructions)}")
    print(f"Validation samples: {len(val_instructions)}")

    # Training loop
    print("\nTraining...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle
        perm = torch.randperm(len(train_instructions))

        for i in range(0, len(train_instructions), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_inst = train_instructions[batch_idx]
            batch_pc = train_pc[batch_idx]
            batch_labels = train_labels[batch_idx]

            optimizer.zero_grad()

            # Forward pass
            dep_matrix, readiness = model(batch_inst, batch_pc, use_hybrid=True)

            # Loss: BCE for dependency prediction
            loss = F.binary_cross_entropy(dep_matrix, batch_labels)

            # Add regularization for parallelism (encourage finding independent instructions)
            parallel_bonus = -0.1 * (1 - dep_matrix).mean()
            loss = loss + parallel_bonus

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_dep, val_ready = model(val_instructions, val_pc, use_hybrid=True)
            val_loss = F.binary_cross_entropy(val_dep, val_labels).item()

            # Compute accuracy
            pred = (val_dep > 0.5).float()
            accuracy = (pred == val_labels).float().mean().item()

        avg_loss = total_loss / num_batches

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss={avg_loss:.4f} Val={val_loss:.4f} Acc={accuracy*100:.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")

    return model


def benchmark_neural_ooo(model: Optional[NeuralOoOScheduler] = None):
    """Benchmark Neural OoO against rule-based OoO"""
    print("\n" + "=" * 60)
    print("Neural OoO Benchmark")
    print("=" * 60)

    # Initialize CPUs
    print("\nInitializing execution engines...")

    try:
        bb_cpu = metal.BBCacheMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"BBCache init failed: {e}")
        bb_cpu = None

    try:
        ooo_cpu = metal.OoOMetalCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"OoO init failed: {e}")
        ooo_cpu = None

    try:
        neural_cpu = metal.NeuralOoOCPU(memory_size=4*1024*1024, cycles_per_batch=10_000_000)
    except Exception as e:
        print(f"NeuralOoO init failed: {e}")
        neural_cpu = None

    # Load trained weights if model provided
    if model is not None and neural_cpu is not None:
        print("Loading trained weights into Neural OoO...")
        weights = model.export_weights()
        # Pad to match expected size
        expected_size = neural_cpu.weight_count()
        if len(weights) < expected_size:
            weights = np.pad(weights, (0, expected_size - len(weights)))
        neural_cpu.load_weights(weights.tolist())

    # Test program: Parallel ADDs (best case for OoO)
    print("\nTest: Parallel Independent ADDs (50K iterations)")
    code = bytearray()
    # 4 independent ADDs
    code.extend((0x21, 0x04, 0x00, 0x91))  # ADD X1, X1, #1
    code.extend((0x42, 0x08, 0x00, 0x91))  # ADD X2, X2, #2
    code.extend((0x63, 0x0C, 0x00, 0x91))  # ADD X3, X3, #3
    code.extend((0x84, 0x10, 0x00, 0x91))  # ADD X4, X4, #4
    code.extend((0x00, 0x04, 0x00, 0xD1))  # SUB X0, X0, #1
    code.extend((0x1F, 0x00, 0x00, 0xF1))  # CMP X0, #0
    code.extend((0x41, 0xFF, 0xFF, 0x54))  # B.NE loop
    code.extend((0x00, 0x00, 0x40, 0xD4))  # HLT
    program = bytes(code)
    iterations = 50_000

    results = {}

    if bb_cpu:
        bb_cpu.reset()
        bb_cpu.load_program(list(program), 0)
        bb_cpu.set_pc(0)
        bb_cpu.set_register(0, iterations)
        result = bb_cpu.execute(max_batches=100, timeout_seconds=30.0)
        results['BBCache'] = {'cycles': result.total_cycles, 'ips': result.ips}
        print(f"  BBCache   : {result.total_cycles:>10,} cyc | {result.ips:>12,.0f} IPS")

    if ooo_cpu:
        ooo_cpu.reset()
        ooo_cpu.load_program(list(program), 0)
        ooo_cpu.set_pc(0)
        ooo_cpu.set_register(0, iterations)
        result = ooo_cpu.execute(max_batches=100, timeout_seconds=30.0)
        results['OoO'] = {'cycles': result.total_cycles, 'ips': result.ips,
                         'parallel': result.parallel_executions}
        print(f"  OoO       : {result.total_cycles:>10,} cyc | {result.ips:>12,.0f} IPS | P:{result.parallel_executions}")

    if neural_cpu:
        neural_cpu.reset()
        neural_cpu.load_program(list(program), 0)
        neural_cpu.set_pc(0)
        neural_cpu.set_register(0, iterations)
        result = neural_cpu.execute(max_batches=100, timeout_seconds=30.0)
        results['NeuralOoO'] = {'cycles': result.total_cycles, 'ips': result.ips,
                               'parallel': result.parallel_executions}
        print(f"  NeuralOoO : {result.total_cycles:>10,} cyc | {result.ips:>12,.0f} IPS | P:{result.parallel_executions}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neural OoO Scheduler Training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip training, just benchmark")
    parser.add_argument("--load", type=str, help="Load pretrained weights")
    args = parser.parse_args()

    model = None

    if not args.benchmark_only:
        model = train_neural_ooo(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
    elif args.load:
        print(f"Loading model from {args.load}")
        model = NeuralOoOScheduler()
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['model_state_dict'])

    benchmark_neural_ooo(model)


if __name__ == "__main__":
    main()
