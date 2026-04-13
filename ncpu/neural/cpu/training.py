"""
Training mixin for NeuralCPU.

Self-supervised training of neural dispatcher and execution engine
using lookup tables as ground truth.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .constants import OpType

logger = logging.getLogger(__name__)


class TrainingMixin:
    """Training methods for NeuralCPU neural components."""

    # ════════════════════════════════════════════════════════════════════════════════
    # NEURAL DISPATCHER TRAINING - Uses lookup tables as ground truth
    # ════════════════════════════════════════════════════════════════════════════════

    def train_neural_dispatcher(self, num_samples: int = 100000, epochs: int = 10, batch_size: int = 256):
        """
        Train the neural dispatcher using lookup tables as ground truth.

        This is SELF-SUPERVISED: we generate random instructions and use
        our lookup tables to label them, then train the neural network!
        """
        import torch.optim as optim

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING NEURAL INSTRUCTION DISPATCHER")
        logger.info("=" * 70)
        logger.info(f"  Samples: {num_samples:,}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info("")

        # Generate training data from lookup tables
        logger.info("  Generating training data from lookup tables...")
        X_train = []  # Instruction bits [N, 32]
        y_train = []  # Op types [N]

        # Sample instructions that we KNOW how to decode
        known_op_bytes = []
        for i in range(256):
            if self.op_type_table[i].item() != 0:
                known_op_bytes.append(i)

        known_op_codes = []
        for i in range(512):
            if self.op_code_table[i].item() != 0:
                known_op_codes.append(i)

        # Generate samples
        for _ in range(num_samples):
            # 80% from op_byte table, 20% from op_code table
            if torch.rand(1).item() < 0.8 and known_op_bytes:
                op_byte = known_op_bytes[int(torch.randint(len(known_op_bytes), (1,)).item())]
                # Random lower bits
                lower_bits = int(torch.randint(0x1000000, (1,)).item())
                inst = (op_byte << 24) | lower_bits
                op_type_val = self.op_type_table[op_byte].item()
            elif known_op_codes:
                op_code = known_op_codes[int(torch.randint(len(known_op_codes), (1,)).item())]
                # Random remaining bits
                lower_bits = int(torch.randint(0x800000, (1,)).item())
                inst = (op_code << 23) | lower_bits
                op_type_val = self.op_code_table[op_code].item()
            else:
                continue

            # Convert to bits
            bits = [float((inst >> j) & 1) for j in range(32)]
            X_train.append(bits)
            y_train.append(op_type_val)

        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.long, device=self.device)
        logger.info(f"  Generated {len(X_train):,} training samples")

        # Train the neural dispatcher
        self.neural_dispatcher.train()
        optimizer = optim.Adam(self.neural_dispatcher.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        num_batches = len(X_train) // batch_size

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(len(X_train), device=self.device)
            X_train = X_train[perm]
            y_train = y_train[perm]

            total_loss = 0.0
            correct = 0
            total = 0

            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                optimizer.zero_grad()

                # Forward pass through neural dispatcher
                # Note: need to process each sample (batched version)
                op_logits_list = []
                for j in range(len(X_batch)):
                    op_logits, _, _, _ = self.neural_dispatcher(X_batch[j])
                    op_logits_list.append(op_logits)

                op_logits = torch.stack(op_logits_list)
                loss = criterion(op_logits, y_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = op_logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            acc = 100.0 * correct / total
            avg_loss = total_loss / num_batches
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.1f}%%")

        self.neural_dispatcher.eval()
        self.dispatcher_trained = True
        logger.info(f"\n  Neural dispatcher trained! Accuracy: {acc:.1f}%%")
        logger.info("=" * 70)


    def neural_step(self) -> torch.Tensor:
        """
        Execute one instruction using FULLY NEURAL execution engine.

        NO Python if/elif dispatch! Soft attention over ops.
        Tensor-based memory addressing.

        Returns:
            continue_tensor: scalar tensor, >0.5 means continue
        """
        # Tensor-based halt check
        if self.halted:
            return torch.tensor(0.0, device=self.device)

        # PC as tensor - fetch instruction via tensor indexing
        pc_clamped = self.pc.clamp(0, self.mem_size - 4)

        # Fetch 4 bytes using tensor indexing (no .item())
        byte0 = self.memory[pc_clamped.long()]
        byte1 = self.memory[(pc_clamped + 1).long()]
        byte2 = self.memory[(pc_clamped + 2).long()]
        byte3 = self.memory[(pc_clamped + 3).long()]

        # Combine into instruction (little-endian) - all tensor ops
        inst_tensor = (byte0.long() |
                      (byte1.long() << 8) |
                      (byte2.long() << 16) |
                      (byte3.long() << 24))

        # Check for halt instruction (all zeros)
        is_halt = (inst_tensor == 0)
        if is_halt:
            self.halted = True
            return torch.tensor(0.0, device=self.device)

        # Convert instruction to bit tensor using tensor ops (no .item()!)
        bit_indices = torch.arange(32, device=self.device)
        inst_bits = ((inst_tensor >> bit_indices) & 1).float()

        # Neural dispatch - get op type weights
        op_logits, rd_logits, rn_logits, rm_logits = self.neural_dispatcher(inst_bits)

        # Soft dispatch weights (attention over ops)
        op_weights = F.softmax(op_logits, dim=-1)

        # Execute through neural engine
        new_regs, new_memory, new_flags = self.neural_engine(
            inst_bits, op_weights, self.regs, self.memory, self.flags
        )

        # Update state
        self.regs = new_regs
        self.flags = new_flags

        # Advance PC (tensor operation)
        self.pc = self.pc + 4
        self.inst_count = self.inst_count + 1

        return torch.tensor(1.0, device=self.device)

    def neural_run(self, max_instructions: int = 1000) -> Tuple[int, float]:
        """
        Run using fully neural execution engine.

        Returns:
            (instructions_executed, elapsed_time)
        """
        import time
        start = time.perf_counter()

        executed = 0
        for _ in range(max_instructions):
            continue_flag = self.neural_step()
            executed += 1
            if continue_flag < 0.5:  # Tensor comparison
                break

        elapsed = time.perf_counter() - start
        return executed, elapsed

    def train_neural_engine(self, num_samples: int = 10000, epochs: int = 20, batch_size: int = 64):
        """
        Train the neural execution engine using ground truth from step().
        """
        import torch.optim as optim

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING NEURAL EXECUTION ENGINE")
        logger.info("=" * 70)

        # Instruction templates
        templates = [
            lambda: 0x91000000 | (torch.randint(0,31,(1,)).item()) | (torch.randint(0,31,(1,)).item()<<5) | (torch.randint(0,256,(1,)).item()<<10),
            lambda: 0xD1000000 | (torch.randint(0,31,(1,)).item()) | (torch.randint(0,31,(1,)).item()<<5) | (torch.randint(0,256,(1,)).item()<<10),
            lambda: 0x8B000000 | (torch.randint(0,31,(1,)).item()) | (torch.randint(0,31,(1,)).item()<<5) | (torch.randint(0,31,(1,)).item()<<16),
            lambda: 0xCB000000 | (torch.randint(0,31,(1,)).item()) | (torch.randint(0,31,(1,)).item()<<5) | (torch.randint(0,31,(1,)).item()<<16),
            lambda: 0xD2800000 | (torch.randint(0,31,(1,)).item()) | (torch.randint(0,1000,(1,)).item()<<5),
        ]

        X_bits, X_regs, Y_regs = [], [], []
        logger.info(f"  Generating {num_samples:,} samples...")

        for i in range(num_samples):
            inst = templates[i % len(templates)]()
            init_regs = torch.randint(-1000, 1000, (32,), dtype=torch.int64, device=self.device)

            old_regs, old_pc, old_halted = self.regs.clone(), self.pc.clone(), self.halted
            self.regs, self.pc, self.halted = init_regs.clone(), torch.tensor(0x1000, dtype=torch.int64, device=self.device), False

            for j in range(4):
                self.memory[0x1000 + j] = (inst >> (j * 8)) & 0xFF

            self.step()

            bit_indices = torch.arange(32, device=self.device)
            X_bits.append(((torch.tensor(inst, device=self.device) >> bit_indices) & 1).float())
            X_regs.append(init_regs.float())
            Y_regs.append(self.regs.float())

            self.regs, self.pc, self.halted = old_regs, old_pc, old_halted

        X_bits, X_regs, Y_regs = torch.stack(X_bits), torch.stack(X_regs), torch.stack(Y_regs)

        self.neural_engine.train()
        optimizer = optim.Adam(self.neural_engine.parameters(), lr=0.001)

        logger.info(f"  Training for {epochs} epochs...")
        for epoch in range(epochs):
            perm = torch.randperm(num_samples, device=self.device)
            total_loss, num_batches = 0, 0

            for b in range(0, num_samples, batch_size):
                batch_bits = X_bits[perm[b:b+batch_size]]
                batch_in = X_regs[perm[b:b+batch_size]]
                batch_out = Y_regs[perm[b:b+batch_size]]

                optimizer.zero_grad()
                losses = []

                for i in range(len(batch_bits)):
                    with torch.no_grad():
                        op_logits, _, _, _ = self.neural_dispatcher(batch_bits[i])
                    op_weights = F.softmax(op_logits, dim=-1).detach()
                    pred_regs, _, _ = self.neural_engine(batch_bits[i], op_weights, batch_in[i].long(), self.memory, self.flags)
                    losses.append(F.mse_loss(pred_regs.float(), batch_out[i]))

                batch_loss = torch.stack(losses).mean()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item() / len(batch_bits)
                num_batches += 1

            if (epoch + 1) % 5 == 0:
                logger.info(f"    Epoch {epoch+1}: loss = {total_loss/num_batches:.4f}")

        self.neural_engine.eval()
        logger.info("  Neural engine trained!")
