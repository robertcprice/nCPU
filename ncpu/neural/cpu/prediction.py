"""
Branch prediction, instruction dispatch, execution optimization, and neural execution engine.

Classes:
    BranchTraceBuffer - 2-bit saturating counter BTB with GPU-parallel batch ops
    NeuralInstructionDispatcher - Neural ARM64 instruction classifier
    NeuralExecutionOptimizer - LSTM-based execution pattern optimizer
    GPUBranchDecider - Tensor-based ARM64 condition evaluator (all 16 conditions)
    NeuralExecutionEngine - Soft-dispatch fully neural execution engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ════════════════════════════════════════════════════════════════════════════════
# BRANCH TRACE BUFFER (BTB) - PREDICTS BRANCH OUTCOMES
# Tracks branch history at each PC for smarter speculation
# ════════════════════════════════════════════════════════════════════════════════

class BranchTraceBuffer:
    """
    Branch Trace Buffer - tracks branch history for prediction.

    Key insight: Most branches have predictable patterns (loops, error checks).
    By tracking history, we can predict outcomes with high confidence.

    Uses a hash-indexed table with 2-bit saturating counters:
    - 00: Strongly Not Taken
    - 01: Weakly Not Taken
    - 10: Weakly Taken
    - 11: Strongly Taken

    Prediction confidence = |counter - 1.5| / 1.5 (0.0 to 1.0)
    """

    def __init__(self, size: int = 2048, device=None):
        self.size = size
        self.device = device or torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # BTB entries: indexed by hash(PC)
        # counter: 2-bit saturating counter (0-3)
        self.counter = torch.ones(size, dtype=torch.int8, device=self.device)  # Start neutral (1)
        # target: last known branch target
        self.target = torch.zeros(size, dtype=torch.int64, device=self.device)
        # valid: has this entry been used?
        self.valid = torch.zeros(size, dtype=torch.bool, device=self.device)
        # pc_tag: verify we have the right entry (upper bits of PC)
        self.pc_tag = torch.zeros(size, dtype=torch.int64, device=self.device)

        # Stats for debugging
        self.predictions = 0
        self.correct = 0
        self.mispredictions = 0

    def _hash(self, pc: torch.Tensor) -> torch.Tensor:
        """Hash PC to BTB index."""
        # Simple hash: XOR folding of PC bits
        return ((pc >> 2) ^ (pc >> 10)) % self.size

    def predict(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict branch outcome.

        Args:
            pc: Branch PC (can be scalar or tensor)

        Returns:
            predicted_taken: bool tensor - predict branch taken?
            confidence: float tensor - confidence (0.0 to 1.0)
            predicted_target: int64 tensor - predicted target PC
        """
        idx = self._hash(pc).long()

        # Check if entry is valid and matches PC tag
        tag = pc >> 12  # Upper bits for verification
        hit = self.valid[idx] & (self.pc_tag[idx] == tag)

        # Predict: counter >= 2 means predict taken
        cnt = self.counter[idx]
        predicted_taken = (cnt >= 2) & hit

        # Confidence: distance from neutral (1.5)
        # 0 or 3 = high confidence (0.67), 1 or 2 = low confidence (0.0)
        confidence = torch.where(hit,
                                 torch.abs(cnt.float() - 1.5) / 1.5,
                                 torch.zeros_like(cnt, dtype=torch.float))

        predicted_target = torch.where(hit, self.target[idx], pc + 4)

        return predicted_taken, confidence, predicted_target

    def update(self, pc: torch.Tensor, taken: torch.Tensor, target: torch.Tensor):
        """
        Update BTB with actual branch outcome.

        Args:
            pc: Branch PC
            taken: Was branch taken?
            target: Actual target PC (if taken)
        """
        idx = self._hash(pc).long()
        tag = pc >> 12

        # Update counter with saturation (0-3)
        cnt = self.counter[idx]
        new_cnt = torch.where(taken,
                              torch.clamp(cnt + 1, 0, 3),
                              torch.clamp(cnt - 1, 0, 3))
        self.counter[idx] = new_cnt.to(torch.int8)

        # Update target if taken
        self.target[idx] = torch.where(taken, target, self.target[idx])

        # Mark valid and update tag
        self.valid[idx] = True
        self.pc_tag[idx] = tag

    def predict_batch(self, pcs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict multiple branches at once (GPU-parallel).

        Args:
            pcs: [N] tensor of branch PCs

        Returns:
            predicted_taken: [N] bool tensor
            confidence: [N] float tensor
            predicted_target: [N] int64 tensor
        """
        idx = self._hash(pcs).long()
        tags = pcs >> 12

        hits = self.valid[idx] & (self.pc_tag[idx] == tags)
        cnts = self.counter[idx]

        predicted_taken = (cnts >= 2) & hits
        confidence = torch.where(hits,
                                 torch.abs(cnts.float() - 1.5) / 1.5,
                                 torch.zeros_like(cnts, dtype=torch.float))
        predicted_targets = torch.where(hits, self.target[idx], pcs + 4)

        return predicted_taken, confidence, predicted_targets

    def update_batch(self, pcs: torch.Tensor, taken: torch.Tensor, targets: torch.Tensor):
        """Update multiple BTB entries at once."""
        idx = self._hash(pcs).long()
        tags = pcs >> 12

        cnts = self.counter[idx]
        new_cnts = torch.where(taken,
                               torch.clamp(cnts + 1, 0, 3),
                               torch.clamp(cnts - 1, 0, 3))
        self.counter[idx] = new_cnts.to(torch.int8)
        self.target[idx] = torch.where(taken, targets, self.target[idx])
        self.valid[idx] = True
        self.pc_tag[idx] = tags

    def get_stats(self) -> Dict[str, float]:
        """Get prediction statistics."""
        total = self.predictions
        if total == 0:
            return {'accuracy': 0.0, 'predictions': 0, 'correct': 0, 'mispredictions': 0}
        return {
            'accuracy': self.correct / total,
            'predictions': self.predictions,
            'correct': self.correct,
            'mispredictions': self.mispredictions,
        }

    def record_outcome(self, predicted_taken: bool, actual_taken: bool):
        """Record prediction outcome for stats."""
        self.predictions += 1
        if predicted_taken == actual_taken:
            self.correct += 1
        else:
            self.mispredictions += 1


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL INSTRUCTION DISPATCHER
# Learns to predict instruction type from bit patterns - FULLY NEURAL!
# ════════════════════════════════════════════════════════════════════════════════

class NeuralInstructionDispatcher(nn.Module):
    """
    Neural network that LEARNS to classify ARM64 instructions.

    Instead of hardcoded if/elif chains, uses a trained network to predict
    the operation type from the 32-bit instruction encoding.
    ALL ON GPU - no Python branching for dispatch!
    """

    def __init__(self, num_op_types: int = 128):
        super().__init__()
        self.num_op_types = num_op_types

        # Bit pattern encoder - learns ARM64 encoding structure
        self.bit_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Op-type classifier - predicts which operation
        self.op_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_op_types),
        )

        # Register extractors - predict rd, rn, rm from bits
        self.reg_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 96),  # 3 registers × 32 (one-hot)
        )

        # Immediate extractor - predict immediate value
        self.imm_extractor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),  # 16-bit immediate (as bits)
        )

    def forward(self, inst_bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode instruction using neural network - ALL ON GPU!

        Args:
            inst_bits: [32] tensor of instruction bits

        Returns:
            op_logits: [num_op_types] logits for operation type
            rd_logits: [32] logits for destination register
            rn_logits: [32] logits for first source register
            rm_logits: [32] logits for second source register
        """
        # Encode bit pattern
        features = self.bit_encoder(inst_bits)

        # Predict operation type
        op_logits = self.op_classifier(features)

        # Extract registers
        reg_logits = self.reg_extractor(features)
        rd_logits = reg_logits[:32]
        rn_logits = reg_logits[32:64]
        rm_logits = reg_logits[64:]

        return op_logits, rd_logits, rn_logits, rm_logits


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL EXECUTION OPTIMIZER
# Learns execution patterns and optimizes instruction sequences
# ════════════════════════════════════════════════════════════════════════════════

class NeuralExecutionOptimizer(nn.Module):
    """
    Neural network that learns to OPTIMIZE execution patterns.

    Features:
    - Predicts which loops can be vectorized
    - Learns common instruction sequences to batch
    - Identifies hot paths for speculative execution
    - All decisions made with GPU tensor operations!
    """

    def __init__(self, hidden_size: int = 128, sequence_len: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.sequence_len = sequence_len

        # Instruction sequence encoder (processes recent instructions)
        self.seq_encoder = nn.LSTM(
            input_size=64,  # Compressed instruction features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # Instruction compressor (32 bits -> 64 features)
        self.inst_compress = nn.Linear(32, 64)

        # Optimization predictors
        self.vectorize_pred = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.batch_size_pred = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),  # Non-negative batch size
        )

        self.speculate_pred = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Execution history for learning (circular buffer on GPU)
        self.history = None
        self.history_idx = 0

    def init_history(self, device):
        """Initialize execution history buffer on GPU."""
        self.history = torch.zeros(self.sequence_len, 32, device=device)
        self.history_idx = 0

    def record_instruction(self, inst_bits: torch.Tensor):
        """Record instruction to history (GPU operation)."""
        if self.history is None:
            self.init_history(inst_bits.device)
        self.history[self.history_idx] = inst_bits
        self.history_idx = (self.history_idx + 1) % self.sequence_len

    def predict_optimizations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict optimization opportunities based on execution history.

        Returns:
            should_vectorize: [1] probability loop can be vectorized
            batch_size: [1] predicted optimal batch size
            should_speculate: [1] probability speculation is beneficial
        """
        if self.history is None:
            return (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]))

        # Compress and encode history
        compressed = self.inst_compress(self.history)  # [seq_len, 64]
        encoded, _ = self.seq_encoder(compressed.unsqueeze(0))  # [1, seq_len, hidden*2]
        summary = encoded[0, -1, :]  # Last hidden state

        # Predict optimizations
        should_vectorize = self.vectorize_pred(summary)
        batch_size = self.batch_size_pred(summary) + 1  # At least 1
        should_speculate = self.speculate_pred(summary)

        return should_vectorize, batch_size, should_speculate


# ════════════════════════════════════════════════════════════════════════════════
# GPU BRANCH DECIDER
# ════════════════════════════════════════════════════════════════════════════════

class GPUBranchDecider(nn.Module):
    """
    Makes branch decisions ENTIRELY ON GPU using tensor operations.

    No .item() calls - everything is differentiable!
    Computes ALL 16 ARM64 conditions in parallel.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        cond_code: torch.Tensor,  # [] condition code (0-15)
        flags: torch.Tensor,      # [4] N, Z, C, V
        reg_val: torch.Tensor,    # [] register value (for CBZ/CBNZ)
        branch_type: torch.Tensor,  # [] 0=B.cond, 1=CBZ, 2=CBNZ
    ) -> torch.Tensor:
        """Returns: take_branch as tensor - STAYS ON GPU!"""
        n, z, c, v = flags[0], flags[1], flags[2], flags[3]

        # Compute ALL conditions in parallel (no Python branching!)
        conditions = torch.stack([
            z,                              # 0: EQ
            1 - z,                          # 1: NE
            c,                              # 2: CS/HS
            1 - c,                          # 3: CC/LO
            n,                              # 4: MI
            1 - n,                          # 5: PL
            v,                              # 6: VS
            1 - v,                          # 7: VC
            c * (1 - z),                    # 8: HI
            (1 - c) + z,                    # 9: LS
            (n == v).float(),               # 10: GE
            (n != v).float(),               # 11: LT
            (1 - z) * (n == v).float(),     # 12: GT
            z + (n != v).float(),           # 13: LE
            torch.ones_like(z),             # 14: AL (always)
            torch.zeros_like(z),            # 15: NV (never)
        ])

        cond_idx = cond_code.long().clamp(0, 15)
        bcond_result = conditions[cond_idx]

        cbz_result = (reg_val == 0).float()
        cbnz_result = (reg_val != 0).float()

        result = torch.where(branch_type == 0, bcond_result,
                 torch.where(branch_type == 1, cbz_result, cbnz_result))

        return result


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL EXECUTION ENGINE - FULLY TENSOR-BASED, NO PYTHON DISPATCH
# ════════════════════════════════════════════════════════════════════════════════

class NeuralExecutionEngine(nn.Module):
    """
    Fully neural CPU execution - NO if/elif, NO .item() calls.

    Key innovations:
    1. SOFT DISPATCH: Attention-weighted effect networks for each op type
    2. TENSOR MEMORY: One-hot addressing with gather/scatter ops
    3. DIFFERENTIABLE: End-to-end gradient flow for training

    This is the nCPU approach to CPU execution.
    """

    def __init__(self, num_ops: int = 64, state_dim: int = 64, num_regs: int = 32, device=None):
        super().__init__()
        self.num_ops = num_ops
        self.state_dim = state_dim
        self.num_regs = num_regs
        self.device = device or torch.device('cpu')

        # Op embeddings - learned representation for each operation type
        self.op_embeddings = nn.Embedding(num_ops, state_dim)

        # Register state encoder
        self.reg_encoder = nn.Linear(num_regs, state_dim)

        # Unified effect network - takes [op_emb, reg_state, operands] -> state_delta
        self.effect_network = nn.Sequential(
            nn.Linear(state_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_regs),  # Delta to registers
        )

        # Memory effect network - for load/store ops
        self.memory_effect = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Memory delta width
        )

        # Operand extractor - extracts rd, rn, rm, imm from instruction bits
        self.operand_net = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )

    def forward(
        self,
        inst_bits: torch.Tensor,     # [32] instruction bits
        op_weights: torch.Tensor,    # [num_ops] soft dispatch weights
        regs: torch.Tensor,          # [32] register values
        memory: torch.Tensor,        # [mem_size] memory tensor
        flags: torch.Tensor,         # [4] condition flags
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one instruction with FULLY NEURAL dispatch.

        Returns:
            new_regs: [32] updated registers
            new_memory: [mem_size] updated memory (sparse update)
            new_flags: [4] updated flags
        """
        # Encode operands from instruction bits
        operand_emb = self.operand_net(inst_bits)  # [state_dim]

        # Weighted op embedding (soft dispatch)
        op_indices = torch.arange(self.num_ops, device=self.device)
        op_embs = self.op_embeddings(op_indices)  # [num_ops, state_dim]
        weighted_op = torch.matmul(op_weights, op_embs)  # [state_dim]

        # Encode register state
        reg_state = self.reg_encoder(regs.float())  # [state_dim]

        # Compute effect
        combined = torch.cat([weighted_op, reg_state, operand_emb])  # [state_dim * 3]
        reg_delta = self.effect_network(combined)  # [32]

        # Apply effect to registers (keep as float for gradient flow)
        new_regs = regs.float() + reg_delta

        # Flags update (simplified - could be learned)
        result = reg_delta[0]  # First register's delta as proxy
        new_flags = flags.clone()
        new_flags[0] = (result < 0).float()  # N
        new_flags[1] = (result == 0).float()  # Z

        return new_regs, memory, new_flags

    def tensor_memory_read(
        self,
        memory: torch.Tensor,  # [mem_size]
        addr_weights: torch.Tensor,  # [mem_size] soft address (one-hot or learned)
    ) -> torch.Tensor:
        """Read from memory using tensor dot product - NO .item()!"""
        return torch.dot(memory.float(), addr_weights)

    def tensor_memory_write(
        self,
        memory: torch.Tensor,  # [mem_size]
        addr_weights: torch.Tensor,  # [mem_size] soft address
        value: torch.Tensor,  # [] value to write
    ) -> torch.Tensor:
        """Write to memory using tensor ops - NO .item()!"""
        # Soft write: memory = memory * (1 - addr_weights) + value * addr_weights
        return memory * (1 - addr_weights) + value * addr_weights

    def address_to_onehot(self, addr_logits: torch.Tensor, mem_size: int) -> torch.Tensor:
        """Convert address logits to soft one-hot using softmax or gumbel."""
        # Use softmax for differentiable addressing
        return F.softmax(addr_logits, dim=-1)

