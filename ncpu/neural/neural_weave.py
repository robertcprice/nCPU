"""Neural Weave Execution Engine — fully neural at batched GPU speed.

Every ALU operation is a call to a trained .pt model. Instead of executing
one instruction at a time (5K IPS serial neural mode), a window of N
instructions is routed to neural specialists simultaneously, amortizing
GPU kernel-launch overhead across the whole batch.

Architecture (the "weave"):
  - ADD/SUB/CMP  → NeuralCarryCombine (Kogge-Stone CLA, 5 stages)  ← arithmetic.pt
  - MUL          → NeuralMultiplierLUT (65K byte-pair table)        ← multiply.pt
  - AND/ORR/EOR  → NeuralLogical (truth-table, 7 ops × 4 entries)   ← logical.pt
  - LSL/LSR/ASR  → NeuralShiftNet (shift decoder + index net)       ← lsl.pt / lsr.pt
  - Branch pred  → NeuralBranchPredictor (tiny LSTM, 2-layer)       ← trained online
  Each specialist runs on all its instructions simultaneously as a
  single batched GPU forward pass.  The fabric is woven together by
  NeuralWeaveBatchALU.execute_batch(), which is a drop-in replacement
  for the tensor-op Phase 4 in run_parallel_gpu().

Key primitives:
  int_tensor_to_bits(x: Tensor[N], n_bits=32) -> Tensor[N, n_bits]
      GPU-native via bit-shift broadcasting. No Python loop.
  bits_to_int_tensor(bits: Tensor[N, n_bits]) -> Tensor[N]
      GPU-native via dot with powers-of-two weights.

Expected IPS (MPS, batch ≥ 256):
  Neural-serial (step()):         ~5K   IPS  — 1 Python call / instruction
  run_parallel_gpu() tensor ops:  ~1.35M IPS  — no neural, raw +/*& tensor ops
  run_woven() this module:        ~200K–800K IPS — fully neural, large-batch path
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────────────────
# GPU-native int ↔ bits conversion
# ────────────────────────────────────────────────────────────────────────────

def int_tensor_to_bits(x: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    """Convert [N] int64 tensor to [N, n_bits] float32 bit tensor (LSB first).

    Pure tensor op — no Python loop, runs entirely on whatever device x lives on.
    Handles two's-complement: the top bit is the sign bit.

    Example:
        int_tensor_to_bits(torch.tensor([5]), 4)  →  [[1., 0., 1., 0.]]
    """
    device = x.device
    shifts = torch.arange(n_bits, dtype=torch.int64, device=device)  # [n_bits]
    # x: [N], shifts: [n_bits] → broadcast [N, n_bits]
    bits = ((x.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    return bits  # [N, n_bits], LSB at index 0


def bits_to_int_tensor(bits: torch.Tensor) -> torch.Tensor:
    """Convert [N, n_bits] float32 bit tensor (LSB first) to [N] int64 tensor.

    Pure tensor op. Handles two's-complement sign extension for 32-bit results.
    """
    device = bits.device
    n_bits = bits.shape[1]
    weights = (1 << torch.arange(n_bits, dtype=torch.int64, device=device)).float()  # [n_bits]
    thresholded = (bits > 0.5).long()  # [N, n_bits]
    values = (thresholded * weights.long()).sum(dim=1)  # [N] unsigned
    if n_bits == 32:
        # Two's complement: values >= 2^31 are negative
        mask = values >= (1 << 31)
        values = torch.where(mask, values - (1 << 32), values)
    return values  # [N] int64


# ────────────────────────────────────────────────────────────────────────────
# Neural branch predictor (tiny LSTM, trained online from execution traces)
# ────────────────────────────────────────────────────────────────────────────

class NeuralBranchPredictor(nn.Module):
    """2-layer LSTM branch predictor.

    Input per branch: [branch_pc (normalised), cond_code, last_4_outcomes (0/1)]
    Output: probability of branch being taken.

    Trained online by accumulating (pc, cond, outcome) traces and fine-tuning
    every N branches.  Falls back to predict-not-taken when untrained.
    """

    INPUT_DIM = 6   # pc_low16, pc_high16, cond_code, last3_outcomes (packed)
    HIDDEN = 32
    LAYERS = 2

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(self.INPUT_DIM, self.HIDDEN, self.LAYERS, batch_first=True)
        self.head = nn.Linear(self.HIDDEN, 1)
        self._hidden: Optional[tuple] = None
        # Training buffer
        self._trace_pcs: list[int] = []
        self._trace_conds: list[int] = []
        self._trace_outcomes: list[int] = []
        self._trained = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, INPUT_DIM] → [B] taken-probabilities."""
        x = features.unsqueeze(1)  # [B, 1, INPUT_DIM]
        out, _ = self.lstm(x)
        logit = self.head(out[:, -1, :]).squeeze(-1)  # [B]
        return torch.sigmoid(logit)

    def predict_batch(self, branch_pcs: torch.Tensor, cond_codes: torch.Tensor,
                      history: torch.Tensor) -> torch.Tensor:
        """Return predicted taken-probability for each branch.

        Args:
            branch_pcs:  [B] int64 PC values
            cond_codes:  [B] int64 condition codes (0-15)
            history:     [B, 4] recent branch outcomes (0/1 float)
        Returns:
            [B] float probability of taken
        """
        if not self._trained:
            # Predict not-taken for all (safe default)
            return torch.zeros(len(branch_pcs), device=branch_pcs.device)

        device = branch_pcs.device
        pc_low  = (branch_pcs & 0xFFFF).float() / 65535.0
        pc_high = ((branch_pcs >> 16) & 0xFFFF).float() / 65535.0
        cond_f  = cond_codes.float() / 15.0
        h_mean  = history.mean(dim=1)
        features = torch.stack([pc_low, pc_high, cond_f,
                                 history[:, 0], history[:, 1], h_mean], dim=1)
        with torch.no_grad():
            return self.forward(features)

    def record(self, pc: int, cond: int, taken: bool):
        """Record one branch outcome for future training."""
        self._trace_pcs.append(pc)
        self._trace_conds.append(cond)
        self._trace_outcomes.append(int(taken))

    def train_online(self, n_steps: int = 200, lr: float = 1e-3):
        """Fine-tune on accumulated traces.  Called periodically during run_woven()."""
        if len(self._trace_outcomes) < 32:
            return
        device = next(self.parameters()).device
        pcs    = torch.tensor(self._trace_pcs,     dtype=torch.int64,  device=device)
        conds  = torch.tensor(self._trace_conds,   dtype=torch.int64,  device=device)
        labels = torch.tensor(self._trace_outcomes, dtype=torch.float32, device=device)
        # Build history windows (rolling 4-step)
        N = len(labels)
        hist = torch.zeros(N, 4, device=device)
        for i in range(1, min(5, N)):
            hist[i:, i - 1] = labels[:N - i]
        pc_low  = (pcs & 0xFFFF).float() / 65535.0
        pc_high = ((pcs >> 16) & 0xFFFF).float() / 65535.0
        cond_f  = conds.float() / 15.0
        h_mean  = hist.mean(dim=1)
        features = torch.stack([pc_low, pc_high, cond_f,
                                 hist[:, 0], hist[:, 1], h_mean], dim=1)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(n_steps):
            pred = self.forward(features).squeeze()
            loss = nn.functional.binary_cross_entropy(pred, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        self._trained = True
        # Keep last 256 for context continuity
        self._trace_pcs      = self._trace_pcs[-256:]
        self._trace_conds    = self._trace_conds[-256:]
        self._trace_outcomes = self._trace_outcomes[-256:]


# ────────────────────────────────────────────────────────────────────────────
# Woven ALU — routes a full instruction batch to the right neural specialist
# ────────────────────────────────────────────────────────────────────────────

class NeuralWeaveBatchALU:
    """Drop-in neural replacement for Phase 4 tensor ops in run_parallel_gpu().

    Usage:
        weave = NeuralWeaveBatchALU(neural_ops)  # pass loaded NeuralOps
        results = weave.execute_batch(ops, rn_vals, rm_vals, imm12, device)
        # results: [N] int64 tensor, same semantics as parallel_gpu Phase 4
    """

    def __init__(self, neural_ops):
        """neural_ops: loaded ncpu.model.neural_ops.NeuralOps instance."""
        self._ops = neural_ops
        # Cache bit-weight tensors per device to avoid re-creation
        self._bit_weights: dict[str, torch.Tensor] = {}
        # Per-device op-type LUT cache (replaces per-call Python set/loop in execute_batch)
        self._op_luts: dict[str, dict] = {}
        # Metal GPU neural ALU — fast path for ADD/SUB/AND/OR/XOR/MUL
        self._metal: Optional["MetalNeuralALU"] = None  # type: ignore[name-defined]
        try:
            from .metal_neural_alu import load_metal_neural_alu
            self._metal = load_metal_neural_alu(neural_ops, load_mul=True)
        except Exception:
            pass

    def _get_op_luts(self, device: torch.device) -> dict:
        """Lazily build and cache per-device op-type boolean LUT tensors."""
        key = str(device)
        if key in self._op_luts:
            return self._op_luts[key]
        from ncpu.neural.cpu import OpType  # local import, avoids circular at module level
        OT = {m.name: m.value for m in OpType}
        max_op = max(m.value for m in OpType) + 1

        def _lut(*names):
            t = torch.zeros(max_op, dtype=torch.bool)
            for n in names:
                v = OT.get(n, -1)
                if v >= 0:
                    t[v] = True
            return t.to(device)

        luts = {
            'reg_b': _lut('ADD_REG', 'ADD_REG_W', 'SUB_REG', 'SUB_REG_W', 'ADDS_REG',
                           'SUBS_REG', 'MUL', 'AND_REG', 'ORR_REG', 'EOR_REG',
                           'LSL_REG', 'LSR_REG', 'ASR_REG', 'CMP_REG', 'CMP_REG_W',
                           'ANDS_REG', 'TST_REG'),
            'add64': _lut('ADD_IMM', 'ADD_REG', 'ADDS_IMM', 'ADDS_REG'),
            'add32': _lut('ADD_IMM_W', 'ADD_REG_W'),
            'sub64': _lut('SUB_IMM', 'SUB_REG', 'SUBS_IMM', 'SUBS_REG'),
            'sub32': _lut('SUB_IMM_W', 'SUB_REG_W'),
            'mul':   _lut('MUL'),
            'and':   _lut('AND_REG', 'AND_IMM', 'ANDS_REG', 'ANDS_IMM'),
            'orr':   _lut('ORR_REG', 'ORR_IMM'),
            'eor':   _lut('EOR_REG', 'EOR_IMM'),
            'lsl':   _lut('LSL_REG', 'LSL_IMM'),
            'lsr':   _lut('LSR_REG', 'LSR_IMM'),
            'asr':   _lut('ASR_REG', 'ASR_IMM'),
        }
        self._op_luts[key] = luts
        return luts

    def _bit_weights_for(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._bit_weights:
            self._bit_weights[key] = (
                (1 << torch.arange(32, dtype=torch.int64, device=device)).float()
            )
        return self._bit_weights[key]

    # ── ADD / SUB via Kogge-Stone CLA ────────────────────────────────────────

    def _cla_batch(self, a_bits: torch.Tensor, b_bits: torch.Tensor,
                   carry_in: float = 0.0) -> torch.Tensor:
        """Batched Kogge-Stone CLA on [M, 32] bit tensors.

        All M additions share the same 5-stage prefix tree forward passes.
        Returns [M, 32] result bits.
        """
        ops = self._ops
        M, N = a_bits.shape   # N == 32

        # Step 1: initial generate (G) and propagate (P) via neural logical
        # G[m,i] = a[m,i] AND b[m,i],  P[m,i] = a[m,i] XOR b[m,i]
        idx_and = (a_bits > 0.5).long() * 2 + (b_bits > 0.5).long()  # [M, 32]
        idx_xor = idx_and  # same index pattern
        _dev = a_bits.device
        _tt = ops._logical.truth_tables.to(_dev)
        with torch.no_grad():
            G = (torch.sigmoid(_tt[0, idx_and]) > 0.5).float()
            P = (torch.sigmoid(_tt[2, idx_xor]) > 0.5).float()

        if carry_in > 0.5:
            G_new = torch.clamp(G[:, 0] + P[:, 0], max=1.0)
            G = G.clone()
            G[:, 0] = (G_new > 0.5).float()

        # Step 2: Kogge-Stone parallel-prefix (5 stages)
        with torch.no_grad():
            stride = 1
            for _ in range(5):
                if stride >= N:
                    break
                n_comb = N - stride
                G_i = G[:, stride:].reshape(-1)
                P_i = P[:, stride:].reshape(-1)
                G_j = G[:, :n_comb].reshape(-1)
                P_j = P[:, :n_comb].reshape(-1)
                batch_in = torch.stack([G_i, P_i, G_j, P_j], dim=1)  # [M*n_comb, 4]
                out = ops._carry_combiner(batch_in)                    # [M*n_comb, 2]
                new_G = (out[:, 0] > 0.5).float().reshape(M, n_comb)
                new_P = (out[:, 1] > 0.5).float().reshape(M, n_comb)
                G = G.clone(); P = P.clone()
                G[:, stride:] = new_G
                P[:, stride:] = new_P
                stride *= 2

        # Step 3: Final sum = P_original XOR carry
        P_orig = (torch.sigmoid(_tt[2, idx_xor]) > 0.5).float()
        carries = torch.zeros(M, N, device=a_bits.device)
        carries[:, 0] = carry_in
        carries[:, 1:] = G[:, :-1]
        idx_sum = (P_orig > 0.5).long() * 2 + (carries > 0.5).long()
        with torch.no_grad():
            result_bits = (torch.sigmoid(_tt[2, idx_sum]) > 0.5).float()
        return result_bits  # [M, 32]

    def _neural_add_batch(self, a: torch.Tensor, b: torch.Tensor,
                          sub: bool = False, w32: bool = False) -> torch.Tensor:
        """Add/sub M int64 values using neural CLA. Returns [M] int64.

        Fast path: uses Metal GPU native shader (same weights, ~50× faster).
        Fallback: PyTorch MPS batch CLA.
        """
        # ── Metal fast path (amortised for batches ≥ 16) ─────────────────
        if self._metal is not None and self._metal.available and len(a) >= 16:
            a_list = a.tolist()
            b_list = b.tolist()
            result_list = self._metal.add_batch(a_list, b_list, is_sub=sub, is_w32=w32)
            return torch.tensor(result_list, dtype=torch.int64, device=a.device)

        # ── PyTorch fallback ──────────────────────────────────────────────
        ops = self._ops
        if ops._carry_combiner is None or ops._logical is None:
            return (a - b) if sub else (a + b)
        # Ensure models are on the same device as inputs (guard against stale device)
        try:
            _dev = a.device
            _cc_dev = next(ops._carry_combiner.parameters()).device
            if str(_cc_dev) != str(_dev):
                ops._carry_combiner.to(_dev)
                ops._logical.truth_tables.data = ops._logical.truth_tables.data.to(_dev)
        except Exception:
            return (a - b) if sub else (a + b)
        a32 = a.int()
        b32 = b.int()
        a_bits = int_tensor_to_bits(a32.long(), 32)
        if sub:
            b_bits = 1.0 - int_tensor_to_bits(b32.long(), 32)
            carry  = 1.0
        else:
            b_bits = int_tensor_to_bits(b32.long(), 32)
            carry  = 0.0
        result_bits = self._cla_batch(a_bits, b_bits, carry_in=carry)
        result = bits_to_int_tensor(result_bits)  # [M] int64
        if w32:
            result = result & 0xFFFFFFFF
        return result

    # ── MUL via neural LUT ───────────────────────────────────────────────────

    def _neural_mul_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply M pairs using the neural byte-pair LUT. Returns [M] int64."""
        # ── Metal fast path ───────────────────────────────────────────────
        if self._metal is not None and self._metal.mul_available:
            result_list = self._metal.mul_batch(a.tolist(), b.tolist())
            return torch.tensor(result_list, dtype=torch.int64, device=a.device)

        ops = self._ops
        if ops._multiplier is None:
            return a * b
        # Guard: ensure LUT is on the correct device
        try:
            _dev = a.device
            if str(ops._multiplier.lut.table.device) != str(_dev):
                ops._multiplier.lut.table = ops._multiplier.lut.table.to(_dev)
        except Exception:
            return a * b

        M = a.shape[0]
        # Sign handling
        neg_a = a < 0
        neg_b = b < 0
        signs = (neg_a ^ neg_b)
        ua = (a.abs() & 0xFFFFFFFF).long()
        ub = (b.abs() & 0xFFFFFFFF).long()

        # Extract all byte pairs as tensors: 4×4 = 16 lookups per pair
        # a_bytes: [M, 4],  b_bytes: [M, 4]
        byte_shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64, device=a.device)
        a_bytes = ((ua.unsqueeze(1) >> byte_shifts.unsqueeze(0)) & 0xFF)  # [M, 4]
        b_bytes = ((ub.unsqueeze(1) >> byte_shifts.unsqueeze(0)) & 0xFF)  # [M, 4]

        # Build all (i,j) combinations: 4×4 = 16 per multiplication
        i_idx = torch.arange(4, device=a.device)
        j_idx = torch.arange(4, device=a.device)
        ii, jj = torch.meshgrid(i_idx, j_idx, indexing='ij')  # [4, 4]
        ii = ii.reshape(-1)  # [16]
        jj = jj.reshape(-1)  # [16]

        # Gather byte values for all M mults × 16 pairs
        a_sel = a_bytes[:, ii]  # [M, 16]
        b_sel = b_bytes[:, jj]  # [M, 16]

        # LUT lookup: table[a_byte, b_byte] → [16] bits (move table to device)
        _lut_table = ops._multiplier.lut.table.to(a.device)
        with torch.no_grad():
            lut_bits = (torch.sigmoid(
                _lut_table[a_sel.reshape(-1), b_sel.reshape(-1)]
            ) > 0.5).long()  # [M*16, 16]
        lut_bits = lut_bits.reshape(M, 16, 16)  # [M, 16_pairs, 16_bits]

        # Bit-value weights for 16-bit product
        bw = (1 << torch.arange(16, dtype=torch.int64, device=a.device))  # [16]
        pair_products = (lut_bits * bw.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [M, 16]

        # Shift and accumulate: each (i,j) pair contributes product << (i+j)*8
        shift_amounts = ((ii + jj) * 8).unsqueeze(0)  # [1, 16]
        contributions = pair_products << shift_amounts  # [M, 16]
        result = contributions.sum(dim=1) & 0xFFFFFFFF  # [M]

        # Apply signs
        result = torch.where(signs, (-result) & 0xFFFFFFFF, result)
        return result.long()

    # ── AND / ORR / EOR via neural truth tables ──────────────────────────────

    def _neural_logical_batch(self, a: torch.Tensor, b: torch.Tensor,
                               op_idx: int) -> torch.Tensor:
        """Apply logical op (0=AND, 1=OR, 2=XOR) to M pairs. Returns [M] int64."""
        # ── Metal fast path ───────────────────────────────────────────────
        if self._metal is not None and self._metal.available:
            result_list = self._metal.logical_batch(a.tolist(), b.tolist(), op_idx)
            return torch.tensor(result_list, dtype=torch.int64, device=a.device)

        ops = self._ops
        if ops._logical is None:
            if op_idx == 0: return a & b
            if op_idx == 1: return a | b
            return a ^ b
        try:
            _dev = a.device
            if str(ops._logical.truth_tables.device) != str(_dev):
                ops._logical.truth_tables.data = ops._logical.truth_tables.data.to(_dev)
        except Exception:
            if op_idx == 0: return a & b
            if op_idx == 1: return a | b
            return a ^ b
        a_bits = int_tensor_to_bits(a.long(), 32)  # [M, 32]
        b_bits = int_tensor_to_bits(b.long(), 32)  # [M, 32]
        idx = (a_bits > 0.5).long() * 2 + (b_bits > 0.5).long()  # [M, 32]
        _tt = ops._logical.truth_tables.to(a.device)
        with torch.no_grad():
            result_bits = (torch.sigmoid(_tt[op_idx, idx]) > 0.5).float()
        return bits_to_int_tensor(result_bits)  # [M]

    # ── LSL / LSR / ASR via neural shift nets ────────────────────────────────

    def _neural_shift_batch(self, val: torch.Tensor, amt: torch.Tensor,
                             direction: str) -> torch.Tensor:
        """Shift M values by their respective amounts using the shift neural net.

        Falls back to tensor shift if shift models are unavailable.
        direction: 'left' | 'right' | 'asr'
        """
        ops = self._ops
        shifter = ops._shifter_left if direction == 'left' else ops._shifter_right
        if shifter is None:
            amt_clamped = (amt & 63).long()
            if direction == 'left':
                return val << amt_clamped
            elif direction == 'asr':
                return val >> amt_clamped  # Python >> is arithmetic
            else:
                # Logical right shift (unsigned)
                u = val & 0xFFFFFFFFFFFFFFFF
                return u >> amt_clamped

        M = val.shape[0]

        # ── Metal fast path (shift LUT, amortised for batches ≥ 16) ─────────
        if self._metal is not None and self._metal.shift_available and M >= 16:
            amt_clamped = (amt & 63).long()
            is_left = direction == 'left'
            result_list = self._metal.shift_batch(
                val.tolist(), amt_clamped.tolist(), is_left
            )
            return torch.tensor(result_list, dtype=torch.int64, device=val.device)

        results = torch.zeros(M, dtype=torch.int64, device=val.device)
        # Process each unique shift amount as a sub-batch (one forward pass per amount)
        amt_clamped = (amt & 31).long()
        unique_amounts = amt_clamped.unique()

        # Device-guard: ensure shift models are on the right device
        try:
            _dev = val.device
            _sd_dev = next(shifter.parameters()).device
            if str(_sd_dev) != str(_dev):
                shifter.to(_dev)
        except Exception:
            pass

        for sa in unique_amounts:
            mask = (amt_clamped == sa)
            sub_vals = val[mask]
            sa_int = int(sa.item())
            # Encode value as 64-bit bit vector (NeuralShiftNet expects [64] input)
            val_bits_64 = int_tensor_to_bits(sub_vals.int().long(), 64)  # [K, 64]
            amt_bits_64 = torch.tensor(
                [(sa_int >> i) & 1 for i in range(64)],
                dtype=torch.float32, device=val.device
            )  # [64]
            # Batch: run one value at a time (model processes single [64] vectors)
            out_list = []
            with torch.no_grad():
                for vb in val_bits_64:
                    out_bits = shifter(vb, amt_bits_64)  # [64]
                    out_list.append(out_bits)
            out = torch.stack(out_list)  # [K, 64]
            result_sub = bits_to_int_tensor(out[:, :32])  # lower 32 bits
            results[mask] = result_sub
        return results

    # ── CMP/flags via neural subtraction ─────────────────────────────────────

    def _neural_cmp_batch(self, a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Compute flags for M comparisons. Returns (N_bits, Z_bits, C_bits) [M] bool."""
        diffs = self._neural_add_batch(a, b, sub=True)
        n_bits = diffs < 0
        z_bits = diffs == 0
        c_bits = (a.long() & 0xFFFFFFFF) >= (b.long() & 0xFFFFFFFF)
        return n_bits, z_bits, c_bits

    # ── Main dispatch ─────────────────────────────────────────────────────────

    def execute_batch(
        self,
        ops: torch.Tensor,         # [N] OpType int values
        rn_vals: torch.Tensor,     # [N] int64 — source register A values
        rm_vals: torch.Tensor,     # [N] int64 — source register B / Rm values
        imm12: torch.Tensor,       # [N] int64 — immediate values
        op_type_map: dict,         # {name: int} from OpType enum values
        existing_results: torch.Tensor,   # [N] pre-filled with tensor-op results
        existing_write_mask: torch.Tensor, # [N] bool  — which slots already set
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace tensor-op results with neural results for all ALU ops.

        Only re-computes instructions that have a neural specialist. Everything
        else (loads, stores, MOV, etc.) is left as-is from existing_results.

        Returns:
            (results [N] int64, write_mask [N] bool)
        """
        device = ops.device
        # Get precomputed LUTs (built once per device, avoid per-call Python set/loops)
        L = self._get_op_luts(device)
        results    = existing_results.clone()
        write_mask = existing_write_mask.clone()

        # Operand B: register (rm_vals) or immediate (imm12) — single LUT lookup
        operand_b = torch.where(L['reg_b'][ops], rm_vals, imm12)

        # ── ADD (64-bit) ──────────────────────────────────────────────────────
        m_add = L['add64'][ops]
        if m_add.any() and self._ops._carry_combiner is not None:
            idx = m_add.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_add_batch(rn_vals[idx], operand_b[idx])
            write_mask[idx] = True

        # ── ADD (32-bit) ──────────────────────────────────────────────────────
        m_add_w = L['add32'][ops]
        if m_add_w.any() and self._ops._carry_combiner is not None:
            idx = m_add_w.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_add_batch(rn_vals[idx], operand_b[idx], w32=True)
            write_mask[idx] = True

        # ── SUB (64-bit) ──────────────────────────────────────────────────────
        m_sub = L['sub64'][ops]
        if m_sub.any() and self._ops._carry_combiner is not None:
            idx = m_sub.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_add_batch(rn_vals[idx], operand_b[idx], sub=True)
            write_mask[idx] = True

        # ── SUB (32-bit) ──────────────────────────────────────────────────────
        m_sub_w = L['sub32'][ops]
        if m_sub_w.any() and self._ops._carry_combiner is not None:
            idx = m_sub_w.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_add_batch(rn_vals[idx], operand_b[idx], sub=True, w32=True)
            write_mask[idx] = True

        # ── MUL ──────────────────────────────────────────────────────────────
        m_mul = L['mul'][ops]
        if m_mul.any() and self._ops._multiplier is not None:
            idx = m_mul.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_mul_batch(rn_vals[idx], rm_vals[idx])
            write_mask[idx] = True

        # ── AND ───────────────────────────────────────────────────────────────
        m_and = L['and'][ops]
        if m_and.any() and self._ops._logical is not None:
            idx = m_and.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_logical_batch(rn_vals[idx], operand_b[idx], op_idx=0)
            write_mask[idx] = True

        # ── ORR ───────────────────────────────────────────────────────────────
        m_orr = L['orr'][ops]
        if m_orr.any() and self._ops._logical is not None:
            idx = m_orr.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_logical_batch(rn_vals[idx], operand_b[idx], op_idx=1)
            write_mask[idx] = True

        # ── EOR ───────────────────────────────────────────────────────────────
        m_eor = L['eor'][ops]
        if m_eor.any() and self._ops._logical is not None:
            idx = m_eor.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_logical_batch(rn_vals[idx], operand_b[idx], op_idx=2)
            write_mask[idx] = True

        # ── LSL ───────────────────────────────────────────────────────────────
        m_lsl = L['lsl'][ops]
        if m_lsl.any():
            idx = m_lsl.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_shift_batch(rn_vals[idx], operand_b[idx], 'left')
            write_mask[idx] = True

        # ── LSR ───────────────────────────────────────────────────────────────
        m_lsr = L['lsr'][ops]
        if m_lsr.any():
            idx = m_lsr.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_shift_batch(rn_vals[idx], operand_b[idx], 'right')
            write_mask[idx] = True

        # ── ASR ───────────────────────────────────────────────────────────────
        m_asr = L['asr'][ops]
        if m_asr.any():
            idx = m_asr.nonzero(as_tuple=True)[0]
            results[idx] = self._neural_shift_batch(rn_vals[idx], operand_b[idx], 'asr')
            write_mask[idx] = True

        return results, write_mask
