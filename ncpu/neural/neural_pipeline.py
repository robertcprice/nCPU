"""Full Neural Pipeline — every stage of the CPU is a neural network.

Pipeline stages (all neural):
  1. NeuralPrefetcher      — LSTM predicts upcoming memory addresses
                             activates the dormant memory_oracle + prefetch.pt
  2. NeuralARM64Decoder    — Transformer decodes raw instruction bits
                             wraps arm64_decoder.pt (bit-embed → field-attn → heads)
  3. NeuralSpeculator      — checkpoint/rollback for branch speculation
                             pairs with NeuralBranchPredictor from neural_weave.py
  4. NeuralWeaveBatchALU   — ALU execution (arithmetic/logical/shift models)
                             already implemented in neural_weave.py
  5. NeuralCacheManager    — cache_replace.pt LSTM decides superblock eviction
  6. NeuralSyscallRouter   — MLP routes syscall numbers to handlers
  7. NeuralRegisterFile    — register_file.pt XZR/SP disambiguation
  8. NeuralMemoryArithmetic — pointer.pt/stack.pt neural address computation

All components degrade gracefully to their classical fallbacks when models
are unavailable or confidence is below threshold.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# ─────────────────────────────────────────────────────────────────────────────
# 1. NEURAL PREFETCHER — activates the dormant prefetch.pt LSTM
# ─────────────────────────────────────────────────────────────────────────────

class _PrefetchNet(nn.Module):
    """Reconstructed architecture matching prefetch.pt state dict.

    addr_embed: Embedding(65536, 32) — 65536 address slots, 32-dim embedding
    lstm:       LSTM(32, 64, 1, batch_first=True)
    predictor:  Linear(64, 4) — predict 4 next addresses
    """

    def __init__(self):
        super().__init__()
        self.addr_embed = nn.Embedding(65536, 32)
        self.lstm       = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.predictor  = nn.Linear(64, 4)

    def forward(self, addr_seq: torch.Tensor) -> torch.Tensor:
        """
        addr_seq: [1, T] int64 — sequence of recent address indices (addr >> 2 & 65535)
        Returns:  [4] int64 — predicted next address indices
        """
        emb = self.addr_embed(addr_seq)          # [1, T, 32]
        out, _ = self.lstm(emb)                  # [1, T, 64]
        last = out[:, -1, :]                     # [1, 64]
        pred = self.predictor(last).squeeze(0)   # [4]
        return pred.long().clamp(0, 65535)


class NeuralPrefetcher:
    """Wires the trained prefetch.pt LSTM into the execution hot path.

    Every `interval` memory accesses, runs the LSTM over the recent address
    history to predict the next 4 addresses and pre-touch them in memory.
    Falls back to sequential prefetch when model unavailable.
    """

    N_HISTORY = 16   # length of address history fed to LSTM

    def __init__(self, oracle, memory: torch.Tensor, interval: int = 32,
                 models_dir: Path = MODELS_DIR):
        self._oracle   = oracle
        self._mem      = memory
        self._interval = interval
        self._counter  = 0
        self._history: list[int] = []   # circular address history
        self._model:  Optional[_PrefetchNet] = None
        self._loaded  = False
        self._load(models_dir)

    def _load(self, models_dir: Path):
        path = models_dir / "os" / "prefetch.pt"
        if not path.exists():
            return
        try:
            net   = _PrefetchNet()
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.eval()
            self._model  = net
            self._loaded = True
        except Exception:
            self._loaded = False

    def on_memory_access(self, addr: int, is_write: bool):
        """Called after every load/store.  Fires prefetch every `interval` calls."""
        slot = (addr >> 2) & 0xFFFF        # address → embedding slot
        self._history.append(slot)
        if len(self._history) > self.N_HISTORY:
            self._history = self._history[-self.N_HISTORY:]
        self._counter += 1
        if self._counter % self._interval == 0:
            self._fire_prefetch()

    def _fire_prefetch(self):
        """Run LSTM → touch predicted addresses."""
        if self._loaded and self._model is not None and len(self._history) >= 4:
            try:
                hist_t = torch.tensor([self._history], dtype=torch.int64)
                with torch.no_grad():
                    pred_slots = self._model(hist_t)   # [4]
                mem_size = self._mem.shape[0]
                for slot in pred_slots.tolist():
                    base = (int(slot) << 2) & ~0x3F   # 64-byte cache line
                    end  = min(base + 64, mem_size)
                    if 0 <= base < end:
                        _ = self._mem[base:end].sum()
                return
            except Exception:
                pass
        # Fallback: try oracle
        try:
            candidates = self._oracle.get_prefetch_candidates()
            if candidates is not None:
                mem_size = self._mem.shape[0]
                for addr in candidates[:4]:
                    base = int(addr) & ~0x3F
                    end  = min(base + 64, mem_size)
                    if 0 <= base < end:
                        _ = self._mem[base:end].sum()
        except Exception:
            pass

    def warm_instruction_stream(self, pc: int, n_insts: int, mem_size: int):
        """Pre-touch the next n_insts instructions (sequential prefetch)."""
        end = min(pc + n_insts * 4, mem_size)
        if 0 <= pc < end:
            _ = self._mem[pc:end].sum()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# 2. NEURAL ARM64 DECODER — wraps arm64_decoder.pt Transformer
# ─────────────────────────────────────────────────────────────────────────────

class _ARM64DecoderNet(nn.Module):
    """Reconstructed architecture matching arm64_decoder.pt state dict.

    Architecture (verified from saved weights):
      Encoder:
        bit_embed  : Embedding(2, 64)    — embed each bit (0/1) → 64d
        pos_embed  : Embedding(32, 64)   — positional embed per bit position
        combine    : Linear(128, 256)    — [N, 32, 128] → [N, 32, 256]
      Field extractor:
        field_queries: Parameter(6, 256) — 6 learned field queries
        self_attn    : MHA(256, heads=4) — [N, 32, 256] self-attention
        field_attn   : MHA(256, heads=4) — queries × bit-encoding cross-attn
        norm1, norm2 : LayerNorm(256)
      Decode heads (operating on each of 6 field outputs):
        category_head : Linear(256,128)→ReLU→Linear(128,10)
        operation_head: Linear(256,256)→ReLU→Linear(256,128)
        rd_head       : Linear(256, 32)
        rn_head       : Linear(256, 32)
        rm_head       : Linear(256, 32)
        imm_head      : Linear(256,256)→ReLU→Linear(256,26)
        flags_head    : Linear(256, 3)
      Refine:
        Linear(1536, 512)   — fuse all 6 field outputs → 512d
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Module()
        self.encoder.bit_embed = nn.Embedding(2, 64)
        self.encoder.pos_embed = nn.Embedding(32, 64)
        self.encoder.combine   = nn.Linear(128, 256)

        # Field extractor
        self.field_extractor = nn.Module()
        self.field_extractor.field_queries = nn.Parameter(torch.zeros(6, 256))
        self.field_extractor.self_attn = nn.MultiheadAttention(
            256, num_heads=4, batch_first=True, bias=True)
        self.field_extractor.field_attn = nn.MultiheadAttention(
            256, num_heads=4, batch_first=True, bias=True)
        self.field_extractor.norm1 = nn.LayerNorm(256)
        self.field_extractor.norm2 = nn.LayerNorm(256)

        # Decode heads
        class _Heads(nn.Module):
            def __init__(self):
                super().__init__()
                self.category_head  = nn.Sequential(
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 10))
                self.operation_head = nn.Sequential(
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128))
                self.rd_head    = nn.Linear(256, 32)
                self.rn_head    = nn.Linear(256, 32)
                self.rm_head    = nn.Linear(256, 32)
                self.imm_head   = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 26))
                self.flags_head = nn.Linear(256, 3)
        self.decoder_head = _Heads()

        # Refine (fuses all 6 field representations)
        self.refine = nn.Sequential(nn.Linear(1536, 512))

    def forward(self, insts: torch.Tensor) -> dict:
        """
        Args:
            insts: [N] int32 instruction words
        Returns dict with keys: operation [N], rd [N], rn [N], rm [N], imm [N], flags [N,3]
        """
        N = insts.shape[0]
        device = insts.device

        # Bit expansion: [N] → [N, 32]
        shifts = torch.arange(32, device=device, dtype=torch.int32)
        bits   = ((insts.unsqueeze(1).int() >> shifts.unsqueeze(0)) & 1).long()  # [N, 32]

        # Embed bits + positions: [N, 32, 64] + [N, 32, 64] → [N, 32, 128]
        pos_idx = torch.arange(32, device=device)
        bit_e   = self.encoder.bit_embed(bits)                    # [N, 32, 64]
        pos_e   = self.encoder.pos_embed(pos_idx).unsqueeze(0)    # [1, 32, 64]
        x       = torch.cat([bit_e, pos_e.expand(N, -1, -1)], dim=2)  # [N, 32, 128]
        x       = F.relu(self.encoder.combine(x))                 # [N, 32, 256]

        # Self-attention over bit positions
        x, _  = self.field_extractor.self_attn(x, x, x)          # [N, 32, 256]
        x     = self.field_extractor.norm1(x)

        # Cross-attention: 6 field queries attend to bit encoding
        q = self.field_extractor.field_queries.unsqueeze(0).expand(N, -1, -1)  # [N, 6, 256]
        fields, _ = self.field_extractor.field_attn(q, x, x)                   # [N, 6, 256]
        fields    = self.field_extractor.norm2(fields)

        # Aggregate fields: mean over 6 queries for shared representation
        agg = fields.mean(dim=1)            # [N, 256]
        fused = self.refine(fields.reshape(N, -1))  # [N, 1536] → [N, 512]

        # Decode heads
        op_logits  = self.decoder_head.operation_head(agg)  # [N, 128]
        rd_logits  = self.decoder_head.rd_head(agg)          # [N, 32]
        rn_logits  = self.decoder_head.rn_head(agg)          # [N, 32]
        rm_logits  = self.decoder_head.rm_head(agg)          # [N, 32]
        imm_logits = self.decoder_head.imm_head(agg)         # [N, 26]

        return {
            "operation": op_logits.argmax(dim=1),   # [N] predicted operation index
            "rd":  rd_logits.argmax(dim=1),          # [N] destination register
            "rn":  rn_logits.argmax(dim=1),          # [N] source register A
            "rm":  rm_logits.argmax(dim=1),          # [N] source register B
            "imm": imm_logits,                        # [N, 26] raw imm logits
            "confidence": op_logits.softmax(dim=1).max(dim=1).values,  # [N] confidence
        }


class NeuralARM64Decoder:
    """Loads arm64_decoder.pt and decodes batches of ARM64 instructions.

    Falls back to the classical op_type_table when:
    - Model file not found
    - Confidence below threshold
    - Any forward-pass error
    """

    CONF_THRESHOLD = 0.70   # below this: fall back to table-based decode

    def __init__(self, op_type_table: torch.Tensor,
                 models_dir: Path = MODELS_DIR,
                 device: str = "cpu"):
        self._table    = op_type_table   # [256] fallback lookup
        self._device   = device
        self._model: Optional[_ARM64DecoderNet] = None
        self._loaded   = False
        self._model_path = models_dir / "decoder" / "arm64_decoder.pt"
        self._load()

    def _load(self):
        if not self._model_path.exists():
            return
        try:
            net = _ARM64DecoderNet()
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=False)  # strict=False: ignore extra keys
            net.eval()
            net = net.to(self._device)
            self._model  = net
            self._loaded = True
        except Exception:
            self._model  = None
            self._loaded = False

    def decode_batch(self, insts: torch.Tensor,
                     fallback_ops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode N instructions using the neural decoder.

        Args:
            insts:       [N] int64 instruction words
            fallback_ops:[N] op types from classical table decode

        Returns:
            (ops [N], rds [N], rns [N]) — neural decode where confident,
            classical decode otherwise.
        """
        if self._model is None or not self._loaded:
            rds = insts & 0x1F
            rns = (insts >> 5) & 0x1F
            return fallback_ops, rds, rns

        N = insts.shape[0]
        try:
            with torch.no_grad():
                out = self._model(insts.int().to(self._device))
            conf  = out["confidence"]                        # [N]
            high  = conf >= self.CONF_THRESHOLD              # [N] bool mask
            neural_ops = out["operation"].to(fallback_ops.device)
            neural_rds = out["rd"].to(fallback_ops.device)
            neural_rns = out["rn"].to(fallback_ops.device)
            high = high.to(fallback_ops.device)
            ops = torch.where(high, neural_ops, fallback_ops)
            rds = torch.where(high, neural_rds, insts & 0x1F)
            rns = torch.where(high, neural_rns, (insts >> 5) & 0x1F)
            return ops, rds, rns
        except Exception:
            return fallback_ops, insts & 0x1F, (insts >> 5) & 0x1F

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEURAL SPECULATOR — checkpoint / rollback for branch speculation
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSpeculator:
    """Enables speculative execution past branch instructions.

    How it works:
      1. run_full_neural() sees a branch approaching.
      2. NeuralBranchPredictor gives taken-probability p.
      3. If p > CONF_THRESHOLD or p < (1 - CONF_THRESHOLD): speculate.
      4. Speculator saves a register-file + PC checkpoint.
      5. Execution continues on the predicted path.
      6. When the branch actually resolves:
           - Prediction correct  → commit() discards checkpoint.
           - Prediction wrong    → rollback() restores checkpoint.
    """

    CONF_THRESHOLD = 0.80

    def __init__(self):
        self._active      = False
        self._ckpt_regs: Optional[torch.Tensor] = None
        self._ckpt_pc:   Optional[torch.Tensor] = None
        self._ckpt_flags: Optional[torch.Tensor] = None
        self._pred_taken: bool = False
        self.commits   = 0
        self.rollbacks = 0

    def should_speculate(self, taken_prob: float) -> tuple[bool, bool]:
        """Return (should_speculate, predicted_taken)."""
        if taken_prob >= self.CONF_THRESHOLD:
            return True, True
        if taken_prob <= (1.0 - self.CONF_THRESHOLD):
            return True, False
        return False, False

    def enter(self, regs: torch.Tensor, pc: torch.Tensor,
              flags: torch.Tensor, pred_taken: bool):
        """Save checkpoint before executing speculative path."""
        self._ckpt_regs  = regs.clone()
        self._ckpt_pc    = pc.clone()
        self._ckpt_flags = flags.clone()
        self._pred_taken = pred_taken
        self._active     = True

    def commit(self):
        """Prediction was correct — discard checkpoint."""
        self._active = False
        self._ckpt_regs = self._ckpt_pc = self._ckpt_flags = None
        self.commits += 1

    def rollback(self, regs: torch.Tensor, pc: torch.Tensor,
                 flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction was wrong — restore checkpoint and return corrected (pc, flags)."""
        if not self._active or self._ckpt_regs is None:
            return pc, flags
        regs[:] = self._ckpt_regs
        pc       = self._ckpt_pc.clone()
        flags[:] = self._ckpt_flags
        self._active = False
        self._ckpt_regs = self._ckpt_pc = self._ckpt_flags = None
        self.rollbacks += 1
        return pc, flags

    @property
    def active(self) -> bool:
        return self._active

    @property
    def predicted_taken(self) -> bool:
        return self._pred_taken

    def accuracy(self) -> float:
        total = self.commits + self.rollbacks
        return self.commits / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEURAL CACHE MANAGER — cache_replace.pt LSTM for superblock eviction
# ─────────────────────────────────────────────────────────────────────────────

class _CacheReplaceNet(nn.Module):
    """Reconstructed architecture matching cache_replace.pt state dict.

    lstm:   LSTM(input_size=4, hidden_size=64) — processes per-entry features
    scorer: Linear(68, 64) → ReLU → Linear(64, 1)
             input: concat(lstm_hidden[64], entry_features[4]) = 68
    """

    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(4, 64, num_layers=1, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(68, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [1, N, 4] — batch=1, N cache entries, 4 features each
                  features: [recency, frequency, valid, tag_norm]
        Returns:  [N] eviction scores (higher = evict first)
        """
        out, _ = self.lstm(features)        # [1, N, 64]
        hidden = out.squeeze(0)             # [N, 64]
        feat   = features.squeeze(0)       # [N, 4]
        combined = torch.cat([hidden, feat], dim=1)  # [N, 68]
        scores = self.scorer(combined).squeeze(1)    # [N]
        return scores


class NeuralCacheManager:
    """Uses cache_replace.pt LSTM to decide which superblock to evict.

    Replaces the heuristic circular-buffer policy with a trained LSTM
    that scores each cache entry based on recency, frequency, and validity.
    Falls back to LRU when model unavailable.
    """

    def __init__(self, n_entries: int = 8, models_dir: Path = MODELS_DIR,
                 device: str = "cpu"):
        self._n      = n_entries
        self._dev    = device
        self._net: Optional[_CacheReplaceNet] = None
        self._loaded = False
        self._access_count = [0] * n_entries
        self._last_access  = [0] * n_entries
        self._tick         = 0
        self._load(models_dir)

    def _load(self, models_dir: Path):
        path = models_dir / "os" / "cache_replace.pt"
        if not path.exists():
            return
        try:
            net   = _CacheReplaceNet()
            state = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=True)
            net.eval()
            self._net    = net
            self._loaded = True
        except Exception:
            self._loaded = False

    def on_access(self, slot: int):
        """Record access to a superblock slot."""
        self._tick += 1
        if 0 <= slot < self._n:
            self._access_count[slot] += 1
            self._last_access[slot]   = self._tick

    def evict_slot(self, valid_mask: list[bool]) -> int:
        """Return the index of the slot to evict."""
        # First try an empty slot
        for i, valid in enumerate(valid_mask):
            if not valid:
                return i
        if self._loaded and self._net is not None:
            return self._neural_evict(valid_mask)
        return self._lru_evict(valid_mask)

    def _lru_evict(self, valid_mask: list[bool]) -> int:
        best_slot, oldest = 0, float('inf')
        for i in range(self._n):
            if self._last_access[i] < oldest:
                oldest, best_slot = self._last_access[i], i
        return best_slot

    def _neural_evict(self, valid_mask: list[bool]) -> int:
        """Run LSTM scorer over all cache entries, evict highest-scored."""
        try:
            features = []
            for i in range(self._n):
                recency   = 1.0 / max(1, self._tick - self._last_access[i])
                frequency = math.log1p(self._access_count[i]) / 10.0
                valid     = 1.0 if (i < len(valid_mask) and valid_mask[i]) else 0.0
                tag_norm  = float(i) / max(1, self._n - 1)
                features.append([recency, frequency, valid, tag_norm])

            feat_t = torch.tensor([features], dtype=torch.float32)  # [1, N, 4]
            with torch.no_grad():
                scores = self._net(feat_t)   # [N]
            return int(scores.argmax().item())
        except Exception:
            return self._lru_evict(valid_mask)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# 5. NEURAL SYSCALL ROUTER — MLP routes syscall numbers to handler classes
# ─────────────────────────────────────────────────────────────────────────────

SYSCALL_ROUTES = {
    0: [64, 66],
    1: [93, 94],
    2: [215, 216, 226, 214],
    3: [63, 64, 56, 57, 62, 80, 79],
    4: [172, 178, 220, 260, 129],
    5: [113, 169],
    6: [134, 135, 132],
    7: [29, 25, 23, 59, 198],
}
_NUM_TO_CLASS = {num: cls for cls, nums in SYSCALL_ROUTES.items() for num in nums}
N_SYSCALL_CLASSES = len(SYSCALL_ROUTES)


class _SyscallRouterNet(nn.Module):
    N_SYSCALL_EMBED = 16
    N_REG_FEATS    = 6   # 6 register values, normalised

    def __init__(self):
        super().__init__()
        self.syscall_embed = nn.Embedding(512, self.N_SYSCALL_EMBED)
        self.mlp = nn.Sequential(
            nn.Linear(self.N_SYSCALL_EMBED + self.N_REG_FEATS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_SYSCALL_CLASSES),
        )

    def forward(self, syscall_num: torch.Tensor,
                reg_vals: torch.Tensor) -> torch.Tensor:
        num_clamped = syscall_num.clamp(0, 511).long()
        emb      = self.syscall_embed(num_clamped)           # [B, 16]
        reg_norm = (reg_vals.float() / (2**31 + 1)).clamp(-1, 1)  # [B, 6]
        x        = torch.cat([emb, reg_norm], dim=1)
        return self.mlp(x)


class NeuralSyscallRouter:
    """Routes syscalls via a trained MLP, falls back to class mapping if untrained.

    Pre-trains from the static SYSCALL_ROUTES table at construction so it
    is active from the very first syscall.
    """

    CONF_THRESHOLD = 0.75

    def __init__(self, device: str = "cpu"):
        self._device  = device
        self._net     = _SyscallRouterNet().to(device)
        self._trained = False
        self._trace_nums:   list[int] = []
        self._trace_regs:   list      = []
        self._trace_labels: list[int] = []
        self._pretrain_from_table()

    def _pretrain_from_table(self, n_per_syscall: int = 64, lr: float = 1e-3,
                              n_epochs: int = 200):
        """Train on synthetic samples from the static routing table.

        Generates n_per_syscall examples per known syscall number with
        randomised register values (which don't affect routing).
        This gives ~100% accuracy on known syscalls from the first call.
        """
        import random
        nums:   list[int] = []
        regs:   list      = []
        labels: list[int] = []
        for cls, syscall_list in SYSCALL_ROUTES.items():
            for num in syscall_list:
                for _ in range(n_per_syscall):
                    nums.append(num)
                    regs.append([random.randint(0, 0x7FFF) for _ in range(6)])
                    labels.append(cls)
        # Unknown syscalls → class 7 (misc)
        for num in range(512):
            if num not in _NUM_TO_CLASS:
                for _ in range(4):
                    nums.append(num)
                    regs.append([0] * 6)
                    labels.append(7)

        nums_t   = torch.tensor(nums,   dtype=torch.int64,  device=self._device)
        regs_t   = torch.tensor(regs,   dtype=torch.int64,  device=self._device)
        labels_t = torch.tensor(labels, dtype=torch.int64,  device=self._device)

        self._net.train()
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        try:
            for _ in range(n_epochs):
                logits = self._net(nums_t, regs_t)
                loss   = F.cross_entropy(logits, labels_t)
                opt.zero_grad()
                loss.backward()
                opt.step()
        except Exception:
            pass
        self._net.eval()
        self._trained = True

    def route(self, syscall_num: int, reg_vals: torch.Tensor) -> int:
        """Return routing class (0-7) for this syscall."""
        static = _NUM_TO_CLASS.get(syscall_num, 7)
        if not self._trained:
            return static
        try:
            num_t = torch.tensor([syscall_num], dtype=torch.int64, device=self._device)
            reg_t = reg_vals[:6].unsqueeze(0).to(self._device)
            with torch.no_grad():
                logits = self._net(num_t, reg_t)[0]
                probs  = logits.softmax(0)
                conf, pred = probs.max(0)
                if conf.item() >= self.CONF_THRESHOLD:
                    return int(pred.item())
        except Exception:
            pass
        return static

    def record(self, syscall_num: int, reg_vals: torch.Tensor, actual_class: int):
        self._trace_nums.append(syscall_num)
        self._trace_regs.append(reg_vals[:6].cpu().tolist())
        self._trace_labels.append(actual_class)

    def train_online(self, n_steps: int = 100, lr: float = 3e-4):
        """Fine-tune on accumulated syscall traces."""
        if len(self._trace_labels) < 16:
            return
        nums   = torch.tensor(self._trace_nums,   dtype=torch.int64,  device=self._device)
        regs   = torch.tensor(self._trace_regs,   dtype=torch.int64,  device=self._device)
        labels = torch.tensor(self._trace_labels, dtype=torch.int64,  device=self._device)
        self._net.train()
        opt = torch.optim.Adam(self._net.parameters(), lr=lr)
        for _ in range(n_steps):
            logits = self._net(nums, regs)
            loss   = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self._net.eval()
        self._trained = True
        self._trace_nums   = self._trace_nums[-64:]
        self._trace_regs   = self._trace_regs[-64:]
        self._trace_labels = self._trace_labels[-64:]


# ─────────────────────────────────────────────────────────────────────────────
# 6. NEURAL REGISTER FILE — register_file.pt XZR/SP disambiguation
# ─────────────────────────────────────────────────────────────────────────────

class _XZRDetectorNet(nn.Module):
    """Detects whether register 31 should be treated as XZR (reads 0)
    rather than SP.  Input: 5-bit one-hot register index.
    """
    def __init__(self):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, reg_idx: torch.Tensor) -> torch.Tensor:
        """reg_idx: [N] int64 register indices → [N] bool (True = is XZR)"""
        bits = ((reg_idx.unsqueeze(1) >> torch.arange(5, device=reg_idx.device)) & 1).float()
        logit = self.detector(bits).squeeze(1) / self.temp.clamp(min=0.01)
        return torch.sigmoid(logit) > 0.5


class _SPSwitchNet(nn.Module):
    """Determines whether register 31 is SP vs XZR in a given context.
    Input: 6-dimensional context (instruction opcode bits + access type).
    """
    def __init__(self):
        super().__init__()
        self.switch = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.is_31_detector = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [N, 6] → [N] bool (True = is SP, not XZR)"""
        logit = self.switch(context).squeeze(1) / self.temp.clamp(min=0.01)
        return torch.sigmoid(logit) > 0.5


class NeuralRegisterFile:
    """Wraps register_file.pt for neural XZR/SP disambiguation.

    register_file.pt contains:
      - xzr_detector: detects when reg 31 is used as XZR (returns 0 on read)
      - sp_switch:    in LDR/STR context, reg 31 is SP not XZR
      - base.index_encoder: encodes register index for routing

    Falls back to the classical rule (31=XZR for most ops, 31=SP for load/store).
    """

    def __init__(self, models_dir: Path = MODELS_DIR, device: str = "cpu"):
        self._device  = device
        self._xzr_net: Optional[_XZRDetectorNet] = None
        self._sp_net:  Optional[_SPSwitchNet]    = None
        self._loaded  = False
        self._load(models_dir)

    def _load(self, models_dir: Path):
        path = models_dir / "register" / "register_file.pt"
        if not path.exists():
            return
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)

            xzr = _XZRDetectorNet()
            xzr.detector[0].weight.data = sd["xzr_detector.detector.0.weight"]
            xzr.detector[0].bias.data   = sd["xzr_detector.detector.0.bias"]
            xzr.detector[2].weight.data = sd["xzr_detector.detector.2.weight"]
            xzr.detector[2].bias.data   = sd["xzr_detector.detector.2.bias"]
            xzr.detector[4].weight.data = sd["xzr_detector.detector.4.weight"]
            xzr.detector[4].bias.data   = sd["xzr_detector.detector.4.bias"]
            xzr.temp.data               = sd["xzr_detector.temp"].reshape(1)
            xzr.eval()
            self._xzr_net = xzr

            sp = _SPSwitchNet()
            sp.switch[0].weight.data = sd["sp_switch.switch.0.weight"]
            sp.switch[0].bias.data   = sd["sp_switch.switch.0.bias"]
            sp.switch[2].weight.data = sd["sp_switch.switch.2.weight"]
            sp.switch[2].bias.data   = sd["sp_switch.switch.2.bias"]
            sp.switch[4].weight.data = sd["sp_switch.switch.4.weight"]
            sp.switch[4].bias.data   = sd["sp_switch.switch.4.bias"]
            sp.switch[6].weight.data = sd["sp_switch.switch.6.weight"]
            sp.switch[6].bias.data   = sd["sp_switch.switch.6.bias"]
            sp.is_31_detector[0].weight.data = sd["sp_switch.is_31_detector.0.weight"]
            sp.is_31_detector[0].bias.data   = sd["sp_switch.is_31_detector.0.bias"]
            sp.is_31_detector[2].weight.data = sd["sp_switch.is_31_detector.2.weight"]
            sp.is_31_detector[2].bias.data   = sd["sp_switch.is_31_detector.2.bias"]
            sp.temp.data = sd["sp_switch.temp"].reshape(1)
            sp.eval()
            self._sp_net  = sp

            self._loaded = True
        except Exception:
            self._loaded = False

    def apply_xzr_mask(self, reg_indices: torch.Tensor,
                        reg_vals: torch.Tensor,
                        is_memory_op: bool = False) -> torch.Tensor:
        """Zero out values where register 31 is XZR (not SP).

        Classical rule: in memory ops (LDR/STR), r31 = SP. Elsewhere, r31 = XZR (=0).
        Neural rule: run xzr_detector on all indices, zero where XZR is predicted.

        Args:
            reg_indices: [N] int64 register indices (0-31)
            reg_vals:    [N] int64 register values from gather
            is_memory_op: if True, r31 is SP → don't zero

        Returns: [N] int64 with XZR reads corrected to 0
        """
        if is_memory_op:
            return reg_vals  # r31 = SP in memory context

        if self._loaded and self._xzr_net is not None:
            try:
                with torch.no_grad():
                    is_xzr = self._xzr_net(reg_indices.to("cpu"))  # [N] bool
                is_xzr = is_xzr.to(reg_vals.device)
                return torch.where(is_xzr, torch.zeros_like(reg_vals), reg_vals)
            except Exception:
                pass

        # Classical fallback: any r31 read outside memory ops = 0
        xzr_mask = (reg_indices == 31)
        return torch.where(xzr_mask, torch.zeros_like(reg_vals), reg_vals)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# 7. NEURAL MEMORY ARITHMETIC — pointer.pt/stack.pt neural address computation
# ─────────────────────────────────────────────────────────────────────────────

class _FullAdderNet(nn.Module):
    """Reconstructed carry combiner from pointer.pt / stack.pt.

    All three memory models share the same addr_arith.full_adder architecture:
      net.0: Linear(3, 64) + ReLU  (G, P, carry_in → features)
      net.2: Linear(64, 32) + ReLU
      net.4: Linear(32, 2)         → (G_out, P_out) logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralMemoryArithmetic:
    """Neural address computation using pointer.pt / stack.pt.

    All three memory models (pointer, stack, function_call) share the same
    full_adder carry-combiner architecture.  Loads pointer.pt by default.

    Used in Phase 4 of run_woven() to compute LDR/STR effective addresses
    via neural ADD instead of tensor ops.  Falls back to tensor addition
    when model unavailable.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, device: str = "cpu"):
        self._device = device
        self._adder: Optional[_FullAdderNet] = None
        self._adder_device: Optional[torch.device] = None
        self._loaded = False
        self._load(models_dir)

    def _load(self, models_dir: Path):
        # Try pointer.pt first, then stack.pt
        for name in ("pointer", "stack"):
            path = models_dir / "memory" / f"{name}.pt"
            if not path.exists():
                continue
            try:
                sd = torch.load(path, map_location="cpu", weights_only=True)
                net = _FullAdderNet()
                net.net[0].weight.data = sd["addr_arith.full_adder.net.0.weight"]
                net.net[0].bias.data   = sd["addr_arith.full_adder.net.0.bias"]
                net.net[2].weight.data = sd["addr_arith.full_adder.net.2.weight"]
                net.net[2].bias.data   = sd["addr_arith.full_adder.net.2.bias"]
                net.net[4].weight.data = sd["addr_arith.full_adder.net.4.weight"]
                net.net[4].bias.data   = sd["addr_arith.full_adder.net.4.bias"]
                net.eval()
                self._adder        = net
                self._adder_device = torch.device("cpu")
                self._loaded       = True
                return
            except Exception:
                continue

    def compute_address(self, base: torch.Tensor,
                         offset: torch.Tensor) -> torch.Tensor:
        """Neural add: base + offset → effective address.

        Uses the pointer.pt carry combiner bit-by-bit (32-bit), then sign-extends.
        Runs on the same device as `base` (lazy model migration on first non-CPU call).
        Falls back to tensor addition when model unavailable.

        Args:
            base:   [N] int64
            offset: [N] int64
        Returns:    [N] int64 effective addresses
        """
        if not self._loaded or self._adder is None:
            return base + offset

        try:
            device = base.device
            # Lazy model migration to input device (e.g. MPS/CUDA)
            if self._adder_device != device:
                self._adder        = self._adder.to(device)
                self._adder_device = device

            N = base.shape[0]
            with torch.no_grad():
                result   = torch.zeros(N, dtype=torch.int64,    device=device)
                carry    = torch.zeros(N, dtype=torch.float32,  device=device)
                base32   = (base   & 0xFFFFFFFF).long()
                offset32 = (offset & 0xFFFFFFFF).long()

                for bit in range(32):
                    a_bits = ((base32   >> bit) & 1).float()  # [N]
                    b_bits = ((offset32 >> bit) & 1).float()  # [N]
                    # G = AND(a, b), P = XOR(a, b)
                    g_init = a_bits * b_bits
                    p_init = (a_bits + b_bits - 2 * g_init)   # XOR via arith
                    inp    = torch.stack([g_init, p_init, carry], dim=1)  # [N, 3]
                    out    = self._adder(inp)                              # [N, 2]
                    g_out  = torch.sigmoid(out[:, 0]) > 0.5
                    # Sum bit: P_orig XOR carry_in (standard full-adder)
                    sum_bit = (p_init.bool() ^ carry.bool()).long()
                    result  = result | (sum_bit << bit)
                    carry   = g_out.float()

            # Sign-extend 32-bit result to int64
            sign_mask = result & 0x80000000
            ext       = torch.tensor(0xFFFFFFFF00000000, dtype=torch.int64, device=device)
            result    = torch.where(sign_mask.bool(), result | ext, result)
            return result
        except Exception:
            return base + offset

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ─────────────────────────────────────────────────────────────────────────────
# 8. FULL NEURAL PIPELINE — orchestrates all stages
# ─────────────────────────────────────────────────────────────────────────────

class FullNeuralPipeline:
    """Combines all neural components into a unified CPU pipeline.

    Components:
        prefetcher   : NeuralPrefetcher     — prefetch.pt LSTM memory prediction
        decoder      : NeuralARM64Decoder   — arm64_decoder.pt Transformer
        speculator   : NeuralSpeculator     — branch speculation checkpoint
        cache_mgr    : NeuralCacheManager   — cache_replace.pt LSTM eviction
        syscall_rtr  : NeuralSyscallRouter  — pre-trained MLP (active from start)
        reg_file     : NeuralRegisterFile   — register_file.pt XZR/SP detection
        mem_arith    : NeuralMemoryArithmetic — pointer.pt neural address compute
        branch_pred  : NeuralBranchPredictor (from neural_weave) — taken/not
        alu_weave    : NeuralWeaveBatchALU (from neural_weave) — ALU execution
    """

    def __init__(self, cpu_ref, neural_ops):
        from .neural_weave import NeuralWeaveBatchALU, NeuralBranchPredictor
        device = str(cpu_ref.device)

        self.prefetcher  = NeuralPrefetcher(
            oracle   = cpu_ref.memory_oracle,
            memory   = cpu_ref.memory,
            interval = 16,
        )
        self.decoder = NeuralARM64Decoder(
            op_type_table = cpu_ref.op_type_table,
            device        = device,
        )
        self.speculator  = NeuralSpeculator()
        self.cache_mgr   = NeuralCacheManager(
            n_entries = int(cpu_ref._sb_valid.shape[0]),
            device    = device,
        )
        self.syscall_rtr = NeuralSyscallRouter(device=device)
        self.reg_file    = NeuralRegisterFile(device=device)
        self.mem_arith   = NeuralMemoryArithmetic(device=device)
        self.branch_pred = NeuralBranchPredictor().to(cpu_ref.device)
        self.branch_pred.eval()
        self.alu_weave   = NeuralWeaveBatchALU(neural_ops)

        self._branch_step   = 0
        self._syscall_step  = 0
        self.branch_train_every  = 2048
        self.syscall_train_every = 512

    def on_branch(self, pc: int, cond: int, taken: bool,
                  regs: torch.Tensor, flags: torch.Tensor) -> None:
        self.branch_pred.record(pc, cond, taken)
        self._branch_step += 1
        if self._branch_step % self.branch_train_every == 0:
            self.branch_pred.train_online()

    def on_syscall(self, num: int, regs: torch.Tensor):
        actual_class = _NUM_TO_CLASS.get(num, 7)
        self.syscall_rtr.record(num, regs, actual_class)
        self._syscall_step += 1
        if self._syscall_step % self.syscall_train_every == 0:
            self.syscall_rtr.train_online()

    def stats(self) -> dict:
        return {
            "prefetcher_loaded":    self.prefetcher.is_loaded,
            "decoder_loaded":       self.decoder.is_loaded,
            "cache_loaded":         self.cache_mgr.is_loaded,
            "reg_file_loaded":      self.reg_file.is_loaded,
            "mem_arith_loaded":     self.mem_arith.is_loaded,
            "syscall_trained":      self.syscall_rtr._trained,
            "speculator_accuracy":  self.speculator.accuracy(),
            "speculator_commits":   self.speculator.commits,
            "speculator_rollbacks": self.speculator.rollbacks,
            "branch_trained":       self.branch_pred._trained,
        }
