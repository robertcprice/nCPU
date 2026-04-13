"""Metal Neural ALU — runs trained .pt models on Metal GPU via native Rust shader.

Exports weights from NeuralOps (.pt files) and dispatches ADD/SUB/AND/OR/XOR/MUL
entirely on the Metal GPU using the neural_alu.rs kernel — no PyTorch inference
at runtime, just GPU buffer reads and Metal shader execution.

IPS profile (Apple M-series):
  PyTorch MPS woven:     ~21K  IPS  (fully neural, Python/PyTorch dispatch)
  Metal neural ALU:      ~500K–1M IPS (same weights, native Metal shader)

Usage:
    from ncpu.neural.metal_neural_alu import MetalNeuralALU, load_metal_neural_alu
    alu = load_metal_neural_alu(neural_ops)  # pass a loaded NeuralOps instance
    if alu.available:
        results = alu.add_batch(a_list, b_list)
        results = alu.logical_batch(a_list, b_list, op_idx=2)  # 0=AND 1=OR 2=XOR
        results = alu.mul_batch(a_list, b_list)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ncpu.model.neural_ops import NeuralOps


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_cc_weights(ops: "NeuralOps") -> Optional[list[float]]:
    """Extract carry_combiner weights as flat float list (2466 values).

    Layout matches neural_alu.rs NEURAL_ALU_SHADER offsets:
      [0    .. 255  ]  FC1 weight [64, 4]
      [256  .. 319  ]  FC1 bias   [64]
      [320  .. 2367 ]  FC2 weight [32, 64]
      [2368 .. 2399 ]  FC2 bias   [32]
      [2400 .. 2463 ]  FC3 weight [2, 32]
      [2464 .. 2465 ]  FC3 bias   [2]
    """
    cc = ops._carry_combiner
    if cc is None:
        return None
    try:
        sd = cc.state_dict()
        flat: list[float] = []
        # FC1
        flat.extend(sd["net.0.weight"].flatten().tolist())   # [64, 4]
        flat.extend(sd["net.0.bias"].tolist())               # [64]
        # FC2
        flat.extend(sd["net.2.weight"].flatten().tolist())   # [32, 64]
        flat.extend(sd["net.2.bias"].tolist())               # [32]
        # FC3
        flat.extend(sd["net.4.weight"].flatten().tolist())   # [2, 32]
        flat.extend(sd["net.4.bias"].tolist())               # [2]
        assert len(flat) == 2466, f"cc_weights: expected 2466, got {len(flat)}"
        return flat
    except Exception:
        return None


def _extract_truth_tables(ops: "NeuralOps") -> Optional[list[float]]:
    """Extract logical truth_tables as flat float list (28 values = [7, 4] row-major)."""
    log = ops._logical
    if log is None:
        return None
    try:
        tt = log.truth_tables
        flat = tt.data.flatten().tolist()
        assert len(flat) == 28, f"truth_tables: expected 28, got {len(flat)}"
        return flat
    except Exception:
        return None


def _extract_mul_lut(ops: "NeuralOps") -> Optional[list[float]]:
    """Extract multiply LUT as flat float list (256*256*16 = 1 048 576 values)."""
    mul = ops._multiplier
    if mul is None:
        return None
    try:
        lut = mul.lut.table  # [256, 256, 16]
        flat = lut.detach().flatten().tolist()
        assert len(flat) == 256 * 256 * 16
        return flat
    except Exception:
        return None


def _extract_asr_rol_luts() -> Optional[tuple[list[float], list[float]]]:
    """Precompute ASR and ROL LUTs from asr.pt / rol.pt.

    ASR LUT: fill positions replaced by one_hot(31) so sign bit is propagated.
    ROL LUT: 64-bit rotate-left, LUT[k,i,j] = weight from bit j → output bit i.
    Both return flat [64*64*64] float lists (1 MB each).
    """
    import torch
    import torch.nn as nn
    from pathlib import Path

    MODELS_DIR = Path(__file__).parent.parent.parent / "models"
    asr_path = MODELS_DIR / "shifts" / "asr.pt"
    rol_path = MODELS_DIR / "shifts" / "rol.pt"
    if not asr_path.exists() or not rol_path.exists():
        return None

    try:
        sd_asr = torch.load(str(asr_path), map_location="cpu", weights_only=True)
        sd_rol = torch.load(str(rol_path), map_location="cpu", weights_only=True)

        def _build_decoder_bn(sd):
            """Reconstruct shift_decoder with BatchNorm from state dict."""
            seq = nn.Sequential(
                nn.Linear(64, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, 64),
            )
            seq[0].weight.data = sd["shift_decoder.0.weight"]
            seq[0].bias.data   = sd["shift_decoder.0.bias"]
            seq[1].weight.data = sd["shift_decoder.1.weight"]
            seq[1].bias.data   = sd["shift_decoder.1.bias"]
            seq[3].weight.data = sd["shift_decoder.3.weight"]
            seq[3].bias.data   = sd["shift_decoder.3.bias"]
            seq[4].weight.data = sd["shift_decoder.4.weight"]
            seq[4].bias.data   = sd["shift_decoder.4.bias"]
            seq[6].weight.data = sd["shift_decoder.6.weight"]
            seq[6].bias.data   = sd["shift_decoder.6.bias"]
            return seq

        def _build_index_net(sd):
            net = nn.Sequential(
                nn.Linear(128, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 64),
            )
            net[0].weight.data = sd["index_net.0.weight"]
            net[0].bias.data   = sd["index_net.0.bias"]
            net[2].weight.data = sd["index_net.2.weight"]
            net[2].bias.data   = sd["index_net.2.bias"]
            net[4].weight.data = sd["index_net.4.weight"]
            net[4].bias.data   = sd["index_net.4.bias"]
            net.eval()
            return net

        def _compute_shift_soft(sd, decoder):
            """Run all 64 shift amounts through BN decoder as one batch."""
            amts = torch.stack([
                torch.tensor([(k >> i) & 1 for i in range(64)], dtype=torch.float32)
                for k in range(64)
            ])  # [64, 64]
            decoder.train()  # use batch statistics (no saved running_mean/var)
            with torch.no_grad():
                enc = decoder(amts.repeat(10, 1))[-64:]  # [64, 64]
                return torch.softmax(enc, dim=1)  # [64, 64]

        # ── ASR LUT ───────────────────────────────────────────────────────
        asr_decoder  = _build_decoder_bn(sd_asr)
        asr_idx_net  = _build_index_net(sd_asr)
        asr_fill_net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        asr_fill_net[0].weight.data = sd_asr["fill_net.0.weight"]
        asr_fill_net[0].bias.data   = sd_asr["fill_net.0.bias"]
        asr_fill_net[2].weight.data = sd_asr["fill_net.2.weight"]
        asr_fill_net[2].bias.data   = sd_asr["fill_net.2.bias"]
        asr_fill_net.eval()
        asr_temp = sd_asr["temperature"]

        asr_soft = _compute_shift_soft(sd_asr, asr_decoder)
        positions = torch.eye(64)
        lut_asr = torch.zeros(64, 64, 64)
        # One-hot for sign bit (bit 31) — used for fill positions
        sign_bit_row = torch.zeros(64)
        sign_bit_row[31] = 1.0  # all weight on bit 31

        with torch.no_grad():
            for k in range(64):
                shift_exp = asr_soft[k].unsqueeze(0).expand(64, -1)
                combined  = torch.cat([positions, shift_exp], dim=1)  # [64, 128]
                idx_w     = torch.softmax(asr_idx_net(combined) / asr_temp, dim=1)  # [64, 64]
                fill_gate = torch.sigmoid(asr_fill_net(combined).squeeze(1)) > 0.5  # [64]
                # Where fill is active: route all weight to sign bit (bit 31)
                for i in range(64):
                    if fill_gate[i].item():
                        lut_asr[k, i] = sign_bit_row
                    else:
                        lut_asr[k, i] = idx_w[i]

        # ── ROL LUT ───────────────────────────────────────────────────────
        rol_decoder = _build_decoder_bn(sd_rol)
        rol_idx_net = _build_index_net(sd_rol)
        rol_temp    = sd_rol["temperature"]

        rol_soft = _compute_shift_soft(sd_rol, rol_decoder)
        lut_rol  = torch.zeros(64, 64, 64)

        with torch.no_grad():
            for k in range(64):
                shift_exp = rol_soft[k].unsqueeze(0).expand(64, -1)
                combined  = torch.cat([positions, shift_exp], dim=1)
                idx_w     = torch.softmax(rol_idx_net(combined) / rol_temp, dim=1)
                lut_rol[k] = idx_w

        asr_flat = lut_asr.flatten().tolist()
        rol_flat = lut_rol.flatten().tolist()
        assert len(asr_flat) == 64 * 64 * 64
        assert len(rol_flat) == 64 * 64 * 64
        return asr_flat, rol_flat

    except Exception:
        return None


def _extract_shift_luts(ops: "NeuralOps") -> Optional[tuple[list[float], list[float]]]:
    """Precompute shift LUTs from NeuralShiftNet for all 64 shift amounts.

    Returns (lsl_flat, lsr_flat) each of length 64*64*64 = 262 144 floats.
    lut[k, i, j] = effective weight: source bit j → output bit i for shift_amount k.
    At runtime: output[i] = Σ_j(lut[k,i,j] * val_bits[j]) > 0.5
    """
    sl = ops._shifter_left
    sr = ops._shifter_right
    if sl is None or sr is None:
        return None
    try:
        import torch

        def _build_lut(model) -> list[float]:
            lut = torch.zeros(64, 64, 64)
            with torch.no_grad():
                for k in range(64):
                    amt_bits = torch.tensor(
                        [(k >> i) & 1 for i in range(64)], dtype=torch.float32
                    )
                    shift_enc = model.shift_decoder(amt_bits.unsqueeze(0))[0]  # [64]
                    shift_soft = torch.softmax(shift_enc, dim=0)               # [64]
                    positions = torch.eye(64)                                   # [64, 64]
                    shift_exp = shift_soft.unsqueeze(0).expand(64, -1)         # [64, 64]
                    combined = torch.cat([positions, shift_exp], dim=1)        # [64, 128]
                    idx_logits = model.index_net(combined)                     # [64, 64]
                    idx_w = torch.softmax(idx_logits / model.temperature, dim=1)  # [64, 64]
                    valid = (
                        torch.sigmoid(model.validity_net(combined).squeeze(1)) > 0.5
                    ).float()                                                   # [64]
                    lut[k] = idx_w * valid.unsqueeze(1)                        # [64, 64]
            return lut.flatten().tolist()

        lsl_flat = _build_lut(sl)
        lsr_flat = _build_lut(sr)
        assert len(lsl_flat) == 64 * 64 * 64
        assert len(lsr_flat) == 64 * 64 * 64
        return lsl_flat, lsr_flat
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MetalNeuralALU — main class
# ─────────────────────────────────────────────────────────────────────────────

class MetalNeuralALU:
    """Wraps NeuralALUKernel (Rust/Metal) with automatic weight loading.

    Falls back to None if Metal is unavailable or weights can't be extracted.
    """

    def __init__(self, kernel, has_mul: bool, has_shift: bool = False,
                 has_asr_rol: bool = False):
        self._kernel = kernel
        self._has_mul = has_mul
        self._has_shift = has_shift
        self._has_asr_rol = has_asr_rol

    @property
    def available(self) -> bool:
        return self._kernel is not None and self._kernel.is_ready()

    @property
    def mul_available(self) -> bool:
        return self._has_mul and self._kernel is not None and self._kernel.mul_ready()

    @property
    def shift_available(self) -> bool:
        return self._has_shift and self._kernel is not None and self._kernel.shift_ready()

    @property
    def asr_available(self) -> bool:
        return self._has_asr_rol and self._kernel is not None and self._kernel.asr_ready()

    @property
    def rol_available(self) -> bool:
        return self._has_asr_rol and self._kernel is not None and self._kernel.rol_ready()

    # ── ADD / SUB ─────────────────────────────────────────────────────────────

    def add_batch(self, a_vals: list[int], b_vals: list[int],
                  is_sub: bool = False, is_w32: bool = False) -> list[int]:
        """Run neural ADD (or SUB) for all (a, b) pairs via Metal GPU."""
        return self._kernel.execute_add(a_vals, b_vals, is_sub, is_w32)

    # ── Logical (AND / OR / XOR / …) ─────────────────────────────────────────

    def logical_batch(self, a_vals: list[int], b_vals: list[int],
                      op_idx: int) -> list[int]:
        """Run neural logical op via Metal GPU.
        op_idx: 0=AND, 1=OR, 2=XOR, 3=BIC, 4=ORN, 5=EON.
        """
        return self._kernel.execute_logical(a_vals, b_vals, op_idx)

    # ── LSL / LSR ─────────────────────────────────────────────────────────────

    def shift_batch(self, a_vals: list[int], shift_amts: list[int],
                    is_left: bool) -> list[int]:
        """Run neural LSL (is_left=True) or LSR (is_left=False) via Metal GPU."""
        if not self.shift_available:
            raise RuntimeError("Shift LUTs not loaded — pass load_shift=True to load_metal_neural_alu()")
        return self._kernel.execute_shift(a_vals, shift_amts, is_left)

    # ── ASR / ROL ─────────────────────────────────────────────────────────────

    def asr_batch(self, a_vals: list[int], shift_amts: list[int]) -> list[int]:
        """Neural ASR (arithmetic shift right, 32-bit) via Metal GPU."""
        if not self.asr_available:
            raise RuntimeError("ASR LUT not loaded")
        return self._kernel.execute_asr(a_vals, shift_amts)

    def rol_batch(self, a_vals: list[int], rot_amts: list[int]) -> list[int]:
        """Neural ROL (64-bit rotate left) via Metal GPU."""
        if not self.rol_available:
            raise RuntimeError("ROL LUT not loaded")
        return self._kernel.execute_rol(a_vals, rot_amts)

    # ── MUL ───────────────────────────────────────────────────────────────────

    def mul_batch(self, a_vals: list[int], b_vals: list[int]) -> list[int]:
        """Run neural MUL via Metal GPU using the byte-pair LUT."""
        if not self.mul_available:
            raise RuntimeError("MUL LUT not loaded — pass load_mul=True to load_metal_neural_alu()")
        return self._kernel.execute_mul(a_vals, b_vals)

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, n: int = 1024) -> dict:
        """Measure ADD throughput. Returns dict with n_ops, elapsed, ips."""
        n_ops, elapsed, ips = self._kernel.benchmark_add(n)
        return {"n_ops": n_ops, "elapsed_s": elapsed, "ips": ips}


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def load_metal_neural_alu(ops: "NeuralOps",
                           load_mul: bool = True,
                           load_shift: bool = True,
                           load_asr_rol: bool = True,
                           verbose: bool = False) -> MetalNeuralALU:
    """Load neural model weights and create a MetalNeuralALU instance.

    Args:
        ops:      A loaded NeuralOps instance (carry_combiner + logical models)
        load_mul: Also load the MUL LUT (4 MB, optional)
        verbose:  Print loading progress

    Returns:
        MetalNeuralALU with .available = True if everything loaded OK,
        or .available = False if Metal / weights are unavailable.
    """
    try:
        import importlib.util

        # Locate ncpu_metal — prefer the system path, fall back to known .so locations.
        # We use spec_from_file_location to avoid polluting sys.path with the venv,
        # which would break the system torch that loaded NeuralOps.
        # Always load via spec_from_file_location to avoid polluting sys.path
        # with the venv (which has an incompatible torch that breaks system torch).
        ncpu_metal = None
        _so_candidates = [
            "/Users/bobbyprice/projects/.venv/lib/python3.13/site-packages/ncpu_metal/ncpu_metal.abi3.so",
            "/Users/bobbyprice/projects/nCPU/kernels/rust_metal/ncpu_metal.abi3.so",
        ]
        import sys as _sys
        # If already imported (e.g. in a test), reuse it
        if "ncpu_metal" in _sys.modules and hasattr(_sys.modules["ncpu_metal"], "NeuralALUKernel"):
            ncpu_metal = _sys.modules["ncpu_metal"]
        else:
            for _so_path in _so_candidates:
                try:
                    spec = importlib.util.spec_from_file_location("ncpu_metal", _so_path)
                    if spec is not None and spec.loader is not None:
                        _m = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(_m)  # type: ignore[union-attr]
                        ncpu_metal = _m
                        _sys.modules["ncpu_metal"] = _m  # cache for next call
                        break
                except Exception:
                    continue
            if ncpu_metal is None:
                try:
                    import ncpu_metal as _m  # last-resort: from sys.path
                    ncpu_metal = _m
                except ImportError:
                    pass

        if ncpu_metal is None or not hasattr(ncpu_metal, "NeuralALUKernel"):
            if verbose:
                print("[MetalNeuralALU] NeuralALUKernel not available")
            return MetalNeuralALU(None, False)

        kernel = ncpu_metal.NeuralALUKernel()

        cc_weights = _extract_cc_weights(ops)
        tt_vals = _extract_truth_tables(ops)
        if cc_weights is None or tt_vals is None:
            if verbose:
                print("[MetalNeuralALU] Could not extract CC/logical weights")
            return MetalNeuralALU(None, False)

        kernel.load_weights(cc_weights, tt_vals)
        if verbose:
            print(f"[MetalNeuralALU] Loaded carry_combiner ({len(cc_weights)} params) "
                  f"+ truth_tables ({len(tt_vals)} values) → GPU buffer")

        has_mul = False
        if load_mul:
            lut_flat = _extract_mul_lut(ops)
            if lut_flat is not None:
                kernel.load_mul_lut(lut_flat)
                has_mul = True
                if verbose:
                    print(f"[MetalNeuralALU] Loaded MUL LUT ({len(lut_flat)} logits, 4 MB) → GPU buffer")
            elif verbose:
                print("[MetalNeuralALU] MUL LUT not available")

        has_shift = False
        if load_shift:
            shift_luts = _extract_shift_luts(ops)
            if shift_luts is not None:
                lsl_flat, lsr_flat = shift_luts
                kernel.load_shift_luts(lsl_flat, lsr_flat)
                has_shift = True
                if verbose:
                    print(f"[MetalNeuralALU] Loaded shift LUTs "
                          f"(LSL+LSR, {len(lsl_flat)} floats each, 1 MB each) → GPU buffers")
            elif verbose:
                print("[MetalNeuralALU] Shift LUTs not available")

        has_asr_rol = False
        if load_asr_rol:
            asr_rol_luts = _extract_asr_rol_luts()
            if asr_rol_luts is not None:
                asr_flat, rol_flat = asr_rol_luts
                kernel.load_asr_rol_luts(asr_flat, rol_flat)
                has_asr_rol = True
                if verbose:
                    print(f"[MetalNeuralALU] Loaded ASR+ROL LUTs "
                          f"({len(asr_flat)} floats each, 1 MB each) → GPU buffers")
            elif verbose:
                print("[MetalNeuralALU] ASR/ROL LUTs not available")

        return MetalNeuralALU(kernel, has_mul, has_shift, has_asr_rol)

    except Exception as e:
        if verbose:
            print(f"[MetalNeuralALU] Failed to initialize: {e}")
        return MetalNeuralALU(None, False)
