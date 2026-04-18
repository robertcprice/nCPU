"""
PyTorch SoftUniversalProgram — GPU-accelerated program synthesis via autograd.

Key design: precompute ALL slot/loop/return weight vectors in a handful of
batched matrix ops, then execute the sequential register-update pass using
those precomputed weights. Reduces ~1300 small MPS kernel launches to ~10.

Architecture (n_args=1):
  Pool = [a, c0..c5, v0..v2, s0..s5, p0, p1]  — 18 registers
  Init  (3 slots) → v0, v1, v2
  Loop  (6 slots × MAX_LOOP_ITER) → s0..s5, soft-gated by loop condition
  Post  (2 slots) → p0, p1
  Return: soft-select from full pool

Usage:
  python3 scripts/soft_synth.py --examples '[[[1],2],[[2],3]]' --n-steps 200
  python3 scripts/soft_synth.py --batch in.jsonl --out results.jsonl
"""

import sys, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants (must match synthesis.rs) ────────────────────────────────────
N_OPS_EXT   = 6    # +, -, *, /, %, identity
N_CMPS      = 6    # <, <=, ==, >=, >, !=
N_CONSTS    = 6
CONST_VALS  = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0]
MAX_LOOP_ITER = 32
N_INIT_SLOTS  = 3
N_LOOP_SLOTS  = 6
N_POST_SLOTS  = 2
N_UNIV_SLOTS  = 11  # 3 + 6 + 2

def _pool(n): return n + N_CONSTS + N_UNIV_SLOTS
def _lip(n):  return n + N_CONSTS + N_INIT_SLOTS
def _sps(p):  return N_OPS_EXT + 5*p + N_CMPS     # params per slot

def n_params_for(n_args):
    p   = _pool(n_args)
    lip = _lip(n_args)
    return (N_UNIV_SLOTS * _sps(p)
            + N_LOOP_SLOTS * lip          # loop-state init weights
            + N_CMPS + 2*p                # loop condition (cmp + lhs + rhs)
            + p                           # return weights
            + N_CONSTS)                   # learned constants


# ─── Soft primitives ─────────────────────────────────────────────────────────

def _soft_op_ext_batch(a, b, w):
    """a,b:(B,)  w:(N_OPS_EXT,) → (B,)"""
    safe_b = torch.where(b.abs() < 1e-6, torch.ones_like(b), b)
    # stack ops: (B, 6)
    ops = torch.stack([a+b, a-b, a*b, a/safe_b,
                       a-(a/safe_b).trunc()*safe_b, a], dim=1)
    return (ops * w.unsqueeze(0)).sum(1)

def _soft_cmp_batch(a, b, w, t):
    """a,b:(B,)  w:(N_CMPS,)  t:float → (B,)"""
    t_c = max(min(float(t), 2.0), 0.5)
    gv  = max(t_c*t_c*0.5, 0.125)
    d   = a - b
    sp  = torch.sigmoid(d / t_c)
    sn  = torch.sigmoid(-d / t_c)
    g   = torch.exp(-(d*d) / gv)
    cmps = torch.stack([sp, sn, sp, sn, g, 1.0 - g], dim=1)  # (B,6)
    return (cmps * w.unsqueeze(0)).sum(1)


class SoftUniversalProgram(nn.Module):
    def __init__(self, n_args: int = 1):
        super().__init__()
        self.n_args    = n_args
        self.p         = _pool(n_args)
        self.lip       = _lip(n_args)
        self.sps_      = _sps(self.p)
        np_            = n_params_for(n_args)
        self.params    = nn.Parameter(torch.zeros(np_))

        # Offsets into params vector
        self._lo_off   = N_UNIV_SLOTS * self.sps_                   # loop-init start
        self._lc_off   = self._lo_off + N_LOOP_SLOTS * self.lip     # loop cond start
        self._ret_off  = self._lc_off + N_CMPS + 2*self.p           # return weights start
        self._co_off   = self._ret_off + self.p                     # constants start

        # Init constants
        with torch.no_grad():
            for i, v in enumerate(CONST_VALS):
                self.params[self._co_off + i] = float(v)

    # ── Precompute all weight vectors in batched ops ──────────────────────────
    def _precompute_weights(self, temp: float):
        """
        Returns a dict of weight tensors:
          slots_w:  (N_UNIV_SLOTS, 7, p_or_N_OPS_EXT_or_N_CMPS)  — variable sizes
        Instead we return flat per-slot dicts (no dynamic shapes in one tensor).
        Uses 2 big softmax calls to cover all slot weights at once.
        """
        p, lip, sps = self.p, self.lip, self.sps_
        inv_t = 1.0 / temp

        # ── All slot logits (N_UNIV_SLOTS × sps) in one tensor ───────────────
        slot_logits = self.params[:N_UNIV_SLOTS * sps].view(N_UNIV_SLOTS, sps)
        # Each slot row: [op(6) | s1(p) | s2(p) | cmp(6) | gl(p) | gr(p) | el(p)]
        # Softmax each sub-region independently
        op_log  = slot_logits[:,  :N_OPS_EXT]                             # (S, 6)
        s1_log  = slot_logits[:,  N_OPS_EXT:N_OPS_EXT+p]                  # (S, p)
        s2_log  = slot_logits[:,  N_OPS_EXT+p:N_OPS_EXT+2*p]             # (S, p)
        cmp_log = slot_logits[:,  N_OPS_EXT+2*p:N_OPS_EXT+2*p+N_CMPS]   # (S, 6)
        gl_log  = slot_logits[:,  N_OPS_EXT+2*p+N_CMPS:N_OPS_EXT+3*p+N_CMPS]  # (S, p)
        gr_log  = slot_logits[:,  N_OPS_EXT+3*p+N_CMPS:N_OPS_EXT+4*p+N_CMPS]  # (S, p)
        el_log  = slot_logits[:,  N_OPS_EXT+4*p+N_CMPS:]                       # (S, p)

        op_w  = F.softmax(op_log  * inv_t, dim=1)   # (S, 6)
        s1_w  = F.softmax(s1_log  * inv_t, dim=1)   # (S, p)
        s2_w  = F.softmax(s2_log  * inv_t, dim=1)   # (S, p)
        cmp_w = F.softmax(cmp_log * inv_t, dim=1)   # (S, 6)
        gl_w  = F.softmax(gl_log  * inv_t, dim=1)   # (S, p)
        gr_w  = F.softmax(gr_log  * inv_t, dim=1)   # (S, p)
        el_w  = F.softmax(el_log  * inv_t, dim=1)   # (S, p)

        # ── Loop-init weights (N_LOOP_SLOTS × lip) ───────────────────────────
        lo_log = self.params[self._lo_off:self._lo_off + N_LOOP_SLOTS*lip]
        lo_w   = F.softmax(lo_log.view(N_LOOP_SLOTS, lip) * inv_t, dim=1)  # (6, lip)

        # ── Loop condition ────────────────────────────────────────────────────
        lco    = self._lc_off
        lc_cmp_w = F.softmax(self.params[lco:lco+N_CMPS] * inv_t, dim=0)          # (6,)
        lc_lhs_w = F.softmax(self.params[lco+N_CMPS:lco+N_CMPS+p] * inv_t, dim=0) # (p,)
        lc_rhs_w = F.softmax(self.params[lco+N_CMPS+p:lco+N_CMPS+2*p] * inv_t, dim=0)  # (p,)

        # ── Return weights ────────────────────────────────────────────────────
        ret_w = F.softmax(self.params[self._ret_off:self._ret_off+p] * inv_t, dim=0)  # (p,)

        return dict(op_w=op_w, s1_w=s1_w, s2_w=s2_w, cmp_w=cmp_w,
                    gl_w=gl_w, gr_w=gr_w, el_w=el_w,
                    lo_w=lo_w, lc_cmp_w=lc_cmp_w, lc_lhs_w=lc_lhs_w,
                    lc_rhs_w=lc_rhs_w, ret_w=ret_w)

    @staticmethod
    def _reg_mat(reg: list) -> torch.Tensor:
        """Stack list of (B,) tensors → (B, p). Called once per slot."""
        return torch.stack(reg, dim=1)

    def _exec_slot_precomp(self, slot: int, reg: list, w: dict, temp: float):
        """reg: list[p] of (B,) tensors → (B,). Uses precomputed weight slices."""
        s    = slot
        rm   = self._reg_mat(reg)          # (B, p) — one kernel launch
        s1   = (rm * w["s1_w"][s]).sum(1)  # (B,) dot product
        s2   = (rm * w["s2_w"][s]).sum(1)
        then_= _soft_op_ext_batch(s1, s2, w["op_w"][s])
        gate = _soft_cmp_batch(
                   (rm * w["gl_w"][s]).sum(1),
                   (rm * w["gr_w"][s]).sum(1),
                   w["cmp_w"][s], temp)
        else_= (rm * w["el_w"][s]).sum(1)
        return gate * then_ + (1.0 - gate) * else_

    def forward(self, inputs: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        """inputs:(B,n_args) → (B,)"""
        B, n = inputs.shape[0], self.n_args
        p, lip = self.p, self.lip
        dev = self.params.device

        # ── Precompute all weights ────────────────────────────────────────────
        w = self._precompute_weights(temp)

        # ── Register file as list of (B,) tensors (avoids inplace autograd issues)
        zeros = inputs.new_zeros(B)
        reg   = [zeros] * p  # share zero tensor for unused slots initially

        # Args
        for i in range(n):
            reg[i] = inputs[:, i]

        # Constants (learned, broadcast to B)
        for i in range(N_CONSTS):
            reg[n + i] = self.params[self._co_off + i].expand(B)

        # ── Phase 1: init slots ───────────────────────────────────────────────
        for s in range(N_INIT_SLOTS):
            reg = list(reg)   # shallow copy of list so we can update
            reg[n + N_CONSTS + s] = self._exec_slot_precomp(s, reg, w, temp)

        # ── Phase 2: loop state init ──────────────────────────────────────────
        # lip pool = [args, consts, v0..v2] — v0..v2 just updated by init slots
        rm_lip = self._reg_mat(reg[:lip])  # (B, lip) — computed after init slots
        for ls in range(N_LOOP_SLOTS):
            reg = list(reg)
            reg[n + N_CONSTS + N_INIT_SLOTS + ls] = (rm_lip * w["lo_w"][ls]).sum(1)

        # ── Phase 3: loop ─────────────────────────────────────────────────────
        for _it in range(MAX_LOOP_ITER):
            rm   = self._reg_mat(reg)      # (B, p) — recomputed each iter
            lhs  = (rm * w["lc_lhs_w"]).sum(1)
            rhs  = (rm * w["lc_rhs_w"]).sum(1)
            cond = _soft_cmp_batch(lhs, rhs, w["lc_cmp_w"], temp)  # (B,)
            for ls in range(N_LOOP_SLOTS):
                slot = N_INIT_SLOTS + ls
                idx  = n + N_CONSTS + slot
                out  = self._exec_slot_precomp(slot, reg, w, temp)
                reg  = list(reg)
                reg[idx] = cond * out + (1.0 - cond) * reg[idx]

        # ── Phase 4: post slots ───────────────────────────────────────────────
        for pi in range(N_POST_SLOTS):
            slot = N_INIT_SLOTS + N_LOOP_SLOTS + pi
            reg  = list(reg)
            reg[n + N_CONSTS + slot] = self._exec_slot_precomp(slot, reg, w, temp)

        # ── Return ────────────────────────────────────────────────────────────
        return sum(w["ret_w"][i] * reg[i] for i in range(p))

    def mse_loss(self, inputs, targets, temp):
        return F.mse_loss(self.forward(inputs, temp), targets)


# ─── Discrete integer evaluation (matches Rust discrete_eval exactly) ────────

def _argmax_slice(params: torch.Tensor, start: int, length: int) -> int:
    return int(params[start:start+length].argmax().item())

def discrete_eval(params: torch.Tensor, inputs: list, n_args: int) -> object:
    """
    Exact port of Rust SoftUniversalProgram::discrete_eval.
    Uses argmax + hard integer ops — NOT soft forward.
    Returns an int, or None if the program divides by zero / overflows.
    """
    p   = _pool(n_args)
    lip = _lip(n_args)
    sps = _sps(p)
    lo_off  = N_UNIV_SLOTS * sps
    lc_off  = lo_off + N_LOOP_SLOTS * lip
    ret_off = lc_off + N_CMPS + 2*p
    co_off  = ret_off + p

    # Build integer register file
    reg = [0] * p
    for i in range(n_args):
        reg[i] = int(inputs[i])
    for i in range(N_CONSTS):
        v = float(params[co_off + i].item())
        reg[n_args + i] = int(round(v)) if (v == v) else 0  # guard NaN

    def disc_exec_slot(slot):
        off   = slot * sps
        op_i  = _argmax_slice(params, off,                          N_OPS_EXT)
        s1_i  = _argmax_slice(params, off + N_OPS_EXT,              p)
        s2_i  = _argmax_slice(params, off + N_OPS_EXT + p,          p)
        cb    = off + N_OPS_EXT + 2*p
        cmp_i = _argmax_slice(params, cb,                           N_CMPS)
        gl_i  = _argmax_slice(params, cb + N_CMPS,                  p)
        gr_i  = _argmax_slice(params, cb + N_CMPS + p,              p)
        el_i  = _argmax_slice(params, cb + N_CMPS + 2*p,            p)

        s1 = reg[s1_i]; s2 = reg[s2_i]
        try:
            if   op_i == 0: tv = s1 + s2
            elif op_i == 1: tv = s1 - s2
            elif op_i == 2: tv = s1 * s2
            elif op_i == 3:
                if s2 == 0: return None
                tv = int(s1 / s2) if (s1 < 0) != (s2 < 0) and s1 % s2 != 0 else s1 // s2
            elif op_i == 4:
                if s2 == 0: return None
                # Rust truncated modulo: a - trunc(a/b)*b
                tv = s1 - int(s1 / s2) * s2
            else: tv = s1   # identity
        except OverflowError:
            return None
        gl = reg[gl_i]; gr = reg[gr_i]
        gate = [gl<gr, gl<=gr, gl==gr, gl>=gr, gl>gr, gl!=gr][cmp_i % N_CMPS]
        return tv if gate else reg[el_i]

    # Phase 1: init slots
    for s in range(N_INIT_SLOTS):
        v = disc_exec_slot(s)
        if v is None: return None
        reg[n_args + N_CONSTS + s] = v

    # Phase 2: loop state init
    for ls in range(N_LOOP_SLOTS):
        io    = lo_off + ls * lip
        src_i = _argmax_slice(params, io, lip)
        reg[n_args + N_CONSTS + N_INIT_SLOTS + ls] = reg[src_i]

    # Phase 3: loop
    cmp_i = _argmax_slice(params, lc_off,          N_CMPS)
    lhs_i = _argmax_slice(params, lc_off + N_CMPS, p)
    rhs_i = _argmax_slice(params, lc_off + N_CMPS + p, p)
    for _ in range(MAX_LOOP_ITER):
        lhs = reg[lhs_i]; rhs = reg[rhs_i]
        cont = [lhs<rhs, lhs<=rhs, lhs==rhs, lhs>=rhs, lhs>rhs, lhs!=rhs][cmp_i % N_CMPS]
        if not cont: break
        for ls in range(N_LOOP_SLOTS):
            v = disc_exec_slot(N_INIT_SLOTS + ls)
            if v is None: return None
            reg[n_args + N_CONSTS + N_INIT_SLOTS + ls] = v

    # Phase 4: post slots
    for pi in range(N_POST_SLOTS):
        v = disc_exec_slot(N_INIT_SLOTS + N_LOOP_SLOTS + pi)
        if v is None: return None
        reg[n_args + N_CONSTS + N_INIT_SLOTS + N_LOOP_SLOTS + pi] = v

    ret_i = _argmax_slice(params, ret_off, p)
    return reg[ret_i]


def check_discrete(params: torch.Tensor, examples: list, n_args: int) -> bool:
    """True if discrete_eval matches all examples."""
    for inputs, target in examples:
        result = discrete_eval(params, inputs, n_args)
        if result is None or result != int(target):
            return False
    return True


def perturb_search(params: torch.Tensor, examples: list, n_args: int,
                   depth: int = 1) -> torch.Tensor | None:
    """
    1-hop (depth=1) or 2-hop (depth=2) perturbation search in description space.
    Tries every single-field change to 'params' and returns a corrected params tensor
    if one passes check_discrete, or None if no 1-field fix exists.

    Covers:
      - ret_src (18 values)
      - cond_cmp (6), cond_lhs (18), cond_rhs (18)
      - loop_init[j] (lip values each, 6 slots)
      - per slot: op (6), s1 (18), s2 (18), gate_cmp (6), gate_lhs (18), gate_rhs (18), else_val (18)

    This fixes near-miss predictions like "correct everywhere except ret_src=7 vs 8".
    Total ~1,200 checks at <0.1ms each = ~120ms.
    """
    p   = _pool(n_args)
    lip = _lip(n_args)
    sps = _sps(p)
    HI, LO = 4.0, -4.0

    lo_off  = N_UNIV_SLOTS * sps
    lc_off  = lo_off + N_LOOP_SLOTS * lip
    ret_off = lc_off + N_CMPS + 2*p

    def try_set(idx: int, val: int, width: int) -> bool:
        """Set params[idx+val]=HI (suppressing old HI), check, restore."""
        old_hi = int(params[idx:idx+width].argmax().item())
        if old_hi == val:
            return False  # no change
        orig_old = params[idx + old_hi].item()
        orig_new = params[idx + val].item()
        params[idx + old_hi] = LO
        params[idx + val]    = HI
        ok = check_discrete(params, examples, n_args)
        params[idx + old_hi] = orig_old
        params[idx + val]    = orig_new
        return ok

    candidates = []

    # ret_src
    ret_base = ret_off
    for v in range(p):
        if try_set(ret_base, v, p):
            c = params.clone()
            old = int(params[ret_base:ret_base+p].argmax().item())
            c[ret_base + old] = LO; c[ret_base + v] = HI
            return c

    # loop condition fields
    for idx, width in [(lc_off, N_CMPS), (lc_off + N_CMPS, p), (lc_off + N_CMPS + p, p)]:
        for v in range(width):
            if try_set(idx, v, width):
                c = params.clone()
                old = int(params[idx:idx+width].argmax().item())
                c[idx + old] = LO; c[idx + v] = HI
                return c

    # loop_init fields
    for ls in range(N_LOOP_SLOTS):
        idx = lo_off + ls * lip
        for v in range(lip):
            if try_set(idx, v, lip):
                c = params.clone()
                old = int(params[idx:idx+lip].argmax().item())
                c[idx + old] = LO; c[idx + v] = HI
                return c

    # per-slot fields
    for s in range(N_UNIV_SLOTS):
        base = s * sps
        fields = [
            (base, N_OPS_EXT),                          # op
            (base + N_OPS_EXT, p),                      # s1
            (base + N_OPS_EXT + p, p),                  # s2
            (base + N_OPS_EXT + 2*p, N_CMPS),           # gate_cmp
            (base + N_OPS_EXT + 2*p + N_CMPS, p),       # gate_lhs
            (base + N_OPS_EXT + 2*p + N_CMPS + p, p),   # gate_rhs
            (base + N_OPS_EXT + 2*p + N_CMPS + 2*p, p), # else_val
        ]
        for idx, width in fields:
            for v in range(width):
                if try_set(idx, v, width):
                    c = params.clone()
                    old = int(params[idx:idx+width].argmax().item())
                    c[idx + old] = LO; c[idx + v] = HI
                    return c

    return None


# ─── Warm-start init from UniversalProgramDescription ────────────────────────

def description_to_params(desc: dict, n_args: int) -> torch.Tensor:
    """Convert UniversalProgramDescription dict (Rust serde JSON format) to params tensor."""
    p, lip, sps = _pool(n_args), _lip(n_args), _sps(_pool(n_args))
    n   = n_params_for(n_args)
    HI  =  4.0   # selected logit  (matches Rust: +4.0)
    LO  = -4.0   # suppressed logit (matches Rust: -4.0)
    params = torch.full((n,), LO)

    lo_off  = N_UNIV_SLOTS * sps
    lc_off  = lo_off + N_LOOP_SLOTS * lip
    ret_off = lc_off + N_CMPS + 2*p
    co_off  = ret_off + p

    # Slot fields use Rust JSON names: op, s1, s2, gate_cmp, gate_lhs, gate_rhs, else_val
    for s_idx, slot in enumerate(desc.get("slots", [])[:N_UNIV_SLOTS]):
        off = s_idx * sps
        params[off + min(int(slot.get("op",      5)), N_OPS_EXT-1)] = HI
        params[off + N_OPS_EXT + min(int(slot.get("s1", 0)), p-1)] = HI
        params[off + N_OPS_EXT + p + min(int(slot.get("s2", 0)), p-1)] = HI
        cb = off + N_OPS_EXT + 2*p
        params[cb + min(int(slot.get("gate_cmp", 4)), N_CMPS-1)] = HI
        params[cb + N_CMPS + min(int(slot.get("gate_lhs", 0)), p-1)] = HI
        params[cb + N_CMPS + p + min(int(slot.get("gate_rhs", 0)), p-1)] = HI
        params[cb + N_CMPS + 2*p + min(int(slot.get("else_val", 0)), p-1)] = HI

    # Loop-init fields
    for ls, src in enumerate(desc.get("loop_init", list(range(N_LOOP_SLOTS)))[:N_LOOP_SLOTS]):
        io = lo_off + ls * lip
        params[io + min(int(src), lip-1)] = HI

    # Loop condition — Rust JSON uses cond_cmp, cond_lhs, cond_rhs
    params[lc_off + min(int(desc.get("cond_cmp", 4)), N_CMPS-1)] = HI
    params[lc_off + N_CMPS + min(int(desc.get("cond_lhs", 0)), p-1)] = HI
    params[lc_off + N_CMPS + p + min(int(desc.get("cond_rhs", 0)), p-1)] = HI

    # Return — Rust JSON uses ret_src
    params[ret_off + min(int(desc.get("ret_src", 0)), p-1)] = HI

    # Constants — use actual values from description (not hardcoded defaults)
    consts = desc.get("consts", CONST_VALS)
    for i, v in enumerate(list(consts)[:N_CONSTS]):
        params[co_off + i] = float(v)

    return params


def params_to_description(params: torch.Tensor, n_args: int) -> dict:
    """Convert a params tensor back into a UniversalProgramDescription dict."""
    params = params.detach().cpu()
    p, lip, sps = _pool(n_args), _lip(n_args), _sps(_pool(n_args))

    lo_off  = N_UNIV_SLOTS * sps
    lc_off  = lo_off + N_LOOP_SLOTS * lip
    ret_off = lc_off + N_CMPS + 2*p
    co_off  = ret_off + p

    def choice(offset: int, width: int) -> int:
        return int(params[offset:offset + width].argmax().item())

    slots = []
    for s_idx in range(N_UNIV_SLOTS):
        off = s_idx * sps
        slots.append({
            "op": choice(off, N_OPS_EXT),
            "s1": choice(off + N_OPS_EXT, p),
            "s2": choice(off + N_OPS_EXT + p, p),
            "gate_cmp": choice(off + N_OPS_EXT + 2*p, N_CMPS),
            "gate_lhs": choice(off + N_OPS_EXT + 2*p + N_CMPS, p),
            "gate_rhs": choice(off + N_OPS_EXT + 2*p + N_CMPS + p, p),
            "else_val": choice(off + N_OPS_EXT + 2*p + N_CMPS + 2*p, p),
        })

    loop_init = [choice(lo_off + ls * lip, lip) for ls in range(N_LOOP_SLOTS)]
    consts = [float(params[co_off + i].item()) for i in range(N_CONSTS)]
    return {
        "n_args": n_args,
        "slots": slots,
        "loop_init": loop_init,
        "cond_cmp": choice(lc_off, N_CMPS),
        "cond_lhs": choice(lc_off + N_CMPS, p),
        "cond_rhs": choice(lc_off + N_CMPS + p, p),
        "ret_src": choice(ret_off, p),
        "consts": consts,
    }


# ─── Synthesis loop ───────────────────────────────────────────────────────────

def _get_device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def _run_one(model, examples, inputs_t, targets_t, n_steps, lr, temp_start=2.0, temp_end=0.1):
    """Single gradient run. Uses discrete_eval for verification (matches Rust)."""
    n_args = model.n_args
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_range = temp_start - temp_end
    for step in range(n_steps):
        temp = max(temp_start - t_range * (step / n_steps), temp_end)
        opt.zero_grad()
        loss = model.mse_loss(inputs_t, targets_t, temp)
        if not torch.isfinite(loss):  # NaN/Inf guard — restart this run
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent gradient explosion
        opt.step()
        # NaN recovery: reset to small random if params go NaN
        with torch.no_grad():
            if not torch.isfinite(model.params).all():
                model.params.data = torch.randn_like(model.params) * 0.1
                break
        if step % 10 == 0 or loss.item() < 1e-2:
            with torch.no_grad():
                if check_discrete(model.params, examples, n_args):
                    return True, step + 1
    with torch.no_grad():
        return check_discrete(model.params, examples, n_args), n_steps


def synthesize(examples, n_args: int = 1, n_steps: int = 800,
               lr: float = 0.05, init_params=None, n_restarts: int = 5,
               device=None, warm_lr: float = 0.02,
               warm_temp_start: float = 0.5, warm_temp_end: float = 0.05) -> dict:
    if device is None:
        device = _get_device()

    inputs_t  = torch.tensor([[float(x) for x in inp] for inp, _ in examples],
                              dtype=torch.float32, device=device)
    targets_t = torch.tensor([float(t) for _, t in examples],
                              dtype=torch.float32, device=device)

    n = n_params_for(n_args)

    # ── Warm-start path ────────────────────────────────────────────────────────
    if init_params is not None:
        ip = init_params.to(device)

        # 1. Check immediately — might already be correct (0 gradient steps)
        with torch.no_grad():
            if check_discrete(ip, examples, n_args):
                return {
                    "solved": True,
                    "steps": 0,
                    "loss": 0.0,
                    "method": "warm_exact",
                    "description": params_to_description(ip, n_args),
                }

        # 2. Perturbation search — fix near-miss predictions (1 field wrong)
        with torch.no_grad():
            corrected = perturb_search(ip, examples, n_args)
        if corrected is not None:
            return {
                "solved": True,
                "steps": 0,
                "loss": 0.0,
                "method": "warm_perturb",
                "description": params_to_description(corrected, n_args),
            }

        # 3. Gradient refinement from warm params with cool temperature schedule
        #    (don't blast the good init with temp=2.0)
        model = SoftUniversalProgram(n_args).to(device)
        with torch.no_grad():
            model.params.copy_(ip)
        solved, steps = _run_one(model, examples, inputs_t, targets_t,
                                 n_steps, warm_lr,
                                 temp_start=warm_temp_start, temp_end=warm_temp_end)
        if solved:
            return {
                "solved": True,
                "steps": steps,
                "loss": 0.0,
                "method": "warm_grad",
                "description": params_to_description(model.params, n_args),
            }

    # ── Cold restarts ──────────────────────────────────────────────────────────
    for _ in range(n_restarts):
        model = SoftUniversalProgram(n_args).to(device)
        with torch.no_grad():
            model.params.data[:n - N_CONSTS] = torch.randn(n - N_CONSTS) * 0.5
            for i, v in enumerate(CONST_VALS):
                model.params[model._co_off + i] = float(v)
        solved, steps = _run_one(model, examples, inputs_t, targets_t, n_steps, lr)
        if solved:
            return {
                "solved": True,
                "steps": steps,
                "loss": 0.0,
                "method": "cold",
                "description": params_to_description(model.params, n_args),
            }

    return {"solved": False, "steps": n_steps, "loss": 1.0, "method": "failed"}


def synthesize_warm(examples, n_args, description, n_steps=400, device=None):
    """Warm-start synthesis: perturbation search first, then cool gradient refinement."""
    return synthesize(examples, n_args, n_steps=n_steps,
                      init_params=description_to_params(description, n_args),
                      device=device)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _run_batch(batch_path, out_path, n_steps, warm):
    device = _get_device()
    print(f"Device: {device}", file=sys.stderr)
    with open(batch_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    results = []
    for rec in records:
        name     = rec["name"]
        examples = [(inp, tgt) for inp, tgt in rec["examples"]]
        n_args   = len(examples[0][0])
        desc     = rec.get("description")
        t0 = time.time()
        res = (synthesize_warm(examples, n_args, desc, n_steps, device)
               if (warm and desc) else
               synthesize(examples, n_args, n_steps, device=device))
        dt = time.time() - t0
        tag = "WARM" if (res["solved"] and warm) else ("COLD" if res["solved"] else "FAIL")
        print(f"  {name} → {tag} {res['steps']} steps ({dt:.2f}s)", file=sys.stderr)
        results.append({"name": name, **res})
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    solved = sum(1 for r in results if r["solved"])
    print(f"\nSolved: {solved}/{len(results)}  ({100*solved//max(len(results),1)}%)", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples")
    ap.add_argument("--n-args",   type=int, default=1)
    ap.add_argument("--n-steps",  type=int, default=800)
    ap.add_argument("--batch")
    ap.add_argument("--out")
    ap.add_argument("--warm",     action="store_true")
    ap.add_argument("--description")
    args = ap.parse_args()

    if args.batch:
        _run_batch(args.batch, args.out or "/tmp/soft_synth_out.jsonl", args.n_steps, args.warm)
        return

    if args.examples:
        examples = [([float(x) for x in inp], float(tgt))
                    for inp, tgt in json.loads(args.examples)]
        n_args = len(examples[0][0]) if examples else args.n_args
        desc   = json.loads(args.description) if args.description else None
        t0  = time.time()
        res = (synthesize_warm(examples, n_args, desc, args.n_steps)
               if (args.warm and desc) else
               synthesize(examples, n_args, args.n_steps))
        dt  = time.time() - t0
        print(json.dumps({**res, "time_s": round(dt, 3)}))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
