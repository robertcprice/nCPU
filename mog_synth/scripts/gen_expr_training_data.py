#!/usr/bin/env python3
"""
Generate training data for the expr/two_precomp/branch meta-learner.

Two data sources:
  1. Benchmark problems: runs each of the 95 benchmark factories through
     Python-side gradient synthesis (SoftExprProgram, SoftTwoPrecompExprProgram,
     SoftBranchProgram) and captures winning f32 logit vectors.
  2. Synthetic problems: generates random 1–4 arg expressions with random
     constants, extracts I/O examples, synthesizes, and captures winning params.

Output: JSONL to stdout or --out FILE. Each line:
  {"inputs": [[i1,o1],[i2,o2],...], "n_args": N,
   "program_type": "expr"|"two_precomp"|"branch",
   "params": [f32 array], "code": "fn foo(...) -> i64 { ... }"}

Usage:
  python3 scripts/gen_expr_training_data.py --synthetic 2000 --out data/expr_train.jsonl
  python3 scripts/gen_expr_training_data.py --benchmark --out data/expr_bench.jsonl
  python3 scripts/gen_expr_training_data.py --benchmark --synthetic 2000 --out data/expr_combined.jsonl

Requires: the mog_synth release binary for benchmark problem extraction.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ─── Constants (must match synthesis.rs) ─────────────────────────────────────
N_OPS = 5       # +, -, *, /, %
N_OPS_EXT = 6   # +, -, *, /, %, identity
N_CMPS = 6      # <, <=, ==, >=, >, !=
N_CONSTS = 6
CONST_VALS = [0.0, 1.0, -1.0, 2.0, -2.0, 10.0]
N_BRANCHES = 3

# ─── Param size calculations ─────────────────────────────────────────────────

def expr_n_params(n_args: int) -> int:
    """SoftExprProgram param count."""
    ns = n_args + N_CONSTS
    ne = ns + 1
    return 1 + 2 * ns + N_OPS + 2 * ne + N_OPS + N_CONSTS

def two_precomp_n_params(n_args: int) -> int:
    """SoftTwoPrecompExprProgram param count."""
    ns = n_args + N_CONSTS
    ne1 = ns + 1
    ne2 = ns + 2
    return 1 + 2 * ns + N_OPS + 1 + 2 * ne1 + N_OPS + 2 * ne2 + N_OPS + N_CONSTS

def branch_n_params(n_args: int) -> int:
    """SoftBranchProgram param count."""
    ns = n_args + N_CONSTS
    ne = ns + 1
    branch_size = N_CMPS + 4 * ne + 6
    return 1 + 2 * ns + N_OPS + N_BRANCHES * branch_size + 2 * ne + 6 + N_CONSTS


# ─── Soft primitives (match Rust exactly) ────────────────────────────────────

def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def softmax_temp(logits: list[float], temp: float) -> list[float]:
    inv_t = 1.0 / temp
    mx = max(logits)
    exps = [math.exp((x * inv_t) - mx * inv_t) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def soft_read(storage: list[float], weights: list[float]) -> float:
    return sum(s * w for s, w in zip(storage, weights))

def soft_op(a: float, b: float, weights: list[float]) -> float:
    safe_b = 1.0 if abs(b) < 1e-6 else b
    results = [a + b, a - b, a * b, a / safe_b,
               a - math.trunc(a / safe_b) * safe_b]
    return sum(w * r for w, r in zip(weights, results))

def soft_op_ext(a: float, b: float, weights: list[float]) -> float:
    safe_b = 1.0 if abs(b) < 1e-6 else b
    results = [a + b, a - b, a * b, a / safe_b,
               a - math.trunc(a / safe_b) * safe_b, a]
    return sum(w * r for w, r in zip(weights, results))

def soft_cmp(a: float, b: float, weights: list[float], cmp_temp: float) -> float:
    d = a - b
    t = max(0.5, min(2.0, cmp_temp))
    gv = max(t * t * 0.5, 0.125)
    results = [
        sigmoid(-d / t),
        sigmoid(-d / t),
        math.exp(-(d * d) / gv),
        sigmoid(d / t),
        sigmoid(d / t),
        1.0 - math.exp(-(d * d) / gv),
    ]
    return sum(w * r for w, r in zip(weights, results))

def argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])


# ─── SoftExprProgram (pure Python, matches Rust layout) ─────────────────────

class SoftExprProgram:
    def __init__(self, n_args: int, params: Optional[list[float]] = None):
        self.n_args = n_args
        self.ns = n_args + N_CONSTS
        self.ne = self.ns + 1
        n = expr_n_params(n_args)
        if params is not None:
            self.params = list(params)
        else:
            self.params = [0.0] * n
            # Default init matching Rust
            self.params[0] = -4.0  # pre_enable off
            off = 1 + 2 * self.ns + N_OPS
            if n_args > 0:
                self.params[off] = 2.0  # src1 -> arg0
            off2 = off + self.ne
            if n_args > 1:
                self.params[off2 + 1] = 2.0
            else:
                self.params[off2 + n_args] = 2.0
            off3 = off2 + self.ne
            self.params[off3] = 2.0  # op -> +
            # consts
            coff = off3 + N_OPS
            for i, v in enumerate(CONST_VALS):
                self.params[coff + i] = v

    def const_offset(self) -> int:
        return 1 + 2 * self.ns + N_OPS + 2 * self.ne + N_OPS

    def forward(self, inputs: list[float], temp: float) -> float:
        ns, ne = self.ns, self.ne
        p = self.params
        coff = self.const_offset()
        storage = [0.0] * ns
        for i in range(min(self.n_args, len(inputs))):
            storage[i] = inputs[i]
        for i in range(N_CONSTS):
            storage[self.n_args + i] = p[coff + i]

        # Pre-compute
        pre_en = sigmoid(p[0])
        pre_s1 = soft_read(storage, softmax_temp(p[1:1 + ns], temp))
        pre_s2 = soft_read(storage, softmax_temp(p[1 + ns:1 + 2 * ns], temp))
        pre_op_w = softmax_temp(p[1 + 2 * ns:1 + 2 * ns + N_OPS], temp)
        v0 = soft_op(pre_s1, pre_s2, pre_op_w) * pre_en

        ext = storage + [v0]

        # Return expression
        off = 1 + 2 * ns + N_OPS
        s1 = soft_read(ext, softmax_temp(p[off:off + ne], temp))
        s2 = soft_read(ext, softmax_temp(p[off + ne:off + 2 * ne], temp))
        op_w = softmax_temp(p[off + 2 * ne:off + 2 * ne + N_OPS], temp)
        return soft_op(s1, s2, op_w)

    def loss(self, examples: list[tuple[list[float], float]], temp: float) -> float:
        preds = [self.forward(inp, temp) for inp, _ in examples]
        targets = [t for _, t in examples]
        return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)

    def discretize_and_emit(self, fn_name: str, param_names: list[str]) -> str:
        ns, ne = self.ns, self.ne
        p = self.params
        coff = self.const_offset()
        consts = [p[coff + i] for i in range(N_CONSTS)]

        src_names = list(param_names) + [str(int(round(c))) for c in consts] + ["v0"]
        ops = ["+", "-", "*", "/", "%"]

        pre_en = sigmoid(p[0])
        pre_on = pre_en > 0.3
        pre_s1_idx = argmax(p[1:1 + ns])
        pre_s2_idx = argmax(p[1 + ns:1 + 2 * ns])
        pre_op_idx = argmax(p[1 + 2 * ns:1 + 2 * ns + N_OPS])

        off = 1 + 2 * ns + N_OPS
        s1_idx = argmax(p[off:off + ne])
        s2_idx = argmax(p[off + ne:off + 2 * ne])
        op_idx = argmax(p[off + 2 * ne:off + 2 * ne + N_OPS])

        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        body = ""
        if pre_on and pre_s1_idx < len(src_names) - 1 and pre_s2_idx < len(src_names) - 1:
            body += f"    v0: i64 = {src_names[pre_s1_idx]} {ops[pre_op_idx]} {src_names[pre_s2_idx]};\n"
        rn1 = src_names[s1_idx] if s1_idx < len(src_names) else "0"
        rn2 = src_names[s2_idx] if s2_idx < len(src_names) else "0"
        body += f"    return {rn1} {ops[op_idx]} {rn2};\n"
        return f"fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n"


# ─── SoftTwoPrecompExprProgram ───────────────────────────────────────────────

class SoftTwoPrecompExprProgram:
    def __init__(self, n_args: int, params: Optional[list[float]] = None):
        self.n_args = n_args
        self.ns = n_args + N_CONSTS
        self.ne1 = self.ns + 1
        self.ne2 = self.ns + 2
        n = two_precomp_n_params(n_args)
        if params is not None:
            self.params = list(params)
        else:
            self.params = [0.0] * n
            ns, ne1, ne2 = self.ns, self.ne1, self.ne2
            p = self.params
            p[0] = -4.0  # pre1 off
            if n_args > 0:
                p[1] = 2.0
            ps2 = 1 + ns
            if n_args > 1:
                p[ps2 + 1] = 2.0
            else:
                p[ps2 + max(0, min(n_args - 1, ns - 1))] = 2.0
            p[1 + 2 * ns] = 2.0  # pre1_op = +

            p2 = 1 + 2 * ns + N_OPS
            p[p2] = -4.0  # pre2 off
            p[p2 + 1 + ne1 - 1] = 2.0  # pre2_s1 = v0
            p[p2 + 1 + ne1 + n_args] = 2.0  # pre2_s2 = const0
            p[p2 + 1 + 2 * ne1] = 2.0  # pre2_op = +

            roff = p2 + 1 + 2 * ne1 + N_OPS
            if n_args > 0:
                p[roff] = 2.0
            rs2 = roff + ne2
            if n_args > 1:
                p[rs2 + 1] = 2.0
            else:
                p[rs2 + n_args] = 2.0
            p[roff + 2 * ne2] = 2.0  # ret_op = +

            coff = roff + 2 * ne2 + N_OPS
            for i, v in enumerate(CONST_VALS):
                p[coff + i] = v

    def const_offset(self) -> int:
        ns, ne1, ne2 = self.ns, self.ne1, self.ne2
        return 1 + 2 * ns + N_OPS + 1 + 2 * ne1 + N_OPS + 2 * ne2 + N_OPS

    def forward(self, inputs: list[float], temp: float) -> float:
        ns, ne1, ne2 = self.ns, self.ne1, self.ne2
        p = self.params
        coff = self.const_offset()
        consts = [p[coff + i] for i in range(N_CONSTS)]

        storage = [0.0] * ns
        for i in range(min(self.n_args, len(inputs))):
            storage[i] = inputs[i]
        for i, c in enumerate(consts):
            storage[self.n_args + i] = c

        # Pre1
        pre1_en = sigmoid(p[0])
        p1s1 = soft_read(storage, softmax_temp(p[1:1 + ns], temp))
        p1s2 = soft_read(storage, softmax_temp(p[1 + ns:1 + 2 * ns], temp))
        p1op = softmax_temp(p[1 + 2 * ns:1 + 2 * ns + N_OPS], temp)
        v0 = soft_op(p1s1, p1s2, p1op) * pre1_en

        ext1 = storage + [v0]

        # Pre2
        p2_base = 1 + 2 * ns + N_OPS
        pre2_en = sigmoid(p[p2_base])
        p2s1 = soft_read(ext1, softmax_temp(p[p2_base + 1:p2_base + 1 + ne1], temp))
        p2s2 = soft_read(ext1, softmax_temp(p[p2_base + 1 + ne1:p2_base + 1 + 2 * ne1], temp))
        p2op = softmax_temp(p[p2_base + 1 + 2 * ne1:p2_base + 1 + 2 * ne1 + N_OPS], temp)
        v1 = soft_op(p2s1, p2s2, p2op) * pre2_en

        ext2 = ext1 + [v1]

        # Return
        roff = p2_base + 1 + 2 * ne1 + N_OPS
        rs1 = soft_read(ext2, softmax_temp(p[roff:roff + ne2], temp))
        rs2 = soft_read(ext2, softmax_temp(p[roff + ne2:roff + 2 * ne2], temp))
        rop = softmax_temp(p[roff + 2 * ne2:roff + 2 * ne2 + N_OPS], temp)
        return soft_op(rs1, rs2, rop)

    def loss(self, examples: list[tuple[list[float], float]], temp: float) -> float:
        preds = [self.forward(inp, temp) for inp, _ in examples]
        targets = [t for _, t in examples]
        return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)

    def discretize_and_emit(self, fn_name: str, param_names: list[str]) -> str:
        ns, ne1, ne2 = self.ns, self.ne1, self.ne2
        p = self.params
        coff = self.const_offset()
        consts = [p[coff + i] for i in range(N_CONSTS)]

        src_names = list(param_names) + [str(int(round(c))) for c in consts] + ["v0"]
        src_names_ext = list(src_names) + ["v1"]
        ops = ["+", "-", "*", "/", "%"]

        pre1_on = sigmoid(p[0]) > 0.3
        p1s1i = argmax(p[1:1 + ns])
        p1s2i = argmax(p[1 + ns:1 + 2 * ns])
        p1opi = argmax(p[1 + 2 * ns:1 + 2 * ns + N_OPS])

        p2_base = 1 + 2 * ns + N_OPS
        pre2_on = sigmoid(p[p2_base]) > 0.3
        p2s1i = argmax(p[p2_base + 1:p2_base + 1 + ne1])
        p2s2i = argmax(p[p2_base + 1 + ne1:p2_base + 1 + 2 * ne1])
        p2opi = argmax(p[p2_base + 1 + 2 * ne1:p2_base + 1 + 2 * ne1 + N_OPS])

        roff = p2_base + 1 + 2 * ne1 + N_OPS
        rs1i = argmax(p[roff:roff + ne2])
        rs2i = argmax(p[roff + ne2:roff + 2 * ne2])
        ropi = argmax(p[roff + 2 * ne2:roff + 2 * ne2 + N_OPS])

        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        body = ""
        if pre1_on and p1s1i < ns and p1s2i < ns:
            body += f"    v0: i64 = {src_names[p1s1i]} {ops[min(p1opi, N_OPS-1)]} {src_names[p1s2i]};\n"
        if pre2_on and p2s1i < ne1 and p2s2i < ne1:
            n1 = src_names[min(p2s1i, len(src_names)-1)]
            n2 = src_names[min(p2s2i, len(src_names)-1)]
            body += f"    v1: i64 = {n1} {ops[min(p2opi, N_OPS-1)]} {n2};\n"
        rn1 = src_names_ext[rs1i] if rs1i < len(src_names_ext) else "0"
        rn2 = src_names_ext[rs2i] if rs2i < len(src_names_ext) else "0"
        body += f"    return {rn1} {ops[min(ropi, N_OPS-1)]} {rn2};\n"
        return f"fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n"


# ─── SoftBranchProgram ───────────────────────────────────────────────────────

class SoftBranchProgram:
    def __init__(self, n_args: int, params: Optional[list[float]] = None):
        self.n_args = n_args
        self.ns = n_args + N_CONSTS
        self.ne = self.ns + 1
        n = branch_n_params(n_args)
        if params is not None:
            self.params = list(params)
        else:
            self.params = [0.0] * n
            ns, ne = self.ns, self.ne
            p = self.params
            p[0] = -4.0  # pre off
            branch_size = N_CMPS + 4 * ne + 6
            boff = 1 + 2 * ns + N_OPS
            # Branch 0 default
            if n_args > 0:
                p[boff + N_CMPS] = 2.0  # lhs = arg0
                p[boff + N_CMPS + ne + n_args] = 2.0  # rhs = const0
                p[boff] = 2.0  # cmp = >
                p[boff + N_CMPS + 2 * ne] = 2.0  # ret_s1 = arg0
                p[boff + N_CMPS + 4 * ne + 5] = 2.0  # ret_op = identity
            for b in range(1, N_BRANCHES):
                bo = boff + b * branch_size
                p[bo + N_CMPS] = 2.0
                p[bo + N_CMPS + ne] = 2.0
                for k in range(N_CMPS - 1):
                    p[bo + k] = -8.0
                p[bo + N_CMPS - 1] = 8.0
                p[bo + N_CMPS + 2 * ne] = 2.0
                p[bo + N_CMPS + 4 * ne + 5] = 2.0
            doff = boff + N_BRANCHES * branch_size
            if n_args > 1:
                p[doff + 1] = 2.0
            else:
                p[doff + n_args] = 2.0
            p[doff + 2 * ne + 5] = 2.0
            coff = doff + 2 * ne + 6
            for i, v in enumerate(CONST_VALS):
                p[coff + i] = v

    def const_offset(self) -> int:
        ne = self.ne
        branch_size = N_CMPS + 4 * ne + 6
        doff = 1 + 2 * self.ns + N_OPS + N_BRANCHES * branch_size
        return doff + 2 * ne + 6

    def forward(self, inputs: list[float], temp: float) -> float:
        ns, ne = self.ns, self.ne
        p = self.params
        coff = self.const_offset()
        storage = [0.0] * ns
        for i in range(min(self.n_args, len(inputs))):
            storage[i] = inputs[i]
        for i in range(N_CONSTS):
            storage[self.n_args + i] = p[coff + i]

        # Pre-compute
        pre_en = sigmoid(p[0])
        pre_s1 = soft_read(storage, softmax_temp(p[1:1 + ns], temp))
        pre_s2 = soft_read(storage, softmax_temp(p[1 + ns:1 + 2 * ns], temp))
        pre_op_w = softmax_temp(p[1 + 2 * ns:1 + 2 * ns + N_OPS], temp)
        v0 = soft_op(pre_s1, pre_s2, pre_op_w) * pre_en

        ext = storage + [v0]

        branch_size = N_CMPS + 4 * ne + 6
        boff = 1 + 2 * ns + N_OPS

        output = 0.0
        remaining = 1.0

        for b in range(N_BRANCHES):
            bo = boff + b * branch_size
            cmp_w = softmax_temp(p[bo:bo + N_CMPS], temp)
            lhs = soft_read(ext, softmax_temp(p[bo + N_CMPS:bo + N_CMPS + ne], temp))
            rhs = soft_read(ext, softmax_temp(p[bo + N_CMPS + ne:bo + N_CMPS + 2 * ne], temp))
            cond = soft_cmp(lhs, rhs, cmp_w, temp)
            rs1 = soft_read(ext, softmax_temp(p[bo + N_CMPS + 2 * ne:bo + N_CMPS + 3 * ne], temp))
            rs2 = soft_read(ext, softmax_temp(p[bo + N_CMPS + 3 * ne:bo + N_CMPS + 4 * ne], temp))
            rop_w = softmax_temp(p[bo + N_CMPS + 4 * ne:bo + N_CMPS + 4 * ne + 6], temp)
            ret_val = soft_op_ext(rs1, rs2, rop_w)
            fire = cond * remaining
            output += fire * ret_val
            remaining *= 1.0 - cond

        doff = boff + N_BRANCHES * branch_size
        ds1 = soft_read(ext, softmax_temp(p[doff:doff + ne], temp))
        ds2 = soft_read(ext, softmax_temp(p[doff + ne:doff + 2 * ne], temp))
        dop_w = softmax_temp(p[doff + 2 * ne:doff + 2 * ne + 6], temp)
        def_val = soft_op_ext(ds1, ds2, dop_w)
        output += remaining * def_val
        return output

    def loss(self, examples: list[tuple[list[float], float]], temp: float) -> float:
        preds = [self.forward(inp, temp) for inp, _ in examples]
        targets = [t for _, t in examples]
        return sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(preds)

    def discretize_and_emit(self, fn_name: str, param_names: list[str]) -> str:
        ns, ne = self.ns, self.ne
        p = self.params
        coff = self.const_offset()
        consts = [p[coff + i] for i in range(N_CONSTS)]

        src_names = list(param_names) + [str(int(round(c))) for c in consts] + ["v0"]
        ops = ["+", "-", "*", "/", "%", ""]
        cmps_str = [">", "<", ">=", "<=", "==", "!="]

        pre_on = sigmoid(p[0]) > 0.3
        pre_s1_idx = argmax(p[1:1 + ns])
        pre_s2_idx = argmax(p[1 + ns:1 + 2 * ns])
        pre_op_idx = argmax(p[1 + 2 * ns:1 + 2 * ns + N_OPS])

        params_sig = ", ".join(f"{n}: i64" for n in param_names)
        body = ""
        if pre_on and pre_s1_idx < ns and pre_s2_idx < ns:
            body += f"    v0: i64 = {src_names[pre_s1_idx]} {ops[pre_op_idx]} {src_names[pre_s2_idx]};\n"

        branch_size = N_CMPS + 4 * ne + 6
        boff = 1 + 2 * ns + N_OPS
        for b in range(N_BRANCHES):
            bo = boff + b * branch_size
            cmp_idx = argmax(p[bo:bo + N_CMPS])
            lhs_idx = argmax(p[bo + N_CMPS:bo + N_CMPS + ne])
            rhs_idx = argmax(p[bo + N_CMPS + ne:bo + N_CMPS + 2 * ne])
            rs1_idx = argmax(p[bo + N_CMPS + 2 * ne:bo + N_CMPS + 3 * ne])
            rs2_idx = argmax(p[bo + N_CMPS + 3 * ne:bo + N_CMPS + 4 * ne])
            rop_idx = argmax(p[bo + N_CMPS + 4 * ne:bo + N_CMPS + 4 * ne + 6])
            lhs_n = src_names[lhs_idx] if lhs_idx < len(src_names) else "0"
            rhs_n = src_names[rhs_idx] if rhs_idx < len(src_names) else "0"
            rs1_n = src_names[rs1_idx] if rs1_idx < len(src_names) else "0"
            rs2_n = src_names[rs2_idx] if rs2_idx < len(src_names) else "0"
            if rop_idx == 5:
                ret_expr = rs1_n
            else:
                ret_expr = f"{rs1_n} {ops[rop_idx]} {rs2_n}"
            body += f"    if {lhs_n} {cmps_str[cmp_idx]} {rhs_n} {{ return {ret_expr}; }}\n"

        doff = boff + N_BRANCHES * branch_size
        ds1_idx = argmax(p[doff:doff + ne])
        ds2_idx = argmax(p[doff + ne:doff + 2 * ne])
        dop_idx = argmax(p[doff + 2 * ne:doff + 2 * ne + 6])
        dn1 = src_names[ds1_idx] if ds1_idx < len(src_names) else "0"
        dn2 = src_names[ds2_idx] if ds2_idx < len(src_names) else "0"
        if dop_idx == 5:
            body += f"    return {dn1};\n"
        else:
            body += f"    return {dn1} {ops[dop_idx]} {dn2};\n"

        return f"fn {fn_name}({params_sig}) -> i64 {{\n{body}}}\n"


# ─── Discrete evaluation for verification ───────────────────────────────────

def discrete_eval_expr(params: list[float], inputs: list[int], n_args: int) -> Optional[int]:
    """Integer evaluation of SoftExprProgram (argmax + hard ops)."""
    ns = n_args + N_CONSTS
    ne = ns + 1
    coff = 1 + 2 * ns + N_OPS + 2 * ne + N_OPS

    storage = [0] * ns
    for i in range(n_args):
        storage[i] = int(inputs[i])
    for i in range(N_CONSTS):
        storage[n_args + i] = int(round(params[coff + i]))

    # Pre-compute
    pre_en = sigmoid(params[0]) > 0.3
    if pre_en:
        s1_idx = argmax(params[1:1 + ns])
        s2_idx = argmax(params[1 + ns:1 + 2 * ns])
        op_idx = argmax(params[1 + 2 * ns:1 + 2 * ns + N_OPS])
        s1, s2 = storage[s1_idx], storage[s2_idx]
        v0 = _int_op(s1, s2, op_idx)
        if v0 is None:
            return None
    else:
        v0 = 0

    ext = storage + [v0]

    off = 1 + 2 * ns + N_OPS
    s1_idx = argmax(params[off:off + ne])
    s2_idx = argmax(params[off + ne:off + 2 * ne])
    op_idx = argmax(params[off + 2 * ne:off + 2 * ne + N_OPS])
    s1 = ext[s1_idx] if s1_idx < len(ext) else 0
    s2 = ext[s2_idx] if s2_idx < len(ext) else 0
    return _int_op(s1, s2, op_idx)


def discrete_eval_two_precomp(params: list[float], inputs: list[int], n_args: int) -> Optional[int]:
    """Integer evaluation of SoftTwoPrecompExprProgram."""
    ns = n_args + N_CONSTS
    ne1 = ns + 1
    ne2 = ns + 2
    coff = 1 + 2 * ns + N_OPS + 1 + 2 * ne1 + N_OPS + 2 * ne2 + N_OPS

    storage = [0] * ns
    for i in range(n_args):
        storage[i] = int(inputs[i])
    for i in range(N_CONSTS):
        storage[n_args + i] = int(round(params[coff + i]))

    # Pre1
    pre1_on = sigmoid(params[0]) > 0.3
    if pre1_on:
        s1i = argmax(params[1:1 + ns])
        s2i = argmax(params[1 + ns:1 + 2 * ns])
        opi = argmax(params[1 + 2 * ns:1 + 2 * ns + N_OPS])
        v0 = _int_op(storage[s1i], storage[s2i], opi)
        if v0 is None:
            return None
    else:
        v0 = 0

    ext1 = storage + [v0]

    # Pre2
    p2 = 1 + 2 * ns + N_OPS
    pre2_on = sigmoid(params[p2]) > 0.3
    if pre2_on:
        s1i = argmax(params[p2 + 1:p2 + 1 + ne1])
        s2i = argmax(params[p2 + 1 + ne1:p2 + 1 + 2 * ne1])
        opi = argmax(params[p2 + 1 + 2 * ne1:p2 + 1 + 2 * ne1 + N_OPS])
        e1 = ext1[s1i] if s1i < len(ext1) else 0
        e2 = ext1[s2i] if s2i < len(ext1) else 0
        v1 = _int_op(e1, e2, opi)
        if v1 is None:
            return None
    else:
        v1 = 0

    ext2 = ext1 + [v1]

    roff = p2 + 1 + 2 * ne1 + N_OPS
    s1i = argmax(params[roff:roff + ne2])
    s2i = argmax(params[roff + ne2:roff + 2 * ne2])
    opi = argmax(params[roff + 2 * ne2:roff + 2 * ne2 + N_OPS])
    e1 = ext2[s1i] if s1i < len(ext2) else 0
    e2 = ext2[s2i] if s2i < len(ext2) else 0
    return _int_op(e1, e2, opi)


def discrete_eval_branch(params: list[float], inputs: list[int], n_args: int) -> Optional[int]:
    """Integer evaluation of SoftBranchProgram."""
    ns = n_args + N_CONSTS
    ne = ns + 1
    branch_size = N_CMPS + 4 * ne + 6
    doff = 1 + 2 * ns + N_OPS + N_BRANCHES * branch_size
    coff = doff + 2 * ne + 6

    storage = [0] * ns
    for i in range(n_args):
        storage[i] = int(inputs[i])
    for i in range(N_CONSTS):
        storage[n_args + i] = int(round(params[coff + i]))

    # Pre-compute
    pre_on = sigmoid(params[0]) > 0.3
    if pre_on:
        s1i = argmax(params[1:1 + ns])
        s2i = argmax(params[1 + ns:1 + 2 * ns])
        opi = argmax(params[1 + 2 * ns:1 + 2 * ns + N_OPS])
        v0 = _int_op(storage[s1i], storage[s2i], opi)
        if v0 is None:
            return None
    else:
        v0 = 0

    ext = storage + [v0]

    boff = 1 + 2 * ns + N_OPS
    for b in range(N_BRANCHES):
        bo = boff + b * branch_size
        cmp_idx = argmax(params[bo:bo + N_CMPS])
        lhs_idx = argmax(params[bo + N_CMPS:bo + N_CMPS + ne])
        rhs_idx = argmax(params[bo + N_CMPS + ne:bo + N_CMPS + 2 * ne])
        lhs = ext[lhs_idx] if lhs_idx < len(ext) else 0
        rhs = ext[rhs_idx] if rhs_idx < len(ext) else 0
        if _int_cmp(lhs, rhs, cmp_idx):
            rs1i = argmax(params[bo + N_CMPS + 2 * ne:bo + N_CMPS + 3 * ne])
            rs2i = argmax(params[bo + N_CMPS + 3 * ne:bo + N_CMPS + 4 * ne])
            ropi = argmax(params[bo + N_CMPS + 4 * ne:bo + N_CMPS + 4 * ne + 6])
            e1 = ext[rs1i] if rs1i < len(ext) else 0
            e2 = ext[rs2i] if rs2i < len(ext) else 0
            return _int_op_ext(e1, e2, ropi)

    # Default
    ds1i = argmax(params[doff:doff + ne])
    ds2i = argmax(params[doff + ne:doff + 2 * ne])
    dopi = argmax(params[doff + 2 * ne:doff + 2 * ne + 6])
    e1 = ext[ds1i] if ds1i < len(ext) else 0
    e2 = ext[ds2i] if ds2i < len(ext) else 0
    return _int_op_ext(e1, e2, dopi)


def _int_op(a: int, b: int, op_idx: int) -> Optional[int]:
    try:
        if op_idx == 0: return a + b
        if op_idx == 1: return a - b
        if op_idx == 2: return a * b
        if op_idx == 3:
            if b == 0: return None
            return int(a / b) if (a < 0) != (b < 0) and a % b != 0 else a // b
        if op_idx == 4:
            if b == 0: return None
            return a - int(a / b) * b
        return a  # identity
    except (OverflowError, ValueError):
        return None


def _int_op_ext(a: int, b: int, op_idx: int) -> Optional[int]:
    """Op with 6 options (last = identity)."""
    if op_idx == 5:
        return a
    return _int_op(a, b, op_idx)


def _int_cmp(a: int, b: int, cmp_idx: int) -> bool:
    # Rust order: >, <, >=, <=, ==, !=
    if cmp_idx == 0: return a > b
    if cmp_idx == 1: return a < b
    if cmp_idx == 2: return a >= b
    if cmp_idx == 3: return a <= b
    if cmp_idx == 4: return a == b
    if cmp_idx == 5: return a != b
    return False


def check_discrete_expr(params, examples, n_args):
    for inputs, target in examples:
        r = discrete_eval_expr(params, inputs, n_args)
        if r is None or r != int(target):
            return False
    return True


def check_discrete_two_precomp(params, examples, n_args):
    for inputs, target in examples:
        r = discrete_eval_two_precomp(params, inputs, n_args)
        if r is None or r != int(target):
            return False
    return True


def check_discrete_branch(params, examples, n_args):
    for inputs, target in examples:
        r = discrete_eval_branch(params, inputs, n_args)
        if r is None or r != int(target):
            return False
    return True


# ─── Adam optimizer (pure Python, matches Rust) ─────────────────────────────

class Adam:
    def __init__(self, n: int, lr: float = 0.05, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * n
        self.v = [0.0] * n
        self.t = 0

    def step(self, params: list[float], grads: list[float]):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] ** 2
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            params[i] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)


def fd_grad(params: list[float], loss_fn, temp: float, h: float = 0.01) -> list[float]:
    """Finite-difference gradient."""
    base_loss = loss_fn(params, temp)
    grads = [0.0] * len(params)
    for i in range(len(params)):
        old = params[i]
        params[i] = old + h
        loss_plus = loss_fn(params, temp)
        params[i] = old
        grads[i] = (loss_plus - base_loss) / h
    return grads


# ─── Training loop (matches Rust train_program) ─────────────────────────────

def train_program(prog_cls, n_args: int, examples: list[tuple[list[float], float]],
                  check_fn, n_steps: int = 500,
                  init_params: Optional[list[float]] = None) -> Optional[list[float]]:
    """
    Gradient descent on a soft program. Returns winning params or None.
    """
    if init_params is not None:
        prog = prog_cls(n_args, list(init_params))
    else:
        prog = prog_cls(n_args)

    int_examples = [([int(x) for x in inp], int(t)) for inp, t in examples]

    # Check initial params
    if check_fn(prog.params, int_examples, n_args):
        return prog.params

    n = len(prog.params)
    opt = Adam(n, lr=0.05)
    best_loss = float("inf")
    best_params = list(prog.params)
    last_check_loss = float("inf")

    chk1 = n_steps // 4
    chk2 = n_steps // 2
    loss_at_chk1 = float("inf")
    loss_at_chk2 = float("inf")

    def loss_fn(p, t):
        prog.params = p
        return prog.loss(examples, t)

    for step in range(n_steps):
        if step == chk1:
            loss_at_chk1 = best_loss
        if step == chk2:
            loss_at_chk2 = best_loss
        if step == chk2 and best_loss > loss_at_chk1 * 0.98:
            break
        if step > n_steps * 3 // 4 and best_loss > loss_at_chk2 * 0.90:
            break

        temp = max(2.0 * (1.0 - step / n_steps), 0.1)
        prog.params = list(best_params) if step > 0 and random.random() < 0.01 else prog.params
        loss = loss_fn(list(prog.params), temp)

        if math.isnan(loss) or math.isinf(loss):
            break

        if loss < best_loss:
            best_loss = loss
            best_params = list(prog.params)

        should_check = loss < 1.0 or (loss < last_check_loss * 0.9) or (step % 50 == 49)
        if should_check:
            last_check_loss = min(loss, last_check_loss)
            if check_fn(prog.params, int_examples, n_args):
                return list(prog.params)
            if best_loss < loss and check_fn(best_params, int_examples, n_args):
                return list(best_params)

        grads = fd_grad(list(prog.params), loss_fn, temp)
        opt.step(prog.params, grads)

    # Final checks
    if check_fn(prog.params, int_examples, n_args):
        return list(prog.params)
    if check_fn(best_params, int_examples, n_args):
        return list(best_params)
    return None


def pseudo_rand(seed: int) -> float:
    x = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return ((x >> 33) & 0xFFFFFFFF) / 0xFFFFFFFF


# ─── Synthesize one problem across all expr types ────────────────────────────

def synthesize_expr(examples: list[tuple[list[int], int]], n_args: int,
                    n_steps: int = 500, n_restarts: int = 5,
                    verbose: bool = False) -> Optional[dict]:
    """
    Try all expr program types. Returns dict with program_type, params, code
    or None if nothing works.
    """
    float_examples = [([float(x) for x in inp], float(t)) for inp, t in examples]
    default_names = ["a", "b", "c", "d", "e", "f"]
    param_names = [default_names[i] if i < len(default_names) else f"x{i}" for i in range(n_args)]

    for restart in range(n_restarts):
        # 1. SoftExprProgram
        prog = SoftExprProgram(n_args)
        if restart > 0:
            noise = restart * 0.3
            for idx in range(len(prog.params)):
                prog.params[idx] += (pseudo_rand(restart * 7919 + idx) - 0.5) * noise

        result = train_program(SoftExprProgram, n_args, float_examples,
                               check_discrete_expr, n_steps, prog.params)
        if result is not None:
            code = SoftExprProgram(n_args, result).discretize_and_emit("f", param_names)
            return {"program_type": "expr", "params": result, "code": code}

        # 2. SoftTwoPrecompExprProgram
        prog = SoftTwoPrecompExprProgram(n_args)
        if restart > 0:
            noise = restart * 0.3
            for idx in range(len(prog.params)):
                prog.params[idx] += (pseudo_rand(restart * 8831 + idx) - 0.5) * noise

        result = train_program(SoftTwoPrecompExprProgram, n_args, float_examples,
                               check_discrete_two_precomp, n_steps, prog.params)
        if result is not None:
            code = SoftTwoPrecompExprProgram(n_args, result).discretize_and_emit("f", param_names)
            return {"program_type": "two_precomp", "params": result, "code": code}

        # 3. SoftBranchProgram
        prog = SoftBranchProgram(n_args)
        if restart > 0:
            noise = restart * 0.3
            for idx in range(len(prog.params)):
                prog.params[idx] += (pseudo_rand(restart * 9901 + idx) - 0.5) * noise

        result = train_program(SoftBranchProgram, n_args, float_examples,
                               check_discrete_branch, n_steps, prog.params)
        if result is not None:
            code = SoftBranchProgram(n_args, result).discretize_and_emit("f", param_names)
            return {"program_type": "branch", "params": result, "code": code}

    return None


# ─── Benchmark data extraction ───────────────────────────────────────────────

MOG_SYNTH_BIN = str(Path(__file__).parent.parent / "target" / "release" / "mog_synth")


def get_benchmark_problems() -> list[dict]:
    """
    Extract benchmark problems by calling the mog_synth binary.
    Falls back to manually constructing known scalar problems.
    """
    problems = []

    # Known simple scalar benchmark problems with their I/O examples
    # We generate these inline since calling the binary for each is slow
    benchmarks = _known_scalar_benchmarks()
    for name, n_args, examples, holdouts in benchmarks:
        problems.append({
            "name": name,
            "n_args": n_args,
            "examples": examples,
            "holdouts": holdouts,
        })
    return problems


def _known_scalar_benchmarks() -> list[tuple[str, int, list[tuple[list[int], int]], list[tuple[list[int], int]]]]:
    """Inline definitions of scalar benchmark problems (only integer-arg ones)."""
    return [
        ("add_two", 2, [([2, 3], 5), ([10, -4], 6), ([7, 8], 15), ([-3, -2], -5)],
         [([100, -37], 63), ([-12, -8], -20)]),
        ("abs_diff", 2, [([3, 7], 4), ([7, 3], 4), ([0, 5], 5), ([-3, 2], 5)],
         [([-10, 7], 17), ([9, -4], 13)]),
        ("max2", 2, [([2, 3], 3), ([10, -4], 10), ([7, 7], 7), ([-3, -2], -2)],
         [([-3, 9], 9), ([12, 12], 12)]),
        ("clamp", 3, [([5, 0, 10], 5), ([-3, 0, 10], 0), ([15, 0, 10], 10), ([7, 5, 8], 7)],
         [([3, 3, 3], 3), ([100, -10, 50], 50)]),
        ("sign", 1, [([5], 1), ([-3], -1), ([0], 0), ([100], 1)],
         [([-999], -1), ([1], 1)]),
        ("is_even", 1, [([4], 1), ([7], 0), ([0], 1), ([-3], 0)],
         [([100], 1), ([-6], 1)]),
        ("rectangle_area", 2, [([3, 4], 12), ([5, 6], 30), ([1, 1], 1), ([10, 2], 20)],
         [([7, 8], 56), ([0, 5], 0)]),
        ("cube", 1, [([2], 8), ([3], 27), ([1], 1), ([4], 64)],
         [([5], 125), ([0], 0)]),
        ("square_plus_n", 1, [([1], 2), ([2], 6), ([3], 12), ([4], 20)],
         [([5], 30), ([10], 110)]),
        ("bilinear3", 3, [([1, 2, 3], 11), ([0, 0, 0], 0), ([2, 3, 1], 13), ([1, 1, 1], 6)],
         [([3, 2, 1], 14), ([5, 5, 5], 55)]),
        ("scaled_sum", 2, [([3, 4], 10), ([1, 2], 4), ([0, 0], 0), ([5, 1], 11)],
         [([10, 10], 30), ([-1, -2], -4)]),
        ("product_offset", 2, [([3, 4], 9), ([2, 5], 8), ([1, 1], 0), ([4, 3], 8)],
         [([5, 2], 5), ([0, 7], 0)]),
        ("safe_div_or_neg1", 2, [([10, 2], 5), ([7, 0], -1), ([0, 3], 0), ([15, 5], 3)],
         [([100, 10], 10), ([1, 0], -1)]),
        ("positive_or_default", 2, [([5, 0], 5), ([-3, 10], 10), ([0, 7], 7), ([8, -1], 8)],
         [([-5, 100], 100), ([1, 0], 1)]),
        ("celsius_to_fahrenheit", 1, [([0], 32), ([100], 212), ([-40], -40), ([37], 98)],
         [([20], 68), ([-10], 14)]),
        ("sum_to_n", 1, [([1], 1), ([2], 3), ([3], 6), ([5], 15)],
         [([10], 55), ([100], 5050)]),
        ("factorial", 1, [([0], 1), ([1], 1), ([2], 2), ([5], 120)],
         [([3], 6), ([6], 720)]),
        ("fibonacci", 1, [([0], 0), ([1], 1), ([2], 1), ([6], 8)],
         [([10], 55), ([7], 13)]),
        ("power", 2, [([2, 3], 8), ([3, 2], 9), ([5, 0], 1), ([2, 10], 1024)],
         [([10, 3], 1000), ([1, 100], 1)]),
        ("polynomial", 2, [([1, 2], 5), ([2, 3], 11), ([0, 0], 0), ([3, 1], 10)],
         [([4, 5], 21), ([5, 0], 25)]),
        ("min3", 3, [([3, 1, 2], 1), ([5, 5, 5], 5), ([-1, 0, 1], -1), ([10, 3, 7], 3)],
         [([0, -5, 5], -5), ([100, 200, 50], 50)]),
        ("sum_squares", 1, [([1], 1), ([2], 5), ([3], 14), ([4], 30)],
         [([5], 55), ([10], 385)]),
        ("product_1_to_n", 1, [([1], 1), ([2], 2), ([3], 6), ([4], 24)],
         [([5], 120), ([6], 720)]),
        ("count_divisors", 1, [([1], 1), ([6], 4), ([12], 6), ([7], 2)],
         [([28], 6), ([100], 9)]),
        ("max_pair_diff", 2, [([10, 3], 7), ([5, 5], 0), ([1, 8], 7), ([-3, 2], 5)],
         [([100, 1], 99), ([-5, -10], 5)]),
        ("leading_digit", 1, [([123], 1), ([9], 9), ([456], 4), ([7890], 7)],
         [([10], 1), ([99999], 9)]),
        ("popcount", 1, [([0], 0), ([1], 1), ([7], 3), ([255], 8)],
         [([1023], 10), ([128], 1)]),
        ("digital_root", 1, [([0], 0), ([5], 5), ([38], 2), ([627], 6)],
         [([999], 9), ([1], 1)]),
        ("nth_triangle", 1, [([1], 1), ([2], 3), ([3], 6), ([4], 10)],
         [([10], 55), ([100], 5050)]),
        ("is_prime", 1, [([2], 1), ([3], 1), ([4], 0), ([7], 1)],
         [([1], 0), ([11], 1)]),
        ("collatz_steps", 1, [([1], 0), ([2], 1), ([3], 7), ([6], 8)],
         [([7], 16), ([27], 111)]),
        ("digit_sum", 1, [([123], 6), ([0], 0), ([999], 27), ([5], 5)],
         [([1234], 10), ([9999], 36)]),
        ("reverse_digits", 1, [([123], 321), ([400], 4), ([1], 1), ([9876], 6789)],
         [([10], 1), ([12345], 54321)]),
        ("digit_count", 1, [([0], 1), ([5], 1), ([123], 3), ([9999], 4)],
         [([10], 2), ([100000], 6)]),
        ("fib_iter", 1, [([0], 0), ([1], 1), ([2], 1), ([6], 8)],
         [([10], 55), ([7], 13)]),
        ("gcd", 2, [([12, 8], 4), ([7, 3], 1), ([100, 75], 25), ([6, 6], 6)],
         [([48, 18], 6), ([17, 13], 1)]),
        ("sum_of_divisors", 1, [([1], 1), ([6], 12), ([12], 28), ([7], 8)],
         [([28], 56), ([100], 217)]),
        ("is_perfect_square", 1, [([0], 1), ([1], 1), ([4], 1), ([5], 0)],
         [([16], 1), ([15], 0)]),
        ("next_power_of_2", 1, [([1], 1), ([2], 2), ([3], 4), ([5], 8)],
         [([7], 8), ([16], 16)]),
        ("harmonic_sum", 1, [([1], 1), ([2], 1), ([3], 1), ([4], 2)],
         [([6], 2), ([10], 2)]),
        ("triangular_check", 1, [([1], 1), ([3], 1), ([6], 1), ([7], 0)],
         [([10], 1), ([11], 0)]),
        ("euler_totient", 1, [([1], 1), ([2], 1), ([6], 2), ([7], 6)],
         [([12], 4), ([10], 4)]),
        ("lucas_number", 1, [([0], 2), ([1], 1), ([2], 3), ([5], 11)],
         [([10], 123), ([7], 29)]),
        ("digit_product", 1, [([123], 6), ([0], 0), ([999], 729), ([5], 5)],
         [([111], 1), ([234], 24)]),
        ("max_digit", 1, [([123], 3), ([0], 0), ([999], 9), ([517], 7)],
         [([9876], 9), ([111], 1)]),
        ("count_even_digits", 1, [([123], 1), ([0], 1), ([2468], 4), ([135], 0)],
         [([24680], 5), ([13579], 0)]),
        ("sum_odd_digits", 1, [([123], 4), ([0], 0), ([2468], 0), ([13579], 25)],
         [([999], 27), ([24681], 9)]),
        ("count_zeros", 1, [([100], 2), ([0], 1), ([101], 1), ([12345], 0)],
         [([10000], 4), ([1020304], 3)]),
    ]


def generate_benchmark_data(out, n_steps: int = 500, n_restarts: int = 5,
                             verbose: bool = False) -> int:
    """
    Run expr synthesis on each benchmark problem and write solved ones.
    Returns number of records written.
    """
    problems = get_benchmark_problems()
    count = 0
    for prob in problems:
        name = prob["name"]
        n_args = prob["n_args"]
        examples = prob["examples"]
        holdouts = prob.get("holdouts", [])

        if verbose:
            print(f"  [{name}] n_args={n_args}, {len(examples)} examples...", file=sys.stderr, end="", flush=True)

        result = synthesize_expr(examples, n_args, n_steps=n_steps, n_restarts=n_restarts, verbose=verbose)
        if result is not None:
            # Verify on holdouts too
            ok = True
            if holdouts:
                check_fn = {
                    "expr": check_discrete_expr,
                    "two_precomp": check_discrete_two_precomp,
                    "branch": check_discrete_branch,
                }[result["program_type"]]
                for inp, tgt in holdouts:
                    if not check_fn(result["params"], [(inp, tgt)], n_args):
                        ok = False
                        break

            if ok:
                # Build the JSONL record
                io_pairs = [[list(inp), int(tgt)] for inp, tgt in examples]
                record = {
                    "inputs": io_pairs,
                    "n_args": n_args,
                    "program_type": result["program_type"],
                    "params": result["params"],
                    "code": result["code"],
                    "name": name,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                count += 1
                if verbose:
                    print(f" SOLVED ({result['program_type']})", file=sys.stderr, flush=True)
            else:
                if verbose:
                    print(f" SOLVED but failed holdouts", file=sys.stderr, flush=True)
        else:
            if verbose:
                print(f" FAILED", file=sys.stderr, flush=True)

    return count


# ─── Synthetic data generation ───────────────────────────────────────────────

def _random_expr_tree(n_args: int, depth: int, rng: random.Random) -> tuple:
    """
    Generate a random expression tree.
    Returns (tree, is_leaf). Tree is either ('const', val), ('arg', idx),
    or ('op', op_idx, left, right).
    """
    if depth == 0 or (depth < 3 and rng.random() < 0.4):
        # Leaf: arg or const
        if rng.random() < 0.6 and n_args > 0:
            return ("arg", rng.randint(0, n_args - 1))
        else:
            # Use standard consts or small random ints
            c = rng.choice([0, 1, -1, 2, -2, 10, 3, 5, -5, 7, 100])
            return ("const", c)
    else:
        op = rng.randint(0, 4)  # +, -, *, /, %
        # Prefer +-* for simpler data
        if rng.random() < 0.7:
            op = rng.randint(0, 2)
        left = _random_expr_tree(n_args, depth - 1, rng)
        right = _random_expr_tree(n_args, depth - 1, rng)
        return ("op", op, left, right)


def _eval_expr_tree(tree, args: list[int]) -> Optional[int]:
    """Evaluate an expression tree on integer args. Returns None on div-by-zero or overflow."""
    if tree[0] == "const":
        return tree[1]
    elif tree[0] == "arg":
        return args[tree[1]]
    elif tree[0] == "op":
        _, op_idx, left, right = tree
        a = _eval_expr_tree(left, args)
        b = _eval_expr_tree(right, args)
        if a is None or b is None:
            return None
        if abs(a) > 10**9 or abs(b) > 10**9:
            return None  # overflow guard
        return _int_op(a, b, op_idx)
    return None


def _tree_depth(tree) -> int:
    if tree[0] in ("const", "arg"):
        return 0
    return 1 + max(_tree_depth(tree[2]), _tree_depth(tree[3]))


def _tree_to_str(tree, arg_names) -> str:
    if tree[0] == "const":
        return str(tree[1])
    elif tree[0] == "arg":
        return arg_names[tree[1]]
    elif tree[0] == "op":
        _, op_idx, left, right = tree
        ops = ["+", "-", "*", "/", "%"]
        l_str = _tree_to_str(left, arg_names)
        r_str = _tree_to_str(right, arg_names)
        return f"({l_str} {ops[op_idx]} {r_str})"
    return "?"


def _tree_needs_branch(tree) -> bool:
    """Check if tree uses division or modulo (may need branch for safety)."""
    if tree[0] in ("const", "arg"):
        return False
    if tree[0] == "op":
        if tree[1] in (3, 4):  # div, mod
            return True
        return _tree_needs_branch(tree[2]) or _tree_needs_branch(tree[3])
    return False


def generate_synthetic_data(out, n_records: int, seed: int = 42,
                             n_steps: int = 400, verbose: bool = False) -> int:
    """
    Generate synthetic expression problems and solve them.
    Returns number of records written.
    """
    rng = random.Random(seed)
    count = 0
    attempts = 0
    max_attempts = n_records * 10  # limit total attempts

    while count < n_records and attempts < max_attempts:
        attempts += 1
        n_args = rng.randint(1, 4)

        # Generate random expression tree
        max_depth = rng.choice([1, 1, 2, 2, 2, 3])
        tree = _random_expr_tree(n_args, max_depth, rng)

        # Skip very deep trees (unlikely to be solvable by expr programs)
        if _tree_depth(tree) > 3:
            continue

        # Generate random I/O examples
        default_names = ["a", "b", "c", "d"]
        arg_names = default_names[:n_args]
        examples = []
        valid = True

        # Generate 8 examples with varied inputs
        input_sets = set()
        for _ in range(50):  # try up to 50 random inputs
            if len(examples) >= 8:
                break
            args = [rng.randint(-10, 20) for _ in range(n_args)]
            key = tuple(args)
            if key in input_sets:
                continue
            input_sets.add(key)
            result = _eval_expr_tree(tree, args)
            if result is None or abs(result) > 10**9:
                continue
            examples.append((args, result))

        if len(examples) < 4:
            continue

        # Check that the function is non-trivial (not constant)
        outputs = set(t for _, t in examples)
        if len(outputs) == 1:
            continue

        # Split into train + holdout
        train_examples = examples[:min(6, len(examples))]
        holdout_examples = examples[min(6, len(examples)):]

        if verbose and attempts % 100 == 0:
            expr_str = _tree_to_str(tree, arg_names)
            print(f"  attempt {attempts}: n_args={n_args} expr={expr_str}...", file=sys.stderr, flush=True)

        # Try synthesis
        result = synthesize_expr(train_examples, n_args, n_steps=n_steps, n_restarts=3)
        if result is not None:
            # Verify on holdouts
            ok = True
            if holdout_examples:
                check_fn = {
                    "expr": check_discrete_expr,
                    "two_precomp": check_discrete_two_precomp,
                    "branch": check_discrete_branch,
                }[result["program_type"]]
                for inp, tgt in holdout_examples:
                    if not check_fn(result["params"], [(inp, tgt)], n_args):
                        ok = False
                        break

            if ok:
                io_pairs = [[list(inp), int(tgt)] for inp, tgt in train_examples]
                expr_str = _tree_to_str(tree, arg_names)
                record = {
                    "inputs": io_pairs,
                    "n_args": n_args,
                    "program_type": result["program_type"],
                    "params": result["params"],
                    "code": result["code"],
                    "name": f"synth_{n_args}arg_{count}",
                    "expr": expr_str,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                count += 1
                if verbose and count % 10 == 0:
                    print(f"  generated {count}/{n_records} records ({attempts} attempts)", file=sys.stderr, flush=True)

    return count


# ─── Also generate known-solution records (no synthesis needed) ──────────────

def generate_known_solutions(out, verbose: bool = False) -> int:
    """
    Generate training records for problems where we know the exact params.
    These are hand-crafted (no gradient descent needed) and are always correct.
    """
    count = 0
    default_names = ["a", "b", "c", "d"]

    # ── 1-arg expr programs ──
    known_1arg = [
        # a + 1
        ("a_plus_1", 1, [([0], 1), ([1], 2), ([5], 6), ([-3], -2)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=n+1, ret_op=0)),
        # a - 1
        ("a_minus_1", 1, [([0], -1), ([1], 0), ([5], 4), ([-3], -4)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=n+1, ret_op=1)),
        # a * 2
        ("a_times_2", 1, [([0], 0), ([1], 2), ([5], 10), ([-3], -6)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=n+3, ret_op=2)),
        # a * a (need precompute)
        ("a_squared", 1, [([0], 0), ([1], 1), ([3], 9), ([5], 25)],
         lambda n: _make_expr_params(n, pre_on=True, pre_s1=0, pre_s2=0, pre_op=2, ret_s1=n+N_CONSTS, ret_s2=n+1, ret_op=0)),
        # a + 10
        ("a_plus_10", 1, [([0], 10), ([1], 11), ([5], 15), ([-10], 0)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=n+5, ret_op=0)),
        # a * 10
        ("a_times_10", 1, [([0], 0), ([1], 10), ([5], 50), ([-3], -30)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=n+5, ret_op=2)),
    ]

    # ── 2-arg expr programs ──
    known_2arg = [
        # a + b
        ("a_plus_b", 2, [([2, 3], 5), ([10, -4], 6), ([7, 8], 15), ([-3, -2], -5)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=1, ret_op=0)),
        # a - b
        ("a_minus_b", 2, [([5, 3], 2), ([10, 4], 6), ([1, 1], 0), ([-3, 2], -5)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=1, ret_op=1)),
        # a * b
        ("a_times_b", 2, [([2, 3], 6), ([5, 4], 20), ([0, 7], 0), ([-3, 2], -6)],
         lambda n: _make_expr_params(n, pre_on=False, ret_s1=0, ret_s2=1, ret_op=2)),
        # a + b + 1 (v0 = a+b, return v0+1)
        ("a_plus_b_plus_1", 2, [([1, 1], 3), ([0, 0], 1), ([5, 3], 9), ([-1, -1], -1)],
         lambda n: _make_expr_params(n, pre_on=True, pre_s1=0, pre_s2=1, pre_op=0, ret_s1=n+N_CONSTS, ret_s2=n+1, ret_op=0)),
        # a * b + 1
        ("a_times_b_plus_1", 2, [([2, 3], 7), ([1, 1], 2), ([0, 5], 1), ([4, 2], 9)],
         lambda n: _make_expr_params(n, pre_on=True, pre_s1=0, pre_s2=1, pre_op=2, ret_s1=n+N_CONSTS, ret_s2=n+1, ret_op=0)),
    ]

    for name, n_args, examples, params_fn in known_1arg + known_2arg:
        params = params_fn(n_args)
        int_examples = [(inp, tgt) for inp, tgt in examples]
        if check_discrete_expr(params, int_examples, n_args):
            io_pairs = [[list(inp), int(tgt)] for inp, tgt in examples]
            code = SoftExprProgram(n_args, params).discretize_and_emit("f", default_names[:n_args])
            record = {
                "inputs": io_pairs,
                "n_args": n_args,
                "program_type": "expr",
                "params": params,
                "code": code,
                "name": name,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            count += 1
            if verbose:
                print(f"  known: {name} OK", file=sys.stderr, flush=True)
        else:
            if verbose:
                print(f"  known: {name} FAILED verify", file=sys.stderr, flush=True)

    return count


def _make_expr_params(n_args: int, pre_on: bool = False,
                       pre_s1: int = 0, pre_s2: int = 0, pre_op: int = 0,
                       ret_s1: int = 0, ret_s2: int = 0, ret_op: int = 0) -> list[float]:
    """Build a SoftExprProgram params vector with specific hard selections."""
    ns = n_args + N_CONSTS
    ne = ns + 1
    n = expr_n_params(n_args)
    HI = 4.0
    LO = -4.0
    params = [LO] * n

    # pre_enable
    params[0] = 4.0 if pre_on else -4.0
    # pre sources
    params[1 + min(pre_s1, ns - 1)] = HI
    params[1 + ns + min(pre_s2, ns - 1)] = HI
    params[1 + 2 * ns + min(pre_op, N_OPS - 1)] = HI

    # Return expr
    off = 1 + 2 * ns + N_OPS
    params[off + min(ret_s1, ne - 1)] = HI
    params[off + ne + min(ret_s2, ne - 1)] = HI
    params[off + 2 * ne + min(ret_op, N_OPS - 1)] = HI

    # Constants
    coff = off + 2 * ne + N_OPS
    for i, v in enumerate(CONST_VALS):
        params[coff + i] = v

    return params


# ─── Data augmentation: I/O permutation + noise ─────────────────────────────

def augment_record(record: dict, rng: random.Random) -> Optional[dict]:
    """
    Create an augmented version by:
    - Shuffling I/O example order
    - Possibly regenerating with different input values (if we have the code)
    """
    new_rec = dict(record)
    io_pairs = list(record["inputs"])
    rng.shuffle(io_pairs)
    new_rec["inputs"] = io_pairs
    new_rec["name"] = record["name"] + "_aug"
    return new_rec


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate expr meta-learner training data")
    ap.add_argument("--benchmark", action="store_true",
                    help="Solve benchmark problems and capture winning params")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="Generate N synthetic expression problems")
    ap.add_argument("--known", action="store_true",
                    help="Include hand-crafted known-solution records")
    ap.add_argument("--out", type=str, default=None,
                    help="Output JSONL file (default: stdout)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=500,
                    help="Gradient steps per synthesis attempt")
    ap.add_argument("--n-restarts", type=int, default=5,
                    help="Number of random restarts")
    ap.add_argument("--augment", type=int, default=0,
                    help="Number of augmented copies per record")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not args.benchmark and args.synthetic == 0 and not args.known:
        print("Error: specify --benchmark, --synthetic N, or --known (or combine them)",
              file=sys.stderr)
        sys.exit(1)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(args.out, "w")
    else:
        out = sys.stdout

    total = 0
    rng = random.Random(args.seed)
    records_for_augment = []

    if args.known:
        print("Generating known-solution records...", file=sys.stderr, flush=True)
        n = generate_known_solutions(out, verbose=args.verbose)
        total += n
        print(f"  Known solutions: {n} records", file=sys.stderr, flush=True)

    if args.benchmark:
        print("Solving benchmark problems...", file=sys.stderr, flush=True)
        n = generate_benchmark_data(out, n_steps=args.n_steps,
                                     n_restarts=args.n_restarts, verbose=args.verbose)
        total += n
        print(f"  Benchmarks: {n} records", file=sys.stderr, flush=True)

    if args.synthetic > 0:
        print(f"Generating {args.synthetic} synthetic records...", file=sys.stderr, flush=True)
        n = generate_synthetic_data(out, args.synthetic, seed=args.seed,
                                     n_steps=args.n_steps, verbose=args.verbose)
        total += n
        print(f"  Synthetic: {n} records", file=sys.stderr, flush=True)

    # Augmentation pass: re-read the file and add shuffled copies
    if args.augment > 0 and args.out:
        out.close()
        print(f"Augmenting with {args.augment} copies per record...", file=sys.stderr, flush=True)
        with open(args.out) as f:
            records = [json.loads(line) for line in f if line.strip()]
        with open(args.out, "a") as f:
            aug_count = 0
            for rec in records:
                for _ in range(args.augment):
                    aug = augment_record(rec, rng)
                    if aug:
                        f.write(json.dumps(aug) + "\n")
                        aug_count += 1
            total += aug_count
            print(f"  Augmented: {aug_count} records", file=sys.stderr, flush=True)
    elif args.out:
        out.close()

    print(f"Total: {total} records", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
