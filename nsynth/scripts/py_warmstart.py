"""
py_warmstart.py — Python warm-start synthesis for Rust fallback integration.

Called by the Rust solver as a subprocess when gradient synthesis fails.
Runs: meta-learner prediction → perturbation search → gradient refinement.

Inputs (JSON from stdin):
  {"name": "sum_to_n_v0", "examples": [[[1], 1], [[2], 3], [[3], 6]], "n_args": 1}
  {"name": "sum_to_n_v0", "io_examples": [[[1], 1], [[2], 3], [[3], 6]], "n_args": 1}

Output (JSON to stdout):
  {"solved": true, "code": "fn f(a: i64) -> i64 { ... }", "method": "py_warm_perturb"}
  {"solved": false, "error": "..."}

Usage:
  echo '{"name":"sum_to_n_v0","examples":[[[1],1],[[2],3]],"n_args":1}' | python3 scripts/py_warmstart.py
  python3 scripts/py_warmstart.py --model models/metalearner_1arg_v3.pt
"""

import sys, json, time, argparse
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent))

import torch
from soft_synth import (
    description_to_params, check_discrete, perturb_search,
    synthesize, n_params_for, CONST_VALS, N_UNIV_SLOTS, N_LOOP_SLOTS,
    N_CMPS, N_OPS_EXT, _pool, _sps, _lip, params_to_description,
)
from train_metalearner import load_model

DEFAULT_MODEL = str(Path(__file__).parent.parent / "models" / "metalearner_1arg_v5.pt")


def description_to_code(desc: dict, name: str = "f") -> str:
    """
    Convert a UniversalProgramDescription to Rust-like pseudocode.
    This gives the Rust solver something human-readable for the 'code' field.
    """
    n_args = desc.get("n_args", 1)
    p = _pool(n_args)
    lip = _lip(n_args)
    consts = desc.get("consts", [0.0, 1.0, -1.0, 2.0, -2.0, 10.0])
    args = [f"a{i}" for i in range(n_args)]

    POOL_NAMES = (args
                  + [f"c{i}" for i in range(len(consts))]
                  + [f"v{i}" for i in range(3)]
                  + [f"s{i}" for i in range(6)]
                  + [f"p{i}" for i in range(2)])
    POOL_NAMES = POOL_NAMES[:p]

    OPS = ["+", "-", "*", "/", "%", "id"]
    CMPS = ["<", "<=", "==", ">=", ">", "!="]

    def reg(i):
        return POOL_NAMES[i] if i < len(POOL_NAMES) else f"r{i}"

    lines = [f"fn {name}({', '.join(f'{a}: i64' for a in args)}) -> i64 {{"]
    # Consts
    for i, v in enumerate(consts):
        lines.append(f"    let c{i} = {int(v)};")

    # Init slots
    slots = desc.get("slots", [])
    vi = 0
    for si, slot in enumerate(slots[:3]):  # init slots → v0,v1,v2
        op_i = slot.get("op", 5)
        s1, s2 = reg(slot.get("s1", 0)), reg(slot.get("s2", 0))
        gc, gl, gr, ev = slot.get("gate_cmp", 5), slot.get("gate_lhs", 0), slot.get("gate_rhs", 0), slot.get("else_val", 0)
        then = f"{s1}" if op_i == 5 else f"({s1} {OPS[op_i]} {s2})"
        lines.append(f"    let v{si} = if {reg(gl)} {CMPS[gc]} {reg(gr)} {{ {then} }} else {{ {reg(ev)} }};")

    # Loop init
    loop_init = desc.get("loop_init", [1] * 6)
    lip_names = args + [f"c{i}" for i in range(len(consts))] + ["v0", "v1", "v2"]
    for lsi, src in enumerate(loop_init[:6]):
        src_name = lip_names[src] if src < len(lip_names) else f"r{src}"
        lines.append(f"    let mut s{lsi} = {src_name};")

    # Loop
    cc, cl, cr = desc.get("cond_cmp", 5), desc.get("cond_lhs", 0), desc.get("cond_rhs", 0)
    lines.append(f"    while {reg(cl)} {CMPS[cc]} {reg(cr)} {{")
    for lsi, slot in enumerate(slots[3:9]):  # loop slots → s0..s5
        s_idx = lsi
        op_i = slot.get("op", 5)
        s1, s2 = reg(slot.get("s1", 0)), reg(slot.get("s2", 0))
        gc, gl, gr, ev = slot.get("gate_cmp", 5), slot.get("gate_lhs", 0), slot.get("gate_rhs", 0), slot.get("else_val", 0)
        then = f"{s1}" if op_i == 5 else f"({s1} {OPS[op_i]} {s2})"
        lines.append(f"        s{s_idx} = if {reg(gl)} {CMPS[gc]} {reg(gr)} {{ {then} }} else {{ {reg(ev)} }};")
    lines.append("    }")

    # Post slots
    for pi, slot in enumerate(slots[9:11]):  # post slots → p0,p1
        op_i = slot.get("op", 5)
        s1, s2 = reg(slot.get("s1", 0)), reg(slot.get("s2", 0))
        gc, gl, gr, ev = slot.get("gate_cmp", 5), slot.get("gate_lhs", 0), slot.get("gate_rhs", 0), slot.get("else_val", 0)
        then = f"{s1}" if op_i == 5 else f"({s1} {OPS[op_i]} {s2})"
        lines.append(f"    let p{pi} = if {reg(gl)} {CMPS[gc]} {reg(gr)} {{ {then} }} else {{ {reg(ev)} }};")

    lines.append(f"    {reg(desc.get('ret_src', 0))}")
    lines.append("}")
    return "\n".join(lines)


def run(req: dict, model_path: str, n_steps: int = 400, fast: bool = True) -> dict:
    """
    Main synthesis routine. Returns dict with solved/code/method.

    fast=True (default for Rust subprocess): only warm-exact + perturbation (~200ms).
    fast=False: also tries gradient refinement (~30-90s depending on steps/restarts).
    """
    name    = req.get("name", "unknown")
    raw_ex  = req.get("examples")
    if raw_ex is None:
        raw_ex = req.get("io_examples", [])
    n_args  = req.get("n_args", 1)

    if not raw_ex:
        return {"solved": False, "error": "no examples provided"}

    # Normalise examples → list of (inputs_list, int_target)
    examples = []
    for ex in raw_ex:
        if isinstance(ex, (list, tuple)) and len(ex) == 2:
            inp, out = ex
            inp = list(inp) if hasattr(inp, '__iter__') else [inp]
            examples.append((inp, int(out)))
        else:
            return {"solved": False, "error": f"bad example format: {ex}"}

    # Load meta-learner
    if not Path(model_path).exists():
        return {"solved": False, "error": f"model not found: {model_path}"}

    model = load_model(model_path)
    model.eval()

    # Predict description
    io_pairs = [(inp, tgt) for inp, tgt in examples[:8]]
    desc = model.predict_description(io_pairs)

    # Step 1: Exact check (0 gradient steps)
    params = description_to_params(desc, n_args)
    if check_discrete(params, examples, n_args):
        exact_desc = params_to_description(params, n_args)
        code = description_to_code(exact_desc, name)
        return {"solved": True, "code": code, "method": "py_warm_exact", "description": exact_desc}

    # Step 2: Perturbation search (1-field corrections, ~120ms)
    corrected = perturb_search(params, examples, n_args)
    if corrected is not None:
        corrected_desc = params_to_description(corrected, n_args)
        code = description_to_code(corrected_desc, name)
        return {
            "solved": True,
            "code": code,
            "method": "py_warm_perturb",
            "description": corrected_desc,
        }

    if fast:
        return {"solved": False, "method": "py_fast_miss",
                "error": "No 1-field correction found (fast mode)"}

    # Step 3: Gradient refinement from warm params (slow path, for offline use)
    result = synthesize(
        examples, n_args, n_steps=n_steps,
        init_params=params, n_restarts=0,  # warm only, no cold restarts
    )
    if result.get("solved"):
        solved_desc = result.get("description", desc)
        code = description_to_code(solved_desc, name)
        return {
            "solved": True,
            "code": code,
            "method": result.get("method", "py_warm_grad"),
            "description": solved_desc,
        }

    # Step 4: Cold restarts (very slow, last resort)
    result = synthesize(examples, n_args, n_steps=n_steps, n_restarts=3)
    if result.get("solved"):
        solved_desc = result.get("description", desc)
        code = description_to_code(solved_desc, name)
        return {"solved": True, "code": code, "method": "py_cold", "description": solved_desc}

    return {"solved": False, "method": "py_failed",
            "error": f"Python warm-start failed after {n_steps} gradient steps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fast", action="store_true", default=True,
                    help="Fast mode: only exact+perturbation (no gradient). Default on.")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="Slow mode: also try gradient refinement and cold restarts.")
    args = ap.parse_args()

    # Read JSON request from stdin
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"solved": False, "error": "no input on stdin"}))
        sys.exit(1)

    try:
        req = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"solved": False, "error": f"JSON parse error: {e}"}))
        sys.exit(1)

    t0 = time.time()
    result = run(req, args.model, args.n_steps, fast=args.fast)
    result["time_s"] = round(time.time() - t0, 3)

    if args.verbose:
        tag = "SOLVED" if result.get("solved") else "FAILED"
        print(f"[py_warmstart] {req.get('name','?')} → {tag} ({result.get('method','?')}) {result['time_s']}s",
              file=sys.stderr)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
