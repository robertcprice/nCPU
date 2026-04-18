"""
eval_metalearner.py — Test all trained meta-learner models on known benchmark problems.

Strategy:
  1. Load bench_known_v2.jsonl → canonical I/O + ground-truth descriptions per fn_name
  2. Map fn_name → benchmark problem name (suffix extraction)
  3. For each model checkpoint:
       a. Run batch inference on canonical I/O examples
       b. Test predicted description with discrete_eval
  4. Report per-model accuracy + which problems each model solves

Usage:
  python3 scripts/eval_metalearner.py
  python3 scripts/eval_metalearner.py --models models/metalearner_1arg_v3.pt
  python3 scripts/eval_metalearner.py --verify-ground-truth   # sanity check first
"""

import sys, json, time, argparse
from pathlib import Path

# Make sure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent))

import torch
from soft_synth import description_to_params, check_discrete, perturb_search, n_params_for, CONST_VALS
from train_metalearner import load_model


BENCH_KNOWN = "data/bench_known_v2.jsonl"
DEFAULT_MODELS = [
    "models/metalearner_1arg.pt",
    "models/metalearner_1arg_50k.pt",
    "models/metalearner_1arg_50k_norm.pt",
    "models/metalearner_1arg_known.pt",
    "models/metalearner_1arg_v2.pt",
    "models/metalearner_1arg_v3.pt",
]


# ── Step 1: Build canonical problem index from bench_known_v2.jsonl ─────────

def load_benchmark_index(path: str):
    """
    Returns dict: bench_name → {"ios": [(inputs, target)...], "description": {...}, "fn_name": str}
    Uses first occurrence of each fn_name for canonical I/O, then maps to bench name.
    """
    # First pass: collect one canonical entry per fn_name
    by_fn: dict = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            fn = rec["fn_name"]
            if fn not in by_fn:
                ios = [([float(x) for x in ex[0]], float(ex[1])) for ex in rec["io_examples"]]
                by_fn[fn] = {
                    "ios":         ios,
                    "description": rec["description"],
                    "fn_name":     fn,
                    "n_args":      rec["description"]["n_args"],
                }

    print(f"Loaded {len(by_fn)} unique fn_names from {path}")
    return by_fn


def extract_bench_name(fn_name: str, known_bench_names: set) -> str | None:
    """
    fn_name format: {program_type}_{bench_name}, e.g. sign_fn_sign_v0 → sign_v0
    Tries all suffixes (shortest prefix removed first) until one matches.
    """
    parts = fn_name.split("_")
    for start in range(1, len(parts)):
        candidate = "_".join(parts[start:])
        if candidate in known_bench_names:
            return candidate
    return None


def build_bench_name_map(by_fn: dict, all_pred_names: set | None = None):
    """
    Map fn_name → bench_name.
    If all_pred_names is given, only try to match those names (avoids false positives).
    Otherwise uses the fn_names themselves as fallback.
    """
    result = {}
    for fn_name, data in by_fn.items():
        if all_pred_names:
            bench = extract_bench_name(fn_name, all_pred_names)
        else:
            # Fallback: try self-referential (some fn_names ARE bench names)
            bench = fn_name
        result[fn_name] = bench or fn_name
    return result


# ── Step 2: Verify ground truth ───────────────────────────────────────────────

def verify_ground_truth(by_fn: dict) -> int:
    """Sanity check: all ground-truth descriptions should pass discrete_eval."""
    ok = 0
    for fn_name, data in by_fn.items():
        desc = data["description"]
        n_args = data["n_args"]
        ios = data["ios"]
        params = description_to_params(desc, n_args)
        if check_discrete(params, ios, n_args):
            ok += 1
        else:
            print(f"  FAIL ground truth: {fn_name}")
    print(f"Ground truth check: {ok}/{len(by_fn)} pass discrete_eval")
    return ok


# ── Step 3: Meta-learner inference ────────────────────────────────────────────

def predict_all(model, problems: list) -> list:
    """
    problems: list of {"fn_name", "bench_name", "ios", "n_args"}
    Returns list of predicted descriptions (one per problem).
    """
    descs = []
    for prob in problems:
        io_pairs = [(inp, tgt) for inp, tgt in prob["ios"][:8]]  # cap at 8
        desc = model.predict_description(io_pairs)
        descs.append(desc)
    return descs


# ── Step 4: Full eval for one model ───────────────────────────────────────────

def eval_model(model_path: str, problems: list, verbose: bool = False) -> dict:
    """
    Returns {"solved": int, "total": int, "details": [{fn_name, bench_name, solved}...]}
    """
    if not Path(model_path).exists():
        return {"error": f"Not found: {model_path}"}

    t0 = time.time()
    model = load_model(model_path)
    model.eval()

    details = []
    solved = 0

    solved_exact   = 0
    solved_perturb = 0

    for prob in problems:
        fn_name   = prob["fn_name"]
        bench     = prob["bench_name"]
        ios       = prob["ios"]
        n_args    = prob["n_args"]
        gt_desc   = prob["description"]

        # Predict
        io_pairs = [(inp, tgt) for inp, tgt in ios[:8]]
        pred_desc = model.predict_description(io_pairs)

        # Test exact prediction
        params = description_to_params(pred_desc, n_args)
        ok_exact = check_discrete(params, ios, n_args)

        # Try perturbation search if exact fails
        ok_perturb = False
        if not ok_exact:
            corrected = perturb_search(params, ios, n_args)
            ok_perturb = corrected is not None

        ok = ok_exact or ok_perturb
        if ok_exact:   solved_exact   += 1
        elif ok_perturb: solved_perturb += 1

        if verbose or not ok:
            tag = "✓" if ok_exact else ("~" if ok_perturb else "✗")
            print(f"  {tag} {fn_name} (→{bench})")

        details.append({"fn_name": fn_name, "bench_name": bench,
                         "solved": ok, "solved_exact": ok_exact,
                         "solved_perturb": ok_perturb,
                         "pred": pred_desc, "gt": gt_desc})

    dt = time.time() - t0
    total = len(problems)
    return {
        "solved":          solved_exact + solved_perturb,
        "solved_exact":    solved_exact,
        "solved_perturb":  solved_perturb,
        "total":           total,
        "pct":             100 * (solved_exact + solved_perturb) // max(total, 1),
        "time_s":          round(dt, 2),
        "details":         details,
    }


# ── Step 5: Ground-truth exact-match score ────────────────────────────────────

def exact_match_rate(details: list) -> float:
    """Fraction of predictions where all discrete fields exactly match ground truth."""
    def flatten(desc):
        slots = desc.get("slots", [])
        fields = []
        for s in slots:
            fields += [s.get("op",0), s.get("s1",0), s.get("s2",0),
                       s.get("gate_cmp",0), s.get("gate_lhs",0), s.get("gate_rhs",0), s.get("else_val",0)]
        fields += desc.get("loop_init", [])
        fields += [desc.get("cond_cmp",0), desc.get("cond_lhs",0), desc.get("cond_rhs",0), desc.get("ret_src",0)]
        return fields

    total = len(details)
    exact = 0
    for d in details:
        if flatten(d["pred"]) == flatten(d["gt"]):
            exact += 1
    return 100 * exact // max(total, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",            nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--bench",             default=BENCH_KNOWN)
    ap.add_argument("--verify-ground-truth", action="store_true")
    ap.add_argument("--verbose",           action="store_true")
    args = ap.parse_args()

    # Change to project root if needed
    root = Path(__file__).parent.parent
    import os; os.chdir(root)

    # Load benchmark index
    by_fn = load_benchmark_index(args.bench)

    # Build problem list (deduplicate by fn_name, keeping canonical I/O)
    # Extract bench names using prediction name set as constraint
    # First get all prediction names from a quick model inference pass (or just use fn_names)
    # We'll try to map using fn_name suffix extraction
    pred_name_hints = {
        "cube_v0", "sign_v0", "digit_sum_v0", "digit_product_v0", "digit_count_v0",
        "is_even_v0", "is_odd_v0", "factorial_v0", "product_1_to_n_v0", "sum_to_n_v0",
        "popcount_v0", "reverse_digits_v0", "leading_digit_v0", "max_digit_v0",
        "lucas_number_v0", "fibonacci_v0", "fib_iter_v0", "square_plus_n_v0",
        "sum_squares_v0", "clamp_positive_v0", "celsius_to_fahrenheit_v0",
        "abs_val_v0", "double_v0", "triple_v0", "add_one_v0",
        "nth_triangle_v0", "positive_or_default_v0",
    }

    fn_to_bench = build_bench_name_map(by_fn, pred_name_hints)
    # Show unmatched (fallback to fn_name itself)
    unmatched = [fn for fn, bn in fn_to_bench.items() if bn == fn]
    if unmatched:
        print(f"Note: {len(unmatched)} fn_names could not be mapped to a known bench name, using fn_name as-is:")
        for fn in unmatched:
            print(f"  {fn}")

    problems = [
        {
            "fn_name":    fn_name,
            "bench_name": fn_to_bench[fn_name],
            "ios":        data["ios"],
            "n_args":     data["n_args"],
            "description": data["description"],
        }
        for fn_name, data in by_fn.items()
    ]

    print(f"\nEvaluating on {len(problems)} benchmark problems\n")

    # Verify ground truth first
    gt_ok = verify_ground_truth(by_fn)
    if gt_ok < len(by_fn):
        print("WARNING: some ground-truth descriptions fail discrete_eval — check bench_known_v2.jsonl")
    print()

    if args.verify_ground_truth:
        return

    # Evaluate each model
    results = {}
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"SKIP {model_path} (not found)")
            continue

        name = Path(model_path).stem
        print(f"── {name} ─────────────────────────────")
        res = eval_model(model_path, problems, verbose=args.verbose)

        if "error" in res:
            print(f"  ERROR: {res['error']}")
            continue

        em = exact_match_rate(res["details"])
        print(f"  Correct exact pred:  {res['solved_exact']}/{res['total']}")
        print(f"  Correct +perturb:    {res['solved']}/{res['total']}  ({res['pct']}%)")
        print(f"  Exact field match:   {em}%")
        print(f"  Time: {res['time_s']}s")

        # Show failures
        failures = [d for d in res["details"] if not d["solved"]]
        if failures:
            print(f"  Failed ({len(failures)}):", ", ".join(d["fn_name"] for d in failures))

        results[name] = res
        print()

    # Summary table
    if len(results) > 1:
        print("═" * 70)
        print(f"{'Model':<35} {'Exact':>6}  {'+Perturb':>9}  {'%':>5}  {'FieldMatch':>11}")
        print("─" * 70)
        for name, res in sorted(results.items(), key=lambda kv: -kv[1]["solved"]):
            em = exact_match_rate(res["details"])
            t  = res["total"]
            print(f"  {name:<33} {res['solved_exact']:>3}/{t}  {res['solved']:>4}/{t}  "
                  f"{res['pct']:>4}%  {em:>9}%")
        print("═" * 70)


if __name__ == "__main__":
    main()
