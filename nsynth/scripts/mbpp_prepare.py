#!/usr/bin/env python3
"""Fetch MBPP + emit the nsynth-runnable, in-domain subset.

MBPP (Mostly Basic Python Problems, 974 tasks) ships `assert func(args) == out`
test cases. This extracts (fn, [(inputs, output)]) from those asserts and keeps
ONLY tasks whose entire I/O is int / list-of-int / bool — the domain the LLM-free
engine can represent today. Emits one JSON line per kept task:
    {"id", "fn", "examples":[{"in":[..], "out":..}]}

Usage:  python3 scripts/mbpp_prepare.py [out.jsonl]   (default /tmp/mbpp_bench.jsonl)
Then:   cargo build --release --bin mbpp_solve_one
        bash scripts/run_mbpp_bench.sh <out.jsonl> 5
"""
import ast, json, sys, urllib.request

URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def classify(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return "intlist"
    return None


def representable(v):
    """True if v maps to a benchmark::Value the interpreter runs: int/bool/str and
    arbitrarily-nested lists/TUPLES thereof (a Python tuple serializes to a JSON
    array, and Mog builds fixed arrays with `[a, b]`, so an int-tuple return like
    (min,max) is solvable). Excludes float and dict (no Value type yet)."""
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(representable(x) for x in v)
    return False


def parse_assert(a):
    """assert FNAME(args) == EXPECTED  (or  assert FNAME(args)  -> True)."""
    try:
        stmt = ast.parse(a.strip()).body[0]
        if not isinstance(stmt, ast.Assert):
            return None
        t = stmt.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            call, exp_node = t.left, t.comparators[0]
        else:
            call, exp_node = t, ast.Constant(True)
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        args = [ast.literal_eval(x) for x in call.args]
        expected = ast.literal_eval(exp_node)
        return call.func.id, args, expected
    except Exception:
        return None


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mbpp_bench.jsonl"
    raw = urllib.request.urlopen(URL, timeout=30).read().decode()
    tasks = [json.loads(l) for l in raw.splitlines() if l.strip()]

    kept = []
    for t in tasks:
        ios, fn, ok = [], None, True
        for a in t["test_list"]:
            p = parse_assert(a)
            if p is None:
                ok = False
                break
            fn, args, exp = p
            if any(classify(x) is None for x in args) or classify(exp) is None:
                ok = False
                break
            ios.append((args, exp))
        if ok and fn and len(ios) >= 3:
            kept.append({"id": t["task_id"], "fn": fn, "examples": [{"in": a, "out": o} for a, o in ios]})

    with open(out_path, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(f"total MBPP tasks: {len(tasks)}")
    print(f"in-domain (int/list-of-int/bool I/O, >=3 tests): {len(kept)} -> {out_path}")


if __name__ == "__main__":
    main()
