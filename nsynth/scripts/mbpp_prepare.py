#!/usr/bin/env python3
"""Fetch MBPP + emit the nsynth-runnable, in-domain subset.

MBPP (Mostly Basic Python Problems, 974 tasks) ships `assert func(args) == out`
test cases. This extracts (fn, [(inputs, output)]) from those asserts and keeps
tasks whose entire I/O is int / bool / (arbitrarily-nested) int-list-or-tuple — the
domain the LLM-free engine can represent today (tuples serialize to JSON arrays and
the runtime emits fixed `[a, b]`). Emits one JSON line per kept task:
    {"id", "fn", "examples":[{"in":[..], "out":..}]}

Usage:  python3 scripts/mbpp_prepare.py [out.jsonl]   (default /tmp/mbpp_bench.jsonl)
Then:   cargo build --release --bin mbpp_solve_one
        bash scripts/run_mbpp_bench.sh <out.jsonl> 5
"""
import ast, json, os, sys, urllib.request

URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def _all_int_nested(v):
    """True for int/bool and arbitrarily-nested lists/tuples of them. A Python tuple
    serializes to a JSON array and the runtime emits fixed arrays `[a, b]`, so an
    int-tuple return like `(min, max)` and nested int-lists are all runnable today —
    they were being dropped only because the old flat `intlist` check missed them."""
    if isinstance(v, bool) or isinstance(v, int):
        return True
    if isinstance(v, (list, tuple)):
        return all(_all_int_nested(x) for x in v)
    return False


def classify(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    # Flat int-list, int-tuple, and nested int structures -> one runnable kind.
    if isinstance(v, (list, tuple)) and _all_int_nested(v):
        return "intnest"
    return None


def representable(v):
    """True if v maps to a benchmark::Value the interpreter runs: int/float/bool/str
    and arbitrarily-nested lists/TUPLES thereof (a Python tuple serializes to a JSON
    array, and Mog builds fixed arrays with `[a, b]`, so an int-tuple return like
    (min,max) is solvable). Float is IN scope: `Value::Float` + the float affine
    lane (`solver::search_float`) + the sound float/int output bridge are real.
    Excludes only dict (no `Value::Map` variant), so those tasks are counted as a
    named out-of-scope gap rather than silently dropped."""
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(representable(x) for x in v)  # nested; empty list ok
    if isinstance(v, dict):
        # Value::Map: keys and values must themselves be representable.
        return all(representable(k) and representable(x) for k, x in v.items())
    return False


def encode(v):
    """Wire-encode for the bench jsonl. Dicts become {"__map__": [[k, v], ...]}
    (NOT a JSON object — json.dumps would stringify int keys and lose the key
    type); the solver binary decodes that marker into Value::Map. Everything else
    passes through, recursing into containers."""
    if isinstance(v, dict):
        return {"__map__": [[encode(k), encode(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    return v


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

    # MBPP_DOMAIN=int keeps the historical int/bool/nested-int-only subset (for a
    # clean scalar-domain A/B); default 'all' emits the FULL engine-representable
    # frontier (adds strings + string-nested + mixed tuples). Coverage lives in the
    # engine (Value::Str + the string search tiers are real); this only stops the
    # harness from hiding that reach behind an artificially narrow filter.
    domain = os.environ.get("MBPP_DOMAIN", "all")
    in_scope = (lambda x: classify(x) is not None) if domain == "int" else representable

    kept, reasons = [], {"dict/other": 0, "unparseable": 0}
    domains = {"int": 0, "string": 0, "float": 0, "dict": 0}
    for t in tasks:
        ios, fn, ok, bad = [], None, True, None
        for a in t["test_list"]:
            p = parse_assert(a)
            if p is None:
                ok, bad = False, "unparseable"
                break
            fn, args, exp = p
            for x in list(args) + [exp]:
                if not in_scope(x):
                    ok = False
                    bad = "dict/other"
                    break
            if not ok:
                break
            ios.append((args, exp))
        if ok and fn and len(ios) >= 3:
            kept.append({
                "id": t["task_id"], "fn": fn,
                "examples": [{"in": [encode(x) for x in a], "out": encode(o)} for a, o in ios],
            })
            if any(_has_dict(a) or _has_dict(o) for a, o in ios):
                domains["dict"] += 1
            elif any(_has_float(a) or _has_float(o) for a, o in ios):
                domains["float"] += 1
            elif any(_has_str(a) or _has_str(o) for a, o in ios):
                domains["string"] += 1
            else:
                domains["int"] += 1
        elif bad in reasons:
            reasons[bad] += 1

    with open(out_path, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(f"total MBPP tasks: {len(tasks)}")
    print(f"ATTEMPTABLE (engine-representable, >=3 tests, domain={domain}): {len(kept)} -> {out_path}")
    print(f"  by domain: int/bool/list {domains['int']}  |  string-involving {domains['string']}"
          f"  |  float-involving {domains['float']}  |  dict-involving {domains['dict']}")
    print(f"OUT-OF-SCOPE: other {reasons['dict/other']} (set/None/custom types)  |  "
          f"unparseable {reasons['unparseable']}")


def _has_float(v):
    if isinstance(v, float):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_float(x) for x in v)
    if isinstance(v, dict):
        return any(_has_float(x) for x in v.values())
    return False


def _has_str(v):
    if isinstance(v, str):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_str(x) for x in v)
    return False


def _has_dict(v):
    if isinstance(v, dict):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_dict(x) for x in v)
    return False


if __name__ == "__main__":
    main()
