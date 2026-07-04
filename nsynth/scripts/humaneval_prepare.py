#!/usr/bin/env python3
"""Fetch HumanEval + emit the nsynth-runnable subset in the SAME jsonl format as
mbpp_prepare.py — an UNTUNED generalization run: no ops were written against
this benchmark, the engine runs exactly as it stands.

HumanEval's tests live inside `def check(candidate)` and are more varied than
MBPP's flat asserts. Extracted forms:
    assert candidate(args) == LITERAL
    assert candidate(args)                      -> expected True
    assert candidate(args) == LITERAL, "msg"
    assert abs(candidate(args) - FLOAT) < TOL   -> expected FLOAT
Asserts using variables, loops, or non-literal args are skipped; a task is
ATTEMPTABLE only if >=3 asserts extract AND every value is engine-representable
(same bar as MBPP). Everything else is counted, not silently dropped.

Usage: python3 scripts/humaneval_prepare.py [out.jsonl]  (default /tmp/humaneval_bench.jsonl)
Then:  bash /tmp/bench_resume-style driver or scripts/run_mbpp_bench.sh <out.jsonl> 5
"""
import ast, gzip, io, json, os, sys, urllib.request

URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"


def representable(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(representable(x) for x in v)
    if isinstance(v, dict):
        return all(representable(k) and representable(x) for k, x in v.items())
    return False


def encode(v):
    if isinstance(v, dict):
        return {"__map__": [[encode(k), encode(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    return v


def literal(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def is_candidate_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "candidate"
    )


def extract_examples(test_src):
    """Walk every Assert in the check() body; return [(args, expected)]."""
    try:
        tree = ast.parse(test_src)
    except Exception:
        return None
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        t = node.test
        call, expected = None, None
        # assert candidate(args) == LITERAL
        if (
            isinstance(t, ast.Compare)
            and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Eq)
            and is_candidate_call(t.left)
        ):
            ok, expected = literal(t.comparators[0])
            if not ok:
                continue
            call = t.left
        # assert candidate(args)  -> True
        elif is_candidate_call(t):
            call, expected = t, True
        # assert candidate(args) == True/False via `is`
        elif (
            isinstance(t, ast.Compare)
            and len(t.ops) == 1
            and isinstance(t.ops[0], ast.Is)
            and is_candidate_call(t.left)
        ):
            ok, expected = literal(t.comparators[0])
            if not ok:
                continue
            call = t.left
        # assert abs(candidate(args) - FLOAT) < TOL
        elif (
            isinstance(t, ast.Compare)
            and len(t.ops) == 1
            and isinstance(t.ops[0], (ast.Lt, ast.LtE))
            and isinstance(t.left, ast.Call)
            and isinstance(t.left.func, ast.Name)
            and t.left.func.id == "abs"
            and len(t.left.args) == 1
            and isinstance(t.left.args[0], ast.BinOp)
            and isinstance(t.left.args[0].op, ast.Sub)
            and is_candidate_call(t.left.args[0].left)
        ):
            ok, expected = literal(t.left.args[0].right)
            if not ok:
                continue
            call = t.left.args[0].left
        else:
            continue
        args = []
        good = True
        for a in call.args:
            ok, v = literal(a)
            if not ok:
                good = False
                break
            args.append(v)
        if good:
            out.append((args, expected))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/humaneval_bench.jsonl"
    raw = urllib.request.urlopen(URL, timeout=30).read()
    text = gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode()
    tasks = [json.loads(l) for l in text.splitlines() if l.strip()]

    kept, too_few, unrep = [], 0, 0
    for t in tasks:
        tid = int(t["task_id"].split("/")[1])
        ios = extract_examples(t["test"]) or []
        ios = [
            (a, e)
            for a, e in ios
            if all(representable(x) for x in a) and representable(e)
        ]
        if len(ios) < 3:
            too_few += 1
            continue
        if not ios:
            unrep += 1
            continue
        kept.append({
            "id": tid,
            "fn": t["entry_point"],
            "examples": [{"in": [encode(x) for x in a], "out": encode(e)} for a, e in ios],
        })

    with open(out_path, "w") as f:
        for k in kept:
            f.write(json.dumps(k) + "\n")
    print(f"total HumanEval tasks: {len(tasks)}")
    print(f"ATTEMPTABLE (>=3 literal asserts, representable): {len(kept)} -> {out_path}")
    print(f"OUT-OF-SCOPE: <3 extractable asserts {too_few} (loops/variables in tests)")


if __name__ == "__main__":
    main()
