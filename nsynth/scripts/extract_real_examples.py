#!/usr/bin/env python3
"""COMPREHENSIVE example extractor for synthesis validation.

Pulls (func, [(args, expected)]) I/O from a repo's tests AND doctests, covering
the full value domain the engine can represent: int / float / bool / str / nested
lists / tuples / dict. Emits nsynth-bench JSONL {id, fn, text, examples:[{in,out}]}.

Sources:
  * unittest:  self.assertEqual(A, B) | assertTrue(CALL) | assertFalse(CALL)
  * pytest:    assert CALL == LIT | assert LIT == CALL | assert CALL | assert not CALL
  * doctests:  >>> func(args)  \n  expected      (parsed via the doctest module)

Only functions whose EVERY example is engine-representable, has consistent arg
count, and has >= MINEX distinct examples are emitted. dict -> {"__map__": [[k,v]..]}.

Usage: python3 extract_all.py <repo_dir> [minex=3] > funcs.jsonl
"""
import ast, sys, json, glob, os, doctest
from collections import defaultdict

MINEX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_LIST, MAX_STR = 64, 200

def to_json(v):
    """Runtime value (from literal_eval) -> engine JSON, or None if unrepresentable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, str):
        return v if len(v) <= MAX_STR else None
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_LIST:
            return None
        out = [to_json(x) for x in v]
        return out if all(x is not None for x in out) else None
    if isinstance(v, dict):
        pairs = []
        for k, val in v.items():
            jk, jv = to_json(k), to_json(val)
            if jk is None or jv is None:
                return None
            pairs.append([jk, jv])
        return {"__map__": pairs}
    return None  # None, set, object, etc.

def node_to_py(node):
    """ast literal node -> python value via literal_eval, or raise."""
    return ast.literal_eval(node)

def call_name_args(call):
    """Call node -> (fname, [python arg values]) if it's a bare/attr call with all
    LITERAL args; else None."""
    if not isinstance(call, ast.Call):
        return None
    if isinstance(call.func, ast.Name):
        fname = call.func.id
    elif isinstance(call.func, ast.Attribute):
        fname = call.func.attr
    else:
        return None
    if call.keywords:
        return None
    args = []
    for a in call.args:
        try:
            args.append(node_to_py(a))
        except Exception:
            return None
    return (fname, args)

funcs = defaultdict(list)

def add(fname, args, expected):
    ji = [to_json(a) for a in args]
    jo = to_json(expected)
    if not args or jo is None or any(x is None for x in ji):
        return
    funcs[fname].append((ji, jo))

def from_asserts(tree):
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in ("assertEqual", "assertEquals") and len(n.args) >= 2:
            a, b = n.args[0], n.args[1]
            for expr, other in ((a, b), (b, a)):
                ca = call_name_args(expr)
                if ca:
                    try:
                        exp = node_to_py(other)
                    except Exception:
                        continue
                    add(ca[0], ca[1], exp); break
        elif isinstance(f, ast.Attribute) and f.attr in ("assertTrue", "assertFalse") and n.args:
            ca = call_name_args(n.args[0])
            if ca:
                add(ca[0], ca[1], f.attr == "assertTrue")
    # pytest: assert CALL == LIT  /  assert LIT == CALL  /  assert CALL / assert not CALL
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assert):
            continue
        t = n.test
        if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
            for expr, other in ((t.left, t.comparators[0]), (t.comparators[0], t.left)):
                ca = call_name_args(expr)
                if ca:
                    try:
                        exp = node_to_py(other)
                    except Exception:
                        continue
                    add(ca[0], ca[1], exp); break
        elif isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            ca = call_name_args(t.operand)
            if ca: add(ca[0], ca[1], False)
        else:
            ca = call_name_args(t)
            if ca: add(ca[0], ca[1], True)

def from_doctests(source):
    try:
        tests = doctest.DocTestParser().get_examples(source)
    except Exception:
        return
    for ex in tests:
        src, want = ex.source.strip(), ex.want.strip()
        if not want:
            continue
        try:
            call = ast.parse(src, mode="eval").body
            ca = call_name_args(call)
            exp = ast.literal_eval(want)
        except Exception:
            continue
        if ca:
            add(ca[0], ca[1], exp)

for path in glob.glob(os.path.join(sys.argv[1], "**", "*.py"), recursive=True):
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
        tree = ast.parse(src)
    except Exception:
        continue
    from_asserts(tree)
    from_doctests(src)

def typ(v):
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str): return "str"
    if isinstance(v, dict): return "map"
    if isinstance(v, list): return "list"
    return "?"

def type_consistent(rows, arity):
    """Drop examples whose type at ANY input position — or whose output type —
    differs from the dominant (majority) type for that slot. A single stray
    float/str example (e.g. decimal_to_hexadecimal's [17.0], or is_palindrome's
    ['10101'] among int inputs) otherwise makes an i64 program fail the
    reproduce-every-example gate, blocking a function the engine CAN solve on its
    consistent examples. int and float are kept DISTINCT (an engine i64 program
    can't run on a float arg), as are bool and int."""
    from collections import Counter
    pos_types = [Counter(typ(a[p]) for a, _ in rows).most_common(1)[0][0]
                 for p in range(arity)]
    out_type = Counter(typ(e) for _, e in rows).most_common(1)[0][0]
    return [(a, e) for a, e in rows
            if typ(e) == out_type
            and all(typ(a[p]) == pos_types[p] for p in range(arity))]

i = 0
bytype = defaultdict(int)
for fname, ios in sorted(funcs.items()):
    seen, rows = set(), []
    for args, exp in ios:
        key = json.dumps([args, exp], sort_keys=True)
        if key not in seen:
            seen.add(key); rows.append((args, exp))
    if len(rows) < MINEX:
        continue
    arity = len(rows[0][0])
    rows = [(a, e) for a, e in rows if len(a) == arity]  # consistent arity
    if len(rows) < MINEX:
        continue
    rows = type_consistent(rows, arity)  # drop stray-type (contaminant) examples
    if len(rows) < MINEX:
        continue
    i += 1
    bytype[typ(rows[0][1])] += 1
    print(json.dumps({
        "id": i, "fn": fname, "text": f"compute {fname}",
        "examples": [{"in": a, "out": e} for a, e in rows],
    }))
sys.stderr.write(f"emitted {i} functions; by output type: {dict(bytype)}\n")
