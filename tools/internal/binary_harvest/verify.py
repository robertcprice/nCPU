#!/usr/bin/env python3
"""
Verify harvested cache rows: does the stored Python implementation
actually reproduce the stored stdout on the stored stdin?

For the harvest to be training-quality we need the reference Python
`solve(stdin)` functions to be faithful reimplementations of the
coreutils behaviour on the sampled inputs. A row where the reference
disagrees with the binary would teach the model wrong behaviour.

Run before feeding the dataset into distillation:
    python3 tools/binary_harvest/verify.py --cache /tmp/harvest_all.tsv
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
from llm_solution_cache import _load_all  # noqa: E402


def exec_solve(code: str, stdin: str, timeout_s: int = 5) -> Tuple[bool, str]:
    ns: dict = {"__builtins__": __builtins__}
    def _h(signum, frame): raise TimeoutError("solve() timed out")
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(timeout_s)
    try:
        exec(code, ns)
        if "solve" not in ns:
            return (False, "no solve()")
        got = ns["solve"](stdin)
        if not isinstance(got, str):
            return (False, f"non-str return: {type(got).__name__}")
        return (True, got)
    except TimeoutError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"exec: {e!r}"[:200])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.environ["NSYNTH_LLM_CACHE_PATH"] = args.cache
    # Re-import to pick up the new cache path.
    for m in ("llm_solution_cache",):
        if m in sys.modules:
            del sys.modules[m]
    from llm_solution_cache import _load_all as _lla

    rows = _lla()
    binary_rows = [(fp, r) for fp, r in rows.items()
                    if r.get("model", "").startswith("binary:")]
    print(f"[verify] {len(binary_rows)} binary-harvest rows in cache")

    per_tool: dict = {}
    for fp, r in binary_rows:
        tool = r["model"].split(":", 1)[1]
        per_tool.setdefault(tool, {"pass": 0, "fail": 0, "fails": []})
        code = r["code"]
        ex = r.get("examples", [])
        if not ex:
            per_tool[tool]["fail"] += 1
            continue
        stdin = ex[0]["inputs"][0]
        expected = ex[0]["expected"]
        ok, got = exec_solve(code, stdin)
        if ok and got == expected:
            per_tool[tool]["pass"] += 1
        else:
            per_tool[tool]["fail"] += 1
            if len(per_tool[tool]["fails"]) < 2:
                diag = got if ok else got  # got is error msg on not-ok
                per_tool[tool]["fails"].append({
                    "fp": fp[:8], "expected": expected[:100],
                    "got": diag[:100] if isinstance(diag, str) else str(diag)[:100],
                })

    total_pass = total_fail = 0
    for tool, d in sorted(per_tool.items()):
        p, f = d["pass"], d["fail"]
        total_pass += p; total_fail += f
        pct = 100.0 * p / max(p + f, 1)
        print(f"  {tool:<6}  {p:>3}/{p+f:<3}  ({pct:>5.1f}%)")
        if args.verbose and d["fails"]:
            for fail in d["fails"]:
                print(f"    ✗ fp={fail['fp']}  expected={fail['expected']!r}  "
                      f"got={fail['got']!r}")
    total = total_pass + total_fail
    print(f"[verify] TOTAL {total_pass}/{total} "
          f"({100.0 * total_pass / max(total, 1):.1f}%) reference implementations "
          f"match binary output")
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
