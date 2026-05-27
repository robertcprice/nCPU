#!/usr/bin/env python3
"""
Measure retrieval hit rate as a function of cache size.

The self-improving story — "the system gets better as users solve more
problems" — needs a quantitative back-up. This tool computes the
retrieval hit rate on a held-out problem set at a sequence of cache
sizes, producing a growth curve.

Methodology:
  1. Load a seed cache (6-col TSV with examples persisted).
  2. Load a held-out problem set (JSONL with `examples` field).
  3. For each cache size N in [10, 30, 100, 300, ...]:
       - Take the first N rows of the cache
       - For each held-out problem: run semantic_lookup, record
         top-sim and whether top-sim ≥ threshold
     Report hit rate + mean top-sim at each size.
  4. Emit a CSV + markdown table.

This tells you:
  - How many verified solves you need before retrieval pays off
  - At what cache size does the curve plateau (diminishing returns)
  - What threshold makes sense for your problem distribution

Usage:
    python3 tools/benchmarks/measure_cache_growth.py \\
        --cache /tmp/retr_v2_corpus.tsv \\
        --holdout tools/benchmarks/humaneval_lite.jsonl \\
        --thresholds 0.5,0.7,0.85 \\
        --sizes 5,10,20,30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_solution_cache import _load_all, _encode  # noqa: E402


def write_subset_cache(rows: Dict[str, dict], n: int, path: Path) -> None:
    """Write first N rows of `rows` back out as a 6-col TSV so the
    semantic_lookup consumer can read them via the standard loader."""
    with path.open("w") as f:
        for i, (fp, row) in enumerate(rows.items()):
            if i >= n:
                break
            examples = row.get("examples") or []
            ex_enc = _encode(json.dumps(examples, sort_keys=True)) if examples else ""
            f.write(
                f"{fp}\t{row['model']}\t{row['success_count']}\t"
                f"{row['last_used_at']}\t{_encode(row['code'])}\t{ex_enc}\n"
            )


def measure(cache_path: Path, holdout_problems: List[dict],
             sizes: List[int], thresholds: List[float]) -> List[dict]:
    """Return a list of row dicts, one per (size, threshold) pair,
    each carrying hit_rate and mean_top_sim."""
    # Reimport semantic_lookup fresh each call so the env-path swap
    # is respected.
    full_rows = _load_all()
    if not full_rows:
        raise SystemExit(f"[growth] cache at {cache_path} is empty")
    results = []

    with tempfile.TemporaryDirectory() as tdir:
        tdir_p = Path(tdir)
        os.environ["NSYNTH_LLM_CACHE_PATH"] = str(tdir_p / "subset.tsv")
        for n in sizes:
            write_subset_cache(full_rows, n, tdir_p / "subset.tsv")

            # Fresh import to pick up new cache path.
            if "semantic_cache" in sys.modules:
                del sys.modules["semantic_cache"]
            if "llm_solution_cache" in sys.modules:
                del sys.modules["llm_solution_cache"]
            from semantic_cache import semantic_lookup  # noqa: E402

            for th in thresholds:
                hits = 0
                top_sims = []
                for p in holdout_problems:
                    ex = p.get("examples") or []
                    if not ex:
                        continue
                    m = semantic_lookup(ex, k=1, min_similarity=0.0)
                    if m:
                        top_sims.append(m[0]["similarity"])
                        if m[0]["similarity"] >= th:
                            hits += 1
                denom = len(holdout_problems) or 1
                mean_sim = sum(top_sims) / len(top_sims) if top_sims else 0.0
                results.append({
                    "size": n,
                    "threshold": th,
                    "hits": hits,
                    "total": denom,
                    "hit_rate": hits / denom,
                    "mean_top_sim": mean_sim,
                })
    return results


def format_markdown(results: List[dict]) -> str:
    sizes = sorted({r["size"] for r in results})
    thresholds = sorted({r["threshold"] for r in results})
    lines = [
        "# Cache-growth retrieval hit rate", "",
        "Held-out hit rate = fraction of held-out problems that retrieve "
        "at least one cached solution with similarity ≥ threshold.",
        "",
        "| size | " + " | ".join(f"sim≥{t:.2f}" for t in thresholds) +
        " | mean top sim |",
        "|--:|" + "|".join(["---:"] * (len(thresholds) + 1)) + "|",
    ]
    for s in sizes:
        row_cells = [f"{s}"]
        for t in thresholds:
            r = next(r for r in results
                     if r["size"] == s and r["threshold"] == t)
            row_cells.append(f"{r['hits']}/{r['total']} ({100*r['hit_rate']:.0f}%)")
        # Mean top sim is threshold-independent; take first.
        mean = next(r["mean_top_sim"] for r in results if r["size"] == s)
        row_cells.append(f"{mean:.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True,
                    help="Path to a 6-col cache TSV (with examples).")
    ap.add_argument("--holdout", required=True,
                    help="Path to a JSONL problem file with 'examples' field.")
    ap.add_argument("--sizes", default="5,10,20,30,50",
                    help="Comma-separated cache sizes to evaluate.")
    ap.add_argument("--thresholds", default="0.5,0.7,0.85",
                    help="Comma-separated similarity thresholds.")
    ap.add_argument("--out-md", default=None,
                    help="Write markdown table to this path.")
    ap.add_argument("--out-csv", default=None,
                    help="Write CSV to this path.")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    os.environ["NSYNTH_LLM_CACHE_PATH"] = args.cache
    # Re-import to pin cache path.
    for m in ("llm_solution_cache", "semantic_cache"):
        if m in sys.modules:
            del sys.modules[m]

    holdout_lines = Path(args.holdout).read_text().splitlines()
    holdout = [json.loads(l) for l in holdout_lines if l.strip()]
    # Filter problems that have examples (required for retrieval).
    holdout = [p for p in holdout if p.get("examples")]
    print(f"[growth] {len(holdout)} held-out problems, "
          f"cache={args.cache}, sizes={sizes}, thresholds={thresholds}")

    results = measure(Path(args.cache), holdout, sizes, thresholds)

    md = format_markdown(results)
    print(md)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md)
        print(f"[growth] wrote {args.out_md}")
    if args.out_csv:
        import csv as _csv
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_csv).open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"[growth] wrote {args.out_csv}")


if __name__ == "__main__":
    main()
