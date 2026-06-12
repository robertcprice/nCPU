#!/usr/bin/env python3
"""Phase A eval (stage A4, v1): 105-problem bench with prior net OFF vs ON.

Runs the nsynth Rust bench twice with fresh, isolated memory banks
(solved-cache disabled, fresh bias bank, fresh rejected bank) so the
measurement captures the prior itself, not the persistent caches:

  OFF: NSYNTH_PRIOR_NET=0
  ON:  NSYNTH_PRIOR_NET=1

v1 artifact schema: {"v1": <current run>, "v0": <2026-06-11 baseline>} —
the v0 history is preserved verbatim on every rerun. The v1 run records the
gate config (tau / signal / model) from confidence_calibration.json.

Writes artifacts/prior_net_phase_a.json + artifacts/prior_net_phase_a.md.
Honesty rule: the measured delta is reported whatever it is.

Usage:
  python3 training/prior_net/eval_phase_a.py [--skip-build]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NSYNTH = PROJECT_ROOT / "nsynth"
BINARY = NSYNTH / "target/release/mog_synth"
ARTIFACT_JSON = PROJECT_ROOT / "artifacts/prior_net_phase_a.json"
ARTIFACT_MD = PROJECT_ROOT / "artifacts/prior_net_phase_a.md"
CALIBRATION = PROJECT_ROOT / "training/prior_net/confidence_calibration.json"
PROPOSER_COST = PROJECT_ROOT / "artifacts/prior_net_proposer_cost.json"


def prior_config() -> dict:
    """The gate config the ON run uses (mirrors prior_gen.rs resolution)."""
    cfg = {"tau_env": os.environ.get("NSYNTH_PRIOR_NET_TAU"),
           "signal_env": os.environ.get("NSYNTH_PRIOR_NET_SIGNAL")}
    if CALIBRATION.exists():
        cal = json.loads(CALIBRATION.read_text())
        cfg["calibration"] = {
            "chosen_tau": cal.get("chosen_tau"),
            "chosen_rule": cal.get("chosen_rule"),
            "signal": cal.get("signal"),
            "model": cal.get("model"),
        }
    for name in ("prior_net_v1.pt", "prior_net_v0.pt"):
        p = PROJECT_ROOT / "training/prior_net" / name
        if p.exists():
            cfg["model_resolved"] = str(p)
            break
    if PROPOSER_COST.exists():
        cost = json.loads(PROPOSER_COST.read_text())
        cfg["proposer_cost"] = {
            "v1_server_startup_seconds": cost["v1_server"]["startup_seconds"],
            "v1_server_per_request_ms": cost["v1_server"]["per_request_ms"],
            "v0_oneshot_per_call_seconds": cost["v0_oneshot"]["per_call_seconds"],
        }
    return cfg


def run_bench(label: str, prior_on: bool) -> tuple[list[dict], dict, float]:
    env = dict(os.environ)
    env["NSYNTH_PRIOR_NET"] = "1" if prior_on else "0"
    env["NSYNTH_CACHE_PATH"] = ""  # disable cross-run solved cache
    bias = Path(f"/tmp/prior_eval_bias_{label}.jsonl")
    rej = Path(f"/tmp/prior_eval_rej_{label}.jsonl")
    for p in (bias, rej):
        if p.exists():
            p.unlink()
    env["NSYNTH_BIAS_BANK_PATH"] = str(bias)
    env["NSYNTH_REJECTED_PATH"] = str(rej)

    t0 = time.time()
    proc = subprocess.run(
        [str(BINARY), "--per-problem-json"],
        cwd=str(NSYNTH),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    wall = time.time() - t0

    rows: list[dict] = []
    summary: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if obj.get("summary"):
            summary = obj
        else:
            rows.append(obj)
    return rows, summary, wall


# The 16 problems whose production method was univ_arr_gradient in the
# April-2026 portfolio — i.e. the problems known to live in the universal-
# array program space the prior was trained on. The modern search-teacher
# catalog pre-empts the fallback for all of them in the full pipeline, so
# the direct-fallback eval below bypasses those stages to measure the prior
# against the 26-restart cascade head-to-head.
UNIV_ARR_PROBLEMS = [
    "max_pair_diff_v0", "second_max_v0", "array_range_v0",
    "max_consecutive_sum_v0", "min_consecutive_sum_v0", "max_stock_profit_v0",
    "is_sorted_v0", "longest_increasing_run_v0", "longest_plateau_v0",
    "prefix_max_sum_v0", "max_abs_v0", "min_positive_v0", "count_peaks_v0",
    "alternating_sum_v0", "prefix_sum_k_v0", "is_palindrome_arr_v0",
]

GEN_BIN = NSYNTH / "target/release/gen_prior_data"


def run_fallback_direct(label: str, prior_on: bool) -> tuple[list[dict], float]:
    env = dict(os.environ)
    env["NSYNTH_PRIOR_NET"] = "1" if prior_on else "0"
    env["NSYNTH_CACHE_PATH"] = ""
    bias = Path(f"/tmp/prior_fallback_bias_{label}.jsonl")
    rej = Path(f"/tmp/prior_fallback_rej_{label}.jsonl")
    for p in (bias, rej):
        if p.exists():
            p.unlink()
    env["NSYNTH_BIAS_BANK_PATH"] = str(bias)
    env["NSYNTH_REJECTED_PATH"] = str(rej)
    t0 = time.time()
    proc = subprocess.run(
        [str(GEN_BIN), "--eval-fallback", "--problems", ",".join(UNIV_ARR_PROBLEMS)],
        cwd=str(NSYNTH),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    wall = time.time() - t0
    rows = [
        json.loads(l)
        for l in proc.stdout.splitlines()
        if l.strip().startswith("{") and '"summary"' not in l
    ]
    return rows, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    if not args.skip_build:
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "mog_synth", "--bin", "gen_prior_data"],
            cwd=str(NSYNTH),
            check=True,
        )

    print("[phase_a] running bench with prior OFF ...")
    rows_off, sum_off, wall_off = run_bench("off", prior_on=False)
    print(f"[phase_a] OFF: {sum_off.get('passed')}/{sum_off.get('problem_count')} in {wall_off:.0f}s")
    print("[phase_a] running bench with prior ON ...")
    rows_on, sum_on, wall_on = run_bench("on", prior_on=True)
    print(f"[phase_a] ON:  {sum_on.get('passed')}/{sum_on.get('problem_count')} in {wall_on:.0f}s")

    # Direct fallback measurement: the prior head-to-head against the
    # 26-restart cascade on the 16 universal-array problems, with the
    # search-teacher stages bypassed.
    print("[phase_a] direct fallback eval, prior OFF ...")
    fb_off, fb_wall_off = run_fallback_direct("off", prior_on=False)
    print(f"[phase_a] fallback OFF: {sum(r['solved'] for r in fb_off)}/{len(fb_off)} in {fb_wall_off:.0f}s")
    print("[phase_a] direct fallback eval, prior ON ...")
    fb_on, fb_wall_on = run_fallback_direct("on", prior_on=True)
    print(f"[phase_a] fallback ON:  {sum(r['solved'] for r in fb_on)}/{len(fb_on)} in {fb_wall_on:.0f}s")

    by_name_off = {r["name"]: r for r in rows_off}
    by_name_on = {r["name"]: r for r in rows_on}

    zero_search = [r["name"] for r in rows_on if r["method"] == "prior_net"]
    warm_solves = [r["name"] for r in rows_on if r["method"] == "prior_net_warm"]

    # Per-problem wall deltas (ON - OFF), focused on the universal-array
    # problems where tier-0 actually fires (OFF method == univ_arr_gradient)
    # plus anything the prior touched.
    touched = set(zero_search) | set(warm_solves)
    univ_off = {n for n, r in by_name_off.items() if r["method"] == "univ_arr_gradient"}
    focus = sorted(touched | univ_off)
    per_problem = []
    for name in sorted(by_name_off):
        if name not in by_name_on:
            continue
        off_r, on_r = by_name_off[name], by_name_on[name]
        per_problem.append(
            {
                "name": name,
                "method_off": off_r["method"],
                "method_on": on_r["method"],
                "seconds_off": off_r["seconds"],
                "seconds_on": on_r["seconds"],
                "delta_seconds": round(on_r["seconds"] - off_r["seconds"], 4),
            }
        )
    focus_deltas = [p["delta_seconds"] for p in per_problem if p["name"] in focus]

    fb_off_by = {r["name"]: r for r in fb_off}
    fb_on_by = {r["name"]: r for r in fb_on}
    fb_rows = []
    for name in UNIV_ARR_PROBLEMS:
        o, n = fb_off_by.get(name), fb_on_by.get(name)
        if not o or not n:
            continue
        fb_rows.append(
            {
                "name": name,
                "solved_off": o["solved"],
                "solved_on": n["solved"],
                "method_on": n["method"],
                "seconds_off": o["seconds"],
                "seconds_on": n["seconds"],
                "delta_seconds": round(n["seconds"] - o["seconds"], 3),
            }
        )
    fb_zero = [r["name"] for r in fb_rows if r["method_on"] == "prior_net"]
    fb_warm = [r["name"] for r in fb_rows if r["method_on"] == "prior_net_warm"]
    fb_deltas = [r["delta_seconds"] for r in fb_rows]

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prior_config": prior_config(),
        "direct_fallback": {
            "problems": len(fb_rows),
            "solved": {
                "off": sum(r["solved_off"] for r in fb_rows),
                "on": sum(r["solved_on"] for r in fb_rows),
            },
            "zero_search_solves": len(fb_zero),
            "zero_search_names": fb_zero,
            "warm_solves": len(fb_warm),
            "warm_names": fb_warm,
            "wall_seconds": {
                "off": round(fb_wall_off, 1),
                "on": round(fb_wall_on, 1),
                "delta": round(fb_wall_on - fb_wall_off, 1),
            },
            "per_problem_delta": {
                "mean": round(statistics.mean(fb_deltas), 3) if fb_deltas else None,
                "median": round(statistics.median(fb_deltas), 3) if fb_deltas else None,
            },
            "rows": fb_rows,
        },
        "coverage_both_runs": {
            "off": sum_off.get("passed"),
            "on": sum_on.get("passed"),
            "total": sum_off.get("problem_count"),
        },
        "zero_search_solves": len(zero_search),
        "zero_search_names": zero_search,
        "prior_warm_solves": len(warm_solves),
        "prior_warm_names": warm_solves,
        "wall_seconds": {"off": round(wall_off, 1), "on": round(wall_on, 1),
                          "delta": round(wall_on - wall_off, 1)},
        "bench_wall_seconds": {
            "off": sum_off.get("wall_seconds"),
            "on": sum_on.get("wall_seconds"),
        },
        "focus_problems": focus,
        "focus_wall_delta": {
            "mean": round(statistics.mean(focus_deltas), 3) if focus_deltas else None,
            "median": round(statistics.median(focus_deltas), 3) if focus_deltas else None,
            "sum": round(sum(focus_deltas), 3) if focus_deltas else None,
        },
        "method_counts": {
            "off": sum_off.get("method_counts"),
            "on": sum_on.get("method_counts"),
        },
        "failures": {"off": sum_off.get("failures"), "on": sum_on.get("failures")},
        "per_problem": per_problem,
        "isolation": {
            "NSYNTH_CACHE_PATH": "(disabled)",
            "NSYNTH_BIAS_BANK_PATH": "/tmp/prior_eval_bias_{off,on}.jsonl (fresh)",
            "NSYNTH_REJECTED_PATH": "/tmp/prior_eval_rej_{off,on}.jsonl (fresh)",
        },
    }
    # Versioned artifact: keep the v0 (2026-06-11) baseline verbatim.
    v0_history = None
    if ARTIFACT_JSON.exists():
        old = json.loads(ARTIFACT_JSON.read_text())
        v0_history = old.get("v0", old if "v1" not in old else None)
    ARTIFACT_JSON.write_text(json.dumps({"v1": artifact, "v0": v0_history}, indent=2))
    print(f"[phase_a] wrote {ARTIFACT_JSON}")

    cov = artifact["coverage_both_runs"]
    md = ["# Prior Net Phase A — measured result (stage A4)", ""]
    md.append("## v1 (confidence gate + persistent async server)")
    md.append("")
    pc = artifact["prior_config"]
    cal = pc.get("calibration") or {}
    md.append(f"- Gate: signal `{cal.get('signal')}`, tau {cal.get('chosen_tau')} "
              f"(rule: {cal.get('chosen_rule')}); model `{Path(pc.get('model_resolved', '?')).name}`")
    if "proposer_cost" in pc:
        c = pc["proposer_cost"]
        md.append(f"- Proposer cost: server startup {c['v1_server_startup_seconds']}s "
                  f"(async, off the solve path) + {c['v1_server_per_request_ms']['median']}ms/request "
                  f"median, vs v0 one-shot {c['v0_oneshot_per_call_seconds']['median']}s/problem")
    md.append(f"Generated {artifact['generated_at']}. Fresh isolated banks; solved-cache disabled.")
    md.append("")
    md.append(f"- Coverage OFF: **{cov['off']}/{cov['total']}**, ON: **{cov['on']}/{cov['total']}**")
    md.append(f"- Zero-search solves (prior proposal verified verbatim): **{len(zero_search)}**"
              + (f" — {', '.join(zero_search)}" if zero_search else ""))
    md.append(f"- Warm-refine solves (proposal + ≤120 Adam steps): **{len(warm_solves)}**"
              + (f" — {', '.join(warm_solves)}" if warm_solves else ""))
    md.append(f"- Total bench wall: OFF {wall_off:.0f}s → ON {wall_on:.0f}s "
              f"(delta {wall_on - wall_off:+.0f}s)")
    fd = artifact["focus_wall_delta"]
    if focus:
        md.append(f"- Wall delta on the {len(focus)} universal-array/prior-touched problems: "
                  f"mean {fd['mean']}s, median {fd['median']}s, sum {fd['sum']}s")
    else:
        md.append("- No problem in the full bench reached the universal-array fallback "
                  "(search stages pre-empt it) — see the direct head-to-head below.")
    md.append("")
    md.append("## Direct fallback head-to-head (search stages bypassed)")
    md.append("")
    md.append(f"The {len(fb_rows)} universal-array problems, run straight through "
              "`synthesize_universal_array_fallback` with fresh banks:")
    md.append("")
    md.append(f"- Solved: OFF {sum(r['solved_off'] for r in fb_rows)}/{len(fb_rows)}, "
              f"ON {sum(r['solved_on'] for r in fb_rows)}/{len(fb_rows)}")
    md.append(f"- Zero-search (proposal verified verbatim): **{len(fb_zero)}**"
              + (f" — {', '.join(fb_zero)}" if fb_zero else ""))
    md.append(f"- Warm-refine wins: **{len(fb_warm)}**"
              + (f" — {', '.join(fb_warm)}" if fb_warm else ""))
    md.append(f"- Wall: OFF {fb_wall_off:.0f}s → ON {fb_wall_on:.0f}s "
              f"({fb_wall_on - fb_wall_off:+.0f}s)")
    md.append("")
    md.append("| problem | OFF s | ON s | Δs | ON method |")
    md.append("|---|---|---|---|---|")
    for r in fb_rows:
        md.append(f"| {r['name']} | {r['seconds_off']} | {r['seconds_on']} "
                  f"| {r['delta_seconds']:+} | {r['method_on']} |")
    md.append("")
    md.append("## Method shifts (OFF -> ON)")
    md.append("")
    md.append("| problem | method OFF | method ON | s OFF | s ON | Δs |")
    md.append("|---|---|---|---|---|---|")
    for p in per_problem:
        if p["method_off"] != p["method_on"] or p["name"] in focus:
            md.append(
                f"| {p['name']} | {p['method_off']} | {p['method_on']} "
                f"| {p['seconds_off']} | {p['seconds_on']} | {p['delta_seconds']:+} |"
            )
    md.append("")
    if v0_history:
        v0fb = v0_history.get("direct_fallback", {})
        v0w = v0fb.get("wall_seconds", {})
        v0cov = v0_history.get("coverage_both_runs", {})
        md.append("## v0 history (2026-06-11 — one-shot subprocess, ungated)")
        md.append("")
        md.append(f"- Coverage OFF {v0cov.get('off')}/{v0cov.get('total')}, "
                  f"ON {v0cov.get('on')}/{v0cov.get('total')}; full bench never reached "
                  "the fallback (search stages pre-empt).")
        md.append(f"- Direct fallback head-to-head: {v0fb.get('zero_search_solves')} zero-search wins "
                  f"({', '.join(v0fb.get('zero_search_names', []))}); wall OFF {v0w.get('off')}s -> "
                  f"ON {v0w.get('on')}s ({v0w.get('delta'):+}s). Net negative: each miss paid the "
                  "~7s torch import + model load in a fresh subprocess, plus K=4 warm refines.")
        md.append("")
    ARTIFACT_MD.write_text("\n".join(md) + "\n")
    print(f"[phase_a] wrote {ARTIFACT_MD}")


if __name__ == "__main__":
    main()
