#!/usr/bin/env python3
"""Measure prior-net proposer cost: v1 persistent server vs v0 one-shot.

Feeds real bench problems (the JSONL dumped by the ignored Rust test
`dump_bench_fallback_requests`) through both proposer modes and reports:

  - v1: server startup (spawn + torch import + checkpoint load, paid once)
        and per-request round-trip latency (the steady-state per-problem cost)
  - v0: full one-shot subprocess cost per problem (spawn + import + load +
        inference), sampled on the first N problems

Also records the gate confidence the server emits per problem — the signal
the calibrated tau acts on — so threshold placement can be sanity-checked
against the actual bench distribution.

Usage:
  python3 training/prior_net/measure_proposer_cost.py \
      [--requests /tmp/prior_bench_requests.jsonl] \
      [--model training/prior_net/prior_net_v0.pt] \
      [--oneshot-samples 4] [--out -]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROPOSE = PROJECT_ROOT / "nsynth/scripts/prior_net/propose.py"


def load_requests(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def measure_server(model: str, requests: list[dict]) -> dict:
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(PROPOSE), "--serve", "--model", model, "--tau", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    ready = json.loads(proc.stdout.readline())
    startup = time.time() - t0
    assert ready.get("ready") is True, f"server failed: {ready}"

    rows = []
    for req in requests:
        payload = {"n_scalar": req["n_scalar"], "examples": req["examples"]}
        t1 = time.time()
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        dt_ms = (time.time() - t1) * 1000.0
        rows.append({
            "name": req.get("name", "?"),
            "ms": round(dt_ms, 1),
            "confidence": resp.get("confidence"),
            "n_proposals": len(resp.get("proposals", [])),
        })
    proc.stdin.close()
    proc.wait(timeout=10)

    lats = [r["ms"] for r in rows]
    return {
        "startup_seconds": round(startup, 2),
        "per_request_ms": {
            "mean": round(statistics.mean(lats), 1),
            "median": round(statistics.median(lats), 1),
            "max": round(max(lats), 1),
        },
        "rows": rows,
    }


def measure_oneshot(model: str, requests: list[dict], n: int) -> dict:
    times = []
    for req in requests[:n]:
        payload = {"n_scalar": req["n_scalar"], "examples": req["examples"]}
        t0 = time.time()
        subprocess.run(
            [sys.executable, str(PROPOSE), "--model", model, "--tau", "0"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        times.append(time.time() - t0)
    return {
        "samples": n,
        "per_call_seconds": {
            "mean": round(statistics.mean(times), 2),
            "median": round(statistics.median(times), 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", default="/tmp/prior_bench_requests.jsonl")
    ap.add_argument("--model", default=str(PROJECT_ROOT / "training/prior_net/prior_net_v0.pt"))
    ap.add_argument("--oneshot-samples", type=int, default=4)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    requests = load_requests(Path(args.requests))
    server = measure_server(args.model, requests)
    oneshot = measure_oneshot(args.model, requests, args.oneshot_samples)

    report = {
        "model": args.model,
        "n_requests": len(requests),
        "v1_server": server,
        "v0_oneshot": oneshot,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    text = json.dumps(report, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
        print(f"v1 startup {server['startup_seconds']}s, "
              f"per-request median {server['per_request_ms']['median']}ms; "
              f"v0 one-shot median {oneshot['per_call_seconds']['median']}s")


if __name__ == "__main__":
    main()
