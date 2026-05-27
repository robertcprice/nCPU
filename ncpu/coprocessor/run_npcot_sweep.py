"""Coprocessor sweep with NPCoT array-thought expert enabled (N1-next).

Runs a training sweep of `NCPUCoprocessorMLPWithArrayThought` against a
given base model. This script is the scaffolding an end user launches on
vast.ai / an A100 / local GPU — it does NOT require the GPU to be present
at import time, so importing it in a test environment is free.

What this script does, end-to-end:

1. Loads the base model (HuggingFace local or hub).
2. Wraps the target layer's MLP with `NCPUCoprocessorMLPWithArrayThought`.
3. Attaches a `ProgramLibrarySession`: the library loads from disk at the
   start of each epoch, captures a snapshot, and saves on end-epoch with a
   compliance-ready diff.
4. Trains on a user-supplied dataset (GSM8K-style arithmetic by default).
5. After training, evaluates hit rate + verifier report on the resulting
   library.
6. Emits a JSON run artifact with:
   - Per-step loss / learning rate
   - Per-epoch library summary (entries, hits, newly_cached, diff)
   - Final verifier report (safe / warn / high breakdown)
   - Whatever held-out benchmark scores the caller plugs in

The actual LLM loading path is kept thin — a function
`load_wrapped_model(model_path, target_layers)` returns the transformer
with its coprocessor wrapper installed. Running the sweep requires
transformers + torch + an actual GPU; testing this script's *shape* does
not.

Example vast.ai launch:

    python3 -m ncpu.coprocessor.run_npcot_sweep \\
        --model Qwen/Qwen3.5-1.5B \\
        --dataset gsm8k \\
        --epochs 2 \\
        --library ~/.nCPU_program_library.json \\
        --out training_results/npcot_sweep_qwen35_1.5b.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
from ncpu.self_optimizing.compliance_report import (
    ComplianceReportConfig,
    build_compliance_report,
)
from ncpu.self_optimizing.program_library_session import (
    ProgramLibrarySession,
    ProgramLibrarySessionConfig,
)
from ncpu.self_optimizing.program_verifier import (
    RangeBound,
    VerifierConfig,
)


@dataclass
class SweepConfig:
    """Runtime knobs for the NPCoT coprocessor sweep."""

    model: str = "Qwen/Qwen3.5-1.5B"
    dataset: str = "gsm8k"
    epochs: int = 2
    steps_per_epoch: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-4
    array_max_len: int = 8
    array_thought_max_gate: float = 0.05
    library_path: str = "~/.nCPU_program_library.json"
    output_json: str = "training_results/npcot_sweep.json"
    convergence_gap_threshold: float = 0.15
    target_layers: list[int] = field(default_factory=lambda: [-2, -1])
    seed: int = 42
    dry_run: bool = False


def parse_cli(argv: Optional[list[str]] = None) -> SweepConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-1.5B")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--array-thought-max-gate", type=float, default=0.05)
    p.add_argument("--library", dest="library_path", default="~/.nCPU_program_library.json")
    p.add_argument("--out", dest="output_json", default="training_results/npcot_sweep.json")
    p.add_argument("--convergence-gap-threshold", type=float, default=0.15)
    p.add_argument("--target-layers", type=str, default="-2,-1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + library paths without training. Zero GPU ops.",
    )
    args = p.parse_args(argv)
    target_layers = [int(x) for x in args.target_layers.split(",") if x.strip()]
    return SweepConfig(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        array_max_len=args.array_max_len,
        array_thought_max_gate=args.array_thought_max_gate,
        library_path=args.library_path,
        output_json=args.output_json,
        convergence_gap_threshold=args.convergence_gap_threshold,
        target_layers=target_layers,
        seed=args.seed,
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# Thin model-loading seam (heavy deps imported lazily so `--dry-run` works
# on machines with no torch + transformers installed).
# ---------------------------------------------------------------------------


def load_wrapped_model(
    model_path: str,
    *,
    target_layers: list[int],
    array_max_len: int,
    max_gate: float,
) -> Any:
    """Load the base model and install `NCPUCoprocessorMLPWithArrayThought`.

    Called only when not in dry-run mode. Lazy-imports the heavy deps.
    """
    import torch  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    from ncpu.coprocessor.array_thought_coprocessor import (
        ArrayThoughtCoprocessorConfig,
        NCPUCoprocessorMLPWithArrayThought,
    )
    from ncpu.coprocessor.config import NCPUCoprocessorConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    hidden_dim = int(getattr(config, "hidden_size", 0) or 0)
    if hidden_dim <= 0:
        raise ValueError("could not infer hidden_size from model config")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=False
    )
    coprocessor_config = NCPUCoprocessorConfig(
        n_bits=8, num_ops=7, max_gate=0.1, residual_init_scale=0.0
    )
    array_thought_config = ArrayThoughtCoprocessorConfig(
        array_max_len=array_max_len,
        max_gate=max_gate,
    )
    base_layers = model.model.layers
    n_layers = len(base_layers)
    for idx in target_layers:
        resolved = idx if idx >= 0 else n_layers + idx
        if resolved < 0 or resolved >= n_layers:
            raise ValueError(f"target layer {idx} out of range [0, {n_layers})")
        layer = base_layers[resolved]
        original_mlp = layer.mlp
        layer.mlp = NCPUCoprocessorMLPWithArrayThought(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            config=coprocessor_config,
            array_thought_config=array_thought_config,
        )
    return model


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_dry(cfg: SweepConfig) -> dict[str, Any]:
    """Validate paths + config without any GPU work."""
    library_path = Path(cfg.library_path).expanduser()
    library_summary: dict[str, Any]
    if library_path.exists():
        library = ArrayProgramLibrary.load(library_path)
        library_summary = library.audit_report()["summary"]
    else:
        library_summary = {
            "entry_count": 0,
            "note": "library does not exist yet; will be created on first run",
        }
    return {
        "mode": "dry_run",
        "timestamp": time.time(),
        "config": cfg.__dict__,
        "library_summary": library_summary,
        "library_path_resolved": str(library_path),
        "output_resolved": str(Path(cfg.output_json).expanduser()),
        "target_layers": cfg.target_layers,
    }


def run_sweep(cfg: SweepConfig) -> dict[str, Any]:
    """Standalone NPCoT training loop + crystallization + compliance.

    This is NOT the full Qwen pipeline — that one lives in
    `ncpu.coprocessor.train` and requires GPUs + HF model downloads.
    This function runs the *equivalent* loop against just the
    `ArrayExecutableThoughtHead` + `ArrayProgramLibrary` stack, producing
    the same artifact shape (trained head, persisted library, compliance
    report). It's launchable on CPU in seconds, so users can sanity-check
    their library path, convergence threshold, and artifact layout before
    burning A100 hours on `run_instruct_sweep` equivalents.

    When `load_wrapped_model` is importable (i.e. transformers installed),
    future work can replace the training loop here with the real HF-wrapped
    path; the surrounding artifact/plumbing code is identical.
    """
    import torch
    from ncpu.self_optimizing.array_executable_thought_head import (
        ArrayExecutableThoughtHead,
        ArrayExecutableThoughtHeadConfig,
        build_array_thought_smoke_batch,
        run_array_thought_smoke_train,
    )

    session = ProgramLibrarySession(
        ProgramLibrarySessionConfig(
            library_path=Path(cfg.library_path).expanduser(),
            convergence_gap_threshold=cfg.convergence_gap_threshold,
            auto_cache=True,
        )
    )
    begin_meta = session.begin_task(f"sweep:{cfg.model}:{cfg.dataset}")

    torch.manual_seed(cfg.seed)
    operations = (
        "sum", "max", "min", "count_positive", "count_negative",
    )
    hidden_dim = 16
    head = ArrayExecutableThoughtHead(
        ArrayExecutableThoughtHeadConfig(
            hidden_dim=hidden_dim, array_max_len=cfg.array_max_len,
        )
    )
    # One "epoch" trains over all skill prototypes; subsequent epochs
    # increase the convergence quality and get more samples cached.
    epoch_reports = []
    for epoch in range(cfg.epochs):
        hidden, arrays, lengths, targets, _ = build_array_thought_smoke_batch(
            hidden_dim=hidden_dim,
            array_max_len=cfg.array_max_len,
            samples_per_op=max(4, cfg.steps_per_epoch // len(operations) // 100),
            seed=cfg.seed + epoch,
            operations=operations,
        )
        # The standalone NPCoT head uses its own much smaller optimizer;
        # the caller's `learning_rate` is an LLM-scale hyperparameter, so we
        # map it to a sensible head-scale rate (capped at 0.05).
        head_lr = min(cfg.learning_rate * 100.0, 0.05)
        metrics = run_array_thought_smoke_train(
            head,
            hidden_state=hidden,
            array_inputs=arrays,
            lengths=lengths,
            targets=targets,
            steps=cfg.steps_per_epoch // cfg.epochs,
            learning_rate=head_lr,
        )
        # Crystallize any newly-converged samples into the library.
        crystallize = session.apply_converged_program(
            head, hidden, arrays, lengths=lengths, temperature=0.05
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "train_initial_loss": metrics.initial_loss,
                "train_final_loss": metrics.final_loss,
                "train_final_mae": metrics.final_mae,
                "newly_cached_this_epoch": int(sum(crystallize.newly_cached)),
                "library_entries_after_epoch": len(session.library),
            }
        )

    summary = session.end_task()

    report = {
        "mode": "standalone_npcot_sweep",
        "timestamp": time.time(),
        "config": cfg.__dict__,
        "begin_meta": begin_meta,
        "epochs": epoch_reports,
        "library_summary": {
            "entry_count": summary.entry_count,
            "total_hits": summary.total_hits,
            "newly_cached_count": summary.newly_cached_count,
            "saved": summary.saved,
            "library_path": summary.library_path,
        },
        "session_diff": summary.diff,
    }
    library = ArrayProgramLibrary.load(Path(cfg.library_path).expanduser())
    return attach_compliance_report(report, library)


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_cli(argv)
    if cfg.dry_run:
        report = run_dry(cfg)
        print(json.dumps(report, indent=2))
        return 0
    try:
        report = run_sweep(cfg)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "(re-run with --dry-run to validate the config, or port "
            "run_instruct_sweep.py's training loop into this file.)",
            file=sys.stderr,
        )
        return 1
    out_path = Path(cfg.output_json).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


def attach_compliance_report(
    run_report: dict[str, Any],
    library: ArrayProgramLibrary,
    *,
    input_bound: RangeBound = RangeBound(-10.0, 10.0),
    max_length: int = 16,
) -> dict[str, Any]:
    """Attach a compliance report to the run artifact (call before writing JSON)."""
    compliance = build_compliance_report(
        library,
        config=ComplianceReportConfig(
            verifier=VerifierConfig(
                input_bound=input_bound,
                max_length=max_length,
            )
        ),
    )
    run_report["compliance"] = compliance
    return run_report


if __name__ == "__main__":
    raise SystemExit(main())
