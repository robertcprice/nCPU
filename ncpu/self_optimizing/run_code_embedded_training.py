"""Train coprocessor with code-embedded arithmetic data on vast.ai.

This trains the coprocessor to activate on code patterns (array indexing,
loop bounds, etc.) rather than raw arithmetic, improving transfer to
real coding tasks.

Usage:
    python run_code_embedded_training.py --model Qwen/Qwen3.5-2B --steps 2000

The script uses code-embedded training data by default, which helps the
coprocessor learn WHEN to activate during code generation.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add ncpu to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ncpu.coprocessor.train import train_coprocessor, TrainingConfig

logger = logging.getLogger(__name__)


DATASET_ALIASES = {
    "code+synthetic": "synthetic+code",
}


def normalize_dataset(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def default_output_dir(model_name: str, dataset: str) -> str:
    model_short = model_name.split("/")[-1].lower().replace(".", "")
    root = "training_results/code_embedded" if "code" in dataset else "training_results/coprocessor"
    return f"{root}/{model_short}"


def main():
    parser = argparse.ArgumentParser(description="Train coprocessor with code-embedded data")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3.5-2B)")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dataset", default="code",
                        choices=["code", "synthetic", "synthetic+code", "code+synthetic", "code+gsm8k"],
                        help="Training dataset mix")
    parser.add_argument("--synthetic-size", type=int, default=10000, help="Number of synthetic/code samples")
    parser.add_argument("--max-value", type=int, default=255, help="Maximum operand value")
    parser.add_argument("--confidence-aware", action="store_true", help="Enable confidence-aware gating")
    parser.add_argument("--max-gate", type=float, default=0.1, help="Max gate value")
    parser.add_argument("--gate-warmup-steps", type=int, default=0, help="Gate warmup steps")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--layers", nargs="+", type=int, default=[-1], help="Layer indices")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset = normalize_dataset(args.dataset)
    output_dir = args.output_dir or default_output_dir(args.model, dataset)

    config = TrainingConfig(
        model_name=args.model,
        layers=args.layers,
        dataset=dataset,
        synthetic_size=args.synthetic_size,
        max_value=args.max_value,
        difficulty="mixed",
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_steps=100,
        eval_every=500,
        target_load=0.01,
        output_dir=output_dir,
        models_dir="models",
        synthetic_only=False,
        confidence_aware=args.confidence_aware,
        max_gate=args.max_gate,
        gate_warmup_steps=args.gate_warmup_steps,
        grad_accum_steps=args.grad_accum_steps,
    )

    logger.info(f"Starting code-embedded training for {args.model}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Dataset: {dataset}, size: {config.synthetic_size}, max_value: {config.max_value}")
    logger.info(f"  Steps: {args.steps}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"  Confidence-aware: {args.confidence_aware}, max_gate: {args.max_gate}")

    result = train_coprocessor(config)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Steps: {result.steps_completed}")
    print(f"  Final loss: {result.final_loss:.4f}")
    print(f"  Eval accuracy: {result.eval_accuracy:.1%}")
    print(f"  Trainable params: {result.trainable_params:,}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")
    print(f"  Weights: {output_dir}/coprocessor_weights.pt")


if __name__ == "__main__":
    main()
