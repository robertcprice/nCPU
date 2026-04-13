"""Evaluate Mog diffusion models on the Mog execution benchmark.

Supports two modes:
- baseline generation via generate_mog
- execution-guided generation via execution_guided_generate_mog

Metrics:
- pass@1
- compile@1 (optional, when using real compiler checks)
- average syntax/static score
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from egdc.mog.model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog.tokenizer import MogCodeTokenizer
from egdc.mog.train import generate_mog, get_device
from egdc.mog.benchmark import get_benchmark, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog.training.execution_guidance import execution_guided_generate_mog
from egdc.mog.eval import evaluate_mog_program


def load_model(checkpoint: str | None, model_size: str, seq_len: int, spec_len: int, device_pref: str = "cpu") -> tuple[MogMaskedDiffusion, torch.device]:
    if model_size == "tiny":
        config = MogDiffusionConfig.tiny()
    elif model_size == "small":
        config = MogDiffusionConfig.small()
    else:
        config = MogDiffusionConfig.medium()
    config.max_seq_len = max(config.max_seq_len, seq_len + spec_len + 64)
    model = MogMaskedDiffusion(config)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    device = get_device(device_pref)
    model = model.to(device)
    model.eval()
    return model, device


def evaluate_model(
    model: MogMaskedDiffusion,
    tokenizer: MogCodeTokenizer,
    problems,
    seq_len: int,
    mode: str,
    num_candidates: int,
    num_steps: int,
    temperature: float,
    device: torch.device,
    use_real_compiler: bool,
) -> dict[str, Any]:
    results = []
    passed = 0
    compiled = 0
    syntax_scores = []
    static_scores = []

    for idx, problem in enumerate(problems):
        if mode == "guided":
            guided = execution_guided_generate_mog(
                model=model,
                tokenizer=tokenizer,
                problem=problem,
                seq_len=seq_len,
                num_candidates=num_candidates,
                num_steps=num_steps,
                temperature=temperature,
                device=device,
                use_real_compiler=use_real_compiler,
            )
            code = guided["best_code"]
        else:
            prompt = f"{problem.signature}\n// {problem.description}"
            spec_tokens = torch.tensor([tokenizer.pad(tokenizer.encode(prompt, add_bos_eos=False), 64)], dtype=torch.long, device=device)
            sample = generate_mog(
                model,
                spec_tokens=spec_tokens,
                seq_len=seq_len,
                num_steps=num_steps,
                temperature=temperature,
                device=device,
            )
            code = tokenizer.decode(sample[0].tolist())

        interp = evaluate_solution(problem, code)
        comp = evaluate_solution_with_compiler(problem, code) if use_real_compiler else None
        static = evaluate_mog_program(code)

        syntax_scores.append(static.syntactic_validity)
        static_scores.append(static.overall_score)
        if interp.passed:
            passed += 1
        if comp is not None and comp.passed:
            compiled += 1

        results.append({
            "problem": problem.name,
            "interp_pass": interp.passed,
            "compiler_pass": comp.passed if comp is not None else None,
            "syntax": static.syntactic_validity,
            "static": static.overall_score,
            "code": code,
        })

        print(f"[{idx+1:>3d}/{len(problems)}] {problem.name} interp={'PASS' if interp.passed else 'FAIL'}" + (f" compiler={'PASS' if comp and comp.passed else 'FAIL'}" if comp is not None else ""))

    summary = {
        "num_problems": len(problems),
        "pass_at_1": passed / max(len(problems), 1),
        "compile_pass_at_1": compiled / max(len(problems), 1) if use_real_compiler else None,
        "avg_syntax": sum(syntax_scores) / max(len(syntax_scores), 1),
        "avg_static": sum(static_scores) / max(len(static_scores), 1),
        "results": results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate Mog diffusion model on the Mog benchmark")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--spec_len", type=int, default=64)
    ap.add_argument("--num_steps", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--mode", choices=["baseline", "guided"], default="guided")
    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--num_problems", type=int, default=10)
    ap.add_argument("--variants_per_factory", type=int, default=1)
    ap.add_argument("--use_real_compiler", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    args = ap.parse_args()

    model, device = load_model(args.checkpoint, args.model_size, args.seq_len, args.spec_len, args.device)
    tokenizer = MogCodeTokenizer()
    problems = get_benchmark(seed=42, variants_per_factory=args.variants_per_factory)[: args.num_problems]

    summary = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        seq_len=args.seq_len,
        mode=args.mode,
        num_candidates=args.num_candidates,
        num_steps=args.num_steps,
        temperature=args.temperature,
        device=device,
        use_real_compiler=args.use_real_compiler,
    )

    print("\nSummary")
    print("-------")
    print(f"Problems:      {summary['num_problems']}")
    print(f"Pass@1:        {summary['pass_at_1']:.3f}")
    if summary['compile_pass_at_1'] is not None:
        print(f"Compile Pass@1:{summary['compile_pass_at_1']:.3f}")
    print(f"Avg syntax:    {summary['avg_syntax']:.3f}")
    print(f"Avg static:    {summary['avg_static']:.3f}")


if __name__ == "__main__":
    main()
