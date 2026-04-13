"""Evaluate Mog body-completion checkpoints on benchmark reference scaffolds."""

from __future__ import annotations

import argparse
import torch

from egdc.mog.model import MogMaskedDiffusion, MogDiffusionConfig
from egdc.mog.tokenizer import MogCodeTokenizer
from egdc.mog.train import get_device
from egdc.mog.training.completion import build_completion_tokens, complete_mog_from_initial
from egdc.mog.benchmark import get_benchmark, evaluate_solution, evaluate_solution_with_compiler
from egdc.mog.eval import evaluate_mog_program


def load_model(checkpoint: str, model_size: str, seq_len: int, spec_len: int, device_pref: str):
    if model_size == "tiny":
        config = MogDiffusionConfig.tiny()
    elif model_size == "small":
        config = MogDiffusionConfig.small()
    else:
        config = MogDiffusionConfig.medium()
    config.max_seq_len = max(config.max_seq_len, seq_len + spec_len + 64)
    model = MogMaskedDiffusion(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    device = get_device(device_pref)
    model = model.to(device)
    model.eval()
    return model, device


def main():
    ap = argparse.ArgumentParser(description="Evaluate Mog completion checkpoints")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", choices=["tiny", "small", "medium"], default="tiny")
    ap.add_argument("--num_problems", type=int, default=10)
    ap.add_argument("--variants_per_factory", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--spec_len", type=int, default=128)
    ap.add_argument("--num_steps", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--num_candidates", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--use_real_compiler", action="store_true")
    args = ap.parse_args()

    model, device = load_model(args.checkpoint, args.model_size, args.seq_len, args.spec_len, args.device)
    tokenizer = MogCodeTokenizer()
    problems = get_benchmark(seed=42, variants_per_factory=args.variants_per_factory)[: args.num_problems]

    passed = 0
    compile_passed = 0
    exact_match = 0
    avg_syntax = 0.0
    avg_static = 0.0

    for idx, problem in enumerate(problems):
        prompt = f"{problem.signature}\n// {problem.description}"
        spec_tokens = torch.tensor([tokenizer.pad(tokenizer.encode(prompt, add_bos_eos=False), args.spec_len)], dtype=torch.long, device=device)
        initial_tokens, fixed_positions, _orig = build_completion_tokens(problem.reference_solution or "", tokenizer, args.seq_len)

        best = None
        best_score = None
        for _cand in range(args.num_candidates):
            completed = complete_mog_from_initial(
                model=model,
                initial_tokens=initial_tokens.unsqueeze(0),
                fixed_positions=fixed_positions.unsqueeze(0),
                spec_tokens=spec_tokens,
                num_steps=args.num_steps,
                temperature=args.temperature,
                device=device,
            )
            code = tokenizer.decode(completed[0].tolist())
            static = evaluate_mog_program(code)
            interp = evaluate_solution(problem, code)
            comp = evaluate_solution_with_compiler(problem, code) if args.use_real_compiler else None
            score = (
                (10.0 if (comp and comp.passed) else 0.0)
                + (5.0 if interp.passed else 0.0)
                + static.syntactic_validity
                + static.overall_score
            )
            if best is None or score > best_score:
                best = (code, interp, comp, static)
                best_score = score

        assert best is not None
        code, interp, comp, static = best
        avg_syntax += static.syntactic_validity
        avg_static += static.overall_score
        if interp.passed:
            passed += 1
        if comp is not None and comp.passed:
            compile_passed += 1
        if code.strip() == (problem.reference_solution or '').strip():
            exact_match += 1
        print(f"[{idx+1:>3d}/{len(problems)}] {problem.name} interp={'PASS' if interp.passed else 'FAIL'}" + (f" compiler={'PASS' if comp and comp.passed else 'FAIL'}" if comp is not None else "") + (" exact=PASS" if code.strip() == (problem.reference_solution or '').strip() else " exact=FAIL"))

    n = max(len(problems), 1)
    print("\nSummary")
    print("-------")
    print(f"Problems:      {len(problems)}")
    print(f"Pass@1:        {passed / n:.3f}")
    if args.use_real_compiler:
        print(f"Compile Pass@1:{compile_passed / n:.3f}")
    print(f"Exact match:   {exact_match / n:.3f}")
    print(f"Avg syntax:    {avg_syntax / n:.3f}")
    print(f"Avg static:    {avg_static / n:.3f}")


if __name__ == "__main__":
    main()
