"""HumanEval evaluator harness with optional NPCoT library integration (BENCH-1).

Loads a HuggingFace causal LM, optionally wraps selected MLP layers with
`NCPUCoprocessorMLPWithArrayThought`, attaches a pre-trained
`ArrayProgramLibrary`, and runs the HumanEval benchmark. Produces pass@1
numbers per configuration.

Two design decisions that matter:

1. **Lazy imports**. `transformers`, `datasets`, and `torch` are
   imported inside the functions that need them, not at module top. This
   lets the test suite import this module on CPU-only CI hosts without
   needing the HF stack installed.

2. **`--dry-run` mode**. Validates config + library paths + target layer
   indices without actually loading the LLM or downloading datasets.
   Cheap sanity check before burning GPU time on vast.ai.

Usage::

    # Cheap sanity on your laptop:
    python3 -m ncpu.self_optimizing.humaneval_runner --dry-run --model Qwen/Qwen3.5-1.5B

    # Real run on a rented GPU:
    python3 -m ncpu.self_optimizing.humaneval_runner \\
        --model Qwen/Qwen3.5-1.5B \\
        --library ~/.nCPU_program_library.json \\
        --max-problems 164 \\
        --out humaneval_run.json

    # Baseline (no library):
    python3 -m ncpu.self_optimizing.humaneval_runner \\
        --model Qwen/Qwen3.5-1.5B \\
        --no-library \\
        --max-problems 164 \\
        --out humaneval_baseline.json

Output is a JSON artifact with pass@1, per-problem details, and compliance
report attached to the library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class HumanEvalConfig:
    model: str = "Qwen/Qwen3.5-1.5B"
    library_path: Optional[Path] = None
    coprocessor_checkpoint: Optional[Path] = None
    target_layers: list[int] = field(default_factory=lambda: [-2, -1])
    array_max_len: int = 8
    array_thought_max_gate: float = 0.05
    max_problems: int = 164
    max_new_tokens: int = 400
    temperature: float = 0.0
    output_json: Path = Path("humaneval_run.json")
    dry_run: bool = False
    device: str = "auto"
    trust_remote_code: bool = False
    use_npcot: bool = True
    quantize: bool = False


def parse_cli(argv: list[str] | None = None) -> HumanEvalConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3.5-4B-Instruct")
    p.add_argument("--library", dest="library_path", type=Path, default=None)
    p.add_argument("--coprocessor-checkpoint", dest="coprocessor_checkpoint", type=Path, default=None,
                   help="Trained NCPUCoprocessorMLPWithArrayThought state_dict (from npcot_qwen_training --out-checkpoint).")
    p.add_argument("--no-library", action="store_true",
                   help="Run without NPCoT library (baseline mode).")
    p.add_argument("--target-layers", default="-2,-1")
    p.add_argument("--array-max-len", type=int, default=8)
    p.add_argument("--array-thought-max-gate", type=float, default=0.05)
    p.add_argument("--max-problems", type=int, default=164)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", dest="output_json", type=Path, default=Path("humaneval_run.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--quantize", action="store_true",
                   help="Apply dynamic int8 quantization (CPU only, ~2x speed).")
    args = p.parse_args(argv)
    return HumanEvalConfig(
        model=args.model,
        library_path=args.library_path,
        coprocessor_checkpoint=args.coprocessor_checkpoint,
        target_layers=[int(x) for x in args.target_layers.split(",") if x.strip()],
        array_max_len=args.array_max_len,
        array_thought_max_gate=args.array_thought_max_gate,
        max_problems=args.max_problems,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_json=args.output_json,
        dry_run=args.dry_run,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        use_npcot=not args.no_library,
        quantize=args.quantize,
    )


# ---------------------------------------------------------------------------
# Dry-run: validate config without loading GPU deps
# ---------------------------------------------------------------------------


def run_dry(cfg: HumanEvalConfig) -> dict:
    checks: list[tuple[str, bool, str]] = []

    lib_ok = True
    lib_entries = 0
    if cfg.use_npcot:
        if cfg.library_path is None:
            lib_ok = False
            checks.append(("library_path", False, "use-npcot requires --library PATH"))
        else:
            resolved = cfg.library_path.expanduser()
            if not resolved.exists():
                lib_ok = False
                checks.append(("library_exists", False, f"{resolved} not found"))
            else:
                from ncpu.self_optimizing.array_program_library import (
                    ArrayProgramLibrary,
                )
                lib = ArrayProgramLibrary.load(resolved)
                lib_entries = len(lib)
                checks.append(("library_exists", True, f"{lib_entries} entries"))
    else:
        checks.append(("library_mode", True, "baseline (no library)"))

    # Layer index sanity (without actually loading the model).
    for idx in cfg.target_layers:
        if not isinstance(idx, int):
            checks.append(("target_layer_type", False, f"{idx!r} is not int"))
        else:
            checks.append((f"layer_{idx}", True, "ok"))

    # Output path writable?
    out_parent = cfg.output_json.expanduser().parent
    checks.append((
        "output_path_writable",
        out_parent.exists() or out_parent.parent.exists(),
        str(cfg.output_json),
    ))

    return {
        "mode": "dry_run",
        "timestamp": time.time(),
        "config": {
            "model": cfg.model,
            "library_path": str(cfg.library_path) if cfg.library_path else None,
            "target_layers": cfg.target_layers,
            "use_npcot": cfg.use_npcot,
            "max_problems": cfg.max_problems,
        },
        "checks": [
            {"name": n, "ok": ok, "detail": d} for n, ok, d in checks
        ],
        "library_entries": lib_entries,
        "all_ok": all(ok for _, ok, _ in checks),
    }


# ---------------------------------------------------------------------------
# Real run — requires torch + transformers + datasets
# ---------------------------------------------------------------------------


_CODE_EXTRACT_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

# HumanEval stop-tokens — when a base model continues into a *new* top-level
# definition or test block, we should stop accepting its output at that point.
_STOP_PATTERNS = (
    "\nclass ",
    "\ndef ",
    "\nif __name__",
    "\n#",
    "\nprint(",
    "\nassert ",
    "\n```",
)


def _truncate_at_stop(code: str) -> str:
    """Stop the generation at the first top-level drift away from the body."""
    best = len(code)
    for pat in _STOP_PATTERNS:
        idx = code.find(pat)
        if idx != -1 and idx < best:
            best = idx
    return code[:best].rstrip()


def _extract_code(generated: str, prompt: str) -> str:
    """Normalize model output into a runnable function body continuation.

    HumanEval prompts end mid-function. The base-model completion style is
    "raw body tokens continuing from where the prompt stopped"; the
    instruct/chat style is "fenced code block containing the whole function".
    This function handles both — it returns a string such that
    ``problem['prompt'] + returned_string`` is a runnable Python module.
    """
    # Strip any re-echoed prompt.
    if prompt and generated.startswith(prompt):
        generated = generated[len(prompt):]

    # Fenced block (instruct-style): extract content, check if it already
    # contains the function definition, and if so drop the prefix so we
    # don't double-define the function when concatenated with the prompt.
    match = _CODE_EXTRACT_RE.search(generated)
    if match is not None:
        inside = match.group(1).strip()
        # If the fenced block opens with the same `def ...` as the prompt,
        # strip everything up to and including the first `:\n` so we're
        # left with just the body. Otherwise treat the block as a full file
        # and return it (we'll concat the prompt separately).
        func_name = _prompt_function_name(prompt)
        if func_name and f"def {func_name}" in inside:
            # The instruct model re-wrote the whole function; return just
            # the body with the same indentation as the prompt expects.
            return _strip_function_header(inside)
        return _truncate_at_stop(inside)

    # Base-model raw continuation: truncate at top-level drift.
    return _truncate_at_stop(generated)


def _prompt_function_name(prompt: str) -> Optional[str]:
    """Pull the entry-point function name out of a HumanEval prompt."""
    match = re.search(r"def\s+(\w+)\s*\(", prompt)
    return match.group(1) if match else None


def _strip_function_header(code: str) -> str:
    """Given a full function definition, return just the body text."""
    match = re.search(r"def\s+\w+\s*\([^)]*\)[^:]*:\s*\n", code)
    if match is None:
        return code
    return code[match.end():]


def _check_solution(
    problem: dict, solution_code: str, *, timeout_s: float = 5.0
) -> tuple[bool, str]:
    """Execute the solution under a subprocess with timeout, return pass/fail."""
    import subprocess
    import tempfile

    # Build the harness without dedent — solution_code has its own indent
    # structure that we must preserve byte-for-byte.
    harness_parts = [
        "import sys",
        "# === solution ===",
        solution_code,
        "# === test ===",
        problem["test"],
        f"check({problem['entry_point']})",
        "print('OK')",
    ]
    harness = "\n".join(harness_parts)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as fh:
        fh.write(harness)
        tmp_path = fh.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        error = (result.stderr or result.stdout).strip().splitlines()[-1:]
        return False, (error[0] if error else "nonzero exit")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def load_humaneval_problems(max_problems: int) -> list[dict]:
    """Load HumanEval from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= max_problems:
            break
        problems.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "canonical_solution": row.get("canonical_solution", ""),
        })
    return problems


def _force_model_dtype(model, target_dtype) -> None:
    """Cast every parameter and buffer to `target_dtype` in place.

    `model.to(dtype=...)` handles most of this, but Qwen3.5 VL config
    embeds its own `dtype: "bfloat16"` in the text_config that leaks into
    sub-module construction. We do an explicit cast afterwards to be
    certain every tensor ends up in the requested dtype.
    """
    import torch

    model.to(dtype=target_dtype)
    for module in model.modules():
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype != target_dtype:
                module._buffers[name] = buf.to(dtype=target_dtype)


def _load_hf_model_vl_aware(model_name: str, *, dtype, trust_remote_code: bool):
    """Load a HuggingFace causal LM, handling Qwen3.5 VL text-only extraction.

    Qwen/Qwen3.5-{4B,9B} are multimodal (Qwen3_5ForConditionalGeneration).
    For HumanEval / MBPP text-only coding eval we need only the language
    tower. This function:

    1. Tries `AutoModelForCausalLM` first — works for non-VL architectures.
    2. On failure, falls back to loading the full
       `Qwen3_5ForConditionalGeneration` and returns its `.language_model`
       attribute so the rest of the eval pipeline sees a standard text
       causal LM.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        pass

    # Fallback: VL model. Load the Qwen3.5 VL class directly — its forward
    # accepts text-only inputs (input_ids + attention_mask) when no
    # `pixel_values` is supplied, so we don't need to extract the text tower
    # into a fresh CausalLM instance. This avoids the dtype-propagation
    # bug where reconstructing `Qwen3_5ForCausalLM(text_cfg)` and then
    # calling `load_state_dict(strict=False)` left mismatched-key weights
    # at their default-init float32 dtype.
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    arch = (config.architectures or [""])[0]
    if arch == "Qwen3_5ForConditionalGeneration":
        from transformers import Qwen3_5ForConditionalGeneration
        full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        # Expose the text tower at `.model.layers` for the layer-wrap code
        # path. If the VL model has a different attribute layout, patch it
        # so callers can always do `model.model.layers[idx].mlp = wrapper`.
        if not hasattr(full_model, "model") or not hasattr(full_model.model, "layers"):
            # Try to find the layers attribute somewhere deeper.
            for attr_path in (
                ("model", "language_model", "layers"),
                ("language_model", "layers"),
                ("model", "layers"),
            ):
                obj = full_model
                ok = True
                for attr in attr_path:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        ok = False
                        break
                if ok and isinstance(obj, torch.nn.ModuleList):
                    # Make the layers accessible via model.model.layers.
                    if not hasattr(full_model, "model"):
                        class _Shim:
                            pass
                        full_model.model = _Shim()
                    full_model.model.layers = obj
                    break
        return full_model

    raise RuntimeError(
        f"Unsupported model architecture for VL-text-only extraction: {arch}"
    )


def load_model_with_optional_npcot(cfg: HumanEvalConfig):
    """Load the HF model; wrap target layers with coprocessor if --library given."""
    import torch
    from transformers import AutoTokenizer

    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = cfg.device
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model, trust_remote_code=cfg.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_hf_model_vl_aware(
        cfg.model,
        dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    ).to(device)
    model.eval()

    if cfg.quantize and device == "cpu":
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8,
        )
        print("[lcb] applied dynamic int8 quantization", flush=True)

    if not cfg.use_npcot:
        return model, tokenizer, device, None

    # Wrap target layers.
    from ncpu.coprocessor.array_thought_coprocessor import (
        ArrayThoughtCoprocessorConfig,
        NCPUCoprocessorMLPWithArrayThought,
    )
    from ncpu.coprocessor.config import NCPUCoprocessorConfig
    from ncpu.self_optimizing.array_program_library import (
        ArrayProgramLibrary,
    )

    hidden_dim = int(getattr(model.config, "hidden_size", 0) or 0)
    if hidden_dim <= 0:
        raise ValueError("could not infer hidden_size from model config")
    layers = model.model.layers
    n_layers = len(layers)
    coproc_cfg = NCPUCoprocessorConfig(
        n_bits=8, num_ops=7, max_gate=0.1, residual_init_scale=0.0,
    )
    array_cfg = ArrayThoughtCoprocessorConfig(
        array_max_len=cfg.array_max_len,
        max_gate=cfg.array_thought_max_gate,
    )
    wrapped_layers: list[int] = []
    wrappers: list = []
    for raw_idx in cfg.target_layers:
        idx = raw_idx if raw_idx >= 0 else n_layers + raw_idx
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"target layer {raw_idx} out of range [0, {n_layers})")
        original_mlp = layers[idx].mlp
        wrapper = NCPUCoprocessorMLPWithArrayThought(
            original_mlp=original_mlp,
            hidden_dim=hidden_dim,
            config=coproc_cfg,
            array_thought_config=array_cfg,
        ).to(device=device, dtype=dtype)
        layers[idx].mlp = wrapper
        wrappers.append(wrapper)
        wrapped_layers.append(idx)

    # Load trained coprocessor weights if available. Without this, the
    # wrapper's projection layers (array_proj, length_proj, output_proj,
    # gate_proj) are random — attaching a library then injects noise into
    # the base model's hidden states. That destroys generation (observed
    # pass@1 -> 0% on HumanEval). The trained checkpoint carries the
    # weights that make the library's contribution meaningful.
    ckpt_info: dict | None = None
    if cfg.coprocessor_checkpoint is not None:
        ckpt_path = cfg.coprocessor_checkpoint.expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"coprocessor checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        loaded_params = 0
        for i, wrapper in enumerate(wrappers):
            # Support two checkpoint formats:
            #  1) Legacy: coprocessor_state_dict with wrapper_{i}.* keys
            #  2) Training: layer_{i}_router / layer_{i}_expert dicts
            router_key = f"layer_{i}_router"
            expert_key = f"layer_{i}_expert"
            if router_key in payload or expert_key in payload:
                state: dict = {}
                if router_key in payload:
                    for k, v in payload[router_key].items():
                        state[f"base.router.{k}"] = v.to(dtype=dtype)
                if expert_key in payload:
                    for k, v in payload[expert_key].items():
                        state[f"base.expert.{k}"] = v.to(dtype=dtype)
                missing, unexpected = wrapper.load_state_dict(state, strict=False)
                loaded_params += len(state)
            else:
                coproc_state = payload.get("coprocessor_state_dict", {})
                prefix = f"wrapper_{i}."
                wrapper_state = {
                    k[len(prefix):]: v.to(dtype=dtype)
                    for k, v in coproc_state.items()
                    if k.startswith(prefix)
                }
                if wrapper_state:
                    missing, unexpected = wrapper.load_state_dict(wrapper_state, strict=False)
                    loaded_params += len(wrapper_state)
        ckpt_info = {
            "checkpoint_path": str(ckpt_path),
            "loaded_params": int(loaded_params),
            "num_wrappers": len(wrappers),
        }
    else:
        ckpt_info = {
            "checkpoint_path": None,
            "warning": "coprocessor projections are RANDOM — library attachment will likely degrade generation. Pass --coprocessor-checkpoint PATH to fix.",
        }

    library = ArrayProgramLibrary.load(cfg.library_path.expanduser())
    # Attach library to every wrapped coprocessor layer.
    for module in model.modules():
        if hasattr(module, "attach_library") and module.__class__.__name__ == (
            "NCPUCoprocessorMLPWithArrayThought"
        ):
            module.attach_library(library, task_name="humaneval")

    return model, tokenizer, device, {
        "wrapped_layers": wrapped_layers,
        "library_entries": len(library),
        "coprocessor_checkpoint": ckpt_info,
    }


def generate_solution(
    model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float, device: str
) -> str:
    import torch

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "You are a Python code generator. Output ONLY executable Python code. No explanation, no markdown prose, no comments about approach."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return generated


def run_humaneval(cfg: HumanEvalConfig) -> dict:
    from ncpu.self_optimizing.compliance_report import (
        ComplianceReportConfig,
        build_compliance_report,
    )

    print(f"[humaneval] loading model {cfg.model}", flush=True)
    model, tokenizer, device, npcot_meta = load_model_with_optional_npcot(cfg)
    print(f"[humaneval] loaded on {device}; NPCoT: {npcot_meta}", flush=True)

    print(f"[humaneval] loading up to {cfg.max_problems} problems", flush=True)
    problems = load_humaneval_problems(cfg.max_problems)
    print(f"[humaneval] {len(problems)} problems loaded", flush=True)

    per_problem: list[dict] = []
    pass_count = 0
    t_start = time.perf_counter()
    for i, problem in enumerate(problems):
        t0 = time.perf_counter()
        generated = generate_solution(
            model, tokenizer, problem["prompt"],
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            device=device,
        )
        gen_s = time.perf_counter() - t0
        code = _extract_code(generated, problem["prompt"])
        # Concatenate the original prompt body (function signature) with the
        # model's continuation. HumanEval expects the entry-point function
        # in scope.
        full_solution = problem["prompt"] + code
        passed, err = _check_solution(problem, full_solution)
        if passed:
            pass_count += 1
        per_problem.append({
            "task_id": problem["task_id"],
            "passed": passed,
            "gen_seconds": round(gen_s, 3),
            "code_chars": len(code),
            "error": err if not passed else None,
        })
        if (i + 1) % 10 == 0 or passed:
            print(
                f"[humaneval] {i+1}/{len(problems)} "
                f"{problem['task_id']}: {'PASS' if passed else 'FAIL'} "
                f"(pass@1 so far {pass_count}/{i+1} = {pass_count/(i+1)*100:.1f}%)",
                flush=True,
            )
    total_s = time.perf_counter() - t_start
    pass_at_1 = pass_count / max(len(problems), 1)

    report = {
        "mode": "humaneval_real_run",
        "timestamp": time.time(),
        "config": {
            "model": cfg.model,
            "library_path": str(cfg.library_path) if cfg.library_path else None,
            "use_npcot": cfg.use_npcot,
            "target_layers": cfg.target_layers,
            "max_problems": cfg.max_problems,
            "temperature": cfg.temperature,
        },
        "npcot_meta": npcot_meta,
        "results": {
            "pass_at_1": pass_at_1,
            "pass_count": pass_count,
            "total_problems": len(problems),
            "total_seconds": round(total_s, 2),
            "mean_gen_seconds": round(
                sum(r["gen_seconds"] for r in per_problem) / max(len(per_problem), 1),
                3,
            ),
        },
        "per_problem": per_problem,
    }

    if cfg.use_npcot and cfg.library_path is not None:
        from ncpu.self_optimizing.array_program_library import ArrayProgramLibrary
        library = ArrayProgramLibrary.load(cfg.library_path.expanduser())
        report["compliance"] = build_compliance_report(
            library,
            config=ComplianceReportConfig(library_name=cfg.library_path.stem),
        )

    return report


def main(argv: list[str] | None = None) -> int:
    cfg = parse_cli(argv)
    if cfg.dry_run:
        report = run_dry(cfg)
        print(json.dumps(report, indent=2))
        return 0 if report["all_ok"] else 1

    try:
        report = run_humaneval(cfg)
    except ImportError as exc:
        print(f"error: missing dependency ({exc}). Install transformers + datasets first.",
              file=sys.stderr)
        return 2

    out = cfg.output_json.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    r = report["results"]
    print(f"\n[humaneval] done. pass@1 = {r['pass_at_1'] * 100:.2f}% "
          f"({r['pass_count']}/{r['total_problems']}), wall {r['total_seconds']}s")
    print(f"[humaneval] report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
