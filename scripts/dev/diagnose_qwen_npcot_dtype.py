"""Minimal dtype-propagation diagnostic for Qwen3.5 + NPCoT wrapper.

Loads Qwen3.5-0.8B (smallest Qwen3.5), wraps one MLP layer with
NCPUCoprocessorMLPWithArrayThought, and runs a single forward pass. At each
step, reports the dtype of representative tensors so we can see exactly
where bf16 / float32 diverge.

Run with::

    python3 -B scripts/diagnose_qwen_npcot_dtype.py

`-B` disables bytecode caching so we're guaranteed to use the current
source. That matters because the earlier VM2 runs seemed to execute stale
bytecode even after explicit cache clears.
"""

from __future__ import annotations

import sys
import torch


def bar(label: str) -> None:
    print(f"\n=== {label} " + "=" * (60 - len(label)), flush=True)


def dtype_of(module_or_tensor):
    if hasattr(module_or_tensor, "parameters"):
        for p in module_or_tensor.parameters():
            if p.is_floating_point():
                return p.dtype
        return None
    return module_or_tensor.dtype


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from ncpu.self_optimizing.humaneval_runner import _load_hf_model_vl_aware
    from ncpu.coprocessor.array_thought_coprocessor import (
        ArrayThoughtCoprocessorConfig,
        NCPUCoprocessorMLPWithArrayThought,
    )
    from ncpu.coprocessor.config import NCPUCoprocessorConfig

    model_name = args.model
    target_dtype = torch.bfloat16

    bar("1. LOAD MODEL")
    print(f"target_dtype: {target_dtype}", flush=True)
    model = _load_hf_model_vl_aware(
        model_name, dtype=target_dtype, trust_remote_code=False
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"model class: {type(model).__name__}", flush=True)
    print(f"model.config.torch_dtype: {getattr(model.config, 'torch_dtype', 'N/A')}", flush=True)
    print(f"hidden_size: {model.config.hidden_size}", flush=True)
    print(f"num layers: {len(model.model.layers)}", flush=True)

    bar("2. AUDIT BASE MODEL DTYPES (first 10 params)")
    count_by_dtype: dict = {}
    for i, (name, p) in enumerate(model.named_parameters()):
        count_by_dtype[str(p.dtype)] = count_by_dtype.get(str(p.dtype), 0) + 1
        if i < 10:
            print(f"  {name}: {p.dtype}", flush=True)
    print(f"total-param dtype histogram: {count_by_dtype}", flush=True)

    bar("3. AUDIT UNWRAPPED MLP ON LAST LAYER")
    # Find the layers attribute — different for VL vs non-VL models.
    layers = None
    for path in (("model", "layers"), ("model", "language_model", "layers"), ("language_model", "layers")):
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.ModuleList):
            layers = obj
            print(f"  found layers at model.{'.'.join(path)}", flush=True)
            break
    if layers is None:
        print("  CANNOT FIND LAYERS ATTRIBUTE — dumping model structure:", flush=True)
        for name, mod in model.named_modules():
            if "layers" in name or "language_model" in name:
                print(f"    {name}: {type(mod).__name__}", flush=True)
        return 1
    original_mlp = layers[-1].mlp
    print(f"original_mlp class: {type(original_mlp).__name__}", flush=True)
    for name, p in original_mlp.named_parameters():
        print(f"  {name}: {p.dtype} shape={tuple(p.shape)}", flush=True)

    bar("4. CONSTRUCT WRAPPER")
    hidden_dim = model.config.hidden_size
    coproc_cfg = NCPUCoprocessorConfig(n_bits=8, num_ops=7, max_gate=0.1, residual_init_scale=0.0)
    array_cfg = ArrayThoughtCoprocessorConfig(array_max_len=8, max_gate=0.05)
    wrapper = NCPUCoprocessorMLPWithArrayThought(
        original_mlp=original_mlp,
        hidden_dim=hidden_dim,
        config=coproc_cfg,
        array_thought_config=array_cfg,
    )
    print("wrapper dtypes BEFORE .to(bf16):", flush=True)
    for name, p in wrapper.named_parameters():
        if "original_mlp" not in name:
            print(f"  {name}: {p.dtype}", flush=True)
            break
    wrapper = wrapper.to(device="cuda", dtype=target_dtype)
    print("wrapper dtypes AFTER .to(bf16) (sample):", flush=True)
    mismatches = 0
    for name, p in wrapper.named_parameters():
        if p.dtype != target_dtype:
            print(f"  MISMATCH {name}: {p.dtype}", flush=True)
            mismatches += 1
    print(f"total param mismatches in wrapper: {mismatches}", flush=True)

    # Buffers too.
    buf_mismatches = 0
    for mod_name, mod in wrapper.named_modules():
        for bname, buf in mod._buffers.items():
            if buf is not None and buf.is_floating_point() and buf.dtype != target_dtype:
                print(f"  BUFFER MISMATCH {mod_name}.{bname}: {buf.dtype}", flush=True)
                buf_mismatches += 1
    print(f"total buffer mismatches in wrapper: {buf_mismatches}", flush=True)

    bar("5. ATTACH WRAPPER TO MODEL LAYER")
    layers[-1].mlp = wrapper
    # Verify the swap worked.
    new_mlp = layers[-1].mlp
    print(f"new_mlp class: {type(new_mlp).__name__}", flush=True)

    bar("6. FORWARD PASS — 2 TOKENS")
    inputs = tokenizer("def f(x):", return_tensors="pt").to("cuda")
    print(f"inputs.input_ids.shape: {inputs.input_ids.shape}", flush=True)
    print(f"inputs.input_ids.dtype: {inputs.input_ids.dtype}", flush=True)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"SUCCESS: outputs.logits.dtype={outputs.logits.dtype}, shape={outputs.logits.shape}", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"\nFAIL: {exc}\n", flush=True)

        bar("7. POST-MORTEM: what's in the last-layer mlp NOW?")
        mlp = layers[-1].mlp
        print(f"  outer class: {type(mlp).__name__}", flush=True)
        if hasattr(mlp, "base"):
            print(f"  mlp.base class: {type(mlp.base).__name__}", flush=True)
            if hasattr(mlp.base, "original_mlp"):
                om = mlp.base.original_mlp
                print(f"  mlp.base.original_mlp class: {type(om).__name__}", flush=True)
                for name, p in om.named_parameters():
                    print(f"    {name}: {p.dtype}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
