#!/usr/bin/env python3
"""
Local-model adapter: open-source model ↔ our benchmark runners.

Exposes an `anthropic`-like `.messages.create(...)` surface backed by
any local model (MLX on Apple Silicon, HuggingFace transformers on
GPU, or an OpenAI-compatible HTTP endpoint like vLLM/llama.cpp). Our
existing runners can drop this in by changing one import:

    # Before (Claude API):
    import anthropic; client = anthropic.Anthropic(api_key=...)

    # After (local Qwen3.5 on MLX):
    from local_model_adapter import LocalModelClient
    client = LocalModelClient(backend="mlx",
        model="mlx-community/Qwen3.5-4B-Instruct-4bit")

    # Same interface from here on.
    resp = client.messages.create(
        model="qwen3.5-4b", max_tokens=512, temperature=0.0,
        messages=[{"role": "user", "content": "Write Python..."}],
    )
    print(resp.content[0].text)

Supported backends:
  - "mlx"     → mlx-lm (Apple Silicon, 4-bit quant fits 4-8 B in
                unified memory)
  - "hf"      → transformers (needs a GPU; pulls the full model)
  - "openai"  → OpenAI-compatible REST (works with vLLM, llama.cpp,
                LM Studio, Ollama — any server that implements
                /v1/chat/completions)

The adapter is *deliberately thin*. No caching, no retries, no
constrained decoding — those are layered on by `inference_enhanced.py`
and the existing runners. This module just answers "can you call an
open-source model with the same interface the benchmarks use?" — yes.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class _TextBlock:
    text: str
    # Name `type` for loose parity with anthropic's ContentBlock shape.
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    content: List[_TextBlock]
    usage: _Usage


class LocalModelClient:
    """Drop-in replacement for `anthropic.Anthropic` against local
    open-source models. Implements only what our benchmark runners
    use: `.messages.create(model, max_tokens, temperature, messages)`.

    Each backend lazy-loads its dependency, so a user running the MLX
    backend doesn't need torch, and vice versa."""

    def __init__(
        self,
        backend: str,
        model: str,
        device: Optional[str] = None,
        api_base: Optional[str] = None,
        adapter_path: Optional[str] = None,
        adapter_routing: str = "always",
    ):
        self.backend = backend
        self.model = model
        self.device = device
        self.api_base = api_base or os.environ.get("LOCAL_MODEL_API_BASE")
        self.adapter_path = adapter_path
        self.adapter_routing = adapter_routing
        self._impl = None
        self._base_impl = None
        self._adapter_impl = None

        self.messages = _Messages(self)

    def _get_impl(self, use_adapter: bool):
        if self.backend == "mlx":
            if use_adapter:
                if self._adapter_impl is None:
                    self._adapter_impl = _MLXBackend(self.model, self.adapter_path)
                return self._adapter_impl
            if self._base_impl is None:
                self._base_impl = _MLXBackend(self.model, None)
            return self._base_impl
        if self._impl is not None:
            return self._impl
        if self.backend == "hf":
            if self.adapter_path:
                raise ValueError("adapter_path is only supported for backend='mlx'")
            self._impl = _HFBackend(self.model, self.device)
        elif self.backend == "openai":
            if self.adapter_path:
                raise ValueError("adapter_path is only supported for backend='mlx'")
            self._impl = _OpenAICompatBackend(self.model, self.api_base)
        else:
            raise ValueError(f"unknown backend: {self.backend}")
        return self._impl

    def _should_use_adapter(self, prompt: str, messages: Optional[list]) -> bool:
        if not self.adapter_path:
            return False
        if self.adapter_routing == "always":
            return True
        if self.adapter_routing == "never":
            return False
        if self.adapter_routing != "utility_only":
            raise ValueError(f"unknown adapter_routing: {self.adapter_routing}")
        text_parts = [prompt]
        for m in messages or []:
            text_parts.append(str(m.get("content", "")))
        text = "\n".join(text_parts)
        score = 0
        markers = [
            "Reimplement the Unix utility",
            "solve(stdin: str) -> str",
            "must return exactly the stdout",
            "Expected output:",
            "Given the stdin below",
        ]
        score += sum(1 for marker in markers if marker in text)
        if re.search(r"Unix utility `[^`]+`", text):
            score += 1
        return score >= 2

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        messages: Optional[list] = None,
    ) -> tuple[str, int, int]:
        impl = self._get_impl(self._should_use_adapter(prompt, messages))
        return impl.generate(prompt, max_tokens, temperature, messages=messages)


class _Messages:
    """anthropic.Anthropic().messages shim."""

    def __init__(self, client: LocalModelClient):
        self._client = client

    def create(
        self,
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.0,
        messages: Optional[list] = None,
    ) -> Response:
        messages = messages or []
        # Build a plain fallback prompt, but pass the structured
        # messages through so backends with native chat templating can
        # stay on the model's expected format.
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt += f"<|{role}|>\n{content}\n"
        prompt += "<|assistant|>\n"

        text, in_tok, out_tok = self._client._generate(
            prompt, max_tokens, temperature, messages=messages
        )
        return Response(
            content=[_TextBlock(text=text)],
            usage=_Usage(input_tokens=in_tok, output_tokens=out_tok),
        )


# ─── Backend implementations ─────────────────────────────────────────────────


class _MLXBackend:
    """mlx-lm on Apple Silicon. 4-bit quant recommended."""

    def __init__(self, model_id: str, adapter_path: Optional[str] = None):
        try:
            from mlx_lm import load, generate  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. `pip install mlx-lm` on macOS/Apple Silicon."
            ) from e
        self._load_fn = load
        self._generate_fn = generate
        self._model, self._tokenizer = load(model_id, adapter_path=adapter_path)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        messages: Optional[list] = None,
    ):
        from mlx_lm.sample_utils import make_sampler  # type: ignore
        if messages:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        sampler = make_sampler(temp=max(temperature, 1e-4))
        out = self._generate_fn(
            self._model, self._tokenizer,
            prompt=prompt, max_tokens=max_tokens,
            sampler=sampler, verbose=False,
        )
        in_tok = len(self._tokenizer.encode(prompt))
        out_tok = len(self._tokenizer.encode(out))
        return (out, in_tok, out_tok)


class _HFBackend:
    """transformers + accelerate. Needs a CUDA GPU for practical speed."""

    def __init__(self, model_id: str, device: Optional[str]):
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM, AutoTokenizer,
            )
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch not installed. "
                "`pip install transformers accelerate torch`."
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Qwen3.5 ships as VL by default. Text-only code synthesis must
        # go through Qwen3_5ForCausalLM (or the equivalent text-config
        # cast). Detect and fall back cleanly.
        try:
            from transformers import Qwen3_5ForCausalLM  # type: ignore
            if "qwen3.5" in model_id.lower():
                self._model = Qwen3_5ForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16,
                    device_map=device or "auto",
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16,
                    device_map=device or "auto",
                )
        except ImportError:
            # Older transformers lacks the VL-specific loader; the base
            # AutoModelForCausalLM still works.
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map=device or "auto",
            )
        self._model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        messages: Optional[list] = None,
    ):
        if messages and hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        ids = self._tokenizer(prompt, return_tensors="pt")
        ids = {k: v.to(self._model.device) for k, v in ids.items()}
        with self._torch.inference_mode():
            out = self._model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 1e-4,
                temperature=max(temperature, 1e-4),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = out[0][ids["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        in_tok = int(ids["input_ids"].shape[1])
        out_tok = int(gen_ids.shape[0])
        return (text, in_tok, out_tok)


class _OpenAICompatBackend:
    """HTTP backend for vLLM / llama.cpp server / LM Studio / Ollama.
    Any server that exposes /v1/chat/completions works."""

    def __init__(self, model_id: str, api_base: Optional[str]):
        import urllib.request  # stdlib, no dep
        self._urllib = urllib.request
        self._model_id = model_id
        self._api_base = api_base or "http://localhost:8000/v1"

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        messages: Optional[list] = None,
    ):
        body = {
            "model": self._model_id,
            "messages": messages or [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": max(temperature, 0.0),
        }
        req = self._urllib.Request(
            f"{self._api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer local"},  # dummy bearer
        )
        with self._urllib.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        return (
            text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


# ─── CLI sanity check ────────────────────────────────────────────────────────


def _smoke() -> int:
    """`python3 tools/benchmarks/local_model_adapter.py --backend ... --model ...`
    runs one tiny prompt and prints the response. Good for confirming
    the install before firing off a full benchmark."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "hf", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--adapter-routing", default="always",
                    choices=["always", "never", "utility_only"])
    ap.add_argument("--prompt", default="Write a Python function to compute fibonacci(n).")
    args = ap.parse_args()

    client = LocalModelClient(
        backend=args.backend,
        model=args.model,
        api_base=args.api_base,
        adapter_path=args.adapter_path,
        adapter_routing=args.adapter_routing,
    )
    resp = client.messages.create(
        model=args.model, max_tokens=200, temperature=0.0,
        messages=[{"role": "user", "content": args.prompt}],
    )
    text = resp.content[0].text
    print(f"--- {args.backend}:{args.model} ---")
    print(text[:800])
    print(f"--- tokens: in={resp.usage.input_tokens} out={resp.usage.output_tokens} ---")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
