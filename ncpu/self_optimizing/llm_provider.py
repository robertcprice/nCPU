"""
LLM Provider Interface for SOME

Supports multiple LLM backends:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Local models (via OpenAI-compatible API)
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum
from pathlib import Path
import json


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HF_LOCAL = "hf_local"
    HF_SEGMENTED_CACHE = "hf_segmented_cache"
    HF_FAST_WEIGHTS = "hf_fast_weights"
    GLM = "glm"  # GLM via MCP - 10x cheaper than OpenAI
    MINIMAX = "minimax"  # MiniMax API - cheap and fast
    QWEN = "qwen"  # Qwen via API or local
    DEEPSEEK = "deepseek"  # DeepSeek API
    TOGETHER = "together"  # Together AI - aggregates many models
    REPLICATE = "replicate"  # Replicate API


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 2048


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    _hf_local_cache: dict[str, tuple[object, object, str]] = {}

    @staticmethod
    def create_provider(
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create an LLM provider function.

        Returns a function that takes a prompt and returns text.
        """
        if provider == "openai":
            return LLMProviderFactory._openai_provider(model, api_key, **kwargs)
        elif provider == "anthropic":
            return LLMProviderFactory._anthropic_provider(model, api_key, **kwargs)
        elif provider == "local":
            return LLMProviderFactory._local_provider(model, **kwargs)
        elif provider == "hf_local":
            return LLMProviderFactory._hf_local_provider(model, **kwargs)
        elif provider == "hf_segmented_cache":
            return LLMProviderFactory._hf_segmented_cache_provider(model, **kwargs)
        elif provider == "hf_fast_weights":
            return LLMProviderFactory._hf_fast_weights_provider(model, **kwargs)
        elif provider == "glm":
            return LLMProviderFactory._glm_provider(model, **kwargs)
        elif provider == "minimax":
            return LLMProviderFactory._minimax_provider(model, **kwargs)
        elif provider == "qwen":
            return LLMProviderFactory._qwen_provider(model, **kwargs)
        elif provider == "deepseek":
            return LLMProviderFactory._deepseek_provider(model, **kwargs)
        elif provider == "together":
            return LLMProviderFactory._together_provider(model, **kwargs)
        elif provider == "replicate":
            return LLMProviderFactory._replicate_provider(model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def _openai_provider(
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """Create OpenAI provider"""
        try:
            import openai
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        except ImportError:
            raise ImportError("OpenAI not installed. Run: pip install openai")

        def provider(prompt: str) -> str:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content

        return provider

    @staticmethod
    def _anthropic_provider(
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """Create Anthropic provider"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic not installed. Run: pip install anthropic")

        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        def provider(prompt: str) -> str:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs,
            )
            return response.content[0].text

        return provider

    @staticmethod
    def _local_provider(
        model: str,
        base_url: str = "http://localhost:11434",
        **kwargs,
    ) -> Callable[[str], str]:
        """Create local provider - supports Ollama and OpenAI-compatible APIs"""
        # System prompt to get clean code
        system_prompt = "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code."
        request_timeout = float(kwargs.get("request_timeout", kwargs.get("timeout", 120)))

        def provider(prompt: str) -> str:
            import re
            import requests

            # Check if it's Ollama (default port 11434)
            if "11434" in base_url or "ollama" in base_url.lower():
                # Use Ollama API - prepend system prompt
                full_prompt = f"{system_prompt}\n\n{prompt}"
                response = requests.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", 0.3),
                            "num_predict": kwargs.get("max_tokens", 2048),
                            "num_ctx": kwargs.get("num_ctx", 4096),
                        }
                    },
                    timeout=request_timeout,
                )
                response.raise_for_status()
                result = response.json()
                code = result.get("response", "")
                code = re.sub(r'^```\w*\n?', '', code)
                code = re.sub(r'\n?```$', '', code)
                return code.strip()

            # vLLM / OpenAI-compatible via requests (faster, no openai import)
            if "8000" in base_url or "vllm" in base_url.lower():
                full_prompt = f"{system_prompt}\n\n{prompt}"
                response = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": full_prompt,
                        "max_tokens": int(kwargs.get("max_tokens", 2048)),
                        "temperature": float(kwargs.get("temperature", 0.3)),
                    },
                    timeout=request_timeout,
                )
                response.raise_for_status()
                result = response.json()
                code = result["choices"][0]["text"]
                code = re.sub(r'^```\w*\n?', '', code)
                code = re.sub(r'\n?```$', '', code)
                return code.strip()

            # Otherwise use OpenAI-compatible API
            try:
                import openai
                openai.api_key = "dummy"
                openai.base_url = base_url
            except ImportError:
                raise ImportError("OpenAI not installed. Run: pip install openai")

            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                timeout=request_timeout,
                **kwargs,
            )
            return response.choices[0].message.content

        return provider

    @staticmethod
    def _resolve_hf_device(preferred_device: Optional[str] = None) -> str:
        import torch

        if preferred_device:
            return preferred_device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_hf_local_model(
        model: str,
        *,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        use_cache: bool = True,
    ) -> tuple[object, object, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = LLMProviderFactory._resolve_hf_device(device)
        model_path = Path(model).expanduser()
        cache_key = f"{model_path.resolve() if model_path.exists() else model}|{resolved_device}"
        if use_cache:
            cached = LLMProviderFactory._hf_local_cache.get(cache_key)
            if cached is not None:
                return cached

        tokenizer = AutoTokenizer.from_pretrained(str(model_path if model_path.exists() else model), trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_path.exists() and (model_path / "adapter_config.json").exists():
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("hf_local adapter loading requires 'peft'. Run: pip install peft") from exc

            adapter_config = json.loads((model_path / "adapter_config.json").read_text(encoding="utf-8"))
            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-1.5B")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=trust_remote_code,
            )
            loaded_model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                str(model_path if model_path.exists() else model),
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=trust_remote_code,
            )

        loaded_model = loaded_model.to(resolved_device)
        loaded_model.eval()
        result = (loaded_model, tokenizer, resolved_device)
        if use_cache:
            LLMProviderFactory._hf_local_cache[cache_key] = result
        return result

    @staticmethod
    def _hf_local_provider(
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Callable[[str], dict]:
        """Create a local Hugging Face / LoRA provider."""
        import re
        import torch

        decode_backend = kwargs.pop("decode_backend", None)
        loaded_model, tokenizer, resolved_device = LLMProviderFactory._load_hf_local_model(
            model,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        do_sample = kwargs.get("do_sample", temperature > 0)
        decode_runtime = None
        if decode_backend == "segmented_kv":
            decode_runtime = LLMProviderFactory._build_segmented_decode_runtime(
                model_obj=loaded_model,
                tokenizer=tokenizer,
                device=resolved_device,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

        def provider(prompt: str) -> dict:
            started = time.perf_counter()
            if decode_runtime is not None:
                return decode_runtime.generate(prompt)
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(resolved_device)

            generate_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature

            with torch.no_grad():
                outputs = loaded_model.generate(**inputs, **generate_kwargs)

            prompt_tokens = int(inputs["input_ids"].shape[-1])
            generated_ids = outputs[0][prompt_tokens:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            elapsed = time.perf_counter() - started

            return {
                "text": text.strip(),
                "eval_count": int(generated_ids.shape[-1]),
                "prompt_eval_count": prompt_tokens,
                "total_duration": int(elapsed * 1_000_000_000),
                "device": resolved_device,
            }

        return provider

    @staticmethod
    def _hf_segmented_cache_provider(
        model: str,
        **kwargs,
    ) -> Callable[[str], dict]:
        """Create a Hugging Face provider that always uses segmented-cache decoding."""
        kwargs = dict(kwargs)
        kwargs["decode_backend"] = "segmented_kv"
        return LLMProviderFactory._hf_local_provider(model, **kwargs)

    @staticmethod
    def _hf_fast_weights_provider(
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Callable[[str], dict]:
        """Create a Hugging Face provider with task-local fast-weight updates."""
        from ncpu.self_optimizing.ncpu_adaptation_backend import (
            NCPUAdaptationBackend,
            NCPUAdaptationConfig,
        )
        from ncpu.self_optimizing.latent_descriptor_head import (
            LatentDescriptorGenerator,
            LatentDescriptorHeadConfig,
            load_latent_descriptor_head,
        )
        from ncpu.self_optimizing.state_patch_head import (
            StatePatchHead,
            StatePatchHeadConfig,
            load_state_patch_head,
        )
        from ncpu.self_optimizing.task_local_fast_weights import (
            HFTaskLocalFastWeightsProvider,
            TaskLocalFastWeightConfig,
        )

        decode_backend = kwargs.pop("decode_backend", None)
        config = kwargs.pop("fast_weights_config", None)
        if config is None:
            target_modules = kwargs.pop("fast_weights_target_modules", None)
            if target_modules is None:
                target_modules_tuple: tuple[str, ...] = TaskLocalFastWeightConfig().target_modules
            elif isinstance(target_modules, str):
                target_modules_tuple = tuple(
                    part.strip() for part in target_modules.split(",") if part.strip()
                )
            else:
                target_modules_tuple = tuple(target_modules)
            config = TaskLocalFastWeightConfig(
                rank=int(kwargs.pop("fast_weights_rank", 8)),
                learning_rate=float(kwargs.pop("fast_weights_learning_rate", 5e-3)),
                gradient_steps=int(kwargs.pop("fast_weights_gradient_steps", 1)),
                adapter_scale=float(kwargs.pop("fast_weights_adapter_scale", 1.0)),
                max_target_tokens=int(kwargs.pop("fast_weights_max_target_tokens", 256)),
                target_modules=target_modules_tuple,
                weight_decay=float(kwargs.pop("fast_weights_weight_decay", 0.0)),
            )

        adaptation_backend = kwargs.pop("ncpu_adaptation_backend", None)
        use_ncpu_adaptation = bool(
            kwargs.pop(
                "fast_weights_use_ncpu_adaptation",
                kwargs.pop("use_ncpu_adaptation", True),
            )
        )
        if adaptation_backend is None and use_ncpu_adaptation:
            adaptation_config = kwargs.pop("ncpu_adaptation_config", None)
            if adaptation_config is None:
                adaptation_config = NCPUAdaptationConfig(
                    compression_type=str(kwargs.pop("fast_weights_ncpu_compression_type", "top_k")),
                    top_k_ratio=float(kwargs.pop("fast_weights_ncpu_top_k_ratio", 0.1)),
                    quantization_bits=int(kwargs.pop("fast_weights_ncpu_quantization_bits", 8)),
                    gradient_clip=float(kwargs.pop("fast_weights_ncpu_gradient_clip", 1.0)),
                    min_learning_rate_scale=float(kwargs.pop("fast_weights_ncpu_min_lr_scale", 0.75)),
                    max_learning_rate_scale=float(kwargs.pop("fast_weights_ncpu_max_lr_scale", 1.5)),
                    max_gradient_steps=int(kwargs.pop("fast_weights_ncpu_max_gradient_steps", 3)),
                    verify_failure_boost=float(kwargs.pop("fast_weights_ncpu_verify_failure_boost", 1.2)),
            )
            adaptation_backend = NCPUAdaptationBackend(config=adaptation_config)

        latent_descriptor_head = kwargs.pop("latent_descriptor_head", None)
        latent_descriptor_head_path = kwargs.pop("latent_descriptor_head_path", None)
        if latent_descriptor_head is None and latent_descriptor_head_path:
            latent_descriptor_head_config = kwargs.pop("latent_descriptor_head_config", None)
            if latent_descriptor_head_config is None and any(
                key in kwargs
                for key in (
                    "latent_descriptor_head_numeric_feature_count",
                    "latent_descriptor_head_hash_bucket_count",
                    "latent_descriptor_head_hidden_dim",
                    "latent_descriptor_head_output_dim",
                )
            ):
                latent_descriptor_head_config = LatentDescriptorHeadConfig(
                    numeric_feature_count=int(
                        kwargs.pop(
                            "latent_descriptor_head_numeric_feature_count",
                            LatentDescriptorHeadConfig().numeric_feature_count,
                        )
                    ),
                    hash_bucket_count=int(kwargs.pop("latent_descriptor_head_hash_bucket_count", 16)),
                    hidden_dim=int(kwargs.pop("latent_descriptor_head_hidden_dim", 64)),
                    output_dim=int(kwargs.pop("latent_descriptor_head_output_dim", 16)),
                    dropout=float(kwargs.pop("latent_descriptor_head_dropout", 0.0)),
                )
            head = load_latent_descriptor_head(
                path=latent_descriptor_head_path,
                device=device or LLMProviderFactory._resolve_hf_device(None),
                config=latent_descriptor_head_config,
            )
            latent_descriptor_head = LatentDescriptorGenerator(
                head=head,
                device=device or LLMProviderFactory._resolve_hf_device(None),
                config=head.config,
            )

        state_patch_head = kwargs.pop("state_patch_head", None)
        state_patch_head_path = kwargs.pop("state_patch_head_path", None)
        if state_patch_head is None and state_patch_head_path:
            state_patch_head_config = kwargs.pop("state_patch_head_config", None)
            if state_patch_head_config is None and any(
                key in kwargs for key in ("state_patch_head_input_dim", "state_patch_head_hidden_dim", "state_patch_head_output_dim")
            ):
                state_patch_head_config = StatePatchHeadConfig(
                    input_dim=int(kwargs.pop("state_patch_head_input_dim", 16)),
                    hidden_dim=int(kwargs.pop("state_patch_head_hidden_dim", 64)),
                    output_dim=int(kwargs.pop("state_patch_head_output_dim", 16)),
                    dropout=float(kwargs.pop("state_patch_head_dropout", 0.0)),
                )
            state_patch_head = load_state_patch_head(
                path=state_patch_head_path,
                device=device or LLMProviderFactory._resolve_hf_device(None),
                config=state_patch_head_config,
            )
        elif state_patch_head is None and kwargs.pop("state_patch_head_enabled", False):
            state_patch_head_config = kwargs.pop("state_patch_head_config", None) or StatePatchHeadConfig(
                input_dim=int(kwargs.pop("state_patch_head_input_dim", 16)),
                hidden_dim=int(kwargs.pop("state_patch_head_hidden_dim", 64)),
                output_dim=int(kwargs.pop("state_patch_head_output_dim", 16)),
                dropout=float(kwargs.pop("state_patch_head_dropout", 0.0)),
            )
            state_patch_head = StatePatchHead(state_patch_head_config)
            state_patch_head = state_patch_head.to(device or LLMProviderFactory._resolve_hf_device(None))
            state_patch_head.eval()

        provider = HFTaskLocalFastWeightsProvider(
            model=model,
            config=config,
            adaptation_backend=adaptation_backend,
            latent_descriptor_head=latent_descriptor_head,
            state_patch_head=state_patch_head,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
            trust_remote_code=trust_remote_code,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        if decode_backend == "segmented_kv":
            provider.decode_runtime = LLMProviderFactory._build_segmented_decode_runtime(
                model_obj=provider.model,
                tokenizer=provider.tokenizer,
                device=provider.device,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )
        return provider

    @staticmethod
    def _build_segmented_decode_runtime(
        *,
        model_obj: object,
        tokenizer: object,
        device: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        repetition_penalty: float,
        **kwargs,
    ) -> object:
        from ncpu.self_optimizing.descriptor_decode_runtime import (
            DescriptorDecodeConfig,
            DescriptorDecodeRuntime,
        )
        from ncpu.self_optimizing.recurrent_commit_policy import RecurrentCommitPolicyConfig
        from ncpu.self_optimizing.segmented_kv_cache import SegmentedKVCacheConfig

        cache_config = SegmentedKVCacheConfig(
            recent_window_tokens=int(kwargs.pop("segmented_cache_recent_window_tokens", 256)),
            commit_segment_tokens=int(kwargs.pop("segmented_cache_commit_segment_tokens", 128)),
            descriptor_tokens_per_segment=int(kwargs.pop("segmented_cache_descriptor_tokens_per_segment", 4)),
            max_memory_segments=int(kwargs.pop("segmented_cache_max_memory_segments", 16)),
            min_prompt_tokens_for_compression=int(kwargs.pop("segmented_cache_min_prompt_tokens_for_compression", 384)),
            min_tokens_to_commit=int(kwargs.pop("segmented_cache_min_tokens_to_commit", 64)),
        )
        commit_policy = RecurrentCommitPolicyConfig(
            recent_window_tokens=cache_config.recent_window_tokens,
            commit_segment_tokens=cache_config.commit_segment_tokens,
            min_tokens_to_commit=cache_config.min_tokens_to_commit,
        )
        decode_config = DescriptorDecodeConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            cache=cache_config,
            commit_policy=commit_policy,
        )
        return DescriptorDecodeRuntime(
            model=model_obj,
            tokenizer=tokenizer,
            device=device,
            config=decode_config,
        )

    @staticmethod
    def _glm_provider(
        model: str = "sonnet",
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create GLM provider using MCP server.

        GLM is 10x cheaper than OpenAI and supports:
        - haiku (fastest, cheapest)
        - sonnet (balanced)
        - opus (highest quality)
        """
        # Map model names
        model_map = {
            "glm-4": "sonnet",
            "glm-4-flash": "haiku",
            "glm-4-plus": "opus",
            "default": "sonnet",
        }
        glm_model = model_map.get(model.lower(), model)

        def provider(prompt: str) -> str:
            try:
                from .glm_client import glm_ask
                return glm_ask(prompt, model=glm_model)
            except ImportError:
                raise ImportError("GLM MCP not available. Configure glm-agent MCP server.")

        return provider

    @staticmethod
    def _minimax_provider(
        model: str = "MiniMax-M2.5",
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """Create MiniMax provider using ~/.minimax.json config"""
        import json
        from pathlib import Path

        # Load from ~/.minimax.json
        config_path = Path.home() / ".minimax.json"
        if not config_path.exists():
            raise ImportError("MiniMax not configured. Create ~/.minimax.json with apiKey and apiUrl")

        with open(config_path) as f:
            config = json.load(f)

        api_key = config.get("apiKey", "")
        # Override URL - use platform.minimax.io
        api_url = "https://api.minimax.io/v1/text/chatcompletion_v2"

        if not api_key:
            raise ValueError("MiniMax apiKey not found in ~/.minimax.json")

        def provider(prompt: str) -> str:
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": 2048,
            }
            # MiniMax endpoint is the full URL
            response = requests.post(
                api_url,
                headers=headers,
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            # Handle both regular content and reasoning content
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            # Return reasoning if content is empty (reasoning models)
            return content or reasoning

        return provider

    @staticmethod
    def _qwen_provider(
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create Qwen provider - Alibaba's Qwen models.

        Supports:
        - qwen-turbo (fast)
        - qwen-plus (balanced)
        - qwen-max (highest quality)
        - qwen-coder (code-specialized)
        """
        import requests

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ImportError("Qwen not configured. Set DASHSCOPE_API_KEY environment variable")

        # Map model names for DashScope
        model_map = {
            "qwen-2.5": "qwen-turbo",
            "qwen-2.5-coder": "qwen-coder-plus",
            "qwen-max": "qwen-max",
            "default": "qwen-turbo",
        }
        qwen_model = model_map.get(model.lower(), model)

        def provider(prompt: str) -> str:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": qwen_model,
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code."
                        },
                        {"role": "user", "content": prompt}
                    ]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": kwargs.get("max_tokens", 2048),
                }
            }
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["output"]["choices"][0]["message"]["content"]

        return provider

    @staticmethod
    def _deepseek_provider(
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create DeepSeek provider - DeepSeek's models.

        Supports:
        - deepseek-chat (general chat)
        - deepseek-coder (code-specialized)
        """
        import requests

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ImportError("DeepSeek not configured. Set DEEPSEEK_API_KEY environment variable")

        # Map model names
        model_map = {
            "deepseek-coder": "deepseek-coder",
            "default": "deepseek-chat",
        }
        deepseek_model = model_map.get(model.lower(), model)

        def provider(prompt: str) -> str:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": deepseek_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        return provider

    @staticmethod
    def _together_provider(
        model: str = "meta-llama/Llama-3-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create Together AI provider - aggregates many models.

        Supports thousands of models including:
        - Meta Llama models
        - Mistral models
        - Qwen models
        - Code models
        """
        import requests

        api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ImportError("Together AI not configured. Set TOGETHER_API_KEY environment variable")

        def provider(prompt: str) -> str:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 2048),
            }
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        return provider

    @staticmethod
    def _replicate_provider(
        model: str = "meta/llama-3-8b-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> Callable[[str], str]:
        """
        Create Replicate provider - run various models via API.

        Supports many models including:
        - Llama models
        - Mistral
        - Phi
        - And many more from the Replicate registry
        """
        import requests

        api_key = api_key or os.environ.get("REPLICATE_API_KEY")
        if not api_key:
            raise ImportError("Replicate not configured. Set REPLICATE_API_KEY environment variable")

        # For Replicate, we need to start a prediction
        def provider(prompt: str) -> str:
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
            # Format model for Replicate (owner/name:version)
            model_version = kwargs.get("version", "latest")

            # First, create the prediction
            data = {
                "version": model_version,
                "input": {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": kwargs.get("max_tokens", 2048),
                    "system_prompt": "You are a code assistant. Output ONLY raw Python code with no markdown formatting, no backticks, no explanations. Just the code.",
                }
            }
            response = requests.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers=headers,
                json=data,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Poll for completion
            prediction_url = result["urls"]["get"]
            while True:
                status_response = requests.get(prediction_url, headers=headers)
                status_response.raise_for_status()
                status = status_response.json()

                if status["status"] == "succeeded":
                    output = status["output"]
                    # Handle various output formats
                    if isinstance(output, list):
                        return output[0]
                    return str(output)
                elif status["status"] == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {status.get('error')}")

                time.sleep(1)

        return provider


# Example usage:
"""
# Using OpenAI
provider = LLMProviderFactory.create_provider(
    provider="openai",
    model="gpt-4",
    temperature=0.3
)
result = provider("Write a fibonacci function")

# Using Anthropic
provider = LLMProviderFactory.create_provider(
    provider="anthropic",
    model="claude-3-opus-20240229"
)
result = provider("Write a fibonacci function")

# Using local model (like llama.cpp server)
provider = LLMProviderFactory.create_provider(
    provider="local",
    model="llama-2-7b",
    base_url="http://localhost:8080/v1"
)
result = provider("Write a fibonacci function")
"""
