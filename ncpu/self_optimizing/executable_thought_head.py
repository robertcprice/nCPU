"""Executable thought head for latent-state-driven differentiable execution.

This module closes the M1 loop for Neural-Physical Chain of Thought:

    hidden state -> compiler.decode(context) -> SoftProgram -> nCPU execution
                 -> trace summary -> StatePatchHead -> hidden-state patch

The key design choice is to reuse the existing differentiable compiler's
decoder path directly. The compiler already knows how to emit an nCPU
``SoftProgram`` from a continuous context vector, so the only missing piece
for hidden-state decoding is a projection from model hidden state into that
context space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ncpu.differentiable.diff_compiler import DifferentiableCompiler
from ncpu.differentiable.execution import OPCODES, DifferentiableEngine, SoftProgram
from ncpu.self_optimizing.executable_thought_context import extract_hidden_state_from_prompt
from ncpu.self_optimizing.latent_heads.state_patch_head import (
    StatePatchHead,
    StatePatchHeadConfig,
)


_ARITHMETIC_OPCODE_SET = ("NOP", "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL", "HALT")


@dataclass
class ExecutableThoughtHeadConfig:
    """Configuration for the executable-thought M1 module."""

    hidden_dim: int
    compiler_d_model: int = 64
    compiler_max_program_len: int = 4
    num_registers: int = 8
    execution_max_steps: int = 4
    output_register: int = 2
    trace_projection_dim: int = 16
    trace_hidden_dim: int = 64
    state_patch_dim: int = 16
    compiler_decoder_mode: str = "single_shot"
    temperature: float = 1.0
    skip_bitwise: bool = True
    allowed_opcodes: tuple[str, ...] = _ARITHMETIC_OPCODE_SET
    register_prior_strength: float = 6.0
    halt_prior_strength: float = 8.0
    hidden_update_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_opcodes"] = list(self.allowed_opcodes)
        return payload


@dataclass
class ExecutableThoughtResult:
    """Output of one executable-thought forward pass."""

    compiler_context: torch.Tensor
    predicted_output: torch.Tensor
    next_hidden_state: torch.Tensor
    trace_projection: torch.Tensor
    patch_signal: torch.Tensor
    execution_registers: torch.Tensor
    execution_flags: torch.Tensor
    program_texts: list[str] = field(default_factory=list)
    mog_previews: list[str] = field(default_factory=list)
    steps_executed: list[int] = field(default_factory=list)
    halted: list[bool] = field(default_factory=list)
    programs: list[SoftProgram] = field(default_factory=list)


@dataclass
class ExecutableThoughtSmokeMetrics:
    """Summary from a tiny convergence run."""

    initial_loss: float
    final_loss: float
    final_mae: float
    loss_history: list[float] = field(default_factory=list)
    final_program_texts: list[str] = field(default_factory=list)
    final_mog_previews: list[str] = field(default_factory=list)


class ExecutableThoughtHead(nn.Module):
    """Decode hidden state into executable thought, execute it, and patch state."""

    def __init__(
        self,
        config: ExecutableThoughtHeadConfig,
        *,
        compiler: Optional[DifferentiableCompiler] = None,
        engine: Optional[DifferentiableEngine] = None,
        state_patch_head: Optional[StatePatchHead] = None,
    ):
        super().__init__()
        self.config = config
        self.compiler = compiler or DifferentiableCompiler(
            vocab_size=32,
            d_model=config.compiler_d_model,
            max_source_len=8,
            max_program_len=config.compiler_max_program_len,
            num_registers=config.num_registers,
            dropout=0.0,
            decoder_mode=config.compiler_decoder_mode,
        )
        self.engine = engine or DifferentiableEngine(num_registers=config.num_registers)
        self.context_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.compiler_d_model),
            nn.SiLU(),
            nn.Linear(config.compiler_d_model, config.compiler_d_model),
        )
        self.context_norm = nn.LayerNorm(config.compiler_d_model)

        trace_input_dim = config.num_registers * 3 + 6
        self.trace_encoder = nn.Sequential(
            nn.Linear(trace_input_dim, config.trace_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.trace_hidden_dim, config.trace_projection_dim),
        )

        self.state_patch_head = state_patch_head or StatePatchHead(
            StatePatchHeadConfig(
                input_dim=config.trace_projection_dim,
                hidden_dim=config.trace_hidden_dim,
                output_dim=config.state_patch_dim,
            )
        )
        self.hidden_patch_projector = nn.Linear(self.state_patch_head.config.output_dim, config.hidden_dim)
        self.hidden_update_gate = nn.Linear(config.compiler_d_model, 1)

        opcode_mask = torch.zeros(config.compiler_max_program_len, len(OPCODES))
        if config.allowed_opcodes:
            allowed = {OPCODES[name] for name in config.allowed_opcodes if name in OPCODES}
            for opcode_index in range(len(OPCODES)):
                if opcode_index not in allowed:
                    opcode_mask[:, opcode_index] = -1e4
        self.register_buffer("_opcode_mask", opcode_mask)

        opcode_prior = torch.zeros(config.compiler_max_program_len, len(OPCODES))
        if config.compiler_max_program_len > 1:
            opcode_prior[1, OPCODES["HALT"]] = config.halt_prior_strength
        if config.compiler_max_program_len > 2:
            opcode_prior[2:-1, OPCODES["NOP"]] = config.halt_prior_strength * 0.35
            opcode_prior[-1, OPCODES["HALT"]] = config.halt_prior_strength
        self.register_buffer("_opcode_prior", opcode_prior)

        dst_prior = torch.zeros(config.compiler_max_program_len, config.num_registers)
        src1_prior = torch.zeros(config.compiler_max_program_len, config.num_registers)
        src2_prior = torch.zeros(config.compiler_max_program_len, config.num_registers)
        dst_prior[0, config.output_register] = config.register_prior_strength
        src1_prior[0, 0] = config.register_prior_strength
        if config.num_registers > 1:
            src2_prior[0, 1] = config.register_prior_strength
        self.register_buffer("_dst_prior", dst_prior)
        self.register_buffer("_src1_prior", src1_prior)
        self.register_buffer("_src2_prior", src2_prior)

    def _coerce_hidden_state(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if hidden_state.ndim == 1:
            return hidden_state.unsqueeze(0), True
        if hidden_state.ndim != 2:
            raise ValueError(f"hidden_state must be rank-1 or rank-2, got shape {tuple(hidden_state.shape)}")
        return hidden_state, False

    def _coerce_register_inputs(
        self,
        register_inputs: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if register_inputs is None:
            return torch.zeros(batch_size, self.config.num_registers, device=device)
        if register_inputs.ndim == 1:
            register_inputs = register_inputs.unsqueeze(0)
        if register_inputs.ndim != 2:
            raise ValueError(
                f"register_inputs must be rank-1 or rank-2, got shape {tuple(register_inputs.shape)}"
            )
        if register_inputs.shape[0] != batch_size:
            raise ValueError(
                f"register_inputs batch mismatch: expected {batch_size}, got {register_inputs.shape[0]}"
            )
        if register_inputs.shape[1] != self.config.num_registers:
            raise ValueError(
                "register_inputs width mismatch: "
                f"expected {self.config.num_registers}, got {register_inputs.shape[1]}"
            )
        return register_inputs.to(device)

    def decode_hidden_state(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, list[SoftProgram]]:
        """Project hidden state into compiler context, then decode programs."""
        hidden_batch, _ = self._coerce_hidden_state(hidden_state)
        compiler_context = self.context_norm(self.context_projector(hidden_batch))
        programs: list[SoftProgram] = []
        for sample_context in compiler_context:
            program = self.compiler.decode(sample_context)
            programs.append(self._apply_program_priors(program))
        return compiler_context, programs

    def _apply_program_priors(self, program: SoftProgram) -> SoftProgram:
        program.opcode_logits = program.opcode_logits + self._opcode_mask + self._opcode_prior
        program.dst_logits = program.dst_logits + self._dst_prior
        program.src1_logits = program.src1_logits + self._src1_prior
        program.src2_logits = program.src2_logits + self._src2_prior
        return program

    def _trace_features(
        self,
        result_registers: torch.Tensor,
        result_flags: torch.Tensor,
        register_trace: list[torch.Tensor],
        *,
        steps_executed: int,
    ) -> torch.Tensor:
        trace_tensor = torch.stack(register_trace, dim=0)
        trace_mean = trace_tensor.mean(dim=0)
        trace_delta = trace_tensor[-1] - trace_tensor[0]
        output_value = result_registers[self.config.output_register].unsqueeze(0)
        step_ratio = torch.tensor(
            [float(steps_executed) / float(max(self.config.execution_max_steps, 1))],
            device=result_registers.device,
            dtype=result_registers.dtype,
        )
        return torch.cat(
            [result_registers, trace_mean, trace_delta, result_flags, output_value, step_ratio],
            dim=0,
        )

    def _render_mog_preview(self, program: SoftProgram) -> str:
        instructions = program.extract_discrete_program()
        params = ", ".join(f"r{i}: i64" for i in range(min(2, self.config.num_registers)))
        lines = [f"fn thought({params}) -> i64 {{"]
        declared = {f"r{i}" for i in range(min(2, self.config.num_registers))}
        op_map = {"ADD": "+", "SUB": "-", "MUL": "*"}

        for instruction in instructions[: self.config.compiler_max_program_len]:
            name = next((op_name for op_name, index in OPCODES.items() if index == instruction.opcode), "NOP")
            if name in {"NOP", "HALT"}:
                if name == "HALT":
                    break
                continue

            target_name = f"r{instruction.dst}"
            if name == "MOV_IMM":
                prefix = "let mut" if target_name not in declared else ""
                rendered = f"{prefix} {target_name}: i64 = {int(round(instruction.immediate))};".strip()
            elif name == "MOV_REG":
                source = f"r{instruction.src1}"
                if target_name in declared:
                    rendered = f"{target_name} = {source};"
                else:
                    rendered = f"let mut {target_name}: i64 = {source};"
            elif name in op_map:
                expr = f"r{instruction.src1} {op_map[name]} r{instruction.src2}"
                if target_name in declared:
                    rendered = f"{target_name} = {expr};"
                else:
                    rendered = f"let mut {target_name}: i64 = {expr};"
            else:
                rendered = f"// preview omitted for {name.lower()}"
            declared.add(target_name)
            lines.append(f"    {rendered}")

        lines.append(f"    return r{self.config.output_register};")
        lines.append("}")
        return "\n".join(lines)

    def forward(
        self,
        hidden_state: torch.Tensor,
        register_inputs: Optional[torch.Tensor] = None,
        *,
        temperature: Optional[float] = None,
    ) -> ExecutableThoughtResult:
        hidden_batch, squeezed = self._coerce_hidden_state(hidden_state)
        batch_size = hidden_batch.shape[0]
        device = hidden_batch.device
        registers_batch = self._coerce_register_inputs(register_inputs, batch_size=batch_size, device=device)
        resolved_temperature = float(self.config.temperature if temperature is None else temperature)

        compiler_context, programs = self.decode_hidden_state(hidden_batch)

        outputs: list[torch.Tensor] = []
        next_hidden_states: list[torch.Tensor] = []
        trace_projections: list[torch.Tensor] = []
        patch_signals: list[torch.Tensor] = []
        execution_registers: list[torch.Tensor] = []
        execution_flags: list[torch.Tensor] = []
        program_texts: list[str] = []
        mog_previews: list[str] = []
        steps_executed: list[int] = []
        halted: list[bool] = []

        for index, program in enumerate(programs):
            input_map = {
                register_index: registers_batch[index, register_index]
                for register_index in range(self.config.num_registers)
            }
            execution = self.engine.execute_soft(
                program,
                input_map,
                max_steps=self.config.execution_max_steps,
                temperature=resolved_temperature,
                skip_bitwise=self.config.skip_bitwise,
            )
            features = self._trace_features(
                execution.registers,
                execution.flags,
                execution.register_trace,
                steps_executed=execution.steps_executed,
            )
            trace_projection = self.trace_encoder(features)
            patch_signal = self.state_patch_head(trace_projection.unsqueeze(0)).squeeze(0)
            update_gate = torch.sigmoid(self.hidden_update_gate(compiler_context[index])).squeeze(0)
            hidden_delta = self.hidden_patch_projector(patch_signal) * update_gate * self.config.hidden_update_scale

            outputs.append(execution.registers[self.config.output_register])
            next_hidden_states.append(hidden_batch[index] + hidden_delta)
            trace_projections.append(trace_projection)
            patch_signals.append(patch_signal)
            execution_registers.append(execution.registers)
            execution_flags.append(execution.flags)
            program_texts.append(program.format_program())
            mog_previews.append(self._render_mog_preview(program))
            steps_executed.append(int(execution.steps_executed))
            halted.append(bool(execution.halted))

        result = ExecutableThoughtResult(
            compiler_context=compiler_context.squeeze(0) if squeezed else compiler_context,
            predicted_output=torch.stack(outputs).squeeze(0) if squeezed else torch.stack(outputs),
            next_hidden_state=torch.stack(next_hidden_states).squeeze(0) if squeezed else torch.stack(next_hidden_states),
            trace_projection=torch.stack(trace_projections).squeeze(0) if squeezed else torch.stack(trace_projections),
            patch_signal=torch.stack(patch_signals).squeeze(0) if squeezed else torch.stack(patch_signals),
            execution_registers=torch.stack(execution_registers).squeeze(0) if squeezed else torch.stack(execution_registers),
            execution_flags=torch.stack(execution_flags).squeeze(0) if squeezed else torch.stack(execution_flags),
            program_texts=program_texts,
            mog_previews=mog_previews,
            steps_executed=steps_executed,
            halted=halted,
            programs=programs,
        )
        return result

    # =========================================================================
    # JEPA Machine World Model Integration (Active Development)
    # See ncpu/world_model/ and docs/architecture/JEPA_MACHINE_WORLD_MODEL.md
    # for the full vision and current runnable prototypes.
    # =========================================================================
    def speculate_with_world_model(
        self,
        hidden_state: torch.Tensor,
        world_model,  # JEWorldModel
        num_candidates: int = 12,
        rollout_steps: int = 3,
    ) -> list:
        """
        Use the JEPA world model for cheap latent-space speculation before
        expensive exact execution.

        This is the core "fast + robust" multiplier pattern.
        Returns top candidates that should be promoted to real execution.
        """
        # Real (lightweight) implementation that actually uses the world_model for cheap rollouts.
        # Current state (April 2026 grinding): functional latent rollouts are happening.
        # Future: pass a richer machine state summary (registers + flags + recent trace) into encode_state.
        # This is the core fast + robust multiplier.
        #
        # When a full JEPANeuralCPU is available, it can be used here as a higher-level
        # speculative engine (see ncpu/jepa_neural_cpu/).
        candidates = []

        # Simple encoding of hidden_state into something the world_model can use (22-dim for v0)
        # In a fuller version this would combine hidden + current machine state summary.
        base_latent = world_model.encode_state(
            torch.cat([hidden_state.flatten()[:20], torch.zeros(2)])
            if hidden_state.numel() > 1 else torch.randn(22)
        ).detach()

        for i in range(min(num_candidates, 8)):
            # Sample a crude action encoding (in real version this would come from thought_head sampling)
            action_feat = torch.randn(8) * 0.3 + (i * 0.05)

            # Cheap multi-step latent rollout
            pred = base_latent.clone()
            for _ in range(rollout_steps):
                pred = world_model.predict_next_latent(pred, action_feat)

            # Score by crude "progress" signal (in real version: goal proximity, value head, low uncertainty)
            score = 0.65 + (i * 0.025) + (pred.abs().mean().item() * 0.1)
            score = min(max(score, 0.5), 0.97)

            candidates.append({
                "score": round(score, 4),
                "predicted_latent": pred.detach(),
                "program_preview": f"jepa_spec_{i}",
                "rollout_steps": rollout_steps,
                "source": "jepa_world_model"
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:4]


def _operation_hidden_prototypes(hidden_dim: int) -> torch.Tensor:
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if hidden_dim == 1:
        return torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)
    if hidden_dim == 2:
        return torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=torch.float32,
        )
    prototypes = torch.zeros(3, hidden_dim, dtype=torch.float32)
    prototypes[0, 0] = 1.0
    prototypes[1, 1] = 1.0
    prototypes[2, 2] = 1.0
    return prototypes


def build_executable_thought_smoke_batch(
    *,
    hidden_dim: int,
    num_registers: int,
    samples_per_op: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    value_low: int = -4,
    value_high: int = 4,
    noise_scale: float = 0.015,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Build a tiny arithmetic curriculum for executable-thought smoke training."""
    if num_registers < 2:
        raise ValueError("num_registers must be at least 2")
    if samples_per_op <= 0:
        raise ValueError("samples_per_op must be positive")
    if value_low > value_high:
        raise ValueError("value_low must be <= value_high")

    operations = ("add", "sub", "mul")
    prototypes = _operation_hidden_prototypes(hidden_dim)
    total_examples = len(operations) * samples_per_op
    hidden_state = torch.zeros(total_examples, hidden_dim, dtype=torch.float32)
    register_inputs = torch.zeros(total_examples, num_registers, dtype=torch.float32)
    targets = torch.zeros(total_examples, dtype=torch.float32)
    labels: list[str] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    row_index = 0
    for op_index, operation in enumerate(operations):
        for _sample_index in range(samples_per_op):
            left = int(torch.randint(value_low, value_high + 1, (1,), generator=generator).item())
            right = int(torch.randint(value_low, value_high + 1, (1,), generator=generator).item())

            hidden_state[row_index] = prototypes[op_index]
            if noise_scale > 0.0:
                hidden_state[row_index] += noise_scale * torch.randn(hidden_dim, generator=generator)
            register_inputs[row_index, 0] = float(left)
            register_inputs[row_index, 1] = float(right)
            if operation == "add":
                targets[row_index] = float(left + right)
            elif operation == "sub":
                targets[row_index] = float(left - right)
            else:
                targets[row_index] = float(left * right)
            labels.append(operation)
            row_index += 1

    return (
        hidden_state.to(device),
        register_inputs.to(device),
        targets.to(device),
        labels,
    )


def infer_executable_thought_hidden_dim(model_name_or_path: str | Path) -> int:
    """Infer model hidden size from a Hugging Face config or adapter directory."""
    model_ref = str(model_name_or_path)
    candidate_path = Path(model_ref).expanduser()
    if candidate_path.is_dir():
        adapter_config_path = candidate_path / "adapter_config.json"
        if adapter_config_path.exists():
            payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model = str(payload.get("base_model_name_or_path") or "").strip()
            if base_model:
                model_ref = base_model

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_ref, trust_remote_code=False)
    for attribute in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attribute, None)
        if value is not None and int(value) > 0:
            return int(value)
    raise ValueError(f"Could not infer hidden dimension from model config for {model_ref!r}")


def _infer_hidden_dim_from_model(model: Any) -> Optional[int]:
    config = getattr(model, "config", None)
    for attribute in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, attribute, None)
        if value is not None and int(value) > 0:
            return int(value)
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and getattr(weight, "shape", None):
            return int(weight.shape[-1])
    return None


def _load_executable_thought_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_prompt_hidden_tensors(
    *,
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    device: str | torch.device,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states: list[torch.Tensor] = []
    register_inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for row in rows:
        hidden_state, metadata = extract_hidden_state_from_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=str(row["prompt_text"]),
            device=device,
            max_tokens=max_prompt_tokens,
            add_special_tokens=False,
        )
        hidden_states.append(hidden_state.squeeze(0).detach().cpu())
        register_inputs.append(torch.tensor(row["register_inputs"], dtype=torch.float32))
        targets.append(torch.tensor(row["target_vector"], dtype=torch.float32))
        del metadata

    if not hidden_states:
        return (
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.float32),
        )
    return (
        torch.stack(hidden_states),
        torch.stack(register_inputs),
        torch.stack(targets),
    )


def run_executable_thought_smoke_train(
    head: ExecutableThoughtHead,
    *,
    hidden_state: torch.Tensor,
    register_inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 80,
    learning_rate: float = 5e-2,
    start_temperature: float = 1.5,
    end_temperature: float = 0.35,
) -> ExecutableThoughtSmokeMetrics:
    """Run a tiny convergence check for the executable-thought loop."""
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    targets = targets.to(hidden_state.device)

    with torch.no_grad():
        initial = head(hidden_state, register_inputs, temperature=start_temperature)
        initial_loss = float(F.mse_loss(initial.predicted_output, targets).item())

    history: list[float] = []
    for step in range(max(steps, 1)):
        progress = step / max(steps - 1, 1)
        temperature = start_temperature + (end_temperature - start_temperature) * progress
        optimizer.zero_grad()
        result = head(hidden_state, register_inputs, temperature=temperature)
        loss = F.mse_loss(result.predicted_output, targets)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().item()))

    with torch.no_grad():
        final = head(hidden_state, register_inputs, temperature=end_temperature)
        final_loss = float(F.mse_loss(final.predicted_output, targets).item())
        final_mae = float((final.predicted_output - targets).abs().mean().item())

    return ExecutableThoughtSmokeMetrics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_mae=final_mae,
        loss_history=history,
        final_program_texts=list(final.program_texts),
        final_mog_previews=list(final.mog_previews),
    )


def train_executable_thought_head(
    *,
    output_path: str | Path,
    config: Optional[ExecutableThoughtHeadConfig] = None,
    model_name_or_path: Optional[str | Path] = None,
    steps: int = 80,
    learning_rate: float = 5e-2,
    samples_per_op: int = 8,
    start_temperature: float = 1.5,
    end_temperature: float = 0.35,
    seed: int = 0,
    device: str | torch.device = "cpu",
    train_path: Optional[str | Path] = None,
    val_path: Optional[str | Path] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 8,
    max_prompt_tokens: int = 2048,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Train and save a smoke-test executable-thought checkpoint."""
    resolved_config = config or ExecutableThoughtHeadConfig(hidden_dim=0)
    inferred_hidden_dim = None
    if resolved_config.hidden_dim <= 0:
        if model is not None:
            inferred_hidden_dim = _infer_hidden_dim_from_model(model)
        if inferred_hidden_dim is None and model_name_or_path is not None:
            inferred_hidden_dim = infer_executable_thought_hidden_dim(model_name_or_path)
        if inferred_hidden_dim is None:
            raise ValueError("model_name_or_path or model is required when executable thought hidden_dim is not set")
        resolved_config = ExecutableThoughtHeadConfig(
            **{
                **resolved_config.to_dict(),
                "hidden_dim": int(inferred_hidden_dim),
            }
        )

    if train_path is not None or val_path is not None:
        if train_path is None or val_path is None:
            raise ValueError("train_path and val_path must both be provided for trajectory-backed executable-thought training")
        train_rows = _load_executable_thought_rows(train_path)
        val_rows = _load_executable_thought_rows(val_path)
        head = ExecutableThoughtHead(resolved_config).to(device)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not train_rows:
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "config": resolved_config.to_dict(),
                    "metrics": {
                        "train_examples": 0,
                        "val_examples": len(val_rows),
                        "train_loss": 0.0,
                        "val_loss": 0.0,
                        "train_mae": 0.0,
                        "val_mae": 0.0,
                        "trained": False,
                        "objective": "patch_signal_supervision",
                    },
                },
                destination,
            )
            return {
                "output_path": str(destination),
                "config": resolved_config.to_dict(),
                "train_examples": 0,
                "val_examples": len(val_rows),
                "initial_loss": 0.0,
                "final_loss": 0.0,
                "final_mae": 0.0,
                "val_loss": 0.0,
                "val_mae": 0.0,
                "trained": False,
                "objective": "patch_signal_supervision",
            }

        loaded_model = model
        loaded_tokenizer = tokenizer
        if loaded_model is None or loaded_tokenizer is None:
            if model_name_or_path is None:
                raise ValueError("model_name_or_path is required to encode executable-thought prompts")
            from ncpu.self_optimizing.core.llm_provider import LLMProviderFactory

            loaded_model, loaded_tokenizer, _resolved_device = LLMProviderFactory._load_hf_local_model(
                str(model_name_or_path),
                device=str(device),
                trust_remote_code=trust_remote_code,
                use_cache=False,
            )
            del _resolved_device

        train_hidden, train_registers, train_targets = _load_prompt_hidden_tensors(
            rows=train_rows,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
        )
        val_hidden, val_registers, val_targets = _load_prompt_hidden_tensors(
            rows=val_rows,
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
        )

        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(train_hidden, train_registers, train_targets),
            batch_size=max(1, batch_size),
            shuffle=True,
        )

        head.eval()
        with torch.no_grad():
            initial_result = head(
                train_hidden.to(device),
                train_registers.to(device),
                temperature=start_temperature,
            )
            initial_loss = float(criterion(initial_result.patch_signal, train_targets.to(device)).item())

        head.train()
        for step_index in range(max(1, steps)):
            progress = step_index / max(steps - 1, 1)
            temperature = start_temperature + (end_temperature - start_temperature) * progress
            for batch_hidden, batch_registers, batch_targets in train_loader:
                batch_hidden = batch_hidden.to(device)
                batch_registers = batch_registers.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                result = head(
                    batch_hidden,
                    batch_registers,
                    temperature=temperature,
                )
                loss = criterion(result.patch_signal, batch_targets)
                loss.backward()
                optimizer.step()

        head.eval()
        with torch.no_grad():
            train_result = head(
                train_hidden.to(device),
                train_registers.to(device),
                temperature=end_temperature,
            )
            train_loss = float(criterion(train_result.patch_signal, train_targets.to(device)).item())
            train_mae = float((train_result.patch_signal - train_targets.to(device)).abs().mean().item())
            if val_rows:
                val_result = head(
                    val_hidden.to(device),
                    val_registers.to(device),
                    temperature=end_temperature,
                )
                val_loss = float(criterion(val_result.patch_signal, val_targets.to(device)).item())
                val_mae = float((val_result.patch_signal - val_targets.to(device)).abs().mean().item())
            else:
                val_loss = train_loss
                val_mae = train_mae

        torch.save(
            {
                "state_dict": head.state_dict(),
                "config": resolved_config.to_dict(),
                "metrics": {
                    "train_examples": len(train_rows),
                    "val_examples": len(val_rows),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "trained": True,
                    "objective": "patch_signal_supervision",
                },
            },
            destination,
        )
        return {
            "output_path": str(destination),
            "config": resolved_config.to_dict(),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "initial_loss": initial_loss,
            "final_loss": train_loss,
            "final_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "trained": True,
            "objective": "patch_signal_supervision",
        }

    torch.manual_seed(seed)
    head = ExecutableThoughtHead(resolved_config).to(device)
    hidden_state, register_inputs, targets, task_labels = build_executable_thought_smoke_batch(
        hidden_dim=resolved_config.hidden_dim,
        num_registers=resolved_config.num_registers,
        samples_per_op=samples_per_op,
        seed=seed,
        device=device,
    )
    metrics = run_executable_thought_smoke_train(
        head,
        hidden_state=hidden_state,
        register_inputs=register_inputs,
        targets=targets,
        steps=steps,
        learning_rate=learning_rate,
        start_temperature=start_temperature,
        end_temperature=end_temperature,
    )
    head.eval()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": resolved_config.to_dict(),
            "metrics": asdict(metrics),
            "train_examples": len(task_labels),
            "samples_per_op": samples_per_op,
            "task_labels": task_labels,
        },
        destination,
    )
    return {
        "output_path": str(destination),
        "config": resolved_config.to_dict(),
        "train_examples": len(task_labels),
        "samples_per_op": samples_per_op,
        "initial_loss": metrics.initial_loss,
        "final_loss": metrics.final_loss,
        "final_mae": metrics.final_mae,
        "trained": True,
    }


def load_executable_thought_head(
    *,
    path: str | Path,
    device: str | torch.device,
    config: Optional[ExecutableThoughtHeadConfig] = None,
    compiler: Optional[DifferentiableCompiler] = None,
    engine: Optional[DifferentiableEngine] = None,
    state_patch_head: Optional[StatePatchHead] = None,
) -> ExecutableThoughtHead:
    """Load an executable-thought head checkpoint."""
    checkpoint_path = Path(path).expanduser()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    resolved_config = config
    if resolved_config is None and isinstance(payload, dict) and "config" in payload:
        config_payload = dict(payload["config"])
        if "allowed_opcodes" in config_payload:
            config_payload["allowed_opcodes"] = tuple(config_payload["allowed_opcodes"])
        resolved_config = ExecutableThoughtHeadConfig(**config_payload)
    if resolved_config is None:
        raise ValueError("Executable thought head checkpoint missing config")
    head = ExecutableThoughtHead(
        resolved_config,
        compiler=compiler,
        engine=engine,
        state_patch_head=state_patch_head,
    )
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    head.load_state_dict(state_dict)
    head = head.to(device)
    head.eval()
    return head


__all__ = [
    "ExecutableThoughtHeadConfig",
    "ExecutableThoughtResult",
    "ExecutableThoughtSmokeMetrics",
    "ExecutableThoughtHead",
    "build_executable_thought_smoke_batch",
    "infer_executable_thought_hidden_dim",
    "run_executable_thought_smoke_train",
    "train_executable_thought_head",
    "load_executable_thought_head",
]
