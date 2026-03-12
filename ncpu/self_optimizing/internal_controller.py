"""Buffered internal controller for hidden think-run-patch-commit inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Optional

from ncpu.self_optimizing.hidden_workspace import HiddenWorkspace
from ncpu.self_optimizing.latent_action_policy import LatentActionDecision
from ncpu.self_optimizing.latent_halt_policy import LatentHaltDecision
from ncpu.self_optimizing.latent_controller_state import LatentControllerState
from ncpu.self_optimizing.sandbox_actions import SandboxActionResult, SandboxActionRunner
from ncpu.self_optimizing.task_local_fast_weights import FastWeightUpdateResult
from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger


@dataclass
class InternalDeliberationTask:
    """Task specification for hidden buffered inference."""

    name: str
    prompt: str
    verifier: Optional[Callable[[str], Any]] = None
    test_cases: Optional[list[dict[str, Any]]] = None
    category: str = "coding"
    response_format: str = "raw Python code"
    feedback_builder: Optional[Callable[[str, dict[str, Any]], str]] = None


@dataclass
class InternalControllerConfig:
    """Configuration for buffered internal inference."""

    max_generation_attempts: int = 4
    plan_before_generate: bool = True
    allow_unverified_commit: bool = False
    commit_on_first_success: bool = True
    max_plan_tokens_hint: int = 128
    fast_weight_updates_on_plan: bool = True
    fast_weight_updates_on_repair_plan: bool = True
    fast_weight_updates_on_verify_failure: bool = True
    fast_weight_updates_on_verify_success: bool = False
    max_fast_weight_updates_per_task: int = 4
    descriptor_updates_on_plan: bool = True
    descriptor_updates_on_verify_failure: bool = True
    max_descriptor_updates_per_task: int = 4
    prefer_latent_action_policy: bool = True
    prefer_latent_memory_updater: bool = True
    prefer_latent_halt_policy: bool = True


@dataclass
class InternalModelResponse:
    """Normalized internal model response."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BufferedInternalController:
    """
    Hidden controller that reasons, verifies, repairs, and only then commits output.

    This is still an external controller in implementation terms, but it behaves like
    a private inference workspace to the end user: only the committed output escapes.
    """

    def __init__(
        self,
        *,
        llm_provider: Callable[[str], Any],
        action_provider: Optional[Callable[[str], Any]] = None,
        latent_action_policy: Optional[Any] = None,
        latent_memory_updater: Optional[Any] = None,
        latent_halt_policy: Optional[Any] = None,
        config: Optional[InternalControllerConfig] = None,
        sandbox_runner: Optional[SandboxActionRunner] = None,
        trajectory_logger: Optional[TrajectoryLogger] = None,
    ):
        self.llm_provider = llm_provider
        self.action_provider = action_provider
        self.latent_action_policy = latent_action_policy
        self.latent_memory_updater = latent_memory_updater
        self.latent_halt_policy = latent_halt_policy
        self.config = config or InternalControllerConfig()
        self.sandbox_runner = sandbox_runner or SandboxActionRunner()
        self.trajectory_logger = trajectory_logger or TrajectoryLogger()

    def _normalize_response(self, response: Any) -> InternalModelResponse:
        if isinstance(response, InternalModelResponse):
            return response
        if isinstance(response, str):
            return InternalModelResponse(text=response)
        if isinstance(response, dict):
            if "text" in response:
                metadata = {key: value for key, value in response.items() if key != "text"}
                return InternalModelResponse(text=str(response["text"]), metadata=metadata)
            return InternalModelResponse(text=str(response))
        if hasattr(response, "text"):
            return InternalModelResponse(
                text=str(getattr(response, "text")),
                metadata=dict(getattr(response, "metadata", {}) or {}),
            )
        return InternalModelResponse(text=str(response))

    def _call_model(self, prompt: str) -> InternalModelResponse:
        return self._normalize_response(self.llm_provider(prompt))

    def _normalize_action_decision(self, response: Any) -> str:
        normalized = self._normalize_response(response).text.strip().lower()
        if not normalized:
            return ""
        first_line = normalized.splitlines()[0].strip()
        token = first_line.split()[0]
        return token.strip("`'\".,:;()[]{}")

    def _call_action_policy(self, prompt: str) -> str:
        if self.action_provider is None:
            return ""
        return self._normalize_action_decision(self.action_provider(prompt))

    def _summarize_recent_steps(self, workspace: HiddenWorkspace, *, limit: int = 6) -> str:
        if not workspace.steps:
            return "(none)"

        recent = workspace.steps[-limit:]
        lines: list[str] = []
        for step in recent:
            parts = [step.action]
            if step.success is True:
                parts.append("success")
            elif step.success is False:
                parts.append("failure")
            if step.error:
                parts.append(f"error={step.error}")
            if step.response_text and step.action in {"think", "write", "patch"}:
                text = step.response_text.replace("\n", "\\n")
                if len(text) > 120:
                    text = text[:117] + "..."
                parts.append(f"response={text}")
            lines.append("- " + " | ".join(parts))
        return "\n".join(lines)

    def _build_action_prompt(
        self,
        task: InternalDeliberationTask,
        workspace: HiddenWorkspace,
        *,
        allowed_actions: list[str],
    ) -> str:
        return (
            "You are choosing the next hidden controller action inside a private "
            "code/reasoning workspace.\n"
            f"Task: {task.name}\n"
            f"Category: {task.category}\n"
            f"Max generation attempts: {workspace.max_generation_attempts}\n"
            f"Current generation attempts used: {workspace.generation_attempts}\n"
            f"Last verification error: {workspace.last_error or 'none'}\n"
            "Recent hidden history:\n"
            f"{self._summarize_recent_steps(workspace)}\n\n"
            f"Choose exactly one next action from: {', '.join(allowed_actions)}.\n"
            "Return only the action label."
        )

    def _resolve_latent_action_policy(self) -> Optional[Any]:
        if self.latent_action_policy is not None:
            return self.latent_action_policy
        for candidate in (self.action_provider, self.llm_provider):
            if candidate is None:
                continue
            embedded = getattr(candidate, "latent_action_policy", None)
            if embedded is not None:
                return embedded
            if hasattr(candidate, "choose_action"):
                return candidate
            if hasattr(candidate, "select_latent_action"):
                return candidate
        return None

    def _call_latent_action_policy(
        self,
        *,
        workspace: HiddenWorkspace,
        allowed_actions: list[str],
        fallback: str,
    ) -> Optional[LatentActionDecision]:
        policy = self._resolve_latent_action_policy()
        if policy is None or not self.config.prefer_latent_action_policy:
            return None

        choose_action = getattr(policy, "choose_action", None)
        if callable(choose_action):
            decision = choose_action(
                workspace=workspace,
                allowed_actions=allowed_actions,
                fallback=fallback,
            )
            if isinstance(decision, LatentActionDecision):
                return decision
            if isinstance(decision, dict):
                action = str(decision.get("action", fallback))
                return LatentActionDecision(
                    action=action,
                    source=str(decision.get("source", "latent_action_policy")),
                    scores=dict(decision.get("scores") or {}),
                    confidence=float(decision.get("confidence", 0.0) or 0.0),
                    feature_summary=dict(decision.get("feature_summary") or {}),
                )

        select_latent_action = getattr(policy, "select_latent_action", None)
        if callable(select_latent_action):
            decision = select_latent_action(
                workspace=workspace,
                allowed_actions=allowed_actions,
                fallback=fallback,
            )
            if isinstance(decision, LatentActionDecision):
                return decision
        return None

    def _choose_action(
        self,
        task: InternalDeliberationTask,
        workspace: HiddenWorkspace,
        *,
        allowed_actions: list[str],
        fallback: str,
    ) -> tuple[str, dict[str, Any]]:
        latent_decision = self._call_latent_action_policy(
            workspace=workspace,
            allowed_actions=allowed_actions,
            fallback=fallback,
        )
        if latent_decision is not None and latent_decision.action in allowed_actions:
            return latent_decision.action, {
                "policy_selected_action": latent_decision.action,
                "policy_source": latent_decision.source,
                "policy_confidence": latent_decision.confidence,
                "policy_scores": dict(latent_decision.scores),
                "policy_feature_summary": dict(latent_decision.feature_summary),
            }

        if self.action_provider is None:
            return fallback, {
                "policy_selected_action": fallback,
                "policy_source": "fallback",
            }

        action_prompt = self._build_action_prompt(task, workspace, allowed_actions=allowed_actions)
        decision = self._call_action_policy(action_prompt)
        if decision not in allowed_actions:
            return fallback, {
                "policy_selected_action": fallback,
                "policy_source": "fallback",
                "policy_prompt_used": True,
            }
        return decision, {
            "policy_selected_action": decision,
            "policy_source": "prompt_action_policy",
            "policy_prompt_used": True,
        }

    def _resolve_latent_halt_policy(self) -> Optional[Any]:
        if self.latent_halt_policy is not None:
            return self.latent_halt_policy
        for candidate in (self.action_provider, self.llm_provider):
            if candidate is None:
                continue
            embedded = getattr(candidate, "latent_halt_policy", None)
            if embedded is not None:
                return embedded
            if hasattr(candidate, "choose_halt_action"):
                return candidate
        return None

    def _resolve_latent_memory_updater(self) -> Optional[Any]:
        if self.latent_memory_updater is not None:
            return self.latent_memory_updater
        for candidate in (self.action_provider, self.llm_provider):
            if candidate is None:
                continue
            embedded = getattr(candidate, "latent_memory_updater", None)
            if embedded is not None:
                return embedded
            if hasattr(candidate, "build_memory_delta"):
                return candidate
        return None

    def _call_latent_halt_policy(
        self,
        *,
        workspace: HiddenWorkspace,
        verification_success: bool,
        verification_error: str,
        allowed_actions: list[str],
        fallback: str,
    ) -> Optional[LatentHaltDecision]:
        policy = self._resolve_latent_halt_policy()
        if policy is None or not self.config.prefer_latent_halt_policy:
            return None

        choose_halt_action = getattr(policy, "choose_halt_action", None)
        if not callable(choose_halt_action):
            return None
        decision = choose_halt_action(
            workspace=workspace,
            verification_success=verification_success,
            verification_error=verification_error,
            allowed_actions=allowed_actions,
            fallback=fallback,
        )
        if isinstance(decision, LatentHaltDecision):
            return decision
        if isinstance(decision, dict):
            action = str(decision.get("action", fallback))
            return LatentHaltDecision(
                action=action,
                source=str(decision.get("source", "latent_halt_policy")),
                scores=dict(decision.get("scores") or {}),
                confidence=float(decision.get("confidence", 0.0) or 0.0),
                feature_summary=dict(decision.get("feature_summary") or {}),
            )
        return None

    def _choose_halt_action(
        self,
        *,
        workspace: HiddenWorkspace,
        verification_success: bool,
        verification_error: str,
        allowed_actions: list[str],
        fallback: str,
    ) -> tuple[str, dict[str, Any]]:
        latent_decision = self._call_latent_halt_policy(
            workspace=workspace,
            verification_success=verification_success,
            verification_error=verification_error,
            allowed_actions=allowed_actions,
            fallback=fallback,
        )
        if latent_decision is not None and latent_decision.action in allowed_actions:
            return latent_decision.action, {
                "halt_selected_action": latent_decision.action,
                "halt_source": latent_decision.source,
                "halt_confidence": latent_decision.confidence,
                "halt_scores": dict(latent_decision.scores),
                "halt_feature_summary": dict(latent_decision.feature_summary),
            }
        return fallback, {
            "halt_selected_action": fallback,
            "halt_source": "fallback",
        }

    def _build_plan_prompt(self, task: InternalDeliberationTask) -> str:
        return (
            "You are operating inside a hidden inference workspace.\n"
            "Think privately about a robust solution and likely failure modes.\n"
            "Do not produce the final user-visible answer.\n"
            f"Keep the plan under {self.config.max_plan_tokens_hint} tokens.\n\n"
            f"Task:\n{task.prompt}"
        )

    def _build_generation_prompt(
        self,
        task: InternalDeliberationTask,
        *,
        plan: Optional[str],
        latent_state: Optional[LatentControllerState] = None,
    ) -> str:
        plan_section = f"Hidden plan:\n{plan}\n\n" if plan else ""
        latent_state_section = ""
        if latent_state is not None:
            latent_state_section = f"Latent state summary:\n{latent_state.to_prompt_summary()}\n\n"
        return (
            "You are operating inside a hidden inference workspace.\n"
            f"Generate a candidate solution as {task.response_format}.\n"
            "Return only the candidate output with no markdown or explanation.\n\n"
            f"{plan_section}"
            f"{latent_state_section}"
            f"Task:\n{task.prompt}"
        )

    def _build_repair_plan_prompt(
        self,
        task: InternalDeliberationTask,
        *,
        workspace: HiddenWorkspace,
        result: SandboxActionResult,
    ) -> str:
        return (
            "You are re-planning a hidden repair attempt inside a private inference workspace.\n"
            "Think privately about why the previous candidate failed and what to change.\n"
            "Do not produce the final user-visible answer.\n\n"
            f"Original task:\n{task.prompt}\n\n"
            f"Current candidate:\n{workspace.candidate_solution}\n\n"
            f"Verification failure:\n{result.failure_summary()}\n"
        )

    def _build_repair_prompt(
        self,
        task: InternalDeliberationTask,
        *,
        workspace: HiddenWorkspace,
        result: SandboxActionResult,
        plan: Optional[str] = None,
    ) -> str:
        if task.feedback_builder is not None:
            return task.feedback_builder(task.prompt, result.to_feedback_payload())

        failure_summary = result.failure_summary()
        plan_section = f"Hidden re-plan:\n{plan}\n\n" if plan else ""
        latent_state_section = f"Latent state summary:\n{workspace.latent_state.to_prompt_summary()}\n\n"
        return (
            "You are repairing a hidden candidate before user-visible commit.\n"
            f"Return only corrected {task.response_format}.\n\n"
            f"{plan_section}"
            f"{latent_state_section}"
            f"Original task:\n{task.prompt}\n\n"
            f"Current candidate:\n{workspace.candidate_solution}\n\n"
            f"Verification failure:\n{failure_summary}\n"
        )

    def _build_verified_refine_prompt(
        self,
        task: InternalDeliberationTask,
        *,
        workspace: HiddenWorkspace,
    ) -> str:
        return (
            "You are refining a hidden candidate that already passed verification.\n"
            "Preserve exact behavior, keep the same output contract, and only improve clarity or robustness.\n"
            f"Return only corrected {task.response_format}.\n\n"
            f"Latent state summary:\n{workspace.latent_state.to_prompt_summary()}\n\n"
            f"Original task:\n{task.prompt}\n\n"
            f"Current verified candidate:\n{workspace.candidate_solution}\n"
        )

    def _record_step(
        self,
        workspace: HiddenWorkspace,
        *,
        action: str,
        prompt: str,
        response_text: str,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        step = workspace.record_step(
            action=action,
            prompt=prompt,
            response_text=response_text,
            success=success,
            error=error,
            metadata=metadata,
        )
        workspace.latent_state.record_action(action)
        if action == "fast_weight_update":
            workspace.latent_state.record_fast_weight_update(
                success=bool(success),
                kind=str((metadata or {}).get("kind", "unknown")),
            )
        if action == "descriptor_update":
            workspace.latent_state.record_descriptor_update(
                success=bool(success),
                kind=str((metadata or {}).get("kind", "unknown")),
            )
        self._apply_latent_memory_update(
            workspace,
            event_kind=action,
            response_text=response_text,
            error_text=error or "",
            success=success,
            step=step,
        )
        self.trajectory_logger.log_step(workspace, step)

    def _apply_latent_memory_update(
        self,
        workspace: HiddenWorkspace,
        *,
        event_kind: str,
        response_text: str,
        error_text: str,
        success: Optional[bool],
        step: Any,
    ) -> None:
        updater = self._resolve_latent_memory_updater()
        if updater is None or not self.config.prefer_latent_memory_updater:
            return
        build_memory_delta = getattr(updater, "build_memory_delta", None)
        if not callable(build_memory_delta):
            return
        try:
            delta, summary = build_memory_delta(
                latent_state=workspace.latent_state,
                workspace=workspace,
                event_kind=event_kind,
                response_text=response_text,
                error_text=error_text,
                success=success,
            )
        except Exception as exc:
            step.metadata["latent_memory_update_error"] = f"{type(exc).__name__}: {exc}"
            return
        if not isinstance(delta, list) or not delta:
            step.metadata["latent_memory_update"] = {
                "applied": False,
                "source": "latent_memory_updater",
                "event_kind": event_kind,
            }
            return
        workspace.latent_state.apply_memory_delta(delta, scale=0.35)
        step.metadata["latent_memory_update"] = {
            "applied": True,
            "source": "latent_memory_updater",
            "event_kind": event_kind,
            "delta_norm": sum(abs(float(value)) for value in delta),
            "feature_summary": dict(summary or {}),
        }

    def _fast_weight_update_count(self, workspace: HiddenWorkspace) -> int:
        return sum(1 for step in workspace.steps if step.action == "fast_weight_update" and step.success)

    def _fast_weight_updates_remaining(self, workspace: HiddenWorkspace) -> int:
        return max(self.config.max_fast_weight_updates_per_task - self._fast_weight_update_count(workspace), 0)

    def _can_apply_fast_weight_update(self, workspace: HiddenWorkspace) -> bool:
        return self._fast_weight_updates_remaining(workspace) > 0

    def _descriptor_update_count(self, workspace: HiddenWorkspace) -> int:
        return sum(1 for step in workspace.steps if step.action == "descriptor_update" and step.success)

    def _descriptor_updates_remaining(self, workspace: HiddenWorkspace) -> int:
        return max(self.config.max_descriptor_updates_per_task - self._descriptor_update_count(workspace), 0)

    def _can_apply_descriptor_update(self, workspace: HiddenWorkspace) -> bool:
        return self._descriptor_updates_remaining(workspace) > 0

    def _build_failure_update_prompt(
        self,
        task: InternalDeliberationTask,
        *,
        workspace: HiddenWorkspace,
        result: SandboxActionResult,
    ) -> str:
        return (
            "You are updating hidden task-local inference state after a failed verification.\n"
            "Absorb only the error pattern and the repair constraints. Do not emit final user output.\n\n"
            f"Task:\n{task.prompt}\n\n"
            f"Current candidate:\n{workspace.candidate_solution}\n\n"
            f"Failure summary:\n{result.failure_summary()}\n"
        )

    def _build_failure_update_target(
        self,
        task: InternalDeliberationTask,
        *,
        workspace: HiddenWorkspace,
        result: SandboxActionResult,
    ) -> str:
        return (
            f"Task: {task.name}\n"
            "Repair constraints:\n"
            f"- Output format: {task.response_format}\n"
            f"- Failure: {result.failure_summary()}\n"
            "- Preserve correct logic from the current candidate where possible.\n"
            "- Change only what is needed to satisfy verification.\n"
        )

    def _begin_fast_weights(
        self,
        task: InternalDeliberationTask,
        workspace: HiddenWorkspace,
    ) -> None:
        begin_task = getattr(self.llm_provider, "begin_task", None)
        if begin_task is None:
            return
        try:
            result = begin_task(task.name, task.prompt)
        except Exception as exc:
            workspace.metadata["fast_weights_begin_error"] = f"{type(exc).__name__}: {exc}"
            return
        if isinstance(result, dict):
            workspace.metadata["fast_weights"] = dict(result)

    def _normalize_fast_weight_result(self, payload: Any, *, update_kind: str) -> FastWeightUpdateResult:
        if isinstance(payload, FastWeightUpdateResult):
            return payload
        if isinstance(payload, dict):
            return FastWeightUpdateResult(
                success=bool(payload.get("success", True)),
                kind=str(payload.get("kind", update_kind)),
                updated_modules=list(payload.get("updated_modules") or []),
                task_name=payload.get("task_name"),
                loss=payload.get("loss"),
                steps=int(payload.get("steps", 0) or 0),
                target_tokens=int(payload.get("target_tokens", 0) or 0),
                elapsed_seconds=float(payload.get("elapsed_seconds", 0.0) or 0.0),
                error=payload.get("error"),
            )
        return FastWeightUpdateResult(success=bool(payload), kind=update_kind)

    def _apply_fast_weight_update(
        self,
        task: InternalDeliberationTask,
        workspace: HiddenWorkspace,
        *,
        prompt: str,
        target_text: str,
        update_kind: str,
    ) -> None:
        apply_update = getattr(self.llm_provider, "apply_self_update", None)
        if apply_update is None or not target_text.strip():
            return
        if not self._can_apply_fast_weight_update(workspace):
            self._record_step(
                workspace,
                action="fast_weight_update",
                prompt=f"fast-weight {update_kind}",
                response_text=f"{update_kind}: budget exhausted",
                success=False,
                error="fast-weight update budget exhausted",
                metadata={
                    "kind": update_kind,
                    "remaining_updates": 0,
                },
            )
            return

        started = time.perf_counter()
        try:
            result = apply_update(
                prompt=prompt,
                target_text=target_text,
                update_kind=update_kind,
                task_name=task.name,
            )
        except TypeError:
            result = apply_update(prompt, target_text)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._record_step(
                workspace,
                action="fast_weight_update",
                prompt=f"fast-weight {update_kind}",
                response_text=f"{update_kind}: failed",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"kind": update_kind, "elapsed_seconds": elapsed},
            )
            return

        normalized = self._normalize_fast_weight_result(result, update_kind=update_kind)
        metadata = normalized.to_dict()
        if metadata.get("elapsed_seconds", 0.0) == 0.0:
            metadata["elapsed_seconds"] = time.perf_counter() - started
        if normalized.task_name is None:
            metadata["task_name"] = task.name
        metadata["remaining_updates"] = self._fast_weight_updates_remaining(workspace) - (1 if normalized.success else 0)
        self._record_step(
            workspace,
            action="fast_weight_update",
            prompt=f"fast-weight {update_kind}",
            response_text=(
                f"{update_kind}: updated {len(normalized.updated_modules)} modules"
                if normalized.success
                else f"{update_kind}: update skipped"
            ),
            success=normalized.success,
            error=normalized.error,
            metadata=metadata,
        )

    def _end_fast_weights(self, workspace: HiddenWorkspace) -> None:
        end_task = getattr(self.llm_provider, "end_task", None)
        if end_task is None:
            return
        try:
            result = end_task()
        except Exception as exc:
            workspace.metadata["fast_weights_end_error"] = f"{type(exc).__name__}: {exc}"
            return
        if isinstance(result, dict):
            workspace.metadata["fast_weights_last_task"] = dict(result)

    def _apply_state_descriptor_update(
        self,
        task: InternalDeliberationTask,
        workspace: HiddenWorkspace,
        *,
        update_kind: str,
        error_text: str = "",
    ) -> None:
        apply_descriptor_update = getattr(self.llm_provider, "apply_state_descriptor_update", None)
        if apply_descriptor_update is None:
            return
        if not self._can_apply_descriptor_update(workspace):
            self._record_step(
                workspace,
                action="descriptor_update",
                prompt=f"descriptor {update_kind}",
                response_text=f"{update_kind}: budget exhausted",
                success=False,
                error="descriptor update budget exhausted",
                metadata={"kind": update_kind, "remaining_updates": 0},
            )
            return

        started = time.perf_counter()
        try:
            result = apply_descriptor_update(
                task_name=task.name,
                update_kind=update_kind,
                latent_state=workspace.latent_state,
                error_text=error_text,
                candidate_text=workspace.candidate_solution,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._record_step(
                workspace,
                action="descriptor_update",
                prompt=f"descriptor {update_kind}",
                response_text=f"{update_kind}: failed",
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata={"kind": update_kind, "elapsed_seconds": elapsed},
            )
            return

        normalized = self._normalize_fast_weight_result(result, update_kind=update_kind)
        metadata = normalized.to_dict()
        if metadata.get("elapsed_seconds", 0.0) == 0.0:
            metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["remaining_updates"] = self._descriptor_updates_remaining(workspace) - (1 if normalized.success else 0)
        self._record_step(
            workspace,
            action="descriptor_update",
            prompt=f"descriptor {update_kind}",
            response_text=(
                f"{update_kind}: applied latent descriptor"
                if normalized.success
                else f"{update_kind}: descriptor update skipped"
            ),
            success=normalized.success,
            error=normalized.error,
            metadata=metadata,
        )

    def run_task(self, task: InternalDeliberationTask) -> HiddenWorkspace:
        workspace = HiddenWorkspace(
            task_name=task.name,
            task_prompt=task.prompt,
            category=task.category,
            status="running",
            max_generation_attempts=self.config.max_generation_attempts,
        )
        if self._resolve_latent_memory_updater() is not None and self.config.prefer_latent_memory_updater:
            workspace.latent_state.enable_heuristic_memory_updates = False
            workspace.metadata["latent_memory_mode"] = "learned_recurrent_updater"
        self.trajectory_logger.log_workspace_initialized(workspace)
        self._begin_fast_weights(task, workspace)

        try:
            plan_text: Optional[str] = None
            initial_action, initial_policy_metadata = self._choose_action(
                task,
                workspace,
                allowed_actions=(["think", "write"] if self.config.plan_before_generate else ["write"]),
                fallback=("think" if self.config.plan_before_generate else "write"),
            )
            if initial_action == "think":
                plan_prompt = self._build_plan_prompt(task)
                started = time.perf_counter()
                plan_response = self._call_model(plan_prompt)
                elapsed = time.perf_counter() - started
                plan_text = plan_response.text.strip()
                workspace.latent_state.record_plan(plan_text, kind="plan")
                plan_metadata = dict(plan_response.metadata)
                plan_metadata["model_elapsed_seconds"] = elapsed
                plan_metadata.update(initial_policy_metadata)
                self._record_step(
                    workspace,
                    action="think",
                    prompt=plan_prompt,
                    response_text=plan_text,
                    metadata=plan_metadata,
                )
                if self.config.descriptor_updates_on_plan:
                    self._apply_state_descriptor_update(
                        task,
                        workspace,
                        update_kind="plan_descriptor",
                    )
                if self.config.fast_weight_updates_on_plan:
                    self._apply_fast_weight_update(
                        task,
                        workspace,
                        prompt=plan_prompt,
                        target_text=plan_text,
                        update_kind="plan",
                    )

            current_prompt = self._build_generation_prompt(
                task,
                plan=plan_text,
                latent_state=workspace.latent_state,
            )

            for attempt_index in range(1, self.config.max_generation_attempts + 1):
                generation_action = "write" if attempt_index == 1 else "patch"
                repair_policy_metadata: dict[str, Any] = {}
                if attempt_index > 1:
                    next_action, next_policy_metadata = self._choose_action(
                        task,
                        workspace,
                        allowed_actions=["patch", "think", "fail"],
                        fallback="patch",
                    )
                    if next_action == "fail":
                        failure_message = "action policy terminated the repair loop"
                        if workspace.last_error:
                            failure_message = f"{failure_message}: {workspace.last_error}"
                        self._record_step(
                            workspace,
                            action="fail",
                            prompt="hidden fail decision",
                            response_text=failure_message,
                            success=False,
                            error=failure_message,
                            metadata=dict(next_policy_metadata),
                        )
                        workspace.fail(failure_message)
                        self.trajectory_logger.log_failed(workspace)
                        return workspace
                    if next_action == "think":
                        repair_plan_prompt = self._build_repair_plan_prompt(
                            task,
                            workspace=workspace,
                            result=verification_result,
                        )
                        plan_started = time.perf_counter()
                        repair_plan_response = self._call_model(repair_plan_prompt)
                        plan_elapsed = time.perf_counter() - plan_started
                        plan_text = repair_plan_response.text.strip()
                        workspace.latent_state.record_plan(plan_text, kind="repair_plan")
                        repair_plan_metadata = dict(repair_plan_response.metadata)
                        repair_plan_metadata["model_elapsed_seconds"] = plan_elapsed
                        repair_plan_metadata.update(next_policy_metadata)
                        self._record_step(
                            workspace,
                            action="think",
                            prompt=repair_plan_prompt,
                            response_text=plan_text,
                            metadata=repair_plan_metadata,
                        )
                        if self.config.fast_weight_updates_on_repair_plan:
                            self._apply_fast_weight_update(
                                task,
                                workspace,
                                prompt=repair_plan_prompt,
                                target_text=plan_text,
                                update_kind="repair_plan",
                            )
                        current_prompt = self._build_repair_prompt(
                            task,
                            workspace=workspace,
                            result=verification_result,
                            plan=plan_text,
                        )
                    else:
                        repair_policy_metadata = dict(next_policy_metadata)
                    generation_action = "patch"
                started = time.perf_counter()
                response = self._call_model(current_prompt)
                generation_elapsed = time.perf_counter() - started
                workspace.generation_attempts = attempt_index
                workspace.set_candidate(response.text.strip())
                workspace.latent_state.record_candidate(workspace.candidate_solution)
                generation_metadata = dict(response.metadata)
                generation_metadata["model_elapsed_seconds"] = generation_elapsed
                if attempt_index == 1:
                    if initial_action == "write":
                        generation_metadata.update(initial_policy_metadata)
                    else:
                        generation_metadata["policy_selected_action"] = generation_action
                        generation_metadata.setdefault("policy_source", "implicit_after_think")
                else:
                    if repair_policy_metadata:
                        generation_metadata.update(repair_policy_metadata)
                    else:
                        generation_metadata["policy_selected_action"] = generation_action
                        generation_metadata.setdefault("policy_source", "implicit_generation")
                self._record_step(
                    workspace,
                    action=generation_action,
                    prompt=current_prompt,
                    response_text=workspace.candidate_solution,
                    metadata=generation_metadata,
                )

                verification_started = time.perf_counter()
                verification_result = self.sandbox_runner.verify_candidate(
                    workspace.candidate_solution,
                    verifier=task.verifier,
                    test_cases=task.test_cases,
                )
                verification_elapsed = time.perf_counter() - verification_started
                workspace.record_verification(
                    success=verification_result.success,
                    verification=verification_result.verification,
                    error=verification_result.error,
                )
                workspace.latent_state.record_verification(
                    success=verification_result.success,
                    verification=verification_result.verification,
                    error=verification_result.error,
                )
                verification_metadata = dict(verification_result.verification or {})
                verification_metadata["verification_elapsed_seconds"] = verification_elapsed
                self._record_step(
                    workspace,
                    action="verify",
                    prompt=f"hidden verification attempt {attempt_index}",
                    response_text="pass" if verification_result.success else verification_result.failure_summary(),
                    success=verification_result.success,
                    error=None if verification_result.success else verification_result.error,
                    metadata=verification_metadata,
                )

                if verification_result.success:
                    halt_action, halt_metadata = self._choose_halt_action(
                        workspace=workspace,
                        verification_success=True,
                        verification_error="",
                        allowed_actions=["commit", "continue"],
                        fallback=("commit" if self.config.commit_on_first_success else "continue"),
                    )
                    if halt_action == "continue" and attempt_index >= self.config.max_generation_attempts:
                        halt_action = "commit"
                        halt_metadata = {
                            **halt_metadata,
                            "halt_selected_action": "commit",
                            "halt_source": "forced_commit_after_final_verified_attempt",
                            "halt_override": "no_attempts_remaining",
                        }
                    if halt_action == "commit":
                        if self.config.fast_weight_updates_on_verify_success:
                            self._apply_fast_weight_update(
                                task,
                                workspace,
                                prompt=current_prompt,
                                target_text=verification_result.normalized_candidate,
                                update_kind="verified_candidate",
                            )
                        self._record_step(
                            workspace,
                            action="commit",
                            prompt="hidden commit decision",
                            response_text="commit verified candidate",
                            success=True,
                            metadata={
                                "verified": True,
                                "committed_attempt": attempt_index,
                                "policy_selected_action": "commit",
                                **halt_metadata,
                            },
                        )
                        workspace.commit(
                            verification_result.normalized_candidate,
                            verified=True,
                            metadata={"committed_attempt": attempt_index},
                        )
                        self.trajectory_logger.log_commit(workspace)
                        return workspace

                    current_prompt = self._build_verified_refine_prompt(
                        task,
                        workspace=workspace,
                    )
                    continue

                halt_action, halt_metadata = self._choose_halt_action(
                    workspace=workspace,
                    verification_success=False,
                    verification_error=verification_result.failure_summary(),
                    allowed_actions=["continue", "fail"],
                    fallback=(
                        "continue"
                        if attempt_index < self.config.max_generation_attempts
                        else "fail"
                    ),
                )
                if halt_action == "continue" and attempt_index >= self.config.max_generation_attempts:
                    halt_action = "fail"
                    halt_metadata = {
                        **halt_metadata,
                        "halt_selected_action": "fail",
                        "halt_source": "forced_fail_after_final_attempt",
                        "halt_override": "no_attempts_remaining",
                    }
                if halt_action == "fail":
                    failure_message = verification_result.failure_summary()
                    self._record_step(
                        workspace,
                        action="fail",
                        prompt="hidden halt decision",
                        response_text=failure_message,
                        success=False,
                        error=failure_message,
                        metadata=dict(halt_metadata),
                    )
                    workspace.fail(failure_message)
                    self.trajectory_logger.log_failed(workspace)
                    return workspace

                if self.config.descriptor_updates_on_verify_failure:
                    self._apply_state_descriptor_update(
                        task,
                        workspace,
                        update_kind="verify_failure_descriptor",
                        error_text=verification_result.failure_summary(),
                    )
                if self.config.fast_weight_updates_on_verify_failure:
                    self._apply_fast_weight_update(
                        task,
                        workspace,
                        prompt=self._build_failure_update_prompt(
                            task,
                            workspace=workspace,
                            result=verification_result,
                        ),
                        target_text=self._build_failure_update_target(
                            task,
                            workspace=workspace,
                            result=verification_result,
                        ),
                        update_kind="verify_failure",
                    )

                current_prompt = self._build_repair_prompt(
                    task,
                    workspace=workspace,
                    result=verification_result,
                )

            if self.config.allow_unverified_commit and workspace.candidate_solution:
                self._record_step(
                    workspace,
                    action="commit",
                    prompt="hidden commit decision",
                    response_text="commit unverified candidate after exhausting verification budget",
                    success=False,
                    error=workspace.last_error,
                    metadata={
                        "verified": False,
                        "committed_attempt": workspace.generation_attempts,
                        "policy_selected_action": "commit",
                    },
                )
                workspace.commit(
                    workspace.candidate_solution,
                    verified=False,
                    metadata={"committed_attempt": workspace.generation_attempts},
                )
                self.trajectory_logger.log_commit(workspace)
                return workspace

            failure_message = workspace.last_error or "verification budget exhausted"
            self._record_step(
                workspace,
                action="fail",
                prompt="hidden fail decision",
                response_text=failure_message,
                success=False,
                error=failure_message,
                metadata={"policy_selected_action": "fail"},
            )
            workspace.fail(failure_message)
            self.trajectory_logger.log_failed(workspace)
            return workspace
        finally:
            self._end_fast_weights(workspace)

    def infer(self, task: InternalDeliberationTask) -> str:
        workspace = self.run_task(task)
        if workspace.committed_output is None:
            raise RuntimeError(workspace.last_error or "buffered inference failed without a committed output")
        return workspace.committed_output
