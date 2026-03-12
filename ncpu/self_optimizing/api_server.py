"""
SOME REST API Server

FastAPI-based REST API for remote execution of the Self-Optimizing Machine Engine.
Allows clients to submit tasks, monitor execution, and retrieve results.

Usage:
    # Start server
    uvicorn ncpu.self_optimizing.api_server:app --reload

    # Or run directly
    python -m ncpu.self_optimizing.api_server
"""

import json
from pathlib import Path
import re
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import threading
import uuid
import time
from datetime import datetime, timezone

# Try to import fastapi, make optional
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = object
    BaseModel = object

    def Field(default=None, **_kwargs):
        return default

    BackgroundTasks = object
    HTTPException = Exception
    JSONResponse = dict
    StreamingResponse = dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_filename_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# =============================================================================
# Data Models
# =============================================================================

class TaskStatus(str, Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VerificationMode(str, Enum):
    """Verification strategy for buffered internal inference."""

    AUTO = "auto"
    CODE_TESTS = "code_tests"
    EXACT_MATCH = "exact_match"
    NONE = "none"


class TaskRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request to execute a task."""
    description: str = Field(..., description="Natural language description of the task")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for the task")
    expected_output: Optional[Any] = Field(None, description="Expected output for verification")
    max_iterations: int = Field(10, description="Maximum optimization iterations")
    timeout_seconds: float = Field(30.0, description="Execution timeout")
    use_gpu: bool = Field(True, description="Whether to use GPU acceleration")


class TaskResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Response from task submission."""
    task_id: str
    status: TaskStatus
    created_at: str


class ExecutionResultResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    iterations_completed: int = 0
    feedback_signal: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


class ModelInfo(BaseModel if FASTAPI_AVAILABLE else object):
    """Information about a model in the zoo."""
    name: str
    category: str
    description: str
    accuracy: float
    avg_execution_time_ms: float


class BufferedInferenceRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request for hidden think-run-patch-commit inference."""

    prompt: str = Field(..., description="Task prompt for the model")
    task_name: Optional[str] = Field(None, description="Optional stable task identifier")
    category: str = Field("coding", description="Task category, e.g. coding or reasoning")
    response_format: str = Field("raw Python code", description="Expected final output format")
    controller_bundle_path: Optional[str] = Field(None, description="Optional controller bundle manifest path")
    provider: str = Field("local", description="LLM provider name")
    model: str = Field("qwen3.5:9b", description="Model identifier")
    base_url: str = Field("http://localhost:11434", description="Provider base URL for local/OpenAI-compatible models")
    api_key: Optional[str] = Field(None, description="Optional provider API key")
    temperature: float = Field(0.2, description="Sampling temperature")
    max_tokens: int = Field(2048, description="Maximum generation tokens")
    request_timeout_seconds: float = Field(120.0, description="Provider request timeout in seconds")
    action_provider: Optional[str] = Field(None, description="Optional hidden action-policy provider; defaults to the main provider when action settings are supplied")
    action_model: Optional[str] = Field(None, description="Optional hidden action-policy model; defaults to the main model when action settings are supplied")
    action_base_url: Optional[str] = Field(None, description="Optional base URL for the hidden action-policy provider")
    action_api_key: Optional[str] = Field(None, description="Optional API key for the hidden action-policy provider")
    action_temperature: Optional[float] = Field(None, description="Optional sampling temperature for hidden action-policy calls")
    action_max_tokens: Optional[int] = Field(None, description="Optional max tokens for hidden action-policy calls")
    action_request_timeout_seconds: Optional[float] = Field(None, description="Optional request timeout for hidden action-policy calls")
    verification_mode: VerificationMode = Field(
        VerificationMode.AUTO,
        description="How the server verifies hidden candidates before commit",
    )
    test_cases: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Python test cases in CodeVerifier format",
    )
    expected_output: Optional[Any] = Field(
        None,
        description="Expected final output for exact-match verification",
    )
    max_generation_attempts: int = Field(4, description="Maximum hidden generation/repair attempts")
    plan_before_generate: bool = Field(True, description="Generate a hidden plan before first candidate")
    allow_unverified_commit: bool = Field(False, description="Allow final output even if verification never passes")
    commit_on_first_success: bool = Field(True, description="Commit immediately on first verified success")
    max_plan_tokens_hint: int = Field(128, description="Planning token budget hint")
    include_hidden_trace: bool = Field(False, description="Include the full hidden workspace snapshot in the response")
    persist_trajectory: bool = Field(True, description="Persist hidden trajectory JSONL for training/evidence")
    trajectory_path: Optional[str] = Field(
        None,
        description="Optional explicit JSONL path for hidden trajectory logging",
    )


class BufferedInferenceResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Response from buffered hidden inference."""

    task_name: str
    category: str
    status: str
    provider: str
    model: str
    controller_bundle_path: Optional[str] = None
    action_provider: Optional[str] = None
    action_model: Optional[str] = None
    verification_mode: VerificationMode
    committed_output: Optional[str] = None
    committed_verified: bool = False
    generation_attempts: int = 0
    max_generation_attempts: int = 0
    hidden_step_count: int = 0
    hidden_actions: List[str] = Field(default_factory=list)
    last_error: Optional[str] = None
    last_verification: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    trajectory_path: Optional[str] = None
    hidden_workspace: Optional[Dict[str, Any]] = None


class BufferedInferenceTaskResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Submission response for async buffered inference."""

    task_id: str
    status: TaskStatus
    task_name: str
    created_at: str


class BufferedInferenceTaskStatusResponse(BaseModel if FASTAPI_AVAILABLE else object):
    """Status view for an async buffered inference task."""

    task_id: str
    status: TaskStatus
    task_name: str
    category: str
    provider: str
    model: str
    controller_bundle_path: Optional[str] = None
    action_provider: Optional[str] = None
    action_model: Optional[str] = None
    verification_mode: VerificationMode
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_ms: float = 0.0
    hidden_step_count: int = 0
    last_event: Optional[str] = None
    last_error: Optional[str] = None
    committed_verified: bool = False
    committed_output: Optional[str] = None
    trajectory_path: Optional[str] = None


class BufferedInferenceEvent(BaseModel if FASTAPI_AVAILABLE else object):
    """Sanitized event from hidden buffered inference."""

    sequence: int
    timestamp: str
    event: str
    task_name: str
    status: Optional[str] = None
    step_index: Optional[int] = None
    action: Optional[str] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    committed_verified: Optional[bool] = None
    generation_attempts: Optional[int] = None


# =============================================================================
# In-Memory Task Store (for demo - use Redis in production)
# =============================================================================

class TaskStore:
    """Simple in-memory task storage."""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def create_task(self, request: TaskRequest) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "task_id": task_id,
            "description": request.description,
            "input_data": request.input_data,
            "expected_output": request.expected_output,
            "max_iterations": request.max_iterations,
            "timeout_seconds": request.timeout_seconds,
            "use_gpu": request.use_gpu,
            "status": TaskStatus.PENDING,
            "created_at": _utc_now(),
            "output": None,
            "error": None,
            "execution_time_ms": 0.0,
            "iterations_completed": 0,
            "feedback_signal": None,
        }
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, updates: Dict[str, Any]):
        if task_id in self.tasks:
            self.tasks[task_id].update(updates)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return sorted(tasks, key=lambda t: t["created_at"], reverse=True)


class BufferedInferenceTaskStore:
    """In-memory status and event store for async buffered inference."""

    TERMINAL_STATUSES = {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED}

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_task(
        self,
        *,
        request: BufferedInferenceRequest,
        verification_mode: VerificationMode,
        trajectory_path: Optional[str],
    ) -> str:
        task_id = str(uuid.uuid4())
        created_at = _utc_now()
        response_config = _resolve_component_config(request)
        with self._lock:
            self.tasks[task_id] = {
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "task_name": request.task_name or _slugify_task_name(request.prompt[:48]),
                "category": request.category,
                "provider": response_config["provider"],
                "model": response_config["model"],
                "controller_bundle_path": request.controller_bundle_path,
                "action_provider": _resolve_action_provider_name(request),
                "action_model": _resolve_action_model_name(request),
                "verification_mode": verification_mode,
                "created_at": created_at,
                "started_at": None,
                "completed_at": None,
                "execution_time_ms": 0.0,
                "hidden_step_count": 0,
                "last_event": None,
                "last_error": None,
                "committed_verified": False,
                "committed_output": None,
                "trajectory_path": trajectory_path,
                "events": [],
            }
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return None
            return {
                **task,
                "events": list(task["events"]),
            }

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(updates)

    def append_event(self, task_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            task = self.tasks.get(task_id)
            if task is None:
                return
            task["events"].append(event)
            task["hidden_step_count"] = len(task["events"])
            task["last_event"] = event.get("event") or event.get("action")
            if event.get("error"):
                task["last_error"] = event["error"]


class InternalTaskEventLogger:
    """Trajectory logger that also mirrors sanitized events into the API task store."""

    def __init__(
        self,
        *,
        task_store: BufferedInferenceTaskStore,
        task_id: str,
        trajectory_path: Optional[str] = None,
    ):
        from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger

        self._task_store = task_store
        self._task_id = task_id
        self._trajectory_logger = TrajectoryLogger(trajectory_path)

    def _append(self, event: Dict[str, Any]) -> None:
        self._trajectory_logger._append(event)
        self._task_store.append_event(self._task_id, _sanitize_internal_event(event))

    def log_workspace_initialized(self, workspace: Any) -> None:
        self._trajectory_logger.log_workspace_initialized(workspace)
        event = {
            "timestamp": _utc_now(),
            "event": "workspace_initialized",
            "task_name": workspace.task_name,
            "status": workspace.status,
        }
        self._task_store.append_event(self._task_id, _sanitize_internal_event(event))

    def log_step(self, workspace: Any, step: Any) -> None:
        event = {
            "timestamp": step.timestamp,
            "event": "workspace_step",
            "task_name": workspace.task_name,
            "status": workspace.status,
            "step_index": step.index,
            "action": step.action,
            "success": step.success,
            "error": step.error,
            "metadata": step.metadata,
            "prompt": step.prompt,
            "response_text": step.response_text,
        }
        self._append(event)

    def log_commit(self, workspace: Any) -> None:
        event = {
            "timestamp": _utc_now(),
            "event": "workspace_committed",
            "task_name": workspace.task_name,
            "status": workspace.status,
            "committed_verified": workspace.committed_verified,
            "generation_attempts": workspace.generation_attempts,
            "last_error": workspace.last_error,
            "committed_output": workspace.committed_output,
        }
        self._append(event)

    def log_failed(self, workspace: Any) -> None:
        event = {
            "timestamp": _utc_now(),
            "event": "workspace_failed",
            "task_name": workspace.task_name,
            "status": workspace.status,
            "generation_attempts": workspace.generation_attempts,
            "last_error": workspace.last_error,
        }
        self._append(event)


# Global task store
task_store = TaskStore()
buffered_inference_task_store = BufferedInferenceTaskStore()


# =============================================================================
# Buffered Internal Inference Helpers
# =============================================================================

def _slugify_task_name(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "buffered_inference"


def _coerce_exact_match_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False

        try:
            if "." in stripped:
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped
    return value


def _resolve_verification_mode(request: BufferedInferenceRequest) -> VerificationMode:
    if request.verification_mode != VerificationMode.AUTO:
        return request.verification_mode
    if request.test_cases:
        return VerificationMode.CODE_TESTS
    if request.expected_output is not None:
        return VerificationMode.EXACT_MATCH
    return VerificationMode.NONE


def _build_exact_match_verifier(expected_output: Any) -> Callable[[str], tuple[bool, Optional[str]]]:
    normalized_expected = _coerce_exact_match_value(expected_output)

    def verifier(candidate_text: str) -> tuple[bool, Optional[str]]:
        actual = _coerce_exact_match_value(candidate_text)
        success = actual == normalized_expected
        if success:
            return True, None
        return False, f"expected {normalized_expected!r}, got {actual!r}"

    return verifier


def _build_request_verifier(
    request: BufferedInferenceRequest,
) -> tuple[VerificationMode, Optional[Callable[[str], Any]], Optional[List[Dict[str, Any]]]]:
    mode = _resolve_verification_mode(request)
    if mode == VerificationMode.CODE_TESTS:
        if not request.test_cases:
            raise ValueError("verification_mode=code_tests requires non-empty test_cases")
        return mode, None, request.test_cases
    if mode == VerificationMode.EXACT_MATCH:
        if request.expected_output is None:
            raise ValueError("verification_mode=exact_match requires expected_output")
        return mode, _build_exact_match_verifier(request.expected_output), None
    return mode, None, None


def _resolve_trajectory_path(request: BufferedInferenceRequest) -> Optional[str]:
    if request.trajectory_path:
        return request.trajectory_path
    if not request.persist_trajectory:
        return None

    timestamp = _utc_filename_stamp()
    task_name = request.task_name or request.prompt[:48]
    filename = f"{_slugify_task_name(task_name)}_{timestamp}_{uuid.uuid4().hex[:8]}.jsonl"
    return str(PROJECT_ROOT / "benchmarks" / "internal_trajectories" / filename)


def _request_field_set(request: BufferedInferenceRequest, field_name: str) -> bool:
    fields = getattr(request, "model_fields_set", None)
    if fields is None:
        fields = getattr(request, "__fields_set__", set())
    return field_name in fields


def _load_request_bundle(request: BufferedInferenceRequest) -> Optional[Any]:
    if not request.controller_bundle_path:
        return None

    from ncpu.self_optimizing.controller_bundle import load_controller_bundle

    try:
        return load_controller_bundle(request.controller_bundle_path)
    except FileNotFoundError as exc:
        raise ValueError(f"controller bundle not found: {request.controller_bundle_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"controller bundle is not valid JSON: {request.controller_bundle_path}") from exc
    except KeyError as exc:
        raise ValueError(f"controller bundle is missing required field: {exc}") from exc


def _resolve_internal_controller_config(request: BufferedInferenceRequest) -> Any:
    from ncpu.self_optimizing.internal_controller import InternalControllerConfig

    bundle = _load_request_bundle(request)
    config_kwargs = dict(getattr(bundle, "controller_config", {}) or {})
    explicit_or_default = {
        "max_generation_attempts": request.max_generation_attempts,
        "plan_before_generate": request.plan_before_generate,
        "allow_unverified_commit": request.allow_unverified_commit,
        "commit_on_first_success": request.commit_on_first_success,
        "max_plan_tokens_hint": request.max_plan_tokens_hint,
    }
    for field_name, value in explicit_or_default.items():
        if _request_field_set(request, field_name) or field_name not in config_kwargs:
            config_kwargs[field_name] = value
    return InternalControllerConfig(**config_kwargs)


def _has_action_policy_config(request: BufferedInferenceRequest) -> bool:
    explicit = any(
        value is not None
        for value in (
            request.action_provider,
            request.action_model,
            request.action_base_url,
            request.action_api_key,
            request.action_temperature,
            request.action_max_tokens,
            request.action_request_timeout_seconds,
        )
    )
    if explicit:
        return True

    bundle = _load_request_bundle(request)
    return bool(bundle and bundle.action is not None)


def _resolve_component_config(
    request: BufferedInferenceRequest,
    *,
    action: bool = False,
) -> Optional[dict[str, Any]]:
    bundle = _load_request_bundle(request)
    bundle_component = None
    if bundle is not None:
        bundle_component = bundle.action if action else bundle.response

    if action and bundle_component is None and not _has_action_policy_config(request):
        return None

    if action:
        provider = (
            request.action_provider
            if _request_field_set(request, "action_provider")
            else (
                bundle_component.provider
                if bundle_component is not None
                else request.provider
            )
        )
        model = (
            request.action_model
            if _request_field_set(request, "action_model")
            else (
                bundle_component.model
                if bundle_component is not None
                else request.model
            )
        )
        base_url = (
            request.action_base_url
            if _request_field_set(request, "action_base_url")
            else (
                bundle_component.base_url
                if bundle_component is not None and bundle_component.base_url is not None
                else request.base_url
            )
        )
        api_key = (
            request.action_api_key
            if _request_field_set(request, "action_api_key")
            else (
                bundle_component.api_key
                if bundle_component is not None and bundle_component.api_key is not None
                else request.api_key
            )
        )
        temperature = (
            request.action_temperature
            if _request_field_set(request, "action_temperature")
            else (
                bundle_component.temperature
                if bundle_component is not None and bundle_component.temperature is not None
                else request.temperature
            )
        )
        max_tokens = (
            request.action_max_tokens
            if _request_field_set(request, "action_max_tokens")
            else (
                bundle_component.max_tokens
                if bundle_component is not None and bundle_component.max_tokens is not None
                else request.max_tokens
            )
        )
        request_timeout = (
            request.action_request_timeout_seconds
            if _request_field_set(request, "action_request_timeout_seconds")
            else (
                bundle_component.request_timeout
                if bundle_component is not None and bundle_component.request_timeout is not None
                else request.request_timeout_seconds
            )
        )
    else:
        provider = (
            request.provider
            if _request_field_set(request, "provider") or bundle_component is None
            else bundle_component.provider
        )
        model = (
            request.model
            if _request_field_set(request, "model") or bundle_component is None
            else bundle_component.model
        )
        base_url = (
            request.base_url
            if _request_field_set(request, "base_url") or bundle_component is None or bundle_component.base_url is None
            else bundle_component.base_url
        )
        api_key = (
            request.api_key
            if _request_field_set(request, "api_key") or bundle_component is None or bundle_component.api_key is None
            else bundle_component.api_key
        )
        temperature = (
            request.temperature
            if _request_field_set(request, "temperature") or bundle_component is None or bundle_component.temperature is None
            else bundle_component.temperature
        )
        max_tokens = (
            request.max_tokens
            if _request_field_set(request, "max_tokens") or bundle_component is None or bundle_component.max_tokens is None
            else bundle_component.max_tokens
        )
        request_timeout = (
            request.request_timeout_seconds
            if _request_field_set(request, "request_timeout_seconds") or bundle_component is None or bundle_component.request_timeout is None
            else bundle_component.request_timeout
        )

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_timeout": request_timeout,
        "provider_kwargs": dict(bundle_component.provider_kwargs if bundle_component is not None else {}),
    }


def _resolve_action_provider_name(request: BufferedInferenceRequest) -> Optional[str]:
    config = _resolve_component_config(request, action=True)
    return None if config is None else str(config["provider"])


def _resolve_action_model_name(request: BufferedInferenceRequest) -> Optional[str]:
    config = _resolve_component_config(request, action=True)
    return None if config is None else str(config["model"])


def _create_llm_provider(request: BufferedInferenceRequest, *, action: bool = False) -> Callable[[str], Any]:
    from ncpu.self_optimizing.llm_provider import LLMProviderFactory

    config = _resolve_component_config(request, action=action)
    if config is None:
        raise ValueError("No action-policy provider configuration supplied")

    return LLMProviderFactory.create_provider(
        provider=config["provider"],
        model=config["model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        request_timeout=config["request_timeout"],
        **config["provider_kwargs"],
    )


def _load_request_latent_action_policy(request: BufferedInferenceRequest):
    from ncpu.self_optimizing.controller_runtime import load_bundle_latent_action_policy

    return load_bundle_latent_action_policy(
        controller_bundle_path=request.controller_bundle_path,
    )


def _load_request_latent_halt_policy(request: BufferedInferenceRequest):
    from ncpu.self_optimizing.controller_runtime import load_bundle_latent_halt_policy

    return load_bundle_latent_halt_policy(
        controller_bundle_path=request.controller_bundle_path,
    )


def _load_request_latent_memory_updater(request: BufferedInferenceRequest):
    from ncpu.self_optimizing.controller_runtime import load_bundle_latent_memory_updater

    return load_bundle_latent_memory_updater(
        controller_bundle_path=request.controller_bundle_path,
    )


def _build_buffered_inference_response(
    *,
    request: BufferedInferenceRequest,
    verification_mode: VerificationMode,
    workspace: Any,
    execution_time_ms: float,
    trajectory_path: Optional[str],
) -> BufferedInferenceResponse:
    hidden_workspace = workspace.snapshot() if request.include_hidden_trace else None
    response_config = _resolve_component_config(request)
    return BufferedInferenceResponse(
        task_name=workspace.task_name,
        category=workspace.category,
        status=workspace.status,
        provider=response_config["provider"],
        model=response_config["model"],
        controller_bundle_path=request.controller_bundle_path,
        action_provider=_resolve_action_provider_name(request),
        action_model=_resolve_action_model_name(request),
        verification_mode=verification_mode,
        committed_output=workspace.committed_output,
        committed_verified=workspace.committed_verified,
        generation_attempts=workspace.generation_attempts,
        max_generation_attempts=workspace.max_generation_attempts,
        hidden_step_count=len(workspace.steps),
        hidden_actions=[step.action for step in workspace.steps],
        last_error=workspace.last_error,
        last_verification=workspace.last_verification,
        execution_time_ms=execution_time_ms,
        trajectory_path=trajectory_path,
        hidden_workspace=hidden_workspace,
    )


def _sanitize_internal_event(event: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {
        "timestamp": event.get("timestamp", _utc_now()),
        "event": event.get("event"),
        "task_name": event.get("task_name"),
        "status": event.get("status"),
        "step_index": event.get("step_index"),
        "action": event.get("action"),
        "success": event.get("success"),
        "error": event.get("error") or event.get("last_error"),
        "committed_verified": event.get("committed_verified"),
        "generation_attempts": event.get("generation_attempts"),
    }
    return {key: value for key, value in sanitized.items() if value is not None}


def _build_buffered_task_status_response(task: Dict[str, Any]) -> BufferedInferenceTaskStatusResponse:
    return BufferedInferenceTaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        task_name=task["task_name"],
        category=task["category"],
        provider=task["provider"],
        model=task["model"],
        controller_bundle_path=task.get("controller_bundle_path"),
        action_provider=task.get("action_provider"),
        action_model=task.get("action_model"),
        verification_mode=task["verification_mode"],
        created_at=task["created_at"],
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        execution_time_ms=task.get("execution_time_ms", 0.0),
        hidden_step_count=task.get("hidden_step_count", 0),
        last_event=task.get("last_event"),
        last_error=task.get("last_error"),
        committed_verified=task.get("committed_verified", False),
        committed_output=task.get("committed_output"),
        trajectory_path=task.get("trajectory_path"),
    )


def _build_buffered_task_events(task: Dict[str, Any]) -> List[BufferedInferenceEvent]:
    return [
        BufferedInferenceEvent(sequence=index, **event)
        for index, event in enumerate(task.get("events", []), start=1)
    ]


def _run_buffered_inference_task(task_id: str, request: BufferedInferenceRequest) -> None:
    from ncpu.self_optimizing.internal_controller import (
        BufferedInternalController,
        InternalControllerConfig,
        InternalDeliberationTask,
    )

    task = buffered_inference_task_store.get_task(task_id)
    if task is None:
        return

    buffered_inference_task_store.update_task(
        task_id,
        {
            "status": TaskStatus.RUNNING,
            "started_at": _utc_now(),
        },
    )

    started = time.perf_counter()
    try:
        verification_mode, verifier, test_cases = _build_request_verifier(request)
        llm_provider = _create_llm_provider(request)
        action_provider = _create_llm_provider(request, action=True) if _has_action_policy_config(request) else None
        trajectory_path = task.get("trajectory_path")
        controller = BufferedInternalController(
            llm_provider=llm_provider,
            action_provider=action_provider,
            latent_action_policy=_load_request_latent_action_policy(request),
            latent_memory_updater=_load_request_latent_memory_updater(request),
            latent_halt_policy=_load_request_latent_halt_policy(request),
            config=_resolve_internal_controller_config(request),
            trajectory_logger=InternalTaskEventLogger(
                task_store=buffered_inference_task_store,
                task_id=task_id,
                trajectory_path=trajectory_path,
            ),
        )
        workspace = controller.run_task(
            InternalDeliberationTask(
                name=task["task_name"],
                prompt=request.prompt,
                verifier=verifier,
                test_cases=test_cases,
                category=request.category,
                response_format=request.response_format,
            )
        )
        buffered_inference_task_store.update_task(
            task_id,
            {
                "status": TaskStatus.SUCCESS if workspace.status == "committed" else TaskStatus.FAILED,
                "completed_at": _utc_now(),
                "execution_time_ms": (time.perf_counter() - started) * 1000.0,
                "committed_verified": workspace.committed_verified,
                "committed_output": workspace.committed_output,
                "last_error": workspace.last_error,
                "verification_mode": verification_mode,
            },
        )
    except Exception as exc:
        buffered_inference_task_store.append_event(
            task_id,
            _sanitize_internal_event(
                {
                    "timestamp": _utc_now(),
                    "event": "task_exception",
                    "task_name": task["task_name"],
                    "status": "failed",
                    "error": str(exc),
                }
            ),
        )
        buffered_inference_task_store.update_task(
            task_id,
            {
                "status": TaskStatus.FAILED,
                "completed_at": _utc_now(),
                "execution_time_ms": (time.perf_counter() - started) * 1000.0,
                "last_error": str(exc),
            },
        )


def _stream_buffered_task_events(task_id: str):
    sent = 0
    while True:
        task = buffered_inference_task_store.get_task(task_id)
        if task is None:
            payload = {"event": "task_missing", "task_id": task_id}
            yield f"data: {json.dumps(payload)}\n\n"
            return

        events = task.get("events", [])
        while sent < len(events):
            payload = {
                "sequence": sent + 1,
                **events[sent],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            sent += 1

        if task["status"] in BufferedInferenceTaskStore.TERMINAL_STATUSES:
            yield f"data: {json.dumps({'event': 'stream_complete', 'task_id': task_id, 'status': task['status']})}\n\n"
            return

        time.sleep(0.1)


# =============================================================================
# FastAPI App
# =============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SOME API",
        description="Self-Optimizing Machine Engine - REST API",
        version="1.0.0",
    )

    @app.get("/")
    def root():
        """Root endpoint with API info."""
        return {
            "name": "SOME API",
            "version": "1.0.0",
            "description": "Self-Optimizing Machine Engine REST API",
            "endpoints": {
                "health": "/health",
                "submit_task": "/tasks (POST)",
                "get_task": "/tasks/{task_id} (GET)",
                "list_tasks": "/tasks (GET)",
                "cancel_task": "/tasks/{task_id}/cancel (POST)",
                "model_zoo": "/models (GET)",
                "execute_model": "/models/{name}/execute (POST)",
                "optimize": "/optimize (POST)",
                "benchmark": "/benchmark (POST)",
                "internal_infer": "/internal/infer (POST)",
                "internal_submit": "/internal/tasks (POST)",
                "internal_status": "/internal/tasks/{task_id} (GET)",
                "internal_events": "/internal/tasks/{task_id}/events (GET)",
                "internal_stream": "/internal/tasks/{task_id}/stream (GET)",
            },
        }

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": _utc_now(),
            "gpu_available": True,  # Would check actual GPU
        }

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    @app.post("/tasks", response_model=TaskResponse)
    def submit_task(request: TaskRequest, background_tasks: BackgroundTasks):
        """Submit a new task for execution."""
        task_id = task_store.create_task(request)

        # Run task in background
        background_tasks.add_task(execute_task_async, task_id)

        return TaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=task_store.get_task(task_id)["created_at"],
        )

    @app.get("/tasks/{task_id}", response_model=ExecutionResultResponse)
    def get_task_result(task_id: str):
        """Get the result of a task."""
        task = task_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        return ExecutionResultResponse(**task)

    @app.get("/tasks", response_model=List[ExecutionResultResponse])
    def list_tasks(status: Optional[TaskStatus] = None):
        """List all tasks, optionally filtered by status."""
        return task_store.list_tasks(status)

    @app.post("/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        """Cancel a running task."""
        task = task_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        if task["status"] in (TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED):
            raise HTTPException(status_code=400, detail="Task already completed")

        task_store.update_task(task_id, {"status": TaskStatus.CANCELLED})
        return {"task_id": task_id, "status": "cancelled"}

    # -------------------------------------------------------------------------
    # Model Zoo
    # -------------------------------------------------------------------------

    @app.get("/models", response_model=List[ModelInfo])
    def list_models():
        """List all available models in the zoo."""
        from ncpu.self_optimizing.model_zoo import get_model_zoo, OperationCategory

        zoo = get_model_zoo()
        models = []

        for name in zoo.list_models():
            model = zoo.get(name)
            if model:
                models.append(ModelInfo(
                    name=model.name,
                    category=model.category.value,
                    description=model.description,
                    accuracy=model.accuracy,
                    avg_execution_time_ms=model.avg_execution_time_ms,
                ))

        return models

    @app.post("/models/{name}/execute")
    def execute_model(name: str, input_data: Dict[str, Any]):
        """Execute a pre-trained model from the zoo."""
        from ncpu.self_optimizing.model_zoo import get_model_zoo

        zoo = get_model_zoo()
        model = zoo.get(name)

        if model is None:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

        # Execute the model (would use actual executor in production)
        # For now, return mock result
        return {
            "model": name,
            "operation": model.descriptor.operation,
            "input": input_data,
            "output": {"result": "mock_output"},
            "execution_time_ms": model.avg_execution_time_ms,
        }

    # -------------------------------------------------------------------------
    # Optimization Endpoints
    # -------------------------------------------------------------------------

    @app.post("/optimize")
    def optimize_parameters(
        objective: str,
        parameters: Dict[str, float],
        method: str = "genetic",
    ):
        """Optimize parameters using genetic algorithm or other methods."""
        # Would use actual optimizer
        return {
            "objective": objective,
            "method": method,
            "best_parameters": parameters,
            "best_fitness": 0.95,
            "iterations": 50,
            "execution_time_ms": 100.0,
        }

    @app.post("/benchmark")
    def run_benchmark(
        operations: List[str],
        input_sizes: List[int],
    ):
        """Run benchmark on specified operations."""
        results = []

        for op in operations:
            for size in input_sizes:
                results.append({
                    "operation": op,
                    "input_size": size,
                    "execution_time_ms": size * 0.001,  # Mock
                    "memory_bytes": size * 4,
                    "throughput": size / (size * 0.001),
                })

        return {
            "operations": operations,
            "input_sizes": input_sizes,
            "results": results,
            "total_time_ms": sum(r["execution_time_ms"] for r in results),
        }

    @app.post("/internal/infer", response_model=BufferedInferenceResponse)
    def buffered_internal_infer(request: BufferedInferenceRequest):
        """
        Run hidden think-run-patch-commit inference and return only the committed result.

        This endpoint is the first API surface for the "internal CPU" approximation:
        the model can plan, draft, verify, repair, and only then emit user-visible output.
        """
        from ncpu.self_optimizing.internal_controller import (
            BufferedInternalController,
            InternalControllerConfig,
            InternalDeliberationTask,
        )
        from ncpu.self_optimizing.trajectory_logger import TrajectoryLogger

        try:
            verification_mode, verifier, test_cases = _build_request_verifier(request)
            _load_request_bundle(request)
            llm_provider = _create_llm_provider(request)
            action_provider = _create_llm_provider(request, action=True) if _has_action_policy_config(request) else None
            trajectory_path = _resolve_trajectory_path(request)
            controller = BufferedInternalController(
                llm_provider=llm_provider,
                action_provider=action_provider,
                latent_action_policy=_load_request_latent_action_policy(request),
                latent_memory_updater=_load_request_latent_memory_updater(request),
                latent_halt_policy=_load_request_latent_halt_policy(request),
                config=_resolve_internal_controller_config(request),
                trajectory_logger=TrajectoryLogger(trajectory_path),
            )
            task_name = request.task_name or _slugify_task_name(request.prompt[:48])
            task = InternalDeliberationTask(
                name=task_name,
                prompt=request.prompt,
                verifier=verifier,
                test_cases=test_cases,
                category=request.category,
                response_format=request.response_format,
            )

            started = time.perf_counter()
            workspace = controller.run_task(task)
            execution_time_ms = (time.perf_counter() - started) * 1000.0
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"buffered inference failed: {exc}") from exc

        return _build_buffered_inference_response(
            request=request,
            verification_mode=verification_mode,
            workspace=workspace,
            execution_time_ms=execution_time_ms,
            trajectory_path=trajectory_path,
        )

    @app.post("/internal/tasks", response_model=BufferedInferenceTaskResponse)
    def submit_buffered_internal_task(request: BufferedInferenceRequest):
        """Submit buffered inference as an async task with live sanitized event tracking."""
        try:
            verification_mode, _, _ = _build_request_verifier(request)
            _load_request_bundle(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        trajectory_path = _resolve_trajectory_path(request)
        task_id = buffered_inference_task_store.create_task(
            request=request,
            verification_mode=verification_mode,
            trajectory_path=trajectory_path,
        )
        worker = threading.Thread(
            target=_run_buffered_inference_task,
            args=(task_id, request),
            daemon=True,
        )
        worker.start()
        task = buffered_inference_task_store.get_task(task_id)
        return BufferedInferenceTaskResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            task_name=task["task_name"],
            created_at=task["created_at"],
        )

    @app.get("/internal/tasks/{task_id}", response_model=BufferedInferenceTaskStatusResponse)
    def get_buffered_internal_task(task_id: str):
        """Fetch async buffered inference task status and final committed output if complete."""
        task = buffered_inference_task_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Buffered inference task not found")
        return _build_buffered_task_status_response(task)

    @app.get("/internal/tasks/{task_id}/events", response_model=List[BufferedInferenceEvent])
    def list_buffered_internal_task_events(task_id: str):
        """Return sanitized hidden events for an async buffered inference task."""
        task = buffered_inference_task_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Buffered inference task not found")
        return _build_buffered_task_events(task)

    @app.get("/internal/tasks/{task_id}/stream")
    def stream_buffered_internal_task_events(task_id: str):
        """Server-sent event stream of sanitized hidden actions."""
        task = buffered_inference_task_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Buffered inference task not found")
        return StreamingResponse(
            _stream_buffered_task_events(task_id),
            media_type="text/event-stream",
        )

    # -------------------------------------------------------------------------
    # WebSocket for Real-time Updates
    # -------------------------------------------------------------------------

    # Note: WebSocket support would require additional setup
    # @app.websocket("/ws/tasks/{task_id}")
    # async def task_updates(websocket: WebSocket, task_id: str):
    #     """WebSocket for real-time task updates."""
    #     await websocket.accept()
    #     while True:
    #         task = task_store.get_task(task_id)
    #         if task:
    #             await websocket.send_json(task)
    #         if task["status"] in (TaskStatus.SUCCESS, TaskStatus.FAILED):
    #             break
    #         await asyncio.sleep(0.5)
    #     await websocket.close()


# =============================================================================
# Task Execution Logic
# =============================================================================

def execute_task_async(task_id: str):
    """Execute a task asynchronously."""
    task = task_store.get_task(task_id)
    if task is None:
        return

    # Update status to running
    task_store.update_task(task_id, {"status": TaskStatus.RUNNING})

    try:
        # In production, this would:
        # 1. Generate code using LLM
        # 2. Execute on GPU
        # 3. Verify result
        # 4. Apply feedback

        # Simulate execution
        time.sleep(0.1)  # Mock delay

        # Mark as success
        task_store.update_task(task_id, {
            "status": TaskStatus.SUCCESS,
            "output": {"result": "mock_result"},
            "execution_time_ms": 100.0,
            "iterations_completed": task["max_iterations"],
            "completed_at": _utc_now(),
            "feedback_signal": {
                "feedback_type": "success",
                "improvement_direction": "positive",
                "correctness_score": 1.0,
            },
        })

    except Exception as e:
        task_store.update_task(task_id, {
            "status": TaskStatus.FAILED,
            "error": str(e),
            "completed_at": _utc_now(),
        })


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="SOME API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if not FASTAPI_AVAILABLE:
        print("Error: fastapi and uvicorn required. Install with:")
        print("  pip install fastapi uvicorn")
        return

    uvicorn.run(
        "ncpu.self_optimizing.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
