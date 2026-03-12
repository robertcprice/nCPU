"""Blueprints for evolving SOME toward an internal weight-compute architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional


RESEARCH_REFERENCES = [
    {
        "title": "Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
        "year": 2024,
        "url": "https://arxiv.org/abs/2407.04620",
        "relevance": "fast weights / test-time training in hidden state",
    },
    {
        "title": "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking",
        "year": 2024,
        "url": "https://arxiv.org/abs/2403.09629",
        "relevance": "hidden reasoning before output",
    },
    {
        "title": "Training Large Language Models to Reason in a Continuous Latent Space",
        "year": 2024,
        "url": "https://arxiv.org/abs/2412.06769",
        "relevance": "continuous latent thought instead of explicit tokens",
    },
    {
        "title": "Titans: Learning to Memorize at Test Time",
        "year": 2025,
        "url": "https://arxiv.org/abs/2501.00663",
        "relevance": "persistent neural memory at test time",
    },
    {
        "title": "Fast Quiet-STaR: Thinking Without Thought Tokens",
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.17746",
        "relevance": "internalized reasoning with lower inference overhead",
    },
    {
        "title": "Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection",
        "year": 2025,
        "url": "https://arxiv.org/abs/2505.19475",
        "relevance": "verifier-guided low-rank adaptation during inference",
    },
    {
        "title": "End-to-End Test-Time Training for Long Context",
        "year": 2025,
        "url": "https://arxiv.org/abs/2512.23675",
        "relevance": "compressing context into weights during inference",
    },
    {
        "title": "Looped Transformers as Programmable Computers",
        "year": 2023,
        "url": "https://arxiv.org/abs/2301.13196",
        "relevance": "transformers used as iterative programmable compute",
    },
]


@dataclass
class WeightMemorySpec:
    """How weights are allowed to change during hidden inference."""

    mechanism: str
    update_scope: list[str] = field(default_factory=list)
    rank: Optional[int] = None
    learning_rate: Optional[float] = None
    max_update_steps: Optional[int] = None
    trigger: str = "verify_failure_or_uncertainty"


@dataclass
class LatentComputeSpec:
    """Hidden compute loop before any visible commit."""

    latent_steps: int
    hidden_state_type: str
    halt_policy: str
    verifier_visible_to_user: bool = False
    external_tool_visible_to_user: bool = False


@dataclass
class WeightCPUTrainingStage:
    """Ordered training stage for the architecture roadmap."""

    name: str
    objective: str
    data_source: str
    trainable_components: list[str]
    success_metric: str


@dataclass
class WeightCPUBlueprint:
    """Serializable architecture blueprint for the long-term SOME direction."""

    name: str
    version: int = 1
    controller_bundle_path: Optional[str] = None
    current_runtime: str = "external_buffered_controller"
    target_runtime: str = "latent_controller_with_fast_weights"
    short_term_memory: str = "attention_and_hidden_workspace"
    long_term_memory: str = "task-local_fast_weights"
    latent_compute: LatentComputeSpec = field(
        default_factory=lambda: LatentComputeSpec(
            latent_steps=8,
            hidden_state_type="continuous_thought_state",
            halt_policy="learned_commit_or_fail",
        )
    )
    weight_memory: WeightMemorySpec = field(
        default_factory=lambda: WeightMemorySpec(
            mechanism="low_rank_test_time_training",
            update_scope=["attention.q_proj", "attention.v_proj", "mlp.up_proj", "mlp.down_proj"],
            rank=8,
            learning_rate=1e-4,
            max_update_steps=4,
        )
    )
    verifier_bridge: dict[str, Any] = field(default_factory=dict)
    training_stages: list[WeightCPUTrainingStage] = field(default_factory=list)
    benchmark_targets: list[str] = field(default_factory=list)
    research_references: list[dict[str, Any]] = field(default_factory=lambda: list(RESEARCH_REFERENCES))
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WeightCPUBlueprint":
        return cls(
            name=str(payload["name"]),
            version=int(payload.get("version", 1)),
            controller_bundle_path=payload.get("controller_bundle_path"),
            current_runtime=str(payload.get("current_runtime", "external_buffered_controller")),
            target_runtime=str(payload.get("target_runtime", "latent_controller_with_fast_weights")),
            short_term_memory=str(payload.get("short_term_memory", "attention_and_hidden_workspace")),
            long_term_memory=str(payload.get("long_term_memory", "task-local_fast_weights")),
            latent_compute=LatentComputeSpec(**payload.get("latent_compute", {})),
            weight_memory=WeightMemorySpec(**payload.get("weight_memory", {})),
            verifier_bridge=dict(payload.get("verifier_bridge") or {}),
            training_stages=[
                WeightCPUTrainingStage(**stage)
                for stage in payload.get("training_stages", [])
            ],
            benchmark_targets=list(payload.get("benchmark_targets") or []),
            research_references=list(payload.get("research_references") or []),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "controller_bundle_path": self.controller_bundle_path,
            "current_runtime": self.current_runtime,
            "target_runtime": self.target_runtime,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
            "latent_compute": asdict(self.latent_compute),
            "weight_memory": asdict(self.weight_memory),
            "verifier_bridge": self.verifier_bridge,
            "training_stages": [asdict(stage) for stage in self.training_stages],
            "benchmark_targets": self.benchmark_targets,
            "research_references": self.research_references,
            "metadata": self.metadata,
        }


def build_default_weight_cpu_blueprint(
    *,
    name: str,
    controller_bundle_path: Optional[str] = None,
) -> WeightCPUBlueprint:
    """Construct the default research-backed roadmap from current SOME to weight CPU."""
    return WeightCPUBlueprint(
        name=name,
        controller_bundle_path=controller_bundle_path,
        verifier_bridge={
            "current_mode": "external_execute_verify_feedback",
            "target_mode": "hidden verifier state plus occasional external execution",
            "commit_rule": "only emit visible output after latent verifier confidence or external pass",
        },
        training_stages=[
            WeightCPUTrainingStage(
                name="stage1_hidden_controller_distillation",
                objective="distill external SOME trajectories into hidden think/write/patch/commit behavior",
                data_source="benchmarks/internal_trajectories",
                trainable_components=["response_adapter", "action_adapter"],
                success_metric="higher pass@1 with fewer visible retries",
            ),
            WeightCPUTrainingStage(
                name="stage2_latent_reasoning_internalization",
                objective="replace textual hidden thoughts with latent recurrent thought states",
                data_source="successful hidden controller trajectories plus latent-thought finetuning",
                trainable_components=["latent_compute_head", "halt_head"],
                success_metric="same-or-better benchmark accuracy with fewer thought tokens",
            ),
            WeightCPUTrainingStage(
                name="stage3_fast_weight_adaptation",
                objective="update small task-local low-rank weights during inference before commit",
                data_source="verifier-selected self-generated patches and failures",
                trainable_components=["fast_weight_adapter"],
                success_metric="improved hard-benchmark pass rate without external retry loops",
            ),
            WeightCPUTrainingStage(
                name="stage4_meta_training_of_weight_updates",
                objective="train the model to learn how to update its own task-local weights efficiently",
                data_source="benchmark tasks with verifier rewards",
                trainable_components=["update_rule", "fast_weight_adapter", "commit_head"],
                success_metric="fewer update steps and better generalization on unseen tasks",
            ),
        ],
        benchmark_targets=[
            "HumanEval+",
            "MBPP+",
            "BigCodeBench-Hard",
        ],
        metadata={
            "design_goal": "approximate a CPU-like hidden compute substrate inside model state and small weight updates",
            "implementation_bias": "prefer low-rank task-local fast weights over full-model online finetuning",
        },
    )


def save_weight_cpu_blueprint(blueprint: WeightCPUBlueprint, path: str | Path) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(blueprint.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_weight_cpu_blueprint(path: str | Path) -> WeightCPUBlueprint:
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return WeightCPUBlueprint.from_dict(payload)
