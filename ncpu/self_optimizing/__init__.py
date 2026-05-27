"""
Self-Optimizing Machine Engine (SOME).

Keep package import lightweight so standalone benchmark runners can import
`ncpu.self_optimizing.*` on stripped-down benchmark boxes without pulling in the
full engine dependency graph.
"""

__all__: list[str] = []

try:
    from ncpu.self_optimizing.core.engine import (
        ExecutionResult,
        SelfOptimizingEngine,
        Task,
        VerifiedCodeGenerator,
    )

    __all__ += [
        "SelfOptimizingEngine",
        "VerifiedCodeGenerator",
        "ExecutionResult",
        "Task",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.executable_thought_head import (
        ExecutableThoughtHead,
        ExecutableThoughtHeadConfig,
        ExecutableThoughtResult,
        ExecutableThoughtSmokeMetrics,
        build_executable_thought_smoke_batch,
        infer_executable_thought_hidden_dim,
        load_executable_thought_head,
        run_executable_thought_smoke_train,
        train_executable_thought_head,
    )
    from ncpu.self_optimizing.executable_thought_training import (
        ExecutableThoughtTrainingExample,
        build_executable_thought_dataset,
        build_executable_thought_training_bundle,
        build_executable_thought_training_examples,
        write_executable_thought_dataset,
    )
    from ncpu.self_optimizing.executable_thought_evaluation import (
        ExecutableThoughtEvalMetrics,
        evaluate_executable_thought_head,
    )

    __all__ += [
        "ExecutableThoughtHeadConfig",
        "ExecutableThoughtResult",
        "ExecutableThoughtSmokeMetrics",
        "ExecutableThoughtHead",
        "build_executable_thought_smoke_batch",
        "infer_executable_thought_hidden_dim",
        "load_executable_thought_head",
        "run_executable_thought_smoke_train",
        "train_executable_thought_head",
        "ExecutableThoughtTrainingExample",
        "build_executable_thought_training_examples",
        "build_executable_thought_dataset",
        "build_executable_thought_training_bundle",
        "write_executable_thought_dataset",
        "ExecutableThoughtEvalMetrics",
        "evaluate_executable_thought_head",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.core.gradient_feedback import (
        CodePattern,
        ExecutionSignal,
        FeedbackType,
        GradientFeedbackSystem,
        ImprovementDirection,
    )

    __all__ += [
        "GradientFeedbackSystem",
        "ExecutionSignal",
        "FeedbackType",
        "ImprovementDirection",
        "CodePattern",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.core.model_integration import (
        ExecutionVerifiedModel,
        ModelInterface,
        MultiCandidateExplorer,
        TensorDescriptor,
    )

    __all__ += [
        "TensorDescriptor",
        "ModelInterface",
        "ExecutionVerifiedModel",
        "MultiCandidateExplorer",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.core.autoresearch_agent import (
        AutoresearchSOMEAgent,
        ExperimentCandidate,
        ExperimentResult,
        GradientFeedback,
    )

    __all__ += [
        "AutoresearchSOMEAgent",
        "ExperimentCandidate",
        "ExperimentResult",
        "GradientFeedback",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.core.experiment_runner import (
        CheckpointManager,
        Experiment,
        ExperimentConfig,
        ModelWeights,
        SOMExperimentRunner,
    )

    __all__ += [
        "SOMExperimentRunner",
        "ExperimentConfig",
        "Experiment",
        "ModelWeights",
        "CheckpointManager",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.core.gradient_search import (
        GradientGuidedSearch,
        MultiObjectiveSearch,
        SearchDirection,
        SearchPattern,
        SearchSuggestion,
    )

    __all__ += [
        "GradientGuidedSearch",
        "MultiObjectiveSearch",
        "SearchSuggestion",
        "SearchPattern",
        "SearchDirection",
    ]
except ImportError:
    pass

try:
    from ncpu.self_optimizing.controller.bundle import (
        ControllerBundle,
        ControllerComponentConfig,
        load_controller_bundle,
        save_controller_bundle,
    )
    from ncpu.self_optimizing.controller.runtime import (
        ResolvedControllerRuntime,
        load_bundle_latent_action_policy,
        load_bundle_latent_halt_policy,
        load_bundle_latent_memory_updater,
        resolve_controller_runtime,
    )
    from ncpu.self_optimizing.architecture.weight_cpu_blueprint import (
        WeightCPUBlueprint,
        WeightCPUTrainingStage,
        WeightMemorySpec,
        LatentComputeSpec,
        build_default_weight_cpu_blueprint,
        load_weight_cpu_blueprint,
        save_weight_cpu_blueprint,
    )
    from ncpu.self_optimizing.controller.training import (
        ACTION_LABELS,
        ActionPolicyExample,
        ControllerTrainingBundle,
        build_action_policy_examples,
        build_controller_training_bundle,
    )
    from ncpu.self_optimizing.latent_heads.action_training import (
        LatentActionTrainingExample,
        build_latent_action_dataset,
        build_latent_action_training_bundle,
        build_latent_action_training_examples,
        train_latent_action_head,
        write_latent_action_dataset,
    )
    from ncpu.self_optimizing.latent_heads.halt_training import (
        LatentHaltTrainingExample,
        build_latent_halt_dataset,
        build_latent_halt_training_bundle,
        build_latent_halt_training_examples,
        train_latent_halt_head,
        write_latent_halt_dataset,
    )
    from ncpu.self_optimizing.controller.hidden_workspace import HiddenWorkspace, WorkspaceStep
    from ncpu.self_optimizing.latent_heads.controller_state import LatentControllerState
    from ncpu.self_optimizing.latent_heads.action_policy import (
        LATENT_ACTION_LABELS,
        LatentActionDecision,
        LatentActionHead,
        LatentActionHeadConfig,
        LatentActionPolicy,
        encode_latent_action_features,
        load_latent_action_head,
    )
    from ncpu.self_optimizing.latent_heads.descriptor_head import (
        LatentDescriptorGenerator,
        LatentDescriptorHead,
        LatentDescriptorHeadConfig,
        encode_latent_descriptor_features,
        load_latent_descriptor_head,
    )
    from ncpu.self_optimizing.latent_heads.descriptor_training import (
        LatentDescriptorTrainingExample,
        build_latent_descriptor_dataset,
        build_latent_descriptor_training_bundle,
        build_latent_descriptor_training_examples,
        train_latent_descriptor_head,
        write_latent_descriptor_dataset,
    )
    from ncpu.self_optimizing.latent_heads.memory_head import (
        LatentMemoryHead,
        LatentMemoryHeadConfig,
        LatentMemoryUpdater,
        encode_latent_memory_features,
        load_latent_memory_head,
    )
    from ncpu.self_optimizing.latent_heads.memory_training import (
        LatentMemoryTrainingExample,
        build_latent_memory_dataset,
        build_latent_memory_training_bundle,
        build_latent_memory_training_examples,
        train_latent_memory_head,
        write_latent_memory_dataset,
    )
    from ncpu.self_optimizing.latent_heads.halt_policy import (
        LATENT_HALT_LABELS,
        LatentHaltDecision,
        LatentHaltHead,
        LatentHaltHeadConfig,
        LatentHaltPolicy,
        encode_latent_halt_features,
        load_latent_halt_head,
    )
    from ncpu.self_optimizing.controller.internal_controller import (
        BufferedInternalController,
        InternalControllerConfig,
        InternalDeliberationTask,
        InternalModelResponse,
    )
    from ncpu.self_optimizing.architecture.task_local_fast_weights import (
        FastWeightLinear,
        FastWeightUpdateResult,
        HFTaskLocalFastWeightsProvider,
        TaskLocalFastWeightConfig,
        find_target_linear_modules,
    )
    from ncpu.self_optimizing.architecture.ncpu_adaptation_backend import (
        NCPUAdaptationBackend,
        NCPUAdaptationConfig,
        NCPUAdaptationDescriptor,
        NCPUAdaptationSession,
    )
    from ncpu.self_optimizing.latent_heads.state_patch_head import (
        StatePatchHead,
        StatePatchHeadConfig,
        load_state_patch_head,
    )
    from ncpu.self_optimizing.latent_heads.state_patch_training import (
        StatePatchTrainingExample,
        build_state_patch_dataset,
        build_state_patch_training_bundle,
        build_state_patch_training_examples,
        train_state_patch_head,
        write_state_patch_dataset,
    )
    from ncpu.self_optimizing.architecture.segmented_kv_cache import (
        SegmentDescriptor,
        SegmentedKVCacheConfig,
        SegmentedKVCacheState,
        compress_hidden_segment,
    )
    from ncpu.self_optimizing.controller.recurrent_commit_policy import (
        RecurrentCommitDecision,
        RecurrentCommitPolicy,
        RecurrentCommitPolicyConfig,
    )
    from ncpu.self_optimizing.controller.descriptor_decode_runtime import (
        DescriptorDecodeConfig,
        DescriptorDecodeRuntime,
    )
    from ncpu.self_optimizing.core.sandbox_actions import SandboxActionResult, SandboxActionRunner
    from ncpu.self_optimizing.training.trajectory_dataset import (
        DistillationExample,
        LoadedTrajectory,
        build_distillation_dataset,
        build_distillation_examples,
        iter_trajectories,
        load_trajectory,
        summarize_distillation_dataset,
        write_distillation_dataset,
    )
    from ncpu.self_optimizing.training.trajectory_logger import TrajectoryLogger

    __all__ += [
        "ControllerBundle",
        "ControllerComponentConfig",
        "load_controller_bundle",
        "save_controller_bundle",
        "ResolvedControllerRuntime",
        "resolve_controller_runtime",
        "load_bundle_latent_action_policy",
        "load_bundle_latent_halt_policy",
        "load_bundle_latent_memory_updater",
        "WeightCPUBlueprint",
        "WeightCPUTrainingStage",
        "WeightMemorySpec",
        "LatentComputeSpec",
        "build_default_weight_cpu_blueprint",
        "load_weight_cpu_blueprint",
        "save_weight_cpu_blueprint",
        "ACTION_LABELS",
        "ActionPolicyExample",
        "ControllerTrainingBundle",
        "build_action_policy_examples",
        "build_controller_training_bundle",
        "LatentActionTrainingExample",
        "build_latent_action_training_examples",
        "build_latent_action_dataset",
        "build_latent_action_training_bundle",
        "write_latent_action_dataset",
        "train_latent_action_head",
        "LatentHaltTrainingExample",
        "build_latent_halt_training_examples",
        "build_latent_halt_dataset",
        "build_latent_halt_training_bundle",
        "write_latent_halt_dataset",
        "train_latent_halt_head",
        "HiddenWorkspace",
        "WorkspaceStep",
        "LatentControllerState",
        "LATENT_ACTION_LABELS",
        "LatentActionDecision",
        "LatentActionHeadConfig",
        "LatentActionHead",
        "LatentActionPolicy",
        "encode_latent_action_features",
        "load_latent_action_head",
        "LatentDescriptorHeadConfig",
        "LatentDescriptorHead",
        "LatentDescriptorGenerator",
        "encode_latent_descriptor_features",
        "load_latent_descriptor_head",
        "LatentDescriptorTrainingExample",
        "build_latent_descriptor_training_examples",
        "build_latent_descriptor_dataset",
        "build_latent_descriptor_training_bundle",
        "write_latent_descriptor_dataset",
        "train_latent_descriptor_head",
        "LatentMemoryHeadConfig",
        "LatentMemoryHead",
        "LatentMemoryUpdater",
        "encode_latent_memory_features",
        "load_latent_memory_head",
        "LatentMemoryTrainingExample",
        "build_latent_memory_training_examples",
        "build_latent_memory_dataset",
        "build_latent_memory_training_bundle",
        "write_latent_memory_dataset",
        "train_latent_memory_head",
        "LATENT_HALT_LABELS",
        "LatentHaltDecision",
        "LatentHaltHeadConfig",
        "LatentHaltHead",
        "LatentHaltPolicy",
        "encode_latent_halt_features",
        "load_latent_halt_head",
        "BufferedInternalController",
        "InternalControllerConfig",
        "InternalDeliberationTask",
        "InternalModelResponse",
        "TaskLocalFastWeightConfig",
        "FastWeightUpdateResult",
        "FastWeightLinear",
        "find_target_linear_modules",
        "HFTaskLocalFastWeightsProvider",
        "NCPUAdaptationConfig",
        "NCPUAdaptationDescriptor",
        "NCPUAdaptationSession",
        "NCPUAdaptationBackend",
        "StatePatchHeadConfig",
        "StatePatchHead",
        "load_state_patch_head",
        "StatePatchTrainingExample",
        "build_state_patch_training_examples",
        "build_state_patch_dataset",
        "build_state_patch_training_bundle",
        "write_state_patch_dataset",
        "train_state_patch_head",
        "SegmentDescriptor",
        "SegmentedKVCacheConfig",
        "SegmentedKVCacheState",
        "compress_hidden_segment",
        "RecurrentCommitDecision",
        "RecurrentCommitPolicy",
        "RecurrentCommitPolicyConfig",
        "DescriptorDecodeConfig",
        "DescriptorDecodeRuntime",
        "SandboxActionResult",
        "SandboxActionRunner",
        "LoadedTrajectory",
        "DistillationExample",
        "load_trajectory",
        "iter_trajectories",
        "build_distillation_examples",
        "build_distillation_dataset",
        "summarize_distillation_dataset",
        "write_distillation_dataset",
        "TrajectoryLogger",
    ]
except ImportError:
    pass
