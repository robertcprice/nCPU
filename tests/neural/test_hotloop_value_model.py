import json
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ncpu.neural.hotloop_value_model import (
    HOTLOOP_VALUE_FEATURE_NAMES,
    build_hotloop_value_feature_tensor,
    extract_hotloop_value_examples,
    load_hotloop_value_model,
    predict_hotloop_value_score,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_build_hotloop_value_feature_tensor_includes_region_tail_fields():
    sample = {
        "segment": 2,
        "body_word_count": 5,
        "estimated_iterations": 12,
        "estimated_work": 60,
        "pre_sync_bytes": 128,
        "post_sync_bytes": 64,
        "remaining_instructions": 2048,
        "tail_word_count": 3,
        "synthetic_stop": False,
        "region_blocks": 2,
        "nested_branch_count": 1,
        "branch_kind": "bcond",
        "tail_max_imm16": 12288,
        "tail_large_imm16_count": 2,
        "reused_state": True,
        "previous_pre_sync_bytes": 4000,
        "previous_post_sync_bytes": 2000,
        "previous_region_blocks": 2,
        "previous_tail_word_count": 3,
        "previous_tail_max_imm16": 2000,
        "previous_tail_large_imm16_count": 0,
    }

    features = build_hotloop_value_feature_tensor(sample)

    assert tuple(features.shape) == (len(HOTLOOP_VALUE_FEATURE_NAMES),)
    tail_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("tail_word_count")
    synthetic_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("synthetic_stop")
    region_blocks_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("region_blocks")
    branch_bcond_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("branch_kind_bcond")
    tail_max_imm16_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("tail_max_imm16")
    tail_large_imm16_count_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("tail_large_imm16_count")
    segment_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("segment")
    reused_state_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("reused_state")
    prev_pre_sync_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_pre_sync_bytes")
    prev_tail_max_imm16_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_tail_max_imm16")
    assert float(features[tail_idx].item()) == 3.0
    assert float(features[synthetic_idx].item()) == 0.0
    assert float(features[region_blocks_idx].item()) == 2.0
    assert float(features[branch_bcond_idx].item()) == 1.0
    assert float(features[tail_max_imm16_idx].item()) == 12288.0
    assert float(features[tail_large_imm16_count_idx].item()) == 2.0
    assert float(features[segment_idx].item()) == 2.0
    assert float(features[reused_state_idx].item()) == 1.0
    assert float(features[prev_pre_sync_idx].item()) == 4000.0
    assert float(features[prev_tail_max_imm16_idx].item()) == 2000.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_extract_hotloop_value_examples_reconstructs_remaining_budget():
    payload = {
        "results": [
            {
                "workload": "counted",
                "backend": "rust-hotloop",
                "result_ok": True,
                "insts_ok": True,
                "best_rust_speedup_vs_neural": 2.0,
                "hotloop_trace": [
                    {
                        "segment": 1,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 5,
                        "estimated_work": 10,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_after": 9993,
                        "executed_count": 17,
                        "tail_word_count": 0,
                        "synthetic_stop": False,
                    }
                ],
            }
        ],
    }

    examples = extract_hotloop_value_examples(payload)

    assert len(examples) == 1
    assert examples[0]["features"][5] == pytest.approx(10010.0, rel=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_extract_hotloop_value_examples_infers_previous_segment_context():
    payload = {
        "results": [
            {
                "workload": "adjacent-counted",
                "backend": "rust-hotloop",
                "result_ok": True,
                "insts_ok": True,
                "best_rust_speedup_vs_neural": 1.25,
                "hotloop_samples": [
                    {
                        "segment": 1,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 5,
                        "estimated_work": 10,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_instructions": 100,
                        "tail_word_count": 3,
                        "region_blocks": 2,
                        "tail_max_imm16": 2000,
                        "tail_large_imm16_count": 0,
                    },
                    {
                        "segment": 2,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 5,
                        "estimated_work": 10,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_instructions": 90,
                        "tail_word_count": 0,
                        "region_blocks": 1,
                        "tail_max_imm16": 0,
                        "tail_large_imm16_count": 0,
                    },
                ],
            }
        ],
    }

    examples = extract_hotloop_value_examples(payload)
    previous_pre_sync_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_pre_sync_bytes")
    previous_region_blocks_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_region_blocks")
    previous_tail_word_count_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_tail_word_count")
    previous_tail_max_imm16_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("previous_tail_max_imm16")
    reused_state_idx = HOTLOOP_VALUE_FEATURE_NAMES.index("reused_state")

    assert len(examples) == 2
    assert examples[1]["features"][previous_pre_sync_idx] == pytest.approx(0.0, rel=1e-6)
    assert examples[1]["features"][previous_region_blocks_idx] == pytest.approx(2.0, rel=1e-6)
    assert examples[1]["features"][previous_tail_word_count_idx] == pytest.approx(3.0, rel=1e-6)
    assert examples[1]["features"][previous_tail_max_imm16_idx] == pytest.approx(2000.0, rel=1e-6)
    assert examples[1]["features"][reused_state_idx] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_extract_hotloop_value_examples_prefers_torch_baseline_speedup():
    payload = {
        "results": [
            {
                "workload": "counted",
                "backend": "rust-hotloop",
                "result_ok": True,
                "insts_ok": True,
                "best_rust_speedup_vs_neural": 0.9,
                "hotloop_speedup_vs_torch": 1.5,
                "hotloop_samples": [
                    {
                        "segment": 1,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 5,
                        "estimated_work": 10,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_instructions": 100,
                        "tail_word_count": 0,
                        "synthetic_stop": False,
                    }
                ],
            }
        ],
    }

    examples = extract_hotloop_value_examples(payload)

    assert len(examples) == 1
    assert examples[0]["speedup_ratio"] == pytest.approx(1.5, rel=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_predict_hotloop_value_score_accepts_mapping_and_tuple_output():
    class TupleModel:
        def __call__(self, features):
            return torch.tensor([0.73], device=features.device), torch.tensor([1.0], device=features.device)

    sample = {
        "body_word_count": 5,
        "estimated_iterations": 12,
        "estimated_work": 60,
        "pre_sync_bytes": 128,
        "post_sync_bytes": 64,
        "remaining_instructions": 2048,
        "tail_word_count": 3,
        "synthetic_stop": False,
        "region_blocks": 2,
        "nested_branch_count": 1,
        "branch_kind": "bcond",
        "tail_max_imm16": 12288,
        "tail_large_imm16_count": 2,
    }

    score = predict_hotloop_value_score(TupleModel(), sample)

    assert score == pytest.approx(0.73, rel=1e-6)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_train_hotloop_value_model_cli_smoke(tmp_path: Path):
    matrix_path = tmp_path / "gpu_only_matrix.json"
    model_path = tmp_path / "hotloop_value_model.pt"
    payload = {
        "generated_at": "2026-04-16T00:00:00+00:00",
        "results": [
            {
                "workload": "counted",
                "backend": "rust-hotloop",
                "result_ok": True,
                "insts_ok": True,
                "best_rust_speedup_vs_neural": 3.0,
                "hotloop_samples": [
                    {
                        "segment": 1,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 256,
                        "estimated_work": 512,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_instructions": 8192,
                        "tail_word_count": 0,
                        "synthetic_stop": True,
                        "executed_count": 513,
                        "elapsed_seconds": 0.01,
                        "observed_ips": 51300.0,
                    },
                    {
                        "segment": 2,
                        "approved": True,
                        "body_word_count": 5,
                        "estimated_iterations": 64,
                        "estimated_work": 320,
                        "pre_sync_bytes": 16,
                        "post_sync_bytes": 16,
                        "remaining_instructions": 4096,
                        "tail_word_count": 3,
                        "synthetic_stop": False,
                        "executed_count": 323,
                        "elapsed_seconds": 0.01,
                        "observed_ips": 32300.0,
                    },
                ],
            },
            {
                "workload": "bytecopy",
                "backend": "rust-hotloop",
                "result_ok": True,
                "insts_ok": True,
                "best_rust_speedup_vs_neural": 0.95,
                "hotloop_samples": [
                    {
                        "segment": 1,
                        "approved": True,
                        "body_word_count": 2,
                        "estimated_iterations": 32,
                        "estimated_work": 64,
                        "pre_sync_bytes": 0,
                        "post_sync_bytes": 0,
                        "remaining_instructions": 2048,
                        "tail_word_count": 0,
                        "synthetic_stop": False,
                        "executed_count": 127,
                        "elapsed_seconds": 0.01,
                        "observed_ips": 12700.0,
                    }
                ],
            },
        ],
    }
    matrix_path.write_text(json.dumps(payload, indent=2) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "ncpu/neural/train_hotloop_value_model.py"),
            "--input",
            str(matrix_path),
            "--output",
            str(model_path),
            "--epochs",
            "5",
            "--batch-size",
            "2",
            "--negative-augmentations",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert model_path.is_file()

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    assert metadata["validation_mode"] == "in-sample"
    assert metadata["recommended_threshold_1_2x"] < 1.0

    model = load_hotloop_value_model(model_path, device="cpu")
    sample = {
        "body_word_count": 4,
        "estimated_iterations": 128,
        "estimated_work": 512,
        "pre_sync_bytes": 0,
        "post_sync_bytes": 0,
        "remaining_instructions": 4096,
        "tail_word_count": 2,
        "synthetic_stop": False,
    }
    with torch.no_grad():
        score = float(model(build_hotloop_value_feature_tensor(sample)).item())
    assert 0.0 <= score <= 1.0
