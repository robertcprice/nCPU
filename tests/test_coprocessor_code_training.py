import sys
from types import SimpleNamespace

from ncpu.coprocessor.code_arithmetic_data import CodeArithmeticGenerator
from ncpu.coprocessor import train as train_module
from ncpu.self_optimizing import run_code_embedded_training as training_script


def test_accumulator_expressions_use_numeric_operands():
    sample = CodeArithmeticGenerator(seed=123).generate_accumulator_sample()
    assert "x " not in sample.arithmetic_expr


def test_evaluate_training_accuracy_prefers_code_eval(monkeypatch):
    monkeypatch.setattr(
        train_module,
        "evaluate_code_accuracy",
        lambda *args, **kwargs: {"overall_accuracy": 0.75},
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_arithmetic_accuracy",
        lambda *args, **kwargs: {"overall_accuracy": 0.25},
    )

    result = train_module.evaluate_training_accuracy(
        model=None,
        tokenizer=None,
        dataset="synthetic+code",
        device="cpu",
    )

    assert result["overall_accuracy"] == 0.75


def test_run_code_embedded_training_honors_dataset_alias(monkeypatch, capsys):
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return SimpleNamespace(
            steps_completed=1,
            final_loss=0.0,
            eval_accuracy=0.5,
            trainable_params=123,
            wall_time_seconds=0.1,
        )

    monkeypatch.setattr(training_script, "train_coprocessor", fake_train)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_code_embedded_training.py",
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--dataset",
            "code+synthetic",
        ],
    )

    training_script.main()

    assert captured["config"].dataset == "synthetic+code"
    assert "code_embedded" in captured["config"].output_dir
    capsys.readouterr()
