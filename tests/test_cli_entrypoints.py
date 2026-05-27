from __future__ import annotations

import ncpu.__main__ as entry
import ncpu.demo as demo
import ncpu.lab as lab


def test_python_m_ncpu_help_prints_cli_usage(capsys):
    rc = entry.main(["--help"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "nCPU flagship launcher" in out
    assert "discover" in out
    assert "doctor" in out


def test_python_m_ncpu_demo_subcommand_dispatches_to_demo(monkeypatch):
    seen = {}

    def fake_demo(argv):
        seen["argv"] = list(argv)
        return 17

    monkeypatch.setattr(entry, "_run_demo", fake_demo)

    rc = entry.main(["demo", "--headless", "--multiproc"])

    assert rc == 17
    assert seen["argv"] == ["--headless", "--multiproc"]


def test_demo_registry_commands_use_top_level_cli():
    assert lab.DEMO_REGISTRY["discover"]["command"] == "python -m ncpu discover"
    assert lab.DEMO_REGISTRY["text"]["command"] == "python -m ncpu text --interactive"
    assert lab.DEMO_REGISTRY["busybox"]["command"] == "python -m ncpu systems busybox --interactive"
    assert lab.DEMO_REGISTRY["alpine"]["command"] == "python -m ncpu systems alpine --demo"
    assert lab.DEMO_REGISTRY["full-neural"]["command"] == "python -m ncpu full-neural"
    assert lab.DEMO_REGISTRY["meta-compare"]["command"] == "python -m ncpu meta-compare"
    assert lab.DEMO_REGISTRY["coprocessor"]["command"] == "python -m ncpu coprocessor --help-only"


def test_full_neural_subcommand_dispatches_to_registered_demo(monkeypatch):
    seen = {}

    def fake_run_demo(name, argv=None):
        seen["name"] = name
        seen["argv"] = list(argv or [])
        return 23

    monkeypatch.setattr(lab, "_run_demo", fake_run_demo)

    rc = lab.main(
        [
            "full-neural",
            "--device",
            "cpu",
            "--max-instructions",
            "123",
            "--output",
            "/tmp/full-neural.png",
            "--summary-json",
            "/tmp/full-neural.json",
        ]
    )

    assert rc == 23
    assert seen["name"] == "full-neural"
    assert seen["argv"] == [
        "full_neural_demo.py",
        "--device",
        "cpu",
        "--max-instructions",
        "123",
        "--output",
        "/tmp/full-neural.png",
        "--summary-json",
        "/tmp/full-neural.json",
    ]


def test_demo_full_neural_flag_dispatches_to_bottom_up_script(monkeypatch):
    seen = {}

    def fake_run_path(path, run_name):
        seen["path"] = path
        seen["run_name"] = run_name

    monkeypatch.setattr(demo.runpy, "run_path", fake_run_path)

    rc = demo.main(["--full-neural", "--device", "cpu"])

    assert rc == 0
    assert seen["path"].endswith("demos/neural/full_neural_demo.py")
    assert seen["run_name"] == "__main__"


def test_meta_compare_subcommand_dispatches_to_registered_demo(monkeypatch):
    seen = {}

    def fake_run_demo(name, argv=None):
        seen["name"] = name
        seen["argv"] = list(argv or [])
        return 29

    monkeypatch.setattr(lab, "_run_demo", fake_run_demo)

    rc = lab.main(
        [
            "meta-compare",
            "--left-runtime",
            "neural-os",
            "--shell",
            "/bin/sh",
            "--device",
            "cpu",
            "--scale",
            "2",
            "--command",
            "pwd",
            "--command",
            "python3 --version",
            "--capture-dir",
            "/tmp/meta-compare-captures",
            "--summary-json",
            "/tmp/meta-compare.json",
            "--shell-log",
            "/tmp/meta-compare.log",
            "--boot-delay-ms",
            "50",
            "--step-delay-ms",
            "75",
            "--final-hold-ms",
            "25",
            "--max-frames",
            "3",
            "--output",
            "/tmp/meta-compare.png",
        ]
    )

    assert rc == 29
    assert seen["name"] == "meta-compare"
    assert seen["argv"] == [
        "meta_comparison_demo.py",
        "--left-runtime",
        "neural-os",
        "--shell",
        "/bin/sh",
        "--device",
        "cpu",
        "--scale",
        "2",
        "--command",
        "pwd",
        "--command",
        "python3 --version",
        "--capture-dir",
        "/tmp/meta-compare-captures",
        "--summary-json",
        "/tmp/meta-compare.json",
        "--shell-log",
        "/tmp/meta-compare.log",
        "--boot-delay-ms",
        "50",
        "--step-delay-ms",
        "75",
        "--final-hold-ms",
        "25",
        "--max-frames",
        "3",
        "--output",
        "/tmp/meta-compare.png",
    ]


def test_demo_meta_compare_flag_dispatches_to_comparison_script(monkeypatch):
    seen = {}

    def fake_run_path(path, run_name):
        seen["path"] = path
        seen["run_name"] = run_name

    monkeypatch.setattr(demo.runpy, "run_path", fake_run_path)

    rc = demo.main(["--meta-compare", "--device", "cpu"])

    assert rc == 0
    assert seen["path"].endswith("demos/neural/meta_comparison_demo.py")
    assert seen["run_name"] == "__main__"
