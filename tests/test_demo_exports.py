import json
from types import SimpleNamespace

from demos.interactive_discovery import InteractiveDiscovery
from demos.neural_text_machine import export_program_text, save_result_summary


def test_interactive_discovery_export_program(tmp_path):
    repl = InteractiveDiscovery()
    repl.last_result = SimpleNamespace(
        accuracy=1.0,
        loss_history=[0.0],
        program_text="ADD R2, R0, R1\nHALT",
    )
    out = tmp_path / "program.asm"
    repl.export_program(str(out))
    assert out.read_text() == "ADD R2, R0, R1\nHALT\n"


def test_interactive_discovery_save_and_load_session(tmp_path):
    repl = InteractiveDiscovery()
    repl.examples = [({0: 1.0, 1: 2.0}, {2: 3.0})]
    repl.num_input_regs = 2
    repl.num_output_regs = 1
    repl.last_result = SimpleNamespace(
        accuracy=1.0,
        loss_history=[0.0],
        program_text="ADD R2, R0, R1\nHALT",
    )
    out = tmp_path / "session.json"
    repl.save_session(str(out))
    payload = json.loads(out.read_text())
    assert payload["num_input_regs"] == 2
    assert payload["examples"][0]["inputs"]["0"] == 1.0

    other = InteractiveDiscovery()
    other.load_session(str(out))
    assert len(other.examples) == 1
    assert other.num_input_regs == 2
    assert other.num_output_regs == 1


def test_neural_text_machine_export_program(tmp_path):
    result = SimpleNamespace(program_text="MOV R1, R0\nHALT")
    out = tmp_path / "text_program.asm"
    export_program_text(result, str(out))
    assert out.read_text() == "MOV R1, R0\nHALT\n"


def test_neural_text_machine_save_summary(tmp_path):
    result = SimpleNamespace(accuracy=0.75, program_text="MOV R1, R0\nHALT")
    out = tmp_path / "summary.json"
    save_result_summary("cipher hello -> khoor", result, str(out))
    payload = json.loads(out.read_text())
    assert payload["label"] == "cipher hello -> khoor"
    assert payload["accuracy"] == 0.75
