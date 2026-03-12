"""Pure-Python tests for shared trace/disassembly helpers."""

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_UTILS_PATH = PROJECT_ROOT / "ncpu/os/gpu/programs/tools/trace_utils.py"
_SPEC = importlib.util.spec_from_file_location("trace_utils_local", TRACE_UTILS_PATH)
_TRACE_UTILS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TRACE_UTILS)


def test_disassemble_instruction_nop():
    assert _TRACE_UTILS.disassemble_instruction(0xD503201F) == "nop"


def test_disassemble_instruction_movz():
    inst = 0xD2800000 | (42 << 5)
    assert _TRACE_UTILS.disassemble_instruction(inst) == "movz x0, #0x2a"


def test_render_trace_table_contains_decoded_instruction():
    table = _TRACE_UTILS.render_trace_table([
        (0x1000, 0xD2800000 | (42 << 5), 42, 0, 0, 0, 0, 0),
    ], limit=1)

    assert "PC" in table
    assert "movz x0, #0x2a" in table
    assert "0x00001000" in table
