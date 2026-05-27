"""Regression tests for the GPU-only NeuralCPU execution engine."""

import struct

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ncpu.neural.hotloop_value_model import HOTLOOP_VALUE_FEATURE_NAMES

from benchmarks.benchmark_gpu_only import (
    build_adjacent_counted_program,
    build_adjacent_bytecopy_bge_exit_program,
    build_adjacent_bytecopy_program,
    build_bytecopy_program,
    build_bytecopy_cbz_then_bge_exit_program,
    build_bytecopy_then_counted_program,
    build_counted_then_bytecopy_program,
    build_nested_counted_program,
    build_program,
    expected_engine_executed_count,
    load_program,
    make_bytecopy_payload,
    reset_pc,
)


def movz(rd: int, imm16: int) -> int:
    return 0xD2800000 | (imm16 << 5) | rd


def add_reg(rd: int, rn: int, rm: int) -> int:
    return 0x8B000000 | (rm << 16) | (rn << 5) | rd


def add_imm(rd: int, rn: int, imm12: int) -> int:
    return 0x91000000 | (imm12 << 10) | (rn << 5) | rd


def adds_imm(rd: int, rn: int, imm12: int) -> int:
    return 0xB1000000 | (imm12 << 10) | (rn << 5) | rd


def subs_imm(rd: int, rn: int, imm12: int) -> int:
    return 0xF1000000 | (imm12 << 10) | (rn << 5) | rd


def strb(rt: int, rn: int, imm12: int = 0) -> int:
    return 0x39000000 | (imm12 << 10) | (rn << 5) | rt


def ldrb(rt: int, rn: int, imm12: int = 0) -> int:
    return 0x39400000 | (imm12 << 10) | (rn << 5) | rt


def bcond(offset_words: int, cond: int) -> int:
    return 0x54000000 | ((offset_words & 0x7FFFF) << 5) | cond


def cbnz(rt: int, offset_words: int) -> int:
    return 0xB5000000 | ((offset_words & 0x7FFFF) << 5) | rt


def cbz(rt: int, offset_words: int) -> int:
    return 0xB4000000 | ((offset_words & 0x7FFFF) << 5) | rt


def b_uncond(offset_words: int) -> int:
    return 0x14000000 | (offset_words & 0x3FFFFFF)


def cmp_reg(rn: int, rm: int) -> int:
    return 0xEB000000 | (rm << 16) | (rn << 5) | 31


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestGpuOnlyEngine:
    @pytest.fixture
    def cpu(self):
        from ncpu.neural.cpu import NeuralCPU

        cpu = NeuralCPU(device_override="cpu", fast_mode=False)
        cpu._hazard_predictor = None
        cpu._dep_predictor = None
        cpu._neural_scheduler = None
        cpu._neural_loop_detector = None
        cpu._neural_branch_predictor = None
        return cpu

    def test_loop_benchmark_instruction_accounting(self, cpu):
        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert int(cpu.regs[0].item()) == n_iters
        assert executed == expected_engine_executed_count(n_iters)

    def test_torch_bytecopy_loop_honors_store_data_dependency(self, cpu, monkeypatch):
        n_iters = 4
        load_addr = 0x10000
        code = build_bytecopy_program(n_iters, src_addr=0x2000, dst_addr=0x3000)
        payload = make_bytecopy_payload(n_iters, seed=3)

        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "off")
        load_program(cpu, code, load_addr)
        cpu.memory[0x2000:0x2000 + n_iters] = torch.tensor(
            list(payload), dtype=torch.uint8, device=cpu.device
        )
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters, "bytecopy")
        assert cpu._last_gpu_only_backend == "torch-gpu-only"
        assert bytes(cpu.memory[0x3000:0x3000 + n_iters].detach().cpu().tolist()) == payload

    def test_torch_counted_then_bytecopy_preserves_copy_result(self, cpu, monkeypatch):
        n_iters = 4
        load_addr = 0x10000
        code = build_counted_then_bytecopy_program(n_iters, src_addr=0x2000, dst_addr=0x3000)
        payload = make_bytecopy_payload(n_iters, seed=7)

        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "off")
        load_program(cpu, code, load_addr)
        cpu.memory[0x2000:0x2000 + n_iters] = torch.tensor(
            list(payload), dtype=torch.uint8, device=cpu.device
        )
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters, "counted-bytecopy")
        assert int(cpu.regs[0].item()) == n_iters
        assert int(cpu.regs[2].item()) == 0
        assert cpu._last_gpu_only_backend == "torch-gpu-only"
        assert bytes(cpu.memory[0x3000:0x3000 + n_iters].detach().cpu().tolist()) == payload

    def test_fused_subs_branch_uses_immediately_preceding_compare(self, cpu):
        load_addr = 0x10000
        insts = [
            movz(1, 1),           # X1 = 1
            adds_imm(2, 31, 1),   # ADDS X2, XZR, #1 -> Z = 0
            subs_imm(1, 1, 1),    # SUBS X1, X1, #1 -> Z = 1
            bcond(2, 0),          # B.EQ target
            movz(0, 7),           # wrong path if stale flags are used
            movz(0, 42),          # target
            0x00000000,           # HALT
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.run_gpu_only(max_instructions=1_000, batch_size=64)

        assert int(cpu.regs[0].item()) == 42
        assert int(cpu.regs[1].item()) == 0

    def test_branch_predictor_hook_runs_for_backward_three_op_loops(self, cpu):
        class StubBranchPredictor:
            def __init__(self):
                self._flag_history = None
                self.update_calls = 0
                self.predict_calls = []

            def update_flag_history(self, flags, device):
                self.update_calls += 1
                if self._flag_history is None:
                    self._flag_history = torch.zeros(4, 4, device=device)
                self._flag_history = torch.cat(
                    [self._flag_history[1:], flags[:4].unsqueeze(0)],
                    dim=0,
                )

            def __call__(self, cond_code, flags, is_backward, counter_hint=0.0):
                self.predict_calls.append((cond_code, bool(is_backward), float(counter_hint)))
                return torch.tensor(0.95, device=flags.device)

        load_addr = 0x10000
        stub = StubBranchPredictor()
        cpu._neural_branch_predictor = stub

        insts = [
            add_reg(0, 0, 2),     # loop body op 1
            add_reg(4, 4, 3),     # loop body op 2
            subs_imm(1, 1, 1),    # loop body op 3
            bcond((-3) & 0x7FFFF, 1),  # B.NE back to first ADD
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.regs[1] = 4
        cpu.regs[2] = 1
        cpu.regs[3] = 1
        cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert int(cpu.regs[0].item()) == 4
        assert int(cpu.regs[4].item()) == 4
        assert stub.update_calls >= 1
        assert stub.predict_calls
        assert all(is_backward for _, is_backward, _ in stub.predict_calls)

    def test_forced_rust_hotloop_handoff_normalizes_instruction_count(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 18
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10018

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self._regs[0] = 5
                self._regs[1] = 0
                self._pc = 0x10018
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert int(cpu.regs[1].item()) == 0
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_chains_adjacent_counted_loops(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10000
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    self._regs[0] = 3
                    self._regs[1] = 0
                    self._regs[2] = 1
                    self._pc = 0x10018
                    return FakeResult(12)
                if self.execute_calls == 2:
                    self._regs[4] = 2
                    self._regs[5] = 0
                    self._regs[6] = 1
                    self._pc = 0x10030
                    return FakeResult(9)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_adjacent_counted_program(3, 2)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 19
        assert int(cpu.regs[0].item()) == 3
        assert int(cpu.regs[1].item()) == 0
        assert int(cpu.regs[4].item()) == 2
        assert int(cpu.regs[5].item()) == 0
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2
        assert cpu._last_gpu_only_hotloop_samples[1]["reused_state"] is True
        assert cpu._last_gpu_only_hotloop_samples[1]["previous_tail_max_imm16"] == 2
        assert cpu._last_gpu_only_hotloop_samples[1]["previous_region_blocks"] == 2

    def test_adjacent_counted_candidate_stops_before_second_loop_body(self, cpu):
        load_addr = 0x10000
        code = build_adjacent_counted_program(3, 2)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        candidate = cpu._collect_hotloop_candidate()

        assert candidate is not None
        assert candidate["synthetic_stop"] is True
        assert candidate["branch_idx"] == 5
        assert candidate["halt_idx"] == 9
        assert candidate["tail_word_count"] == 3

    def test_forced_rust_hotloop_chains_adjacent_bytecopy_loops(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    assert bytes(self._memory[0x2000:0x2004]) == b"ABCD"
                    self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                    self._regs[1] = 0x2004
                    self._regs[2] = 0
                    self._regs[3] = 0x3004
                    self._pc = 0x10024
                    return FakeResult(27)
                if self.execute_calls == 2:
                    assert bytes(self._memory[0x2800:0x2804]) == b"WXYZ"
                    assert bytes(self._memory[0x3000:0x3004]) == b"ABCD"
                    self._memory[0x3800:0x3804] = self._memory[0x2800:0x2804]
                    self._regs[1] = 0x2804
                    self._regs[2] = 0
                    self._regs[3] = 0x3804
                    self._pc = 0x1004C
                    return FakeResult(27)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_adjacent_bytecopy_program(4, first_src_addr=0x2000, first_dst_addr=0x3000, second_src_addr=0x2800, second_dst_addr=0x3800)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"ABCD"), dtype=torch.uint8)
        cpu.memory[0x2800:0x2804] = torch.tensor(list(b"WXYZ"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "adjacent-bytecopy")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"ABCD"
        assert bytes(cpu.memory[0x3800:0x3804].detach().cpu().tolist()) == b"WXYZ"
        assert int(cpu.regs[1].item()) == 0x2804
        assert int(cpu.regs[2].item()) == 0
        assert int(cpu.regs[3].item()) == 0x3804
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2
        assert cpu._last_gpu_only_hotloop_samples[1]["reused_state"] is True
        # pre_sync covers only the load source (4 bytes); the store destination
        # is tracked in post_sync, not pre_sync, since its pre-state is overwritten.
        assert cpu._last_gpu_only_hotloop_samples[1]["previous_pre_sync_bytes"] == 4
        assert cpu._last_gpu_only_hotloop_samples[1]["previous_post_sync_bytes"] == 4

    def test_forced_rust_hotloop_chains_adjacent_bge_exit_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    assert bytes(self._memory[0x2000:0x2004]) == b"QRST"
                    self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                    self._regs[1] = 0x2004
                    self._regs[2] = 4
                    self._regs[3] = 0x3004
                    self._regs[5] = 4
                    self._pc = 0x10030
                    return FakeResult(38)
                if self.execute_calls == 2:
                    assert bytes(self._memory[0x2800:0x2804]) == b"UVWX"
                    assert bytes(self._memory[0x3000:0x3004]) == b"QRST"
                    self._memory[0x3800:0x3804] = self._memory[0x2800:0x2804]
                    self._regs[1] = 0x2804
                    self._regs[2] = 4
                    self._regs[3] = 0x3804
                    self._regs[5] = 4
                    self._pc = 0x10064
                    return FakeResult(38)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_adjacent_bytecopy_bge_exit_program(4, first_src_addr=0x2000, first_dst_addr=0x3000, second_src_addr=0x2800, second_dst_addr=0x3800)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"QRST"), dtype=torch.uint8)
        cpu.memory[0x2800:0x2804] = torch.tensor(list(b"UVWX"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "adjacent-bytecopy-bge-exit")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"QRST"
        assert bytes(cpu.memory[0x3800:0x3804].detach().cpu().tolist()) == b"UVWX"
        assert int(cpu.regs[1].item()) == 0x2804
        assert int(cpu.regs[2].item()) == 4
        assert int(cpu.regs[3].item()) == 0x3804
        assert int(cpu.regs[5].item()) == 4
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2

    def test_forced_rust_hotloop_reuses_state_across_segments(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10000
                self.execute_calls = 0
                self.set_register_calls = 0
                self.set_flags_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self.set_register_calls += 1
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self.set_flags_calls += 1
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    self._regs[0] = 3
                    self._regs[1] = 0
                    self._regs[2] = 1
                    self._pc = 0x10018
                    return FakeResult(12)
                if self.execute_calls == 2:
                    self._regs[4] = 2
                    self._regs[5] = 0
                    self._regs[6] = 1
                    self._pc = 0x10030
                    return FakeResult(9)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_adjacent_counted_program(3, 2)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 19
        assert fake_cpu.execute_calls == 2
        assert fake_cpu.set_register_calls == 32
        assert fake_cpu.set_flags_calls == 1
        assert cpu._last_gpu_only_hotloop_stats["reused_state_segments"] == 1

    def test_forced_rust_hotloop_chains_counted_then_bytecopy(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    self._regs[0] = 4
                    self._regs[1] = 0
                    self._regs[2] = 1
                    self._pc = 0x10018
                    return FakeResult(15)
                if self.execute_calls == 2:
                    assert bytes(self._memory[0x2000:0x2004]) == b"ABCD"
                    self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                    self._regs[0] = 4
                    self._regs[1] = 0x2004
                    self._regs[2] = 0
                    self._regs[3] = 0x3004
                    self._pc = 0x10040
                    return FakeResult(27)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_counted_then_bytecopy_program(4, src_addr=0x2000, dst_addr=0x3000)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"ABCD"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "counted-bytecopy")
        assert int(cpu.regs[0].item()) == 4
        assert int(cpu.regs[2].item()) == 0
        assert int(cpu.regs[3].item()) == 0x3004
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"ABCD"
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2

    def test_forced_rust_hotloop_chains_bytecopy_then_counted(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    assert bytes(self._memory[0x2000:0x2004]) == b"WXYZ"
                    self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                    self._regs[1] = 0x2004
                    self._regs[2] = 0
                    self._regs[3] = 0x3004
                    self._pc = 0x10024
                    return FakeResult(27)
                if self.execute_calls == 2:
                    self._regs[0] = 4
                    self._regs[1] = 0
                    self._regs[2] = 1
                    self._pc = 0x1003C
                    return FakeResult(15)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_bytecopy_then_counted_program(4, src_addr=0x2000, dst_addr=0x3000)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"WXYZ"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy-counted")
        assert int(cpu.regs[0].item()) == 4
        assert int(cpu.regs[2].item()) == 1
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"WXYZ"
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2

    def test_forced_rust_hotloop_chains_cbz_then_bge_exit_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    assert bytes(self._memory[0x2000:0x2004]) == b"QRST"
                    self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                    self._regs[1] = 0x2004
                    self._regs[2] = 0
                    self._regs[3] = 0x3004
                    self._pc = 0x10028
                    return FakeResult(expected_engine_executed_count(4, "bytecopy-cbz-exit") + 1)
                if self.execute_calls == 2:
                    assert bytes(self._memory[0x2800:0x2804]) == b"UVWX"
                    assert bytes(self._memory[0x3000:0x3004]) == b"QRST"
                    self._memory[0x3800:0x3804] = self._memory[0x2800:0x2804]
                    self._regs[1] = 0x2804
                    self._regs[2] = 4
                    self._regs[3] = 0x3804
                    self._regs[5] = 4
                    self._pc = 0x1005C
                    return FakeResult(expected_engine_executed_count(4, "bytecopy-bge-exit") + 1)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_bytecopy_cbz_then_bge_exit_program(
            4,
            4,
            first_src_addr=0x2000,
            first_dst_addr=0x3000,
            second_src_addr=0x2800,
            second_dst_addr=0x3800,
        )
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"QRST"), dtype=torch.uint8)
        cpu.memory[0x2800:0x2804] = torch.tensor(list(b"UVWX"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy-cbz-then-bge-exit")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"QRST"
        assert bytes(cpu.memory[0x3800:0x3804].detach().cpu().tolist()) == b"UVWX"
        assert int(cpu.regs[2].item()) == 4
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert fake_cpu.execute_calls == 2

    def test_auto_hotloop_backend_rechecks_detector_for_each_segment(self, cpu, monkeypatch):
        class StubLoopDetector:
            def __init__(self):
                self.calls = 0

            def __call__(self, body_bits, regs):
                self.calls += 1
                logits = torch.tensor([0.0, 3.0], device=body_bits.device)
                return logits, torch.tensor([0.0], device=body_bits.device), torch.tensor([4.0], device=body_bits.device)

        class FakeResult:
            def __init__(self, cycles):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10000
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    self._regs[0] = 3
                    self._regs[1] = 0
                    self._regs[2] = 1
                    self._pc = 0x10018
                    return FakeResult(12)
                if self.execute_calls == 2:
                    self._regs[4] = 2
                    self._regs[5] = 0
                    self._regs[6] = 1
                    self._pc = 0x10030
                    return FakeResult(9)
                raise AssertionError("unexpected extra hotloop execute")

        import kernels.mlx.rust_runner as rust_runner

        detector = StubLoopDetector()
        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS", "1")
        cpu._neural_loop_detector = detector
        cpu._neural_hotloop_value_model = None

        load_addr = 0x10000
        code = build_adjacent_counted_program(3, 2)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 19
        assert detector.calls >= 2
        assert fake_cpu.execute_calls == 2
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert len(cpu._last_gpu_only_hotloop_samples) == 2
        assert len(cpu._last_gpu_only_hotloop_trace) == 2
        assert all(entry["approved"] for entry in cpu._last_gpu_only_hotloop_trace)

    def test_auto_hotloop_backend_falls_back_after_second_segment_rejection(self, cpu, monkeypatch):
        class StubLoopDetector:
            def __init__(self):
                self.calls = 0

            def __call__(self, body_bits, regs):
                self.calls += 1
                if self.calls == 1:
                    logits = torch.tensor([0.0, 3.0], device=body_bits.device)
                else:
                    logits = torch.tensor([3.0, 0.0], device=body_bits.device)
                return logits, torch.tensor([0.0], device=body_bits.device), torch.tensor([4.0], device=body_bits.device)

        class FakeResult:
            cycles = 12
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10000
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                self._regs[0] = 3
                self._regs[1] = 0
                self._regs[2] = 1
                self._pc = 0x10018
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        detector = StubLoopDetector()
        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS", "1")
        cpu._neural_loop_detector = detector
        cpu._neural_hotloop_value_model = None

        load_addr = 0x10000
        code = build_adjacent_counted_program(3, 2)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 19
        assert int(cpu.regs[0].item()) == 3
        assert int(cpu.regs[4].item()) == 2
        assert detector.calls >= 2
        assert fake_cpu.execute_calls == 1
        assert cpu._last_gpu_only_backend == "hybrid-hotloop+torch"
        assert cpu._last_gpu_only_hotloop_segments == 1
        assert len(cpu._last_gpu_only_hotloop_samples) == 2
        assert cpu._last_gpu_only_hotloop_trace[-1]["approved"] is False
        assert cpu._last_gpu_only_hotloop_trace[-1]["policy_reason"] == "detector-rejected"

    def test_auto_hotloop_policy_rejects_small_segments(self, cpu, monkeypatch):
        class StubLoopDetector:
            def __call__(self, body_bits, regs):
                logits = torch.tensor([0.0, 3.0], device=body_bits.device)
                return logits, torch.tensor([0.0], device=body_bits.device), torch.tensor([2.0], device=body_bits.device)

        class FakeRustCPU:
            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return getattr(self, "_pc", 0)

            def set_register(self, reg, value):
                pass

            def get_register(self, reg):
                return 0

            def set_flags(self, n, z, c, v):
                pass

            def get_flags(self):
                return (False, False, False, False)

            def execute(self, max_cycles=100_000):
                raise AssertionError("policy-rejected hotloop should not execute")

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_MIN_ESTIMATED_WORK", "1000")
        cpu._neural_loop_detector = StubLoopDetector()
        cpu._neural_hotloop_value_model = None

        n_iters = 3
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert cpu._last_gpu_only_backend == "torch-gpu-only"
        assert cpu._last_gpu_only_hotloop_segments == 0
        assert cpu._last_gpu_only_hotloop_trace[-1]["approved"] is False
        assert cpu._last_gpu_only_hotloop_trace[-1]["policy_reason"] == "estimated-work-below-threshold"

    def test_auto_hotloop_value_model_can_approve_small_segments(self, cpu, monkeypatch):
        class StubLoopDetector:
            def __call__(self, body_bits, regs):
                logits = torch.tensor([0.0, 3.0], device=body_bits.device)
                return logits, torch.tensor([0.0], device=body_bits.device), torch.tensor([2.0], device=body_bits.device)

        class FakeValueModel:
            def __init__(self):
                self.calls = []

            def __call__(self, feature_vec):
                self.calls.append(feature_vec.detach().cpu().tolist())
                return torch.tensor([0.80], device=feature_vec.device)

        class FakeResult:
            cycles = 18
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10018
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                self._regs[0] = 5
                self._regs[1] = 0
                self._pc = 0x10018
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        value_model = FakeValueModel()
        fake_cpu = FakeRustCPU()
        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: fake_cpu)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_MIN_BODY_WORDS", "64")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_VALUE_THRESHOLD", "0.70")
        cpu._neural_loop_detector = StubLoopDetector()
        cpu._neural_hotloop_value_model = value_model

        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert fake_cpu.execute_calls == 1
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 1
        assert cpu._last_gpu_only_hotloop_trace[0]["reason"] == "value-model"
        assert cpu._last_gpu_only_hotloop_samples[0]["policy_reason"] == "value-model"
        assert cpu._last_gpu_only_hotloop_samples[0]["observed_ips"] == pytest.approx(1700.0, rel=1e-5)
        assert cpu._last_gpu_only_hotloop_samples[0]["value_target"] > 0.0
        assert value_model.calls
        assert len(value_model.calls[0][0]) == len(HOTLOOP_VALUE_FEATURE_NAMES)

    def test_auto_hotloop_value_model_rejects_and_falls_back(self, cpu, monkeypatch):
        class StubLoopDetector:
            def __call__(self, body_bits, regs):
                logits = torch.tensor([0.0, 3.0], device=body_bits.device)
                return logits, torch.tensor([0.0], device=body_bits.device), torch.tensor([2.0], device=body_bits.device)

        class FakeValueModel:
            def __call__(self, feature_vec):
                return torch.tensor([0.40], device=feature_vec.device)

        class FakeRustCPU:
            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return getattr(self, "_pc", 0)

            def set_register(self, reg, value):
                pass

            def get_register(self, reg):
                return 0

            def set_flags(self, n, z, c, v):
                pass

            def get_flags(self):
                return (False, False, False, False)

            def execute(self, max_cycles=100_000):
                raise AssertionError("value-threshold rejection should not execute Rust")

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_ALLOW_CPU", "1")
        monkeypatch.setenv("NCPU_GPU_ONLY_AUTO_VALUE_THRESHOLD", "0.70")
        cpu._neural_loop_detector = StubLoopDetector()
        cpu._neural_hotloop_value_model = FakeValueModel()

        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert cpu._last_gpu_only_backend == "torch-gpu-only"
        assert cpu._last_gpu_only_hotloop_segments == 0
        assert cpu._last_gpu_only_hotloop_trace[-1]["approved"] is False
        assert cpu._last_gpu_only_hotloop_trace[-1]["policy_reason"] == "value-threshold"
        assert cpu._last_gpu_only_hotloop_trace[-1]["policy_score"] == pytest.approx(0.40, rel=1e-5)
        assert cpu._last_gpu_only_hotloop_samples[-1]["value_target"] == 0.0

    def test_forced_rust_hotloop_executes_post_loop_tail_region(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 15
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10024
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                self._regs[0] = 3
                self._regs[1] = 0
                self._regs[2] = 1
                self._regs[7] = 41
                self._regs[8] = 44
                self._regs[9] = 45
                self._pc = 0x10024
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(0, 0),
            movz(1, 3),
            movz(2, 1),
            add_reg(0, 0, 2),
            subs_imm(1, 1, 1),
            bcond((-2) & 0x7FFFF, 1),
            movz(7, 41),
            add_reg(8, 0, 7),
            add_reg(9, 8, 2),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 14
        assert int(cpu.regs[0].item()) == 3
        assert int(cpu.regs[8].item()) == 44
        assert int(cpu.regs[9].item()) == 45
        assert cpu.halted is True
        assert cpu._last_gpu_only_backend == "rust-hotloop"
        assert cpu._last_gpu_only_hotloop_segments == 1
        assert cpu._last_gpu_only_hotloop_trace[0]["tail_word_count"] == 3
        assert cpu._last_gpu_only_hotloop_trace[0]["region_blocks"] == 2
        assert cpu._last_gpu_only_hotloop_trace[0]["tail_max_imm16"] == 41
        assert cpu._last_gpu_only_hotloop_trace[0]["branch_kind_bcond"] == 1
        assert cpu._last_gpu_only_hotloop_samples[0]["tail_word_count"] == 3
        assert cpu._last_gpu_only_hotloop_samples[0]["region_blocks"] == 2
        assert cpu._last_gpu_only_hotloop_samples[0]["tail_max_imm16"] == 41
        assert cpu._last_gpu_only_hotloop_samples[0]["branch_kind_bcond"] == 1

    def test_forced_rust_superblock_executes_branchy_memory_region(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 8
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x1001C
                self._memory = bytearray(0x8000)
                self.execute_calls = 0

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                assert self._memory[0x2000] == ord("Q")
                self._memory[0x3000] = self._memory[0x2000]
                self._regs[0] = 1
                self._regs[1] = 0x2000
                self._regs[3] = 0x3000
                self._regs[4] = ord("Q")
                self._regs[5] = 7
                self._pc = 0x1001C
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(0, 1),
            movz(1, 0x2000),
            movz(3, 0x3000),
            cbz(0, 3),
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            movz(5, 7),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        cpu.memory[0x2000] = ord("Q")
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 7
        assert bytes(cpu.memory[0x3000:0x3001].detach().cpu().tolist()) == b"Q"
        assert int(cpu.regs[4].item()) == ord("Q")
        assert int(cpu.regs[5].item()) == 7
        assert cpu.halted is True
        assert cpu._last_gpu_only_backend == "rust-superblock"
        assert cpu._last_gpu_only_hotloop_segments == 1
        assert cpu._last_gpu_only_hotloop_trace[0]["region_kind"] == "superblock"
        assert cpu._last_gpu_only_hotloop_trace[0]["region_blocks"] == 2
        assert cpu._last_gpu_only_hotloop_samples[0]["region_kind"] == "superblock"

    def test_forced_rust_superblock_materializes_side_exit_path(self, cpu, monkeypatch):
        class FakeResult:
            def __init__(self, cycles, stop_reason_name="HALT"):
                self.cycles = cycles
                self.elapsed_seconds = 0.01
                self.stop_reason_name = stop_reason_name

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10000
                self._memory = bytearray(0x8000)
                self.execute_calls = 0
                self.loaded_words = []

            def load_program(self, binary, address=0x10000):
                payload = bytes(binary)
                self.loaded_words = list(struct.unpack(f"<{len(payload) // 4}I", payload))
                self._memory[address:address + len(payload)] = payload

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self.execute_calls += 1
                if self.execute_calls == 1:
                    assert self.loaded_words == [
                        movz(0, 1),
                        movz(1, 7),
                        movz(31, 0),
                        0x00000000,
                    ]
                    self._regs[0] = 1
                    self._regs[1] = 7
                    self._pc = 0x1000C
                    return FakeResult(cycles=4)
                assert self.loaded_words == [
                    movz(2, 42),
                    0x00000000,
                ]
                self._regs[2] = 42
                self._pc = 0x10018
                return FakeResult(cycles=2)

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_WORDS", "4")

        load_addr = 0x10000
        side_exit_pc = load_addr + (5 * 4)
        insts = [
            movz(0, 1),
            movz(1, 7),
            b_uncond(3),
            0xD4000001,
            movz(9, 100),
            movz(2, 42),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 4
        assert int(cpu.regs[0].item()) == 1
        assert int(cpu.regs[1].item()) == 7
        assert int(cpu.regs[2].item()) == 42
        assert cpu.halted is True
        assert int(cpu.pc.item()) == load_addr + (6 * 4)
        assert cpu._last_gpu_only_backend == "rust-superblock"
        assert cpu._last_gpu_only_hotloop_segments == 2
        assert cpu._last_gpu_only_hotloop_trace[0]["synthetic_stop"] is True
        assert cpu._last_gpu_only_hotloop_trace[0]["expected_stop_pc"] == side_exit_pc
        assert cpu._last_gpu_only_hotloop_trace[0]["resolved_stop_pc"] == side_exit_pc
        assert cpu._last_gpu_only_hotloop_trace[1]["pc"] == side_exit_pc
        assert cpu._last_gpu_only_hotloop_samples[0]["expected_stop_pc"] == side_exit_pc

    def test_forced_rust_superblock_materializes_revisited_trace(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 17
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10020
                self.loaded_words = []

            def load_program(self, binary, address=0x10000):
                payload = bytes(binary)
                self.loaded_words = list(struct.unpack(f"<{len(payload) // 4}I", payload))

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                return None

            def read_memory(self, address, size):
                return b"\x00" * size

            def execute(self, max_cycles=100_000):
                assert self.loaded_words == [
                    movz(0, 3),
                    movz(1, 0),
                    movz(31, 0),
                    add_reg(1, 1, 0),
                    subs_imm(0, 0, 1),
                    movz(31, 0),
                    movz(31, 0),
                    add_reg(1, 1, 0),
                    subs_imm(0, 0, 1),
                    movz(31, 0),
                    movz(31, 0),
                    add_reg(1, 1, 0),
                    subs_imm(0, 0, 1),
                    movz(31, 0),
                    movz(31, 0),
                    movz(2, 99),
                    0x00000000,
                ]
                self._regs[0] = 0
                self._regs[1] = 6
                self._regs[2] = 99
                self._pc = 0x10040
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setattr(cpu, "_collect_hotloop_candidate", lambda *args, **kwargs: None)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_WORDS", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_STEPS", "32")

        load_addr = 0x10000
        insts = [
            movz(0, 3),
            movz(1, 0),
            cbz(0, 4),
            add_reg(1, 1, 0),
            subs_imm(0, 0, 1),
            b_uncond((-3) & 0x3FFFFFF),
            movz(2, 99),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 16
        assert int(cpu.regs[0].item()) == 0
        assert int(cpu.regs[1].item()) == 6
        assert int(cpu.regs[2].item()) == 99
        assert cpu.halted is True
        assert cpu._last_gpu_only_backend == "rust-superblock"
        assert cpu._last_gpu_only_hotloop_segments == 1
        assert cpu._last_gpu_only_hotloop_trace[0]["region_kind"] == "superblock"
        assert cpu._last_gpu_only_hotloop_trace[0]["synthetic_stop"] is False
        assert cpu._last_gpu_only_hotloop_trace[0]["materialized_word_count"] == 16
        assert cpu._last_gpu_only_hotloop_trace[0]["region_blocks"] == 3

    def test_superblock_candidate_cache_reuses_same_entry_state(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}

        load_addr = 0x10000
        insts = [
            movz(0, 1),
            movz(1, 7),
            b_uncond(2),
            movz(9, 99),
            movz(2, 42),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        candidate_1 = cpu._collect_superblock_candidate()
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_entries"] == 1

    def test_superblock_candidate_cache_invalidates_when_guarded_memory_changes(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            ldrb(0, 1, 0),
            cbz(0, 3),
            movz(2, 7),
            0x00000000,
            movz(2, 9),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000] = 0

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        candidate_zero = cpu._collect_superblock_candidate()
        cpu.memory[0x2000] = 1
        candidate_nonzero = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_zero is not None
        assert candidate_nonzero is not None
        assert candidate_zero["cache_hit"] is False
        assert candidate_nonzero["cache_hit"] is False
        assert candidate_zero["code_bytes"] != candidate_nonzero["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_entries"] == 1

    def test_superblock_cache_priority_prefers_higher_value_model_candidate(self, cpu, monkeypatch):
        class FakeValueModel:
            def __call__(self, feature_vec):
                estimated_work = float(feature_vec[0][HOTLOOP_VALUE_FEATURE_NAMES.index("estimated_work")].item())
                score = 0.9 if estimated_work >= 3.0 else 0.1
                return torch.tensor([score], device=feature_vec.device)

        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "1")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._neural_hotloop_value_model = FakeValueModel()

        primary_addr = 0x10000
        secondary_addr = 0x11000
        primary_insts = [
            movz(0, 1),
            movz(1, 7),
            b_uncond(2),
            movz(9, 99),
            movz(2, 42),
            0x00000000,
        ]
        secondary_insts = [
            movz(3, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        primary_candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        secondary_candidate = cpu._collect_superblock_candidate()
        reset_pc(cpu, primary_addr)
        primary_candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert primary_candidate_1 is not None
        assert secondary_candidate is not None
        assert primary_candidate_2 is not None
        assert primary_candidate_1["cache_hit"] is False
        assert secondary_candidate["cache_hit"] is False
        assert primary_candidate_2["cache_hit"] is True
        assert primary_candidate_1["cache_priority_source"] == "value-model"
        assert secondary_candidate["cache_priority_source"] == "value-model"
        assert primary_candidate_1["cache_priority_score"] == pytest.approx(0.9, rel=1e-6)
        assert secondary_candidate["cache_priority_score"] == pytest.approx(0.1, rel=1e-6)
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_priority_rejections"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_entries"] == 1

    def test_superblock_template_cache_reuses_path_across_irrelevant_register_changes(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}

        load_addr = 0x10000
        insts = [
            movz(0, 1),
            movz(1, 7),
            b_uncond(2),
            movz(9, 99),
            movz(2, 42),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[11] = 123
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, load_addr)
        cpu.regs[11] = 456
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_1["cache_hit_kind"] == "none"
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "template"
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_misses"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_entries"] == 1

    def test_superblock_template_generalization_merges_same_path_branch_states(self, cpu, monkeypatch):
        class FakeValueModel:
            def __call__(self, feature_vec):
                return torch.tensor([0.8], device=feature_vec.device)

            def encode_features(self, feature_vec):
                return torch.tensor([1.0, 0.0], device=feature_vec.device)

        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_MERGE_SIM", "0.9")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._neural_hotloop_value_model = FakeValueModel()

        load_addr = 0x10000
        insts = [
            cbz(0, 3),
            movz(2, 42),
            0x00000000,
            movz(2, 99),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[0] = 1
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, load_addr)
        cpu.regs[0] = 2
        candidate_2 = cpu._collect_superblock_candidate()
        reset_pc(cpu, load_addr)
        cpu.regs[0] = 3
        candidate_3 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_3 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is False
        assert candidate_3["cache_hit"] is True
        assert candidate_3["cache_hit_kind"] == "template"
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"] == candidate_3["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 3
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_generalizations"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_entries"] == 1

    def test_superblock_template_cross_window_hit_retargets_stop_pc(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}

        primary_addr = 0x10000
        secondary_addr = 0x11000
        insts = [
            movz(0, 1),
            movz(1, 7),
            b_uncond(2),
            movz(9, 99),
            movz(2, 42),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)
        load_program(cpu, code, primary_addr)
        load_program(cpu, code, secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        cpu.regs[11] = 123
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        cpu.regs[11] = 456
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "template-cross-window"
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"]
        relative_stop = int(candidate_1["expected_stop_pc"]) - primary_addr
        assert int(candidate_2["pc"]) == secondary_addr
        assert int(candidate_2["expected_stop_pc"]) == secondary_addr + relative_stop
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_cross_window_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_entries"] == 1

    def test_superblock_shape_cache_reuses_when_only_off_path_words_differ(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x10000
        secondary_addr = 0x11000
        primary_insts = [
            movz(0, 1),
            movz(2, 42),
            b_uncond(2),
            movz(2, 99),
            movz(3, 7),
            0x00000000,
        ]
        secondary_insts = [
            movz(0, 1),
            movz(2, 42),
            b_uncond(2),
            movz(2, 1234),
            movz(3, 7),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "shape-cross-window"
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"]
        relative_stop = int(candidate_1["expected_stop_pc"]) - primary_addr
        assert int(candidate_2["pc"]) == secondary_addr
        assert int(candidate_2["expected_stop_pc"]) == secondary_addr + relative_stop
        assert cpu._last_gpu_only_hotloop_stats["superblock_cache_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_misses"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_cross_window_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_entries"] == 1

    def test_superblock_shape_cache_patches_safe_literals_on_path(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x12000
        secondary_addr = 0x13000
        primary_insts = [
            movz(0, 7),
            movz(2, 1),
            add_reg(3, 0, 2),
            0x00000000,
        ]
        secondary_insts = [
            movz(0, 9),
            movz(2, 1),
            add_reg(3, 0, 2),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "shape-literal-cross-window"
        assert candidate_2["literal_patch_count"] == 1
        assert candidate_1["code_bytes"] != candidate_2["code_bytes"]
        assert candidate_2["words"][0] == secondary_insts[0]
        relative_stop = int(candidate_1["expected_stop_pc"]) - primary_addr
        assert int(candidate_2["pc"]) == secondary_addr
        assert int(candidate_2["expected_stop_pc"]) == secondary_addr + relative_stop
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_cross_window_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_words"] == 1

    def test_superblock_shape_cache_refuses_literal_patches_that_change_branch_path(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x14000
        secondary_addr = 0x15000
        primary_insts = [
            movz(0, 1),
            cbz(0, 3),
            movz(2, 42),
            b_uncond(2),
            movz(2, 99),
            0x00000000,
        ]
        secondary_insts = [
            movz(0, 0),
            cbz(0, 3),
            movz(2, 42),
            b_uncond(2),
            movz(2, 99),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is False
        assert candidate_1["code_bytes"] != candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_words"] == 0

    def test_superblock_shape_cache_patches_safe_add_imm_on_path(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x16000
        secondary_addr = 0x17000
        primary_insts = [
            movz(0, 40),
            add_imm(0, 0, 2),
            movz(1, 1),
            add_reg(3, 0, 1),
            0x00000000,
        ]
        secondary_insts = [
            movz(0, 40),
            add_imm(0, 0, 4),
            movz(1, 1),
            add_reg(3, 0, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "shape-imm-cross-window"
        assert candidate_2["specialization_patch_count"] == 1
        assert candidate_2["literal_patch_count"] == 0
        assert candidate_2["imm_patch_count"] == 1
        assert candidate_1["code_bytes"] != candidate_2["code_bytes"]
        assert candidate_2["words"][1] == secondary_insts[1]
        relative_stop = int(candidate_1["expected_stop_pc"]) - primary_addr
        assert int(candidate_2["pc"]) == secondary_addr
        assert int(candidate_2["expected_stop_pc"]) == secondary_addr + relative_stop
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_cross_window_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_imm_patch_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_imm_patch_words"] == 1

    def test_superblock_shape_cache_refuses_add_imm_patch_that_changes_load_address(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x18000
        secondary_addr = 0x19000
        primary_insts = [
            movz(1, 0x2000),
            add_imm(1, 1, 1),
            ldrb(2, 1),
            movz(3, 1),
            0x00000000,
        ]
        secondary_insts = [
            movz(1, 0x2000),
            add_imm(1, 1, 2),
            ldrb(2, 1),
            movz(3, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is False
        assert candidate_1["code_bytes"] != candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_misses"] == 2
        assert cpu._last_gpu_only_hotloop_stats["superblock_imm_patch_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_imm_patch_words"] == 0

    def test_superblock_template_cache_reuses_store_superblock_across_payload_register_changes(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        load_addr = 0x1A000
        insts = [
            strb(0, 1),
            movz(2, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[0] = 7
        cpu.regs[1] = 0x2000
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, load_addr)
        cpu.regs[0] = 9
        cpu.regs[1] = 0x2000
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "template"
        assert candidate_1["code_bytes"] == candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_misses"] == 1

    def test_superblock_template_cache_refuses_store_superblock_across_address_changes(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        load_addr = 0x1B000
        insts = [
            strb(0, 1),
            movz(2, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[0] = 7
        cpu.regs[1] = 0x2000
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, load_addr)
        cpu.regs[0] = 7
        cpu.regs[1] = 0x2001
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is False
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_hits"] == 0
        assert cpu._last_gpu_only_hotloop_stats["superblock_template_misses"] == 2

    def test_superblock_shape_cache_patches_store_payload_literals(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_SHAPE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        primary_addr = 0x1C000
        secondary_addr = 0x1D000
        primary_insts = [
            movz(0, 7),
            movz(1, 0x2000),
            strb(0, 1),
            movz(2, 1),
            0x00000000,
        ]
        secondary_insts = [
            movz(0, 9),
            movz(1, 0x2000),
            strb(0, 1),
            movz(2, 1),
            0x00000000,
        ]
        load_program(cpu, struct.pack(f"<{len(primary_insts)}I", *primary_insts), primary_addr)
        load_program(cpu, struct.pack(f"<{len(secondary_insts)}I", *secondary_insts), secondary_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, primary_addr)
        candidate_1 = cpu._collect_superblock_candidate()
        reset_pc(cpu, secondary_addr)
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True
        assert candidate_2["cache_hit_kind"] == "shape-literal-cross-window"
        assert candidate_2["literal_patch_count"] == 1
        assert candidate_2["words"][0] == secondary_insts[0]
        assert candidate_1["code_bytes"] != candidate_2["code_bytes"]
        assert cpu._last_gpu_only_hotloop_stats["superblock_shape_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_hits"] == 1
        assert cpu._last_gpu_only_hotloop_stats["superblock_literal_patch_words"] == 1

    def test_simulator_pre_windows_excludes_store_only_ranges(self, cpu):
        store_addr = 0x2200
        cpu.regs[0] = 0x42
        cpu.regs[1] = store_addr
        insts = [strb(0, 1), 0x00000000]
        simulation = cpu._simulate_superblock_path(insts, max_steps=8, base_pc=0)

        assert simulation is not None
        assert list(simulation["pre_windows"]) == []
        assert list(simulation["post_windows"]) == [(store_addr, store_addr + 1)]

    def test_simulator_pre_windows_shadows_read_after_write(self, cpu):
        rw_addr = 0x2300
        cpu.regs[0] = 0x77
        cpu.regs[1] = rw_addr
        # Store X0 at [X1], then load the same byte back into X2.
        insts = [strb(0, 1), ldrb(2, 1), 0x00000000]
        simulation = cpu._simulate_superblock_path(insts, max_steps=8, base_pc=0)

        assert simulation is not None
        # Store contributes only to post; load is fully shadowed so pre stays empty.
        assert list(simulation["pre_windows"]) == []
        assert list(simulation["post_windows"]) == [(rw_addr, rw_addr + 1)]

    def test_simulator_pre_windows_keeps_pure_load_ranges(self, cpu):
        load_addr_bytes = 0x2400
        cpu.regs[1] = load_addr_bytes
        insts = [ldrb(0, 1), 0x00000000]
        simulation = cpu._simulate_superblock_path(insts, max_steps=8, base_pc=0)

        assert simulation is not None
        assert list(simulation["pre_windows"]) == [(load_addr_bytes, load_addr_bytes + 1)]
        assert list(simulation["post_windows"]) == []

    def test_superblock_trace_cache_ignores_store_only_byte_changes(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        load_addr = 0x1E000
        store_addr = 0x2100
        insts = [strb(0, 1), movz(2, 1), 0x00000000]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[0] = 7
        cpu.regs[1] = store_addr
        cpu.memory[store_addr] = 0xAA
        candidate_1 = cpu._collect_superblock_candidate()

        # Benign mutation at the store destination — block overwrites it anyway.
        cpu.memory[store_addr] = 0x55
        reset_pc(cpu, load_addr)
        cpu.regs[0] = 7
        cpu.regs[1] = store_addr
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 1
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is True

    def test_superblock_trace_cache_invalidates_on_load_byte_change(self, cpu, monkeypatch):
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "8")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "4")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}

        load_addr = 0x1F000
        read_addr = 0x2500
        insts = [ldrb(0, 1), movz(2, 1), 0x00000000]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        calls = []
        original = cpu._simulate_superblock_path

        def counting_simulator(words, max_steps=256, *, base_pc=0):
            calls.append((tuple(words), int(max_steps), int(base_pc)))
            return original(words, max_steps=max_steps, base_pc=base_pc)

        monkeypatch.setattr(cpu, "_simulate_superblock_path", counting_simulator)

        reset_pc(cpu, load_addr)
        cpu.regs[1] = read_addr
        cpu.memory[read_addr] = 0xAA
        candidate_1 = cpu._collect_superblock_candidate()

        # Mutation at the load source must invalidate the cached memory snapshot.
        cpu.memory[read_addr] = 0x55
        reset_pc(cpu, load_addr)
        cpu.regs[1] = read_addr
        candidate_2 = cpu._collect_superblock_candidate()

        assert len(calls) == 2
        assert candidate_1 is not None
        assert candidate_2 is not None
        assert candidate_1["cache_hit"] is False
        assert candidate_2["cache_hit"] is False

    def test_adaptive_trace_promotion_skips_trace_lookup_after_misses(self, cpu, monkeypatch):
        """After N consecutive trace-level misses for a program_key, subsequent
        lookups should skip the trace-level snapshot comparison and go straight
        to template/shape cache. Pins the adaptive-promotion behavior."""
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "2")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "2")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION", "2")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}
        cpu._superblock_trace_miss_counter = {}

        load_addr = 0x30000
        insts = [movz(0, 5), movz(1, 7), 0x00000000]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        # First pass populates trace cache but different regs each entry.
        # We force cache_key to differ by varying a register NOT read by the
        # block, so trace-level always misses while template-level hits.
        def _collect_with_reg(extra_reg_value: int):
            reset_pc(cpu, load_addr)
            cpu.regs[5] = extra_reg_value  # r5 never read by the block
            return cpu._collect_superblock_candidate()

        # Miss 1: populate trace + template
        c1 = _collect_with_reg(1)
        assert c1 is not None and c1["cache_hit"] is False
        # Miss 2: different trace key, template should hit (r5 not guarded)
        c2 = _collect_with_reg(2)
        assert c2 is not None and c2["cache_hit_kind"] == "template"
        # Miss 3: counter hits threshold — trace lookup should now be skipped
        c3 = _collect_with_reg(3)
        assert c3 is not None and c3["cache_hit_kind"] == "template"
        # Miss 4: trace lookup skipped (promoted)
        c4 = _collect_with_reg(4)
        assert c4 is not None and c4["cache_hit_kind"] == "template"

        stats = cpu._last_gpu_only_hotloop_stats
        assert stats.get("superblock_trace_promotions", 0) >= 1, (
            f"expected at least one trace promotion, got stats={stats}"
        )
        assert stats.get("superblock_trace_skips", 0) >= 1, (
            f"expected at least one trace skip after promotion, got stats={stats}"
        )

    def test_adaptive_trace_promotion_disabled_at_zero_threshold(self, cpu, monkeypatch):
        """Setting NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION=0 disables the
        optimization so trace-level lookups always run. Pins the off-by-default
        escape hatch for workloads where trace-miss comparison is free."""
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_CACHE_SIZE", "2")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TEMPLATE_PER_KEY", "2")
        monkeypatch.setenv("NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION", "0")
        cpu._last_gpu_only_hotloop_stats = {}
        cpu._superblock_trace_cache = {}
        cpu._superblock_template_cache = {}
        cpu._superblock_shape_cache = {}
        cpu._superblock_trace_miss_counter = {}

        load_addr = 0x31000
        insts = [movz(0, 5), movz(1, 7), 0x00000000]
        load_program(cpu, struct.pack(f"<{len(insts)}I", *insts), load_addr)

        for reg_value in range(5):
            reset_pc(cpu, load_addr)
            cpu.regs[5] = reg_value
            cpu._collect_superblock_candidate()

        stats = cpu._last_gpu_only_hotloop_stats
        assert stats.get("superblock_trace_skips", 0) == 0
        assert stats.get("superblock_trace_promotions", 0) == 0

    def test_rust_hotloop_failure_falls_back_to_torch_backend(self, cpu, monkeypatch):
        import kernels.mlx.rust_runner as rust_runner

        def _boom(**_kwargs):
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(rust_runner, "get_shared_cpu", _boom)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert cpu._last_gpu_only_backend == "torch-gpu-only"

    def test_auto_hotloop_backend_stays_on_torch_for_cpu_device(self, cpu, monkeypatch):
        import kernels.mlx.rust_runner as rust_runner

        def _boom(**_kwargs):
            raise AssertionError("CPU auto mode should not invoke the Rust backend")

        monkeypatch.setattr(rust_runner, "get_shared_cpu", _boom)
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "auto")

        n_iters = 5
        load_addr = 0x10000
        code = build_program(n_iters)
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(n_iters)
        assert int(cpu.regs[0].item()) == n_iters
        assert cpu._last_gpu_only_backend == "torch-gpu-only"

    def test_forced_rust_hotloop_handoff_syncs_memory_windows(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 27
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x1002C
                self._memory = bytearray(0x4000)

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                assert bytes(self._memory[0x2000:0x2004]) == b"ABCD"
                self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                self._regs[1] = 0x2004
                self._regs[2] = 0
                self._regs[3] = 0x3004
                self._pc = 0x1002C
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            movz(3, 0x3000),
            movz(2, 4),
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            add_imm(1, 1, 1),
            add_imm(3, 3, 1),
            subs_imm(2, 2, 1),
            bcond((-5) & 0x7FFFF, 1),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"ABCD"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"ABCD"
        assert int(cpu.regs[1].item()) == 0x2004
        assert int(cpu.regs[2].item()) == 0
        assert int(cpu.regs[3].item()) == 0x3004
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_cbnz_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 27
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10028
                self._memory = bytearray(0x4000)

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                assert bytes(self._memory[0x2000:0x2004]) == b"WXYZ"
                self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                self._regs[1] = 0x2004
                self._regs[2] = 0
                self._regs[3] = 0x3004
                self._pc = 0x10028
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            movz(3, 0x3000),
            movz(2, 4),
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            add_imm(1, 1, 1),
            add_imm(3, 3, 1),
            subs_imm(2, 2, 1),
            cbnz(2, (-5) & 0x7FFFF),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"WXYZ"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"WXYZ"
        assert int(cpu.regs[1].item()) == 0x2004
        assert int(cpu.regs[2].item()) == 0
        assert int(cpu.regs[3].item()) == 0x3004
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_cbz_exit_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = expected_engine_executed_count(4, "bytecopy-cbz-exit") + 1
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10028
                self._memory = bytearray(0x4000)

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                assert bytes(self._memory[0x2000:0x2004]) == b"QRST"
                self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                self._regs[1] = 0x2004
                self._regs[2] = 0
                self._regs[3] = 0x3004
                self._pc = 0x10028
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            movz(3, 0x3000),
            movz(2, 4),
            cbz(2, 7),
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            add_imm(1, 1, 1),
            add_imm(3, 3, 1),
            subs_imm(2, 2, 1),
            b_uncond((-6) & 0x3FFFFFF),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"QRST"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy-cbz-exit")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"QRST"
        assert int(cpu.regs[1].item()) == 0x2004
        assert int(cpu.regs[2].item()) == 0
        assert int(cpu.regs[3].item()) == 0x3004
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_bge_exit_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = expected_engine_executed_count(4, "bytecopy-bge-exit") + 1
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10030
                self._memory = bytearray(0x4000)

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                assert bytes(self._memory[0x2000:0x2004]) == b"UVWX"
                self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                self._regs[1] = 0x2004
                self._regs[2] = 4
                self._regs[3] = 0x3004
                self._regs[5] = 4
                self._pc = 0x10030
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            movz(3, 0x3000),
            movz(2, 0),
            movz(5, 4),
            cmp_reg(2, 5),
            bcond(7, 10),  # B.GE halt
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            add_imm(1, 1, 1),
            add_imm(3, 3, 1),
            add_imm(2, 2, 1),
            b_uncond((-7) & 0x3FFFFFF),
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"UVWX"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy-bge-exit")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"UVWX"
        assert int(cpu.regs[1].item()) == 0x2004
        assert int(cpu.regs[2].item()) == 4
        assert int(cpu.regs[3].item()) == 0x3004
        assert int(cpu.regs[5].item()) == 4
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_blt_memory_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 32
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, False, False, False)
                self._pc = 0x10030
                self._memory = bytearray(0x4000)

            def load_program(self, binary, address=0x10000):
                self._memory[address:address + len(binary)] = bytes(binary)

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def write_memory(self, address, data):
                payload = bytes(data)
                self._memory[address:address + len(payload)] = payload

            def read_memory(self, address, size):
                return bytes(self._memory[address:address + size])

            def execute(self, max_cycles=100_000):
                assert bytes(self._memory[0x2000:0x2004]) == b"LMNO"
                self._memory[0x3000:0x3004] = self._memory[0x2000:0x2004]
                self._regs[1] = 0x2004
                self._regs[2] = 4
                self._regs[3] = 0x3004
                self._regs[5] = 4
                self._pc = 0x10030
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(1, 0x2000),
            movz(3, 0x3000),
            movz(2, 0),
            movz(5, 4),
            ldrb(4, 1, 0),
            strb(4, 3, 0),
            add_imm(1, 1, 1),
            add_imm(3, 3, 1),
            add_imm(2, 2, 1),
            cmp_reg(2, 5),
            bcond((-6) & 0x7FFFF, 11),  # B.LT loop
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)
        cpu.memory[0x2000:0x2004] = torch.tensor(list(b"LMNO"), dtype=torch.uint8)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == expected_engine_executed_count(4, "bytecopy-blt")
        assert bytes(cpu.memory[0x3000:0x3004].detach().cpu().tolist()) == b"LMNO"
        assert int(cpu.regs[1].item()) == 0x2004
        assert int(cpu.regs[2].item()) == 4
        assert int(cpu.regs[3].item()) == 0x3004
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_bgt_compare_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = 20
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10020

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self._regs[0] = 4
                self._regs[1] = 0
                self._regs[5] = 0
                self._pc = 0x10020
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        insts = [
            movz(0, 0),
            movz(1, 4),
            movz(2, 1),
            movz(5, 0),
            add_reg(0, 0, 2),
            subs_imm(1, 1, 1),
            cmp_reg(1, 5),
            bcond((-4) & 0x7FFFF, 12),  # B.GT loop
            0x00000000,
        ]
        code = struct.pack(f"<{len(insts)}I", *insts)

        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=10_000, batch_size=64)

        assert executed == 19
        assert int(cpu.regs[0].item()) == 4
        assert int(cpu.regs[1].item()) == 0
        assert cpu._last_gpu_only_backend == "rust-hotloop"

    def test_forced_rust_hotloop_handoff_supports_nested_counted_loops(self, cpu, monkeypatch):
        class FakeResult:
            cycles = expected_engine_executed_count(0, "nested-counted") + 1
            elapsed_seconds = 0.01
            stop_reason_name = "HALT"

        class FakeRustCPU:
            def __init__(self):
                self._regs = [0] * 32
                self._flags = (False, True, True, False)
                self._pc = 0x10024

            def load_program(self, binary, address=0x10000):
                self._binary = bytes(binary)
                self._address = address

            def set_pc(self, value):
                self._pc = value

            @property
            def pc(self):
                return self._pc

            def set_register(self, reg, value):
                self._regs[reg] = value

            def get_register(self, reg):
                return self._regs[reg]

            def set_flags(self, n, z, c, v):
                self._flags = (n, z, c, v)

            def get_flags(self):
                return self._flags

            def execute(self, max_cycles=100_000):
                self._regs[0] = 0
                self._regs[1] = 0
                self._regs[2] = 128 * 128
                self._pc = 0x10024
                return FakeResult()

        import kernels.mlx.rust_runner as rust_runner

        monkeypatch.setattr(rust_runner, "get_shared_cpu", lambda **_: FakeRustCPU())
        monkeypatch.setenv("NCPU_GPU_ONLY_HOTLOOP_BACKEND", "rust")

        load_addr = 0x10000
        code = build_nested_counted_program()
        load_program(cpu, code, load_addr)
        reset_pc(cpu, load_addr)

        executed, _ = cpu.run_gpu_only(max_instructions=200_000, batch_size=64)

        assert executed == expected_engine_executed_count(0, "nested-counted")
        assert int(cpu.regs[0].item()) == 0
        assert int(cpu.regs[1].item()) == 0
        assert int(cpu.regs[2].item()) == 128 * 128
        assert cpu._last_gpu_only_backend == "rust-hotloop"
