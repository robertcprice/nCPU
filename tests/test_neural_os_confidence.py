import torch

from ncpu.os.gpu import neural_demo
from ncpu.os.neuros.cache import NeuralCache
from ncpu.os.neuros.interrupts import NeuralGIC
from ncpu.os.neuros.scheduler import SchedulerNet, PROCESS_FEATURE_DIM


class DummyVectorModel:
    def __init__(self, logits):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        return self.logits.clone()


class DummyBatchModel:
    def __init__(self, batch_logits):
        self.batch_logits = torch.as_tensor(batch_logits, dtype=torch.float32)

    def __call__(self, windows):
        batch = int(windows.shape[0])
        if self.batch_logits.dim() == 1:
            return self.batch_logits.unsqueeze(0).repeat(batch, 1)
        return self.batch_logits[:batch].clone()


class FakeProcess:
    def __init__(self, pid: int, total_cycles: int):
        self.pid = pid
        self.total_cycles = total_cycles
        self.state = neural_demo.ProcessState.READY


class FakeProcessManager:
    def __init__(self, processes, fallback_pid: int):
        self.processes = {p.pid: p for p in processes}
        self._fallback_pid = fallback_pid

    def schedule_next(self):
        return self._fallback_pid


def test_model_confidence_gate_rejects_low_margin_logits():
    gate = neural_demo.ModelConfidenceGate(min_confidence=0.60, min_margin=0.10)

    allow, summary = gate.should_use(torch.tensor([0.20, 0.19], dtype=torch.float32))

    assert not allow
    assert 0.0 < summary.confidence < 0.60
    assert summary.margin < 0.10


def test_scheduler_wrapper_falls_back_on_low_confidence():
    proc_mgr = FakeProcessManager(
        [FakeProcess(1, 1000), FakeProcess(2, 2000)],
        fallback_pid=1,
    )
    scheduler_state = SchedulerNet(feature_dim=PROCESS_FEATURE_DIM).state_dict()
    wrapper = neural_demo.NeuralSchedulerWrapper(
        proc_mgr,
        scheduler_state,
        torch.device("cpu"),
        confidence_config=neural_demo.NeuralConfidenceConfig(
            scheduler_min_confidence=0.90,
            scheduler_min_margin=0.20,
        ),
    )
    wrapper.net = DummyVectorModel([0.0, 0.0])

    chosen = proc_mgr.schedule_next()
    stats = wrapper.stats()

    assert chosen == 1
    assert stats["model_invocations"] == 1
    assert stats["neural_decisions"] == 0
    assert stats["confidence_fallbacks"] == 1


def test_scheduler_wrapper_uses_neural_decision_when_confident():
    proc_mgr = FakeProcessManager(
        [FakeProcess(1, 1000), FakeProcess(2, 2000)],
        fallback_pid=1,
    )
    scheduler_state = SchedulerNet(feature_dim=PROCESS_FEATURE_DIM).state_dict()
    wrapper = neural_demo.NeuralSchedulerWrapper(
        proc_mgr,
        scheduler_state,
        torch.device("cpu"),
        confidence_config=neural_demo.NeuralConfidenceConfig(
            scheduler_min_confidence=0.60,
            scheduler_min_margin=0.20,
        ),
    )
    wrapper.net = DummyVectorModel([-2.0, 3.5])

    chosen = proc_mgr.schedule_next()
    stats = wrapper.stats()

    assert chosen == 2
    assert stats["model_invocations"] == 1
    assert stats["neural_decisions"] == 1
    assert stats["confidence_fallbacks"] == 0


def test_gic_wrapper_falls_back_to_fixed_priority_on_low_confidence():
    gic = NeuralGIC(device=torch.device("cpu"))
    gic._trained = True
    gic.encoder = DummyVectorModel(torch.zeros(gic.num_irqs))
    wrapper = neural_demo.NeuralGICWrapper(
        gic,
        torch.device("cpu"),
        confidence_config=neural_demo.NeuralConfidenceConfig(
            gic_min_confidence=0.90,
            gic_min_margin=0.20,
        ),
    )

    wrapper.on_syscall(63)  # READ => raises both DISK and SYSCALL IRQs
    stats = wrapper.stats()

    assert stats["model_invocations"] == 1
    assert stats["neural_dispatches"] == 0
    assert stats["fallback_dispatches"] == 1
    assert stats["gic_handled"] == 2
    assert stats["gic_policy"] == "confidence-gated"


def test_compilation_advisor_filters_low_confidence_windows():
    advisor = neural_demo.NeuralCompilationAdvisor(
        DummyBatchModel([[0.10, 0.11, 0.12, 0.11, 0.10]] * 3),
        torch.device("cpu"),
        confidence_config=neural_demo.NeuralConfidenceConfig(
            compiler_min_confidence=0.90,
            compiler_min_margin=0.20,
        ),
    )

    advisor.on_compile("/tmp/example.c", binary_size=36)
    stats = advisor.stats()

    assert stats["compilations_analyzed"] == 1
    assert stats["model_invocations"] == 1
    assert stats["confidence_fallback_windows"] == 3
    assert stats["total_suggestions"] == 0


def test_cache_fs_falls_back_to_lru_on_low_confidence():
    cache = NeuralCache(num_sets=1, ways=4, device=torch.device("cpu"))
    cache._replacer_trained = True
    cache.replacer = DummyVectorModel([0.0, 0.0, 0.0, 0.0])
    cache_fs = neural_demo.NeuralCacheFS(
        object(),
        cache,
        confidence_config=neural_demo.NeuralConfidenceConfig(
            cache_min_confidence=0.90,
            cache_min_margin=0.20,
        ),
    )

    cache.valid[0] = torch.tensor([True, True, True, True], dtype=torch.bool)
    cache.last_access[0] = torch.tensor([10, 1, 5, 7], dtype=torch.int64)
    cache.access_count[0] = torch.tensor([2, 1, 4, 3], dtype=torch.int64)

    victim = cache._neural_victim(0)
    stats = cache_fs.stats()

    assert victim == 1
    assert stats["replacement_model_calls"] == 1
    assert stats["confidence_fallbacks"] == 1
