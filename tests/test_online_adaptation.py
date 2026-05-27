#!/usr/bin/env python3
"""Tests for online model adaptation (cache + scheduler).

Tests cover:
    - OnlineCacheAdapter: replay buffer, gradient updates, hit rate tracking
    - OnlineSchedulerAdapter: decision logging, few-shot fine-tuning, reward tracking
    - AdaptationManager: coordination, metrics, checkpoint save/load, enable/disable
    - attach_to_neuros: transparent monkey-patching of NeurOS
    - ReplayBuffer: capacity, sampling, clearing
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.neural.online_adaptation import (
    AdaptationConfig,
    AdaptationManager,
    OnlineCacheAdapter,
    OnlineSchedulerAdapter,
    ReplayBuffer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def neural_cache(device):
    """Create a NeuralCache with a trained (random-initialized) replacer."""
    from ncpu.os.neuros.cache import NeuralCache
    cache = NeuralCache(num_sets=16, ways=4, device=device)
    # Mark as trained so the adapter will generate training samples
    cache._replacer_trained = True
    cache.replacer.eval()
    return cache


@pytest.fixture
def neural_scheduler(device):
    """Create a NeuralScheduler with a trained (random-initialized) net."""
    from ncpu.os.neuros.process import ProcessTable
    from ncpu.os.neuros.scheduler import NeuralScheduler
    pt = ProcessTable(max_processes=64, device=device)
    sched = NeuralScheduler(process_table=pt, device=device)
    # Mark as trained so the adapter will generate training samples
    sched._trained = True
    sched.net.eval()
    return sched


@pytest.fixture
def config():
    """Fast adaptation config for testing (small intervals)."""
    return AdaptationConfig(
        cache_replay_capacity=100,
        cache_adapt_interval=10,
        cache_adapt_steps=2,
        cache_lr=1e-3,
        cache_min_samples=5,
        sched_replay_capacity=50,
        sched_adapt_interval=5,
        sched_adapt_steps=2,
        sched_lr=1e-3,
        sched_min_samples=3,
        log_interval=20,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBuffer:
    def test_push_and_len(self):
        buf = ReplayBuffer(capacity=5)
        assert len(buf) == 0
        buf.push({"x": 1})
        buf.push({"x": 2})
        assert len(buf) == 2

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.push({"x": i})
        assert len(buf) == 3
        # Oldest items should be discarded
        items = buf.all()
        assert items[0]["x"] == 7
        assert items[-1]["x"] == 9

    def test_sample_batch(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.push({"x": i})
        batch = buf.sample_batch(3)
        assert len(batch) == 3
        assert all("x" in s for s in batch)

    def test_sample_empty(self):
        buf = ReplayBuffer(capacity=5)
        assert buf.sample_batch(3) == []

    def test_sample_undersized(self):
        buf = ReplayBuffer(capacity=10)
        buf.push({"x": 1})
        buf.push({"x": 2})
        batch = buf.sample_batch(5)
        # Should return min(5, 2) = 2
        assert len(batch) == 2

    def test_clear(self):
        buf = ReplayBuffer(capacity=5)
        buf.push({"x": 1})
        buf.clear()
        assert len(buf) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# OnlineCacheAdapter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnlineCacheAdapter:
    def test_init(self, neural_cache, config):
        adapter = OnlineCacheAdapter(neural_cache, config)
        assert adapter.total_accesses == 0
        assert adapter.total_adaptations == 0
        assert len(adapter.replay) == 0

    def test_hit_not_recorded(self, neural_cache, config):
        """Cache hits should not generate training samples (no eviction)."""
        adapter = OnlineCacheAdapter(neural_cache, config)
        adapter.on_access(addr=0x1000, hit=True, write=False, victim_way=None)
        assert adapter.total_accesses == 1
        assert len(adapter.replay) == 0

    def test_miss_with_victim_recorded(self, neural_cache, config):
        """Cache misses with eviction should generate training samples."""
        adapter = OnlineCacheAdapter(neural_cache, config)
        adapter.on_access(addr=0x1000, hit=False, write=False,
                         set_idx=0, victim_way=2)
        assert len(adapter.replay) == 1
        sample = adapter.replay.all()[0]
        assert sample["victim"] == 2
        assert "history" in sample
        assert "line_features" in sample

    def test_miss_without_victim_not_recorded(self, neural_cache, config):
        """Cache misses without eviction (empty way available) should not
        generate training samples."""
        adapter = OnlineCacheAdapter(neural_cache, config)
        adapter.on_access(addr=0x1000, hit=False, write=False,
                         set_idx=0, victim_way=None)
        assert len(adapter.replay) == 0

    def test_adaptation_triggers(self, neural_cache, config):
        """Adapter should run gradient updates after enough accesses."""
        adapter = OnlineCacheAdapter(neural_cache, config)

        # Fill replay buffer above min_samples
        for i in range(config.cache_min_samples + 1):
            adapter.on_access(
                addr=i * 64, hit=False, write=False,
                set_idx=i % neural_cache.num_sets, victim_way=i % neural_cache.ways,
            )

        # Now trigger enough accesses to hit the adapt_interval
        remaining = config.cache_adapt_interval - adapter.accesses_since_adapt
        for i in range(remaining + 1):
            adapter.on_access(addr=0x2000 + i * 64, hit=True, write=False)

        assert adapter.total_adaptations >= 1

    def test_hit_rate_tracking(self, neural_cache, config):
        adapter = OnlineCacheAdapter(neural_cache, config)

        # 10 hits, 10 misses
        for _ in range(10):
            adapter.on_access(addr=0x1000, hit=True, write=False)
        for i in range(10):
            adapter.on_access(addr=i * 64, hit=False, write=False,
                             set_idx=0, victim_way=i % 4)

        m = adapter.metrics()
        assert m["total_accesses"] == 20
        assert 0.4 < m["current_hit_rate"] < 0.6  # ~50%

    def test_metrics_keys(self, neural_cache, config):
        adapter = OnlineCacheAdapter(neural_cache, config)
        m = adapter.metrics()
        required_keys = [
            "total_accesses", "total_adaptations", "replay_buffer_size",
            "baseline_hit_rate", "current_hit_rate",
            "pre_adapt_hit_rate", "post_adapt_hit_rate",
            "hit_rate_improvement", "global_hit_rate",
        ]
        for k in required_keys:
            assert k in m, f"Missing key: {k}"

    def test_reset(self, neural_cache, config):
        adapter = OnlineCacheAdapter(neural_cache, config)
        adapter.on_access(addr=0x1000, hit=False, write=False,
                         set_idx=0, victim_way=1)
        adapter.reset()
        assert adapter.total_accesses == 0
        assert len(adapter.replay) == 0

    def test_untrained_replacer_skips_recording(self, device, config):
        """If the replacer isn't trained, no samples are recorded."""
        from ncpu.os.neuros.cache import NeuralCache
        cache = NeuralCache(num_sets=16, ways=4, device=device)
        assert not cache._replacer_trained

        adapter = OnlineCacheAdapter(cache, config)
        adapter.on_access(addr=0x1000, hit=False, write=False,
                         set_idx=0, victim_way=2)
        assert len(adapter.replay) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# OnlineSchedulerAdapter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOnlineSchedulerAdapter:
    def test_init(self, neural_scheduler, config):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        assert adapter.total_decisions == 0
        assert adapter.total_adaptations == 0

    def test_decision_recorded(self, neural_scheduler, config, device):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        features = torch.randn(3, 8, device=device)
        adapter.on_decision(
            selected_idx=1, process_features=features,
            num_ready=3, throughput=0.8, fairness=0.9,
        )
        assert adapter.total_decisions == 1
        assert len(adapter.replay) == 1

    def test_single_ready_not_recorded(self, neural_scheduler, config, device):
        """With only 1 ready process, there's no meaningful decision to learn."""
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        features = torch.randn(1, 8, device=device)
        adapter.on_decision(
            selected_idx=0, process_features=features,
            num_ready=1, throughput=0.8, fairness=0.9,
        )
        assert adapter.total_decisions == 1
        assert len(adapter.replay) == 0  # skipped because num_ready <= 1

    def test_adaptation_triggers(self, neural_scheduler, config, device):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)

        # Push enough decisions to trigger adaptation
        for i in range(config.sched_adapt_interval + config.sched_min_samples):
            features = torch.randn(4, 8, device=device)
            adapter.on_decision(
                selected_idx=i % 4, process_features=features,
                num_ready=4, throughput=0.7, fairness=0.85,
            )

        assert adapter.total_adaptations >= 1

    def test_reward_tracking(self, neural_scheduler, config, device):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        features = torch.randn(3, 8, device=device)
        adapter.on_decision(
            selected_idx=0, process_features=features,
            num_ready=3, throughput=0.9, fairness=0.95,
        )
        m = adapter.metrics()
        assert m["avg_reward"] == pytest.approx(0.9 * 0.95, abs=1e-4)
        assert m["avg_throughput"] == pytest.approx(0.9, abs=1e-4)

    def test_metrics_keys(self, neural_scheduler, config):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        m = adapter.metrics()
        required_keys = [
            "total_decisions", "total_adaptations", "replay_buffer_size",
            "baseline_fairness", "current_fairness", "avg_reward",
            "avg_throughput", "avg_turnaround",
            "pre_adapt_reward", "post_adapt_reward", "reward_improvement",
        ]
        for k in required_keys:
            assert k in m, f"Missing key: {k}"

    def test_reset(self, neural_scheduler, config, device):
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)
        features = torch.randn(3, 8, device=device)
        adapter.on_decision(
            selected_idx=0, process_features=features,
            num_ready=3, throughput=0.8, fairness=0.9,
        )
        adapter.reset()
        assert adapter.total_decisions == 0
        assert len(adapter.replay) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# AdaptationManager Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptationManager:
    def test_init_both(self, neural_cache, neural_scheduler, config):
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)
        assert mgr.cache_adapter is not None
        assert mgr.sched_adapter is not None
        assert mgr.enabled

    def test_init_cache_only(self, neural_cache, config):
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)
        assert mgr.cache_adapter is not None
        assert mgr.sched_adapter is None

    def test_init_scheduler_only(self, neural_scheduler, config):
        mgr = AdaptationManager(neural_scheduler=neural_scheduler, config=config)
        assert mgr.cache_adapter is None
        assert mgr.sched_adapter is not None

    def test_init_neither(self, config):
        mgr = AdaptationManager(config=config)
        assert mgr.cache_adapter is None
        assert mgr.sched_adapter is None

    def test_enable_disable(self, neural_cache, config):
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)
        mgr.disable()
        assert not mgr.enabled
        mgr.enable()
        assert mgr.enabled

    def test_disabled_no_adaptation(self, neural_cache, config):
        """When disabled, events are counted but no gradient updates happen."""
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)
        mgr.disable()

        # Push many events
        for i in range(50):
            mgr.on_cache_access(
                addr=i * 64, hit=False, write=False,
                set_idx=i % 16, victim_way=i % 4,
            )

        # No adaptations should have occurred
        assert mgr.cache_adapter.total_adaptations == 0

    def test_cache_events_flow(self, neural_cache, config):
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)

        mgr.on_cache_access(addr=0x1000, hit=True, write=False)
        mgr.on_cache_access(addr=0x2000, hit=False, write=True,
                           set_idx=3, victim_way=1)

        assert mgr._total_events == 2
        assert mgr.cache_adapter.total_accesses == 2

    def test_scheduler_events_flow(self, neural_scheduler, config, device):
        mgr = AdaptationManager(neural_scheduler=neural_scheduler, config=config)

        features = torch.randn(4, 8, device=device)
        mgr.on_schedule_decision(
            selected_idx=2, process_features=features,
            num_ready=4, throughput=0.85, fairness=0.92,
        )

        assert mgr._total_events == 1
        assert mgr.sched_adapter.total_decisions == 1

    def test_metrics_combined(self, neural_cache, neural_scheduler, config):
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)
        m = mgr.metrics()
        assert "enabled" in m
        assert "total_events" in m
        assert "cache" in m
        assert "scheduler" in m

    def test_summary_string(self, neural_cache, neural_scheduler, config):
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)
        s = mgr.summary()
        assert "AdaptationManager" in s
        assert "Cache" in s
        assert "Scheduler" in s

    def test_repr(self, neural_cache, neural_scheduler, config):
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)
        r = repr(mgr)
        assert "ON" in r
        assert "cache" in r
        assert "sched" in r

    def test_save_load_checkpoint(self, neural_cache, neural_scheduler, config):
        """Save checkpoint, modify weights, load checkpoint, verify restoration."""
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            saved = mgr.save_checkpoint(tmpdir)
            assert "cache_replace" in saved
            assert "scheduler" in saved
            assert "metrics" in saved

            # Verify files exist
            assert Path(saved["cache_replace"]).exists()
            assert Path(saved["scheduler"]).exists()
            assert Path(saved["metrics"]).exists()

            # Corrupt weights to verify load actually restores
            with torch.no_grad():
                for p in neural_cache.replacer.parameters():
                    p.fill_(999.0)

            # Load
            result = mgr.load_checkpoint(tmpdir)
            assert result.get("cache_replace") is True
            assert result.get("scheduler") is True

            # Verify weights were restored (not all 999.0)
            param = next(neural_cache.replacer.parameters())
            assert not torch.all(param == 999.0)

    def test_save_empty_dir_created(self, neural_cache, config):
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub", "deep")
            saved = mgr.save_checkpoint(subdir)
            assert Path(subdir).exists()

    def test_load_nonexistent_dir(self, neural_cache, config):
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)
        result = mgr.load_checkpoint("/nonexistent/path")
        assert result == {}

    def test_reset(self, neural_cache, neural_scheduler, config):
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)
        mgr.on_cache_access(addr=0x1000, hit=True, write=False)
        mgr.reset()
        assert mgr._total_events == 0
        assert mgr.cache_adapter.total_accesses == 0
        assert mgr.sched_adapter.total_decisions == 0

    def test_logging_at_interval(self, neural_cache, config):
        """Metrics snapshots should be logged at the configured interval."""
        config.log_interval = 5
        mgr = AdaptationManager(neural_cache=neural_cache, config=config)

        for i in range(10):
            mgr.on_cache_access(addr=i * 64, hit=True, write=False)

        # Should have 2 log entries (at event 5 and event 10)
        assert len(mgr._log) == 2
        assert mgr._log[0]["event_count"] == 5
        assert mgr._log[1]["event_count"] == 10


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End: Adaptation Actually Improves
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_cache_weights_change(self, neural_cache, config):
        """After several adaptation rounds, model weights should have changed."""
        adapter = OnlineCacheAdapter(neural_cache, config)

        # Snapshot initial weights
        initial_weights = {
            name: p.clone()
            for name, p in neural_cache.replacer.named_parameters()
        }

        # Generate enough miss events to trigger multiple adaptations
        for i in range(config.cache_adapt_interval * 3):
            adapter.on_access(
                addr=i * 64, hit=False, write=False,
                set_idx=i % neural_cache.num_sets,
                victim_way=i % neural_cache.ways,
            )

        assert adapter.total_adaptations >= 2

        # Check that at least one parameter has changed
        changed = False
        for name, p in neural_cache.replacer.named_parameters():
            if not torch.allclose(p, initial_weights[name], atol=1e-6):
                changed = True
                break
        assert changed, "Model weights should have been updated by adaptation"

    def test_scheduler_weights_change(self, neural_scheduler, config, device):
        """After several adaptation rounds, scheduler weights should change."""
        adapter = OnlineSchedulerAdapter(neural_scheduler, config)

        initial_weights = {
            name: p.clone()
            for name, p in neural_scheduler.net.named_parameters()
        }

        for i in range(config.sched_adapt_interval * 3):
            features = torch.randn(4, 8, device=device)
            adapter.on_decision(
                selected_idx=i % 4, process_features=features,
                num_ready=4, throughput=0.7 + 0.1 * (i % 3),
                fairness=0.8 + 0.1 * (i % 2),
            )

        assert adapter.total_adaptations >= 2

        changed = False
        for name, p in neural_scheduler.net.named_parameters():
            if not torch.allclose(p, initial_weights[name], atol=1e-6):
                changed = True
                break
        assert changed, "Scheduler weights should have been updated"

    def test_full_manager_lifecycle(self, neural_cache, neural_scheduler, config, device):
        """Full lifecycle: init -> events -> adapt -> checkpoint -> load."""
        mgr = AdaptationManager(neural_cache, neural_scheduler, config)

        # Feed events
        for i in range(30):
            mgr.on_cache_access(
                addr=i * 64, hit=(i % 3 == 0), write=(i % 5 == 0),
                set_idx=i % neural_cache.num_sets,
                victim_way=i % neural_cache.ways if (i % 3 != 0) else None,
            )
            features = torch.randn(3, 8, device=device)
            mgr.on_schedule_decision(
                selected_idx=i % 3, process_features=features,
                num_ready=3, throughput=0.75, fairness=0.9,
            )

        # Should have done some adaptations
        m = mgr.metrics()
        assert m["total_events"] == 60

        # Save and restore
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = mgr.save_checkpoint(tmpdir)
            loaded = mgr.load_checkpoint(tmpdir)
            assert loaded.get("cache_replace") is True
            assert loaded.get("scheduler") is True

        # Summary should not crash
        print(mgr.summary())


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with NeurOS Boot
# ═══════════════════════════════════════════════════════════════════════════════

class TestNeurOSIntegration:
    def test_boot_with_adaptation(self, device):
        """Boot NeurOS with online_adaptation=True and verify manager is created."""
        from ncpu.os.neuros.boot import NeurOS
        os_instance = NeurOS(device=device)
        stages = os_instance.boot(load_models=False, quiet=True, online_adaptation=True)
        assert os_instance.adaptation_manager is not None
        assert "adaptation" in stages

    def test_boot_without_adaptation(self, device):
        """Default boot should not create adaptation manager."""
        from ncpu.os.neuros.boot import NeurOS
        os_instance = NeurOS(device=device)
        os_instance.boot(load_models=False, quiet=True)
        assert os_instance.adaptation_manager is None

    def test_attach_to_neuros_patches_access(self, device):
        """attach_to_neuros should monkey-patch cache.access."""
        from ncpu.os.neuros.boot import NeurOS
        from ncpu.neural.online_adaptation import attach_to_neuros

        os_instance = NeurOS(device=device)
        os_instance.boot(load_models=False, quiet=True)

        # Store original method
        original_access = os_instance.cache.access

        mgr = attach_to_neuros(os_instance)

        # Method should be different now (patched)
        assert os_instance.cache.access is not original_access

        # Access should work and feed events to the manager
        os_instance.cache.access(0x1000, write=False)
        assert mgr._total_events == 1

    def test_attach_to_neuros_patches_schedule(self, device):
        """attach_to_neuros should monkey-patch scheduler.schedule."""
        from ncpu.os.neuros.boot import NeurOS
        from ncpu.os.neuros.process import ProcessState
        from ncpu.neural.online_adaptation import attach_to_neuros

        os_instance = NeurOS(device=device)
        os_instance.boot(load_models=False, quiet=True)

        original_schedule = os_instance.scheduler.schedule

        mgr = attach_to_neuros(os_instance)

        assert os_instance.scheduler.schedule is not original_schedule

        # Create a ready process so scheduling has something to do
        proc = os_instance.process_table.create_process("test", priority=128)
        proc.state = ProcessState.READY

        result = os_instance.scheduler.schedule()
        # Scheduling should work (even if no training) and feed events
        # The total events increment depends on whether a decision was made


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
