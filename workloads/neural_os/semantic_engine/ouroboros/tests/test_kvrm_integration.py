#!/usr/bin/env python3
"""
KVRM Integration Test Suite
============================
Tests for the OUROBOROS-KVRM integrated system.
"""

import sys
import os

# Add paths
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvrm_integration import (
    AgentKVRM,
    AgentConfig,
    NarratorKVRM,
    NarratorConfig,
    TrustLevel,
    OuroborosOrganism,
    OuroborosConfig,
    MetaLearner,
    LearningSignal,
)
from core.shared_memory import SharedKVMemory


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  [FAIL] {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_shared_memory():
    """Test SharedKVMemory basic operations."""
    print("\n[TEST] SharedKVMemory")
    results = TestResults()

    memory = SharedKVMemory()

    # Write
    memory.write("test:key1", {"value": 42}, source="test")
    results.record("write", "test:key1" in memory)

    # Read
    entry = memory.read("test:key1")
    results.record("read", entry is not None and entry.value.get("value") == 42)

    # Pattern read
    memory.write("test:key2", {"value": 100}, source="test")
    entries = memory.read_pattern("test:*")
    results.record("pattern_read", len(entries) == 2)

    # Cleanup
    memory.clear()
    results.record("clear", len(memory) == 0)

    return results.summary()


def test_agent_creation():
    """Test AgentKVRM creation and configuration."""
    print("\n[TEST] AgentKVRM Creation")
    results = TestResults()

    # Create competitive agent
    config = AgentConfig(
        name="test_agent_0",
        mode="competitive",
        llm_model="llama3.1:8b",
    )
    agent = AgentKVRM(config)

    results.record("agent_created", agent is not None)
    results.record("agent_name", agent.name == "test_agent_0")
    results.record("agent_mode", agent.agent_config.mode == "competitive")
    results.record("has_llm", agent.llm is not None)

    # Check schemas
    input_schema = agent.get_input_schema()
    results.record("input_schema", "problem:current" in input_schema)

    output_schema = agent.get_output_schema()
    results.record("output_schema", f"solution:{agent.name}" in output_schema)

    return results.summary()


def test_narrator_creation():
    """Test NarratorKVRM with trust levels."""
    print("\n[TEST] NarratorKVRM Creation")
    results = TestResults()

    config = NarratorConfig(
        name="test_narrator",
        trust_level=TrustLevel.GUIDE,
    )
    narrator = NarratorKVRM(config)

    results.record("narrator_created", narrator is not None)
    results.record("trust_level", narrator.trust_level == TrustLevel.GUIDE)
    results.record("pending_overrides_empty", len(narrator.pending_overrides) == 0)

    return results.summary()


def test_narrator_override_requires_approval():
    """Test that OVERRIDE requires human approval."""
    print("\n[TEST] Override Requires Human Approval")
    results = TestResults()

    config = NarratorConfig(
        name="test_narrator",
        trust_level=TrustLevel.OVERRIDE,
    )
    narrator = NarratorKVRM(config)

    # Create a mock override request
    narrator.pending_overrides["test_request"] = {
        "request_id": "test_request",
        "reason": "Test override",
        "status": "PENDING_HUMAN_APPROVAL",
    }

    # Check it requires approval
    results.record("override_pending", len(narrator.pending_overrides) == 1)

    # Approve
    result = narrator.approve_override("test_request", "test_human")
    results.record("approve_success", result.get("success", False))
    results.record("pending_cleared", len(narrator.pending_overrides) == 0)

    return results.summary()


def test_meta_learner():
    """Test MetaLearner tracking."""
    print("\n[TEST] MetaLearner")
    results = TestResults()

    memory = SharedKVMemory()
    learner = MetaLearner(memory)

    # Record signals
    signal = LearningSignal(
        agent_id="test_agent",
        generation=1,
        fitness_before=0.2,
        fitness_after=0.5,
        action_type="mutation",
        action_details="test",
        success=True,
    )
    learner.record_signal(signal)

    results.record("signal_recorded", len(learner.signals) == 1)

    # Check improvement
    results.record("improvement_positive", signal.improvement() == 0.3)

    # Get summary
    summary = learner.get_summary()
    results.record("summary_has_signals", summary.get("total_signals") == 1)

    # Check memory was written
    entries = memory.read_pattern("meta:*")
    results.record("memory_written", len(entries) > 0)

    return results.summary()


def test_organism_creation():
    """Test OuroborosOrganism creation."""
    print("\n[TEST] OuroborosOrganism Creation")
    results = TestResults()

    config = OuroborosConfig(
        num_competitive_agents=2,
        num_cooperative_agents=1,
        narrator_trust_level=TrustLevel.GUIDE,
        max_generations=3,
        enable_meta_learning=True,
        enable_emergence_detection=True,
    )

    organism = OuroborosOrganism(config)

    results.record("organism_created", organism is not None)
    results.record("has_agents", len(organism.kvrms) == 4)  # 2 comp + 1 coop + 1 narrator
    results.record("has_memory", organism.memory is not None)
    results.record("has_meta_learner", organism.meta_learner is not None)

    # Check agent modes
    competitive_count = sum(1 for k in organism.kvrms if hasattr(k, 'agent_config') and k.agent_config.mode == "competitive")
    cooperative_count = sum(1 for k in organism.kvrms if hasattr(k, 'agent_config') and k.agent_config.mode == "cooperative")
    results.record("competitive_agents", competitive_count == 2)
    results.record("cooperative_agents", cooperative_count == 1)

    return results.summary()


def test_organism_set_problem():
    """Test setting a problem for the organism."""
    print("\n[TEST] Organism Set Problem")
    results = TestResults()

    config = OuroborosConfig(
        num_competitive_agents=1,
        num_cooperative_agents=1,
        max_generations=2,
    )
    organism = OuroborosOrganism(config)

    problem = {
        "description": "Test problem",
        "test_cases": [{"input": [1, 2], "expected": 3}],
    }
    organism.set_problem(problem)

    # Check problem was injected into memory
    entries = organism.observe("problem:current")
    results.record("problem_injected", len(entries) > 0)

    if entries:
        results.record("problem_content", entries[0].value.get("description") == "Test problem")
    else:
        results.record("problem_content", False, "No problem entry found")

    return results.summary()


def test_visualization_data():
    """Test that visualization data is generated."""
    print("\n[TEST] Visualization Data")
    results = TestResults()

    config = OuroborosConfig(
        num_competitive_agents=1,
        num_cooperative_agents=1,
        max_generations=2,
    )
    organism = OuroborosOrganism(config)

    viz_data = organism.get_visualization_data()

    results.record("has_fitness_history", "fitness_history" in viz_data)
    results.record("has_emergence_signals", "emergence_signals" in viz_data)
    results.record("has_memory_snapshot", "memory_snapshot" in viz_data)
    results.record("has_event_timeline", "event_timeline" in viz_data)
    results.record("has_narrator_status", "narrator_status" in viz_data)

    return results.summary()


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("       OUROBOROS-KVRM INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        test_shared_memory,
        test_agent_creation,
        test_narrator_creation,
        test_narrator_override_requires_approval,
        test_meta_learner,
        test_organism_creation,
        test_organism_set_problem,
        test_visualization_data,
    ]

    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("       ALL TESTS PASSED")
    else:
        print("       SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
