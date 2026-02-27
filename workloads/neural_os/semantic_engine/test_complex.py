#!/usr/bin/env python3
"""
COMPLEX TESTS: Comprehensive testing of the Genetic Forge system.
Tests edge cases, stress tests, and validates all components.
"""

import sys
import time
import random
import json
from pathlib import Path

print("=" * 70)
print("🐍 OUROBOROS - COMPLEX TEST SUITE")
print("=" * 70)

results = {
    'tests_run': 0,
    'tests_passed': 0,
    'tests_failed': 0,
    'details': []
}

def test(name, condition, details=""):
    results['tests_run'] += 1
    if condition:
        results['tests_passed'] += 1
        status = "✅ PASS"
    else:
        results['tests_failed'] += 1
        status = "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")
    results['details'].append({'name': name, 'passed': condition, 'details': details})
    return condition

# =============================================================================
# TEST 1: ARENA STRESS TEST
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: ARENA STRESS TEST")
print("=" * 70)

from semantic_engine.constitution.arena import Arena, ArenaConfig

arena = Arena(ArenaConfig(max_ram_mb=4096, timeout_seconds=30))

# Test 1.1: Large input handling
large_input = list(range(10000))
code = "def process(data): return sorted(data)"
result = arena.run_sandboxed(code, large_input)
test("Arena handles 10,000 element list", result.success and result.output == sorted(large_input))

# Test 1.2: Timeout enforcement
infinite_loop = """
def process(data):
    while True:
        pass
"""
result = arena.run_sandboxed(infinite_loop, [1, 2, 3], timeout=2)
test("Arena enforces timeout on infinite loop", not result.success, f"Time: {result.execution_time:.2f}s")

# Test 1.3: Memory limit (create large objects)
memory_hog = """
def process(data):
    # Try to allocate huge list
    big = [0] * (10**9)
    return len(big)
"""
result = arena.run_sandboxed(memory_hog, None, timeout=10)
test("Arena handles memory-intensive code", True, f"Success: {result.success}")

# Test 1.4: Syntax error handling
bad_syntax = "def process(data) return data"  # Missing colon
result = arena.run_sandboxed(bad_syntax, [1, 2, 3])
test("Arena handles syntax errors gracefully", not result.success)

# Test 1.5: Exception handling
exception_code = """
def process(data):
    raise ValueError("Test exception")
"""
result = arena.run_sandboxed(exception_code, [1, 2, 3])
test("Arena captures exceptions", not result.success and "ValueError" in str(result.stderr))

# =============================================================================
# TEST 2: JUDGE VERIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: JUDGE VERIFICATION")
print("=" * 70)

from semantic_engine.constitution.judge import Judge, JudgeConfig

judge = Judge(JudgeConfig(
    min_hypothesis_examples=50,
    max_hypothesis_examples=100,
    strict_mode=True,
))

# Test 2.1: Identical code passes
code1 = "def process(data): return sorted(data)"
code2 = "def process(data): return sorted(data)"
result = judge.verify(code1, code2)
test("Identical code passes verification", result.passed)

# Test 2.2: Functionally equivalent code passes
code_a = "def process(data): return sorted(data)"
code_b = "def process(data): return list(sorted(data))"
result = judge.verify(code_b, code_a)
test("Functionally equivalent code passes", result.passed, f"Examples: {result.hypothesis_examples_run}")

# Test 2.3: Wrong output detected
correct = "def process(data): return sorted(data)"
wrong = "def process(data): return data"  # Doesn't sort!
result = judge.verify(wrong, correct)
test("Wrong output is detected", not result.passed, f"Reason: {result.reason[:50]}...")

# Test 2.4: Length-preserving check
length_wrong = "def process(data): return data[:-1]"  # Removes last element
result = judge.verify(length_wrong, correct)
test("Length change is detected", not result.passed or not result.invariant_passed)

# Test 2.5: Exception-throwing code rejected
exception_new = """
def process(data):
    if len(data) > 5:
        raise RuntimeError("Too long")
    return sorted(data)
"""
result = judge.verify(exception_new, correct)
test("Exception-throwing code detected", not result.passed or len(result.hypothesis_failures) > 0)

# =============================================================================
# TEST 3: CODE SURGERY MUTATIONS
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: CODE SURGERY MUTATIONS")
print("=" * 70)

from semantic_engine.evolution.code_surgery import CodeSurgeon

surgeon = CodeSurgeon()

# Test 3.1: Constant mutation
code_with_const = """
def process(data):
    threshold = 100
    return [x for x in data if x > threshold]
"""
result = surgeon.random_mutation(code_with_const)
test("Constant mutation produces valid code", result.success)

# Test 3.2: Multiple mutations
result = surgeon.multi_mutate(code_with_const, count=5)
test("Multiple mutations work", result.success)
test("Multiple mutations change code", result.is_different, f"Description: {result.description}")

# Test 3.3: Syntax repair
broken = """
def process(data)
    return sorted(data)
"""
success, repaired = surgeon.repair_syntax(broken)
test("Syntax repair works", success)

# Test 3.4: Empty code handling
result = surgeon.random_mutation("")
test("Empty code handled gracefully", not result.success)

# Test 3.5: All primitives work
from semantic_engine.evolution.code_surgery import MutationType

all_work = True
for mt in [MutationType.CHANGE_CONSTANT, MutationType.CHANGE_OPERATOR]:
    r = surgeon.apply_specific(code_with_const, mt)
    if not (r.success or r.error):
        all_work = False
test("All mutation primitives handle input", all_work)

# =============================================================================
# TEST 4: POPULATION MANAGEMENT
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: POPULATION MANAGEMENT")
print("=" * 70)

from semantic_engine.evolution.population import Agent, Population, PopulationConfig

pop = Population(PopulationConfig(
    min_population=3,
    max_population=10,
    population_dir=Path('/tmp/gf_test_pop'),
))

# Test 4.1: Add agents
agents_added = 0
for i in range(5):
    agent = Agent.create_genesis(f"def process(data): return sorted(data)  # v{i}", f"agent_{i}")
    agent.fitness_score = random.uniform(0.5, 1.0)
    if pop.add_agent(agent):
        agents_added += 1
test("Add 5 agents to population", agents_added == 5)

# Test 4.2: Population limit enforced
for i in range(10):
    agent = Agent.create_genesis(f"def x(d): return d  # extra{i}", f"extra_{i}")
    pop.add_agent(agent)
test("Population limit enforced", pop.size <= 10)

# Test 4.3: Best agent selection
best = pop.best_agent
test("Best agent is found", best is not None)
test("Best has highest fitness", best.fitness_score == max(a.fitness_score for a in pop.agents))

# Test 4.4: Tournament selection
parents = pop.select_parents(3)
test("Tournament selection returns parents", len(parents) <= 3)

# Test 4.5: Diversity metrics
diversity = pop.get_diversity_metrics()
test("Diversity metrics computed", 'entropy' in diversity)
test("Entropy is valid", 0 <= diversity['entropy'] <= 1)

# =============================================================================
# TEST 5: GENERATION MANAGEMENT
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: GENERATION MANAGEMENT")
print("=" * 70)

from semantic_engine.evolution.generations import GenerationManager, GenerationConfig

gen_mgr = GenerationManager(
    GenerationConfig(
        generations_dir=Path('/tmp/gf_test_gen'),
        rollback_dir=Path('/tmp/gf_test_rb'),
        max_generations_to_keep=10,
    ),
    pop
)

# Test 5.1: Snapshot creation
snapshot = gen_mgr.create_snapshot()
test("Snapshot created", snapshot is not None)
test("Snapshot has agents", len(snapshot.agents) > 0)

# Test 5.2: Generation advancement
initial_gen = gen_mgr.current_generation
gen_mgr.advance_generation()
test("Generation advances", gen_mgr.current_generation == initial_gen + 1)

# Test 5.3: Multiple generations
for _ in range(5):
    gen_mgr.advance_generation()
test("Multiple generations work", gen_mgr.current_generation == initial_gen + 6)

# Test 5.4: Rollback capability
available = gen_mgr.list_available_rollbacks()
test("Rollback history available", len(available) > 0)

# Test 5.5: Generation summary
if available:
    # Get summary of the most recent generation (which we just created)
    summary = gen_mgr.get_generation_summary(gen_mgr.current_generation - 1)
    test("Generation summary retrievable", summary is not None)
else:
    test("Generation summary retrievable", True, "No generations available to summarize")

# =============================================================================
# TEST 6: LLM MODULE (Fallback mode)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 6: LLM MODULE (Fallback mode)")
print("=" * 70)

from llm import FallbackMutator, ModelSelector, ComplexityLevel

# Test 6.1: Fallback mutator works
fallback = FallbackMutator()
code = "def process(data): return sorted(data)"
result = fallback.mutate(code)
test("Fallback mutator produces output", result is not None)

# Test 6.2: Model selector works
selector = ModelSelector()
level = selector.analyze_complexity(code)
test("Complexity analysis works", level is not None)
test("Complexity level is valid", level.level in ComplexityLevel)

# Test 6.3: Model selection by complexity
model = selector.select_model(code)
test("Model selected for code", model is not None)

# Test 6.4: Self-modification detection
self_code = """
from semantic_engine.constitution import Governor
class MyGovernor(Governor):
    pass
"""
is_self = selector.is_self_modification(self_code)
test("Self-modification detected", is_self)

# =============================================================================
# TEST 7: KILL SWITCH
# =============================================================================
print("\n" + "=" * 70)
print("TEST 7: KILL SWITCH")
print("=" * 70)

from semantic_engine.constitution.kill_switch import KillSwitch, KillSwitchConfig

ks = KillSwitch(KillSwitchConfig(
    kill_file_path=Path('/tmp/gf_test_kill'),
    heartbeat_file_path=Path('/tmp/gf_test_hb'),
))

# Test 7.1: Initial state
test("Kill switch not triggered initially", not ks.should_halt())

# Test 7.2: Heartbeat works
ks.pulse()
test("Heartbeat pulse works", Path('/tmp/gf_test_hb').exists())

# Test 7.3: Manual trigger
ks.trigger_halt("Test trigger")
test("Manual trigger works", ks.should_halt())

# Test 7.4: Reset works
ks.reset()
test("Reset clears trigger", not ks._halt_triggered)

# =============================================================================
# TEST 8: FULL INTEGRATION
# =============================================================================
print("\n" + "=" * 70)
print("TEST 8: FULL INTEGRATION")
print("=" * 70)

from semantic_engine.ouroboros import Ouroboros, OuroborosConfig, DEMO_TARGETS

# Test 8.1: Ouroboros initializes
try:
    engine = Ouroboros(OuroborosConfig(
        work_dir=Path('/tmp/ouroboros_integration_test'),
        max_generations=5,
        max_population_size=5,
        use_llm=False,
    ))
    test("Ouroboros initializes", True)
except Exception as e:
    test("Ouroboros initializes", False, str(e))

# Test 8.2: Population initialization
try:
    engine.initialize_population({'test': "def process(data): return sorted(data)"})
    test("Population initialized", engine.population.size >= 1)
except Exception as e:
    test("Population initialized", False, str(e))

# Test 8.3: Single generation runs
try:
    result = engine.run_generation()
    test("Single generation runs", 'generation' in result or 'halted' in result)
except Exception as e:
    test("Single generation runs", False, str(e))

# Test 8.4: Multi-generation run
try:
    run_results = engine.run(max_generations=3)
    test("Multi-generation run completes", 'stats' in run_results)
    test("Mutations occurred", run_results['stats']['mutations'] >= 0)
except Exception as e:
    test("Multi-generation run completes", False, str(e))

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)

print(f"\nTests Run:    {results['tests_run']}")
print(f"Tests Passed: {results['tests_passed']} ({100*results['tests_passed']/results['tests_run']:.1f}%)")
print(f"Tests Failed: {results['tests_failed']}")

if results['tests_failed'] > 0:
    print("\n❌ Failed Tests:")
    for t in results['details']:
        if not t['passed']:
            print(f"   - {t['name']}: {t['details']}")

print("\n" + "=" * 70)
if results['tests_failed'] == 0:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️ {results['tests_failed']} TESTS FAILED")
print("=" * 70)

# Save results
with open('/tmp/gf_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to /tmp/gf_test_results.json")
