#!/usr/bin/env python3
"""
DETAILED TEST: Shows exactly what happens in one evolutionary loop.
This demonstrates the full Ouroboros pipeline step by step.
"""

import sys
import time
from pprint import pprint

print("=" * 70)
print("🔬 OUROBOROS - DETAILED LOOP DEMONSTRATION")
print("=" * 70)

# =============================================================================
# STEP 1: THE TARGET CODE (What we're trying to optimize)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: TARGET CODE TO OPTIMIZE")
print("=" * 70)

original_code = '''
def bubble_sort(arr):
    """Bubble sort - O(n²) time complexity. Can we make it faster?"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''

print("\nOriginal Code:")
print("-" * 40)
print(original_code)
print("-" * 40)

# =============================================================================
# STEP 2: THE ARENA - Sandbox Execution
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: THE ARENA - Testing code in isolated sandbox")
print("=" * 70)

from semantic_engine.constitution.arena import Arena, ArenaConfig

arena = Arena(ArenaConfig(
    max_ram_mb=4096,
    timeout_seconds=30,
))

test_input = [64, 34, 25, 12, 22, 11, 90]
print(f"\nTest Input: {test_input}")

result = arena.run_sandboxed(original_code, test_input.copy())

print(f"\n✅ Arena Result:")
print(f"   Success: {result.success}")
print(f"   Output: {result.output}")
print(f"   Execution Time: {result.execution_time:.6f}s")
print(f"   Exit Code: {result.exit_code}")

# Verify correctness
expected = sorted(test_input)
print(f"\n   Expected: {expected}")
print(f"   Match: {result.output == expected}")

# =============================================================================
# STEP 3: CODE SURGERY - AST-Based Mutation
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: CODE SURGERY - Applying AST-based mutations")
print("=" * 70)

from semantic_engine.evolution.code_surgery import CodeSurgeon, MutationType

surgeon = CodeSurgeon()

print("\nAttempting mutations on the code...")
print("-" * 40)

# Try each mutation type
for mutation_type in [MutationType.CHANGE_CONSTANT, MutationType.CHANGE_OPERATOR, MutationType.REORDER_STATEMENTS]:
    result = surgeon.apply_specific(original_code, mutation_type)
    print(f"\n{mutation_type.name}:")
    print(f"   Success: {result.success}")
    print(f"   Description: {result.description}")
    if result.success and result.is_different:
        # Show the diff
        orig_lines = original_code.strip().split('\n')
        new_lines = result.new_code.strip().split('\n')
        for i, (o, n) in enumerate(zip(orig_lines, new_lines)):
            if o != n:
                print(f"   Line {i+1}: '{o.strip()}' → '{n.strip()}'")

# =============================================================================
# STEP 4: THE JUDGE - Hypothesis Fuzzing Verification
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: THE JUDGE - Property-based testing with Hypothesis")
print("=" * 70)

from semantic_engine.constitution.judge import Judge, JudgeConfig

judge = Judge(JudgeConfig(
    min_hypothesis_examples=100,
    max_hypothesis_examples=200,
    strict_mode=False,
))

# Create a mutated version
mutated_code = '''
def bubble_sort(arr):
    """Bubble sort - O(n²) time complexity. Can we make it faster?"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''

# Create a WRONG mutation to show rejection
wrong_mutation = '''
def bubble_sort(arr):
    """This mutation breaks correctness!"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i):  # BUG: Changed range, causes index error
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
'''

print("\nTesting CORRECT mutation:")
print("-" * 40)
verification = judge.verify(mutated_code, original_code)
print(f"   Passed: {verification.passed}")
print(f"   Hypothesis Examples Run: {verification.hypothesis_examples_run}")
print(f"   Reason: {verification.reason}")
print(f"   Confidence: {verification.confidence:.2f}")

print("\nTesting INCORRECT mutation (should be rejected):")
print("-" * 40)
verification_bad = judge.verify(wrong_mutation, original_code)
print(f"   Passed: {verification_bad.passed}")
print(f"   Reason: {verification_bad.reason}")
if verification_bad.hypothesis_failures:
    print(f"   First failure: {verification_bad.hypothesis_failures[0][:100]}...")

# =============================================================================
# STEP 5: POPULATION MANAGEMENT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: POPULATION - Managing competing agents")
print("=" * 70)

from semantic_engine.evolution.population import Agent, Population, PopulationConfig

pop = Population(PopulationConfig(
    min_population=3,
    max_population=10,
    population_dir='/tmp/gf_demo/pop',
))

# Create agents
agent1 = Agent.create_genesis(original_code, "bubble_v1")
agent1.fitness_score = 0.85

agent2_code = '''
def bubble_sort(arr):
    """Optimized: early exit if no swaps"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr
'''
agent2 = Agent.create_genesis(agent2_code, "bubble_v2_optimized")
agent2.fitness_score = 0.92

pop.add_agent(agent1)
pop.add_agent(agent2)

print("\nPopulation Stats:")
stats = pop.get_fitness_stats()
print(f"   Size: {pop.size}")
print(f"   Min Fitness: {stats['min']:.4f}")
print(f"   Max Fitness: {stats['max']:.4f}")
print(f"   Mean Fitness: {stats['mean']:.4f}")

diversity = pop.get_diversity_metrics()
print(f"\nDiversity Metrics:")
print(f"   Shannon Entropy: {diversity['entropy']:.3f}")
print(f"   Unique Solutions: {diversity['unique_count']}/{diversity['total_count']}")

print("\nTournament Selection:")
parents = pop.select_parents(2)
print(f"   Selected parents: {[p.agent_id for p in parents]}")
print(f"   Their fitness scores: {[p.fitness_score for p in parents]}")

# =============================================================================
# STEP 6: GENERATION CYCLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: GENERATION CYCLE - Creating offspring")
print("=" * 70)

from semantic_engine.evolution.generations import GenerationManager, GenerationConfig

gen_mgr = GenerationManager(
    GenerationConfig(
        generations_dir='/tmp/gf_demo/gen',
        rollback_dir='/tmp/gf_demo/rb',
    ),
    pop
)

print(f"\nCurrent Generation: {gen_mgr.current_generation}")

# Create offspring
print("\nCreating offspring from best parents...")
best_parent = pop.best_agent
print(f"   Best parent: {best_parent.agent_id} (fitness: {best_parent.fitness_score})")

# Mutate the best parent
mutated = surgeon.random_mutation(best_parent.source_code)
if mutated.success:
    print(f"   Mutation applied: {mutated.description}")

    # Verify the mutation
    verify_result = judge.verify(mutated.new_code, best_parent.source_code)
    print(f"   Verification: {'PASSED' if verify_result.passed else 'FAILED'}")

    if verify_result.passed:
        offspring = Agent(
            agent_id=f"gen{gen_mgr.current_generation}_offspring_1",
            generation=gen_mgr.current_generation + 1,
            source_code=mutated.new_code,
            parent_ids=[best_parent.agent_id],
            mutation_type="ast",
        )
        offspring.fitness_score = best_parent.fitness_score * 0.99  # Slightly lower (mutation may not improve)
        pop.add_agent(offspring)
        print(f"   Offspring added: {offspring.agent_id}")

# Advance generation
gen_mgr.advance_generation()
print(f"\nAdvanced to Generation: {gen_mgr.current_generation}")

# =============================================================================
# STEP 7: BENCHMARKING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: BENCHMARKING - Measuring performance")
print("=" * 70)

import random

def benchmark_sort(code: str, sizes: list = [100, 500, 1000]) -> dict:
    """Benchmark sorting code on various input sizes."""
    results = {'sizes': sizes, 'times': [], 'correct': []}

    for size in sizes:
        test_data = [random.randint(0, 10000) for _ in range(size)]
        expected = sorted(test_data)

        result = arena.run_sandboxed(code, test_data.copy(), timeout=10)

        results['times'].append(result.execution_time)
        results['correct'].append(result.output == expected if result.success else False)

    return results

print("\nBenchmarking all agents...")
print("-" * 40)

for agent in pop.agents:
    bench = benchmark_sort(agent.source_code)
    avg_time = sum(bench['times']) / len(bench['times'])
    all_correct = all(bench['correct'])

    print(f"\n{agent.agent_id}:")
    print(f"   Correctness: {'✅ All passed' if all_correct else '❌ Some failed'}")
    print(f"   Avg Time: {avg_time:.4f}s")
    for size, time_taken in zip(bench['sizes'], bench['times']):
        print(f"   Size {size}: {time_taken:.4f}s")

# =============================================================================
# STEP 8: KILL SWITCH
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: KILL SWITCH - Safety mechanism")
print("=" * 70)

from semantic_engine.constitution.kill_switch import KillSwitch, KillSwitchConfig

ks = KillSwitch(KillSwitchConfig(
    kill_file_path='/tmp/gf_demo/kill_switch',
    heartbeat_file_path='/tmp/gf_demo/heartbeat',
))

print("\nKill Switch Status:")
status = ks.get_status()
print(f"   Halt Triggered: {status['halt_triggered']}")
print(f"   Kill File Exists: {status['kill_file_exists']}")
print(f"   Running: {status['running']}")

print("\nTo halt the system, simply run:")
print("   touch /tmp/gf_demo/kill_switch")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("📊 SUMMARY: THE COMPLETE EVOLUTIONARY LOOP")
print("=" * 70)

print("""
1. TARGET CODE is loaded into the system

2. ARENA executes code in isolated sandbox
   - Memory limited (4GB)
   - Time limited (30s)
   - Process isolated

3. CODE SURGERY applies AST mutations
   - Change constants
   - Swap operators
   - Reorder statements
   - (Or LLM suggests improvements)

4. JUDGE verifies mutations with Hypothesis
   - 100-500 random test inputs
   - Differential testing (old vs new)
   - Invariant checking
   - REJECT if ANY test fails

5. POPULATION manages competing agents
   - 5-20 agents compete
   - Tournament selection (best breed)
   - Shannon entropy for diversity
   - Weak agents culled

6. GENERATION advances the evolution
   - Offspring from successful mutations
   - 100 generations kept for rollback
   - System can restart as new generation

7. BENCHMARKING measures fitness
   - Correctness on test inputs
   - Speed comparison
   - Combined fitness score

8. KILL SWITCH allows emergency halt
   - File-based trigger
   - Heartbeat monitoring
   - External to Python process
""")

print("=" * 70)
print("🎉 DETAILED LOOP DEMONSTRATION COMPLETE")
print("=" * 70)
