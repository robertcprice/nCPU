#!/usr/bin/env python3
"""
COMPREHENSIVE TESTS: All moonshots and methods
Tests each component individually and the integrated system
"""

import time
import json
from sympy import Integer, Symbol
from singularity_core import SingularityCore

print("""
╔══════════════════════════════════════════════════════════════════╗
║           COMPREHENSIVE SINGULARITY CORE TESTS                    ║
╚══════════════════════════════════════════════════════════════════╝
""")

# Initialize
core = SingularityCore(enable_all=True)

results = {
    'tests': [],
    'summary': {
        'total': 0,
        'passed': 0,
        'failed': 0,
        'neural_used': 0,
        'coded_used': 0
    }
}

def run_test(name, input_val, target_val, expected_contains=None, use_holographic=True, use_annealing=True):
    """Run a single test and record results"""
    global results

    start = time.time()
    result = core.synthesize(
        Integer(input_val),
        Integer(target_val),
        use_holographic=use_holographic,
        use_annealing=use_annealing
    )
    elapsed = (time.time() - start) * 1000

    solutions = result.get('solutions', [])
    best = result.get('best_solution')
    method = result.get('method')

    # Check if passed
    passed = best is not None
    if expected_contains and passed:
        passed = any(expected_contains.lower() in str(s.get('result', '')).lower()
                    for s in solutions) or expected_contains.lower() in str(best).lower()

    # Track neural vs coded
    for sol in solutions:
        m = sol.get('method', '')
        if m in ['trained_model', 'mco']:
            results['summary']['neural_used'] += 1
        else:
            results['summary']['coded_used'] += 1

    test_result = {
        'name': name,
        'input': input_val,
        'target': target_val,
        'best_solution': best,
        'method': method,
        'solutions_count': len(solutions),
        'methods_used': list(set(s.get('method') for s in solutions)),
        'time_ms': elapsed,
        'passed': passed
    }

    results['tests'].append(test_result)
    results['summary']['total'] += 1
    results['summary']['passed' if passed else 'failed'] += 1

    status = "✅" if passed else "❌"
    print(f"{status} {name}: {input_val}→{target_val} = {best} via {method} ({elapsed:.1f}ms)")

    return passed

print("\n" + "="*60)
print("TEST SUITE 1: BASIC OPERATIONS")
print("="*60)

run_test("identity", 5, 5, "identity")
run_test("double", 5, 10, "double")
run_test("negate", 5, -5, "negate")
run_test("add_ten", 5, 15, "add")

print("\n" + "="*60)
print("TEST SUITE 2: MEDIUM OPERATIONS")
print("="*60)

run_test("square", 5, 25, "square")
run_test("abs_positive", 5, 5, "identity")
run_test("increment", 5, 6, "increment")
run_test("decrement", 5, 4, "decrement")

print("\n" + "="*60)
print("TEST SUITE 3: HARD OPERATIONS")
print("="*60)

run_test("double_square", 3, 36, "double")
run_test("square_add_ten", 3, 19)
run_test("negate_double", 3, -6, "negate")

print("\n" + "="*60)
print("TEST SUITE 4: EXPERT OPERATIONS")
print("="*60)

run_test("cube", 3, 27, "cube")
run_test("triple", 4, 12, "triple")
run_test("quadruple", 3, 12, "quadruple")

print("\n" + "="*60)
print("TEST SUITE 5: EDGE CASES")
print("="*60)

run_test("small_identity", 2, 2, "identity")
run_test("ten_identity", 10, 10, "identity")
run_test("five_squared", 5, 25, "square")
run_test("ten_squared", 10, 100, "square")

print("\n" + "="*60)
print("TEST SUITE 6: METHOD ISOLATION TESTS")
print("="*60)

# Test with only holographic
print("\n--- Holographic Only ---")
run_test("holo_square", 4, 16, "square", use_holographic=True, use_annealing=False)

# Test with only annealing
print("\n--- Annealing Only ---")
run_test("anneal_double", 6, 12, "double", use_holographic=False, use_annealing=True)

# Test with neither (pure neural/coded)
print("\n--- Pure Neural/Coded ---")
run_test("pure_triple", 5, 15, "triple", use_holographic=False, use_annealing=False)

print("\n" + "="*60)
print("TEST SUITE 7: STRESS TEST (Multiple Queries)")
print("="*60)

stress_start = time.time()
stress_tests = [
    (2, 4), (3, 9), (4, 16), (5, 25), (6, 36),  # squares
    (1, 2), (2, 4), (3, 6), (4, 8), (5, 10),     # doubles
    (7, 17), (8, 18), (9, 19), (10, 20),         # add_ten
]
stress_passed = 0
for inp, out in stress_tests:
    result = core.synthesize(Integer(inp), Integer(out))
    if result.get('best_solution'):
        stress_passed += 1
stress_time = (time.time() - stress_start) * 1000

print(f"Stress test: {stress_passed}/{len(stress_tests)} passed in {stress_time:.1f}ms")
print(f"Average: {stress_time/len(stress_tests):.1f}ms per query")

print("\n" + "="*60)
print("TEST SUITE 8: MOONSHOT COMPONENT STATUS")
print("="*60)

status = core.status()
print(f"Capability: {status.get('capability', 0)*100:.0f}%")
print(f"Layers: {status.get('layers', {})}")
print(f"Moonshots: {status.get('moonshots', {})}")

print("\n" + "="*60)
print("TEST SUITE 9: BENCHMARK EVALUATION")
print("="*60)

bench = core.evaluate_on_benchmarks()
print(f"Overall Accuracy: {bench.get('overall_accuracy', 0)*100:.1f}%")
for name, data in bench.get('benchmarks', {}).items():
    print(f"  {name}: {data.get('accuracy', 0)*100:.1f}% ({data.get('samples', 0)} samples)")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary = results['summary']
print(f"""
Total Tests:    {summary['total']}
Passed:         {summary['passed']} ({summary['passed']/summary['total']*100:.1f}%)
Failed:         {summary['failed']}

Neural Uses:    {summary['neural_used']}
Coded Uses:     {summary['coded_used']}
Neural/Total:   {summary['neural_used']/(summary['neural_used']+summary['coded_used'])*100:.1f}%

Method Distribution (across all solutions):
""")

# Count methods
method_counts = {}
for test in results['tests']:
    for m in test.get('methods_used', []):
        method_counts[m] = method_counts.get(m, 0) + 1

for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
    bar = "█" * (count * 2)
    print(f"  {method:20} {bar} ({count})")

# Save results
with open('comprehensive_test_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to comprehensive_test_results.json")

# Determine if system is working well
overall_pass_rate = summary['passed'] / summary['total']
neural_ratio = summary['neural_used'] / (summary['neural_used'] + summary['coded_used'])

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  VERDICT: {"EXCELLENT" if overall_pass_rate >= 0.95 else "NEEDS WORK"} - {overall_pass_rate*100:.0f}% pass rate                              ║
║  Neural/Coded Balance: {neural_ratio*100:.0f}% neural, {(1-neural_ratio)*100:.0f}% coded                 ║
╚══════════════════════════════════════════════════════════════════╝
""")
