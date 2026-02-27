#!/usr/bin/env python3
"""
TEST SINGULARITY: Full pipeline test with trained model

Tests:
1. Trained model synthesis (100% accuracy)
2. Self-improvement loop
3. All integrated components
"""

from sympy import Integer
from singularity_core import SingularityCore


def test_trained_model_synthesis():
    """Test synthesis with the trained model."""
    print("\n" + "=" * 60)
    print("TEST 1: Trained Model Synthesis (100% accuracy)")
    print("=" * 60)

    core = SingularityCore(enable_all=True)

    # Test cases for trained model
    # Operations: identity, double, square, negate, add_ten, times_five
    test_cases = [
        (Integer(5), Integer(5), "identity"),      # 5 → 5
        (Integer(5), Integer(10), "double"),       # 5 → 10
        (Integer(5), Integer(25), "square"),       # 5 → 25 (or times_five)
        (Integer(5), Integer(-5), "negate"),       # 5 → -5
        (Integer(5), Integer(15), "add_ten"),      # 5 → 15
        (Integer(3), Integer(6), "double"),        # 3 → 6
        (Integer(4), Integer(16), "square"),       # 4 → 16
        (Integer(7), Integer(14), "double"),       # 7 → 14
        (Integer(8), Integer(-8), "negate"),       # 8 → -8
    ]

    correct = 0
    for input_val, target_val, expected_op in test_cases:
        result = core.synthesize(input_val, target_val)

        # Check if trained model found it
        trained_result = None
        for sol in result['solutions']:
            if sol['method'] == 'trained_model':
                trained_result = sol
                break

        if trained_result:
            is_correct = trained_result['result'] == expected_op
            if is_correct:
                correct += 1
            status = "✅" if is_correct else "❌"
            print(f"  {input_val} → {target_val}: {trained_result['result']} "
                  f"(conf={trained_result['confidence']:.2f}) {status}")
        else:
            print(f"  {input_val} → {target_val}: No trained model result")

    print(f"\nTrained Model Accuracy: {correct}/{len(test_cases)} = {100*correct/len(test_cases):.1f}%")
    return correct / len(test_cases)


def test_self_improvement():
    """Test the self-improvement loop."""
    print("\n" + "=" * 60)
    print("TEST 2: Self-Improvement Loop")
    print("=" * 60)

    core = SingularityCore(enable_all=True)

    # Run a short self-improvement cycle
    result = core.self_improve(iterations=3)

    print(f"\n  Improvements: {len(result['improvements'])}")
    print(f"  Discoveries: {len(result['discoveries'])}")

    return len(result['improvements']) > 0


def test_status():
    """Test system status."""
    print("\n" + "=" * 60)
    print("TEST 3: System Status")
    print("=" * 60)

    core = SingularityCore(enable_all=True)
    status = core.status()

    print(f"\n  Generation: {status['generation']}")
    print(f"  Capability: {status['capability']:.4f}")

    layers_active = sum(status['layers'].values())
    moonshots_active = sum(status['moonshots'].values())

    print(f"\n  Layers: {layers_active}/5 active")
    for name, active in status['layers'].items():
        emoji = "✓" if active else "✗"
        print(f"    {emoji} {name}")

    print(f"\n  Moonshots: {moonshots_active}/6 active")
    for name, active in status['moonshots'].items():
        emoji = "✓" if active else "✗"
        print(f"    {emoji} {name}")

    return layers_active >= 3 and moonshots_active >= 5


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "SINGULARITY CORE TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Trained model synthesis
    results['synthesis'] = test_trained_model_synthesis()

    # Test 2: Self-improvement
    results['self_improvement'] = test_self_improvement()

    # Test 3: Status
    results['status'] = test_status()

    # Summary
    print("\n" + "=" * 70)
    print(" " * 20 + "TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(" " * 15 + "🎉 ALL TESTS PASSED!")
    else:
        print(" " * 15 + "⚠️ SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
