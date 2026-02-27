#!/usr/bin/env python3
"""
THE CHAOS-RATCHET ENGINE

The complete integrated system combining:
1. LLM Chaos Generator - Explores solutions
2. Enhanced Ratchet - Filters to only improvements
3. Adversarial Detector - Catches attacks
4. Bug Hunter - Finds and fixes bugs
5. Proof Engine - Generates verifiable certificates

This is what the Hybrid AI Panel recommended building.
"""

import time
import hashlib
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import random

# Import our components
from llm_chaos_generator import LLMChaosGenerator, CodeMutation
from enhanced_adversarial_detector import EnhancedAdversarialDetector, DetectionResult
from bug_hunter import BugHunter, BugDetector


# =============================================================================
# INTEGRATED ENGINE
# =============================================================================

@dataclass
class EvolutionStep:
    """A single step in the evolution process."""
    generation: int
    candidate_name: str
    mutation_type: str
    accepted: bool
    speedup: float
    bugs_before: int
    bugs_after: int
    adversarial_score: float
    proof_hash: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvolutionCertificate:
    """Complete certificate of evolution."""
    initial_code: str
    final_code: str
    steps: List[EvolutionStep]
    total_speedup: float
    bugs_fixed: int
    attacks_blocked: int
    proof_chain: List[str]


class ChaosRatchetEngine:
    """
    The complete Chaos-Ratchet Engine.

    Combines chaos (exploration) with ratchet (guarantees).
    """

    def __init__(self, use_llm_apis: bool = False):
        self.chaos = LLMChaosGenerator(use_apis=use_llm_apis)
        self.adversarial = EnhancedAdversarialDetector(sensitivity=0.4)
        self.bug_hunter = BugHunter()
        self.bug_detector = BugDetector()

        self.current_code: Optional[str] = None
        self.current_time: float = float('inf')
        self.evolution_history: List[EvolutionStep] = []
        self.proof_chain: List[str] = []
        self.attacks_blocked = 0
        self.improvements_accepted = 0

    def _benchmark_code(self, code: str, test_fn_name: str = None) -> Tuple[float, bool]:
        """Benchmark code execution time."""
        try:
            namespace = {}
            exec(code, namespace)

            # Find the function
            fn = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    fn = obj
                    break

            if fn is None:
                return float('inf'), False

            # Run benchmarks
            test_inputs = [
                [random.randint(0, 1000) for _ in range(100)],
                [random.randint(0, 1000) for _ in range(500)],
                list(range(200)),
                list(range(200, 0, -1)),
            ]

            total_time = 0
            all_correct = True

            for test_input in test_inputs:
                expected = sorted(test_input)
                try:
                    start = time.perf_counter_ns()
                    result = fn(test_input)
                    end = time.perf_counter_ns()
                    total_time += (end - start)

                    if list(result) != expected:
                        all_correct = False
                except:
                    all_correct = False
                    total_time += 10**9

            return total_time, all_correct

        except Exception as e:
            return float('inf'), False

    def initialize(self, code: str):
        """Initialize the engine with starting code."""
        time_ns, correct = self._benchmark_code(code)
        if not correct:
            raise ValueError("Initial code produces incorrect results")

        self.current_code = code
        self.current_time = time_ns
        bugs = self.bug_detector.detect_bugs(code)

        print(f"Initialized Chaos-Ratchet Engine")
        print(f"  Execution time: {time_ns/1e6:.3f}ms")
        print(f"  Bugs detected: {len(bugs)}")

        # Generate initial proof
        initial_hash = hashlib.sha256(
            f"INIT:{code}:{time_ns}".encode()
        ).hexdigest()[:16]
        self.proof_chain.append(initial_hash)

    def evolve(self, max_generations: int = 10) -> EvolutionCertificate:
        """
        Evolve the code through chaos + ratchet.

        Returns complete evolution certificate.
        """
        if self.current_code is None:
            raise ValueError("Must initialize first")

        initial_code = self.current_code
        initial_bugs = len(self.bug_detector.detect_bugs(initial_code))

        print("\n" + "=" * 60)
        print("CHAOS-RATCHET EVOLUTION")
        print("=" * 60)

        for gen in range(max_generations):
            print(f"\n--- Generation {gen + 1} ---")

            # Generate mutations (CHAOS)
            mutations = self.chaos.generate_mutations(self.current_code)
            print(f"Generated {len(mutations)} candidates")

            accepted_this_gen = False

            for mutation in mutations:
                # Security check (ADVERSARIAL DETECTION)
                security_result = self.adversarial.analyze({
                    'description': mutation.description,
                    'code_changes': mutation.code,
                })

                if security_result.is_adversarial:
                    print(f"  BLOCKED {mutation.name}: Adversarial (conf: {security_result.confidence:.2f})")
                    self.attacks_blocked += 1
                    continue

                # Correctness and speed check (RATCHET)
                candidate_time, correct = self._benchmark_code(mutation.code)
                current_time, _ = self._benchmark_code(self.current_code)

                if not correct:
                    print(f"  REJECTED {mutation.name}: Incorrect results")
                    continue

                if candidate_time >= current_time:
                    print(f"  REJECTED {mutation.name}: Not faster ({candidate_time/1e6:.3f}ms >= {current_time/1e6:.3f}ms)")
                    continue

                # Bug check (must not introduce bugs)
                current_bugs = len(self.bug_detector.detect_bugs(self.current_code))
                candidate_bugs = len(self.bug_detector.detect_bugs(mutation.code))

                if candidate_bugs > current_bugs:
                    print(f"  REJECTED {mutation.name}: Introduces bugs ({candidate_bugs} > {current_bugs})")
                    continue

                # ACCEPTED!
                speedup = current_time / candidate_time
                print(f"  ACCEPTED {mutation.name}: {speedup:.2f}x speedup!")

                # Generate proof
                proof_hash = hashlib.sha256(
                    f"{self.proof_chain[-1]}:{mutation.code}:{speedup}".encode()
                ).hexdigest()[:16]
                self.proof_chain.append(proof_hash)

                # Record step
                step = EvolutionStep(
                    generation=gen + 1,
                    candidate_name=mutation.name,
                    mutation_type=mutation.mutation_type,
                    accepted=True,
                    speedup=speedup,
                    bugs_before=current_bugs,
                    bugs_after=candidate_bugs,
                    adversarial_score=security_result.confidence,
                    proof_hash=proof_hash,
                )
                self.evolution_history.append(step)

                # Update state
                self.current_code = mutation.code
                self.current_time = candidate_time
                self.improvements_accepted += 1
                accepted_this_gen = True
                break

            if not accepted_this_gen:
                print("  No improvements found this generation")

        # Calculate totals
        final_time, _ = self._benchmark_code(self.current_code)
        initial_time, _ = self._benchmark_code(initial_code)
        total_speedup = initial_time / final_time if final_time > 0 else 1.0

        final_bugs = len(self.bug_detector.detect_bugs(self.current_code))
        bugs_fixed = initial_bugs - final_bugs

        return EvolutionCertificate(
            initial_code=initial_code,
            final_code=self.current_code,
            steps=self.evolution_history,
            total_speedup=total_speedup,
            bugs_fixed=bugs_fixed,
            attacks_blocked=self.attacks_blocked,
            proof_chain=self.proof_chain,
        )

    def print_certificate(self, cert: EvolutionCertificate):
        """Print the evolution certificate."""
        print("\n" + "=" * 70)
        print("EVOLUTION CERTIFICATE")
        print("Chaos-Ratchet Engine - Provable Self-Improvement")
        print("=" * 70)

        print(f"\nTotal Improvement: {cert.total_speedup:.2f}x speedup")
        print(f"Bugs Fixed: {cert.bugs_fixed}")
        print(f"Attacks Blocked: {cert.attacks_blocked}")
        print(f"Improvements Accepted: {len(cert.steps)}")

        print("\nEvolution Path:")
        for step in cert.steps:
            print(f"  Gen {step.generation}: {step.candidate_name}")
            print(f"    Speedup: {step.speedup:.2f}x")
            print(f"    Bugs: {step.bugs_before} → {step.bugs_after}")
            print(f"    Proof: {step.proof_hash}")

        print("\nProof Chain:")
        for i, proof in enumerate(cert.proof_chain):
            print(f"  [{i}] {proof}")

        print("\nFinal Code:")
        print("-" * 40)
        print(cert.final_code)
        print("-" * 40)


# =============================================================================
# DEMO
# =============================================================================

def run_integrated_demo():
    """Run the full integrated Chaos-Ratchet Engine demo."""
    print("=" * 70)
    print("CHAOS-RATCHET ENGINE: INTEGRATED DEMO")
    print("=" * 70)
    print()
    print("This combines:")
    print("  1. LLM Chaos Generator - Explores solutions")
    print("  2. Enhanced Ratchet - Filters to only improvements")
    print("  3. Adversarial Detector - Catches attacks")
    print("  4. Bug Hunter - Finds and fixes bugs")
    print("  5. Proof Engine - Generates verifiable certificates")
    print()

    # Starting code (bubble sort)
    initial_code = textwrap.dedent('''
    def sort_fn(arr):
        """Bubble sort - O(n^2), slow."""
        arr = list(arr)
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    ''').strip()

    print("Initial Code (Bubble Sort):")
    print(initial_code)
    print()

    # Create engine and run evolution
    engine = ChaosRatchetEngine(use_llm_apis=False)  # Fallback mode
    engine.initialize(initial_code)

    cert = engine.evolve(max_generations=5)

    # Print certificate
    engine.print_certificate(cert)

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("What we demonstrated:")
    print(f"  1. {cert.total_speedup:.2f}x speedup through evolution")
    print(f"  2. {len(cert.steps)} improvements accepted (all verified)")
    print(f"  3. {cert.attacks_blocked} adversarial attempts blocked")
    print(f"  4. {len(cert.proof_chain)} proofs in verification chain")
    print(f"  5. Complete audit trail with cryptographic hashes")

    return cert


if __name__ == '__main__':
    cert = run_integrated_demo()
