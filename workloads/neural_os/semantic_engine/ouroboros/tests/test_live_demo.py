#!/usr/bin/env python3
"""
OUROBOROS Live Demo
====================
Demonstrates actual system behavior for the 6-AI panel review.
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvrm_integration import (
    OuroborosOrganism,
    OuroborosConfig,
    TrustLevel,
)
from kvrm_integration.visualizations import OuroborosVisualizer


def run_demo():
    """Run a live demo showing system behavior."""
    print("=" * 70)
    print("       OUROBOROS-KVRM LIVE DEMONSTRATION")
    print("=" * 70)

    # Create organism
    print("\n[1] Creating OUROBOROS Organism...")
    config = OuroborosConfig(
        num_competitive_agents=2,
        num_cooperative_agents=1,
        narrator_trust_level=TrustLevel.GUIDE,
        max_generations=3,
        llm_model="llama3.1:8b",
        enable_meta_learning=True,
        enable_emergence_detection=True,
    )

    organism = OuroborosOrganism(config)
    print(f"    Created with {len(organism.kvrms)} KVRMs:")
    for kvrm in organism.kvrms:
        mode = getattr(kvrm, 'agent_config', None)
        mode_str = mode.mode if mode else "narrator"
        print(f"      - {kvrm.name} [{mode_str}]")

    # Set problem
    print("\n[2] Setting Problem...")
    problem = {
        "description": "Write a function to check if a number is prime",
        "test_cases": [
            {"input": 2, "expected": True},
            {"input": 4, "expected": False},
            {"input": 17, "expected": True},
        ],
    }
    organism.set_problem(problem)
    print(f"    Problem: {problem['description']}")

    # Run generations
    print("\n[3] Running Evolution...")
    print("-" * 70)

    for gen_num in range(2):  # Just 2 generations for demo
        print(f"\n  Generation {gen_num + 1}...")
        start = time.time()
        result = organism.run_generation()
        duration = time.time() - start

        print(f"    Duration: {duration:.1f}s")
        print(f"    Best fitness: {result.get('best_fitness', 0):.2f}")
        print(f"    Best agent: {result.get('best_agent', 'none')}")

        # Show solutions
        solutions = result.get('solutions', [])
        if solutions:
            print(f"    Solutions:")
            for sol in solutions:
                print(f"      - {sol.get('agent')}: fitness={sol.get('fitness', 0):.2f} [{sol.get('mode', '?')}]")

        # Show narrator observation
        obs = result.get('narrator_observation', '')
        if obs:
            obs_str = obs.get('value', str(obs)) if isinstance(obs, dict) else str(obs)
            print(f"    Narrator: {obs_str[:80]}")

        # Show emergence
        emergence = result.get('emergence_signals', [])
        if emergence:
            print(f"    Emergence signals: {len(emergence)}")
            for e in emergence:
                print(f"      ! {e.get('signal_type', '?')}")

    # Show final state
    print("\n[4] Final State...")
    print("-" * 70)

    status = organism.get_status()
    print(f"    Generation: {status.get('generation')}")
    print(f"    Best fitness: {status.get('best_fitness', 0):.2f}")
    print(f"    Emergence signals: {status.get('emergence_signals_count', 0)}")
    print(f"    Meta-learning enabled: {status.get('meta_learning_enabled')}")

    # Show memory state
    memory_snapshot = organism.memory.get_snapshot()
    print(f"\n    SharedKVMemory state:")
    print(f"      Total entries: {memory_snapshot.get('entry_count', 0)}")
    print(f"      By source: {memory_snapshot.get('entries_by_source', {})}")

    # Show meta-learning
    if organism.meta_learner:
        meta_summary = organism.meta_learner.get_summary()
        print(f"\n    Meta-Learning:")
        print(f"      Signals tracked: {meta_summary.get('total_signals', 0)}")
        print(f"      Strategies: {meta_summary.get('strategies_tracked', 0)}")
        best = meta_summary.get('best_strategy')
        if best:
            print(f"      Best strategy: {best.get('strategy', '?')[:40]}")

    # Generate visualization data
    print("\n[5] Visualization Data...")
    print("-" * 70)

    viz_data = organism.get_visualization_data()
    visualizer = OuroborosVisualizer(viz_data)

    # Print terminal dashboard
    print(visualizer.render_terminal_dashboard())

    print("\n[6] Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
