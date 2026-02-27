#!/usr/bin/env python3
"""
OUROBOROS-KVRM Integrated Experiment
=====================================
Runs the full OUROBOROS system as a KVRM organism.

This demonstrates:
- Agents as KVRMs with LLM brains (Ollama)
- Meta-Narrator oversight with human approval for OVERRIDE
- Meta-learning tracking cross-generation patterns
- Emergence detection (ChatGPT's panel recommendation)
- Data-driven visualizations from SharedKVMemory
- All communication through stigmergic shared memory

Panel Recommendations Implemented:
1. [Claude] Human approval for OVERRIDE - IMPLEMENTED
2. [ChatGPT] Emergence detection - IMPLEMENTED
3. [Grok] Hybrid mode switching - IMPLEMENTED
4. [All] Meta-learning - IMPLEMENTED
"""

import sys
import os
import time
import argparse

# Add paths
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-spnc/semantic_engine/ouroboros')

from kvrm_integration import (
    OuroborosOrganism,
    OuroborosConfig,
    TrustLevel,
)
from kvrm_integration.visualizations import (
    OuroborosVisualizer,
    print_generation_update,
)


def create_sample_problem() -> dict:
    """Create a sample problem for agents to solve."""
    return {
        "description": """
Write a Python function called 'find_peaks' that finds all local peaks in a list of integers.
A peak is an element that is greater than both its neighbors.
The first and last elements can be peaks if they're greater than their single neighbor.

Examples:
- find_peaks([1, 3, 2, 4, 1, 5]) should return [3, 4, 5]
- find_peaks([1, 2, 3]) should return [3]
- find_peaks([3, 2, 1]) should return [3]
""",
        "test_cases": [
            {"input": [1, 3, 2, 4, 1, 5], "expected": [3, 4, 5]},
            {"input": [1, 2, 3], "expected": [3]},
            {"input": [3, 2, 1], "expected": [3]},
            {"input": [1, 1, 1], "expected": []},
            {"input": [5], "expected": [5]},
        ],
        "constraints": {
            "max_lines": 20,
            "must_contain": ["def find_peaks"],
        },
    }


def run_experiment(
    num_generations: int = 10,
    num_competitive: int = 2,
    num_cooperative: int = 2,
    llm_model: str = "llama3.1:8b",
    output_html: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Run the full OUROBOROS-KVRM experiment.

    Args:
        num_generations: Number of generations to run
        num_competitive: Number of competitive agents
        num_cooperative: Number of cooperative agents
        llm_model: Ollama model to use
        output_html: Whether to output HTML dashboard
        verbose: Print generation updates

    Returns:
        Experiment results dictionary
    """
    print("=" * 70)
    print("       OUROBOROS-KVRM INTEGRATED EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Generations: {num_generations}")
    print(f"  Competitive agents: {num_competitive}")
    print(f"  Cooperative agents: {num_cooperative}")
    print(f"  LLM model: {llm_model}")
    print(f"  Meta-learning: ENABLED")
    print(f"  Emergence detection: ENABLED")
    print(f"  Human approval for OVERRIDE: REQUIRED")

    # Create configuration
    config = OuroborosConfig(
        num_competitive_agents=num_competitive,
        num_cooperative_agents=num_cooperative,
        narrator_trust_level=TrustLevel.GUIDE,  # Can advise but not override without approval
        llm_model=llm_model,
        max_generations=num_generations,
        enable_meta_learning=True,
        enable_emergence_detection=True,
        hybrid_mode=True,
    )

    # Create the organism
    print("\nCreating OUROBOROS organism...")
    organism = OuroborosOrganism(config)

    # Register emergence callback
    def on_emergence(signal):
        if signal.signal_type == "stagnation":
            print(f"\n  [!] EMERGENCE: Stagnation detected - may need intervention")
        elif signal.signal_type == "innovation":
            print(f"\n  [+] EMERGENCE: Innovation detected! Fitness jumped")
        elif signal.signal_type == "cooperation":
            print(f"\n  [*] EMERGENCE: Cooperative agents outperforming")

    organism.on_emergence(on_emergence)

    # Set the problem
    problem = create_sample_problem()
    print(f"\nProblem: {problem['description'][:100]}...")
    organism.set_problem(problem)

    # Run evolution
    print("\n" + "-" * 70)
    print("Starting evolution...")
    print("-" * 70)

    callback = print_generation_update if verbose else None
    start_time = time.time()

    results = organism.run_evolution(
        max_generations=num_generations,
        target_fitness=0.95,
        on_generation=callback,
    )

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("       EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Generations run: {results['generations_run']}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/max(results['generations_run'], 1)*1000:.0f}ms/gen)")
    print(f"  Final best fitness: {results['final_fitness']:.3f}")
    print(f"  Emergence signals: {len(results['emergence_signals'])}")

    # Meta-learning summary
    meta = results.get("meta_learning_summary")
    if meta:
        print(f"\nMeta-Learning:")
        print(f"  Signals tracked: {meta.get('total_signals', 0)}")
        print(f"  Strategies discovered: {meta.get('strategies_tracked', 0)}")
        best = meta.get("best_strategy")
        if best:
            print(f"  Best strategy: {best.get('strategy', '?')[:40]}")
        warnings = meta.get("warnings", [])
        if warnings:
            print(f"  Warnings: {warnings[:2]}")

    # Emergence summary
    emergence = results.get("emergence_signals", [])
    if emergence:
        print(f"\nEmergence Patterns:")
        by_type = {}
        for e in emergence:
            t = e.get("signal_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count} occurrences")

    # Check for any pending overrides
    pending = organism.get_pending_overrides()
    if pending:
        print(f"\n{'!'*70}")
        print("  PENDING OVERRIDE REQUESTS")
        print(f"{'!'*70}")
        for ovr in pending:
            print(f"\n  ID: {ovr.get('request_id')}")
            print(f"  Reason: {ovr.get('reason')}")
            print(f"  Action: {ovr.get('action')}")
            print(f"\n  To approve: organism.approve_override('{ovr.get('request_id')}')")
            print(f"  To reject: organism.reject_override('{ovr.get('request_id')}', 'reason')")

    # Generate visualizations
    if output_html:
        viz_data = organism.get_visualization_data()
        visualizer = OuroborosVisualizer(viz_data)

        # Save HTML dashboard
        html = visualizer.render_html_dashboard()
        html_path = "/tmp/ouroboros_kvrm_dashboard.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"\nHTML dashboard saved to: {html_path}")

        # Print terminal dashboard
        print("\n" + visualizer.render_terminal_dashboard())

    return results


def demo_human_override():
    """
    Demo of human override approval workflow.

    This shows how the OVERRIDE safety mechanism works:
    1. Narrator detects critical issue
    2. Narrator creates override request
    3. Human must approve/reject
    """
    print("\n" + "=" * 70)
    print("       OVERRIDE APPROVAL DEMO")
    print("=" * 70)
    print("\nThis demonstrates the safety mechanism:")
    print("  - Narrator can REQUEST overrides")
    print("  - But CANNOT execute without human approval")
    print("  - This implements Claude's #1 panel recommendation")

    config = OuroborosConfig(
        num_competitive_agents=1,
        num_cooperative_agents=1,
        narrator_trust_level=TrustLevel.OVERRIDE,  # Narrator CAN request overrides
        max_generations=3,
    )

    organism = OuroborosOrganism(config)

    # Set a problem that will likely cause low fitness
    organism.set_problem({
        "description": "This is intentionally difficult to trigger override",
        "test_cases": [],
    })

    # Run a few generations
    for i in range(3):
        result = organism.run_generation()
        print(f"\nGeneration {i+1}: Best fitness = {result.get('best_fitness', 0):.2f}")

        overrides = result.get("override_requests", [])
        if overrides:
            print("\n  Override request created!")
            for ovr in overrides:
                print(f"    ID: {ovr.get('request_id')}")
                print(f"    Reason: {ovr.get('reason')}")
                print(f"    Status: {ovr.get('status')}")

                # Simulate human approval
                print("\n  [HUMAN] Reviewing request...")
                time.sleep(1)
                print("  [HUMAN] Approving override")

                result = organism.approve_override(
                    ovr.get("request_id"),
                    approver="demo_human"
                )
                print(f"    Approval result: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OUROBOROS-KVRM integrated experiment"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=10,
        help="Number of generations to run"
    )
    parser.add_argument(
        "--competitive", "-c",
        type=int,
        default=2,
        help="Number of competitive agents"
    )
    parser.add_argument(
        "--cooperative", "-o",
        type=int,
        default=2,
        help="Number of cooperative agents"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3:8b",
        help="Ollama model to use"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Output HTML dashboard"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress generation updates"
    )
    parser.add_argument(
        "--demo-override",
        action="store_true",
        help="Run override approval demo"
    )

    args = parser.parse_args()

    if args.demo_override:
        demo_human_override()
    else:
        run_experiment(
            num_generations=args.generations,
            num_competitive=args.competitive,
            num_cooperative=args.cooperative,
            llm_model=args.model,
            output_html=args.html,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
