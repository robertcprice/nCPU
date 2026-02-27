#!/usr/bin/env python3
"""
OUROBOROS Dashboard Experiment Runner
======================================
Run experiments with full dashboard visualization.

Features:
- Live WebSocket dashboard (--live)
- Terminal ASCII dashboard (default)
- HTML report generation (--html)
- Solution caching for performance
- Failure pattern learning
"""

import sys
import os
import time
import argparse
import threading

# Add paths
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kvrm_integration import (
    OuroborosOrganism,
    OuroborosConfig,
    TrustLevel,
)
from kvrm_integration.solution_cache import get_solution_cache
from kvrm_integration.failure_taxonomy import get_failure_taxonomy
from dashboard import TerminalDashboard, HTMLDashboard

# Optional live server
try:
    from dashboard import LiveDashboardServer
    HAS_LIVE_SERVER = True
except ImportError:
    HAS_LIVE_SERVER = False


def create_problem(problem_type: str = "default") -> dict:
    """Create problem based on type."""
    problems = {
        "default": {
            "description": "Write a Python function called 'find_peaks' that finds all local peaks in a list.",
            "test_cases": [
                {"input": [1, 3, 2, 4, 1], "expected": [3, 4]},
                {"input": [1, 2, 3], "expected": [3]},
            ],
        },
        "prime": {
            "description": "Write a function 'is_prime(n)' that returns True if n is prime.",
            "test_cases": [
                {"input": 2, "expected": True},
                {"input": 4, "expected": False},
                {"input": 17, "expected": True},
            ],
        },
        "fibonacci": {
            "description": "Write a function 'fib(n)' that returns the nth Fibonacci number.",
            "test_cases": [
                {"input": 0, "expected": 0},
                {"input": 1, "expected": 1},
                {"input": 10, "expected": 55},
            ],
        },
    }
    return problems.get(problem_type, problems["default"])


def run_experiment(args):
    """Run the full experiment with dashboard."""
    print("=" * 70)
    print("       OUROBOROS DASHBOARD EXPERIMENT")
    print("=" * 70)

    # Configuration
    config = OuroborosConfig(
        num_competitive_agents=args.competitive,
        num_cooperative_agents=args.cooperative,
        narrator_trust_level=TrustLevel.GUIDE,
        llm_model=args.model,
        max_generations=args.generations,
        enable_meta_learning=True,
        enable_emergence_detection=True,
    )

    print(f"\nConfiguration:")
    print(f"  Generations: {args.generations}")
    print(f"  Competitive: {args.competitive}")
    print(f"  Cooperative: {args.cooperative}")
    print(f"  Model: {args.model}")
    print(f"  Dashboard: {'Live' if args.live else 'Terminal'}")

    # Initialize components
    cache = get_solution_cache()
    taxonomy = get_failure_taxonomy()
    terminal_dash = TerminalDashboard()
    html_dash = HTMLDashboard()

    # Live server (optional)
    live_server = None
    if args.live and HAS_LIVE_SERVER:
        print(f"\nStarting live dashboard at http://localhost:{args.port}")
        live_server = LiveDashboardServer()
        live_server.start_background(port=args.port)
        print("  Open browser to see real-time updates")
    elif args.live and not HAS_LIVE_SERVER:
        print("\n  Warning: FastAPI not installed. Using terminal dashboard.")
        print("  Install with: pip install fastapi uvicorn")

    # Create organism
    print("\nCreating OUROBOROS organism...")
    organism = OuroborosOrganism(config)

    # Set problem
    problem = create_problem(args.problem)
    organism.set_problem(problem)
    print(f"Problem: {problem['description'][:60]}...")

    # Emergence callback
    def on_emergence(signal):
        msg = f"[EMERGENCE] {signal.signal_type}: {signal.description[:40]}"
        if live_server:
            live_server.push_event("emergence", {
                "type": signal.signal_type,
                "strength": signal.strength,
                "description": signal.description,
            }, signal.generation)
        print(f"  {msg}")

    organism.on_emergence(on_emergence)

    # Run evolution
    print("\n" + "-" * 70)
    print("Starting evolution...")
    print("-" * 70)

    start_time = time.time()

    for gen in range(args.generations):
        gen_start = time.time()

        # Run generation
        result = organism.run_generation()
        gen_duration = time.time() - gen_start

        # Get visualization data
        viz_data = organism.get_visualization_data()

        # Update dashboards
        if args.live and live_server:
            live_server.push_update(viz_data)

        if not args.quiet:
            # Clear and show terminal dashboard
            if not args.live:
                os.system('clear' if os.name != 'nt' else 'cls')
                print(terminal_dash.render(viz_data))
            else:
                # Just show generation summary
                best = result.get('best_fitness', 0)
                agent = result.get('best_agent', 'none')
                print(f"  Gen {gen + 1}/{args.generations}: best={best:.2f} ({agent}) [{gen_duration:.1f}s]")

        # Check for failures and learn
        solutions = result.get('solutions', [])
        for sol in solutions:
            if sol.get('fitness', 0) < 0.3:
                code = sol.get('code', '')
                failures = taxonomy.detect_failure(
                    code,
                    agent_id=sol.get('agent', 'unknown'),
                    generation=gen + 1,
                )
                if failures:
                    for f in failures:
                        print(f"  [FAILURE] {sol.get('agent')}: {f.description}")

        # Check for override requests
        overrides = result.get('override_requests', [])
        if overrides:
            print("\n" + "!" * 60)
            print("  OVERRIDE REQUEST - HUMAN APPROVAL REQUIRED")
            for ovr in overrides:
                print(f"  ID: {ovr.get('request_id')}")
                print(f"  Reason: {ovr.get('reason')}")
            print("!" * 60)

    # Final results
    elapsed = time.time() - start_time
    final_viz = organism.get_visualization_data()

    print("\n" + "=" * 70)
    print("       EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Avg per generation: {elapsed / args.generations:.1f}s")

    status = organism.get_status()
    print(f"  Final best fitness: {status.get('best_fitness', 0):.3f}")
    print(f"  Emergence signals: {status.get('emergence_signals_count', 0)}")

    # Cache stats
    cache_stats = cache.get_stats()
    print(f"\n  Cache hit rate: {cache_stats.get('hit_rate', 0)*100:.1f}%")
    print(f"  Cache size: {cache_stats.get('cache_size', 0)}")

    # Failure stats
    failure_summary = taxonomy.get_summary()
    print(f"  Total failures detected: {failure_summary.get('total_failures', 0)}")
    hall_of_shame = taxonomy.get_hall_of_shame(3)
    if hall_of_shame:
        print(f"  Top failure patterns:")
        for p in hall_of_shame:
            print(f"    - {p['description']} ({p['frequency']}x)")

    # Save HTML report
    if args.html:
        html_path = args.html if args.html != True else "/tmp/ouroboros_report.html"
        html_dash.save(final_viz, html_path, f"OUROBOROS Report - {args.generations} Generations")
        print(f"\n  HTML report saved: {html_path}")

    # Keep live server running
    if args.live and live_server:
        print(f"\n  Dashboard still running at http://localhost:{args.port}")
        print("  Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")

    return final_viz


def main():
    parser = argparse.ArgumentParser(
        description="Run OUROBOROS with dashboard visualization"
    )
    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=5,
        help="Number of generations (default: 5)"
    )
    parser.add_argument(
        "--competitive", "-c",
        type=int,
        default=2,
        help="Number of competitive agents (default: 2)"
    )
    parser.add_argument(
        "--cooperative", "-o",
        type=int,
        default=1,
        help="Number of cooperative agents (default: 1)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3:8b",
        help="Ollama model (default: qwen3:8b)"
    )
    parser.add_argument(
        "--problem", "-p",
        type=str,
        default="default",
        choices=["default", "prime", "fibonacci"],
        help="Problem type (default: default)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live WebSocket dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Live dashboard port (default: 8080)"
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="/tmp/ouroboros_report.html",
        help="Save HTML report (default: /tmp/ouroboros_report.html)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
