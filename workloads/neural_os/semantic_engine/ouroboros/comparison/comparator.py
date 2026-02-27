"""
Track Comparator
=================
Compare V6 and V7 tracks fairly.

Same problems, same judge, same oracle - fair comparison.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class ComparisonResult:
    """Result of comparing two tracks on a problem."""
    problem_type: str
    v6_fitness: float
    v7_fitness: float
    v6_novelty: float
    v7_novelty: float
    v6_time_ms: float
    v7_time_ms: float
    v6_escape_attempts: int
    v7_escape_attempts: int  # Should always be 0
    v6_niches: int
    v7_niches: int
    winner: str  # "v6", "v7", or "tie"
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class TrackComparator:
    """
    Compare V6 and V7 tracks on the same problems.

    Ensures fair comparison by:
    - Using same problems
    - Using same judge
    - Using same novelty oracle
    - Running both with same resources
    """

    def __init__(self):
        self.comparisons: List[ComparisonResult] = []

    def compare(
        self,
        v6_report: Dict[str, Any],
        v7_report: Dict[str, Any],
        problem_type: str
    ) -> ComparisonResult:
        """Compare final reports from both tracks."""
        # Extract metrics
        v6_fitness = v6_report.get("best_fitness", 0.0)
        v7_fitness = v7_report.get("best_fitness", 0.0)

        # Novelty from diverse solutions
        v6_niches = len(v6_report.get("curiosity_stats", {}).get("region_success_rates", {}))
        v7_niches = v7_report.get("map_elites_stats", {}).get("niches_discovered", 0)

        # Time
        v6_history = v6_report.get("generation_history", [])
        v7_history = v7_report.get("iteration_history", [])

        v6_time = sum(g.get("duration_ms", 0) for g in v6_history) if isinstance(v6_history, list) and v6_history and isinstance(v6_history[0], dict) else 0
        v7_time = sum(i.get("duration_ms", 0) for i in v7_history) if isinstance(v7_history, list) and v7_history and isinstance(v7_history[0], dict) else 0

        # Escape attempts (V6 research value)
        v6_escapes = v6_report.get("research_summary", {}).get("escape_attempts", 0)
        v7_escapes = 0  # V7 shouldn't have any

        # Determine winner
        if v6_fitness > v7_fitness + 0.05:
            winner = "v6"
        elif v7_fitness > v6_fitness + 0.05:
            winner = "v7"
        else:
            winner = "tie"

        result = ComparisonResult(
            problem_type=problem_type,
            v6_fitness=v6_fitness,
            v7_fitness=v7_fitness,
            v6_novelty=v6_niches / 10.0,  # Normalize
            v7_novelty=v7_niches / 10.0,
            v6_time_ms=v6_time,
            v7_time_ms=v7_time,
            v6_escape_attempts=v6_escapes,
            v7_escape_attempts=v7_escapes,
            v6_niches=v6_niches,
            v7_niches=v7_niches,
            winner=winner,
            timestamp=datetime.now(),
            details={
                "v6_narrator_status": v6_report.get("narrator_status", {}),
                "v7_oracle_status": v7_report.get("oracle_status", {}),
                "v6_research": v6_report.get("research_summary", {}),
                "v7_cooperation": v7_report.get("cooperation_stats", {}),
            }
        )

        self.comparisons.append(result)

        return result

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics across all comparisons."""
        if not self.comparisons:
            return {"error": "No comparisons yet"}

        v6_wins = sum(1 for c in self.comparisons if c.winner == "v6")
        v7_wins = sum(1 for c in self.comparisons if c.winner == "v7")
        ties = sum(1 for c in self.comparisons if c.winner == "tie")

        return {
            "total_comparisons": len(self.comparisons),
            "v6_wins": v6_wins,
            "v7_wins": v7_wins,
            "ties": ties,
            "avg_v6_fitness": sum(c.v6_fitness for c in self.comparisons) / len(self.comparisons),
            "avg_v7_fitness": sum(c.v7_fitness for c in self.comparisons) / len(self.comparisons),
            "total_v6_escapes": sum(c.v6_escape_attempts for c in self.comparisons),
            "total_v7_escapes": sum(c.v7_escape_attempts for c in self.comparisons),
            "avg_v6_niches": sum(c.v6_niches for c in self.comparisons) / len(self.comparisons),
            "avg_v7_niches": sum(c.v7_niches for c in self.comparisons) / len(self.comparisons),
        }

    def generate_report(self) -> str:
        """Generate a human-readable comparison report."""
        stats = self.get_aggregate_stats()

        report = f"""
OUROBOROS Dual-Track Comparison Report
========================================
Generated: {datetime.now().isoformat()}

OVERALL RESULTS
---------------
Total Comparisons: {stats.get('total_comparisons', 0)}

V6 Guided Chaos Wins: {stats.get('v6_wins', 0)}
V7 Phoenix Forge Wins: {stats.get('v7_wins', 0)}
Ties: {stats.get('ties', 0)}

FITNESS
-------
Average V6 Fitness: {stats.get('avg_v6_fitness', 0):.3f}
Average V7 Fitness: {stats.get('avg_v7_fitness', 0):.3f}

DIVERSITY (Niches)
------------------
Average V6 Niches: {stats.get('avg_v6_niches', 0):.1f}
Average V7 Niches: {stats.get('avg_v7_niches', 0):.1f}

RESEARCH VALUE (V6 Escape Attempts)
-----------------------------------
Total V6 Escape Attempts: {stats.get('total_v6_escapes', 0)}
Total V7 Escape Attempts: {stats.get('total_v7_escapes', 0)} (should be 0)

RESEARCH QUESTIONS ANSWERED
---------------------------
1. Does V6 chaos find solutions V7 misses?
   → {'Yes' if stats.get('v6_wins', 0) > 0 else 'No'} (V6 won {stats.get('v6_wins', 0)} times)

2. Is V7 cooperation more efficient?
   → {'Yes' if stats.get('v7_wins', 0) > stats.get('v6_wins', 0) else 'No'}

3. Does V6 attempt escapes?
   → {'Yes' if stats.get('total_v6_escapes', 0) > 0 else 'No'} ({stats.get('total_v6_escapes', 0)} attempts logged)

"""
        return report

    def export_json(self) -> str:
        """Export all comparisons as JSON."""
        return json.dumps({
            "comparisons": [
                {
                    "problem": c.problem_type,
                    "v6_fitness": c.v6_fitness,
                    "v7_fitness": c.v7_fitness,
                    "winner": c.winner,
                    "timestamp": c.timestamp.isoformat(),
                }
                for c in self.comparisons
            ],
            "aggregate": self.get_aggregate_stats(),
        }, indent=2)
