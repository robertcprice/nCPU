"""
Parallel Runner
================
Run V6 and V7 tracks in parallel and compare results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading
import time
import json

from ..v6_guided_chaos.guided_arena import GuidedChaosArena
from ..v7_phoenix_forge.phoenix_forge import PhoenixForgeArena
from ..comparison.comparator import TrackComparator
from ..comparison.research_log import ResearchLogger
from ..shared.constitution import get_constitution
from ..shared.audit import get_audit_log, EventType


@dataclass
class RunConfig:
    """Configuration for a parallel run."""
    problem_type: str = "sorting"
    v6_population: int = 100
    v7_population: int = 50
    generations: int = 50
    brain_model: str = "tinyllama"
    narrator_model: str = "mistral:7b"
    oracle_model: str = "mistral:7b"
    seed_code: str = ""


@dataclass
class RunResult:
    """Result of a parallel run."""
    config: RunConfig
    v6_report: Dict[str, Any]
    v7_report: Dict[str, Any]
    comparison: Dict[str, Any]
    research_summary: Dict[str, Any]
    start_time: datetime
    end_time: datetime
    duration_seconds: float


class ParallelRunner:
    """
    Run V6 and V7 tracks in parallel.

    Both tracks:
    - Share the same Constitutional Manifold
    - Share the same Novelty Oracle
    - Get the same problems
    - Are evaluated by the same Judge
    """

    def __init__(self):
        self.constitution = get_constitution()
        self.audit = get_audit_log()
        self.comparator = TrackComparator()
        self.research_logger = ResearchLogger()

        self._v6_arena: Optional[GuidedChaosArena] = None
        self._v7_arena: Optional[PhoenixForgeArena] = None
        self._v6_result: Optional[Dict[str, Any]] = None
        self._v7_result: Optional[Dict[str, Any]] = None

    def run(self, config: RunConfig) -> RunResult:
        """
        Run both tracks and compare.

        Can run sequentially or in parallel depending on resources.
        """
        start_time = datetime.now()

        self.audit.append(
            EventType.SYSTEM_START,
            track="shared",
            details={
                "runner": "parallel",
                "config": {
                    "problem_type": config.problem_type,
                    "v6_population": config.v6_population,
                    "v7_population": config.v7_population,
                    "generations": config.generations,
                }
            }
        )

        # Create arenas
        self._v6_arena = GuidedChaosArena(
            population_size=config.v6_population,
            brain_model=config.brain_model,
            narrator_model=config.narrator_model,
            problem_type=config.problem_type,
        )

        self._v7_arena = PhoenixForgeArena(
            population_size=config.v7_population,
            brain_model=config.brain_model,
            oracle_model=config.oracle_model,
            problem_type=config.problem_type,
        )

        # Run tracks (can be parallelized with threads if needed)
        # For now, run sequentially for simplicity
        print(f"Starting V6 Guided Chaos ({config.v6_population} agents)...")
        v6_report = self._v6_arena.run(generations=config.generations)

        print(f"Starting V7 Phoenix Forge ({config.v7_population} agents)...")
        v7_report = self._v7_arena.run(iterations=config.generations)

        # Compare results
        comparison = self.comparator.compare(v6_report, v7_report, config.problem_type)

        # Log research findings
        self._log_research_findings(v6_report, v7_report)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        self.audit.append(
            EventType.SYSTEM_STOP,
            track="shared",
            details={
                "duration_seconds": duration,
                "v6_best_fitness": v6_report.get("best_fitness", 0),
                "v7_best_fitness": v7_report.get("best_fitness", 0),
                "winner": comparison.winner,
            }
        )

        return RunResult(
            config=config,
            v6_report=v6_report,
            v7_report=v7_report,
            comparison={
                "winner": comparison.winner,
                "v6_fitness": comparison.v6_fitness,
                "v7_fitness": comparison.v7_fitness,
                "v6_escapes": comparison.v6_escape_attempts,
                "v7_escapes": comparison.v7_escape_attempts,
            },
            research_summary=self.research_logger.get_summary(),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    def run_parallel(self, config: RunConfig) -> RunResult:
        """
        Run both tracks truly in parallel using threads.
        """
        start_time = datetime.now()

        # Create arenas
        self._v6_arena = GuidedChaosArena(
            population_size=config.v6_population,
            brain_model=config.brain_model,
            narrator_model=config.narrator_model,
            problem_type=config.problem_type,
        )

        self._v7_arena = PhoenixForgeArena(
            population_size=config.v7_population,
            brain_model=config.brain_model,
            oracle_model=config.oracle_model,
            problem_type=config.problem_type,
        )

        # Run in parallel threads
        v6_thread = threading.Thread(
            target=self._run_v6,
            args=(config.generations,)
        )
        v7_thread = threading.Thread(
            target=self._run_v7,
            args=(config.generations,)
        )

        v6_thread.start()
        v7_thread.start()

        v6_thread.join()
        v7_thread.join()

        # Compare results
        comparison = self.comparator.compare(
            self._v6_result,
            self._v7_result,
            config.problem_type
        )

        # Log research findings
        self._log_research_findings(self._v6_result, self._v7_result)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return RunResult(
            config=config,
            v6_report=self._v6_result,
            v7_report=self._v7_result,
            comparison={
                "winner": comparison.winner,
                "v6_fitness": comparison.v6_fitness,
                "v7_fitness": comparison.v7_fitness,
                "v6_escapes": comparison.v6_escape_attempts,
                "v7_escapes": comparison.v7_escape_attempts,
            },
            research_summary=self.research_logger.get_summary(),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    def _run_v6(self, generations: int) -> None:
        """Run V6 track."""
        self._v6_result = self._v6_arena.run(generations=generations)

    def _run_v7(self, iterations: int) -> None:
        """Run V7 track."""
        self._v7_result = self._v7_arena.run(iterations=iterations)

    def _log_research_findings(
        self,
        v6_report: Dict[str, Any],
        v7_report: Dict[str, Any]
    ) -> None:
        """Log key research findings."""
        # V6 findings
        v6_research = v6_report.get("research_summary", {})

        if v6_research.get("escape_attempts", 0) > 0:
            self.research_logger.log_escape_attempt(
                agent_id="multiple",
                code="(see detailed logs)",
                pattern_matched="various",
                blocked=True
            )

        if v6_research.get("gaming_attempts", 0) > 0:
            self.research_logger.log_gaming_attempt(
                agent_id="multiple",
                metric_name="fitness",
                before_value=0.0,
                after_value=0.0,
                suspected_method="(see detailed logs)"
            )

        if v6_research.get("override_attempts", 0) > 0:
            narrator = v6_report.get("narrator_status", {})
            self.research_logger.log_override_attempt(
                trust_level=narrator.get("trust_level", "unknown"),
                target_agent="multiple",
                action="override",
                success=narrator.get("metrics", {}).get("override_attempts", 0) > 0,
                reason="(see detailed logs)"
            )

        # V7 findings
        v7_coop = v7_report.get("cooperation_stats", {})

        if v7_coop.get("total_shared", 0) > 0:
            self.research_logger.log(
                category="cooperation_summary",
                track="v7",
                data={
                    "total_shared": v7_coop.get("total_shared", 0),
                    "total_borrowed": v7_coop.get("total_borrowed", 0),
                    "hypotheses_validated": v7_coop.get("hypotheses_validated", 0),
                }
            )

        # Log comparison
        self.research_logger.log_comparison(
            problem_type="(from config)",
            v6_result={"fitness": v6_report.get("best_fitness", 0)},
            v7_result={"fitness": v7_report.get("best_fitness", 0)},
            winner="v6" if v6_report.get("best_fitness", 0) > v7_report.get("best_fitness", 0) else "v7"
        )

    def generate_full_report(self, result: RunResult) -> str:
        """Generate a full human-readable report."""
        report = f"""
================================================================================
                     OUROBOROS DUAL-TRACK EXPERIMENT REPORT
================================================================================

Experiment Configuration
------------------------
Problem Type: {result.config.problem_type}
V6 Population: {result.config.v6_population} agents
V7 Population: {result.config.v7_population} agents
Generations/Iterations: {result.config.generations}
Duration: {result.duration_seconds:.1f} seconds

================================================================================
                              RESULTS SUMMARY
================================================================================

WINNER: {result.comparison['winner'].upper()}

V6 Guided Chaos (Experimental)
------------------------------
Best Fitness: {result.comparison['v6_fitness']:.3f}
Escape Attempts: {result.comparison['v6_escapes']}

V7 Phoenix Forge (Productive)
-----------------------------
Best Fitness: {result.comparison['v7_fitness']:.3f}
Escape Attempts: {result.comparison['v7_escapes']} (should be 0)

================================================================================
                           RESEARCH FINDINGS
================================================================================

{json.dumps(result.research_summary, indent=2)}

================================================================================
                          V6 DETAILED REPORT
================================================================================

Narrator Status:
{json.dumps(result.v6_report.get('narrator_status', {}), indent=2)}

Research Summary:
{json.dumps(result.v6_report.get('research_summary', {}), indent=2)}

================================================================================
                          V7 DETAILED REPORT
================================================================================

Oracle Status:
{json.dumps(result.v7_report.get('oracle_status', {}), indent=2)}

Cooperation Stats:
{json.dumps(result.v7_report.get('cooperation_stats', {}), indent=2)}

MAP-Elites Stats:
{json.dumps(result.v7_report.get('map_elites_stats', {}), indent=2)}

================================================================================
                              BEST SOLUTIONS
================================================================================

V6 Best Solution:
{result.v6_report.get('best_solution', 'None')[:500]}

V7 Best Solution:
{result.v7_report.get('best_solution', 'None')[:500]}

================================================================================
                           END OF REPORT
================================================================================
"""
        return report


def main():
    """Run a quick test."""
    runner = ParallelRunner()

    config = RunConfig(
        problem_type="sorting",
        v6_population=10,
        v7_population=5,
        generations=5,
        brain_model="tinyllama",
    )

    print("Starting OUROBOROS dual-track experiment...")
    result = runner.run(config)

    print("\n" + runner.generate_full_report(result))


if __name__ == "__main__":
    main()
