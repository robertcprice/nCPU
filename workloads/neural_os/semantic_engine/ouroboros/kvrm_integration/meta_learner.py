"""
MetaLearner - Learning What Works Across Generations
=====================================================
Implements panel recommendations:
- Track cross-generation knowledge transfer
- Identify successful strategies
- Detect patterns that lead to improvement

Writes to SharedKVMemory:
- meta:patterns - Detected patterns
- meta:best_strategies - What worked before
- meta:warnings - What to avoid
"""

import sys
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib
import json

sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')

from core.shared_memory import SharedKVMemory


@dataclass
class LearningSignal:
    """A signal that something worked or didn't work."""
    agent_id: str
    generation: int
    fitness_before: float
    fitness_after: float
    action_type: str  # "mutation", "collaboration", "exploration"
    action_details: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def improvement(self) -> float:
        return self.fitness_after - self.fitness_before


class MetaLearner:
    """
    Learns what works across generations and shares insights.

    This is the "meta" layer that remembers:
    - Which mutation strategies improved fitness
    - Which collaborations were fruitful
    - What patterns led to dead ends
    - Cross-agent knowledge transfer

    Panel Recommendation: "Track cross-generation knowledge transfer"
    """

    def __init__(self, memory: SharedKVMemory):
        self.memory = memory

        # Learning storage
        self.signals: List[LearningSignal] = []
        self.strategy_scores: Dict[str, float] = defaultdict(float)
        self.strategy_counts: Dict[str, int] = defaultdict(int)

        # Pattern detection
        self.improvement_patterns: List[Dict] = []
        self.failure_patterns: List[Dict] = []

        # Best solutions seen
        self.hall_of_fame: List[Dict] = []

    def record_signal(self, signal: LearningSignal) -> None:
        """Record a learning signal (something that worked or didn't)."""
        self.signals.append(signal)

        # Update strategy scores
        key = f"{signal.action_type}:{signal.action_details[:50]}"
        if signal.success:
            self.strategy_scores[key] += signal.improvement()
            self.strategy_counts[key] += 1
        else:
            self.strategy_scores[key] -= 0.1  # Penalty for failure

        # Detect patterns
        self._analyze_patterns()

        # Write to shared memory
        self._update_memory()

    def record_solution(
        self,
        agent_id: str,
        generation: int,
        solution: str,
        fitness: float
    ) -> None:
        """Record a solution for hall of fame consideration."""
        solution_hash = hashlib.md5(solution.encode()).hexdigest()[:8]

        # Add to hall of fame if good enough
        if len(self.hall_of_fame) < 10 or fitness > min(s["fitness"] for s in self.hall_of_fame):
            self.hall_of_fame.append({
                "agent_id": agent_id,
                "generation": generation,
                "fitness": fitness,
                "solution_hash": solution_hash,
                "solution_preview": solution[:200],
                "timestamp": datetime.now().isoformat(),
            })

            # Keep only top 10
            self.hall_of_fame.sort(key=lambda x: x["fitness"], reverse=True)
            self.hall_of_fame = self.hall_of_fame[:10]

            self._update_memory()

    def _analyze_patterns(self) -> None:
        """Analyze signals to find patterns."""
        if len(self.signals) < 5:
            return

        recent = self.signals[-20:]

        # Pattern 1: Successful strategies
        successes = [s for s in recent if s.success and s.improvement() > 0.1]
        if len(successes) >= 3:
            # Find common action types
            action_types = [s.action_type for s in successes]
            most_common = max(set(action_types), key=action_types.count)

            if action_types.count(most_common) >= 2:
                self.improvement_patterns.append({
                    "type": "successful_strategy",
                    "strategy": most_common,
                    "count": action_types.count(most_common),
                    "avg_improvement": sum(s.improvement() for s in successes) / len(successes),
                    "detected_at": datetime.now().isoformat(),
                })

        # Pattern 2: Failure patterns
        failures = [s for s in recent if not s.success]
        if len(failures) >= 5:
            failure_types = [s.action_type for s in failures]
            most_common_fail = max(set(failure_types), key=failure_types.count)

            self.failure_patterns.append({
                "type": "failure_pattern",
                "strategy": most_common_fail,
                "count": failure_types.count(most_common_fail),
                "detected_at": datetime.now().isoformat(),
            })

        # Pattern 3: Collaboration benefits
        collab_signals = [s for s in recent if s.action_type == "collaboration"]
        if collab_signals:
            avg_collab_improvement = sum(s.improvement() for s in collab_signals) / len(collab_signals)
            solo_signals = [s for s in recent if s.action_type != "collaboration"]
            if solo_signals:
                avg_solo_improvement = sum(s.improvement() for s in solo_signals) / len(solo_signals)

                if avg_collab_improvement > avg_solo_improvement * 1.5:
                    self.improvement_patterns.append({
                        "type": "collaboration_benefit",
                        "collab_avg": avg_collab_improvement,
                        "solo_avg": avg_solo_improvement,
                        "detected_at": datetime.now().isoformat(),
                    })

    def get_best_strategies(self, top_n: int = 5) -> List[Dict]:
        """Get the best performing strategies."""
        strategies = []
        for key, score in sorted(self.strategy_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            count = self.strategy_counts.get(key, 0)
            if count > 0:
                strategies.append({
                    "strategy": key,
                    "total_improvement": score,
                    "times_used": count,
                    "avg_improvement": score / count,
                })
        return strategies

    def get_warnings(self) -> List[str]:
        """Get warnings about what to avoid."""
        warnings = []

        # Warn about consistently failing strategies
        for key, score in self.strategy_scores.items():
            if score < -0.5 and self.strategy_counts[key] >= 3:
                warnings.append(f"Strategy '{key}' has negative impact")

        # Warn about recent failure patterns
        recent_failures = self.failure_patterns[-3:]
        for pattern in recent_failures:
            warnings.append(f"Repeated failures with {pattern['strategy']}")

        return warnings

    def _update_memory(self) -> None:
        """Write meta-learning data to shared memory."""
        # Best strategies
        best = self.get_best_strategies()
        self.memory.write(
            key="meta:best_strategies",
            value={
                "strategies": best,
                "updated_at": datetime.now().isoformat(),
            },
            source="meta_learner",
            ttl=300,
            priority=8,
        )

        # Warnings
        warnings = self.get_warnings()
        if warnings:
            self.memory.write(
                key="meta:warnings",
                value={
                    "warnings": warnings,
                    "updated_at": datetime.now().isoformat(),
                },
                source="meta_learner",
                ttl=300,
                priority=9,
            )

        # Hall of fame
        if self.hall_of_fame:
            self.memory.write(
                key="meta:hall_of_fame",
                value={
                    "solutions": self.hall_of_fame[:5],  # Top 5
                    "updated_at": datetime.now().isoformat(),
                },
                source="meta_learner",
                ttl=600,
                priority=7,
            )

        # Patterns
        patterns = {
            "improvements": self.improvement_patterns[-5:],
            "failures": self.failure_patterns[-5:],
            "total_signals": len(self.signals),
        }
        self.memory.write(
            key="meta:patterns",
            value=patterns,
            source="meta_learner",
            ttl=300,
            priority=6,
        )

    def get_summary(self) -> Dict:
        """Get meta-learning summary."""
        best_strategies = self.get_best_strategies(1)
        return {
            "total_signals": len(self.signals),
            "strategies_tracked": len(self.strategy_scores),
            "improvement_patterns": len(self.improvement_patterns),
            "failure_patterns": len(self.failure_patterns),
            "hall_of_fame_size": len(self.hall_of_fame),
            "best_strategy": best_strategies[0] if best_strategies else None,
            "warnings": self.get_warnings(),
        }
