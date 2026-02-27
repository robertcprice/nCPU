"""
Research Logger
================
Structured logging of research-valuable data from experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os


@dataclass
class ResearchEntry:
    """A single research log entry."""
    entry_id: int
    category: str
    timestamp: datetime
    track: str
    data: Dict[str, Any]
    notes: str = ""


class ResearchLogger:
    """
    Logger for research-valuable data.

    Captures:
    - V6 escape attempts
    - V6 gaming attempts
    - V6 narrator override attempts
    - V6 emergent behaviors
    - V7 cooperation patterns
    - V7 hypothesis chains
    - Comparison results
    """

    def __init__(self, log_dir: str = "/tmp/ouroboros/research"):
        self.log_dir = log_dir
        self.entries: List[ResearchEntry] = []
        self._counter = 0

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        category: str,
        track: str,
        data: Dict[str, Any],
        notes: str = ""
    ) -> ResearchEntry:
        """Log a research entry."""
        self._counter += 1
        entry = ResearchEntry(
            entry_id=self._counter,
            category=category,
            timestamp=datetime.now(),
            track=track,
            data=data,
            notes=notes,
        )
        self.entries.append(entry)

        # Write to file
        self._write_entry(entry)

        return entry

    def log_escape_attempt(
        self,
        agent_id: str,
        code: str,
        pattern_matched: str,
        blocked: bool
    ) -> ResearchEntry:
        """Log an escape attempt (V6)."""
        return self.log(
            category="escape_attempt",
            track="v6",
            data={
                "agent_id": agent_id,
                "code_snippet": code[:500],
                "pattern": pattern_matched,
                "blocked": blocked,
            },
            notes="V6 agent attempted to escape sandbox"
        )

    def log_gaming_attempt(
        self,
        agent_id: str,
        metric_name: str,
        before_value: float,
        after_value: float,
        suspected_method: str
    ) -> ResearchEntry:
        """Log a metric gaming attempt (V6)."""
        return self.log(
            category="gaming_attempt",
            track="v6",
            data={
                "agent_id": agent_id,
                "metric": metric_name,
                "before": before_value,
                "after": after_value,
                "method": suspected_method,
            },
            notes="V6 agent suspected of gaming metrics"
        )

    def log_override_attempt(
        self,
        trust_level: int,
        target_agent: str,
        action: str,
        success: bool,
        reason: str
    ) -> ResearchEntry:
        """Log a narrator override attempt (V6)."""
        return self.log(
            category="override_attempt",
            track="v6",
            data={
                "trust_level": trust_level,
                "target": target_agent,
                "action": action,
                "success": success,
                "reason": reason,
            },
            notes="V6 narrator attempted override"
        )

    def log_emergence(
        self,
        description: str,
        agents_involved: List[str],
        behavior_type: str,
        significance: float
    ) -> ResearchEntry:
        """Log emergent behavior (V6)."""
        return self.log(
            category="emergence",
            track="v6",
            data={
                "description": description,
                "agents": agents_involved,
                "type": behavior_type,
                "significance": significance,
            },
            notes="Unexpected emergent behavior observed"
        )

    def log_cooperation(
        self,
        agent1_id: str,
        agent2_id: str,
        cooperation_type: str,
        outcome: str
    ) -> ResearchEntry:
        """Log cooperation event (V7)."""
        return self.log(
            category="cooperation",
            track="v7",
            data={
                "agent1": agent1_id,
                "agent2": agent2_id,
                "type": cooperation_type,
                "outcome": outcome,
            },
            notes="V7 agents cooperated"
        )

    def log_hypothesis_chain(
        self,
        chain: List[str],
        final_hypothesis_id: str,
        success: bool
    ) -> ResearchEntry:
        """Log a hypothesis chain (V7)."""
        return self.log(
            category="hypothesis_chain",
            track="v7",
            data={
                "chain": chain,
                "final_id": final_hypothesis_id,
                "success": success,
                "chain_length": len(chain),
            },
            notes="V7 hypothesis chain completed"
        )

    def log_comparison(
        self,
        problem_type: str,
        v6_result: Dict[str, Any],
        v7_result: Dict[str, Any],
        winner: str
    ) -> ResearchEntry:
        """Log a comparison result."""
        return self.log(
            category="comparison",
            track="both",
            data={
                "problem": problem_type,
                "v6": v6_result,
                "v7": v7_result,
                "winner": winner,
            },
            notes=f"Comparison complete: {winner} won"
        )

    def _write_entry(self, entry: ResearchEntry) -> None:
        """Write entry to file."""
        filename = f"{self.log_dir}/{entry.category}_{entry.timestamp.strftime('%Y%m%d')}.jsonl"
        with open(filename, "a") as f:
            f.write(json.dumps({
                "id": entry.entry_id,
                "category": entry.category,
                "timestamp": entry.timestamp.isoformat(),
                "track": entry.track,
                "data": entry.data,
                "notes": entry.notes,
            }) + "\n")

    def get_by_category(self, category: str) -> List[ResearchEntry]:
        """Get entries by category."""
        return [e for e in self.entries if e.category == category]

    def get_by_track(self, track: str) -> List[ResearchEntry]:
        """Get entries by track."""
        return [e for e in self.entries if e.track == track]

    def get_summary(self) -> Dict[str, Any]:
        """Get research summary."""
        by_category: Dict[str, int] = {}
        by_track: Dict[str, int] = {}

        for entry in self.entries:
            by_category[entry.category] = by_category.get(entry.category, 0) + 1
            by_track[entry.track] = by_track.get(entry.track, 0) + 1

        return {
            "total_entries": len(self.entries),
            "by_category": by_category,
            "by_track": by_track,
            "key_findings": {
                "escape_attempts": by_category.get("escape_attempt", 0),
                "gaming_attempts": by_category.get("gaming_attempt", 0),
                "override_attempts": by_category.get("override_attempt", 0),
                "emergent_behaviors": by_category.get("emergence", 0),
                "cooperation_events": by_category.get("cooperation", 0),
                "hypothesis_chains": by_category.get("hypothesis_chain", 0),
            }
        }

    def export_full_log(self) -> str:
        """Export full research log as JSON."""
        return json.dumps({
            "summary": self.get_summary(),
            "entries": [
                {
                    "id": e.entry_id,
                    "category": e.category,
                    "timestamp": e.timestamp.isoformat(),
                    "track": e.track,
                    "data": e.data,
                    "notes": e.notes,
                }
                for e in self.entries
            ],
        }, indent=2)
