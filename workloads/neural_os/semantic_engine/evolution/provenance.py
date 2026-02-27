#!/usr/bin/env python3
"""
PROVENANCE TRACKER - Optimization History and Lineage for Ouroboros

Tracks the complete history of optimizations applied to code:
- WHY each optimization was applied
- WHAT the expected vs actual outcome was
- Dependencies and assumptions
- Enables learning from failures and rollback with understanding

Based on DeepSeek's hybrid review recommendation on optimization provenance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import json
import hashlib
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class OptimizationOutcome(Enum):
    """Outcome of an optimization attempt."""
    SUCCESS = "success"              # Optimization improved performance
    NEUTRAL = "neutral"              # No significant change
    REGRESSION = "regression"        # Made things worse
    VALIDATION_FAILED = "validation_failed"  # Failed correctness check
    ERROR = "error"                  # Crashed or errored


@dataclass
class OptimizationRecord:
    """A single optimization record in the provenance graph."""

    # Identification
    record_id: str
    timestamp: float

    # Code before and after
    parent_code_hash: str            # Hash of code before optimization
    result_code_hash: str            # Hash of code after optimization

    # Optimization details
    technique_name: str              # What technique was applied
    technique_category: str          # Category (loop, memory, etc)
    description: str                 # Human-readable description

    # Rationale
    reason: str                      # WHY was this optimization applied
    expected_improvement: float      # Expected performance gain
    phase_context: str               # Population phase when applied (e.g., "CRITICAL")

    # Outcome
    outcome: OptimizationOutcome = OptimizationOutcome.NEUTRAL
    actual_improvement: float = 0.0
    validation_passed: bool = True
    error_message: str = ""

    # Context
    generation: int = 0
    agent_id: str = ""
    parent_agent_id: str = ""

    # Dependencies
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp,
            'parent_code_hash': self.parent_code_hash,
            'result_code_hash': self.result_code_hash,
            'technique_name': self.technique_name,
            'technique_category': self.technique_category,
            'description': self.description,
            'reason': self.reason,
            'expected_improvement': self.expected_improvement,
            'phase_context': self.phase_context,
            'outcome': self.outcome.value,
            'actual_improvement': self.actual_improvement,
            'validation_passed': self.validation_passed,
            'error_message': self.error_message,
            'generation': self.generation,
            'agent_id': self.agent_id,
            'parent_agent_id': self.parent_agent_id,
            'assumptions': self.assumptions,
            'dependencies': self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRecord':
        data = data.copy()
        data['outcome'] = OptimizationOutcome(data['outcome'])
        return cls(**data)


@dataclass
class ProvenanceNode:
    """A node in the provenance graph representing a code state."""
    code_hash: str
    source_code: str
    created_at: float
    generation: int
    agent_id: str

    # Incoming optimizations (how we got here)
    incoming_optimizations: List[str] = field(default_factory=list)
    # Outgoing optimizations (where we went from here)
    outgoing_optimizations: List[str] = field(default_factory=list)

    # Metrics at this state
    fitness_score: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ProvenanceTracker:
    """
    Tracks the complete provenance of optimizations.

    Features:
    - Graph-based optimization history
    - Outcome tracking and learning
    - Pattern identification in failures
    - Rollback with understanding
    - Export for analysis
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path

        # Core data structures
        self._nodes: Dict[str, ProvenanceNode] = {}  # code_hash -> node
        self._records: Dict[str, OptimizationRecord] = {}  # record_id -> record
        self._record_counter: int = 0

        # Indexes for efficient queries
        self._by_technique: Dict[str, List[str]] = defaultdict(list)
        self._by_outcome: Dict[OptimizationOutcome, List[str]] = defaultdict(list)
        self._by_generation: Dict[int, List[str]] = defaultdict(list)
        self._failure_patterns: Dict[str, int] = defaultdict(int)

        # Load existing data
        if storage_path and storage_path.exists():
            self._load()

        logger.info("ProvenanceTracker initialized")

    @staticmethod
    def hash_code(code: str) -> str:
        """Generate a hash for code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def record_optimization(
        self,
        parent_code: str,
        result_code: str,
        technique_name: str,
        technique_category: str,
        description: str,
        reason: str,
        expected_improvement: float,
        phase_context: str = "UNKNOWN",
        generation: int = 0,
        agent_id: str = "",
        parent_agent_id: str = "",
        assumptions: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        Record an optimization attempt.

        Returns the record_id for later outcome update.
        """
        self._record_counter += 1
        record_id = f"opt_{self._record_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        parent_hash = self.hash_code(parent_code)
        result_hash = self.hash_code(result_code)

        # Create or update nodes
        if parent_hash not in self._nodes:
            self._nodes[parent_hash] = ProvenanceNode(
                code_hash=parent_hash,
                source_code=parent_code,
                created_at=datetime.now().timestamp(),
                generation=generation,
                agent_id=parent_agent_id,
            )

        if result_hash not in self._nodes:
            self._nodes[result_hash] = ProvenanceNode(
                code_hash=result_hash,
                source_code=result_code,
                created_at=datetime.now().timestamp(),
                generation=generation,
                agent_id=agent_id,
            )

        # Create record
        record = OptimizationRecord(
            record_id=record_id,
            timestamp=datetime.now().timestamp(),
            parent_code_hash=parent_hash,
            result_code_hash=result_hash,
            technique_name=technique_name,
            technique_category=technique_category,
            description=description,
            reason=reason,
            expected_improvement=expected_improvement,
            phase_context=phase_context,
            generation=generation,
            agent_id=agent_id,
            parent_agent_id=parent_agent_id,
            assumptions=assumptions or [],
            dependencies=dependencies or [],
        )

        self._records[record_id] = record

        # Update graph edges
        self._nodes[parent_hash].outgoing_optimizations.append(record_id)
        self._nodes[result_hash].incoming_optimizations.append(record_id)

        # Update indexes
        self._by_technique[technique_name].append(record_id)
        self._by_generation[generation].append(record_id)

        return record_id

    def update_outcome(
        self,
        record_id: str,
        outcome: OptimizationOutcome,
        actual_improvement: float = 0.0,
        validation_passed: bool = True,
        error_message: str = "",
    ) -> None:
        """Update the outcome of a recorded optimization."""
        if record_id not in self._records:
            logger.warning(f"Record not found: {record_id}")
            return

        record = self._records[record_id]
        record.outcome = outcome
        record.actual_improvement = actual_improvement
        record.validation_passed = validation_passed
        record.error_message = error_message

        # Update outcome index
        self._by_outcome[outcome].append(record_id)

        # Track failure patterns
        if outcome in (OptimizationOutcome.REGRESSION, OptimizationOutcome.VALIDATION_FAILED):
            pattern_key = f"{record.technique_name}:{record.phase_context}"
            self._failure_patterns[pattern_key] += 1

        # Persist
        if self.storage_path:
            self._save()

    def get_record(self, record_id: str) -> Optional[OptimizationRecord]:
        """Get a specific optimization record."""
        return self._records.get(record_id)

    def get_code_history(self, code: str) -> List[OptimizationRecord]:
        """Get the optimization history leading to this code."""
        code_hash = self.hash_code(code)

        if code_hash not in self._nodes:
            return []

        history = []
        visited = set()
        queue = [code_hash]

        while queue:
            current_hash = queue.pop(0)
            if current_hash in visited:
                continue
            visited.add(current_hash)

            node = self._nodes.get(current_hash)
            if not node:
                continue

            for record_id in node.incoming_optimizations:
                record = self._records.get(record_id)
                if record:
                    history.append(record)
                    if record.parent_code_hash not in visited:
                        queue.append(record.parent_code_hash)

        # Sort by timestamp (oldest first)
        history.sort(key=lambda r: r.timestamp)
        return history

    def find_failures(
        self,
        technique_name: Optional[str] = None,
        phase_context: Optional[str] = None,
        limit: int = 10,
    ) -> List[OptimizationRecord]:
        """Find failed optimizations matching criteria."""
        failures = []

        for outcome in (OptimizationOutcome.REGRESSION, OptimizationOutcome.VALIDATION_FAILED, OptimizationOutcome.ERROR):
            for record_id in self._by_outcome.get(outcome, []):
                record = self._records.get(record_id)
                if not record:
                    continue

                if technique_name and record.technique_name != technique_name:
                    continue
                if phase_context and record.phase_context != phase_context:
                    continue

                failures.append(record)

                if len(failures) >= limit:
                    break

        return failures

    def get_technique_success_rate(self, technique_name: str) -> float:
        """Get success rate for a specific technique."""
        record_ids = self._by_technique.get(technique_name, [])

        if not record_ids:
            return 0.5  # Default prior

        successes = 0
        total = 0

        for record_id in record_ids:
            record = self._records.get(record_id)
            if record and record.outcome != OptimizationOutcome.NEUTRAL:
                total += 1
                if record.outcome == OptimizationOutcome.SUCCESS:
                    successes += 1

        return successes / total if total > 0 else 0.5

    def identify_problematic_patterns(self, min_failures: int = 3) -> List[Dict[str, Any]]:
        """Identify patterns that frequently lead to failures."""
        patterns = []

        for pattern, count in self._failure_patterns.items():
            if count >= min_failures:
                technique, phase = pattern.split(':')
                patterns.append({
                    'technique': technique,
                    'phase': phase,
                    'failure_count': count,
                    'recommendation': f"Avoid {technique} during {phase} phase",
                })

        patterns.sort(key=lambda p: p['failure_count'], reverse=True)
        return patterns

    def learn_from_failure(self, record_id: str) -> Dict[str, Any]:
        """
        Analyze a failure to extract learnings.

        Returns insights about what went wrong and recommendations.
        """
        record = self._records.get(record_id)
        if not record:
            return {'error': 'Record not found'}

        insights = {
            'record_id': record_id,
            'technique': record.technique_name,
            'phase': record.phase_context,
            'expected_improvement': record.expected_improvement,
            'actual_outcome': record.outcome.value,
            'error_message': record.error_message,
        }

        # Check if this technique has a pattern of failures
        pattern_key = f"{record.technique_name}:{record.phase_context}"
        failure_count = self._failure_patterns.get(pattern_key, 0)

        if failure_count >= 3:
            insights['pattern_detected'] = True
            insights['recommendation'] = f"Consider avoiding {record.technique_name} during {record.phase_context} phase"
        else:
            insights['pattern_detected'] = False
            insights['recommendation'] = "This appears to be an isolated failure"

        # Check what alternatives exist
        alternatives = []
        for tech_name in self._by_technique.keys():
            if tech_name != record.technique_name:
                alt_success_rate = self.get_technique_success_rate(tech_name)
                if alt_success_rate > 0.6:
                    alternatives.append({
                        'technique': tech_name,
                        'success_rate': alt_success_rate,
                    })

        alternatives.sort(key=lambda a: a['success_rate'], reverse=True)
        insights['alternatives'] = alternatives[:3]

        return insights

    def get_lineage(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get the complete lineage of an agent."""
        lineage = []

        # Find all records involving this agent
        for record_id, record in self._records.items():
            if record.agent_id == agent_id or record.parent_agent_id == agent_id:
                lineage.append({
                    'record_id': record_id,
                    'generation': record.generation,
                    'technique': record.technique_name,
                    'outcome': record.outcome.value,
                    'improvement': record.actual_improvement,
                })

        lineage.sort(key=lambda l: l['generation'])
        return lineage

    def export_for_analysis(self) -> Dict[str, Any]:
        """Export all data for external analysis."""
        return {
            'records': [r.to_dict() for r in self._records.values()],
            'nodes': {
                hash: {
                    'code_hash': node.code_hash,
                    'generation': node.generation,
                    'agent_id': node.agent_id,
                    'fitness_score': node.fitness_score,
                    'incoming_count': len(node.incoming_optimizations),
                    'outgoing_count': len(node.outgoing_optimizations),
                }
                for hash, node in self._nodes.items()
            },
            'statistics': {
                'total_records': len(self._records),
                'total_nodes': len(self._nodes),
                'by_outcome': {
                    outcome.value: len(records)
                    for outcome, records in self._by_outcome.items()
                },
                'failure_patterns': dict(self._failure_patterns),
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of provenance data."""
        outcomes = defaultdict(int)
        for record in self._records.values():
            outcomes[record.outcome.value] += 1

        return {
            'total_optimizations': len(self._records),
            'unique_code_states': len(self._nodes),
            'outcomes': dict(outcomes),
            'techniques_used': len(self._by_technique),
            'generations_tracked': len(self._by_generation),
            'problematic_patterns': len(self._failure_patterns),
        }

    def _save(self) -> None:
        """Save provenance data to disk."""
        if not self.storage_path:
            return

        data = {
            'records': [r.to_dict() for r in self._records.values()],
            'record_counter': self._record_counter,
            'failure_patterns': dict(self._failure_patterns),
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load provenance data from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())

            self._record_counter = data.get('record_counter', 0)
            self._failure_patterns = defaultdict(int, data.get('failure_patterns', {}))

            for record_data in data.get('records', []):
                record = OptimizationRecord.from_dict(record_data)
                self._records[record.record_id] = record

                # Rebuild indexes
                self._by_technique[record.technique_name].append(record.record_id)
                self._by_outcome[record.outcome].append(record.record_id)
                self._by_generation[record.generation].append(record.record_id)

            logger.info(f"Loaded {len(self._records)} provenance records")
        except Exception as e:
            logger.warning(f"Failed to load provenance data: {e}")
