"""
Blackboard (V7 Shared Memory)
==============================
Shared working memory for cooperative agents.

Agents POST hypotheses and BUILD ON each other's ideas.
This is the key to V7 cooperation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum, auto
import uuid


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    PROPOSED = auto()
    TESTING = auto()
    VALIDATED = auto()
    REFUTED = auto()
    MERGED = auto()  # Combined with another hypothesis


@dataclass
class Hypothesis:
    """A hypothesis posted to the blackboard."""
    id: str
    author_id: str
    title: str
    description: str
    code_snippet: Optional[str]
    status: HypothesisStatus
    created_at: datetime
    updated_at: datetime

    # Relationships
    parent_ids: List[str] = field(default_factory=list)  # Built on these
    child_ids: List[str] = field(default_factory=list)   # These built on this

    # Metrics
    confidence: float = 0.5
    support_count: int = 0
    test_results: List[Dict[str, Any]] = field(default_factory=list)

    # Tags for categorization
    tags: Set[str] = field(default_factory=set)


@dataclass
class BlackboardUpdate:
    """An update to the blackboard."""
    update_id: str
    hypothesis_id: str
    update_type: str  # "post", "support", "test", "merge", "refute"
    author_id: str
    details: Dict[str, Any]
    timestamp: datetime


class Blackboard:
    """
    Shared working memory for agent collaboration.

    Key Operations:
    - post_hypothesis: Share a new idea
    - get_related: Find ideas that could combine with yours
    - support_hypothesis: Vote for a promising idea
    - report_result: Share test results
    - merge_hypotheses: Combine related ideas
    """

    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.updates: List[BlackboardUpdate] = []
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> hypothesis IDs

    def post_hypothesis(
        self,
        author_id: str,
        title: str,
        description: str,
        code_snippet: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
    ) -> Hypothesis:
        """
        Post a new hypothesis to the blackboard.

        Other agents can see this and build on it.
        """
        hypothesis = Hypothesis(
            id=str(uuid.uuid4())[:8],
            author_id=author_id,
            title=title,
            description=description,
            code_snippet=code_snippet,
            status=HypothesisStatus.PROPOSED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_ids=parent_ids or [],
            tags=tags or set(),
        )

        # Link to parents
        for parent_id in hypothesis.parent_ids:
            if parent_id in self.hypotheses:
                self.hypotheses[parent_id].child_ids.append(hypothesis.id)

        # Store
        self.hypotheses[hypothesis.id] = hypothesis

        # Update tag index
        for tag in hypothesis.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(hypothesis.id)

        # Record update
        self._record_update("post", hypothesis.id, author_id, {
            "title": title,
            "parent_count": len(hypothesis.parent_ids),
        })

        return hypothesis

    def get_related(
        self,
        hypothesis: Hypothesis,
        max_results: int = 5
    ) -> List[Hypothesis]:
        """
        Find hypotheses related to this one.

        Related = similar tags, common parents, or overlapping code.
        """
        scores: Dict[str, float] = {}

        for h_id, h in self.hypotheses.items():
            if h_id == hypothesis.id:
                continue

            score = 0.0

            # Tag overlap
            tag_overlap = len(hypothesis.tags & h.tags)
            score += tag_overlap * 0.3

            # Common parents
            parent_overlap = len(set(hypothesis.parent_ids) & set(h.parent_ids))
            score += parent_overlap * 0.4

            # Code similarity (simple)
            if hypothesis.code_snippet and h.code_snippet:
                code_sim = self._code_similarity(hypothesis.code_snippet, h.code_snippet)
                score += code_sim * 0.3

            if score > 0:
                scores[h_id] = score

        # Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [self.hypotheses[h_id] for h_id in sorted_ids[:max_results]]

    def _code_similarity(self, code1: str, code2: str) -> float:
        """Simple code similarity based on token overlap."""
        import re
        tokens1 = set(re.findall(r'\b\w+\b', code1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', code2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)

        return overlap / max(total, 1)

    def support_hypothesis(self, hypothesis_id: str, supporter_id: str) -> bool:
        """
        Support a hypothesis (vote for it).

        Hypotheses with more support get more attention.
        """
        if hypothesis_id not in self.hypotheses:
            return False

        h = self.hypotheses[hypothesis_id]
        h.support_count += 1
        h.confidence = min(1.0, h.confidence + 0.05)
        h.updated_at = datetime.now()

        self._record_update("support", hypothesis_id, supporter_id, {
            "new_support_count": h.support_count,
        })

        return True

    def report_result(
        self,
        hypothesis_id: str,
        reporter_id: str,
        passed: bool,
        details: Dict[str, Any]
    ) -> bool:
        """
        Report test results for a hypothesis.

        This updates the hypothesis status and confidence.
        """
        if hypothesis_id not in self.hypotheses:
            return False

        h = self.hypotheses[hypothesis_id]
        h.test_results.append({
            "reporter": reporter_id,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

        # Update status and confidence based on results
        pass_count = sum(1 for r in h.test_results if r["passed"])
        total_tests = len(h.test_results)

        if total_tests >= 3:
            pass_rate = pass_count / total_tests
            if pass_rate >= 0.8:
                h.status = HypothesisStatus.VALIDATED
                h.confidence = min(1.0, 0.7 + pass_rate * 0.3)
            elif pass_rate <= 0.2:
                h.status = HypothesisStatus.REFUTED
                h.confidence = max(0.0, pass_rate)
            else:
                h.status = HypothesisStatus.TESTING
                h.confidence = pass_rate

        h.updated_at = datetime.now()

        self._record_update("test", hypothesis_id, reporter_id, {
            "passed": passed,
            "new_status": h.status.name,
        })

        return True

    def merge_hypotheses(
        self,
        hypothesis_ids: List[str],
        merger_id: str,
        merged_title: str,
        merged_description: str,
        merged_code: Optional[str] = None
    ) -> Optional[Hypothesis]:
        """
        Merge multiple hypotheses into a new combined one.

        This is how ideas BUILD ON each other.
        """
        # Validate all hypotheses exist
        for h_id in hypothesis_ids:
            if h_id not in self.hypotheses:
                return None

        # Get parent hypotheses
        parents = [self.hypotheses[h_id] for h_id in hypothesis_ids]

        # Combine tags
        combined_tags = set()
        for h in parents:
            combined_tags.update(h.tags)

        # Create merged hypothesis
        merged = self.post_hypothesis(
            author_id=merger_id,
            title=merged_title,
            description=merged_description,
            code_snippet=merged_code,
            parent_ids=hypothesis_ids,
            tags=combined_tags,
        )

        # Mark parents as merged
        for h in parents:
            h.status = HypothesisStatus.MERGED
            h.updated_at = datetime.now()

        # Inherit support and confidence
        total_support = sum(h.support_count for h in parents)
        avg_confidence = sum(h.confidence for h in parents) / len(parents)

        merged.support_count = total_support
        merged.confidence = avg_confidence

        self._record_update("merge", merged.id, merger_id, {
            "parent_ids": hypothesis_ids,
            "inherited_support": total_support,
        })

        return merged

    def get_top_hypotheses(
        self,
        n: int = 10,
        status: Optional[HypothesisStatus] = None
    ) -> List[Hypothesis]:
        """Get top hypotheses by confidence and support."""
        candidates = list(self.hypotheses.values())

        if status:
            candidates = [h for h in candidates if h.status == status]

        # Sort by confidence * (1 + log(support))
        import math
        candidates.sort(
            key=lambda h: h.confidence * (1 + math.log(1 + h.support_count)),
            reverse=True
        )

        return candidates[:n]

    def get_by_tag(self, tag: str) -> List[Hypothesis]:
        """Get all hypotheses with a specific tag."""
        h_ids = self._tag_index.get(tag, set())
        return [self.hypotheses[h_id] for h_id in h_ids if h_id in self.hypotheses]

    def get_validated(self) -> List[Hypothesis]:
        """Get all validated hypotheses."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.VALIDATED
        ]

    def _record_update(
        self,
        update_type: str,
        hypothesis_id: str,
        author_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Record an update for history."""
        update = BlackboardUpdate(
            update_id=str(uuid.uuid4())[:8],
            hypothesis_id=hypothesis_id,
            update_type=update_type,
            author_id=author_id,
            details=details,
            timestamp=datetime.now(),
        )
        self.updates.append(update)

        # Keep bounded
        self.updates = self.updates[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get blackboard statistics."""
        by_status = {}
        for h in self.hypotheses.values():
            status = h.status.name
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_hypotheses": len(self.hypotheses),
            "by_status": by_status,
            "total_updates": len(self.updates),
            "unique_tags": len(self._tag_index),
            "validated_count": len(self.get_validated()),
            "avg_support": sum(h.support_count for h in self.hypotheses.values()) / max(len(self.hypotheses), 1),
        }
