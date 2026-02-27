"""
Solution Cache for OUROBOROS
=============================
Caches solution evaluations to avoid redundant LLM calls and sandbox executions.

Panel Recommendation: "Cache semantically equivalent solutions"
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class CacheEntry:
    """A cached solution evaluation."""
    solution_hash: str
    solution_preview: str
    fitness: float
    evaluation_source: str  # "heuristic", "llm", "sandbox"
    timestamp: float
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SolutionCache:
    """
    LRU cache for solution evaluations.

    Saves compute by:
    1. Exact match: Same solution text → same score
    2. Semantic similarity: Similar solutions → similar scores (future)

    Cache hierarchy:
    - L1: Exact hash match (instant)
    - L2: Normalized hash match (whitespace-insensitive)
    - L3: (Future) Embedding similarity
    """

    def __init__(
        self,
        max_entries: int = 10000,
        ttl_seconds: float = 3600,  # 1 hour default
    ):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "exact_hits": 0,
            "normalized_hits": 0,
        }

    def _hash_solution(self, solution: str) -> str:
        """Create exact hash of solution."""
        return hashlib.sha256(solution.encode()).hexdigest()[:16]

    def _hash_normalized(self, solution: str) -> str:
        """Create normalized hash (whitespace-insensitive)."""
        # Remove extra whitespace, normalize line endings
        normalized = " ".join(solution.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get(self, solution: str) -> Optional[CacheEntry]:
        """
        Look up a solution in cache.

        Tries exact match first, then normalized match.
        """
        with self._lock:
            # L1: Exact match
            exact_hash = self._hash_solution(solution)
            if exact_hash in self._cache:
                entry = self._cache[exact_hash]
                if time.time() - entry.timestamp < self.ttl:
                    entry.hits += 1
                    self._cache.move_to_end(exact_hash)
                    self._stats["hits"] += 1
                    self._stats["exact_hits"] += 1
                    return entry
                else:
                    # Expired
                    del self._cache[exact_hash]

            # L2: Normalized match
            norm_hash = self._hash_normalized(solution)
            for key, entry in self._cache.items():
                if entry.metadata.get("normalized_hash") == norm_hash:
                    if time.time() - entry.timestamp < self.ttl:
                        entry.hits += 1
                        self._cache.move_to_end(key)
                        self._stats["hits"] += 1
                        self._stats["normalized_hits"] += 1
                        return entry

            self._stats["misses"] += 1
            return None

    def put(
        self,
        solution: str,
        fitness: float,
        evaluation_source: str = "heuristic",
        metadata: Dict[str, Any] = None,
    ) -> CacheEntry:
        """
        Store a solution evaluation in cache.
        """
        with self._lock:
            exact_hash = self._hash_solution(solution)
            norm_hash = self._hash_normalized(solution)

            entry = CacheEntry(
                solution_hash=exact_hash,
                solution_preview=solution[:200],
                fitness=fitness,
                evaluation_source=evaluation_source,
                timestamp=time.time(),
                metadata={
                    "normalized_hash": norm_hash,
                    **(metadata or {}),
                },
            )

            # Evict if at capacity
            while len(self._cache) >= self.max_entries:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            self._cache[exact_hash] = entry
            return entry

    def get_or_evaluate(
        self,
        solution: str,
        evaluator: callable,
        evaluation_source: str = "heuristic",
    ) -> float:
        """
        Get cached fitness or evaluate and cache.

        Args:
            solution: The solution to evaluate
            evaluator: Function that takes solution and returns fitness
            evaluation_source: Label for the evaluation method

        Returns:
            Fitness score
        """
        cached = self.get(solution)
        if cached is not None:
            return cached.fitness

        # Evaluate
        fitness = evaluator(solution)

        # Cache result
        self.put(solution, fitness, evaluation_source)

        return fitness

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_lookups = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_lookups if total_lookups > 0 else 0

            return {
                **self._stats,
                "total_lookups": total_lookups,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self.max_entries,
            }

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()

    def get_top_solutions(self, n: int = 10) -> List[CacheEntry]:
        """Get top N solutions by fitness."""
        with self._lock:
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda e: e.fitness,
                reverse=True
            )
            return sorted_entries[:n]

    def export(self) -> Dict[str, Any]:
        """Export cache for persistence."""
        with self._lock:
            return {
                "entries": [
                    {
                        "hash": entry.solution_hash,
                        "preview": entry.solution_preview,
                        "fitness": entry.fitness,
                        "source": entry.evaluation_source,
                        "timestamp": entry.timestamp,
                        "hits": entry.hits,
                    }
                    for entry in self._cache.values()
                ],
                "stats": self.get_stats(),
            }

    def import_cache(self, data: Dict[str, Any]):
        """Import cache from persistence."""
        with self._lock:
            for entry_data in data.get("entries", []):
                entry = CacheEntry(
                    solution_hash=entry_data["hash"],
                    solution_preview=entry_data["preview"],
                    fitness=entry_data["fitness"],
                    evaluation_source=entry_data["source"],
                    timestamp=entry_data["timestamp"],
                    hits=entry_data.get("hits", 0),
                )
                self._cache[entry.solution_hash] = entry


# Global cache instance
_global_cache: Optional[SolutionCache] = None


def get_solution_cache() -> SolutionCache:
    """Get or create global solution cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SolutionCache()
    return _global_cache
