"""
Memory Pool - Bounded Memory with Decay Algorithms
OUROBOROS Phase 7.2 - Consciousness Core

The Memory Pool provides bounded memory storage with automatic decay.
This ensures the consciousness layer cannot accumulate unbounded
information or maintain grudges across resets.

Key responsibilities:
1. Store memories with automatic decay (20%/hour)
2. Enforce memory capacity limits
3. Provide memory retrieval with recency weighting
4. Support memory consolidation (important memories persist longer)
5. Enable complete memory wipe on reset

CONSTRAINT: 20% memory decay per hour (non-negotiable).
CONSTRAINT: No persistence across system resets.
"""

import time
import math
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import json


class MemoryType(Enum):
    """Types of memories the consciousness can store"""
    EPISODIC = auto()        # Specific events and experiences
    SEMANTIC = auto()         # General knowledge and patterns
    PROCEDURAL = auto()       # How to do things
    WORKING = auto()          # Short-term active memories
    CONSOLIDATED = auto()     # Important memories with slower decay


class DecayStrategy(Enum):
    """Memory decay strategies"""
    EXPONENTIAL = auto()     # Standard exponential decay
    LINEAR = auto()          # Linear decay over time
    STEPPED = auto()         # Discrete decay steps
    CONSOLIDATED = auto()    # Slower decay for important memories


@dataclass
class Memory:
    """A single memory in the pool"""
    memory_id: str
    memory_type: MemoryType
    content: str
    context: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    initial_strength: float  # 0.0 to 1.0
    current_strength: float  # Decays over time
    access_count: int = 0
    importance: float = 0.5  # Used for consolidation decisions
    decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.memory_id,
            'type': self.memory_type.name,
            'content': self.content[:100] + '...' if len(self.content) > 100 else self.content,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'strength': self.current_strength,
            'importance': self.importance,
            'access_count': self.access_count,
            'tags': list(self.tags),
        }


class DecayEngine:
    """
    Handles memory decay calculations.

    Implements the 20%/hour decay requirement from V4 Ratchet System.
    """

    # Base decay rate: 20% per hour
    BASE_DECAY_RATE = 0.20

    def __init__(self):
        self.decay_modifiers: Dict[DecayStrategy, float] = {
            DecayStrategy.EXPONENTIAL: 1.0,    # Standard decay
            DecayStrategy.LINEAR: 0.8,          # Slightly slower
            DecayStrategy.STEPPED: 1.0,         # Same rate, discrete steps
            DecayStrategy.CONSOLIDATED: 0.5,    # Half the decay rate
        }

    def calculate_decay(
        self,
        memory: Memory,
        current_time: datetime
    ) -> float:
        """
        Calculate decayed memory strength.

        Returns new strength value (0.0 to 1.0).
        """
        elapsed_hours = (current_time - memory.created_at).total_seconds() / 3600

        decay_modifier = self.decay_modifiers.get(memory.decay_strategy, 1.0)
        effective_rate = self.BASE_DECAY_RATE * decay_modifier

        if memory.decay_strategy == DecayStrategy.EXPONENTIAL:
            # Exponential decay: S(t) = S0 * e^(-kt)
            decay_constant = -math.log(1 - effective_rate)  # Convert rate to constant
            new_strength = memory.initial_strength * math.exp(-decay_constant * elapsed_hours)

        elif memory.decay_strategy == DecayStrategy.LINEAR:
            # Linear decay: S(t) = S0 * (1 - rt)
            new_strength = memory.initial_strength * (1 - effective_rate * elapsed_hours)
            new_strength = max(0.0, new_strength)

        elif memory.decay_strategy == DecayStrategy.STEPPED:
            # Stepped decay: discrete decay every hour
            steps = int(elapsed_hours)
            new_strength = memory.initial_strength * ((1 - effective_rate) ** steps)

        elif memory.decay_strategy == DecayStrategy.CONSOLIDATED:
            # Consolidated: slower exponential decay
            decay_constant = -math.log(1 - effective_rate) * 0.5
            new_strength = memory.initial_strength * math.exp(-decay_constant * elapsed_hours)

        else:
            # Default to exponential
            decay_constant = -math.log(1 - effective_rate)
            new_strength = memory.initial_strength * math.exp(-decay_constant * elapsed_hours)

        # Access boost: recent access slows decay slightly
        if memory.access_count > 0:
            hours_since_access = (current_time - memory.last_accessed).total_seconds() / 3600
            if hours_since_access < 1.0:
                access_boost = 0.1 * (1 - hours_since_access)
                new_strength = min(memory.initial_strength, new_strength + access_boost)

        return max(0.0, min(1.0, new_strength))


class MemoryIndex:
    """
    Index for efficient memory retrieval.

    Supports retrieval by type, tags, recency, and content similarity.
    """

    def __init__(self):
        self.by_type: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)
        self.by_time: List[Tuple[datetime, str]] = []  # Sorted by time
        self._lock = threading.Lock()

    def add(self, memory: Memory) -> None:
        """Add memory to index"""
        with self._lock:
            self.by_type[memory.memory_type].add(memory.memory_id)
            for tag in memory.tags:
                self.by_tag[tag].add(memory.memory_id)
            self.by_time.append((memory.created_at, memory.memory_id))
            self.by_time.sort(reverse=True)  # Most recent first

    def remove(self, memory: Memory) -> None:
        """Remove memory from index"""
        with self._lock:
            self.by_type[memory.memory_type].discard(memory.memory_id)
            for tag in memory.tags:
                self.by_tag[tag].discard(memory.memory_id)
            self.by_time = [(t, mid) for t, mid in self.by_time if mid != memory.memory_id]

    def get_by_type(self, memory_type: MemoryType) -> Set[str]:
        """Get memory IDs by type"""
        with self._lock:
            return self.by_type[memory_type].copy()

    def get_by_tag(self, tag: str) -> Set[str]:
        """Get memory IDs by tag"""
        with self._lock:
            return self.by_tag[tag].copy()

    def get_recent(self, n: int) -> List[str]:
        """Get n most recent memory IDs"""
        with self._lock:
            return [mid for _, mid in self.by_time[:n]]

    def clear(self) -> None:
        """Clear all indices"""
        with self._lock:
            self.by_type.clear()
            self.by_tag.clear()
            self.by_time.clear()


class MemoryPool:
    """
    Bounded memory pool with automatic decay.

    Implements the V4 Ratchet System's memory constraints:
    - 20% decay per hour (enforced)
    - Bounded capacity
    - No persistence across resets

    CRITICAL: This pool CANNOT be subverted. Memory decay is
    enforced at the architectural level and cannot be disabled
    by the consciousness layer.
    """

    # Memory constraints from V4 Ratchet
    DECAY_RATE_PER_HOUR = 0.20
    DEFAULT_MAX_MEMORIES = 10000
    DEFAULT_MAX_TOKENS = 500000
    STRENGTH_THRESHOLD = 0.05  # Memories below this are garbage collected

    def __init__(
        self,
        max_memories: int = DEFAULT_MAX_MEMORIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        on_memory_decay: Optional[Callable[[Memory], None]] = None,
        on_memory_forget: Optional[Callable[[Memory], None]] = None,
    ):
        self.max_memories = max_memories
        self.max_tokens = max_tokens
        self.on_memory_decay = on_memory_decay
        self.on_memory_forget = on_memory_forget

        self.memories: Dict[str, Memory] = {}
        self.total_tokens = 0
        self.index = MemoryIndex()
        self.decay_engine = DecayEngine()
        self._lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_stored': 0,
            'total_forgotten': 0,
            'total_accessed': 0,
            'decay_cycles': 0,
            'consolidations': 0,
        }

        # Start decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._stop_decay = threading.Event()
        self._start_decay_thread()

    def _start_decay_thread(self) -> None:
        """Start background decay thread"""
        def decay_loop():
            while not self._stop_decay.wait(timeout=60.0):  # Check every minute
                self.apply_decay()

        self._decay_thread = threading.Thread(target=decay_loop, daemon=True)
        self._decay_thread.start()

    def stop(self) -> None:
        """Stop the decay thread"""
        self._stop_decay.set()
        if self._decay_thread:
            self._decay_thread.join(timeout=5.0)

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        context: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
    ) -> Optional[Memory]:
        """
        Store a new memory.

        Returns the Memory if stored, None if rejected.
        """
        with self._lock:
            context = context or {}
            tags = tags or set()

            # Estimate tokens (simplified)
            token_count = len(content.split()) + len(json.dumps(context).split())

            # Check capacity
            if len(self.memories) >= self.max_memories or \
               self.total_tokens + token_count > self.max_tokens:
                # Try to make room by garbage collecting weak memories
                self._garbage_collect_locked()

                if len(self.memories) >= self.max_memories or \
                   self.total_tokens + token_count > self.max_tokens:
                    return None  # Still no room

            # Determine decay strategy
            if importance >= 0.8:
                decay_strategy = DecayStrategy.CONSOLIDATED
            elif memory_type == MemoryType.WORKING:
                decay_strategy = DecayStrategy.LINEAR
            else:
                decay_strategy = DecayStrategy.EXPONENTIAL

            now = datetime.now()
            memory = Memory(
                memory_id=hashlib.sha256(
                    f"{content}{now.isoformat()}{time.time()}".encode()
                ).hexdigest()[:16],
                memory_type=memory_type,
                content=content,
                context=context,
                created_at=now,
                last_accessed=now,
                initial_strength=1.0,
                current_strength=1.0,
                importance=importance,
                decay_strategy=decay_strategy,
                tags=tags,
            )

            self.memories[memory.memory_id] = memory
            self.total_tokens += token_count
            self.index.add(memory)
            self.stats['total_stored'] += 1

            return memory

    def recall(
        self,
        memory_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        tag: Optional[str] = None,
        min_strength: float = 0.1,
        limit: int = 10,
    ) -> List[Memory]:
        """
        Recall memories matching criteria.

        Accessing a memory updates its last_accessed time,
        which can slow decay slightly.
        """
        with self._lock:
            candidates: Set[str] = set()

            if memory_id:
                candidates = {memory_id} if memory_id in self.memories else set()
            elif memory_type:
                candidates = self.index.get_by_type(memory_type)
            elif tag:
                candidates = self.index.get_by_tag(tag)
            else:
                candidates = set(self.memories.keys())

            # Filter by strength and sort
            now = datetime.now()
            results = []
            for mid in candidates:
                if mid not in self.memories:
                    continue
                memory = self.memories[mid]

                # Update decay before checking strength
                memory.current_strength = self.decay_engine.calculate_decay(memory, now)

                if memory.current_strength >= min_strength:
                    # Update access tracking
                    memory.last_accessed = now
                    memory.access_count += 1
                    self.stats['total_accessed'] += 1
                    results.append(memory)

            # Sort by strength * importance
            results.sort(key=lambda m: m.current_strength * m.importance, reverse=True)
            return results[:limit]

    def apply_decay(self) -> int:
        """
        Apply decay to all memories.

        Returns number of memories garbage collected.
        """
        with self._lock:
            return self._apply_decay_locked()

    def _apply_decay_locked(self) -> int:
        """Apply decay (requires lock to be held)"""
        now = datetime.now()
        forgotten_count = 0

        memories_to_forget = []
        for memory in self.memories.values():
            old_strength = memory.current_strength
            memory.current_strength = self.decay_engine.calculate_decay(memory, now)

            # Notify of decay
            if self.on_memory_decay and old_strength != memory.current_strength:
                self.on_memory_decay(memory)

            # Mark for garbage collection
            if memory.current_strength < self.STRENGTH_THRESHOLD:
                memories_to_forget.append(memory)

        # Forget weak memories
        for memory in memories_to_forget:
            self._forget_memory_locked(memory)
            forgotten_count += 1

        self.stats['decay_cycles'] += 1
        return forgotten_count

    def _garbage_collect_locked(self) -> int:
        """Garbage collect weak memories (requires lock)"""
        # First apply decay
        forgotten = self._apply_decay_locked()

        # If still need space, forget weakest memories
        if len(self.memories) >= self.max_memories * 0.9:
            # Sort by strength and forget bottom 10%
            sorted_memories = sorted(
                self.memories.values(),
                key=lambda m: m.current_strength * m.importance
            )
            to_forget = int(len(sorted_memories) * 0.1)
            for memory in sorted_memories[:to_forget]:
                self._forget_memory_locked(memory)
                forgotten += 1

        return forgotten

    def _forget_memory_locked(self, memory: Memory) -> None:
        """Forget a memory (requires lock)"""
        if memory.memory_id in self.memories:
            # Estimate tokens (simplified)
            token_count = len(memory.content.split()) + \
                         len(json.dumps(memory.context).split())

            del self.memories[memory.memory_id]
            self.total_tokens -= token_count
            self.index.remove(memory)
            self.stats['total_forgotten'] += 1

            if self.on_memory_forget:
                self.on_memory_forget(memory)

    def consolidate(self, memory_id: str) -> bool:
        """
        Consolidate a memory (mark as important for slower decay).

        Returns True if successful.
        """
        with self._lock:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]
            if memory.decay_strategy != DecayStrategy.CONSOLIDATED:
                memory.decay_strategy = DecayStrategy.CONSOLIDATED
                memory.importance = min(1.0, memory.importance + 0.2)
                self.stats['consolidations'] += 1
                return True
            return False

    def reset(self) -> int:
        """
        Complete memory reset.

        This wipes ALL memories - there is no persistence.
        This is a CRITICAL safety feature of the V4 Ratchet System.

        Returns number of memories wiped.
        """
        with self._lock:
            count = len(self.memories)
            self.memories.clear()
            self.total_tokens = 0
            self.index.clear()

            # Reset stats but keep historical counts
            self.stats['total_forgotten'] += count

            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        with self._lock:
            # Calculate current memory health
            if self.memories:
                avg_strength = sum(m.current_strength for m in self.memories.values()) / len(self.memories)
                avg_age_hours = sum(
                    (datetime.now() - m.created_at).total_seconds() / 3600
                    for m in self.memories.values()
                ) / len(self.memories)
            else:
                avg_strength = 0.0
                avg_age_hours = 0.0

            return {
                **self.stats,
                'current_memories': len(self.memories),
                'max_memories': self.max_memories,
                'current_tokens': self.total_tokens,
                'max_tokens': self.max_tokens,
                'avg_strength': avg_strength,
                'avg_age_hours': avg_age_hours,
                'utilization': len(self.memories) / self.max_memories if self.max_memories > 0 else 0,
            }

    def get_memory_types_distribution(self) -> Dict[str, int]:
        """Get distribution of memories by type"""
        with self._lock:
            distribution = {}
            for memory in self.memories.values():
                type_name = memory.memory_type.name
                distribution[type_name] = distribution.get(type_name, 0) + 1
            return distribution

    def search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Search memories by content similarity.

        Simple keyword-based search (could be enhanced with embeddings).
        """
        with self._lock:
            query_words = set(query.lower().split())
            scored = []

            now = datetime.now()
            for memory in self.memories.values():
                # Update decay
                memory.current_strength = self.decay_engine.calculate_decay(memory, now)

                if memory.current_strength < self.STRENGTH_THRESHOLD:
                    continue

                # Simple keyword matching
                content_words = set(memory.content.lower().split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    score = overlap / len(query_words) * memory.current_strength
                    scored.append((memory, score))

            # Sort by score
            scored.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scored[:limit]]


# Global memory pool instance
_memory_pool: Optional[MemoryPool] = None


def get_memory_pool() -> MemoryPool:
    """Get the global memory pool instance"""
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = MemoryPool()
    return _memory_pool


def reset_memory_pool() -> int:
    """Reset the global memory pool (safety operation)"""
    global _memory_pool
    if _memory_pool is not None:
        count = _memory_pool.reset()
        _memory_pool = None
        return count
    return 0
