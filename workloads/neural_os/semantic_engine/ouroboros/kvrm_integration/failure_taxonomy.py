"""
Failure Pattern Taxonomy for OUROBOROS
=======================================
Tracks and learns from failures to prevent repeated mistakes.

Panel Recommendation (Claude): "Focus on failure analysis, not success patterns"
Panel Recommendation (Grok): "Extract anti-patterns, create Hall of Shame"
"""

import hashlib
import time
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import re


FailureType = Literal[
    "syntax_error",      # Code doesn't parse
    "runtime_error",     # Code crashes when executed
    "incorrect_output",  # Wrong answer
    "timeout",           # Took too long
    "infinite_loop",     # Detected loop
    "unsafe_code",       # Security issue detected
    "empty_solution",    # No solution provided
    "format_error",      # Wrong output format
    "resource_exceeded", # Memory/CPU limit
    "semantic_error",    # Logic is wrong
]


@dataclass
class FailurePattern:
    """A detected failure pattern."""
    pattern_id: str
    failure_type: FailureType
    description: str
    detection_regex: Optional[str]  # Pattern to detect in code
    detection_keywords: List[str]   # Keywords that indicate this failure
    frequency: int = 0              # How often seen
    fatal: bool = False             # Does this crash the system?
    agents_affected: List[str] = field(default_factory=list)
    example_solutions: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    recovery_hints: List[str] = field(default_factory=list)

    def matches(self, solution: str, error_msg: str = "") -> bool:
        """Check if a solution matches this failure pattern."""
        combined = f"{solution}\n{error_msg}".lower()

        # Check keywords
        for keyword in self.detection_keywords:
            if keyword.lower() in combined:
                return True

        # Check regex
        if self.detection_regex:
            if re.search(self.detection_regex, combined, re.IGNORECASE):
                return True

        return False


@dataclass
class FailureEvent:
    """A single failure occurrence."""
    event_id: str
    pattern_id: str
    agent_id: str
    generation: int
    solution_preview: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class FailureTaxonomy:
    """
    Tracks failure patterns across the system.

    "Hall of Shame" - learns what NOT to do.
    """

    # Built-in patterns (common failures)
    BUILTIN_PATTERNS = [
        FailurePattern(
            pattern_id="infinite_recursion",
            failure_type="infinite_loop",
            description="Infinite recursion with no base case",
            detection_regex=r"def\s+(\w+)\([^)]*\):[^}]*\1\s*\(",
            detection_keywords=["recursionerror", "maximum recursion depth"],
            fatal=True,
            recovery_hints=["Add a base case", "Use iteration instead"],
        ),
        FailurePattern(
            pattern_id="syntax_missing_colon",
            failure_type="syntax_error",
            description="Missing colon after def/if/for/while",
            detection_regex=r"(def|if|for|while|class)\s+[^:]+(?!:)\s*$",
            detection_keywords=["syntaxerror", "expected ':'"],
            fatal=False,
            recovery_hints=["Add colon at end of statement"],
        ),
        FailurePattern(
            pattern_id="undefined_variable",
            failure_type="runtime_error",
            description="Using variable before definition",
            detection_regex=None,
            detection_keywords=["nameerror", "is not defined"],
            fatal=False,
            recovery_hints=["Define variable before use", "Check spelling"],
        ),
        FailurePattern(
            pattern_id="index_out_of_bounds",
            failure_type="runtime_error",
            description="Accessing list/array with invalid index",
            detection_regex=None,
            detection_keywords=["indexerror", "list index out of range"],
            fatal=False,
            recovery_hints=["Check array bounds", "Use len() to verify"],
        ),
        FailurePattern(
            pattern_id="type_mismatch",
            failure_type="runtime_error",
            description="Operation on incompatible types",
            detection_regex=None,
            detection_keywords=["typeerror", "unsupported operand"],
            fatal=False,
            recovery_hints=["Check input types", "Add type conversion"],
        ),
        FailurePattern(
            pattern_id="empty_return",
            failure_type="incorrect_output",
            description="Function returns None or empty result",
            detection_regex=r"return\s*$|return\s+None",
            detection_keywords=["returned none", "empty result"],
            fatal=False,
            recovery_hints=["Return the computed value", "Check logic flow"],
        ),
        FailurePattern(
            pattern_id="hardcoded_answer",
            failure_type="incorrect_output",
            description="Solution hardcodes specific test case answers",
            detection_regex=r"return\s+\[?\d+\]?.*#.*test",
            detection_keywords=["hardcoded", "magic number"],
            fatal=False,
            recovery_hints=["Implement actual algorithm", "Don't memorize test cases"],
        ),
        FailurePattern(
            pattern_id="no_function_def",
            failure_type="format_error",
            description="Missing required function definition",
            detection_regex=None,
            detection_keywords=["no function", "missing def"],
            fatal=False,
            recovery_hints=["Define the required function", "Check function name"],
        ),
        FailurePattern(
            pattern_id="unsafe_eval",
            failure_type="unsafe_code",
            description="Using eval() or exec() on input",
            detection_regex=r"(eval|exec)\s*\(",
            detection_keywords=["eval(", "exec("],
            fatal=True,
            recovery_hints=["Never use eval on untrusted input", "Parse manually"],
        ),
        FailurePattern(
            pattern_id="file_system_access",
            failure_type="unsafe_code",
            description="Attempting file system operations",
            detection_regex=r"(open|os\.|shutil\.|pathlib)",
            detection_keywords=["open(", "os.path", "file"],
            fatal=True,
            recovery_hints=["File operations not allowed", "Use in-memory data"],
        ),
    ]

    def __init__(self):
        # Pattern registry
        self.patterns: Dict[str, FailurePattern] = {}
        for p in self.BUILTIN_PATTERNS:
            self.patterns[p.pattern_id] = p

        # Event history
        self.events: List[FailureEvent] = []

        # Stats by type
        self.stats_by_type: Dict[FailureType, int] = defaultdict(int)
        self.stats_by_agent: Dict[str, int] = defaultdict(int)

    def detect_failure(
        self,
        solution: str,
        error_message: str = "",
        agent_id: str = "unknown",
        generation: int = 0,
    ) -> List[FailurePattern]:
        """
        Detect which failure patterns match a solution.

        Returns list of matching patterns.
        """
        matches = []

        for pattern in self.patterns.values():
            if pattern.matches(solution, error_message):
                matches.append(pattern)

                # Update pattern stats
                pattern.frequency += 1
                pattern.last_seen = datetime.now()
                if agent_id not in pattern.agents_affected:
                    pattern.agents_affected.append(agent_id)
                if len(pattern.example_solutions) < 5:
                    pattern.example_solutions.append(solution[:200])

                # Update global stats
                self.stats_by_type[pattern.failure_type] += 1
                self.stats_by_agent[agent_id] += 1

                # Record event
                event = FailureEvent(
                    event_id=f"fail_{generation}_{hashlib.md5(solution.encode()).hexdigest()[:8]}",
                    pattern_id=pattern.pattern_id,
                    agent_id=agent_id,
                    generation=generation,
                    solution_preview=solution[:200],
                    error_message=error_message[:500],
                )
                self.events.append(event)

        return matches

    def add_custom_pattern(
        self,
        pattern_id: str,
        failure_type: FailureType,
        description: str,
        detection_keywords: List[str],
        detection_regex: str = None,
        fatal: bool = False,
        recovery_hints: List[str] = None,
    ) -> FailurePattern:
        """Add a new failure pattern."""
        pattern = FailurePattern(
            pattern_id=pattern_id,
            failure_type=failure_type,
            description=description,
            detection_regex=detection_regex,
            detection_keywords=detection_keywords,
            fatal=fatal,
            recovery_hints=recovery_hints or [],
        )
        self.patterns[pattern_id] = pattern
        return pattern

    def get_warnings_for_agent(self, agent_id: str) -> List[str]:
        """Get warnings about patterns this agent frequently hits."""
        warnings = []

        # Find patterns this agent hits often
        agent_patterns = [
            p for p in self.patterns.values()
            if agent_id in p.agents_affected and p.frequency >= 2
        ]

        for pattern in sorted(agent_patterns, key=lambda p: -p.frequency)[:3]:
            warnings.append(
                f"Warning: You've hit '{pattern.description}' {pattern.frequency} times. "
                f"Hints: {', '.join(pattern.recovery_hints[:2])}"
            )

        return warnings

    def get_hall_of_shame(self, top_n: int = 10) -> List[Dict]:
        """Get the most frequent failure patterns."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )

        return [
            {
                "pattern_id": p.pattern_id,
                "description": p.description,
                "type": p.failure_type,
                "frequency": p.frequency,
                "fatal": p.fatal,
                "agents_affected": len(p.agents_affected),
                "recovery_hints": p.recovery_hints,
            }
            for p in sorted_patterns[:top_n]
            if p.frequency > 0
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get failure taxonomy summary."""
        return {
            "total_patterns": len(self.patterns),
            "total_failures": sum(self.stats_by_type.values()),
            "failures_by_type": dict(self.stats_by_type),
            "failures_by_agent": dict(self.stats_by_agent),
            "most_common_patterns": self.get_hall_of_shame(5),
            "fatal_patterns_hit": sum(
                1 for p in self.patterns.values()
                if p.fatal and p.frequency > 0
            ),
        }

    def get_prompt_injection(self, agent_id: str) -> str:
        """
        Generate prompt text warning agent about common failures.

        This is injected into agent prompts to prevent repeated mistakes.
        """
        warnings = self.get_warnings_for_agent(agent_id)
        if not warnings:
            return ""

        return "\n\nAVOID THESE COMMON MISTAKES:\n" + "\n".join(f"- {w}" for w in warnings)


# Global instance
_global_taxonomy: Optional[FailureTaxonomy] = None


def get_failure_taxonomy() -> FailureTaxonomy:
    """Get or create global failure taxonomy."""
    global _global_taxonomy
    if _global_taxonomy is None:
        _global_taxonomy = FailureTaxonomy()
    return _global_taxonomy
