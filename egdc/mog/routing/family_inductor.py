"""Automatic family induction from repeated solved programs.

Scans a collection of solved Mog programs and detects shared structural
patterns that could be promoted into reusable synthesis families.

This is the "pathway strengthening" mechanism: when the system keeps solving
problems with similar code structures, it notices the pattern and names it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class InducedPattern:
    name: str
    shared_structure: str
    member_functions: list[str]
    frequency: int


class FamilyInductor:
    """Detects recurring structural patterns in solved Mog programs."""

    def __init__(self):
        # Structural signatures we look for.
        self._structure_extractors: list[tuple[str, str]] = [
            ("for item in", r"for\s+\w+\s+in\s+\w+\s*\{"),
            ("while loop", r"while\s+.+\s*\{"),
            ("if return early", r"if\s*\(.+\)\s*\{\s*return\s"),
            ("match expression", r"match\s+\w+\s*\{"),
            ("recursive call", r"return\s+\w+\(.+\w+\s*-\s*1"),
            ("struct construction", r"\w+\s*\{\s*\w+:\s*"),
            ("accumulator pattern", r"\w+\s*=\s*\w+\s*\+\s"),
        ]

    def _extract_structures(self, code: str) -> list[str]:
        found = []
        for name, pattern in self._structure_extractors:
            if re.search(pattern, code):
                found.append(name)
        return found

    def detect_patterns(self, solved_codes: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Detect shared patterns across solved programs.

        Args:
            solved_codes: list of (function_name, code) pairs.

        Returns:
            List of detected patterns with shared structure info.
        """
        structure_to_functions: dict[str, list[str]] = {}
        for fn_name, code in solved_codes:
            structures = self._extract_structures(code)
            for s in structures:
                if s not in structure_to_functions:
                    structure_to_functions[s] = []
                structure_to_functions[s].append(fn_name)

        patterns = []
        for structure, functions in structure_to_functions.items():
            if len(functions) >= 2:
                patterns.append({
                    "name": f"induced_{structure.replace(' ', '_')}",
                    "shared_structure": structure,
                    "member_functions": functions,
                    "frequency": len(functions),
                })

        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        return patterns

    def suggest_new_families(self, solved_codes: list[tuple[str, str]], min_frequency: int = 3) -> list[dict[str, Any]]:
        """Suggest new family templates based on frequently recurring patterns."""
        patterns = self.detect_patterns(solved_codes)
        return [p for p in patterns if p["frequency"] >= min_frequency]
