"""Automatic sub-program extraction from solved programs.

When multiple solved programs share a common code pattern, extract it as
a reusable sub-program. This is how the system's library grows automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SharedFragment:
    code: str
    found_in: list[str]
    frequency: int


class SubProgramExtractor:
    def find_shared_fragments(self, solved: dict[str, str], min_length: int = 30) -> list[SharedFragment]:
        """Find code fragments shared across multiple solved programs.

        Uses longest common substring detection between pairs of programs,
        then groups by frequency.
        """
        fragments: dict[str, list[str]] = {}

        names = list(solved.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                code_a = self._normalize(solved[names[i]])
                code_b = self._normalize(solved[names[j]])
                shared = self._longest_common_substrings(code_a, code_b, min_length)
                for s in shared:
                    s_stripped = s.strip()
                    if len(s_stripped) < min_length:
                        continue
                    # Deduplicate by normalized form
                    key = s_stripped
                    if key not in fragments:
                        fragments[key] = []
                    if names[i] not in fragments[key]:
                        fragments[key].append(names[i])
                    if names[j] not in fragments[key]:
                        fragments[key].append(names[j])

        result = []
        for code, found_in in fragments.items():
            if len(found_in) >= 2:
                result.append(SharedFragment(code, found_in, len(found_in)))
        result.sort(key=lambda x: (-x.frequency, -len(x.code)))
        return result

    def _normalize(self, code: str) -> str:
        # Strip function wrapper, keep body
        body = re.sub(r"fn\s+\w+\([^)]*\)\s*->\s*\w+\s*\{", "", code)
        body = body.rstrip().rstrip("}")
        return body.strip()

    def _longest_common_substrings(self, a: str, b: str, min_len: int) -> list[str]:
        """Find all common substrings of length >= min_len."""
        results = []
        m, n = len(a), len(b)
        # Dynamic programming table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] >= min_len:
                        substr = a[i - dp[i][j]:i]
                        if substr not in results:
                            results.append(substr)
        return results
