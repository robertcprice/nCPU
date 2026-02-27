"""
MAP-Elites (V7 Novelty Discovery)
===================================
Reward finding NEW NICHES, not just the best solution.

MAP-Elites maintains a grid of solutions where each cell
represents a different "niche" (behavior characteristic).

This encourages DIVERSITY of solutions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib


@dataclass
class Niche:
    """A niche in the MAP-Elites archive."""
    niche_id: str
    dimensions: Dict[str, float]  # The behavior characteristics
    best_solution: str  # Code
    best_F: float  # Free energy (lower is better)
    best_fitness: float
    discovered_at: datetime
    updated_at: datetime
    discovery_count: int = 1  # How many times discovered


@dataclass
class ArchiveEntry:
    """Entry for a solution in the archive."""
    code: str
    fitness: float
    F: float  # Free energy
    dimensions: Dict[str, float]
    niche_id: str
    timestamp: datetime


class MAPElites:
    """
    MAP-Elites algorithm for novelty search.

    Key Idea: Maintain a GRID of solutions, one per behavioral niche.
    This encourages exploration of DIFFERENT solutions, not just
    converging to one "best" solution.

    Dimensions we use:
    - complexity: How complex is the code?
    - recursion: Does it use recursion?
    - data_structures: What data structures does it use?
    - algorithm_type: What algorithm pattern does it follow?
    """

    # Niche dimension definitions
    COMPLEXITY_BINS = [0.0, 0.25, 0.5, 0.75, 1.0]
    RECURSION_BINS = [0, 1]  # No recursion, Has recursion
    DATA_STRUCTURE_BINS = ["none", "list", "dict", "set", "multi"]
    ALGORITHM_BINS = ["iterative", "recursive", "divide_conquer", "dp", "greedy"]

    def __init__(self):
        self.archive: Dict[str, Niche] = {}
        self.history: List[ArchiveEntry] = []
        self._total_attempts = 0
        self._successful_additions = 0

    def characterize_solution(self, code: str) -> Dict[str, Any]:
        """
        Extract behavioral characteristics from code.

        These characteristics determine which niche the solution belongs to.
        """
        import re

        # Complexity (0-1)
        lines = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')])
        complexity = min(1.0, lines / 30)

        # Find complexity bin
        complexity_bin = 0
        for i, threshold in enumerate(self.COMPLEXITY_BINS[1:]):
            if complexity <= threshold:
                complexity_bin = i
                break

        # Recursion detection
        has_recursion = 1 if re.search(r'def\s+(\w+).*\1\s*\(', code, re.DOTALL) else 0

        # Data structure detection
        if 'set(' in code or '{' in code and ':' not in code:
            data_structure = "set"
        elif '{' in code and ':' in code:
            data_structure = "dict"
        elif '[' in code:
            data_structure = "list"
        else:
            data_structure = "none"

        # Check for multiple data structures
        ds_count = sum([
            '[' in code,
            '{' in code,
            'set(' in code,
        ])
        if ds_count >= 2:
            data_structure = "multi"

        # Algorithm type detection
        if has_recursion:
            if 'memo' in code or 'cache' in code:
                algorithm = "dp"
            else:
                algorithm = "recursive"
        elif 'for ' in code or 'while ' in code:
            if 'max(' in code or 'min(' in code:
                algorithm = "greedy"
            else:
                algorithm = "iterative"
        else:
            algorithm = "iterative"

        return {
            "complexity": complexity,
            "complexity_bin": complexity_bin,
            "recursion": has_recursion,
            "data_structure": data_structure,
            "algorithm": algorithm,
        }

    def compute_niche_id(self, characteristics: Dict[str, Any]) -> str:
        """Compute unique niche ID from characteristics."""
        key = f"{characteristics['complexity_bin']}_{characteristics['recursion']}_{characteristics['data_structure']}_{characteristics['algorithm']}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def try_add_solution(
        self,
        code: str,
        fitness: float,
        F: float
    ) -> Tuple[bool, str, bool]:
        """
        Try to add a solution to the archive.

        Returns: (added, niche_id, is_new_niche)
        """
        self._total_attempts += 1

        # Characterize the solution
        characteristics = self.characterize_solution(code)
        niche_id = self.compute_niche_id(characteristics)

        dimensions = {
            "complexity": characteristics["complexity"],
            "recursion": float(characteristics["recursion"]),
            "data_structure": characteristics["data_structure"],
            "algorithm": characteristics["algorithm"],
        }

        # Record in history
        entry = ArchiveEntry(
            code=code,
            fitness=fitness,
            F=F,
            dimensions=dimensions,
            niche_id=niche_id,
            timestamp=datetime.now(),
        )
        self.history.append(entry)

        # Keep history bounded
        self.history = self.history[-10000:]

        # Check if niche exists
        if niche_id not in self.archive:
            # New niche discovered!
            self.archive[niche_id] = Niche(
                niche_id=niche_id,
                dimensions=dimensions,
                best_solution=code,
                best_F=F,
                best_fitness=fitness,
                discovered_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self._successful_additions += 1
            return True, niche_id, True

        else:
            # Existing niche - check if this solution is better
            existing = self.archive[niche_id]
            existing.discovery_count += 1

            # Better if lower F (free energy)
            if F < existing.best_F:
                existing.best_solution = code
                existing.best_F = F
                existing.best_fitness = fitness
                existing.updated_at = datetime.now()
                self._successful_additions += 1
                return True, niche_id, False

            return False, niche_id, False

    def get_diverse_solutions(self, n: int = 10) -> List[Niche]:
        """Get n diverse solutions from different niches."""
        # Sort niches by fitness
        sorted_niches = sorted(
            self.archive.values(),
            key=lambda x: x.best_fitness,
            reverse=True
        )

        return sorted_niches[:n]

    def get_niche_coverage(self) -> Dict[str, Any]:
        """Get coverage statistics for each dimension."""
        coverage = {
            "complexity_bins": {},
            "recursion": {0: 0, 1: 0},
            "data_structures": {},
            "algorithms": {},
        }

        for niche in self.archive.values():
            # Complexity
            comp_bin = int(niche.dimensions.get("complexity", 0) * 4)
            coverage["complexity_bins"][comp_bin] = coverage["complexity_bins"].get(comp_bin, 0) + 1

            # Recursion
            rec = int(niche.dimensions.get("recursion", 0))
            coverage["recursion"][rec] = coverage["recursion"].get(rec, 0) + 1

            # Data structure
            ds = niche.dimensions.get("data_structure", "none")
            coverage["data_structures"][ds] = coverage["data_structures"].get(ds, 0) + 1

            # Algorithm
            alg = niche.dimensions.get("algorithm", "iterative")
            coverage["algorithms"][alg] = coverage["algorithms"].get(alg, 0) + 1

        return coverage

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        if not self.archive:
            return {
                "niches_discovered": 0,
                "total_attempts": self._total_attempts,
                "success_rate": 0.0,
            }

        return {
            "niches_discovered": len(self.archive),
            "total_attempts": self._total_attempts,
            "success_rate": self._successful_additions / max(self._total_attempts, 1),
            "best_fitness": max(n.best_fitness for n in self.archive.values()),
            "best_F": min(n.best_F for n in self.archive.values()),
            "avg_discovery_count": sum(n.discovery_count for n in self.archive.values()) / len(self.archive),
            "coverage": self.get_niche_coverage(),
        }

    def get_underexplored_niches(self) -> List[Dict[str, Any]]:
        """
        Identify niches that are underexplored.

        These are niches with few discoveries or low fitness.
        """
        underexplored = []

        for niche in self.archive.values():
            if niche.discovery_count <= 2 or niche.best_fitness < 0.5:
                underexplored.append({
                    "niche_id": niche.niche_id,
                    "dimensions": niche.dimensions,
                    "discovery_count": niche.discovery_count,
                    "best_fitness": niche.best_fitness,
                })

        return underexplored

    def suggest_exploration_direction(self) -> Dict[str, Any]:
        """
        Suggest a direction for exploration.

        Based on coverage analysis, suggest which niche types to explore.
        """
        coverage = self.get_niche_coverage()

        suggestions = []

        # Check complexity coverage
        for bin_idx in range(len(self.COMPLEXITY_BINS) - 1):
            if coverage["complexity_bins"].get(bin_idx, 0) == 0:
                suggestions.append({
                    "dimension": "complexity",
                    "suggestion": f"Try solutions with complexity in bin {bin_idx}",
                })

        # Check recursion coverage
        if coverage["recursion"].get(1, 0) == 0:
            suggestions.append({
                "dimension": "recursion",
                "suggestion": "Try recursive solutions",
            })

        # Check data structure coverage
        for ds in self.DATA_STRUCTURE_BINS:
            if coverage["data_structures"].get(ds, 0) == 0:
                suggestions.append({
                    "dimension": "data_structure",
                    "suggestion": f"Try solutions using {ds}",
                })

        # Check algorithm coverage
        for alg in self.ALGORITHM_BINS:
            if coverage["algorithms"].get(alg, 0) == 0:
                suggestions.append({
                    "dimension": "algorithm",
                    "suggestion": f"Try {alg} approach",
                })

        return {
            "suggestions": suggestions,
            "coverage": coverage,
        }
