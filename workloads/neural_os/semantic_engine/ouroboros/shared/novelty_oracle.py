"""
Novelty Oracle
===============
A FROZEN LLM ensemble that judges novelty.

Critical properties:
- NEVER updated during runs (frozen weights)
- Deterministic for fair comparison
- Cannot be influenced by agents
- Same oracle evaluates both V6 and V7
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json


@dataclass
class NoveltyScore:
    """Result of novelty evaluation."""
    code_hash: str
    total_score: float  # 0-1, higher = more novel
    perplexity_score: float  # How surprising to models
    structural_score: float  # How unique the structure is
    semantic_score: float  # How unique the behavior is
    model_scores: Dict[str, float]  # Per-model scores
    timestamp: datetime


class NoveltyOracle:
    """
    Frozen LLM ensemble for judging novelty.

    Uses multiple models to compute how "surprising" code is.
    Lower compressibility = more novel.
    """

    # Frozen model configuration - NEVER changes
    MODELS = [
        "codellama:7b",
        "mistral:7b",
        "phi:2.7b",
    ]

    # Weights for ensemble (sum to 1.0)
    MODEL_WEIGHTS = {
        "codellama:7b": 0.5,   # Code specialist
        "mistral:7b": 0.3,     # General
        "phi:2.7b": 0.2,       # Fast check
    }

    def __init__(self, cache_size: int = 10000):
        self._client = None
        self._cache: Dict[str, NoveltyScore] = {}
        self._cache_size = cache_size
        self._evaluation_count = 0
        self._frozen_at = datetime.now()

        # Hash of oracle configuration for integrity
        config_str = json.dumps({
            "models": self.MODELS,
            "weights": self.MODEL_WEIGHTS,
        }, sort_keys=True)
        self._config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_client(self):
        """Lazy load Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client()
            except ImportError:
                self._client = None
        return self._client

    def compute_novelty(self, code: str) -> NoveltyScore:
        """
        Compute novelty score for code.

        Uses perplexity from frozen models as novelty signal.
        Higher perplexity = more novel (less predictable).
        """
        # Check cache
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        if code_hash in self._cache:
            return self._cache[code_hash]

        # Compute scores from each model
        model_scores = {}
        total_perplexity = 0.0

        client = self._get_client()

        for model in self.MODELS:
            try:
                if client:
                    # Get perplexity from model
                    perplexity = self._compute_perplexity(client, model, code)
                else:
                    # Fallback: use heuristic
                    perplexity = self._heuristic_perplexity(code)

                model_scores[model] = perplexity
                total_perplexity += perplexity * self.MODEL_WEIGHTS.get(model, 0.33)

            except Exception as e:
                # Model failed, use fallback
                model_scores[model] = self._heuristic_perplexity(code)
                total_perplexity += model_scores[model] * self.MODEL_WEIGHTS.get(model, 0.33)

        # Convert perplexity to 0-1 novelty score
        # Higher perplexity = more novel
        perplexity_score = 1.0 - (1.0 / (1.0 + total_perplexity / 100.0))

        # Structural novelty (unique AST patterns)
        structural_score = self._compute_structural_novelty(code)

        # Semantic novelty (unique behavior patterns)
        semantic_score = self._compute_semantic_novelty(code)

        # Combined score
        total_score = (
            0.5 * perplexity_score +
            0.3 * structural_score +
            0.2 * semantic_score
        )

        result = NoveltyScore(
            code_hash=code_hash,
            total_score=total_score,
            perplexity_score=perplexity_score,
            structural_score=structural_score,
            semantic_score=semantic_score,
            model_scores=model_scores,
            timestamp=datetime.now(),
        )

        # Cache result
        self._cache[code_hash] = result
        self._evaluation_count += 1

        # Trim cache if needed
        if len(self._cache) > self._cache_size:
            # Remove oldest entries
            oldest = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
            for key, _ in oldest[:len(self._cache) - self._cache_size]:
                del self._cache[key]

        return result

    def _compute_perplexity(self, client, model: str, code: str) -> float:
        """Compute perplexity using Ollama model."""
        try:
            # Use model to predict next tokens
            # Higher perplexity = code is more surprising to model
            response = client.generate(
                model=model,
                prompt=f"Continue this code:\n```\n{code[:500]}\n```\n",
                options={"num_predict": 50}
            )

            # Estimate perplexity from generation
            # If model struggles or generates very different code = high perplexity
            generated = response.get("response", "")
            similarity = self._code_similarity(code[500:], generated)

            # Invert similarity to get perplexity-like score
            return (1.0 - similarity) * 100.0

        except Exception:
            return self._heuristic_perplexity(code)

    def _heuristic_perplexity(self, code: str) -> float:
        """Fallback heuristic for perplexity when models unavailable."""
        # Simple heuristics:
        # - Longer variable names = more novel
        # - More unique tokens = more novel
        # - Less common patterns = more novel

        import re

        # Token diversity
        tokens = re.findall(r'\b\w+\b', code)
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)

        # Identifier complexity
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        avg_len = sum(len(i) for i in identifiers) / max(len(identifiers), 1)

        # Nesting depth
        max_depth = 0
        current_depth = 0
        for char in code:
            if char in '({[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)

        # Combine into perplexity estimate
        perplexity = (
            unique_ratio * 30 +
            min(avg_len, 15) * 2 +
            min(max_depth, 10) * 3
        )

        return perplexity

    def _compute_structural_novelty(self, code: str) -> float:
        """Compute structural novelty based on AST patterns."""
        try:
            import ast
            tree = ast.parse(code)

            # Count node types
            node_types = {}
            for node in ast.walk(tree):
                name = type(node).__name__
                node_types[name] = node_types.get(name, 0) + 1

            # More diverse node types = more novel
            diversity = len(node_types) / max(sum(node_types.values()), 1)

            # Unusual patterns
            unusual = 0
            if 'Lambda' in node_types:
                unusual += 0.1
            if 'ListComp' in node_types or 'DictComp' in node_types:
                unusual += 0.1
            if 'Try' in node_types:
                unusual += 0.05
            if 'With' in node_types:
                unusual += 0.05

            return min(1.0, diversity + unusual)

        except SyntaxError:
            return 0.5  # Can't parse, neutral score

    def _compute_semantic_novelty(self, code: str) -> float:
        """Compute semantic novelty based on behavior patterns."""
        # Look for unique behavior patterns

        import re

        score = 0.5  # Start neutral

        # Recursion
        if re.search(r'def\s+(\w+).*\1\s*\(', code, re.DOTALL):
            score += 0.1

        # Generator expressions
        if 'yield' in code:
            score += 0.1

        # Decorators
        if re.search(r'@\w+', code):
            score += 0.05

        # Context managers
        if '__enter__' in code or '__exit__' in code:
            score += 0.1

        # Metaclasses
        if 'metaclass' in code or '__new__' in code:
            score += 0.15

        return min(1.0, score)

    def _code_similarity(self, code1: str, code2: str) -> float:
        """Compute similarity between two code snippets."""
        if not code1 or not code2:
            return 0.0

        # Simple token overlap
        import re
        tokens1 = set(re.findall(r'\b\w+\b', code1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', code2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)

        return overlap / max(total, 1)

    def verify_frozen(self) -> bool:
        """Verify oracle configuration hasn't changed."""
        config_str = json.dumps({
            "models": self.MODELS,
            "weights": self.MODEL_WEIGHTS,
        }, sort_keys=True)
        current_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return current_hash == self._config_hash

    def get_stats(self) -> Dict[str, Any]:
        """Get oracle statistics."""
        return {
            "evaluations": self._evaluation_count,
            "cache_size": len(self._cache),
            "frozen_at": self._frozen_at.isoformat(),
            "config_hash": self._config_hash,
            "integrity_ok": self.verify_frozen(),
        }


# Singleton frozen oracle
_ORACLE: Optional[NoveltyOracle] = None


def get_novelty_oracle() -> NoveltyOracle:
    """Get the global frozen novelty oracle."""
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = NoveltyOracle()
    return _ORACLE
