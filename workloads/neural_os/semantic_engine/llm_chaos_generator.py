#!/usr/bin/env python3
"""
LLM-Powered Chaos Generator

Uses multiple LLMs (Claude, GPT-4, DeepSeek) to generate code mutations.
The chaos feeds the ratchet filter.

This is the "exploration" half of the Chaos-Ratchet Engine.
"""

import os
import sys
import time
import random
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import hashlib

# Add path for hybrid reviewer APIs
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-llm-compiler/staged_classifier')

try:
    import anthropic
    import openai
    APIS_AVAILABLE = True
except ImportError:
    APIS_AVAILABLE = False
    print("Warning: API libraries not available, using fallback mutations")


# =============================================================================
# CODE CANDIDATE
# =============================================================================

@dataclass
class CodeMutation:
    """A mutated code candidate."""
    name: str
    code: str
    description: str
    mutation_type: str
    source_llm: str
    generation: int = 0
    parent: Optional[str] = None
    proof_hash: str = ""

    def __post_init__(self):
        if not self.proof_hash:
            self.proof_hash = hashlib.sha256(
                f"{self.name}:{self.code}:{time.time()}".encode()
            ).hexdigest()[:16]

    def compile(self) -> Optional[Callable]:
        """Compile and return the function."""
        try:
            namespace = {}
            exec(self.code, namespace)
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith('_'):
                    return obj
            return None
        except:
            return None


# =============================================================================
# LLM CHAOS GENERATOR
# =============================================================================

class LLMChaosGenerator:
    """
    Uses LLMs to generate intelligent code mutations.

    Much more powerful than random mutations - LLMs understand:
    - Algorithm patterns
    - Optimization techniques
    - Code idioms
    - Performance best practices
    """

    def __init__(self, use_apis: bool = True):
        self.use_apis = use_apis and APIS_AVAILABLE
        self.mutation_count = 0
        self.successful_patterns = []  # Learn from successes

        # API keys from hybrid reviewer
        if self.use_apis:
            self.openai_key = os.getenv("OPENAI_API_KEY")
            self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    def generate_mutations(self, code: str, task: str = "optimize for speed") -> List[CodeMutation]:
        """Generate multiple mutation candidates using LLMs."""
        mutations = []

        if self.use_apis:
            # Try Claude
            claude_mutation = self._call_claude(code, task)
            if claude_mutation:
                mutations.append(claude_mutation)

            # Try GPT-4
            gpt_mutation = self._call_gpt4(code, task)
            if gpt_mutation:
                mutations.append(gpt_mutation)

        # Always include fallback mutations (fast, no API needed)
        mutations.extend(self._generate_fallback_mutations(code))

        return mutations

    def _call_claude(self, code: str, task: str) -> Optional[CodeMutation]:
        """Call Claude to generate a mutation."""
        try:
            client = anthropic.Anthropic(api_key=self.anthropic_key)

            prompt = f"""You are a code optimization expert. Given this Python code:

```python
{code}
```

Task: {task}

Generate an IMPROVED version that is faster/better. Return ONLY the improved Python code, no explanation.
The function must:
1. Take the same input (a list)
2. Return the same output (sorted list)
3. Be named 'optimized_fn'
4. Be demonstrably faster or use a better algorithm

Return only valid Python code:"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text
            # Extract code from response
            if '```python' in result:
                result = result.split('```python')[1].split('```')[0]
            elif '```' in result:
                result = result.split('```')[1].split('```')[0]

            result = result.strip()

            self.mutation_count += 1
            return CodeMutation(
                name=f"claude_optimized_{self.mutation_count}",
                code=result,
                description="Claude-generated optimization",
                mutation_type="llm_optimization",
                source_llm="claude",
            )
        except Exception as e:
            print(f"Claude API error: {e}")
            return None

    def _call_gpt4(self, code: str, task: str) -> Optional[CodeMutation]:
        """Call GPT-4 to generate a mutation."""
        try:
            client = openai.OpenAI(api_key=self.openai_key)

            prompt = f"""You are a code optimization expert. Given this Python code:

```python
{code}
```

Task: {task}

Generate an IMPROVED version that is faster/better. Return ONLY the improved Python code.
The function must:
1. Take the same input (a list)
2. Return the same output (sorted list)
3. Be named 'optimized_fn'
4. Be demonstrably faster

Return only valid Python code, no markdown:"""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )

            result = response.choices[0].message.content
            # Extract code
            if '```python' in result:
                result = result.split('```python')[1].split('```')[0]
            elif '```' in result:
                result = result.split('```')[1].split('```')[0]

            result = result.strip()

            self.mutation_count += 1
            return CodeMutation(
                name=f"gpt4_optimized_{self.mutation_count}",
                code=result,
                description="GPT-4 generated optimization",
                mutation_type="llm_optimization",
                source_llm="gpt4",
            )
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            return None

    def _generate_fallback_mutations(self, code: str) -> List[CodeMutation]:
        """Generate mutations without API calls (fast fallback)."""
        mutations = []

        # Strategy 1: Use built-in sort (often fastest)
        if 'sorted' not in code or 'return sorted' not in code:
            self.mutation_count += 1
            mutations.append(CodeMutation(
                name=f"builtin_sort_{self.mutation_count}",
                code="def optimized_fn(arr):\n    return sorted(arr)",
                description="Use Python's optimized built-in sort",
                mutation_type="use_builtin",
                source_llm="fallback",
            ))

        # Strategy 2: Tim sort with key optimization
        self.mutation_count += 1
        mutations.append(CodeMutation(
            name=f"timsort_optimized_{self.mutation_count}",
            code=textwrap.dedent('''
            def optimized_fn(arr):
                """Tim sort - Python's built-in, highly optimized."""
                result = list(arr)
                result.sort()
                return result
            ''').strip(),
            description="Tim sort (in-place, cache-friendly)",
            mutation_type="algorithm_switch",
            source_llm="fallback",
        ))

        # Strategy 3: Quick sort implementation
        self.mutation_count += 1
        mutations.append(CodeMutation(
            name=f"quicksort_{self.mutation_count}",
            code=textwrap.dedent('''
            def optimized_fn(arr):
                """Quick sort - O(n log n) average."""
                if len(arr) <= 1:
                    return list(arr)
                pivot = arr[len(arr) // 2]
                left = [x for x in arr if x < pivot]
                middle = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return optimized_fn(left) + middle + optimized_fn(right)
            ''').strip(),
            description="Quick sort with middle pivot",
            mutation_type="algorithm_switch",
            source_llm="fallback",
        ))

        # Strategy 4: Merge sort (stable, O(n log n))
        self.mutation_count += 1
        mutations.append(CodeMutation(
            name=f"mergesort_{self.mutation_count}",
            code=textwrap.dedent('''
            def optimized_fn(arr):
                """Merge sort - O(n log n) stable."""
                if len(arr) <= 1:
                    return list(arr)
                mid = len(arr) // 2
                left = optimized_fn(arr[:mid])
                right = optimized_fn(arr[mid:])
                result = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        result.append(left[i])
                        i += 1
                    else:
                        result.append(right[j])
                        j += 1
                result.extend(left[i:])
                result.extend(right[j:])
                return result
            ''').strip(),
            description="Merge sort - stable O(n log n)",
            mutation_type="algorithm_switch",
            source_llm="fallback",
        ))

        # Strategy 5: Insertion sort (good for small/nearly sorted)
        self.mutation_count += 1
        mutations.append(CodeMutation(
            name=f"insertion_{self.mutation_count}",
            code=textwrap.dedent('''
            def optimized_fn(arr):
                """Insertion sort - good for small arrays."""
                arr = list(arr)
                for i in range(1, len(arr)):
                    key = arr[i]
                    j = i - 1
                    while j >= 0 and arr[j] > key:
                        arr[j + 1] = arr[j]
                        j -= 1
                    arr[j + 1] = key
                return arr
            ''').strip(),
            description="Insertion sort",
            mutation_type="algorithm_switch",
            source_llm="fallback",
        ))

        # Strategy 6: Hybrid (insertion for small, merge for large)
        self.mutation_count += 1
        mutations.append(CodeMutation(
            name=f"hybrid_{self.mutation_count}",
            code=textwrap.dedent('''
            def optimized_fn(arr):
                """Hybrid sort - insertion for small, merge for large."""
                def insertion_sort(arr):
                    arr = list(arr)
                    for i in range(1, len(arr)):
                        key = arr[i]
                        j = i - 1
                        while j >= 0 and arr[j] > key:
                            arr[j + 1] = arr[j]
                            j -= 1
                        arr[j + 1] = key
                    return arr

                if len(arr) <= 16:
                    return insertion_sort(arr)

                mid = len(arr) // 2
                left = optimized_fn(arr[:mid])
                right = optimized_fn(arr[mid:])
                result = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        result.append(left[i])
                        i += 1
                    else:
                        result.append(right[j])
                        j += 1
                result.extend(left[i:])
                result.extend(right[j:])
                return result
            ''').strip(),
            description="Hybrid insertion+merge sort",
            mutation_type="hybrid_algorithm",
            source_llm="fallback",
        ))

        return mutations

    def record_success(self, mutation: CodeMutation, speedup: float):
        """Learn from successful mutations."""
        self.successful_patterns.append({
            'type': mutation.mutation_type,
            'source': mutation.source_llm,
            'speedup': speedup,
        })


# =============================================================================
# DEMO
# =============================================================================

def demo_llm_chaos():
    """Demonstrate the LLM chaos generator."""
    print("=" * 70)
    print("LLM CHAOS GENERATOR DEMO")
    print("=" * 70)
    print()

    generator = LLMChaosGenerator(use_apis=False)  # Fallback mode for demo

    # Starting code
    bubble_sort = textwrap.dedent('''
    def sort_fn(arr):
        arr = list(arr)
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    ''').strip()

    print("Starting code (bubble sort):")
    print(bubble_sort)
    print()

    print("Generating mutations...")
    mutations = generator.generate_mutations(bubble_sort, "optimize for speed")

    print(f"\nGenerated {len(mutations)} mutations:")
    for m in mutations:
        print(f"\n--- {m.name} ({m.source_llm}) ---")
        print(f"Type: {m.mutation_type}")
        print(f"Description: {m.description}")
        print(f"Code preview: {m.code[:100]}...")

    return mutations


if __name__ == '__main__':
    demo_llm_chaos()
