#!/usr/bin/env python3
"""
OLLAMA MUTATOR - LLM-Powered Code Mutation

Uses Ollama's local LLM to generate intelligent code mutations.
Key features:
- Async HTTP client for non-blocking requests
- Prompt engineering for code optimization
- Repair loop: Feed syntax errors back to LLM for fixing
- Fallback to AST mutations when LLM unavailable

Prompt Strategy:
1. Provide the original code
2. Explain the optimization goal (speed, readability, etc.)
3. Request ONLY code output (no explanations)
4. Parse and validate the response
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
import asyncio
import aiohttp
import json
import time
import logging
import re

logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """Types of optimizations to request from LLM."""
    SPEED = auto()           # Make it faster
    READABILITY = auto()     # Make it cleaner
    MEMORY = auto()          # Use less memory
    SIMPLICITY = auto()      # Reduce complexity
    CORRECTNESS = auto()     # Fix potential bugs
    ALGORITHMIC = auto()     # Improve algorithm (e.g., O(n²) → O(n log n))


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    # Connection
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 60

    # Models by complexity
    simple_model: str = "codellama:7b"
    balanced_model: str = "deepseek-coder:6.7b"
    advanced_model: str = "mistral:7b"

    # Generation settings
    temperature: float = 0.3  # Low for deterministic code generation
    max_tokens: int = 2048
    top_p: float = 0.9

    # Retry settings
    max_retries: int = 3
    repair_attempts: int = 2  # Number of times to try fixing syntax errors


@dataclass
class MutationRequest:
    """Request for LLM mutation."""
    source_code: str
    goal: OptimizationGoal = OptimizationGoal.SPEED
    context: str = ""  # Additional context about the code
    model: Optional[str] = None  # Override model selection
    constraints: List[str] = field(default_factory=list)  # e.g., "preserve function signature"


@dataclass
class MutationResponse:
    """Response from LLM mutation."""
    success: bool
    original_code: str
    mutated_code: str
    model_used: str = ""
    generation_time: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    reasoning: str = ""  # LLM's explanation (if provided)
    attempts: int = 1


class OllamaMutator:
    """
    LLM-powered code mutator using Ollama.

    Usage:
        mutator = OllamaMutator(OllamaConfig())

        # Check if Ollama is available
        if await mutator.is_available():
            # Request a mutation
            request = MutationRequest(
                source_code="def sort(arr): ...",
                goal=OptimizationGoal.SPEED,
            )
            response = await mutator.mutate(request)

            if response.success:
                print(f"Mutated code: {response.mutated_code}")
    """

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._mutation_history: List[MutationResponse] = []
        self._available: Optional[bool] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as resp:
                self._available = resp.status == 200
                return self._available
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False

    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    def _build_prompt(self, request: MutationRequest) -> str:
        """Build the prompt for the LLM."""

        goal_prompts = {
            OptimizationGoal.SPEED: "Optimize this code for maximum speed. Focus on reducing time complexity and unnecessary operations.",
            OptimizationGoal.READABILITY: "Refactor this code for better readability. Use clear variable names and simplify logic.",
            OptimizationGoal.MEMORY: "Optimize this code for memory efficiency. Reduce allocations and use generators where possible.",
            OptimizationGoal.SIMPLICITY: "Simplify this code. Remove unnecessary complexity while maintaining functionality.",
            OptimizationGoal.CORRECTNESS: "Review this code for bugs and fix any issues. Ensure edge cases are handled.",
            OptimizationGoal.ALGORITHMIC: "Improve the algorithm. If possible, reduce time complexity (e.g., O(n²) → O(n log n)).",
        }

        goal_text = goal_prompts.get(request.goal, goal_prompts[OptimizationGoal.SPEED])

        constraints_text = ""
        if request.constraints:
            constraints_text = "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in request.constraints)

        context_text = ""
        if request.context:
            context_text = f"\n\nContext: {request.context}"

        prompt = f"""You are a code optimization expert. Your task is to improve the following Python code.

{goal_text}{constraints_text}{context_text}

IMPORTANT: Return ONLY the improved Python code. No explanations, no markdown, no comments about what you changed. Just the code.

Original code:
```python
{request.source_code}
```

Improved code:"""

        return prompt

    def _extract_code(self, response_text: str) -> str:
        """Extract code from LLM response."""
        text = response_text.strip()

        # Try to extract from markdown code blocks
        code_block_pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try to find code between def/class and end
        if 'def ' in text or 'class ' in text:
            # Find first function/class definition
            lines = text.split('\n')
            code_lines = []
            in_code = False

            for line in lines:
                if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                    in_code = True
                if in_code:
                    # Stop at explanation text
                    if line.strip().startswith(('#', 'This ', 'The ', 'Note:', 'I ')):
                        if not line.strip().startswith('# '):  # Keep comments
                            break
                    code_lines.append(line)

            if code_lines:
                return '\n'.join(code_lines).strip()

        # Just return the text if no patterns match
        return text

    def _build_repair_prompt(self, code: str, error: str) -> str:
        """Build prompt to repair syntax error."""
        return f"""The following Python code has a syntax error. Fix the error and return only the corrected code.

Error: {error}

Code:
```python
{code}
```

Fixed code:"""

    async def _call_ollama(
        self,
        prompt: str,
        model: str,
    ) -> Tuple[bool, str, int]:
        """
        Make a call to Ollama API.

        Returns: (success, response_text, tokens_used)
        """
        try:
            session = await self._get_session()

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                },
            }

            async with session.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return False, f"HTTP {resp.status}: {error_text}", 0

                data = await resp.json()
                response_text = data.get("response", "")
                tokens = data.get("eval_count", 0)

                return True, response_text, tokens

        except asyncio.TimeoutError:
            return False, "Request timed out", 0
        except Exception as e:
            return False, str(e), 0

    async def mutate(self, request: MutationRequest) -> MutationResponse:
        """
        Generate a code mutation using the LLM.

        This:
        1. Builds an optimization prompt
        2. Calls Ollama
        3. Extracts code from response
        4. Validates syntax
        5. Attempts repair if syntax is invalid
        """
        start_time = time.time()

        # Check availability
        if not await self.is_available():
            return MutationResponse(
                success=False,
                original_code=request.source_code,
                mutated_code=request.source_code,
                error="Ollama not available",
            )

        # Select model
        model = request.model or self.config.balanced_model

        # Build prompt
        prompt = self._build_prompt(request)

        # Call Ollama
        attempts = 0
        current_code = request.source_code
        total_tokens = 0

        for attempt in range(self.config.max_retries):
            attempts += 1

            success, response_text, tokens = await self._call_ollama(prompt, model)
            total_tokens += tokens

            if not success:
                logger.warning(f"Ollama call failed (attempt {attempt + 1}): {response_text}")
                continue

            # Extract code
            extracted_code = self._extract_code(response_text)

            if not extracted_code or extracted_code == request.source_code:
                continue

            # Validate syntax
            try:
                import ast
                ast.parse(extracted_code)

                # Success!
                response = MutationResponse(
                    success=True,
                    original_code=request.source_code,
                    mutated_code=extracted_code,
                    model_used=model,
                    generation_time=time.time() - start_time,
                    tokens_used=total_tokens,
                    attempts=attempts,
                )
                self._mutation_history.append(response)
                return response

            except SyntaxError as e:
                # Try to repair
                logger.debug(f"Syntax error in generated code: {e}")

                for repair_attempt in range(self.config.repair_attempts):
                    repair_prompt = self._build_repair_prompt(extracted_code, str(e))
                    repair_success, repair_text, repair_tokens = await self._call_ollama(repair_prompt, model)
                    total_tokens += repair_tokens

                    if repair_success:
                        repaired_code = self._extract_code(repair_text)
                        try:
                            import ast
                            ast.parse(repaired_code)

                            response = MutationResponse(
                                success=True,
                                original_code=request.source_code,
                                mutated_code=repaired_code,
                                model_used=model,
                                generation_time=time.time() - start_time,
                                tokens_used=total_tokens,
                                attempts=attempts,
                            )
                            self._mutation_history.append(response)
                            return response
                        except SyntaxError:
                            continue

        # All attempts failed
        return MutationResponse(
            success=False,
            original_code=request.source_code,
            mutated_code=request.source_code,
            model_used=model,
            generation_time=time.time() - start_time,
            tokens_used=total_tokens,
            error="Failed to generate valid code after all attempts",
            attempts=attempts,
        )

    async def batch_mutate(
        self,
        requests: List[MutationRequest],
        concurrency: int = 3,
    ) -> List[MutationResponse]:
        """
        Process multiple mutation requests concurrently.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_mutate(request: MutationRequest) -> MutationResponse:
            async with semaphore:
                return await self.mutate(request)

        return await asyncio.gather(*[bounded_mutate(r) for r in requests])

    def get_stats(self) -> Dict[str, Any]:
        """Get mutation statistics."""
        if not self._mutation_history:
            return {'total': 0, 'success_rate': 0}

        successes = sum(1 for r in self._mutation_history if r.success)
        total_tokens = sum(r.tokens_used for r in self._mutation_history)
        total_time = sum(r.generation_time for r in self._mutation_history)

        return {
            'total': len(self._mutation_history),
            'successes': successes,
            'success_rate': successes / len(self._mutation_history),
            'total_tokens': total_tokens,
            'total_time': total_time,
            'avg_time': total_time / len(self._mutation_history),
            'available': self._available,
        }


# Synchronous wrapper for use in non-async contexts
class SyncOllamaMutator:
    """Synchronous wrapper around OllamaMutator."""

    def __init__(self, config: OllamaConfig):
        self.config = config
        self._async_mutator = OllamaMutator(config)

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return asyncio.run(self._async_mutator.is_available())

    def mutate(self, request: MutationRequest) -> MutationResponse:
        """Perform a mutation synchronously."""
        return asyncio.run(self._async_mutator.mutate(request))

    def close(self) -> None:
        """Close the mutator."""
        asyncio.run(self._async_mutator.close())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self._async_mutator.get_stats()
