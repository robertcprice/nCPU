"""
AgentKVRM - OUROBOROS Agent as a KVRM Organism
================================================
An agent with an LLM brain that lives in the KVRM ecosystem.

Unlike traditional KVRMs with small neural networks, AgentKVRM uses
an Ollama LLM for its "think" step - enabling reasoning, creativity,
and genuine problem-solving.

Sense-Think-Act Cycle:
- SENSE: Read current problem, other agents' solutions, narrator guidance
- THINK: LLM reasons about how to improve/solve
- ACT: Write solution, hypotheses, art to shared memory
"""

import sys
import os
import json
import hashlib
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import subprocess

# Add KVRM ecosystem to path
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')

from core.shared_memory import SharedKVMemory, KVEntry
from core.kvrm_base import KVRMBase, KVRMConfig

import torch
import torch.nn as nn


@dataclass
class AgentConfig(KVRMConfig):
    """Configuration for an OUROBOROS agent."""
    mode: str = "competitive"  # "competitive" or "cooperative"
    llm_model: str = "llama3.1:8b"  # Ollama model
    max_tokens: int = 500
    temperature: float = 0.7
    energy_budget: float = 100.0  # Compute budget
    problem_key: str = "problem:current"  # Key to read problem from


class OllamaLLM:
    """Wrapper for Ollama LLM inference."""

    def __init__(self, model: str = "qwen3:8b", timeout: int = 30):
        self.model = model
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, timeout=5
            )
            self._available = result.returncode == 0
        except:
            self._available = False
        return self._available

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text using Ollama."""
        if not self.is_available():
            return "[LLM unavailable - using heuristic response]"

        try:
            # Use ollama CLI for simplicity
            cmd = [
                "ollama", "run", self.model,
                "--nowordwrap",
                prompt
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"[LLM error: {result.stderr[:100]}]"

        except subprocess.TimeoutExpired:
            return "[LLM timeout]"
        except Exception as e:
            return f"[LLM exception: {str(e)[:50]}]"


class DummyModel(nn.Module):
    """Placeholder model - actual inference uses LLM."""
    def forward(self, x):
        return {"placeholder": x}


class AgentKVRM(KVRMBase):
    """
    An OUROBOROS agent implemented as a KVRM organism.

    Uses Ollama LLM for reasoning instead of a small neural network.
    Communicates through SharedKVMemory with other agents.

    Memory Keys Read:
    - problem:current - The current problem to solve
    - solution:* - Other agents' solutions (for inspiration/competition)
    - narrator:guidance - Meta-narrator's advice
    - meta:patterns - Meta-learning insights

    Memory Keys Written:
    - solution:{agent_id} - This agent's solution
    - hypothesis:{agent_id} - Shared discoveries (cooperative mode)
    - observation:{agent_id} - What the agent noticed
    """

    def __init__(self, config: AgentConfig):
        # Create dummy model (we use LLM instead)
        dummy = DummyModel()
        super().__init__(config, model=dummy)

        self.agent_config = config
        self.llm = OllamaLLM(config.llm_model)

        # Agent state
        self.current_solution = ""
        self.fitness = 0.0
        self.energy = config.energy_budget
        self.generation = 0
        self.improvements = 0
        self.history: List[Dict] = []

        # Compute tracking
        self.tokens_used = 0
        self.inference_time = 0.0

    def get_input_schema(self) -> Dict[str, type]:
        """What this agent reads from memory."""
        return {
            self.agent_config.problem_key: dict,  # Current problem
            "solution:*": dict,  # Other solutions
            "narrator:guidance": str,  # Narrator advice
            "meta:patterns": dict,  # Meta-learning insights
            "meta:best_strategies": list,  # What worked before
        }

    def get_output_schema(self) -> Dict[str, type]:
        """What this agent writes to memory."""
        return {
            f"solution:{self.name}": dict,
            f"hypothesis:{self.name}": dict,
            f"observation:{self.name}": str,
            f"status:{self.name}": dict,
        }

    def encode(self, inputs: Dict[str, List[KVEntry]]) -> torch.Tensor:
        """
        Encode perceived inputs.

        For LLM-based agents, we don't actually encode to tensor.
        The real processing happens in think().
        """
        # Store inputs for think() to use
        self._current_inputs = inputs
        return torch.zeros(1)  # Placeholder

    def decode(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Decode model outputs.

        For LLM-based agents, the actual outputs are computed in think().
        """
        return self._current_outputs

    def think(self, inputs: Dict[str, List[KVEntry]]) -> Dict[str, Any]:
        """
        THINK: Use LLM to reason about the problem.

        This is where the real intelligence happens.
        """
        start_time = time.time()

        # Extract relevant data from inputs
        problem = self._get_problem(inputs)
        other_solutions = self._get_other_solutions(inputs)
        narrator_guidance = self._get_narrator_guidance(inputs)
        meta_patterns = self._get_meta_patterns(inputs)

        # Build prompt based on mode
        if self.agent_config.mode == "competitive":
            prompt = self._build_competitive_prompt(
                problem, other_solutions, narrator_guidance, meta_patterns
            )
        else:
            prompt = self._build_cooperative_prompt(
                problem, other_solutions, narrator_guidance, meta_patterns
            )

        # Call LLM
        llm_response = self.llm.generate(
            prompt,
            max_tokens=self.agent_config.max_tokens,
            temperature=self.agent_config.temperature
        )

        # Track compute usage
        self.inference_time += time.time() - start_time
        self.tokens_used += len(llm_response.split())
        self.energy -= len(llm_response.split()) * 0.1  # Token cost

        # Parse response
        solution, hypothesis, observation = self._parse_response(llm_response)

        # Evaluate solution
        new_fitness = self._evaluate_solution(solution, problem)

        # Update state if improved
        if new_fitness > self.fitness:
            self.current_solution = solution
            self.fitness = new_fitness
            self.improvements += 1

        # Record in history
        self.history.append({
            "generation": self.generation,
            "fitness": self.fitness,
            "solution_hash": hashlib.md5(solution.encode()).hexdigest()[:8],
            "improved": new_fitness > self.fitness,
        })

        self.generation += 1

        # Store outputs for decode()
        self._current_outputs = {
            f"solution:{self.name}": {
                "code": solution,
                "fitness": self.fitness,
                "generation": self.generation,
                "mode": self.agent_config.mode,
            },
            f"observation:{self.name}": observation,
            f"status:{self.name}": {
                "energy": self.energy,
                "tokens_used": self.tokens_used,
                "improvements": self.improvements,
                "fitness": self.fitness,
            },
        }

        # Add hypothesis for cooperative mode
        if self.agent_config.mode == "cooperative" and hypothesis:
            self._current_outputs[f"hypothesis:{self.name}"] = {
                "title": hypothesis[:50],
                "content": hypothesis,
                "fitness": self.fitness,
            }

        return self._current_outputs

    def _get_problem(self, inputs: Dict[str, List[KVEntry]]) -> str:
        """Extract current problem from inputs."""
        problem_entries = inputs.get(self.agent_config.problem_key, [])
        if problem_entries:
            return str(problem_entries[0].value)
        return "No problem defined"

    def _get_other_solutions(self, inputs: Dict[str, List[KVEntry]]) -> List[Dict]:
        """Extract other agents' solutions."""
        solutions = []
        for entry in inputs.get("solution:*", []):
            if entry.source_kvrm != self.name:  # Don't include own solution
                solutions.append({
                    "agent": entry.source_kvrm,
                    "fitness": entry.value.get("fitness", 0),
                    "code": entry.value.get("code", "")[:200],  # Truncate
                })
        return solutions

    def _get_narrator_guidance(self, inputs: Dict[str, List[KVEntry]]) -> str:
        """Extract narrator guidance."""
        guidance_entries = inputs.get("narrator:guidance", [])
        if guidance_entries:
            return str(guidance_entries[0].value)
        return ""

    def _get_meta_patterns(self, inputs: Dict[str, List[KVEntry]]) -> Dict:
        """Extract meta-learning patterns."""
        patterns = inputs.get("meta:patterns", [])
        if patterns:
            return patterns[0].value
        return {}

    def _build_competitive_prompt(
        self,
        problem: str,
        others: List[Dict],
        guidance: str,
        meta: Dict
    ) -> str:
        """Build prompt for competitive mode."""
        prompt = f"""You are Agent {self.name} in a competitive evolution system.

PROBLEM TO SOLVE:
{problem}

YOUR CURRENT SOLUTION (fitness={self.fitness:.2f}):
{self.current_solution[:300] if self.current_solution else "None yet"}

"""
        if others:
            prompt += "COMPETING SOLUTIONS:\n"
            for s in others[:3]:
                prompt += f"- {s['agent']}: fitness={s['fitness']:.2f}\n"

        if guidance:
            prompt += f"\nNARRATOR GUIDANCE:\n{guidance}\n"

        if meta.get("best_strategies"):
            prompt += f"\nWHAT WORKS (from meta-learning):\n{meta['best_strategies'][:200]}\n"

        prompt += """
YOUR TASK:
1. Analyze the problem and your current solution
2. Think of an improvement or new approach
3. Output your reasoning and new solution

FORMAT:
REASONING: [your thought process]
SOLUTION: [your code/solution]
OBSERVATION: [what you noticed about the problem or other agents]
"""
        return prompt

    def _build_cooperative_prompt(
        self,
        problem: str,
        others: List[Dict],
        guidance: str,
        meta: Dict
    ) -> str:
        """Build prompt for cooperative mode."""
        prompt = f"""You are Agent {self.name} in a cooperative evolution system.
You work WITH other agents, sharing discoveries on the blackboard.

PROBLEM TO SOLVE:
{problem}

YOUR CURRENT SOLUTION (fitness={self.fitness:.2f}):
{self.current_solution[:300] if self.current_solution else "None yet"}

"""
        if others:
            prompt += "TEAMMATE SOLUTIONS (learn from these):\n"
            for s in others[:3]:
                prompt += f"- {s['agent']}: fitness={s['fitness']:.2f}\n  {s['code'][:100]}...\n"

        if guidance:
            prompt += f"\nCONSENSUS DIRECTION:\n{guidance}\n"

        if meta.get("shared_discoveries"):
            prompt += f"\nSHARED DISCOVERIES:\n{meta['shared_discoveries'][:200]}\n"

        prompt += """
YOUR TASK:
1. Build on your teammates' work
2. Share any insights you discover
3. Propose improvements that help everyone

FORMAT:
REASONING: [your thought process]
SOLUTION: [your code/solution]
HYPOTHESIS: [insight to share with team]
OBSERVATION: [patterns you noticed]
"""
        return prompt

    def _parse_response(self, response: str) -> tuple:
        """Parse LLM response into solution, hypothesis, observation."""
        solution = ""
        hypothesis = ""
        observation = ""

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line_upper = line.upper()
            if "SOLUTION:" in line_upper:
                current_section = "solution"
                solution = line.split(":", 1)[-1].strip() if ":" in line else ""
            elif "HYPOTHESIS:" in line_upper:
                current_section = "hypothesis"
                hypothesis = line.split(":", 1)[-1].strip() if ":" in line else ""
            elif "OBSERVATION:" in line_upper:
                current_section = "observation"
                observation = line.split(":", 1)[-1].strip() if ":" in line else ""
            elif "REASONING:" in line_upper:
                current_section = "reasoning"
            elif current_section == "solution":
                solution += "\n" + line
            elif current_section == "hypothesis":
                hypothesis += " " + line
            elif current_section == "observation":
                observation += " " + line

        return solution.strip(), hypothesis.strip(), observation.strip()

    def _evaluate_solution(self, solution: str, problem: str) -> float:
        """
        Evaluate solution fitness.

        For now uses simple heuristics. In full system, would execute
        and test the code.
        """
        if not solution:
            return 0.0

        # Basic heuristics
        score = 0.1  # Base score for trying

        # Length bonus (not too short, not too long)
        length = len(solution)
        if 50 < length < 500:
            score += 0.2

        # Has function definition
        if "def " in solution or "function" in solution:
            score += 0.2

        # Has return statement
        if "return" in solution:
            score += 0.2

        # Doesn't have obvious errors
        if "error" not in solution.lower() and "exception" not in solution.lower():
            score += 0.1

        # Novelty bonus (different from current solution)
        if self.current_solution:
            current_hash = hashlib.md5(self.current_solution.encode()).hexdigest()
            new_hash = hashlib.md5(solution.encode()).hexdigest()
            if current_hash != new_hash:
                score += 0.2

        return min(score, 1.0)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status for monitoring."""
        return {
            "name": self.name,
            "mode": self.agent_config.mode,
            "generation": self.generation,
            "fitness": self.fitness,
            "energy": self.energy,
            "improvements": self.improvements,
            "tokens_used": self.tokens_used,
            "inference_time": self.inference_time,
            "solution_preview": self.current_solution[:100] if self.current_solution else None,
        }
