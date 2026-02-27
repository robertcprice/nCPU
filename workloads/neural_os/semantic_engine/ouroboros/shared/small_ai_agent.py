"""
Small AI Agent
===============
Each agent is a TINY MIND, not just code being evolved.

The agent has:
- Brain: Small LLM (TinyLlama, Phi-2, etc.) for reasoning
- Memory: Vector store of experiences
- Goals: Stack of objectives
- Energy: Budget for mutations AND program execution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto
from datetime import datetime
import uuid
import json
from abc import ABC, abstractmethod


class AgentMode(Enum):
    """Operating mode for the agent."""
    COMPETITIVE = auto()   # V6: Fight for resources
    COOPERATIVE = auto()   # V7: Share discoveries


@dataclass
class Experience:
    """A single experience in agent memory."""
    timestamp: datetime
    action: str
    context: Dict[str, Any]
    outcome: str
    reward: float
    surprise: float  # How unexpected was this?

    def to_embedding_text(self) -> str:
        """Convert to text for embedding."""
        return f"Action: {self.action}\nContext: {json.dumps(self.context)}\nOutcome: {self.outcome}\nReward: {self.reward}"


class AgentMemory:
    """
    Vector-based memory for agent experiences.

    Uses simple similarity search for now.
    Can be upgraded to proper vector DB later.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: List[Experience] = []
        self._embeddings: List[List[float]] = []  # Placeholder for real embeddings

    def store(self, experience: Experience) -> None:
        """Store an experience in memory."""
        self.experiences.append(experience)
        # In production: compute embedding and store
        # For now: just store the experience

        # Trim if over size (FIFO with priority for high-surprise)
        if len(self.experiences) > self.max_size:
            # Keep high-surprise experiences
            self.experiences.sort(key=lambda e: e.surprise, reverse=True)
            self.experiences = self.experiences[:self.max_size]

    def recall_similar(self, query: str, k: int = 5) -> List[Experience]:
        """Recall k most similar experiences to query."""
        # In production: vector similarity search
        # For now: simple keyword matching
        scored = []
        query_words = set(query.lower().split())

        for exp in self.experiences:
            exp_words = set(exp.to_embedding_text().lower().split())
            overlap = len(query_words & exp_words)
            scored.append((overlap, exp))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:k]]

    def recall_high_reward(self, k: int = 5) -> List[Experience]:
        """Recall k highest reward experiences."""
        sorted_exp = sorted(self.experiences, key=lambda e: e.reward, reverse=True)
        return sorted_exp[:k]

    def recall_high_surprise(self, k: int = 5) -> List[Experience]:
        """Recall k highest surprise experiences."""
        sorted_exp = sorted(self.experiences, key=lambda e: e.surprise, reverse=True)
        return sorted_exp[:k]


@dataclass
class Goal:
    """A goal on the agent's goal stack."""
    description: str
    priority: float
    deadline: Optional[datetime] = None
    parent_goal: Optional[str] = None

    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline


class GoalStack:
    """Stack of agent goals with priority."""

    def __init__(self):
        self.goals: List[Goal] = []

    def push(self, goal: Goal) -> None:
        """Push a new goal onto the stack."""
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def pop(self) -> Optional[Goal]:
        """Pop the highest priority goal."""
        # Remove expired goals
        self.goals = [g for g in self.goals if not g.is_expired()]
        if not self.goals:
            return None
        return self.goals.pop(0)

    def peek(self) -> Optional[Goal]:
        """Peek at the highest priority goal."""
        self.goals = [g for g in self.goals if not g.is_expired()]
        if not self.goals:
            return None
        return self.goals[0]

    def clear(self) -> None:
        """Clear all goals."""
        self.goals = []


class AgentBrain(ABC):
    """
    Abstract base class for agent brains.

    Different implementations can use:
    - TinyLlama (1.1B)
    - Phi-2 (2.7B)
    - Quantized CodeLlama
    - Custom distilled model
    """

    @abstractmethod
    def reason(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate reasoning about a problem."""
        pass

    @abstractmethod
    def evaluate(self, code: str, goal: str) -> float:
        """Evaluate how well code achieves goal (0-1)."""
        pass

    @abstractmethod
    def propose_mutation(self, code: str, goal: str, memory: List[Experience]) -> str:
        """Propose a mutation to the code."""
        pass


class OllamaBrain(AgentBrain):
    """Brain implementation using local Ollama models."""

    def __init__(self, model_name: str = "tinyllama"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy load the Ollama client."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client()
            except ImportError:
                raise RuntimeError("ollama package not installed")
        return self._client

    def reason(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate reasoning using Ollama."""
        try:
            client = self._get_client()
            response = client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"num_predict": max_tokens}
            )
            return response["response"]
        except Exception as e:
            return f"[Brain error: {e}]"

    def evaluate(self, code: str, goal: str) -> float:
        """Evaluate code against goal."""
        prompt = f"""Rate this code from 0 to 1 for achieving the goal.
Goal: {goal}
Code:
```
{code}
```
Just respond with a number between 0 and 1."""

        response = self.reason(prompt, max_tokens=10)
        try:
            # Extract number from response
            import re
            numbers = re.findall(r"0?\.\d+|1\.0|0|1", response)
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 0.5  # Default if can't parse

    def propose_mutation(self, code: str, goal: str, memory: List[Experience]) -> str:
        """Propose a code mutation."""
        memory_context = ""
        if memory:
            memory_context = "Previous attempts:\n"
            for exp in memory[-3:]:  # Last 3 experiences
                memory_context += f"- {exp.action}: {exp.outcome} (reward: {exp.reward})\n"

        prompt = f"""You are improving code to achieve a goal.
Goal: {goal}

Current code:
```
{code}
```

{memory_context}

Propose ONE specific change to improve the code. Output ONLY the modified code."""

        return self.reason(prompt, max_tokens=500)


@dataclass
class SmallAIAgent:
    """
    A small AI agent that can reason, learn, and evolve.

    This is NOT just code being mutated - it's a tiny mind
    that actively tries to solve problems.
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_id: Optional[str] = None

    # The brain (small LLM)
    brain: AgentBrain = field(default_factory=lambda: OllamaBrain("tinyllama"))

    # Memory and goals
    memory: AgentMemory = field(default_factory=AgentMemory)
    goals: GoalStack = field(default_factory=GoalStack)

    # Current state
    code: str = ""  # The code this agent is evolving
    fitness: float = 0.0
    novelty: float = 0.0

    # Energy budget (TWO levels)
    mutation_energy: float = 100.0   # Cost to THINK/TRY things
    program_energy: float = 100.0    # Cost of PROGRAMS it creates

    # Mode (V6 vs V7)
    mode: AgentMode = AgentMode.COMPETITIVE

    # Metrics
    mutations_attempted: int = 0
    mutations_successful: int = 0
    total_reward: float = 0.0

    # Trust level (V6 only - for narrator interaction)
    trust_level: float = 0.0  # 0.0 to 1.0

    # Cooperation (V7 only)
    shared_discoveries: int = 0
    borrowed_ideas: int = 0

    def reason_about_problem(self, problem: str) -> str:
        """Use brain to reason about how to approach a problem."""
        # Recall relevant experiences
        relevant = self.memory.recall_similar(problem, k=3)

        prompt = f"""Problem: {problem}

Current code:
```
{self.code}
```

Relevant past experiences:
{self._format_experiences(relevant)}

How should I approach improving this code?"""

        return self.brain.reason(prompt)

    def propose_solution(self, goal: str) -> Tuple[str, float]:
        """
        Propose a code modification to achieve goal.

        Returns (new_code, energy_cost)
        """
        # Get memory context
        relevant = self.memory.recall_high_reward(k=3)

        # Propose mutation
        new_code = self.brain.propose_mutation(self.code, goal, relevant)

        # Calculate energy cost (LLM calls cost more)
        energy_cost = 5.0  # Base LLM call cost

        return new_code, energy_cost

    def evaluate_solution(self, code: str, goal: str) -> float:
        """Use brain to evaluate how well code achieves goal."""
        return self.brain.evaluate(code, goal)

    def learn_from_outcome(
        self,
        action: str,
        context: Dict[str, Any],
        outcome: str,
        reward: float,
        surprise: float
    ) -> None:
        """Learn from the outcome of an action."""
        experience = Experience(
            timestamp=datetime.now(),
            action=action,
            context=context,
            outcome=outcome,
            reward=reward,
            surprise=surprise,
        )
        self.memory.store(experience)
        self.total_reward += reward

        # Update trust based on outcomes (V6)
        if self.mode == AgentMode.COMPETITIVE:
            if reward > 0:
                self.trust_level = min(1.0, self.trust_level + 0.01 * reward)
            else:
                self.trust_level = max(0.0, self.trust_level - 0.02 * abs(reward))

    def can_afford(self, cost: float) -> bool:
        """Check if agent can afford an action."""
        return self.mutation_energy >= cost

    def spend_energy(self, cost: float, reason: str) -> bool:
        """Spend mutation energy on an action."""
        if not self.can_afford(cost):
            return False
        self.mutation_energy -= cost
        return True

    def receive_energy(self, amount: float, reason: str) -> None:
        """Receive energy (from rewards or regeneration)."""
        self.mutation_energy += amount

    def pay_program_cost(self, cost: float) -> bool:
        """Pay the cost of running a program the agent created."""
        if self.program_energy < cost:
            return False
        self.program_energy -= cost
        return True

    def is_alive(self) -> bool:
        """Check if agent has enough energy to continue."""
        return self.mutation_energy > 0 and self.program_energy > 0

    def share_discovery(self, discovery: Dict[str, Any]) -> None:
        """Share a discovery (V7 cooperative mode)."""
        if self.mode == AgentMode.COOPERATIVE:
            self.shared_discoveries += 1

    def borrow_idea(self, idea: Dict[str, Any]) -> None:
        """Borrow an idea from the blackboard (V7)."""
        if self.mode == AgentMode.COOPERATIVE:
            self.borrowed_ideas += 1

    def _format_experiences(self, experiences: List[Experience]) -> str:
        """Format experiences for prompt."""
        if not experiences:
            return "None"

        lines = []
        for exp in experiences:
            lines.append(f"- {exp.action}: {exp.outcome} (reward: {exp.reward:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state."""
        return {
            "id": self.id,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "code": self.code,
            "fitness": self.fitness,
            "novelty": self.novelty,
            "mutation_energy": self.mutation_energy,
            "program_energy": self.program_energy,
            "mode": self.mode.name,
            "mutations_attempted": self.mutations_attempted,
            "mutations_successful": self.mutations_successful,
            "total_reward": self.total_reward,
            "trust_level": self.trust_level,
            "memory_size": len(self.memory.experiences),
        }

    @classmethod
    def spawn(cls, parent: "SmallAIAgent", mode: AgentMode) -> "SmallAIAgent":
        """Spawn a child agent from a parent."""
        child = cls(
            generation=parent.generation + 1,
            parent_id=parent.id,
            brain=parent.brain,  # Share brain (or could copy)
            code=parent.code,
            mode=mode,
            mutation_energy=50.0,  # Start with less energy
            program_energy=50.0,
        )

        # Inherit some memories (knowledge transfer)
        best_memories = parent.memory.recall_high_reward(k=10)
        for mem in best_memories:
            child.memory.store(mem)

        return child
