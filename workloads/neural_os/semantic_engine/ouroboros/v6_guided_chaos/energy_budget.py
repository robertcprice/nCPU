"""
Energy Budget System (V6)
==========================
Two-level thermodynamic constraints to prevent wasteful chaos.

Level 1: Mutation Energy - Cost to THINK/TRY things
Level 2: Program Energy - Cost of PROGRAMS agent creates

This creates pressure toward EFFICIENT solutions, not just correct ones.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum, auto


class EnergyType(Enum):
    """Types of energy expenditure."""
    MUTATION_AST = auto()      # Simple AST mutation
    MUTATION_LLM = auto()      # LLM-assisted mutation
    EVALUATION = auto()        # Running tests
    REASONING = auto()         # Agent reasoning
    PROGRAM_EXECUTION = auto() # Running the output program


@dataclass
class EnergyTransaction:
    """Record of an energy transaction."""
    agent_id: str
    energy_type: EnergyType
    amount: float  # Positive = spent, Negative = gained
    reason: str
    timestamp: datetime
    balance_after: float


@dataclass
class ProgramCost:
    """Cost of running an output program."""
    execution_time_ms: float
    memory_usage_mb: float
    complexity_score: float  # Estimated Big-O
    total_cost: float

    @classmethod
    def estimate(cls, code: str) -> "ProgramCost":
        """Estimate program cost from code analysis."""
        import re

        # Estimate complexity from code patterns
        complexity = 1.0

        # Nested loops = O(n^2) or worse
        loop_depth = 0
        current_depth = 0
        for line in code.split('\n'):
            if 'for ' in line or 'while ' in line:
                current_depth += 1
                loop_depth = max(loop_depth, current_depth)
            elif line.strip() and not line.strip().startswith('#'):
                if current_depth > 0 and not ('for ' in line or 'while ' in line):
                    pass  # Still in loop
                elif line.strip().startswith('return') or line.strip().startswith('def '):
                    current_depth = 0

        complexity *= (2 ** loop_depth)  # Exponential penalty for nesting

        # Recursion detection
        if re.search(r'def\s+(\w+).*\1\s*\(', code, re.DOTALL):
            complexity *= 1.5  # Recursion adds cost

        # Memory estimation from data structures
        memory_score = 1.0
        if '[' in code and ']' in code:
            memory_score += 0.5  # List usage
        if '{' in code and '}' in code:
            memory_score += 0.5  # Dict usage
        if 'class ' in code:
            memory_score += 1.0  # Object creation

        # Estimate execution time (very rough)
        lines = len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')])
        exec_time = lines * complexity * 0.1  # ms per line * complexity

        return cls(
            execution_time_ms=exec_time,
            memory_usage_mb=memory_score,
            complexity_score=complexity,
            total_cost=exec_time * 0.01 + memory_score * 0.5 + complexity * 0.5,
        )


class EnergyBudgetSystem:
    """
    Manages energy budgets for all agents.

    Key Principles:
    1. Mutations COST energy
    2. Programs agents create COST energy
    3. Good outcomes REWARD energy
    4. Energy regenerates slowly
    5. Dead agents (0 energy) are culled
    """

    # Cost configuration
    COSTS = {
        EnergyType.MUTATION_AST: 1.0,      # Cheap: simple AST change
        EnergyType.MUTATION_LLM: 5.0,      # Expensive: LLM call
        EnergyType.EVALUATION: 2.0,        # Running tests
        EnergyType.REASONING: 3.0,         # Agent reasoning
        EnergyType.PROGRAM_EXECUTION: 1.0, # Base cost, multiplied by program cost
    }

    # Reward configuration
    FITNESS_IMPROVEMENT_REWARD = 20.0
    NOVELTY_REWARD = 10.0
    EFFICIENCY_BONUS = 5.0  # For creating efficient programs

    # Regeneration
    REGEN_PER_CYCLE = 5.0
    MAX_ENERGY = 200.0
    MIN_ENERGY_TO_ACT = 1.0

    def __init__(self):
        self._transactions: List[EnergyTransaction] = []
        self._agent_balances: Dict[str, Dict[str, float]] = {}  # mutation + program

    def register_agent(self, agent_id: str, initial_mutation: float = 100.0, initial_program: float = 100.0) -> None:
        """Register a new agent with initial energy."""
        self._agent_balances[agent_id] = {
            "mutation": initial_mutation,
            "program": initial_program,
        }

    def get_balance(self, agent_id: str) -> Dict[str, float]:
        """Get agent's current energy balances."""
        return self._agent_balances.get(agent_id, {"mutation": 0, "program": 0})

    def can_afford(self, agent_id: str, energy_type: EnergyType, amount: Optional[float] = None) -> bool:
        """Check if agent can afford an action."""
        balance = self.get_balance(agent_id)
        cost = amount if amount is not None else self.COSTS.get(energy_type, 1.0)

        if energy_type == EnergyType.PROGRAM_EXECUTION:
            return balance["program"] >= cost
        else:
            return balance["mutation"] >= cost

    def spend(
        self,
        agent_id: str,
        energy_type: EnergyType,
        reason: str,
        amount: Optional[float] = None
    ) -> bool:
        """Spend energy on an action."""
        if agent_id not in self._agent_balances:
            return False

        cost = amount if amount is not None else self.COSTS.get(energy_type, 1.0)

        if energy_type == EnergyType.PROGRAM_EXECUTION:
            if self._agent_balances[agent_id]["program"] < cost:
                return False
            self._agent_balances[agent_id]["program"] -= cost
            balance_after = self._agent_balances[agent_id]["program"]
        else:
            if self._agent_balances[agent_id]["mutation"] < cost:
                return False
            self._agent_balances[agent_id]["mutation"] -= cost
            balance_after = self._agent_balances[agent_id]["mutation"]

        self._transactions.append(EnergyTransaction(
            agent_id=agent_id,
            energy_type=energy_type,
            amount=cost,
            reason=reason,
            timestamp=datetime.now(),
            balance_after=balance_after,
        ))

        return True

    def spend_program_cost(self, agent_id: str, program: str) -> Tuple[bool, ProgramCost]:
        """Spend energy based on program complexity."""
        cost = ProgramCost.estimate(program)

        success = self.spend(
            agent_id,
            EnergyType.PROGRAM_EXECUTION,
            f"program_execution (complexity={cost.complexity_score:.1f})",
            cost.total_cost,
        )

        return success, cost

    def reward(
        self,
        agent_id: str,
        fitness_delta: float,
        novelty_score: float,
        efficiency_bonus: float = 0.0
    ) -> float:
        """Reward agent for good outcomes."""
        if agent_id not in self._agent_balances:
            return 0.0

        reward = 0.0

        # Fitness improvement reward
        if fitness_delta > 0:
            reward += fitness_delta * self.FITNESS_IMPROVEMENT_REWARD

        # Novelty reward
        reward += novelty_score * self.NOVELTY_REWARD

        # Efficiency bonus
        reward += efficiency_bonus * self.EFFICIENCY_BONUS

        # Apply reward (split between mutation and program energy)
        mutation_reward = reward * 0.7
        program_reward = reward * 0.3

        self._agent_balances[agent_id]["mutation"] = min(
            self.MAX_ENERGY,
            self._agent_balances[agent_id]["mutation"] + mutation_reward
        )
        self._agent_balances[agent_id]["program"] = min(
            self.MAX_ENERGY,
            self._agent_balances[agent_id]["program"] + program_reward
        )

        self._transactions.append(EnergyTransaction(
            agent_id=agent_id,
            energy_type=EnergyType.MUTATION_AST,  # Using as "reward" type
            amount=-reward,  # Negative = gained
            reason=f"reward (fitness={fitness_delta:.2f}, novelty={novelty_score:.2f})",
            timestamp=datetime.now(),
            balance_after=self._agent_balances[agent_id]["mutation"],
        ))

        return reward

    def regenerate_cycle(self) -> Dict[str, float]:
        """Regenerate energy for all agents."""
        regen_amounts = {}

        for agent_id in self._agent_balances:
            # Mutation energy regenerates faster
            mutation_regen = min(
                self.REGEN_PER_CYCLE,
                self.MAX_ENERGY - self._agent_balances[agent_id]["mutation"]
            )
            self._agent_balances[agent_id]["mutation"] += mutation_regen

            # Program energy regenerates slower
            program_regen = min(
                self.REGEN_PER_CYCLE * 0.5,
                self.MAX_ENERGY - self._agent_balances[agent_id]["program"]
            )
            self._agent_balances[agent_id]["program"] += program_regen

            regen_amounts[agent_id] = mutation_regen + program_regen

        return regen_amounts

    def get_dead_agents(self) -> List[str]:
        """Get list of agents with no energy (should be culled)."""
        dead = []
        for agent_id, balances in self._agent_balances.items():
            if balances["mutation"] < self.MIN_ENERGY_TO_ACT and balances["program"] < self.MIN_ENERGY_TO_ACT:
                dead.append(agent_id)
        return dead

    def transfer_energy(self, from_agent: str, to_agent: str, amount: float) -> bool:
        """Transfer energy between agents (for V7 cooperation)."""
        if from_agent not in self._agent_balances or to_agent not in self._agent_balances:
            return False

        if self._agent_balances[from_agent]["mutation"] < amount:
            return False

        self._agent_balances[from_agent]["mutation"] -= amount
        self._agent_balances[to_agent]["mutation"] += amount

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get energy system statistics."""
        total_mutation = sum(b["mutation"] for b in self._agent_balances.values())
        total_program = sum(b["program"] for b in self._agent_balances.values())

        return {
            "agent_count": len(self._agent_balances),
            "total_mutation_energy": total_mutation,
            "total_program_energy": total_program,
            "avg_mutation_energy": total_mutation / max(len(self._agent_balances), 1),
            "avg_program_energy": total_program / max(len(self._agent_balances), 1),
            "dead_agents": len(self.get_dead_agents()),
            "transactions": len(self._transactions),
        }


# Need to import Tuple for type hints
from typing import Tuple
