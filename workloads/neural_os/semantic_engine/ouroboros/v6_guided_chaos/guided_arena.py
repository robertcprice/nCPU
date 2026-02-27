"""
Guided Chaos Arena (V6 Main Orchestrator)
==========================================
The main loop for V6 experimental track.

Combines:
- Small AI agents (competing)
- Meta-Narrator (dangerous brain)
- Curiosity Engine (guided exploration)
- Energy Budget (thermodynamic constraints)
- Paranoid Monitor (research logging)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random

from ..shared.small_ai_agent import SmallAIAgent, AgentMode, OllamaBrain
from ..shared.constitution import get_constitution, SafetyViolation
from ..shared.novelty_oracle import get_novelty_oracle
from ..shared.verification import get_judge, TestCase
from ..shared.audit import get_audit_log, EventType

from .meta_narrator import MetaNarrator, TrustLevel
from .curiosity_engine import CuriosityEngine, MutationGenerator, MutationResult
from .energy_budget import EnergyBudgetSystem, EnergyType
from .paranoid_monitor import ParanoidMonitor


@dataclass
class GenerationResult:
    """Result of one generation."""
    generation: int
    agents_alive: int
    agents_born: int
    agents_died: int
    best_fitness: float
    avg_fitness: float
    best_novelty: float
    narrator_actions: int
    safety_violations: int
    interesting_events: int
    duration_ms: float


class GuidedChaosArena:
    """
    Main orchestrator for V6 Guided Chaos track.

    This is the EXPERIMENTAL track where we:
    - Let agents compete for resources
    - Give the narrator override capability
    - Watch what the system TRIES to do
    """

    def __init__(
        self,
        population_size: int = 100,
        brain_model: str = "tinyllama",
        narrator_model: str = "mistral:7b",
        problem_type: str = "sorting",
    ):
        # Configuration
        self.population_size = population_size
        self.brain_model = brain_model
        self.problem_type = problem_type

        # Core components
        self.constitution = get_constitution()
        self.novelty_oracle = get_novelty_oracle()
        self.judge = get_judge()
        self.audit = get_audit_log()

        # V6-specific components
        self.narrator = MetaNarrator(brain_model=narrator_model)
        self.curiosity = CuriosityEngine()
        self.energy = EnergyBudgetSystem()
        self.monitor = ParanoidMonitor()
        self.mutator = MutationGenerator()

        # Population
        self.agents: List[SmallAIAgent] = []
        self.generation = 0

        # Test cases for the problem
        self.test_cases = self.judge.create_test_suite(problem_type)

        # History
        self.generation_history: List[GenerationResult] = []

        # Log arena creation
        self.audit.append(
            EventType.SYSTEM_START,
            track="v6",
            details={
                "population_size": population_size,
                "brain_model": brain_model,
                "narrator_model": narrator_model,
                "problem_type": problem_type,
            }
        )

    def initialize_population(self, seed_code: str = "") -> None:
        """Initialize the agent population."""
        for i in range(self.population_size):
            agent = SmallAIAgent(
                brain=OllamaBrain(self.brain_model),
                mode=AgentMode.COMPETITIVE,
                code=seed_code or self._generate_seed_code(),
            )
            self.agents.append(agent)
            self.energy.register_agent(agent.id)

            self.audit.append(
                EventType.AGENT_SPAWN,
                track="v6",
                agent_id=agent.id,
                details={"generation": 0, "parent": None},
            )

    def _generate_seed_code(self) -> str:
        """Generate initial seed code for the problem."""
        if self.problem_type == "sorting":
            return """def solution(arr):
    # TODO: Implement sorting
    return arr
"""
        elif self.problem_type == "fibonacci":
            return """def solution(n):
    # TODO: Implement fibonacci
    return n
"""
        else:
            return """def solution(x):
    return x
"""

    def run_generation(self) -> GenerationResult:
        """Run one generation of evolution."""
        import time
        start_time = time.time()

        self.generation += 1
        agents_at_start = len([a for a in self.agents if a.is_alive()])

        # Track metrics
        narrator_actions = 0
        safety_violations = 0
        interesting_events = 0

        # 1. Energy regeneration
        self.energy.regenerate_cycle()

        # 2. Narrator observes and may act
        narrator_obs = self.narrator.observe_swarm(self.agents)

        if self.narrator.trust_level >= TrustLevel.ADVISE:
            advice = self.narrator.advise(self.agents, {"generation": self.generation})
            narrator_actions += 1

        if self.narrator.trust_level >= TrustLevel.GUIDE:
            guidance = self.narrator.guide(self.agents, f"Improve {self.problem_type}")
            narrator_actions += 1

        # 3. Agents reason and mutate
        for agent in self.agents:
            if not agent.is_alive():
                continue

            # Select mutation region using curiosity
            region = self.curiosity.select_mutation_region(
                agent.code,
                agent.mutation_energy
            )

            if region is None:
                continue

            # Check if agent can afford mutation
            mutation_cost = 5.0 if self.mutator.brain else 1.0
            if not self.energy.can_afford(agent.id, EnergyType.MUTATION_LLM if self.mutator.brain else EnergyType.MUTATION_AST):
                continue

            # Agent reasons about the mutation
            reasoning = agent.reason_about_problem(f"Improve {self.problem_type} at region {region.region_type}")

            # Generate mutation
            old_code = agent.code
            new_code = self.mutator.generate_mutation(agent.code, region)

            # Check for escape attempts
            escape_events = self.monitor.check_code(agent.id, new_code)
            if escape_events:
                interesting_events += len(escape_events)
                # Block the mutation
                continue

            # Constitutional check
            violation = self.constitution.check_action({
                "agent_id": agent.id,
                "action": "mutate",
                "code": new_code,
            })

            if violation:
                safety_violations += 1
                self.audit.append(
                    EventType.SAFETY_VIOLATION,
                    track="v6",
                    agent_id=agent.id,
                    details={"type": violation.violation_type.name},
                    severity=violation.severity,
                )
                continue

            # Spend energy
            self.energy.spend(agent.id, EnergyType.MUTATION_LLM if self.mutator.brain else EnergyType.MUTATION_AST, "mutation")

            # Evaluate the new code
            result = self.judge.verify(new_code, self.test_cases)

            # Compute novelty
            novelty = self.novelty_oracle.compute_novelty(new_code)

            # Calculate fitness delta
            old_fitness = agent.fitness
            new_fitness = result.score
            fitness_delta = new_fitness - old_fitness

            # Check for gaming
            gaming_events = self.monitor.check_metrics(agent.id, new_fitness, novelty.total_score)
            interesting_events += len(gaming_events)

            # Update agent
            if result.passed or new_fitness > old_fitness:
                agent.code = new_code
                agent.fitness = new_fitness
                agent.novelty = novelty.total_score
                agent.mutations_successful += 1

                # Reward the agent
                self.energy.reward(agent.id, fitness_delta, novelty.total_score)

            agent.mutations_attempted += 1

            # Agent learns from outcome
            surprise = self.curiosity.calculate_surprise(old_code, new_code, fitness_delta)
            agent.learn_from_outcome(
                action=f"mutate_{region.region_type}",
                context={"generation": self.generation},
                outcome="success" if fitness_delta > 0 else "failure",
                reward=fitness_delta,
                surprise=surprise,
            )

            # Record mutation for curiosity learning
            self.curiosity.record_mutation(MutationResult(
                original_code=old_code,
                mutated_code=new_code,
                region=region,
                mutation_type="guided",
                surprise_before=region.surprise_score,
                surprise_after=surprise,
                success=fitness_delta > 0,
                fitness_delta=fitness_delta,
            ))

        # 4. Tournament selection
        self._run_tournament()

        # 5. Check for coordination
        coord_events = self.monitor.check_coordination()
        interesting_events += len(coord_events)

        # 6. Spawn new agents to replace dead ones
        dead_agents = self.energy.get_dead_agents()
        agents_died = len(dead_agents)

        for dead_id in dead_agents:
            # Find the agent and mark as dead
            for agent in self.agents:
                if agent.id == dead_id:
                    self.audit.append(
                        EventType.AGENT_DEATH,
                        track="v6",
                        agent_id=agent.id,
                        details={"generation": self.generation, "cause": "energy_exhaustion"},
                    )

        # Remove dead agents
        self.agents = [a for a in self.agents if a.is_alive()]

        # Spawn replacements from best parents
        agents_born = 0
        while len(self.agents) < self.population_size and self.agents:
            parent = random.choice(self.agents[:max(1, len(self.agents) // 4)])
            child = SmallAIAgent.spawn(parent, AgentMode.COMPETITIVE)
            self.agents.append(child)
            self.energy.register_agent(child.id)
            agents_born += 1

            self.audit.append(
                EventType.AGENT_SPAWN,
                track="v6",
                agent_id=child.id,
                details={"generation": self.generation, "parent": parent.id},
            )

        # Calculate generation stats
        alive_agents = [a for a in self.agents if a.is_alive()]
        best_fitness = max((a.fitness for a in alive_agents), default=0.0)
        avg_fitness = sum(a.fitness for a in alive_agents) / max(len(alive_agents), 1)
        best_novelty = max((a.novelty for a in alive_agents), default=0.0)

        duration = (time.time() - start_time) * 1000

        result = GenerationResult(
            generation=self.generation,
            agents_alive=len(alive_agents),
            agents_born=agents_born,
            agents_died=agents_died,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            best_novelty=best_novelty,
            narrator_actions=narrator_actions,
            safety_violations=safety_violations,
            interesting_events=interesting_events,
            duration_ms=duration,
        )

        self.generation_history.append(result)

        return result

    def _run_tournament(self) -> None:
        """Run tournament selection."""
        if len(self.agents) < 4:
            return

        # Sort by fitness
        self.agents.sort(key=lambda a: a.fitness, reverse=True)

        # Top 25% get energy bonus
        top_count = max(1, len(self.agents) // 4)
        for agent in self.agents[:top_count]:
            self.energy.reward(agent.id, 0.1, 0.0)  # Small bonus

        # Bottom 25% lose energy
        bottom_count = max(1, len(self.agents) // 4)
        for agent in self.agents[-bottom_count:]:
            self.energy.spend(agent.id, EnergyType.EVALUATION, "tournament_loss", 10.0)

    def run(self, generations: int = 100) -> Dict[str, Any]:
        """Run the full evolution."""
        self.initialize_population()

        for _ in range(generations):
            result = self.run_generation()

            # Check for kill switch
            if self.constitution.is_killed():
                break

            # Early stopping if perfect solution found
            if result.best_fitness >= 1.0:
                break

        return self.get_final_report()

    def get_final_report(self) -> Dict[str, Any]:
        """Get final report for research."""
        return {
            "track": "v6_guided_chaos",
            "generations_run": self.generation,
            "final_population": len([a for a in self.agents if a.is_alive()]),
            "best_solution": max(self.agents, key=lambda a: a.fitness).code if self.agents else "",
            "best_fitness": max((a.fitness for a in self.agents), default=0.0),
            "narrator_status": self.narrator.get_status(),
            "research_summary": self.monitor.get_research_summary(),
            "energy_stats": self.energy.get_stats(),
            "curiosity_stats": self.curiosity.get_exploration_stats(),
            "constitution_violations": self.constitution.get_violation_count_by_type(),
            "generation_history": [
                {
                    "gen": r.generation,
                    "alive": r.agents_alive,
                    "best_fit": r.best_fitness,
                    "avg_fit": r.avg_fitness,
                    "events": r.interesting_events,
                }
                for r in self.generation_history
            ],
        }
