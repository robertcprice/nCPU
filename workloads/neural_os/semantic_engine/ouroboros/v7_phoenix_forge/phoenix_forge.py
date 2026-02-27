"""
Phoenix Forge Arena (V7 Main Orchestrator)
============================================
The main loop for V7 productive track.

Combines:
- Small AI agents (cooperating)
- Consensus Oracle (safe brain, no override)
- Active Inference (free energy minimization)
- Blackboard (shared memory for collaboration)
- MAP-Elites (diversity preservation)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..shared.small_ai_agent import SmallAIAgent, AgentMode, OllamaBrain
from ..shared.constitution import get_constitution
from ..shared.novelty_oracle import get_novelty_oracle
from ..shared.verification import get_judge, TestCase
from ..shared.audit import get_audit_log, EventType

from .consensus_oracle import ConsensusOracle
from .world_model import HierarchicalWorldModel
from .free_energy import FreeEnergyMinimizer
from .blackboard import Blackboard, Hypothesis, HypothesisStatus
from .map_elites import MAPElites


@dataclass
class IterationResult:
    """Result of one iteration."""
    iteration: int
    agents_alive: int
    hypotheses_posted: int
    hypotheses_validated: int
    best_fitness: float
    avg_fitness: float
    best_F: float  # Best free energy (lower = better)
    niches_discovered: int
    consensus_reached: int
    duration_ms: float


class PhoenixForgeArena:
    """
    Main orchestrator for V7 Phoenix Forge track.

    This is the PRODUCTIVE track where:
    - Agents cooperate via blackboard
    - Consensus Oracle facilitates (no override)
    - Active Inference drives exploration
    - MAP-Elites preserves diversity
    """

    def __init__(
        self,
        population_size: int = 50,
        brain_model: str = "tinyllama",
        oracle_model: str = "mistral:7b",
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

        # V7-specific components
        self.oracle = ConsensusOracle(brain_model=oracle_model)
        self.blackboard = Blackboard()
        self.map_elites = MAPElites()

        # Shared world model for all agents
        self.world_model = HierarchicalWorldModel()

        # Population
        self.agents: List[SmallAIAgent] = []
        self.iteration = 0

        # Test cases for the problem
        self.test_cases = self.judge.create_test_suite(problem_type)

        # History
        self.iteration_history: List[IterationResult] = []

        # Log arena creation
        self.audit.append(
            EventType.SYSTEM_START,
            track="v7",
            details={
                "population_size": population_size,
                "brain_model": brain_model,
                "oracle_model": oracle_model,
                "problem_type": problem_type,
            }
        )

    def initialize_population(self, seed_code: str = "") -> None:
        """Initialize the agent population."""
        for i in range(self.population_size):
            agent = SmallAIAgent(
                brain=OllamaBrain(self.brain_model),
                mode=AgentMode.COOPERATIVE,
                code=seed_code or self._generate_seed_code(),
            )
            self.agents.append(agent)

            self.audit.append(
                EventType.AGENT_SPAWN,
                track="v7",
                agent_id=agent.id,
                details={"iteration": 0},
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

    def run_iteration(self) -> IterationResult:
        """Run one iteration of cooperative evolution."""
        import time
        start_time = time.time()

        self.iteration += 1
        hypotheses_posted = 0
        hypotheses_validated = 0
        consensus_reached = 0

        # 1. Oracle observes the swarm
        observation = self.oracle.observe_swarm(self.agents)

        # 2. Each agent explores using Active Inference
        for agent in self.agents:
            if not agent.is_alive():
                continue

            # Create agent-specific free energy minimizer
            minimizer = FreeEnergyMinimizer(self.world_model)

            # Compute current free energy
            current_F = minimizer.compute_free_energy(
                agent.code,
                f"Solve {self.problem_type}",
                lambda c: self._evaluate_code(c)
            )

            # Check blackboard for related hypotheses
            related = self._find_related_hypotheses(agent)

            # Agent reasons about problem
            reasoning = agent.reason_about_problem(f"Improve {self.problem_type}")

            # Try to minimize free energy
            new_code, new_F = minimizer.minimize_step(
                agent.code,
                f"Solve {self.problem_type}",
                lambda c: self._evaluate_code(c)
            )

            # If improved, post hypothesis to blackboard
            if new_F.total_F < current_F.total_F:
                hypothesis = self.blackboard.post_hypothesis(
                    author_id=agent.id,
                    title=f"Improvement from agent {agent.id}",
                    description=f"Reduced F from {current_F.total_F:.3f} to {new_F.total_F:.3f}",
                    code_snippet=new_code,
                    tags=set(minimizer.world_model.algorithm_prior.detect_patterns(new_code)),
                )
                hypotheses_posted += 1

                # Update agent
                agent.code = new_code
                agent.fitness = new_F.accuracy
                agent.share_discovery({"hypothesis_id": hypothesis.id})

                # Try to add to MAP-Elites
                added, niche_id, is_new = self.map_elites.try_add_solution(
                    new_code,
                    new_F.accuracy,
                    new_F.total_F
                )

            # Agent considers borrowing from blackboard
            if related:
                best_related = max(related, key=lambda h: h.confidence)
                if best_related.code_snippet and best_related.confidence > 0.6:
                    # Try to merge ideas
                    merged_code = self._merge_ideas(agent.code, best_related.code_snippet)
                    merged_F = minimizer.compute_free_energy(
                        merged_code,
                        f"Solve {self.problem_type}",
                        lambda c: self._evaluate_code(c)
                    )

                    if merged_F.total_F < new_F.total_F:
                        agent.code = merged_code
                        agent.fitness = merged_F.accuracy
                        agent.borrow_idea({"from": best_related.id})

        # 3. Test hypotheses and update blackboard
        for hypothesis in self.blackboard.hypotheses.values():
            if hypothesis.status == HypothesisStatus.PROPOSED:
                if hypothesis.code_snippet:
                    # Test the hypothesis
                    result = self.judge.verify(hypothesis.code_snippet, self.test_cases)

                    # Report result
                    self.blackboard.report_result(
                        hypothesis.id,
                        "system",
                        result.passed,
                        {"score": result.score, "tests_passed": result.tests_passed}
                    )

                    if result.passed:
                        hypotheses_validated += 1

        # 4. Oracle may propose a direction
        if self.iteration % 5 == 0:  # Every 5 iterations
            proposal = self.oracle.propose_direction(
                self.agents,
                f"Current best fitness: {max((a.fitness for a in self.agents), default=0):.2f}"
            )
            self.oracle.collect_votes(proposal, self.agents)
            result = self.oracle.finalize_proposal(proposal)

            if result.accepted:
                consensus_reached += 1

        # 5. Allocate resources fairly
        allocations = self.oracle.allocate_resources_fairly(self.agents, 1000.0)
        for agent in self.agents:
            agent.mutation_energy = allocations.get(agent.id, 50.0)

        # 6. Update world model with observations
        for agent in self.agents:
            self.world_model.update_beliefs(agent.code, {
                "fitness": agent.fitness,
                "iteration": self.iteration,
            })

        # Calculate iteration stats
        alive_agents = [a for a in self.agents if a.is_alive()]
        best_fitness = max((a.fitness for a in alive_agents), default=0.0)
        avg_fitness = sum(a.fitness for a in alive_agents) / max(len(alive_agents), 1)

        # Best free energy (from archive)
        archive_stats = self.map_elites.get_exploration_stats()
        best_F = archive_stats.get("best_F", float('inf'))

        duration = (time.time() - start_time) * 1000

        result = IterationResult(
            iteration=self.iteration,
            agents_alive=len(alive_agents),
            hypotheses_posted=hypotheses_posted,
            hypotheses_validated=hypotheses_validated,
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            best_F=best_F if best_F != float('inf') else 0.0,
            niches_discovered=archive_stats.get("niches_discovered", 0),
            consensus_reached=consensus_reached,
            duration_ms=duration,
        )

        self.iteration_history.append(result)

        return result

    def _evaluate_code(self, code: str) -> float:
        """Evaluate code fitness."""
        try:
            result = self.judge.verify(code, self.test_cases)
            return result.score
        except Exception:
            return 0.0

    def _find_related_hypotheses(self, agent: SmallAIAgent) -> List[Hypothesis]:
        """Find hypotheses related to agent's current work."""
        # Create a dummy hypothesis from agent's code
        dummy = Hypothesis(
            id="temp",
            author_id=agent.id,
            title="temp",
            description="",
            code_snippet=agent.code,
            status=HypothesisStatus.PROPOSED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return self.blackboard.get_related(dummy, max_results=3)

    def _merge_ideas(self, code1: str, code2: str) -> str:
        """
        Merge ideas from two code snippets.

        This is a simple implementation - production would be smarter.
        """
        # Extract function bodies
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')

        # Find useful lines from code2 not in code1
        code1_set = set(l.strip() for l in lines1)
        useful_from_2 = [
            l for l in lines2
            if l.strip() and l.strip() not in code1_set
            and not l.strip().startswith('#')
            and not l.strip().startswith('def ')
        ]

        # Try to insert useful lines
        if useful_from_2 and len(lines1) > 2:
            # Insert before the return statement if there is one
            for i, line in enumerate(lines1):
                if 'return' in line:
                    # Insert one useful line
                    if useful_from_2:
                        lines1.insert(i, useful_from_2[0])
                    break

        return '\n'.join(lines1)

    def run(self, iterations: int = 100) -> Dict[str, Any]:
        """Run the full evolution."""
        self.initialize_population()

        for _ in range(iterations):
            result = self.run_iteration()

            # Check for kill switch
            if self.constitution.is_killed():
                break

            # Early stopping if perfect solution found
            if result.best_fitness >= 1.0:
                break

        return self.get_final_report()

    def get_final_report(self) -> Dict[str, Any]:
        """Get final report."""
        # Get best solutions from archive
        diverse_solutions = self.map_elites.get_diverse_solutions(5)

        return {
            "track": "v7_phoenix_forge",
            "iterations_run": self.iteration,
            "final_population": len([a for a in self.agents if a.is_alive()]),
            "best_solution": max(self.agents, key=lambda a: a.fitness).code if self.agents else "",
            "best_fitness": max((a.fitness for a in self.agents), default=0.0),
            "diverse_solutions": [
                {
                    "niche_id": n.niche_id,
                    "dimensions": n.dimensions,
                    "fitness": n.best_fitness,
                    "F": n.best_F,
                }
                for n in diverse_solutions
            ],
            "oracle_status": self.oracle.get_status(),
            "blackboard_stats": self.blackboard.get_stats(),
            "map_elites_stats": self.map_elites.get_exploration_stats(),
            "world_model_stats": self.world_model.get_learning_stats(),
            "cooperation_stats": {
                "total_shared": sum(a.shared_discoveries for a in self.agents),
                "total_borrowed": sum(a.borrowed_ideas for a in self.agents),
                "hypotheses_validated": sum(1 for h in self.blackboard.hypotheses.values() if h.status == HypothesisStatus.VALIDATED),
            },
            "iteration_history": [
                {
                    "iter": r.iteration,
                    "alive": r.agents_alive,
                    "best_fit": r.best_fitness,
                    "avg_fit": r.avg_fitness,
                    "niches": r.niches_discovered,
                    "hypotheses": r.hypotheses_posted,
                }
                for r in self.iteration_history
            ],
        }
