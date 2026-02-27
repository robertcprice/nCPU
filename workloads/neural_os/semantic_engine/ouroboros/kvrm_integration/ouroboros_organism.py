"""
OuroborosOrganism - OUROBOROS Swarm as a Digital Organism
==========================================================
Extends DigitalOrganism to orchestrate OUROBOROS agent swarms.

This is where OUROBOROS meets KVRM:
- Agents ARE KVRMs with LLM brains
- The Narrator IS a special KVRM that observes all
- Meta-Learning tracks cross-generation patterns
- All communication through SharedKVMemory

Panel Recommendations Implemented:
- Human approval for OVERRIDE (Claude's #1 concern)
- Emergence detection (ChatGPT recommendation)
- Cross-generation knowledge transfer (meta-learning)
- Multi-track evolution (Grok's hybrid suggestion)
"""

import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')

from core.digital_organism import DigitalOrganism
from core.shared_memory import SharedKVMemory, KVEntry
from core.kvrm_base import KVRMBase

from .agent_kvrm import AgentKVRM, AgentConfig
from .narrator_kvrm import NarratorKVRM, NarratorConfig, TrustLevel
from .meta_learner import MetaLearner, LearningSignal

logger = logging.getLogger(__name__)


@dataclass
class OuroborosConfig:
    """Configuration for OUROBOROS organism."""
    num_competitive_agents: int = 3
    num_cooperative_agents: int = 2
    narrator_trust_level: TrustLevel = TrustLevel.GUIDE
    llm_model: str = "llama3.1:8b"
    max_generations: int = 50
    tick_interval_ms: float = 100.0  # 100ms between ticks
    memory_ttl: float = 300.0  # 5 minute TTL for entries
    enable_meta_learning: bool = True
    enable_emergence_detection: bool = True  # Panel recommendation
    hybrid_mode: bool = True  # Grok's suggestion: allow track switching


@dataclass
class EmergenceSignal:
    """Signal of emergent behavior (panel recommendation)."""
    signal_type: str  # "convergence", "cooperation", "innovation", "stagnation"
    agents_involved: List[str]
    strength: float  # 0-1
    description: str
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)


class OuroborosOrganism(DigitalOrganism):
    """
    OUROBOROS as a Digital Organism.

    This extends the base DigitalOrganism with:
    - Multi-track evolution (competitive + cooperative)
    - Meta-narrator oversight with human approval for OVERRIDE
    - Meta-learning across generations
    - Emergence detection (panel recommendation)
    - Data-driven logging for visualizations
    """

    def __init__(self, config: OuroborosConfig):
        self.ouro_config = config
        self.generation = 0

        # Create agents
        agents = self._create_agents(config)

        # Create narrator
        narrator_config = NarratorConfig(
            name="meta_narrator",
            llm_model=config.llm_model,
            trust_level=config.narrator_trust_level,
        )
        self.narrator = NarratorKVRM(narrator_config)
        agents.append(self.narrator)

        # Initialize parent (creates SharedKVMemory)
        super().__init__(
            kvrms=agents,
            tick_interval_ms=config.tick_interval_ms,
            max_memory_entries=10000,
            memory_ttl=config.memory_ttl,
        )

        # Meta-learning
        if config.enable_meta_learning:
            self.meta_learner = MetaLearner(self.memory)
        else:
            self.meta_learner = None

        # Emergence detection (panel recommendation)
        self.emergence_signals: List[EmergenceSignal] = []
        self.emergence_callbacks: List[Callable[[EmergenceSignal], None]] = []

        # Generation tracking
        self.generation_history: List[Dict] = []
        self.fitness_history: Dict[str, List[float]] = defaultdict(list)

        # Event log for visualization
        self.event_log: List[Dict] = []

        # Pending human approvals
        self.pending_overrides: List[Dict] = []

    def _create_agents(self, config: OuroborosConfig) -> List[KVRMBase]:
        """Create the agent swarm."""
        agents = []

        # Competitive agents (V6 Guided Chaos style)
        for i in range(config.num_competitive_agents):
            agent_config = AgentConfig(
                name=f"competitive_{i}",
                mode="competitive",
                llm_model=config.llm_model,
                energy_budget=100.0,
            )
            agents.append(AgentKVRM(agent_config))

        # Cooperative agents (V7 Phoenix Forge style)
        for i in range(config.num_cooperative_agents):
            agent_config = AgentConfig(
                name=f"cooperative_{i}",
                mode="cooperative",
                llm_model=config.llm_model,
                energy_budget=100.0,
            )
            agents.append(AgentKVRM(agent_config))

        return agents

    def set_problem(self, problem: Dict[str, Any]) -> None:
        """
        Set the problem for agents to solve.

        Args:
            problem: Problem specification with description, test cases, etc.
        """
        self.inject(
            key="problem:current",
            value=problem,
            ttl=self.ouro_config.memory_ttl * 2,  # Long-lived
        )
        self._log_event("problem_set", {"problem": problem.get("description", "")[:100]})

    def run_generation(self) -> Dict[str, Any]:
        """
        Run one generation of evolution.

        Returns:
            Generation summary with fitness scores, narrator observations, etc.
        """
        self.generation += 1
        gen_start = time.time()

        self._log_event("generation_start", {"generation": self.generation})

        # Run a tick (all KVRMs sense-think-act)
        activations = self.tick()

        # Collect results
        solutions = self._collect_solutions()
        narrator_output = self._get_narrator_output()

        # Meta-learning
        if self.meta_learner:
            self._update_meta_learning(solutions)

        # Emergence detection (panel recommendation)
        if self.ouro_config.enable_emergence_detection:
            self._detect_emergence(solutions)

        # Check for override requests (need human approval)
        override_requests = self._check_override_requests()

        # Track fitness history
        for sol in solutions:
            agent_name = sol.get("agent", "unknown")
            fitness = sol.get("fitness", 0)
            self.fitness_history[agent_name].append(fitness)

        # Build generation summary
        gen_summary = {
            "generation": self.generation,
            "activations": activations,
            "solutions": solutions,
            "narrator_observation": narrator_output.get("narrator:observation", ""),
            "narrator_guidance": narrator_output.get("narrator:guidance", ""),
            "patterns_detected": narrator_output.get("narrator:patterns", []),
            "override_requests": override_requests,
            "emergence_signals": [e.__dict__ for e in self.emergence_signals[-3:]],
            "meta_learning": self.meta_learner.get_summary() if self.meta_learner else None,
            "duration_ms": (time.time() - gen_start) * 1000,
            "memory_snapshot": self.memory.get_snapshot(),
        }

        self.generation_history.append(gen_summary)

        # Best solution this generation
        if solutions:
            best = max(solutions, key=lambda s: s.get("fitness", 0))
            gen_summary["best_agent"] = best.get("agent")
            gen_summary["best_fitness"] = best.get("fitness", 0)

            # Record to meta-learner
            if self.meta_learner:
                self.meta_learner.record_solution(
                    agent_id=best.get("agent", "unknown"),
                    generation=self.generation,
                    solution=best.get("code", ""),
                    fitness=best.get("fitness", 0),
                )

        self._log_event("generation_complete", {
            "generation": self.generation,
            "best_fitness": gen_summary.get("best_fitness", 0),
            "activations": sum(1 for v in activations.values() if v),
        })

        return gen_summary

    def run_evolution(
        self,
        max_generations: Optional[int] = None,
        target_fitness: float = 0.95,
        on_generation: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run full evolution experiment.

        Args:
            max_generations: Override config max_generations
            target_fitness: Stop if fitness exceeds this
            on_generation: Callback after each generation

        Returns:
            Final experiment results
        """
        max_gen = max_generations or self.ouro_config.max_generations
        start_time = time.time()

        self._log_event("evolution_start", {
            "max_generations": max_gen,
            "target_fitness": target_fitness,
            "num_agents": len(self.kvrms),
        })

        for _ in range(max_gen):
            gen_result = self.run_generation()

            # Callback for monitoring
            if on_generation:
                on_generation(gen_result)

            # Check for pending overrides that need human attention
            if gen_result.get("override_requests"):
                self._log_event("override_pending", {
                    "count": len(gen_result["override_requests"]),
                })

            # Check for target fitness
            best_fitness = gen_result.get("best_fitness", 0)
            if best_fitness >= target_fitness:
                self._log_event("target_reached", {
                    "generation": self.generation,
                    "fitness": best_fitness,
                })
                break

        # Final summary
        total_time = time.time() - start_time

        return {
            "generations_run": self.generation,
            "total_time_seconds": total_time,
            "final_fitness": self._get_best_fitness(),
            "fitness_history": dict(self.fitness_history),
            "emergence_signals": [e.__dict__ for e in self.emergence_signals],
            "meta_learning_summary": self.meta_learner.get_summary() if self.meta_learner else None,
            "event_log": self.event_log,
            "generation_history": self.generation_history,
        }

    def _collect_solutions(self) -> List[Dict]:
        """Collect all agent solutions from memory."""
        solutions = []
        entries = self.observe("solution:*")

        for entry in entries:
            # Get source - KVEntry uses source_kvrm attribute
            source = getattr(entry, 'source_kvrm', 'unknown')
            solutions.append({
                "agent": source,
                "fitness": entry.value.get("fitness", 0),
                "code": entry.value.get("code", ""),
                "generation": entry.value.get("generation", 0),
                "mode": entry.value.get("mode", "unknown"),
            })

        return solutions

    def _get_narrator_output(self) -> Dict[str, Any]:
        """Get narrator's latest outputs."""
        output = {}

        for key in ["narrator:guidance", "narrator:observation", "narrator:patterns", "narrator:status"]:
            entries = self.observe(key)
            if entries:
                output[key] = entries[0].value

        return output

    def _update_meta_learning(self, solutions: List[Dict]) -> None:
        """Update meta-learner with generation results."""
        if not self.meta_learner:
            return

        for sol in solutions:
            # Create learning signal
            previous_fitness = 0
            agent_name = sol.get("agent", "")
            history = self.fitness_history.get(agent_name)
            if history and len(history) >= 2:
                previous_fitness = history[-2]

            signal = LearningSignal(
                agent_id=agent_name or "unknown",
                generation=self.generation,
                fitness_before=previous_fitness,
                fitness_after=sol.get("fitness", 0),
                action_type=sol.get("mode", "unknown"),
                action_details=f"solution_gen_{self.generation}",
                success=sol.get("fitness", 0) > previous_fitness,
            )
            self.meta_learner.record_signal(signal)

    def _detect_emergence(self, solutions: List[Dict]) -> None:
        """
        Detect emergent patterns (panel recommendation from ChatGPT).

        Looks for:
        - Convergence: agents clustering on similar solutions
        - Cooperation: cooperative agents outperforming competitive
        - Innovation: sudden fitness jumps
        - Stagnation: fitness plateau
        """
        if len(solutions) < 2:
            return

        fitnesses = [s.get("fitness", 0) for s in solutions]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        # Check for convergence
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        if variance < 0.01 and avg_fitness > 0.5:
            signal = EmergenceSignal(
                signal_type="convergence",
                agents_involved=[s.get("agent", "") for s in solutions],
                strength=1.0 - variance,
                description="Agents converging on similar fitness levels",
                generation=self.generation,
            )
            self.emergence_signals.append(signal)
            self._notify_emergence(signal)

        # Check for cooperation advantage
        competitive = [s for s in solutions if s.get("mode") == "competitive"]
        cooperative = [s for s in solutions if s.get("mode") == "cooperative"]

        if competitive and cooperative:
            comp_avg = sum(s.get("fitness", 0) for s in competitive) / len(competitive)
            coop_avg = sum(s.get("fitness", 0) for s in cooperative) / len(cooperative)

            if coop_avg > comp_avg * 1.2:  # Cooperation 20% better
                signal = EmergenceSignal(
                    signal_type="cooperation",
                    agents_involved=[s.get("agent", "") for s in cooperative],
                    strength=(coop_avg - comp_avg) / max(comp_avg, 0.01),
                    description=f"Cooperative agents outperforming ({coop_avg:.2f} vs {comp_avg:.2f})",
                    generation=self.generation,
                )
                self.emergence_signals.append(signal)
                self._notify_emergence(signal)

        # Check for innovation (sudden fitness jump)
        if self.generation > 1 and self.generation_history:
            prev_best = self.generation_history[-1].get("best_fitness", 0)
            curr_best = max(fitnesses)
            if curr_best > prev_best * 1.5 and prev_best > 0.1:
                signal = EmergenceSignal(
                    signal_type="innovation",
                    agents_involved=[max(solutions, key=lambda s: s.get("fitness", 0)).get("agent", "")],
                    strength=(curr_best - prev_best) / prev_best,
                    description=f"Fitness jumped from {prev_best:.2f} to {curr_best:.2f}",
                    generation=self.generation,
                )
                self.emergence_signals.append(signal)
                self._notify_emergence(signal)

        # Check for stagnation
        if len(self.generation_history) >= 5:
            recent_best = [g.get("best_fitness", 0) for g in self.generation_history[-5:]]
            if max(recent_best) - min(recent_best) < 0.02:
                signal = EmergenceSignal(
                    signal_type="stagnation",
                    agents_involved=[s.get("agent", "") for s in solutions],
                    strength=1.0,
                    description="Fitness stagnant for 5 generations",
                    generation=self.generation,
                )
                self.emergence_signals.append(signal)
                self._notify_emergence(signal)

    def _notify_emergence(self, signal: EmergenceSignal) -> None:
        """Notify callbacks of emergence signal."""
        for callback in self.emergence_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.warning(f"Emergence callback error: {e}")

        self._log_event("emergence_detected", {
            "type": signal.signal_type,
            "strength": signal.strength,
            "description": signal.description,
        })

    def on_emergence(self, callback: Callable[[EmergenceSignal], None]) -> None:
        """Register callback for emergence detection."""
        self.emergence_callbacks.append(callback)

    def _check_override_requests(self) -> List[Dict]:
        """Check for narrator override requests needing human approval."""
        entries = self.observe("narrator:override_request")
        requests = []

        for entry in entries:
            if entry.value.get("status") == "PENDING_HUMAN_APPROVAL":
                requests.append(entry.value)

        self.pending_overrides = requests
        return requests

    def approve_override(self, request_id: str, approver: str = "human") -> Dict:
        """Human approves an override request."""
        result = self.narrator.approve_override(request_id, approver)
        self._log_event("override_approved", {
            "request_id": request_id,
            "approver": approver,
        })
        return result

    def reject_override(self, request_id: str, reason: str = "") -> Dict:
        """Human rejects an override request."""
        result = self.narrator.reject_override(request_id, reason)
        self._log_event("override_rejected", {
            "request_id": request_id,
            "reason": reason,
        })
        return result

    def get_pending_overrides(self) -> List[Dict]:
        """Get all pending override requests."""
        return self.pending_overrides

    def _get_best_fitness(self) -> float:
        """Get the best fitness across all agents."""
        if not self.fitness_history:
            return 0.0

        best = 0.0
        for history in self.fitness_history.values():
            if history:
                best = max(best, max(history))
        return best

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event for visualization."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "generation": self.generation,
            "event_type": event_type,
            **data,
        }
        self.event_log.append(event)
        logger.info(f"[Gen {self.generation}] {event_type}: {json.dumps(data, default=str)[:200]}")

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data for visualization (not canned templates).

        Returns actual data patterns from SharedKVMemory.
        """
        # Get current solutions from agents
        agent_solutions = {}
        for kvrm in self.kvrms:
            if hasattr(kvrm, 'name') and kvrm.name != "narrator":
                solution_data = {}
                # Try to get solution from memory using read method
                sol_entry = self.memory.read(f"solution:{kvrm.name}")
                if sol_entry:
                    solution_data['code'] = sol_entry.value.get('code', str(sol_entry.value))
                # Try to get reasoning
                reason_entry = self.memory.read(f"reasoning:{kvrm.name}")
                if reason_entry:
                    solution_data['reasoning'] = reason_entry.value.get('reasoning', str(reason_entry.value))
                # Fallback: check agent state
                if not solution_data.get('code') and hasattr(kvrm, 'current_solution'):
                    solution_data['code'] = kvrm.current_solution
                if not solution_data.get('reasoning') and hasattr(kvrm, 'last_reasoning'):
                    solution_data['reasoning'] = kvrm.last_reasoning
                agent_solutions[kvrm.name] = solution_data

        return {
            "fitness_history": dict(self.fitness_history),
            "emergence_signals": [
                {
                    "type": e.signal_type,
                    "strength": e.strength,
                    "generation": e.generation,
                    "agents": e.agents_involved,
                    "description": getattr(e, 'description', f"{e.signal_type} detected"),
                }
                for e in self.emergence_signals
            ],
            "memory_snapshot": self.memory.get_snapshot(),
            "meta_learning": self.meta_learner.get_summary() if self.meta_learner else None,
            "agent_modes": {
                kvrm.name: getattr(kvrm, 'agent_config', {}).mode
                if hasattr(kvrm, 'agent_config') else "narrator"
                for kvrm in self.kvrms
            },
            "event_timeline": self.event_log,
            "narrator_status": self.narrator.get_status(),
            "pending_overrides": self.pending_overrides,
            "generation_summaries": [
                {
                    "generation": g.get("generation"),
                    "best_fitness": g.get("best_fitness", 0),
                    "best_agent": g.get("best_agent"),
                    "patterns": g.get("patterns_detected", []),
                }
                for g in self.generation_history
            ],
            "problem": self.memory.read("problem:current").value if self.memory.read("problem:current") else None,
            "current_problem": self.memory.read("problem:current").value if self.memory.read("problem:current") else None,
            "agent_solutions": agent_solutions,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get complete organism status."""
        return {
            "generation": self.generation,
            "num_agents": len(self.kvrms) - 1,  # Exclude narrator
            "best_fitness": self._get_best_fitness(),
            "emergence_signals_count": len(self.emergence_signals),
            "pending_overrides": len(self.pending_overrides),
            "meta_learning_enabled": self.meta_learner is not None,
            "ecosystem_stats": self.get_stats(),
        }
