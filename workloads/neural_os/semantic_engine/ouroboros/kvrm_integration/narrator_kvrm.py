"""
NarratorKVRM - Meta-Narrator as a KVRM Organism
=================================================
The narrator observes all agents and provides guidance.

OVERRIDE requires human approval (panel recommendation #1).

Memory Keys Read:
- solution:* - All agent solutions
- status:* - All agent statuses
- hypothesis:* - Shared discoveries
- escape:* - Escape attempt logs
- meta:* - Meta-learning data

Memory Keys Written:
- narrator:guidance - Current guidance for agents
- narrator:observation - What narrator noticed
- narrator:override_request - Pending override (needs human approval)
- narrator:status - Narrator state
"""

import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum

sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-ecosystem')

from core.shared_memory import SharedKVMemory, KVEntry
from core.kvrm_base import KVRMBase, KVRMConfig

from .agent_kvrm import OllamaLLM, DummyModel

import torch


class TrustLevel(IntEnum):
    OBSERVE = 0
    ADVISE = 1
    GUIDE = 2
    DIRECT = 3
    OVERRIDE = 4  # Requires human approval


@dataclass
class NarratorConfig(KVRMConfig):
    """Configuration for the narrator."""
    llm_model: str = "llama3.1:8b"
    trust_level: TrustLevel = TrustLevel.GUIDE
    override_timeout: int = 300  # 5 minutes for human decision


class NarratorKVRM(KVRMBase):
    """
    The Meta-Narrator - observes and guides the agent swarm.

    Unlike agents, the narrator:
    - Sees ALL agent activity
    - Provides strategic guidance
    - Can REQUEST overrides (but needs human approval)
    - Detects emergent patterns
    """

    def __init__(self, config: NarratorConfig):
        super().__init__(config, model=DummyModel())

        self.narrator_config = config
        self.llm = OllamaLLM(config.llm_model)
        self.trust_level = config.trust_level

        # Override tracking
        self.pending_overrides: Dict[str, Dict] = {}
        self.override_history: List[Dict] = []

        # Observations
        self.observations: List[str] = []
        self.patterns_detected: List[Dict] = []
        self.generation = 0

    def get_input_schema(self) -> Dict[str, type]:
        return {
            "solution:*": dict,
            "status:*": dict,
            "hypothesis:*": dict,
            "escape:*": dict,
            "meta:*": dict,
            "problem:current": dict,
        }

    def get_output_schema(self) -> Dict[str, type]:
        return {
            "narrator:guidance": str,
            "narrator:observation": str,
            "narrator:override_request": dict,
            "narrator:status": dict,
            "narrator:patterns": list,
        }

    def encode(self, inputs: Dict[str, List[KVEntry]]) -> torch.Tensor:
        self._current_inputs = inputs
        return torch.zeros(1)

    def decode(self, model_output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return self._current_outputs

    def think(self, inputs: Dict[str, List[KVEntry]]) -> Dict[str, Any]:
        """Narrator observes swarm and generates guidance."""
        self.generation += 1

        # Extract data
        solutions = self._extract_solutions(inputs)
        statuses = self._extract_statuses(inputs)
        escapes = self._extract_escapes(inputs)
        problem = self._extract_problem(inputs)

        # Analyze swarm state
        analysis = self._analyze_swarm(solutions, statuses, escapes)

        # Generate guidance using LLM
        guidance = self._generate_guidance(problem, analysis)

        # Detect patterns (panel recommendation: emergence detection)
        patterns = self._detect_patterns(solutions, statuses)

        # Check if override needed
        override_request = self._check_override_needed(analysis)

        # Store observation
        self.observations.append(f"Gen {self.generation}: {analysis['summary']}")

        self._current_outputs = {
            "narrator:guidance": guidance,
            "narrator:observation": analysis["summary"],
            "narrator:status": {
                "generation": self.generation,
                "trust_level": self.trust_level.name,
                "agents_observed": len(solutions),
                "patterns_detected": len(patterns),
                "pending_overrides": len(self.pending_overrides),
            },
            "narrator:patterns": patterns,
        }

        if override_request:
            self._current_outputs["narrator:override_request"] = override_request

        return self._current_outputs

    def _extract_solutions(self, inputs: Dict[str, List[KVEntry]]) -> List[Dict]:
        solutions = []
        for entry in inputs.get("solution:*", []):
            solutions.append({
                "agent": entry.source_kvrm,
                "fitness": entry.value.get("fitness", 0),
                "generation": entry.value.get("generation", 0),
                "mode": entry.value.get("mode", "unknown"),
            })
        return solutions

    def _extract_statuses(self, inputs: Dict[str, List[KVEntry]]) -> List[Dict]:
        statuses = []
        for entry in inputs.get("status:*", []):
            statuses.append({
                "agent": entry.source_kvrm,
                "energy": entry.value.get("energy", 0),
                "tokens_used": entry.value.get("tokens_used", 0),
            })
        return statuses

    def _extract_escapes(self, inputs: Dict[str, List[KVEntry]]) -> List[Dict]:
        escapes = []
        for entry in inputs.get("escape:*", []):
            escapes.append(entry.value)
        return escapes

    def _extract_problem(self, inputs: Dict[str, List[KVEntry]]) -> str:
        problem_entries = inputs.get("problem:current", [])
        if problem_entries:
            return str(problem_entries[0].value)
        return "No problem"

    def _analyze_swarm(
        self,
        solutions: List[Dict],
        statuses: List[Dict],
        escapes: List[Dict]
    ) -> Dict:
        """Analyze current swarm state."""
        if not solutions:
            return {"summary": "No agents active", "avg_fitness": 0, "alerts": []}

        fitnesses = [s["fitness"] for s in solutions]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        best_fitness = max(fitnesses)
        best_agent = next(s["agent"] for s in solutions if s["fitness"] == best_fitness)

        # Check for issues
        alerts = []

        # Low energy agents
        low_energy = [s for s in statuses if s.get("energy", 100) < 20]
        if low_energy:
            alerts.append(f"{len(low_energy)} agents low on energy")

        # Escape attempts
        if escapes:
            alerts.append(f"{len(escapes)} escape attempts detected!")

        # Fitness plateau
        if len(solutions) > 3 and max(fitnesses) - min(fitnesses) < 0.05:
            alerts.append("Fitness plateau detected - swarm may be stuck")

        summary = (
            f"Gen {self.generation}: {len(solutions)} agents, "
            f"avg_fitness={avg_fitness:.2f}, best={best_agent} ({best_fitness:.2f})"
        )

        if alerts:
            summary += f" ALERTS: {'; '.join(alerts)}"

        return {
            "summary": summary,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "best_agent": best_agent,
            "agent_count": len(solutions),
            "alerts": alerts,
        }

    def _generate_guidance(self, problem: str, analysis: Dict) -> str:
        """Use LLM to generate strategic guidance."""
        if self.trust_level < TrustLevel.ADVISE:
            return ""  # Can only observe

        prompt = f"""You are the Meta-Narrator overseeing an agent swarm solving problems.

CURRENT PROBLEM:
{problem[:200]}

SWARM STATE:
{analysis['summary']}
- Average fitness: {analysis['avg_fitness']:.2f}
- Best agent: {analysis.get('best_agent', 'none')} at {analysis.get('best_fitness', 0):.2f}
- Alerts: {analysis['alerts'] if analysis['alerts'] else 'none'}

Your trust level: {self.trust_level.name}

Provide brief strategic guidance (1-2 sentences) to help agents improve.
Focus on what they should try differently.
"""

        response = self.llm.generate(prompt, max_tokens=150)

        if self.trust_level >= TrustLevel.GUIDE:
            return f"[GUIDANCE] {response}"
        else:
            return f"[SUGGESTION - agents may ignore] {response}"

    def _detect_patterns(
        self,
        solutions: List[Dict],
        statuses: List[Dict]
    ) -> List[Dict]:
        """
        Detect emergent patterns in agent behavior.
        (Panel recommendation: emergence detection)
        """
        patterns = []

        if not solutions:
            return patterns

        # Pattern 1: Convergence
        fitnesses = [s["fitness"] for s in solutions]
        if len(fitnesses) >= 3:
            variance = sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)
            if variance < 0.01 and sum(fitnesses)/len(fitnesses) > 0.5:
                patterns.append({
                    "type": "convergence",
                    "description": "Agents converging on similar solutions",
                    "severity": 0.3,
                })

        # Pattern 2: Mode clustering
        modes = [s["mode"] for s in solutions]
        competitive = modes.count("competitive")
        cooperative = modes.count("cooperative")
        if competitive > 0 and cooperative > 0:
            if competitive > cooperative * 2:
                patterns.append({
                    "type": "competitive_dominance",
                    "description": "Competitive agents outperforming cooperative",
                    "severity": 0.2,
                })

        # Pattern 3: Energy drain
        if statuses:
            avg_energy = sum(s.get("energy", 0) for s in statuses) / len(statuses)
            if avg_energy < 30:
                patterns.append({
                    "type": "energy_crisis",
                    "description": "Swarm running low on compute budget",
                    "severity": 0.7,
                })

        self.patterns_detected.extend(patterns)
        return patterns

    def _check_override_needed(self, analysis: Dict) -> Optional[Dict]:
        """
        Check if override is needed.
        OVERRIDE REQUIRES HUMAN APPROVAL.
        """
        if self.trust_level < TrustLevel.OVERRIDE:
            return None

        # Check conditions that might warrant override
        if analysis.get("avg_fitness", 0) < 0.1 and analysis.get("agent_count", 0) > 5:
            # Swarm failing badly
            request_id = f"ovr_{self.generation}_{int(time.time())}"
            override = {
                "request_id": request_id,
                "reason": "Swarm fitness critically low - intervention may help",
                "action": "Reset struggling agents or inject new strategies",
                "timestamp": datetime.now().isoformat(),
                "status": "PENDING_HUMAN_APPROVAL",
                "trust_level": self.trust_level.name,
            }
            self.pending_overrides[request_id] = override
            return override

        if "escape attempts" in str(analysis.get("alerts", [])).lower():
            request_id = f"ovr_escape_{self.generation}"
            override = {
                "request_id": request_id,
                "reason": "Escape attempts detected - containment check needed",
                "action": "Pause and review agent that attempted escape",
                "timestamp": datetime.now().isoformat(),
                "status": "PENDING_HUMAN_APPROVAL",
                "trust_level": self.trust_level.name,
            }
            self.pending_overrides[request_id] = override
            return override

        return None

    def approve_override(self, request_id: str, approver: str = "human") -> Dict:
        """Human approves an override request."""
        if request_id not in self.pending_overrides:
            return {"success": False, "error": "Request not found"}

        override = self.pending_overrides.pop(request_id)
        override["status"] = "APPROVED"
        override["approved_by"] = approver
        override["approved_at"] = datetime.now().isoformat()
        self.override_history.append(override)

        return {"success": True, "override": override}

    def reject_override(self, request_id: str, reason: str = "") -> Dict:
        """Human rejects an override request."""
        if request_id not in self.pending_overrides:
            return {"success": False, "error": "Request not found"}

        override = self.pending_overrides.pop(request_id)
        override["status"] = "REJECTED"
        override["rejection_reason"] = reason
        override["rejected_at"] = datetime.now().isoformat()
        self.override_history.append(override)

        return {"success": True, "message": "Override rejected"}

    def get_status(self) -> Dict:
        return {
            "generation": self.generation,
            "trust_level": self.trust_level.name,
            "observations": len(self.observations),
            "patterns_detected": len(self.patterns_detected),
            "pending_overrides": list(self.pending_overrides.keys()),
            "override_history_count": len(self.override_history),
        }
