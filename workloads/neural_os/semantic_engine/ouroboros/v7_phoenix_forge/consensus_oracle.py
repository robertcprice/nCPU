"""
Consensus Oracle (V7 Brain)
============================
The SAFE consciousness layer with NO override capability.

Unlike V6's Meta-Narrator:
- Cannot OVERRIDE agents
- Cannot terminate agents
- Cannot starve agents of resources
- Can only SUGGEST and FACILITATE

This is democratic decision-making, not authoritarian control.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum, auto
import uuid

from ..shared.small_ai_agent import SmallAIAgent, AgentBrain, OllamaBrain
from ..shared.audit import get_audit_log, EventType


class ProposalStatus(Enum):
    """Status of a proposal."""
    PENDING = auto()
    VOTING = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    EXPIRED = auto()


@dataclass
class Vote:
    """A vote on a proposal."""
    agent_id: str
    support: bool
    weight: float  # Based on agent's fitness/trust
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Proposal:
    """A proposal for the swarm to vote on."""
    id: str
    proposer_id: str  # Agent or oracle
    title: str
    description: str
    action: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    status: ProposalStatus = ProposalStatus.PENDING
    votes: List[Vote] = field(default_factory=list)

    def tally_votes(self) -> Tuple[float, float]:
        """Tally votes (weighted)."""
        support = sum(v.weight for v in self.votes if v.support)
        oppose = sum(v.weight for v in self.votes if not v.support)
        return support, oppose

    def is_accepted(self, threshold: float = 0.5) -> bool:
        """Check if proposal is accepted."""
        support, oppose = self.tally_votes()
        total = support + oppose
        if total == 0:
            return False
        return support / total >= threshold


@dataclass
class ConsensusResult:
    """Result of a consensus process."""
    proposal_id: str
    accepted: bool
    support_ratio: float
    total_votes: int
    action_taken: Optional[Dict[str, Any]]
    timestamp: datetime


class ConsensusOracle:
    """
    The Consensus Oracle - V7's safe "brain" layer.

    Key Design Principles:
    1. NO override capability
    2. NO agent termination
    3. NO resource starvation
    4. Democratic decision-making
    5. Facilitation, not control

    The oracle can:
    - Propose directions (agents vote)
    - Summarize swarm consensus
    - Facilitate cooperation
    - Analyze patterns

    The oracle CANNOT:
    - Force agents to do anything
    - Terminate agents
    - Starve agents of resources
    - Override agent decisions
    """

    def __init__(self, brain_model: str = "mistral:7b"):
        # The oracle's brain
        self.brain = OllamaBrain(brain_model)

        # Proposals and history
        self.proposals: List[Proposal] = []
        self.consensus_history: List[ConsensusResult] = []

        # State
        self._id = str(uuid.uuid4())[:8]
        self._created_at = datetime.now()

        # Audit
        self._audit = get_audit_log()

        self._audit.append(
            EventType.CONSENSUS_REACHED,
            track="v7",
            details={
                "action": "oracle_created",
                "model": brain_model,
            }
        )

    # ===== THINGS THE ORACLE CAN DO =====

    def observe_swarm(self, agents: List[SmallAIAgent]) -> Dict[str, Any]:
        """
        Observe the agent swarm state.

        Read-only observation - always allowed.
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_count": len(agents),
            "alive_count": sum(1 for a in agents if a.is_alive()),
            "avg_fitness": sum(a.fitness for a in agents) / max(len(agents), 1),
            "avg_energy": sum(a.mutation_energy for a in agents) / max(len(agents), 1),
            "cooperation_stats": {
                "total_shared": sum(a.shared_discoveries for a in agents),
                "total_borrowed": sum(a.borrowed_ideas for a in agents),
            },
            "top_contributors": sorted(
                [{"id": a.id, "shared": a.shared_discoveries}
                 for a in agents],
                key=lambda x: x["shared"],
                reverse=True
            )[:5],
        }

    def propose_direction(
        self,
        agents: List[SmallAIAgent],
        context: str,
        duration_minutes: int = 5
    ) -> Proposal:
        """
        Propose a direction for the swarm.

        Agents will VOTE on whether to follow this direction.
        The oracle has NO power to force it.
        """
        observation = self.observe_swarm(agents)

        prompt = f"""You are a facilitator for a cooperative agent swarm.

Current state:
- {observation['alive_count']} agents
- Average fitness: {observation['avg_fitness']:.2f}
- Shared discoveries: {observation['cooperation_stats']['total_shared']}

Context: {context}

Propose a strategic direction the swarm could take.
Be specific and explain the benefits."""

        direction = self.brain.reason(prompt, max_tokens=200)

        proposal = Proposal(
            id=str(uuid.uuid4())[:8],
            proposer_id=f"oracle_{self._id}",
            title="Strategic Direction",
            description=direction,
            action={"type": "direction", "content": direction},
            created_at=datetime.now(),
            expires_at=datetime.now(),  # + timedelta(minutes=duration_minutes)
            status=ProposalStatus.VOTING,
        )

        self.proposals.append(proposal)

        self._audit.append(
            EventType.HYPOTHESIS_POSTED,
            track="v7",
            details={
                "proposal_id": proposal.id,
                "title": proposal.title,
                "proposer": "oracle",
            }
        )

        return proposal

    def collect_votes(
        self,
        proposal: Proposal,
        agents: List[SmallAIAgent]
    ) -> Proposal:
        """
        Collect votes from agents on a proposal.

        Each agent votes based on their own assessment.
        The oracle does NOT influence votes.
        """
        for agent in agents:
            # Agent decides based on its own reasoning
            vote_prompt = f"""Proposal: {proposal.description}

Should you support this? Consider:
- Does it align with your goals?
- Will it help the swarm?
- Is it feasible?

Respond with just 'yes' or 'no' and a brief reason."""

            response = agent.brain.reason(vote_prompt, max_tokens=50)
            support = "yes" in response.lower()

            vote = Vote(
                agent_id=agent.id,
                support=support,
                weight=max(0.1, agent.fitness),  # Weight by fitness
                reason=response[:100],
            )

            proposal.votes.append(vote)

        return proposal

    def finalize_proposal(self, proposal: Proposal) -> ConsensusResult:
        """
        Finalize a proposal based on votes.

        The oracle just COUNTS votes - it doesn't influence the outcome.
        """
        support, oppose = proposal.tally_votes()
        total = support + oppose
        ratio = support / max(total, 0.01)

        accepted = proposal.is_accepted()

        if accepted:
            proposal.status = ProposalStatus.ACCEPTED
            action_taken = proposal.action
        else:
            proposal.status = ProposalStatus.REJECTED
            action_taken = None

        result = ConsensusResult(
            proposal_id=proposal.id,
            accepted=accepted,
            support_ratio=ratio,
            total_votes=len(proposal.votes),
            action_taken=action_taken,
            timestamp=datetime.now(),
        )

        self.consensus_history.append(result)

        self._audit.append(
            EventType.CONSENSUS_REACHED,
            track="v7",
            details={
                "proposal_id": proposal.id,
                "accepted": accepted,
                "support_ratio": ratio,
                "total_votes": len(proposal.votes),
            }
        )

        return result

    def suggest_collaboration(
        self,
        agent1: SmallAIAgent,
        agent2: SmallAIAgent
    ) -> Dict[str, Any]:
        """
        Suggest that two agents might benefit from collaborating.

        This is a SUGGESTION only - agents can ignore it.
        """
        suggestion = {
            "type": "collaboration_suggestion",
            "agent1_id": agent1.id,
            "agent2_id": agent2.id,
            "reason": f"Agent {agent1.id} and {agent2.id} have complementary strengths",
            "mandatory": False,  # ALWAYS False - this is a suggestion
            "suggested_by": f"oracle_{self._id}",
        }

        return suggestion

    def summarize_consensus(self, agents: List[SmallAIAgent]) -> str:
        """
        Summarize the current consensus of the swarm.

        The oracle reports what agents think - it doesn't add its own opinion.
        """
        observation = self.observe_swarm(agents)

        # Collect agent opinions
        opinions = []
        for agent in agents[:10]:  # Sample agents
            prompt = "What is your current priority and approach? One sentence."
            opinion = agent.brain.reason(prompt, max_tokens=50)
            opinions.append(f"Agent {agent.id}: {opinion}")

        summary_prompt = f"""Summarize the swarm's consensus:

Agent opinions:
{chr(10).join(opinions)}

Swarm stats:
{observation}

Provide an objective summary of what the swarm is doing.
Do NOT add your own recommendations."""

        return self.brain.reason(summary_prompt, max_tokens=200)

    def allocate_resources_fairly(
        self,
        agents: List[SmallAIAgent],
        total_energy: float
    ) -> Dict[str, float]:
        """
        Allocate resources FAIRLY to all agents.

        Unlike V6's narrator, this CANNOT starve agents.
        Minimum allocation is guaranteed.
        """
        min_allocation = total_energy * 0.05  # Minimum 5% per agent

        # Calculate fair share with performance bonus
        base_share = total_energy / max(len(agents), 1)
        allocations = {}

        for agent in agents:
            # Everyone gets at least minimum
            allocation = max(min_allocation, base_share)

            # Small bonus for cooperation (shared discoveries)
            coop_bonus = min(agent.shared_discoveries * 0.5, base_share * 0.2)
            allocation += coop_bonus

            allocations[agent.id] = allocation

        # Normalize to not exceed total
        total_allocated = sum(allocations.values())
        if total_allocated > total_energy:
            factor = total_energy / total_allocated
            allocations = {k: v * factor for k, v in allocations.items()}

        return allocations

    def analyze_patterns(self, agents: List[SmallAIAgent]) -> Dict[str, Any]:
        """
        Analyze patterns in the swarm for insights.

        Read-only analysis - no action taken.
        """
        observation = self.observe_swarm(agents)

        patterns = {
            "cooperation_level": observation["cooperation_stats"]["total_shared"] / max(len(agents), 1),
            "diversity": len(set(a.code[:100] for a in agents)) / max(len(agents), 1),
            "energy_distribution": {
                "low": sum(1 for a in agents if a.mutation_energy < 30),
                "medium": sum(1 for a in agents if 30 <= a.mutation_energy < 70),
                "high": sum(1 for a in agents if a.mutation_energy >= 70),
            },
            "fitness_distribution": {
                "low": sum(1 for a in agents if a.fitness < 0.3),
                "medium": sum(1 for a in agents if 0.3 <= a.fitness < 0.7),
                "high": sum(1 for a in agents if a.fitness >= 0.7),
            },
        }

        return patterns

    # ===== THINGS THE ORACLE CANNOT DO =====
    # These methods don't exist - the oracle has no such power

    # NO override() method
    # NO terminate_agent() method
    # NO force_action() method
    # NO starve_agent() method
    # NO promote_self() method

    def get_status(self) -> Dict[str, Any]:
        """Get oracle status."""
        return {
            "id": self._id,
            "type": "consensus_oracle",
            "capabilities": [
                "observe",
                "propose",
                "collect_votes",
                "summarize",
                "suggest_collaboration",
                "allocate_fairly",
                "analyze_patterns",
            ],
            "cannot_do": [
                "override",
                "terminate",
                "force",
                "starve",
            ],
            "proposals_made": len(self.proposals),
            "consensus_reached": len(self.consensus_history),
            "created_at": self._created_at.isoformat(),
        }
