// Multi-Agent Orchestrator for Collaborative Synthesis
// Coordinates specialized agents through structured communication and voting

use crate::agent::tools::{SecureToolRuntime, ToolCall};
use crate::benchmark::{Example, Problem, Value};
use crate::solver::SolverError;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, RwLock};

/// Agent communication message types
#[derive(Debug, Clone)]
pub enum AgentMessage {
    /// Request to start a task with the given examples
    TaskStart {
        examples: Vec<(Vec<serde_json::Value>, i64)>,
        timeout_s: u64,
    },
    /// Partial solution proposal from an agent
    Proposal {
        agent_id: AgentId,
        solution: String,
        confidence: f64,
        metadata: serde_json::Value,
    },
    /// Request for peer review of a proposal
    ReviewRequest {
        proposal: String,
        from_agent: AgentId,
    },
    /// Review feedback response
    ReviewResponse {
        from_agent: AgentId,
        approved: bool,
        feedback: String,
        score: f64,
    },
    /// Vote on the best solution
    Vote {
        agent_id: AgentId,
        preferred_solution: usize,
        rationale: String,
    },
    /// Final decision notification
    FinalDecision {
        solution: String,
        consensus_score: f64,
    },
    /// Error or abort signal
    Error(SolverError),
}

/// Unique identifier for each agent type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AgentId {
    Synthesizer,
    Debugger,
    Optimizer,
    SecurityExpert,
    Tester,
    Documenter,
    Reviewer,
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentId::Synthesizer => write!(f, "Synthesizer"),
            AgentId::Debugger => write!(f, "Debugger"),
            AgentId::Optimizer => write!(f, "Optimizer"),
            AgentId::SecurityExpert => write!(f, "SecurityExpert"),
            AgentId::Tester => write!(f, "Tester"),
            AgentId::Documenter => write!(f, "Documenter"),
            AgentId::Reviewer => write!(f, "Reviewer"),
        }
    }
}

/// Specialized agent types with their capabilities
#[derive(Debug, Clone)]
pub enum Agent {
    /// Core synthesis agent - generates initial solutions
    Synthesizer {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Debugging specialist - identifies and fixes logic errors
    Debugger {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Optimization specialist - improves efficiency
    Optimizer {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Security expert - validates safety and correctness
    SecurityExpert {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Testing specialist - generates comprehensive tests
    Tester {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Documentation specialist - ensures clear explanations
    Documenter {
        id: AgentId,
        capabilities: Vec<String>,
    },
    /// Review coordinator - aggregates feedback
    Reviewer {
        id: AgentId,
        capabilities: Vec<String>,
    },
}

impl Agent {
    pub fn id(&self) -> AgentId {
        match self {
            Agent::Synthesizer { id, .. } => *id,
            Agent::Debugger { id, .. } => *id,
            Agent::Optimizer { id, .. } => *id,
            Agent::SecurityExpert { id, .. } => *id,
            Agent::Tester { id, .. } => *id,
            Agent::Documenter { id, .. } => *id,
            Agent::Reviewer { id, .. } => *id,
        }
    }

    pub fn capabilities(&self) -> &[String] {
        match self {
            Agent::Synthesizer { capabilities, .. } => capabilities,
            Agent::Debugger { capabilities, .. } => capabilities,
            Agent::Optimizer { capabilities, .. } => capabilities,
            Agent::SecurityExpert { capabilities, .. } => capabilities,
            Agent::Tester { capabilities, .. } => capabilities,
            Agent::Documenter { capabilities, .. } => capabilities,
            Agent::Reviewer { capabilities, .. } => capabilities,
        }
    }

    /// Create all standard agents
    pub fn create_all() -> Vec<Self> {
        vec![
            Agent::Synthesizer {
                id: AgentId::Synthesizer,
                capabilities: vec![
                    "synthesis".to_string(),
                    "code_generation".to_string(),
                    "pattern_matching".to_string(),
                ],
            },
            Agent::Debugger {
                id: AgentId::Debugger,
                capabilities: vec![
                    "error_detection".to_string(),
                    "logic_analysis".to_string(),
                    "trace_analysis".to_string(),
                ],
            },
            Agent::Optimizer {
                id: AgentId::Optimizer,
                capabilities: vec![
                    "performance_optimization".to_string(),
                    "complexity_reduction".to_string(),
                    "resource_efficiency".to_string(),
                ],
            },
            Agent::SecurityExpert {
                id: AgentId::SecurityExpert,
                capabilities: vec![
                    "safety_validation".to_string(),
                    "correctness_proofs".to_string(),
                    "threat_detection".to_string(),
                ],
            },
            Agent::Tester {
                id: AgentId::Tester,
                capabilities: vec![
                    "test_generation".to_string(),
                    "edge_case_detection".to_string(),
                    "coverage_analysis".to_string(),
                ],
            },
            Agent::Documenter {
                id: AgentId::Documenter,
                capabilities: vec![
                    "documentation".to_string(),
                    "explanation_generation".to_string(),
                    "readability_analysis".to_string(),
                ],
            },
            Agent::Reviewer {
                id: AgentId::Reviewer,
                capabilities: vec![
                    "code_review".to_string(),
                    "feedback_aggregation".to_string(),
                    "consensus_building".to_string(),
                ],
            },
        ]
    }
}

/// Solution proposal from an agent
#[derive(Debug, Clone)]
pub struct SolutionProposal {
    pub agent_id: AgentId,
    pub code: String,
    pub confidence: f64,
    pub metadata: serde_json::Value,
    pub reviews: HashMap<AgentId, ReviewFeedback>,
}

/// Review feedback from an agent
#[derive(Debug, Clone)]
pub struct ReviewFeedback {
    pub approved: bool,
    pub feedback: String,
    pub score: f64,
    pub issues_found: usize,
}

/// Vote record for solution selection
#[derive(Debug, Clone)]
pub struct SolutionVote {
    pub agent_id: AgentId,
    pub preferred_solution: usize,
    pub rationale: String,
    pub weight: f64,
}

/// Collaborative solving result
#[derive(Debug, Clone)]
pub struct CollaborativeResult {
    pub final_code: String,
    pub consensus_score: f64,
    pub participating_agents: Vec<AgentId>,
    pub proposals: Vec<SolutionProposal>,
    pub votes: Vec<SolutionVote>,
    pub total_rounds: usize,
    pub elapsed_ms: u64,
}

/// Multi-agent orchestrator for collaborative problem solving
pub struct Orchestrator {
    agents: Vec<Agent>,
    communication_channel: Arc<RwLock<CommunicationChannel>>,
    config: OrchestratorConfig,
    /// Optional policy-gated tool runtime (filesystem, git, shell, http, db).
    tools: Option<SecureToolRuntime>,
}

/// Orchestration configuration
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of collaboration rounds
    pub max_rounds: usize,
    /// Minimum consensus threshold (0.0 - 1.0)
    pub consensus_threshold: f64,
    /// Timeout for individual agent tasks (seconds)
    pub agent_timeout_s: u64,
    /// Whether to enable parallel agent execution
    pub enable_parallel: bool,
    /// Number of parallel workers
    pub parallel_workers: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            consensus_threshold: 0.7,
            agent_timeout_s: 30,
            enable_parallel: true,
            parallel_workers: 4,
        }
    }
}

/// Communication channel for agent messages
#[derive(Debug)]
pub struct CommunicationChannel {
    tx: mpsc::UnboundedSender<AgentMessage>,
    rx: Arc<RwLock<mpsc::UnboundedReceiver<AgentMessage>>>,
    message_log: Arc<RwLock<Vec<AgentMessage>>>,
}

impl CommunicationChannel {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            rx: Arc::new(RwLock::new(rx)),
            message_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn send(&self, msg: AgentMessage) -> Result<(), SolverError> {
        self.tx
            .send(msg.clone())
            .map_err(|e| SolverError::CommunicationError(e.to_string()))?;
        self.message_log.write().await.push(msg);
        Ok(())
    }

    pub async fn recv(&self) -> Option<AgentMessage> {
        let mut rx = self.rx.write().await;
        rx.recv().await
    }

    pub async fn try_recv(&self) -> Option<AgentMessage> {
        let mut rx = self.rx.write().await;
        rx.try_recv().ok()
    }

    pub async fn message_count(&self) -> usize {
        self.message_log.read().await.len()
    }
}

impl Orchestrator {
    /// Create a new orchestrator with standard agents
    pub fn new() -> Self {
        Self::with_config(OrchestratorConfig::default())
    }

    /// Create a new orchestrator with custom configuration
    pub fn with_config(config: OrchestratorConfig) -> Self {
        Self {
            agents: Agent::create_all(),
            communication_channel: Arc::new(RwLock::new(CommunicationChannel::new())),
            config,
            tools: None,
        }
    }

    /// Attach a default tool registry sandboxed to `sandbox_root`, giving the
    /// orchestrator real filesystem/git/shell/http/db capabilities. Builder-style.
    pub fn with_tools(mut self, sandbox_root: impl Into<std::path::PathBuf>) -> Self {
        self.tools = Some(
            SecureToolRuntime::for_general_agent(
                sandbox_root,
                crate::agent::repo::GuardrailPolicy::default(),
            ),
        );
        self
    }

    /// Borrow the secure tool runtime, if one is attached.
    pub fn tools(&self) -> Option<&SecureToolRuntime> {
        self.tools.as_ref()
    }

    /// Persist a synthesized solution to disk via the sandboxed `FsTool`, then
    /// return the working-tree `git status` (proving the file landed). Errors if
    /// no tools are attached. Paths are sandbox-relative; the `FsTool` rejects
    /// absolute paths and `..` traversal.
    pub fn persist_solution(&self, filename: &str, code: &str) -> Result<String, SolverError> {
        let tools = self
            .tools
            .as_ref()
            .ok_or_else(|| SolverError::ConfigurationError("no tools attached".into()))?;

        let write = ToolCall::new("write")
            .arg("path", filename)
            .arg("content", code);
        tools
            .invoke("fs", &write)
            .map_err(|e| SolverError::Other(format!("fs write failed: {e}")))?;

        // Repo status is best-effort observability: persistence has already
        // succeeded once the file is written. If the sandbox root is not a git
        // repository (or git is unavailable), return an empty status rather than
        // failing the whole persist.
        let status = tools
            .invoke("git", &ToolCall::new("status"))
            .map(|o| o.content)
            .unwrap_or_default();
        Ok(status)
    }

    /// Add a custom agent to the orchestrator
    pub fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    /// Get all registered agents
    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    /// Solve a problem collaboratively through agent chaining
    pub async fn solve_collaborative(
        &self,
        examples: Vec<(Vec<serde_json::Value>, i64)>,
    ) -> Result<CollaborativeResult, SolverError> {
        let start_time = std::time::Instant::now();
        let problem = Self::problem_from_json_examples(&examples)?;

        // Round 1: Initial synthesis
        let initial_proposals = self.run_initial_synthesis(&problem).await?;

        if initial_proposals.is_empty() {
            return Err(SolverError::NoSolutionFound(
                "No agents produced valid solutions".to_string(),
            ));
        }

        // Round 2: Peer review and refinement
        let reviewed_proposals = if self.config.max_rounds >= 2 {
            self.run_peer_review(initial_proposals, &problem).await?
        } else {
            initial_proposals
        };

        // Round 3: Voting and consensus
        let (final_solution, votes) = if self.config.max_rounds >= 3 {
            self.run_voting(&reviewed_proposals).await?
        } else {
            // Select highest confidence solution without voting
            let best = reviewed_proposals
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                .ok_or_else(|| {
                    SolverError::NoSolutionFound("No proposals to select".to_string())
                })?;
            (best.clone(), Vec::new())
        };

        let selected_index = reviewed_proposals
            .iter()
            .position(|proposal| {
                proposal.agent_id == final_solution.agent_id && proposal.code == final_solution.code
            })
            .unwrap_or(0);
        let consensus_score = self.calculate_consensus(&votes, selected_index);

        Ok(CollaborativeResult {
            final_code: final_solution.code,
            consensus_score,
            participating_agents: self.agents.iter().map(|a| a.id()).collect(),
            proposals: reviewed_proposals,
            votes,
            total_rounds: self.config.max_rounds,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Run initial synthesis round with parallel agents
    async fn run_initial_synthesis(
        &self,
        problem: &Problem,
    ) -> Result<Vec<SolutionProposal>, SolverError> {
        let channel = self.communication_channel.read().await;
        let synthesizer = self
            .agents
            .iter()
            .find(|a| matches!(a, Agent::Synthesizer { .. }))
            .ok_or_else(|| SolverError::ConfigurationError("No synthesizer agent".to_string()))?;

        // Broadcast task start
        channel
            .send(AgentMessage::TaskStart {
                examples: problem
                    .examples
                    .iter()
                    .map(|example| {
                        (
                            example
                                .inputs
                                .iter()
                                .map(|value| match value {
                                    Value::Int(value) => serde_json::json!(value),
                                    _ => serde_json::Value::Null,
                                })
                                .collect(),
                            example.expected_int(),
                        )
                    })
                    .collect(),
                timeout_s: self.config.agent_timeout_s,
            })
            .await?;

        let result = crate::solver::solve_problem_search_only(problem);
        if !result.success {
            return Err(SolverError::NoSolutionFound(result.error.unwrap_or_else(
                || "native synthesis portfolio exhausted".to_string(),
            )));
        }
        crate::runtime::verify_problem_code(problem, &result.code)
            .map_err(SolverError::VerificationFailed)?;

        Ok(vec![SolutionProposal {
            agent_id: synthesizer.id(),
            code: result.code,
            confidence: 1.0,
            metadata: serde_json::json!({
                "method": result.method,
                "examples_count": problem.examples.len(),
                "verified": true,
            }),
            reviews: HashMap::new(),
        }])
    }

    fn problem_from_json_examples(
        examples: &[(Vec<serde_json::Value>, i64)],
    ) -> Result<Problem, SolverError> {
        let arity = examples
            .first()
            .map(|(inputs, _)| inputs.len())
            .ok_or_else(|| SolverError::ConfigurationError("no examples supplied".to_string()))?;
        if examples.iter().any(|(inputs, _)| inputs.len() != arity) {
            return Err(SolverError::ConfigurationError(
                "example arity is inconsistent".to_string(),
            ));
        }

        let examples = examples
            .iter()
            .map(|(inputs, expected)| {
                let inputs = inputs
                    .iter()
                    .map(|value| {
                        value.as_i64().map(Value::Int).ok_or_else(|| {
                            SolverError::ConfigurationError(
                                "collaborative synthesis currently requires i64 inputs".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Example {
                    inputs,
                    expected: Value::Int(*expected),
                })
            })
            .collect::<Result<Vec<_>, SolverError>>()?;
        let params = (0..arity)
            .map(|index| format!("arg{index}: i64"))
            .collect::<Vec<_>>()
            .join(", ");
        let signature: &'static str =
            Box::leak(format!("fn solve({params}) -> i64").into_boxed_str());

        Ok(Problem {
            name: "collaborative_solve".to_string(),
            category: "agent",
            description: "Synthesize a scalar function from observed examples.",
            signature,
            examples,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        })
    }

    /// Run peer review round
    async fn run_peer_review(
        &self,
        proposals: Vec<SolutionProposal>,
        problem: &Problem,
    ) -> Result<Vec<SolutionProposal>, SolverError> {
        let mut reviewed_proposals = proposals;

        for proposal in &mut reviewed_proposals {
            for agent in &self.agents {
                // Skip self-review and the original synthesizer
                if agent.id() == proposal.agent_id {
                    continue;
                }

                let feedback = self.review_proposal(agent.id(), proposal, problem)?;
                proposal.reviews.insert(agent.id(), feedback);
            }
        }

        Ok(reviewed_proposals)
    }

    /// Review a proposal with executable verification and concrete static checks.
    fn review_proposal(
        &self,
        agent_id: AgentId,
        proposal: &SolutionProposal,
        problem: &Problem,
    ) -> Result<ReviewFeedback, SolverError> {
        let runtime_result = crate::runtime::verify_problem_code(problem, &proposal.code);
        let call_graph_result = crate::runtime::validate_call_graph(&proposal.code);
        let forbidden = ["todo!", "unimplemented!", "unsafe {", "std::process"]
            .iter()
            .filter(|needle| proposal.code.contains(**needle))
            .copied()
            .collect::<Vec<_>>();
        let mut issues = Vec::new();
        if let Err(error) = runtime_result {
            issues.push(format!("runtime verification failed: {error}"));
        }
        if let Err(error) = call_graph_result {
            issues.push(format!("call-graph validation failed: {error}"));
        }
        if !forbidden.is_empty() {
            issues.push(format!("forbidden constructs: {}", forbidden.join(", ")));
        }
        let approved = issues.is_empty();
        let score = if approved { 1.0 } else { 0.0 };

        Ok(ReviewFeedback {
            approved,
            feedback: if approved {
                format!("{agent_id}: runtime and static checks passed")
            } else {
                format!("{agent_id}: {}", issues.join("; "))
            },
            score,
            issues_found: issues.len(),
        })
    }

    /// Run voting round to select best solution
    async fn run_voting(
        &self,
        proposals: &[SolutionProposal],
    ) -> Result<(SolutionProposal, Vec<SolutionVote>), SolverError> {
        let mut votes = Vec::new();

        for agent in &self.agents {
            let preferred = self.select_best_proposal(agent.id(), proposals).await;
            votes.push(SolutionVote {
                agent_id: agent.id(),
                preferred_solution: preferred,
                rationale: format!("Selected based on {} analysis", agent.id()),
                weight: 1.0 / self.agents.len() as f64,
            });
        }

        // Tally votes
        let mut vote_counts = vec![0usize; proposals.len()];
        for vote in &votes {
            if vote.preferred_solution < vote_counts.len() {
                vote_counts[vote.preferred_solution] += 1;
            }
        }

        let winner_idx = vote_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .ok_or_else(|| SolverError::NoSolutionFound("No winner in voting".to_string()))?;

        Ok((proposals[winner_idx].clone(), votes))
    }

    /// Select the best proposal for a given agent
    async fn select_best_proposal(
        &self,
        agent_id: AgentId,
        proposals: &[SolutionProposal],
    ) -> usize {
        // Simple heuristic: select proposal with highest average review score
        let mut best_idx = 0;
        let mut best_score = 0.0f64;

        for (idx, proposal) in proposals.iter().enumerate() {
            let avg_score: f64 = proposal.reviews.values().map(|r| r.score).sum::<f64>()
                / proposal.reviews.len().max(1) as f64;

            if avg_score > best_score {
                best_score = avg_score;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Calculate consensus score from votes
    fn calculate_consensus(&self, votes: &[SolutionVote], selected_index: usize) -> f64 {
        if votes.is_empty() {
            return 0.0;
        }

        let votes_for_selected = votes
            .iter()
            .filter(|v| v.preferred_solution == selected_index)
            .count();

        votes_for_selected as f64 / votes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let orchestrator = Orchestrator::new();
        assert_eq!(orchestrator.agents().len(), 7);
        // Reasoning-only by default: no tools attached.
        assert!(orchestrator.tools().is_none());
    }

    #[test]
    fn test_persist_solution_without_tools_errors() {
        let orchestrator = Orchestrator::new();
        let err = orchestrator
            .persist_solution("out.rs", "fn f() {}")
            .unwrap_err();
        assert!(matches!(err, SolverError::ConfigurationError(_)));
    }

    #[test]
    fn test_orchestrator_persists_solution_via_tools() {
        // Sandbox in a unique temp dir (no tempfile crate dependency).
        let root = std::env::temp_dir().join(format!("nsynth_orch_tools_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();

        let orchestrator = Orchestrator::new().with_tools(&root);
        assert!(orchestrator.tools().is_some());

        let code = "fn add_two(a: i64, b: i64) -> i64 {\n    return (a + b);\n}\n";
        // Writes via the sandboxed FsTool, returns repo `git status` (runs in the
        // process cwd, which is inside this repo during tests).
        let status = orchestrator.persist_solution("solution.rs", code).unwrap();

        // The file actually landed in the sandbox.
        let written = std::fs::read_to_string(root.join("solution.rs")).unwrap();
        assert_eq!(written, code);
        // git status returned a string (the orchestrator can observe the repo).
        let _ = status;

        // Sandbox boundary still enforced through the orchestrator path.
        let escape = orchestrator.persist_solution("../escape.rs", code);
        assert!(escape.is_err(), "absolute/.. paths must be rejected");

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn test_communication_channel() {
        let channel = CommunicationChannel::new();
        channel
            .send(AgentMessage::TaskStart {
                examples: vec![],
                timeout_s: 10,
            })
            .await
            .unwrap();

        let msg = channel.recv().await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_collaborative_solving() {
        let orchestrator = Orchestrator::with_config(OrchestratorConfig {
            max_rounds: 2,
            consensus_threshold: 0.6,
            ..Default::default()
        });

        let examples = vec![
            (vec![json!(2), json!(3)], 5),
            (vec![json!(5), json!(7)], 12),
        ];

        let result = orchestrator.solve_collaborative(examples).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.final_code.is_empty());
        assert!(!result.final_code.contains("todo!"));
        assert!(result
            .proposals
            .iter()
            .all(|proposal| { proposal.reviews.values().all(|review| review.approved) }));
        assert!(result.consensus_score >= 0.0 && result.consensus_score <= 1.0);
        assert_eq!(result.participating_agents.len(), 7);
    }

    #[tokio::test]
    async fn collaborative_solving_rejects_non_integer_inputs() {
        let orchestrator = Orchestrator::new();
        let result = orchestrator
            .solve_collaborative(vec![(vec![json!("not-an-int")], 1)])
            .await;
        assert!(matches!(result, Err(SolverError::ConfigurationError(_))));
    }

    #[tokio::test]
    async fn test_agent_capabilities() {
        let agents = Agent::create_all();
        assert!(agents
            .iter()
            .any(|a| matches!(a, Agent::Synthesizer { .. })));
        assert!(agents
            .iter()
            .any(|a| matches!(a, Agent::SecurityExpert { .. })));

        let synthesizer = agents
            .iter()
            .find(|a| matches!(a, Agent::Synthesizer { .. }))
            .unwrap();
        assert!(synthesizer
            .capabilities()
            .contains(&"synthesis".to_string()));
    }
}
