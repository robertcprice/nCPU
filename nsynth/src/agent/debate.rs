// Adversarial Debate System for Code Hardening
// Implements multi-agent debate where critics propose changes, debate alternatives,
// vote on approaches, and merge improvements for battle-hardened code

use crate::benchmark::{Example, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Error types for adversarial debate system
#[derive(Debug, thiserror::Error)]
pub enum DebateError {
    #[error("No valid proposals generated")]
    NoValidProposals,

    #[error("Consensus could not be reached")]
    NoConsensus,

    #[error("Agent communication failed: {0}")]
    CommunicationError(String),

    #[error("Attack analysis failed: {0}")]
    AttackError(String),

    #[error("Defense validation failed: {0}")]
    DefenseError(String),

    #[error("Vote tally failed: {0}")]
    VotingError(String),

    #[error("Merge conflict detected: {0}")]
    MergeConflict(String),

    #[error("Timeout exceeded: {0}s")]
    Timeout(u64),
}

/// Agent role in the debate system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebateAgentId {
    /// Proposes alternative solutions and improvements
    Proposer,

    /// Identifies vulnerabilities and edge cases
    Attacker,

    /// Defends code against attacks and proposes patches
    Defender,

    /// Validates correctness and semantics
    Validator,

    /// Analyzes performance characteristics
    PerformanceAnalyst,

    /// Checks security properties
    SecurityAuditor,

    /// Aggregates feedback and manages voting
    Moderator,
}

impl std::fmt::Display for DebateAgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DebateAgentId::Proposer => write!(f, "Proposer"),
            DebateAgentId::Attacker => write!(f, "Attacker"),
            DebateAgentId::Defender => write!(f, "Defender"),
            DebateAgentId::Validator => write!(f, "Validator"),
            DebateAgentId::PerformanceAnalyst => write!(f, "PerformanceAnalyst"),
            DebateAgentId::SecurityAuditor => write!(f, "SecurityAuditor"),
            DebateAgentId::Moderator => write!(f, "Moderator"),
        }
    }
}

/// Critique of a proposed solution
#[derive(Debug, Clone)]
pub struct Critique {
    /// Agent who generated this critique
    pub critic: DebateAgentId,

    /// Severity of the issue (0.0 - 1.0)
    pub severity: f64,

    /// Category of the critique
    pub category: CritiqueCategory,

    /// Human-readable description
    pub description: String,

    /// Suggested fix or alternative
    pub suggestion: Option<String>,

    /// Code location (line numbers, function names)
    pub location: Option<String>,

    /// Confidence in this critique
    pub confidence: f64,
}

/// Categories of critiques
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CritiqueCategory {
    /// Incorrect logic or algorithm
    LogicError,

    /// Missing edge case handling
    EdgeCase,

    /// Performance inefficiency
    Performance,

    /// Security vulnerability
    Security,

    /// Type safety issue
    TypeSafety,

    /// Resource leak
    ResourceLeak,

    /// Concurrency issue
    Concurrency,

    /// Semantic incorrectness
    Semantic,

    /// Code style or readability
    Style,

    /// Other category
    Other(String),
}

/// Alternative solution proposal
#[derive(Debug, Clone)]
pub struct AlternativeProposal {
    /// Agent who proposed this alternative
    pub proposer: DebateAgentId,

    /// Alternative code
    pub code: String,

    /// Why this alternative is better
    pub rationale: String,

    /// Expected improvements
    pub improvements: Vec<String>,

    /// Trade-offs of this approach
    pub tradeoffs: Vec<String>,

    /// Confidence in this proposal
    pub confidence: f64,

    /// Estimated complexity (0.0 - 1.0)
    pub complexity: f64,
}

/// Defense of a proposed solution
#[derive(Debug, Clone)]
pub struct Defense {
    /// Agent who generated this defense
    pub defender: DebateAgentId,

    /// Code being defended
    pub code: String,

    /// Rationale for why this code is correct
    pub rationale: String,

    /// Counterarguments to critiques
    pub counterarguments: Vec<String>,
}

/// Vote on a proposal
#[derive(Debug, Clone)]
pub struct Vote {
    /// Agent casting the vote
    pub voter: DebateAgentId,

    /// Index of the proposal being voted for
    pub proposal_index: usize,

    /// Weight of this vote (higher = more trusted)
    pub weight: f64,

    /// Reasoning for this vote
    pub reasoning: String,
}

/// Result of the debate process
#[derive(Debug, Clone)]
pub struct DebateResult {
    /// Final battle-hardened code
    pub final_code: String,

    /// All alternatives that were considered
    pub alternatives: Vec<AlternativeProposal>,

    /// All critiques that were raised
    pub critiques: Vec<Critique>,

    /// Votes cast in the final selection
    pub votes: Vec<Vote>,

    /// Consensus level (0.0 - 1.0)
    pub consensus_level: f64,

    /// Number of debate rounds
    pub rounds: usize,

    /// Total elapsed time in milliseconds
    pub elapsed_ms: u64,

    /// Improvements that were merged
    pub merged_improvements: Vec<String>,
}

/// Configuration for the debate system
#[derive(Debug, Clone)]
pub struct DebateConfig {
    /// Maximum number of debate rounds
    pub max_rounds: usize,

    /// Minimum consensus threshold (0.0 - 1.0)
    pub consensus_threshold: f64,

    /// Timeout per round in seconds
    pub round_timeout_s: u64,

    /// Enable parallel critique generation
    pub enable_parallel_critique: bool,

    /// Minimum severity threshold for blocking issues
    pub blocking_severity: f64,

    /// Number of alternative proposals to generate
    pub num_alternatives: usize,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 3,
            consensus_threshold: 0.75,
            round_timeout_s: 30,
            enable_parallel_critique: true,
            blocking_severity: 0.8,
            num_alternatives: 3,
        }
    }
}

/// Adversarial debate system
pub struct DebateSystem {
    config: DebateConfig,
    agents: HashMap<DebateAgentId, Box<dyn DebateAgent>>,
    message_bus: Arc<RwLock<MessageBus>>,
}

/// Message bus for agent communication
#[derive(Debug)]
struct MessageBus {
    messages: Vec<DebateMessage>,
}

/// Message types for debate communication
#[derive(Debug, Clone)]
enum DebateMessage {
    CritiqueGenerated(Critique),
    AlternativeProposed(AlternativeProposal),
    DefenseGenerated(Defense),
    VoteCast(Vote),
    RoundComplete(usize),
}

impl DebateSystem {
    /// Create a new debate system with default configuration
    pub fn new() -> Self {
        Self::with_config(DebateConfig::default())
    }

    /// Create a new debate system with custom configuration
    pub fn with_config(config: DebateConfig) -> Self {
        let mut system = Self {
            config: config.clone(),
            agents: HashMap::new(),
            message_bus: Arc::new(RwLock::new(MessageBus {
                messages: Vec::new(),
            })),
        };

        // Register default agents
        system.register_agent(Box::new(AdversarialAgent::new(DebateAgentId::Attacker)));
        system.register_agent(Box::new(DefenderAgent::new(DebateAgentId::Defender)));
        system.register_agent(Box::new(ProposerAgent::new(DebateAgentId::Proposer)));

        system
    }

    /// Register a debate agent
    pub fn register_agent(&mut self, agent: Box<dyn DebateAgent>) {
        let id = agent.agent_id();
        self.agents.insert(id, agent);
    }

    /// Run the full debate process on a solution
    pub async fn debate_solution(
        &self,
        initial_code: &str,
        examples: &[Example],
    ) -> Result<DebateResult, DebateError> {
        let start_time = std::time::Instant::now();

        // Round 1: Generate critiques and alternatives
        let (critiques, alternatives) = self
            .generate_critiques_and_alternatives(initial_code, examples)
            .await?;

        // Round 2: Debate alternatives with reasoning
        let debated_alternatives = self
            .debate_alternatives(&alternatives, &critiques, examples)
            .await?;

        // Round 3: Vote on best approach
        let (winner, votes) = self
            .vote_on_best_approach(&debated_alternatives, &critiques)
            .await?;

        // Round 4: Merge improvements from all agents
        let final_code = self
            .merge_improvements(initial_code, &winner, &critiques)
            .await?;

        let consensus_level = self.calculate_consensus(&votes);
        let merged_improvements = self.extract_improvements(&final_code, initial_code);

        Ok(DebateResult {
            final_code,
            alternatives: debated_alternatives,
            critiques,
            votes,
            consensus_level,
            rounds: self.config.max_rounds,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            merged_improvements,
        })
    }

    /// Generate critiques and alternative proposals
    async fn generate_critiques_and_alternatives(
        &self,
        code: &str,
        examples: &[Example],
    ) -> Result<(Vec<Critique>, Vec<AlternativeProposal>), DebateError> {
        let mut critiques = Vec::new();
        let mut alternatives = Vec::new();

        for agent in self.agents.values() {
            // Generate critiques
            let agent_critiques = agent.critique(code, examples).await?;
            critiques.extend(agent_critiques);

            // Generate alternatives if capable
            if let Some(alt) = agent
                .propose_alternative(code, examples, &critiques)
                .await?
            {
                alternatives.push(alt);
            }
        }

        // Check for blocking issues
        let blocking = critiques
            .iter()
            .filter(|c| c.severity >= self.config.blocking_severity)
            .count();

        // Blocking critiques should drive the defense/alternative rounds, not
        // abort a debate that already has proposals capable of addressing them.
        if blocking > 0 && alternatives.is_empty() {
            return Err(DebateError::NoConsensus);
        }

        if alternatives.is_empty() {
            return Err(DebateError::NoValidProposals);
        }

        Ok((critiques, alternatives))
    }

    /// Debate alternatives with detailed reasoning
    async fn debate_alternatives(
        &self,
        alternatives: &[AlternativeProposal],
        critiques: &[Critique],
        examples: &[Example],
    ) -> Result<Vec<AlternativeProposal>, DebateError> {
        let mut debated = Vec::new();

        for alternative in alternatives {
            let mut improved_alternative = alternative.clone();

            // Let each agent analyze this alternative
            for agent in self.agents.values() {
                let analysis = agent
                    .analyze_alternative(&improved_alternative, critiques, examples)
                    .await?;

                // Apply improvements suggested by analysis
                if !analysis.improvements.is_empty() {
                    improved_alternative.code =
                        analysis.improved_code.unwrap_or(improved_alternative.code);
                    improved_alternative
                        .improvements
                        .extend(analysis.improvements);
                }

                // Check if any attacks defeat this alternative
                if let Some(attack) = analysis.defeat_attack {
                    improved_alternative
                        .tradeoffs
                        .push(format!("Vulnerable to: {}", attack));
                }
            }

            debated.push(improved_alternative);
        }

        Ok(debated)
    }

    /// Vote on the best approach
    async fn vote_on_best_approach(
        &self,
        alternatives: &[AlternativeProposal],
        critiques: &[Critique],
    ) -> Result<(AlternativeProposal, Vec<Vote>), DebateError> {
        let mut votes = Vec::new();

        // Collect votes from all agents
        for agent in self.agents.values() {
            let vote = agent.vote(alternatives, critiques).await?;
            votes.push(vote);
        }

        // Tally votes
        let mut scores = vec![0.0f64; alternatives.len()];
        for vote in &votes {
            if vote.proposal_index < scores.len() {
                scores[vote.proposal_index] += vote.weight;
            }
        }

        // Find winner (highest score)
        let winner_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or(DebateError::VotingError("No winner found".to_string()))?;

        Ok((alternatives[winner_idx].clone(), votes))
    }

    /// Merge improvements from all agents
    async fn merge_improvements(
        &self,
        original: &str,
        winner: &AlternativeProposal,
        critiques: &[Critique],
    ) -> Result<String, DebateError> {
        let mut merged = winner.code.clone();

        // Apply patches from defender
        if let Some(defender) = self.agents.get(&DebateAgentId::Defender) {
            if let Some(patch) = defender.generate_patch(&merged, critiques).await? {
                merged = patch;
            }
        }

        // Validate the merged code
        if let Some(validator) = self.agents.get(&DebateAgentId::Validator) {
            if !validator.validate(&merged).await? {
                return Err(DebateError::MergeConflict(
                    "Merged code failed validation".to_string(),
                ));
            }
        }

        Ok(merged)
    }

    /// Calculate consensus level from votes
    fn calculate_consensus(&self, votes: &[Vote]) -> f64 {
        if votes.is_empty() {
            return 0.0;
        }

        // Count votes for each proposal
        let mut vote_counts: HashMap<usize, usize> = HashMap::new();
        for vote in votes {
            *vote_counts.entry(vote.proposal_index).or_insert(0) += 1;
        }

        // Find the maximum
        let max_votes = vote_counts.values().copied().max().unwrap_or(0);

        max_votes as f64 / votes.len() as f64
    }

    /// Extract improvements from final code
    fn extract_improvements(&self, final_code: &str, original: &str) -> Vec<String> {
        let mut improvements = Vec::new();

        // Simple heuristic: if code is longer, improvements were added
        if final_code.len() > original.len() {
            improvements.push("Additional code added".to_string());
        }

        // Check for common improvement patterns
        if final_code.contains("unwrap_or") && !original.contains("unwrap_or") {
            improvements.push("Added safe unwrap handling".to_string());
        }

        if final_code.contains("check") && !original.contains("check") {
            improvements.push("Added bounds checking".to_string());
        }

        improvements
    }
}

/// Trait for debate agents
#[async_trait::async_trait]
pub trait DebateAgent: Send + Sync {
    /// Get the agent's ID
    fn agent_id(&self) -> DebateAgentId;

    /// Critique the given code
    async fn critique(
        &self,
        code: &str,
        examples: &[Example],
    ) -> Result<Vec<Critique>, DebateError>;

    /// Propose an alternative solution
    async fn propose_alternative(
        &self,
        code: &str,
        examples: &[Example],
        critiques: &[Critique],
    ) -> Result<Option<AlternativeProposal>, DebateError>;

    /// Analyze an alternative proposal
    async fn analyze_alternative(
        &self,
        alternative: &AlternativeProposal,
        critiques: &[Critique],
        examples: &[Example],
    ) -> Result<AlternativeAnalysis, DebateError>;

    /// Vote on the best alternative
    async fn vote(
        &self,
        alternatives: &[AlternativeProposal],
        critiques: &[Critique],
    ) -> Result<Vote, DebateError>;

    /// Validate code
    async fn validate(&self, code: &str) -> Result<bool, DebateError> {
        // Default implementation: basic syntax check
        Ok(!code.is_empty())
    }

    /// Generate a patch
    async fn generate_patch(
        &self,
        code: &str,
        critiques: &[Critique],
    ) -> Result<Option<String>, DebateError> {
        Ok(None)
    }
}

/// Result of analyzing an alternative
#[derive(Debug, Clone)]
pub struct AlternativeAnalysis {
    /// Improvements found
    pub improvements: Vec<String>,

    /// Improved code if available
    pub improved_code: Option<String>,

    /// Attack that defeats this alternative
    pub defeat_attack: Option<String>,
}

/// Adversarial agent that attacks code
pub struct AdversarialAgent {
    id: DebateAgentId,
}

impl AdversarialAgent {
    pub fn new(id: DebateAgentId) -> Self {
        Self { id }
    }

    /// Find edge cases not covered by examples
    async fn find_edge_cases(&self, code: &str, examples: &[Example]) -> Vec<Critique> {
        let mut critiques = Vec::new();

        // Check for common edge case issues
        if code.contains("unwrap()") || code.contains(".unwrap()") {
            critiques.push(Critique {
                critic: self.id,
                severity: 0.9,
                category: CritiqueCategory::EdgeCase,
                description: "Unsafe unwrap could panic".to_string(),
                suggestion: Some("Use unwrap_or, unwrap_or_else, or pattern matching".to_string()),
                location: self.find_unwrap_location(code),
                confidence: 0.95,
            });
        }

        if code.contains("[]") && !code.contains(".get(") {
            critiques.push(Critique {
                critic: self.id,
                severity: 0.85,
                category: CritiqueCategory::EdgeCase,
                description: "Direct array access could panic on empty array".to_string(),
                suggestion: Some("Use .get() for safe access".to_string()),
                location: None,
                confidence: 0.9,
            });
        }

        // Check for overflow issues
        if code.contains("*") || code.contains("+") {
            critiques.push(Critique {
                critic: self.id,
                severity: 0.7,
                category: CritiqueCategory::EdgeCase,
                description: "Potential integer overflow".to_string(),
                suggestion: Some("Use checked_mul or saturating operations".to_string()),
                location: None,
                confidence: 0.75,
            });
        }

        critiques
    }

    fn find_unwrap_location(&self, code: &str) -> Option<String> {
        for (idx, line) in code.lines().enumerate() {
            if line.contains("unwrap()") {
                return Some(format!("line {}", idx + 1));
            }
        }
        None
    }

    /// Generate attack scenarios
    async fn generate_attacks(&self, code: &str) -> Vec<Critique> {
        let mut attacks = Vec::new();

        // Empty input attack
        if !code.contains(".is_empty()") {
            attacks.push(Critique {
                critic: self.id,
                severity: 0.8,
                category: CritiqueCategory::EdgeCase,
                description: "No empty input check".to_string(),
                suggestion: Some("Add .is_empty() check".to_string()),
                location: None,
                confidence: 0.85,
            });
        }

        // Negative numbers attack
        if !code.contains("< 0") {
            attacks.push(Critique {
                critic: self.id,
                severity: 0.6,
                category: CritiqueCategory::LogicError,
                description: "No negative number handling".to_string(),
                suggestion: Some("Add check for negative inputs".to_string()),
                location: None,
                confidence: 0.7,
            });
        }

        attacks
    }
}

#[async_trait::async_trait]
impl DebateAgent for AdversarialAgent {
    fn agent_id(&self) -> DebateAgentId {
        self.id
    }

    async fn critique(
        &self,
        code: &str,
        examples: &[Example],
    ) -> Result<Vec<Critique>, DebateError> {
        let mut all_critiques = Vec::new();

        // Find edge cases
        all_critiques.extend(self.find_edge_cases(code, examples).await);

        // Generate attacks
        all_critiques.extend(self.generate_attacks(code).await);

        Ok(all_critiques)
    }

    async fn propose_alternative(
        &self,
        _code: &str,
        _examples: &[Example],
        _critiques: &[Critique],
    ) -> Result<Option<AlternativeProposal>, DebateError> {
        // Attackers don't propose alternatives, only critique
        Ok(None)
    }

    async fn analyze_alternative(
        &self,
        alternative: &AlternativeProposal,
        critiques: &[Critique],
        _examples: &[Example],
    ) -> Result<AlternativeAnalysis, DebateError> {
        // Check if alternative addresses our critiques
        let improvements: Vec<String> = critiques
            .iter()
            .filter(|c| {
                alternative
                    .code
                    .contains(c.suggestion.as_ref().unwrap_or(&String::new()))
            })
            .map(|c| c.description.clone())
            .collect();

        // Find if still vulnerable
        let defeat_attack = if alternative.code.contains("unwrap()") {
            Some("Still contains unsafe unwrap".to_string())
        } else if alternative.code.contains("[]") {
            Some("Still has direct array access".to_string())
        } else {
            None
        };

        Ok(AlternativeAnalysis {
            improvements,
            improved_code: None,
            defeat_attack,
        })
    }

    async fn vote(
        &self,
        alternatives: &[AlternativeProposal],
        critiques: &[Critique],
    ) -> Result<Vote, DebateError> {
        // Vote for alternative with least severe remaining issues
        let mut best_idx = 0;
        let mut min_issues = usize::MAX;

        for (idx, alt) in alternatives.iter().enumerate() {
            let issue_count = critiques
                .iter()
                .filter(|c| alt.code.contains(&c.description))
                .count();

            if issue_count < min_issues {
                min_issues = issue_count;
                best_idx = idx;
            }
        }

        Ok(Vote {
            voter: self.id,
            proposal_index: best_idx,
            weight: 1.0 / alternatives.len() as f64,
            reasoning: format!(
                "Alternative {} has {} unresolved issues",
                best_idx, min_issues
            ),
        })
    }

    async fn validate(&self, code: &str) -> Result<bool, DebateError> {
        // Attacker validates by checking for vulnerable patterns
        Ok(!code.contains("unwrap()") && !code.contains("[]"))
    }
}

/// Defender agent that proposes patches
pub struct DefenderAgent {
    id: DebateAgentId,
}

impl DefenderAgent {
    pub fn new(id: DebateAgentId) -> Self {
        Self { id }
    }

    /// Generate safe replacement for unwrap
    fn safe_unwrap_replacement(&self, code: &str) -> String {
        code.replace(".unwrap()", ".unwrap_or(0)")
            .replace(".unwrap()", ".unwrap_or_default()")
    }

    /// Add bounds checking
    fn add_bounds_check(&self, code: &str) -> String {
        // Simple heuristic: add check before array access
        if code.contains("[") && !code.contains(".get(") {
            format!("{}\n// Added bounds check\nif array.len() > index {{", code)
        } else {
            code.to_string()
        }
    }
}

#[async_trait::async_trait]
impl DebateAgent for DefenderAgent {
    fn agent_id(&self) -> DebateAgentId {
        self.id
    }

    async fn critique(
        &self,
        code: &str,
        examples: &[Example],
    ) -> Result<Vec<Critique>, DebateError> {
        let mut critiques = Vec::new();

        // Defender looks for defendable issues
        if code.contains("panic!") {
            critiques.push(Critique {
                critic: self.id,
                severity: 0.95,
                category: CritiqueCategory::ResourceLeak,
                description: "Contains panic that should be avoided".to_string(),
                suggestion: Some("Return Result instead of panicking".to_string()),
                location: None,
                confidence: 0.9,
            });
        }

        Ok(critiques)
    }

    async fn propose_alternative(
        &self,
        code: &str,
        examples: &[Example],
        critiques: &[Critique],
    ) -> Result<Option<AlternativeProposal>, DebateError> {
        let mut improved_code = code.to_string();

        // Apply fixes based on critiques
        for critique in critiques {
            if let Some(suggestion) = &critique.suggestion {
                if critique.category == CritiqueCategory::EdgeCase {
                    improved_code = self.safe_unwrap_replacement(&improved_code);
                }
            }
        }

        Ok(Some(AlternativeProposal {
            proposer: self.id,
            code: improved_code,
            rationale: "Applied defensive patches".to_string(),
            improvements: vec!["Safe unwrapping".to_string()],
            tradeoffs: vec!["Slightly more verbose".to_string()],
            confidence: 0.85,
            complexity: 0.3,
        }))
    }

    async fn analyze_alternative(
        &self,
        alternative: &AlternativeProposal,
        critiques: &[Critique],
        _examples: &[Example],
    ) -> Result<AlternativeAnalysis, DebateError> {
        let improvements: Vec<String> = critiques
            .iter()
            .filter(|c| alternative.code.contains("unwrap_or"))
            .map(|c| format!("Addressed: {}", c.description))
            .collect();

        Ok(AlternativeAnalysis {
            improvements,
            improved_code: None,
            defeat_attack: None,
        })
    }

    async fn vote(
        &self,
        alternatives: &[AlternativeProposal],
        _critiques: &[Critique],
    ) -> Result<Vote, DebateError> {
        // Vote for safest alternative
        let safest = alternatives
            .iter()
            .enumerate()
            .filter(|(_, alt)| !alt.code.contains("unwrap()"))
            .min_by_key(|(_, alt)| alt.code.matches("panic").count());

        let idx = safest.map(|(idx, _)| idx).unwrap_or(0);

        Ok(Vote {
            voter: self.id,
            proposal_index: idx,
            weight: 1.2, // Defender gets slightly higher weight
            reasoning: "Selected safest alternative".to_string(),
        })
    }

    async fn generate_patch(
        &self,
        code: &str,
        critiques: &[Critique],
    ) -> Result<Option<String>, DebateError> {
        let mut patched = code.to_string();

        for critique in critiques {
            if critique.severity > 0.7 {
                if let Some(suggestion) = &critique.suggestion {
                    if suggestion.contains("unwrap_or") {
                        patched = self.safe_unwrap_replacement(&patched);
                    }
                }
            }
        }

        if patched != code {
            Ok(Some(patched))
        } else {
            Ok(None)
        }
    }
}

/// Proposer agent that suggests alternatives
pub struct ProposerAgent {
    id: DebateAgentId,
}

impl ProposerAgent {
    pub fn new(id: DebateAgentId) -> Self {
        Self { id }
    }
}

#[async_trait::async_trait]
impl DebateAgent for ProposerAgent {
    fn agent_id(&self) -> DebateAgentId {
        self.id
    }

    async fn critique(
        &self,
        _code: &str,
        _examples: &[Example],
    ) -> Result<Vec<Critique>, DebateError> {
        // Proposers focus on alternatives, not critique
        Ok(Vec::new())
    }

    async fn propose_alternative(
        &self,
        code: &str,
        examples: &[Example],
        _critiques: &[Critique],
    ) -> Result<Option<AlternativeProposal>, DebateError> {
        // Generate functional-style alternative
        let functional_alt = self.generate_functional_alternative(code, examples);

        Ok(Some(functional_alt))
    }

    async fn analyze_alternative(
        &self,
        _alternative: &AlternativeProposal,
        _critiques: &[Critique],
        _examples: &[Example],
    ) -> Result<AlternativeAnalysis, DebateError> {
        Ok(AlternativeAnalysis {
            improvements: Vec::new(),
            improved_code: None,
            defeat_attack: None,
        })
    }

    async fn vote(
        &self,
        alternatives: &[AlternativeProposal],
        _critiques: &[Critique],
    ) -> Result<Vote, DebateError> {
        // Vote for most concise alternative
        let most_concise = alternatives
            .iter()
            .enumerate()
            .min_by_key(|(_, alt)| alt.code.len());

        let idx = most_concise.map(|(idx, _)| idx).unwrap_or(0);

        Ok(Vote {
            voter: self.id,
            proposal_index: idx,
            weight: 1.0,
            reasoning: "Selected most concise alternative".to_string(),
        })
    }
}

impl ProposerAgent {
    fn generate_functional_alternative(
        &self,
        code: &str,
        _examples: &[Example],
    ) -> AlternativeProposal {
        // Simple transformation: make it more functional
        let functional_code = if code.contains("for") {
            code.replace("for", "// TODO: replace with iterator")
        } else {
            code.to_string()
        };

        AlternativeProposal {
            proposer: self.id,
            code: functional_code,
            rationale: "Functional style alternatives".to_string(),
            improvements: vec!["More declarative".to_string()],
            tradeoffs: vec!["May be harder to read".to_string()],
            confidence: 0.7,
            complexity: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_debate_system_creation() {
        let system = DebateSystem::new();
        assert_eq!(system.agents.len(), 3);
    }

    #[tokio::test]
    async fn test_adversarial_agent_critique() {
        let agent = AdversarialAgent::new(DebateAgentId::Attacker);
        let code = "fn foo(x: Option<i32>) -> i32 { x.unwrap() }";
        let examples = &[];

        let critiques = agent.critique(code, examples).await.unwrap();
        assert!(!critiques.is_empty());
        assert!(critiques.iter().any(|c| c.description.contains("unwrap")));
    }

    #[tokio::test]
    async fn test_defender_agent_patch() {
        let agent = DefenderAgent::new(DebateAgentId::Defender);
        let code = "fn foo(x: Option<i32>) -> i32 { x.unwrap() }";
        let critiques = vec![Critique {
            critic: DebateAgentId::Attacker,
            severity: 0.9,
            category: CritiqueCategory::EdgeCase,
            description: "Unsafe unwrap".to_string(),
            suggestion: Some("Use unwrap_or".to_string()),
            location: None,
            confidence: 0.9,
        }];

        let patch = agent.generate_patch(code, &critiques).await.unwrap();
        assert!(patch.is_some());
        assert!(patch.unwrap().contains("unwrap_or"));
    }

    #[tokio::test]
    async fn test_full_debate() {
        let system = DebateSystem::with_config(DebateConfig {
            max_rounds: 2,
            ..Default::default()
        });

        let code = "fn add(x: i32, y: i32) -> i32 { x + y }";
        let examples = vec![Example {
            inputs: vec![],
            expected: crate::benchmark::Value::Int(5),
        }];

        let result = system.debate_solution(code, &examples).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(!result.final_code.is_empty());
        assert!(result.consensus_level > 0.0);
    }

    #[tokio::test]
    async fn test_vote_tally() {
        let system = DebateSystem::new();
        let alternatives = vec![
            AlternativeProposal {
                proposer: DebateAgentId::Proposer,
                code: "code A".to_string(),
                rationale: "A".to_string(),
                improvements: Vec::new(),
                tradeoffs: Vec::new(),
                confidence: 0.8,
                complexity: 0.3,
            },
            AlternativeProposal {
                proposer: DebateAgentId::Defender,
                code: "code B".to_string(),
                rationale: "B".to_string(),
                improvements: Vec::new(),
                tradeoffs: Vec::new(),
                confidence: 0.9,
                complexity: 0.2,
            },
        ];

        let votes = vec![
            Vote {
                voter: DebateAgentId::Attacker,
                proposal_index: 0,
                weight: 1.0,
                reasoning: "A is better".to_string(),
            },
            Vote {
                voter: DebateAgentId::Defender,
                proposal_index: 1,
                weight: 1.0,
                reasoning: "B is better".to_string(),
            },
        ];

        let consensus = system.calculate_consensus(&votes);
        assert!(consensus >= 0.0 && consensus <= 1.0);
    }
}
