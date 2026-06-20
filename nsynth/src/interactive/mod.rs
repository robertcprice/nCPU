//! Interactive synthesis and human-in-the-loop features
//!
//! This module provides scaffolding for:
//! - Clarifying ambiguous synthesis requirements
//! - Checkpoint/resume for long-running synthesis
//! - Partial execution with intermediate value inspection
//! - Human-in-the-loop refinement

pub mod clarify;

// Re-exports from clarify module
pub use clarify::{
    ask_clarification, execute_interactive, Answer, CandidateAttempt, Checkpoint,
    ClarificationState, InteractiveSession, IntermediateValue, PartialResults, PartialSolution,
    Question, QuestionOption, SynthesisPhase,
};

// Re-export legacy interactive functions for compatibility
pub use crate::interactive_legacy::{
    solve_interactive_problem as solve_interactive_problem_legacy,
    solve_interactive_problem_differentiable_only,
};

use crate::benchmark::{Example, Problem, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::solver::SolveResult;
use serde::{Deserialize, Serialize};

/// Core interactive types (from legacy interactive.rs)
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveTrace {
    pub input_stream: Vec<i64>,
    pub expected_output: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveProblem {
    pub name: String,
    #[serde(skip)]
    pub base_problem: Problem, // Skip serialization to avoid serde issues
    pub traces: Vec<InteractiveTrace>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractiveSolveResult {
    pub success: bool,
    pub code: String,
    pub method: String,
    pub error: Option<String>,
    #[serde(skip)]
    pub metadata: DifferentiableMetadata, // Skip to avoid serde issues
}

// Placeholder functions for compatibility
pub fn lift_problem_to_interactive(_problem: &Problem) -> Result<InteractiveProblem, String> {
    Err("Not implemented".to_string())
}

pub fn solve_interactive_problem(_problem: &Problem) -> InteractiveSolveResult {
    InteractiveSolveResult {
        success: false,
        code: String::new(),
        method: "interactive".to_string(),
        error: Some("Not implemented".to_string()),
        metadata: DifferentiableMetadata::default(),
    }
}

pub fn verify_interactive_program(
    _problem: &InteractiveProblem,
    _code: &str,
) -> Result<(), String> {
    Err("Not implemented".to_string())
}

/// Configuration for interactive synthesis sessions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractiveConfig {
    /// Directory for checkpoint storage
    pub checkpoint_dir: String,
    /// Whether to auto-ask clarifying questions
    pub auto_clarify: bool,
    /// Whether to show intermediate values during synthesis
    pub show_intermediates: bool,
    /// Maximum number of clarification questions to ask
    pub max_questions: usize,
}

impl Default for InteractiveConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: ".nsynth/checkpoints".to_string(),
            auto_clarify: true,
            show_intermediates: true,
            max_questions: 10,
        }
    }
}

/// Result of an interactive synthesis session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InteractiveResult {
    /// Whether synthesis was successful
    pub success: bool,
    /// The synthesized code (if successful)
    pub code: Option<String>,
    /// Method used for synthesis
    pub method: String,
    /// Any error that occurred
    pub error: Option<String>,
    /// Partial results from checkpoints
    pub partial_results: Option<PartialResults>,
    /// Answers to clarification questions
    pub clarifications: Vec<Answer>,
}

impl InteractiveResult {
    /// Create a successful result
    pub fn success(code: String, method: String) -> Self {
        Self {
            success: true,
            code: Some(code),
            method,
            error: None,
            partial_results: None,
            clarifications: Vec::new(),
        }
    }

    /// Create a failed result
    pub fn error(error: String) -> Self {
        Self {
            success: false,
            code: None,
            method: String::new(),
            error: Some(error),
            partial_results: None,
            clarifications: Vec::new(),
        }
    }

    /// Attach partial results to this result
    pub fn with_partial(mut self, partial: PartialResults) -> Self {
        self.partial_results = Some(partial);
        self
    }

    /// Attach clarification answers to this result
    pub fn with_clarifications(mut self, answers: Vec<Answer>) -> Self {
        self.clarifications = answers;
        self
    }
}

/// Convert a SolveResult to an InteractiveResult
impl From<SolveResult> for InteractiveResult {
    fn from(result: SolveResult) -> Self {
        if result.success {
            Self::success(result.code, result.method)
        } else {
            Self::error(result.error.unwrap_or_else(|| "Unknown error".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InteractiveConfig::default();
        assert_eq!(config.checkpoint_dir, ".nsynth/checkpoints");
        assert!(config.auto_clarify);
        assert!(config.show_intermediates);
        assert_eq!(config.max_questions, 10);
    }

    #[test]
    fn test_interactive_result() {
        let success =
            InteractiveResult::success("fn test() {}".to_string(), "test_method".to_string());
        assert!(success.success);
        assert!(success.code.is_some());
        assert!(success.error.is_none());

        let failure = InteractiveResult::error("test error".to_string());
        assert!(!failure.success);
        assert!(failure.code.is_none());
        assert!(failure.error.is_some());
    }
}
