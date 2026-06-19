//! Interactive clarification and human-in-the-loop synthesis
//!
//! Provides scaffolding for:
//! - Asking clarifying questions for ambiguous requirements
//! - Checkpoint/resume functionality for long-running syntheses
//! - Partial execution with intermediate value inspection
//! - Human-in-the-loop refinement of synthesis results

use crate::benchmark::{Problem, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// A clarifying question presented to the user during synthesis
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Question {
    /// Unique identifier for this question
    pub id: String,
    /// The question text presented to the user
    pub text: String,
    /// Available options for the answer
    pub options: Vec<QuestionOption>,
    /// Explanation of why this question is being asked
    pub why: String,
    /// Default answer (if any)
    pub default: Option<usize>,
}

/// A single option for a question
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionOption {
    /// Display text for this option
    pub text: String,
    /// Value that will be returned if this option is selected
    pub value: String,
    /// Additional explanation for this option
    pub explanation: Option<String>,
}

impl Question {
    /// Create a new question with the given text
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            options: Vec::new(),
            why: String::new(),
            default: None,
        }
    }

    /// Add an option to this question
    pub fn with_option(mut self, text: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push(QuestionOption {
            text: text.into(),
            value: value.into(),
            explanation: None,
        });
        self
    }

    /// Add an option with explanation
    pub fn with_option_explained(
        mut self,
        text: impl Into<String>,
        value: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        self.options.push(QuestionOption {
            text: text.into(),
            value: value.into(),
            explanation: Some(explanation.into()),
        });
        self
    }

    /// Set the explanation for why this question is asked
    pub fn with_why(mut self, why: impl Into<String>) -> Self {
        self.why = why.into();
        self
    }

    /// Set the default option index
    pub fn with_default(mut self, default: usize) -> Self {
        self.default = Some(default);
        self
    }
}

/// Answer to a clarifying question
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// ID of the question being answered
    pub question_id: String,
    /// Index of the selected option
    pub selected_index: usize,
    /// The selected value
    pub value: String,
}

/// State of clarification questions
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClarificationState {
    /// Still waiting for questions to be answered
    Pending(Vec<Question>),
    /// All questions answered
    Answered(Vec<Answer>),
    /// No clarification needed
    NotNeeded,
}

impl Default for ClarificationState {
    fn default() -> Self {
        Self::NotNeeded
    }
}

impl ClarificationState {
    /// Get unanswered questions
    pub fn unanswered(&self) -> &[Question] {
        match self {
            ClarificationState::Pending(questions) => questions,
            _ => &[],
        }
    }

    /// Get answered questions
    pub fn answered(&self) -> &[Answer] {
        match self {
            ClarificationState::Answered(answers) => answers,
            _ => &[],
        }
    }

    /// Check if all questions are answered
    pub fn is_complete(&self) -> bool {
        matches!(self, ClarificationState::Answered(_) | ClarificationState::NotNeeded)
    }
}

/// Checkpoint data for resumable synthesis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint ID
    pub id: String,
    /// Timestamp when checkpoint was created
    pub timestamp: u64,
    /// Problem being synthesized (skipped during serialization due to Problem not impl)
    #[serde(skip)]
    pub problem: Option<Problem>,
    /// Current phase of synthesis
    pub phase: SynthesisPhase,
    /// Partial results so far
    pub partial_results: PartialResults,
    /// Clarification state
    pub clarification: ClarificationState,
    /// Metadata about the synthesis
    pub metadata: HashMap<String, String>,
}

/// Phase of synthesis process
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisPhase {
    /// Analyzing the problem
    Analyzing,
    /// Generating candidates
    Generating,
    /// Verifying candidates
    Verifying,
    /// Optimizing solution
    Optimizing,
    /// Complete
    Complete,
}

/// Partial results from ongoing synthesis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartialResults {
    /// Intermediate values computed so far
    pub intermediates: Vec<IntermediateValue>,
    /// Candidate solutions attempted
    pub candidates: Vec<CandidateAttempt>,
    /// Best solution found so far
    pub best: Option<PartialSolution>,
}

/// An intermediate value computed during synthesis
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntermediateValue {
    /// Name/identifier for this value
    pub name: String,
    /// The computed value
    pub value: Value,
    /// When this was computed (phase index)
    pub step: usize,
    /// Human-readable description
    pub description: String,
}

/// A candidate solution attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateAttempt {
    /// Code that was attempted
    pub code: String,
    /// Whether it passed verification
    pub passed: bool,
    /// Error message if it failed
    pub error: Option<String>,
    /// How many examples it passed
    pub passed_examples: usize,
    /// Total examples tested
    pub total_examples: usize,
}

/// A partial solution (not yet verified on all examples)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartialSolution {
    /// The code so far
    pub code: String,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Which examples have been verified
    pub verified_examples: Vec<usize>,
    /// Estimated completion percentage
    pub completion_pct: usize,
}

impl IntermediateValue {
    /// Create a new intermediate value
    pub fn new(
        name: impl Into<String>,
        value: Value,
        step: usize,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            step,
            description: description.into(),
        }
    }
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(id: impl Into<String>, problem: Problem, phase: SynthesisPhase) -> Self {
        Self {
            id: id.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            problem: Some(problem),
            phase,
            partial_results: PartialResults {
                intermediates: Vec::new(),
                candidates: Vec::new(),
                best: None,
            },
            clarification: ClarificationState::NotNeeded,
            metadata: HashMap::new(),
        }
    }

    /// Save checkpoint to disk
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize checkpoint: {}", e))?;
        fs::write(path, json)
            .map_err(|e| format!("Failed to write checkpoint to {}: {}", path.display(), e))
    }

    /// Load checkpoint from disk
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read checkpoint from {}: {}", path.display(), e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize checkpoint: {}", e))
    }

    /// Update the synthesis phase
    pub fn with_phase(mut self, phase: SynthesisPhase) -> Self {
        self.phase = phase;
        self
    }

    /// Add an intermediate value
    pub fn with_intermediate(mut self, value: IntermediateValue) -> Self {
        self.partial_results.intermediates.push(value);
        self
    }

    /// Record a candidate attempt
    pub fn with_candidate(mut self, attempt: CandidateAttempt) -> Self {
        self.partial_results.candidates.push(attempt);
        self
    }

    /// Update the best partial solution
    pub fn with_best(mut self, best: PartialSolution) -> Self {
        self.partial_results.best = Some(best);
        self
    }
}

/// Interactive synthesis session
pub struct InteractiveSession {
    /// Session ID
    id: String,
    /// Current checkpoint
    checkpoint: Checkpoint,
    /// Checkpoint file path
    checkpoint_path: PathBuf,
}

impl InteractiveSession {
    /// Create a new interactive session
    pub fn new(problem: Problem, checkpoint_dir: &Path) -> Result<Self, String> {
        let id = format!("session_{}", SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs());

        fs::create_dir_all(checkpoint_dir)
            .map_err(|e| format!("Failed to create checkpoint directory: {}", e))?;

        let checkpoint_path = checkpoint_dir.join(format!("{}.json", id));
        let checkpoint = Checkpoint::new(id.clone(), problem, SynthesisPhase::Analyzing);

        checkpoint.save(&checkpoint_path)?;

        Ok(Self {
            id,
            checkpoint,
            checkpoint_path,
        })
    }

    /// Resume an existing session
    pub fn resume(checkpoint_path: &Path) -> Result<Self, String> {
        let checkpoint = Checkpoint::load(checkpoint_path)?;
        let id = checkpoint.id.clone();

        Ok(Self {
            id,
            checkpoint,
            checkpoint_path: checkpoint_path.to_path_buf(),
        })
    }

    /// Get the session ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get current checkpoint
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// Check if clarification is needed
    pub fn needs_clarification(&self) -> bool {
        !self.checkpoint.clarification.is_complete()
    }

    /// Ask clarification questions
    ///
    /// Returns the questions that need to be answered.
    pub fn ask_clarification(&self) -> Vec<Question> {
        self.checkpoint.clarification.unanswered().to_vec()
    }

    /// Answer a clarification question
    ///
    /// Updates the checkpoint with the answer. Returns true if all
    /// questions are now answered.
    pub fn answer_question(&mut self, answer: Answer) -> Result<bool, String> {
        let state = std::mem::take(&mut self.checkpoint.clarification);

        let (remaining_questions, mut answers) = match state {
            ClarificationState::Pending(questions) => {
                let answers: Vec<Answer> = Vec::new();
                (questions, answers)
            }
            ClarificationState::Answered(answers) => {
                // Already answered - verify this answer matches
                let existing = answers.iter()
                    .find(|a| a.question_id == answer.question_id);
                if let Some(existing) = existing {
                    if existing.value != answer.value {
                        return Err(format!(
                            "Answer for {} already given as {}, cannot change to {}",
                            answer.question_id, existing.value, answer.value
                        ));
                    }
                }
                return Ok(true);
            }
            ClarificationState::NotNeeded => {
                return Ok(true);
            }
        };

        // Remove the answered question
        let remaining: Vec<Question> = remaining_questions
            .into_iter()
            .filter(|q| q.id != answer.question_id)
            .collect();

        answers.push(answer);

        // Update state
        self.checkpoint.clarification = if remaining.is_empty() {
            ClarificationState::Answered(answers)
        } else {
            ClarificationState::Pending(remaining)
        };

        self.save_checkpoint()?;

        Ok(self.checkpoint.clarification.is_complete())
    }

    /// Execute one step of synthesis with partial execution
    ///
    /// This allows the user to see intermediate values and guide the
    /// synthesis process.
    pub fn execute_step<F>(&mut self, step_fn: F) -> Result<PartialResults, String>
    where
        F: FnOnce(&Problem, &[Answer]) -> Result<PartialResults, String>,
    {
        let answers = self.checkpoint.clarification.answered().to_vec();
        let problem = self.checkpoint.problem.as_ref()
            .ok_or_else(|| "Problem not available in checkpoint".to_string())?;
        let partial = step_fn(problem, &answers)?;

        // Update checkpoint with new partial results
        self.checkpoint.partial_results = partial.clone();

        self.save_checkpoint()?;

        Ok(partial)
    }

    /// Complete the synthesis
    pub fn complete(mut self, solution: PartialSolution) -> Result<(), String> {
        self.checkpoint.phase = SynthesisPhase::Complete;
        self.checkpoint.partial_results.best = Some(solution);
        self.save_checkpoint()?;
        Ok(())
    }

    /// Save current checkpoint state
    fn save_checkpoint(&self) -> Result<(), String> {
        self.checkpoint.save(&self.checkpoint_path)
    }

    /// Clean up checkpoint file
    pub fn cleanup(self) -> Result<(), String> {
        fs::remove_file(&self.checkpoint_path)
            .map_err(|e| format!("Failed to remove checkpoint: {}", e))
    }
}

/// Generate clarification questions for an ambiguous problem
pub fn ask_clarification(problem: &Problem) -> Vec<Question> {
    let mut questions = Vec::new();

    // Check if the problem signature is ambiguous
    if problem.signature.contains("...") || problem.signature.contains("Generic") {
        questions.push(
            Question::new(
                "type_params",
                "What type parameters should be used?"
            )
            .with_why("The problem signature uses generic types")
            .with_option("Integer (i64)", "i64")
            .with_option("Float (f64)", "f64")
            .with_option_explained("String", "String", "For text processing problems")
        );
    }

    // Check if examples are sparse
    if problem.examples.len() < 2 {
        questions.push(
            Question::new("more_examples", "Provide more example cases?")
                .with_why("Only one example provided - more examples help verification")
                .with_option("Yes, provide more", "yes")
                .with_option("No, proceed with current", "no")
        );
    }

    // Check for recursion ambiguity
    if problem.recursive_allowed {
        questions.push(
            Question::new("recursion_style", "Preferred recursion style?")
                .with_why("Problem allows recursive solutions")
                .with_option("Standard recursion", "standard")
                .with_option_explained("Tail recursion", "tail", "Optimized for tail-call")
                .with_option("Explicit stack", "explicit_stack")
                .with_default(0)
        );
    }

    questions
}

/// Convenience function to start an interactive synthesis session
pub fn execute_interactive(
    problem: Problem,
    checkpoint_dir: &Path,
) -> Result<InteractiveSession, String> {
    // Create session
    let mut session = InteractiveSession::new(problem, checkpoint_dir)?;

    // Determine if clarification is needed
    let checkpoint_problem = session.checkpoint.problem.as_ref()
        .ok_or_else(|| "Problem not available in checkpoint".to_string())?;
    let questions = ask_clarification(checkpoint_problem);

    if !questions.is_empty() {
        session.checkpoint = Checkpoint {
            clarification: ClarificationState::Pending(questions),
            ..session.checkpoint.clone()
        };
        session.save_checkpoint()?;
    }

    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_problem() -> Problem {
        Problem {
            name: "test".to_string(),
            category: "test",
            description: "test problem",
            signature: "fn test(x: i64) -> i64",
            examples: vec![],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn test_question_creation() {
        let q = Question::new("q1", "What type?")
            .with_option("Int", "i64")
            .with_option("Float", "f64")
            .with_why("Need to know type")
            .with_default(0);

        assert_eq!(q.id, "q1");
        assert_eq!(q.options.len(), 2);
        assert_eq!(q.default, Some(0));
    }

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.json");

        let problem = make_test_problem();
        let checkpoint = Checkpoint::new("test", problem, SynthesisPhase::Analyzing);

        checkpoint.save(&checkpoint_path).unwrap();
        let loaded = Checkpoint::load(&checkpoint_path).unwrap();

        assert_eq!(loaded.id, "test");
        assert_eq!(loaded.phase, SynthesisPhase::Analyzing);

        let _ = fs::remove_file(checkpoint_path);
    }

    #[test]
    fn test_session_lifecycle() {
        let temp_dir = std::env::temp_dir();
        let problem = make_test_problem();

        let session = InteractiveSession::new(problem, &temp_dir).unwrap();
        assert!(!session.needs_clarification());

        let _ = session.cleanup();
    }

    #[test]
    fn test_clarification_flow() {
        let problem = Problem {
            name: "recursive_test".to_string(),
            category: "test",
            description: "test",
            signature: "fn test(x: Generic) -> Generic",
            examples: vec![],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: true,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        let questions = ask_clarification(&problem);
        assert!(!questions.is_empty());

        let temp_dir = std::env::temp_dir();
        let session = execute_interactive(problem, &temp_dir).unwrap();

        assert!(session.needs_clarification());

        let unanswered = session.ask_clarification();
        assert_eq!(unanswered.len(), 2); // type_params and recursion_style

        let _ = session.cleanup();
    }

    #[test]
    fn test_intermediate_value() {
        let val = IntermediateValue::new("x", Value::Int(42), 0, "test value");
        assert_eq!(val.name, "x");
        assert_eq!(val.step, 0);
    }

    #[test]
    fn test_answer_question() {
        let temp_dir = std::env::temp_dir();
        let problem = make_test_problem();

        let questions = vec![
            Question::new("q1", "What?")
                .with_option("A", "a")
                .with_option("B", "b"),
        ];

        let mut checkpoint = Checkpoint::new("test", problem, SynthesisPhase::Analyzing);
        checkpoint.clarification = ClarificationState::Pending(questions);

        let mut session = InteractiveSession {
            id: "test".to_string(),
            checkpoint,
            checkpoint_path: temp_dir.join("test.json"),
        };

        let answer = Answer {
            question_id: "q1".to_string(),
            selected_index: 0,
            value: "a".to_string(),
        };

        let complete = session.answer_question(answer).unwrap();
        assert!(complete);
        assert!(!session.needs_clarification());
    }
}
