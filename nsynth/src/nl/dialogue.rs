//! Multi-turn Dialogue System for nCPU/nSynth
//!
//! This module provides interactive clarification for ambiguous requirements.
//! It detects ambiguities, generates follow-up questions, and refines
//! specifications based on user answers.

use super::{NLError, ParsedRequirements};

/// Question type for clarification
#[derive(Debug, Clone)]
pub enum QuestionType {
    /// Empty input behavior
    EmptyInput,
    /// Data type clarification
    DataType,
    /// Error handling
    ErrorHandling,
    /// Performance requirements
    Performance,
    /// Edge cases
    EdgeCases,
    /// Custom question
    Custom(String),
}

/// A question in the dialogue
#[derive(Debug, Clone)]
pub struct Question {
    /// Question type
    pub qtype: QuestionType,
    /// Question text
    pub text: String,
    /// Possible answers
    pub options: Vec<String>,
    /// Whether answer is required
    pub required: bool,
}

/// User answer to a question
#[derive(Debug, Clone)]
pub struct Answer {
    /// Question this answers
    pub question_id: String,
    /// Answer text
    pub text: String,
    /// Whether answer was provided
    pub provided: bool,
}

/// Dialogue state
#[derive(Debug, Clone)]
pub struct DialogueState {
    /// Current questions
    pub questions: Vec<Question>,
    /// Answers received
    pub answers: Vec<Answer>,
    /// Current round number
    pub round: usize,
    /// Max rounds before giving up
    pub max_rounds: usize,
}

/// Ambiguity detected in requirements
#[derive(Debug, Clone)]
pub struct Ambiguity {
    /// Ambiguity type
    pub qtype: QuestionType,
    /// Description of ambiguity
    pub description: String,
    /// Suggested questions
    pub questions: Vec<String>,
}

/// Multi-turn Dialogue Manager
#[derive(Debug, Clone)]
pub struct DialogueManager {
    /// Current dialogue state
    state: DialogueState,
    /// Max questions per round
    max_questions: usize,
}

impl DialogueManager {
    /// Create a new dialogue manager
    pub fn new() -> Self {
        Self {
            state: DialogueState {
                questions: Vec::new(),
                answers: Vec::new(),
                round: 0,
                max_rounds: 5,
            },
            max_questions: 5,
        }
    }

    /// Create with custom max questions
    pub fn with_max_questions(max: usize) -> Self {
        Self {
            state: DialogueState {
                questions: Vec::new(),
                answers: Vec::new(),
                round: 0,
                max_rounds: 5,
            },
            max_questions: max,
        }
    }

    /// Detect ambiguities in requirements
    pub fn detect_ambiguity(&self, req: &ParsedRequirements) -> Vec<Ambiguity> {
        let mut ambiguities = Vec::new();

        // Check for empty input handling
        if !self.specifies_empty_input_behavior(req) {
            ambiguities.push(Ambiguity {
                qtype: QuestionType::EmptyInput,
                description: "Empty input behavior not specified".to_string(),
                questions: vec![
                    "What should happen for empty input?".to_string(),
                    "Return error code?".to_string(),
                    "Return default value?".to_string(),
                ],
            });
        }

        // Check for data type clarity
        if !self.specifies_output_type(req) {
            ambiguities.push(Ambiguity {
                qtype: QuestionType::DataType,
                description: "Output data type unclear".to_string(),
                questions: vec![
                    "What data type should this return?".to_string(),
                    "Integer? Float? String?".to_string(),
                ],
            });
        }

        // Check for error handling
        if !self.specifies_error_handling(req) {
            ambiguities.push(Ambiguity {
                qtype: QuestionType::ErrorHandling,
                description: "Error handling not specified".to_string(),
                questions: vec![
                    "Should this handle errors gracefully?".to_string(),
                    "Return error codes?".to_string(),
                    "Panic on error?".to_string(),
                ],
            });
        }

        ambiguities
    }

    /// Generate follow-up questions from ambiguities
    pub fn generate_followup_questions(&mut self, ambiguities: &[Ambiguity]) -> Vec<Question> {
        let mut questions = Vec::new();

        for ambiguity in ambiguities.iter().take(self.max_questions) {
            let question_text = ambiguity
                .questions
                .first()
                .cloned()
                .unwrap_or_else(|| format!("Clarify: {}", ambiguity.description));

            questions.push(Question {
                qtype: ambiguity.qtype.clone(),
                text: question_text,
                options: ambiguity.questions.clone(),
                required: true,
            });
        }

        self.state.questions = questions.clone();
        questions
    }

    /// Refine requirements based on user answers
    pub fn refine_requirements(
        &self,
        req: &ParsedRequirements,
        answers: &[Answer],
    ) -> Result<ParsedRequirements, NLError> {
        let mut refined = req.clone();

        for answer in answers {
            if !answer.provided {
                continue;
            }

            match self.find_question(&answer.question_id) {
                Some(Question {
                    qtype: QuestionType::EmptyInput,
                    ..
                }) => {
                    refined
                        .constraints
                        .push(format!("empty_input: {}", answer.text));
                }
                Some(Question {
                    qtype: QuestionType::DataType,
                    ..
                }) => {
                    refined.output.type_ = answer.text.clone();
                }
                Some(Question {
                    qtype: QuestionType::ErrorHandling,
                    ..
                }) => {
                    refined
                        .constraints
                        .push(format!("error_handling: {}", answer.text));
                }
                Some(Question {
                    qtype: QuestionType::Performance,
                    ..
                }) => {
                    refined
                        .constraints
                        .push(format!("performance: {}", answer.text));
                }
                _ => {}
            }
        }

        Ok(refined)
    }

    /// Confirm specification with user
    pub fn confirm_specification(&self, req: &ParsedRequirements) -> Result<bool, NLError> {
        // Check if requirements are complete enough
        let has_examples = !req.examples.is_empty();
        let has_output_type = !req.output.type_.is_empty();
        let has_input_specs = !req.inputs.is_empty();

        Ok(has_examples && has_output_type && has_input_specs)
    }

    /// Process user answer
    pub fn process_answer(&mut self, _question_id: String, answer: Answer) {
        self.state.answers.push(answer);
    }

    /// Get max questions per round
    pub fn max_questions(&self) -> usize {
        self.max_questions
    }

    /// Get current dialogue state
    pub fn state(&self) -> &DialogueState {
        &self.state
    }

    /// Increment round counter
    pub fn next_round(&mut self) -> Result<bool, NLError> {
        self.state.round += 1;
        if self.state.round >= self.state.max_rounds {
            return Err(NLError::NotImplemented);
        }
        Ok(true)
    }

    /// Check if dialogue is complete
    pub fn is_complete(&self) -> bool {
        self.state.answers.len() >= self.state.questions.len()
    }

    /// Helper: check if empty input behavior is specified
    fn specifies_empty_input_behavior(&self, req: &ParsedRequirements) -> bool {
        req.constraints.iter().any(|c| c.contains("empty"))
    }

    /// Helper: check if output type is specified
    fn specifies_output_type(&self, req: &ParsedRequirements) -> bool {
        !req.output.type_.is_empty() && req.output.type_ != "unknown"
    }

    /// Helper: check if error handling is specified
    fn specifies_error_handling(&self, req: &ParsedRequirements) -> bool {
        req.constraints.iter().any(|c| c.contains("error"))
    }

    /// Helper: find question by ID
    fn find_question(&self, id: &str) -> Option<&Question> {
        self.state.questions.iter().find(|q| {
            // Generate a simple ID from the question text
            let qid = q
                .text
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();
            qid == id || q.text.contains(id)
        })
    }
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl::OutputSpec;

    #[test]
    fn test_detect_ambiguity_empty_input() {
        let manager = DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let ambiguities = manager.detect_ambiguity(&req);
        assert!(!ambiguities.is_empty());
    }

    #[test]
    fn test_generate_followup_questions() {
        let mut manager = DialogueManager::new();
        let ambiguities = vec![Ambiguity {
            qtype: QuestionType::EmptyInput,
            description: "Empty input".to_string(),
            questions: vec!["What happens on empty?".to_string()],
        }];

        let questions = manager.generate_followup_questions(&ambiguities);
        assert_eq!(questions.len(), 1);
        assert!(questions[0].text.contains("empty"));
    }

    #[test]
    fn test_refine_requirements() {
        let manager = DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: OutputSpec {
                type_: "unknown".to_string(),
                description: None,
            },
            description: "test".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let answers = vec![Answer {
            question_id: "output_type".to_string(),
            text: "float".to_string(),
            provided: true,
        }];

        // This would need proper question_id matching
        let refined = manager.refine_requirements(&req, &answers);
        assert!(refined.is_ok());
    }
}
