//! Example Synthesizer for nCPU/nSynth
//!
//! This module provides intelligent example generation for program synthesis.
//! It generates representative I/O examples, validates consistency, and ensures
//! comprehensive coverage of edge cases and diverse test scenarios.

use super::{Example, ParsedRequirements};
use serde_json::json;
use std::collections::HashSet;

/// Result of example generation
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Generated examples
    pub examples: Vec<Example>,

    /// Validation errors if any
    pub validation_errors: Vec<String>,

    /// Coverage statistics
    pub coverage: CoverageStats,
}

/// Coverage statistics for generated examples
#[derive(Debug, Clone)]
pub struct CoverageStats {
    /// Total examples generated
    pub total_examples: usize,

    /// Edge case examples
    pub edge_cases: usize,

    /// Typical case examples
    pub typical_cases: usize,

    /// Boundary condition examples
    pub boundary_cases: usize,

    /// Unique input patterns covered
    pub unique_patterns: usize,
}

/// Example Synthesizer for program synthesis
pub struct ExampleSynthesizer {
    /// Maximum number of examples to generate
    max_examples: usize,

    /// Minimum number of examples to generate
    min_examples: usize,

    /// Whether to include edge cases
    include_edge_cases: bool,

    /// Whether to generate diverse examples
    diverse_generation: bool,
}

impl ExampleSynthesizer {
    /// Create a new example synthesizer with default settings
    pub fn new() -> Self {
        Self {
            max_examples: 10,
            min_examples: 5,
            include_edge_cases: true,
            diverse_generation: true,
        }
    }

    /// Create with custom example count bounds
    pub fn with_bounds(min_examples: usize, max_examples: usize) -> Self {
        Self {
            max_examples,
            min_examples: min_examples.min(max_examples),
            include_edge_cases: true,
            diverse_generation: true,
        }
    }

    /// Set whether to include edge cases
    pub fn with_edge_cases(mut self, include: bool) -> Self {
        self.include_edge_cases = include;
        self
    }

    /// Set whether to use diverse generation
    pub fn with_diverse(mut self, diverse: bool) -> Self {
        self.diverse_generation = diverse;
        self
    }

    /// Get minimum examples bound
    pub fn min_examples(&self) -> usize {
        self.min_examples
    }

    /// Get maximum examples bound
    pub fn max_examples(&self) -> usize {
        self.max_examples
    }

    /// Generate examples from natural language description
    pub fn generate_examples(&self, description: &str) -> SynthesisResult {
        // Parse the description to infer problem characteristics
        let problem_type = self.infer_problem_type(description);

        // Generate base examples
        let mut examples = if self.diverse_generation {
            self.diverse_test_cases(&problem_type, description)
        } else {
            self.generate_typical_cases(&problem_type)
        };

        // Add edge cases if enabled
        if self.include_edge_cases {
            let edge_cases = self.generate_edge_cases(&problem_type);
            examples.extend(edge_cases);
        }

        // Ensure we have at least min_examples
        while examples.len() < self.min_examples {
            let filler = self.generate_filler_example(&problem_type, examples.len());
            examples.push(filler);
        }

        // Trim to max_examples
        examples.truncate(self.max_examples);

        // Validate consistency
        let validation_errors = self.validate_consistency(&examples);

        // Compute coverage statistics
        let coverage = self.compute_coverage(&examples);

        SynthesisResult {
            examples,
            validation_errors,
            coverage,
        }
    }

    /// Generate examples from existing parsed requirements
    pub fn from_requirements(&self, requirements: &ParsedRequirements) -> SynthesisResult {
        // If requirements already have examples, validate them
        if !requirements.examples.is_empty() {
            let examples = requirements.examples.clone();
            let validation_errors = self.validate_consistency(&examples);
            let coverage = self.compute_coverage(&examples);

            return SynthesisResult {
                examples,
                validation_errors,
                coverage,
            };
        }

        // Otherwise, generate from description
        self.generate_examples(&requirements.description)
    }

    /// Infer problem type from description
    fn infer_problem_type(&self, description: &str) -> ProblemType {
        let desc_lower = description.to_lowercase();

        // Pattern matching for common problem types
        if desc_lower.contains("sum") || desc_lower.contains("add") || desc_lower.contains("total")
        {
            ProblemType::Arithmetic(ArithmeticKind::Addition)
        } else if desc_lower.contains("max")
            || desc_lower.contains("maximum")
            || desc_lower.contains("largest")
        {
            ProblemType::Arithmetic(ArithmeticKind::Maximum)
        } else if desc_lower.contains("min")
            || desc_lower.contains("minimum")
            || desc_lower.contains("smallest")
        {
            ProblemType::Arithmetic(ArithmeticKind::Minimum)
        } else if desc_lower.contains("count")
            || desc_lower.contains("length")
            || desc_lower.contains("size")
        {
            ProblemType::Counting
        } else if desc_lower.contains("reverse") || desc_lower.contains("backward") {
            ProblemType::ListTransformation(ListKind::Reverse)
        } else if desc_lower.contains("sort") || desc_lower.contains("order") {
            ProblemType::ListTransformation(ListKind::Sort)
        } else if desc_lower.contains("filter") || desc_lower.contains("select") {
            ProblemType::ListTransformation(ListKind::Filter)
        } else if desc_lower.contains("list") || desc_lower.contains("array") {
            ProblemType::ListTransformation(ListKind::Map)
        } else {
            ProblemType::Generic
        }
    }

    /// Generate typical (non-edge) cases for a problem type
    fn generate_typical_cases(&self, problem_type: &ProblemType) -> Vec<Example> {
        match problem_type {
            ProblemType::Arithmetic(kind) => self.arithmetic_examples(kind, false),
            ProblemType::Counting => self.counting_examples(false),
            ProblemType::ListTransformation(kind) => self.list_transformation_examples(kind, false),
            ProblemType::Generic => self.generic_examples(),
        }
    }

    /// Generate edge cases for a problem type
    fn generate_edge_cases(&self, problem_type: &ProblemType) -> Vec<Example> {
        match problem_type {
            ProblemType::Arithmetic(kind) => self.arithmetic_examples(kind, true),
            ProblemType::Counting => self.counting_examples(true),
            ProblemType::ListTransformation(kind) => self.list_transformation_examples(kind, true),
            ProblemType::Generic => self.generic_edge_cases(),
        }
    }

    /// Generate diverse test cases covering multiple scenarios
    fn diverse_test_cases(&self, problem_type: &ProblemType, description: &str) -> Vec<Example> {
        let mut examples = Vec::new();

        // Add a few typical cases
        examples.extend(self.generate_typical_cases(problem_type));

        // Add boundary cases only if edge cases are enabled
        if self.include_edge_cases {
            examples.extend(self.boundary_cases(problem_type));
        }

        // Add cases based on description hints
        examples.extend(self.description_hints(description, problem_type));

        examples
    }

    /// Generate boundary condition examples
    fn boundary_cases(&self, problem_type: &ProblemType) -> Vec<Example> {
        match problem_type {
            ProblemType::Arithmetic(ArithmeticKind::Addition) => vec![
                Example {
                    inputs: vec![json!(0), json!(0)],
                    expected: json!(0),
                    explanation: Some("Identity element for addition".to_string()),
                },
                Example {
                    inputs: vec![json!(1), json!(1)],
                    expected: json!(2),
                    explanation: Some("Smallest positive addition".to_string()),
                },
            ],
            ProblemType::Arithmetic(ArithmeticKind::Maximum) => vec![Example {
                inputs: vec![json!(5), json!(5)],
                expected: json!(5),
                explanation: Some("Equal elements".to_string()),
            }],
            ProblemType::Counting => vec![
                Example {
                    inputs: vec![json!([])],
                    expected: json!(0),
                    explanation: Some("Empty list".to_string()),
                },
                Example {
                    inputs: vec![json!([1])],
                    expected: json!(1),
                    explanation: Some("Single element".to_string()),
                },
            ],
            ProblemType::ListTransformation(ListKind::Reverse) => vec![
                Example {
                    inputs: vec![json!([])],
                    expected: json!([]),
                    explanation: Some("Empty list reversal".to_string()),
                },
                Example {
                    inputs: vec![json!([1])],
                    expected: json!([1]),
                    explanation: Some("Single element reversal".to_string()),
                },
                Example {
                    inputs: vec![json!([1, 2])],
                    expected: json!([2, 1]),
                    explanation: Some("Two element reversal".to_string()),
                },
            ],
            _ => vec![],
        }
    }

    /// Extract hints from natural language description
    fn description_hints(&self, description: &str, _problem_type: &ProblemType) -> Vec<Example> {
        let mut examples = Vec::new();
        let desc_lower = description.to_lowercase();

        // Look for explicit examples in the description
        if let Some(examples_found) = self.extract_explicit_examples(description) {
            examples.extend(examples_found);
        }

        // Look for hints about input ranges
        if desc_lower.contains("positive") {
            examples.push(Example {
                inputs: vec![json!([1, 2, 3])],
                expected: json!(3), // Assuming counting/max context
                explanation: Some("Positive integers only".to_string()),
            });
        }

        if desc_lower.contains("negative") {
            examples.push(Example {
                inputs: vec![json!([-1, -2, -3])],
                expected: json!(-3), // Assuming min context
                explanation: Some("Negative integers only".to_string()),
            });
        }

        // Look for sorting/order hints
        if desc_lower.contains("sorted") || desc_lower.contains("ordered") {
            examples.push(Example {
                inputs: vec![json!([1, 2, 3, 4, 5])],
                expected: json!(5), // Assuming max/length context
                explanation: Some("Already sorted input".to_string()),
            });

            examples.push(Example {
                inputs: vec![json!([5, 4, 3, 2, 1])],
                expected: json!(5), // Assuming max/length context
                explanation: Some("Reverse sorted input".to_string()),
            });
        }

        examples
    }

    /// Extract explicit examples from description text
    fn extract_explicit_examples(&self, description: &str) -> Option<Vec<Example>> {
        // Look for patterns like "f(1) = 2" or "input: [1] -> output: 2"
        let mut examples = Vec::new();

        // Pattern 1: f(x) = y
        for cap in description.matches(r"f\(\d+\)\s*=\s*\d+") {
            // Simple parsing for demo
            if let Some(start) = cap.find('(') {
                if let Some(end) = cap.find(')') {
                    let input_str = &cap[start + 1..end];
                    if let Some(eq_pos) = cap.find('=') {
                        let output_str = &cap[eq_pos + 1..];
                        if let (Ok(input), Ok(output)) =
                            (input_str.parse::<i64>(), output_str.trim().parse::<i64>())
                        {
                            examples.push(Example {
                                inputs: vec![json!(input)],
                                expected: json!(output),
                                explanation: Some("Extracted from description".to_string()),
                            });
                        }
                    }
                }
            }
        }

        if examples.is_empty() {
            None
        } else {
            Some(examples)
        }
    }

    /// Generate arithmetic operation examples
    fn arithmetic_examples(&self, kind: &ArithmeticKind, edge_cases: bool) -> Vec<Example> {
        let mut examples = Vec::new();

        match kind {
            ArithmeticKind::Addition => {
                if edge_cases {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(0), json!(0)],
                            expected: json!(0),
                            explanation: Some("Zero addition".to_string()),
                        },
                        Example {
                            inputs: vec![json!(-5), json!(5)],
                            expected: json!(0),
                            explanation: Some("Negating addition".to_string()),
                        },
                    ]);
                } else {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(2), json!(3)],
                            expected: json!(5),
                            explanation: Some("Simple addition".to_string()),
                        },
                        Example {
                            inputs: vec![json!(10), json!(15)],
                            expected: json!(25),
                            explanation: Some("Two-digit addition".to_string()),
                        },
                        Example {
                            inputs: vec![json!(-3), json!(7)],
                            expected: json!(4),
                            explanation: Some("Mixed sign addition".to_string()),
                        },
                    ]);
                }
            }
            ArithmeticKind::Maximum => {
                if edge_cases {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(5), json!(5)],
                            expected: json!(5),
                            explanation: Some("Equal values".to_string()),
                        },
                        Example {
                            inputs: vec![json!(-10), json!(-5)],
                            expected: json!(-5),
                            explanation: Some("Negative maximum".to_string()),
                        },
                    ]);
                } else {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(3), json!(7)],
                            expected: json!(7),
                            explanation: Some("Simple maximum".to_string()),
                        },
                        Example {
                            inputs: vec![json!(10), json!(2)],
                            expected: json!(10),
                            explanation: Some("First is maximum".to_string()),
                        },
                    ]);
                }
            }
            ArithmeticKind::Minimum => {
                if edge_cases {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(5), json!(5)],
                            expected: json!(5),
                            explanation: Some("Equal values".to_string()),
                        },
                        Example {
                            inputs: vec![json!(0), json!(100)],
                            expected: json!(0),
                            explanation: Some("Zero vs large".to_string()),
                        },
                    ]);
                } else {
                    examples.extend(vec![
                        Example {
                            inputs: vec![json!(3), json!(7)],
                            expected: json!(3),
                            explanation: Some("Simple minimum".to_string()),
                        },
                        Example {
                            inputs: vec![json!(10), json!(2)],
                            expected: json!(2),
                            explanation: Some("Second is minimum".to_string()),
                        },
                    ]);
                }
            }
        }

        examples
    }

    /// Generate counting operation examples
    fn counting_examples(&self, edge_cases: bool) -> Vec<Example> {
        if edge_cases {
            vec![
                Example {
                    inputs: vec![json!([])],
                    expected: json!(0),
                    explanation: Some("Empty list count".to_string()),
                },
                Example {
                    inputs: vec![json!([1])],
                    expected: json!(1),
                    explanation: Some("Single element count".to_string()),
                },
            ]
        } else {
            vec![
                Example {
                    inputs: vec![json!([1, 2, 3])],
                    expected: json!(3),
                    explanation: Some("Simple count".to_string()),
                },
                Example {
                    inputs: vec![json!([5, 5, 5, 5, 5])],
                    expected: json!(5),
                    explanation: Some("Repeated elements count".to_string()),
                },
            ]
        }
    }

    /// Generate list transformation examples
    fn list_transformation_examples(&self, kind: &ListKind, edge_cases: bool) -> Vec<Example> {
        match kind {
            ListKind::Reverse => {
                if edge_cases {
                    vec![
                        Example {
                            inputs: vec![json!([])],
                            expected: json!([]),
                            explanation: Some("Empty list reversal".to_string()),
                        },
                        Example {
                            inputs: vec![json!([1])],
                            expected: json!([1]),
                            explanation: Some("Single element reversal".to_string()),
                        },
                    ]
                } else {
                    vec![
                        Example {
                            inputs: vec![json!([1, 2, 3])],
                            expected: json!([3, 2, 1]),
                            explanation: Some("Simple reversal".to_string()),
                        },
                        Example {
                            inputs: vec![json!([5, 4, 3, 2, 1])],
                            expected: json!([1, 2, 3, 4, 5]),
                            explanation: Some("Reverse sorted reversal".to_string()),
                        },
                    ]
                }
            }
            ListKind::Sort => {
                if edge_cases {
                    vec![Example {
                        inputs: vec![json!([])],
                        expected: json!([]),
                        explanation: Some("Empty list sort".to_string()),
                    }]
                } else {
                    vec![
                        Example {
                            inputs: vec![json!([3, 1, 2])],
                            expected: json!([1, 2, 3]),
                            explanation: Some("Simple sort".to_string()),
                        },
                        Example {
                            inputs: vec![json!([5, 5, 3, 1, 3])],
                            expected: json!([1, 3, 3, 5, 5]),
                            explanation: Some("With duplicates sort".to_string()),
                        },
                    ]
                }
            }
            ListKind::Map | ListKind::Filter => {
                // For map/filter, return counting examples as a fallback
                self.counting_examples(edge_cases)
            }
        }
    }

    /// Generate generic examples when problem type is unknown
    fn generic_examples(&self) -> Vec<Example> {
        vec![
            Example {
                inputs: vec![json!(1), json!(2)],
                expected: json!(3),
                explanation: Some("Generic case 1".to_string()),
            },
            Example {
                inputs: vec![json!(5)],
                expected: json!(5),
                explanation: Some("Generic case 2".to_string()),
            },
        ]
    }

    /// Generate generic edge cases
    fn generic_edge_cases(&self) -> Vec<Example> {
        vec![
            Example {
                inputs: vec![json!(0)],
                expected: json!(0),
                explanation: Some("Zero input".to_string()),
            },
            Example {
                inputs: vec![json!(1)],
                expected: json!(1),
                explanation: Some("Single unit input".to_string()),
            },
        ]
    }

    /// Generate a filler example to meet minimum count
    fn generate_filler_example(&self, problem_type: &ProblemType, index: usize) -> Example {
        // Generate a reasonable example based on problem type
        match problem_type {
            ProblemType::Arithmetic(ArithmeticKind::Addition) => {
                let a = (index as i64) + 1;
                let b = (index as i64) + 2;
                Example {
                    inputs: vec![json!(a), json!(b)],
                    expected: json!(a + b),
                    explanation: Some(format!("Filler addition: {} + {}", a, b)),
                }
            }
            ProblemType::Counting => {
                let len = (index + 1).min(10);
                let list: Vec<i64> = (1..=len as i64).collect();
                Example {
                    inputs: vec![json!(list)],
                    expected: json!(len as i64),
                    explanation: Some(format!("Filler counting: length {}", len)),
                }
            }
            _ => Example {
                inputs: vec![json!(index as i64)],
                expected: json!(index as i64),
                explanation: Some("Filler example".to_string()),
            },
        }
    }

    /// Validate consistency of generated examples
    pub fn validate_consistency(&self, examples: &[Example]) -> Vec<String> {
        let mut errors = Vec::new();

        // Check for duplicate inputs with different outputs
        let mut seen_inputs = std::collections::HashMap::new();

        for (i, example) in examples.iter().enumerate() {
            // Convert inputs to comparable key
            let inputs_key = self.inputs_to_key(&example.inputs);

            if let Some(&prev_idx) = seen_inputs.get(&inputs_key) {
                let prev_example: &Example = &examples[prev_idx];

                if let (Some(prev_val), Some(curr_val)) =
                    (prev_example.expected.as_i64(), example.expected.as_i64())
                {
                    if prev_val != curr_val {
                        errors.push(format!(
                            "Inconsistency: Examples {} and {} have same inputs but different outputs ({} vs {})",
                            prev_idx, i, prev_val, curr_val
                        ));
                    }
                }
            } else {
                seen_inputs.insert(inputs_key, i);
            }
        }

        // Check for type consistency
        for (i, example) in examples.iter().enumerate() {
            if let Err(e) = self.check_example_types(example) {
                errors.push(format!("Example {}: {}", i, e));
            }
        }

        errors
    }

    /// Convert inputs to hashable key
    fn inputs_to_key(&self, inputs: &[serde_json::Value]) -> String {
        format!("{:?}", inputs)
    }

    /// Check type consistency of an example
    fn check_example_types(&self, example: &Example) -> Result<(), String> {
        // All inputs should be valid JSON values
        for (i, input) in example.inputs.iter().enumerate() {
            if input.is_null() {
                return Err(format!("Input {} is null", i));
            }
        }

        // Output should not be null
        if example.expected.is_null() {
            return Err("Output is null".to_string());
        }

        Ok(())
    }

    /// Compute coverage statistics
    fn compute_coverage(&self, examples: &[Example]) -> CoverageStats {
        let total = examples.len();

        let mut edge_cases = 0;
        let mut boundary_cases = 0;
        let mut typical_cases = 0;

        let mut unique_patterns = HashSet::new();

        for example in examples {
            // Classify example type
            let is_edge = self.is_edge_case(example);
            let is_boundary = self.is_boundary_case(example);

            if is_edge {
                edge_cases += 1;
            } else if is_boundary {
                boundary_cases += 1;
            } else {
                typical_cases += 1;
            }

            // Track unique input patterns (simplified)
            let pattern = self.input_pattern(example);
            unique_patterns.insert(pattern);
        }

        CoverageStats {
            total_examples: total,
            edge_cases,
            boundary_cases,
            typical_cases,
            unique_patterns: unique_patterns.len(),
        }
    }

    /// Determine if example is an edge case
    fn is_edge_case(&self, example: &Example) -> bool {
        // Check for empty inputs
        if example.inputs.iter().any(|i| {
            if let Some(arr) = i.as_array() {
                arr.is_empty()
            } else {
                false
            }
        }) {
            return true;
        }

        // Check for zero values
        if example.inputs.iter().any(|i| {
            if let Some(n) = i.as_i64() {
                n == 0
            } else {
                false
            }
        }) {
            return true;
        }

        // Check for negative values
        if example.inputs.iter().any(|i| {
            if let Some(n) = i.as_i64() {
                n < 0
            } else {
                false
            }
        }) {
            return true;
        }

        false
    }

    /// Determine if example is a boundary case
    fn is_boundary_case(&self, example: &Example) -> bool {
        // Check for single elements
        if example.inputs.iter().any(|i| {
            if let Some(arr) = i.as_array() {
                arr.len() == 1
            } else {
                false
            }
        }) {
            return true;
        }

        // Check for very small positive values (1)
        if example.inputs.iter().any(|i| {
            if let Some(n) = i.as_i64() {
                n == 1
            } else {
                false
            }
        }) {
            return true;
        }

        false
    }

    /// Extract input pattern for uniqueness tracking
    fn input_pattern(&self, example: &Example) -> String {
        let mut pattern = String::new();

        for input in &example.inputs {
            if input.is_array() {
                pattern.push('L'); // List
            } else if input.is_i64() {
                pattern.push('I'); // Integer
            } else if input.is_string() {
                pattern.push('S'); // String
            } else {
                pattern.push('?');
            }
        }

        pattern
    }
}

/// Problem type classification
#[derive(Debug, Clone, PartialEq)]
enum ProblemType {
    /// Arithmetic operations
    Arithmetic(ArithmeticKind),

    /// Counting/length operations
    Counting,

    /// List transformations
    ListTransformation(ListKind),

    /// Generic/unknown problem type
    Generic,
}

/// Arithmetic operation kind
#[derive(Debug, Clone, Copy, PartialEq)]
enum ArithmeticKind {
    Addition,
    Maximum,
    Minimum,
}

/// List transformation kind
#[derive(Debug, Clone, Copy, PartialEq)]
enum ListKind {
    Reverse,
    Sort,
    Filter,
    Map,
}

impl Default for ExampleSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl::{InputSpec, OutputSpec};

    #[test]
    fn test_basic_synthesis() {
        let synthesizer = ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add two numbers");

        assert!(!result.examples.is_empty(), "Should generate examples");
        assert!(
            result.examples.len() >= 5,
            "Should generate at least min_examples"
        );
        assert!(
            result.examples.len() <= 10,
            "Should not exceed max_examples"
        );
    }

    #[test]
    fn test_custom_bounds() {
        let synthesizer = ExampleSynthesizer::with_bounds(3, 7);
        let result = synthesizer.generate_examples("count list length");

        assert!(result.examples.len() >= 3, "Should respect min_examples");
        assert!(result.examples.len() <= 7, "Should respect max_examples");
    }

    #[test]
    fn test_edge_cases_generation() {
        let synthesizer = ExampleSynthesizer::new().with_edge_cases(true);
        let result = synthesizer.generate_examples("count list elements");

        // Should have edge cases like empty list
        let has_empty = result.examples.iter().any(|e| {
            e.inputs.iter().any(|i| {
                if let Some(arr) = i.as_array() {
                    arr.is_empty()
                } else {
                    false
                }
            })
        });

        assert!(has_empty, "Should include empty list edge case");
    }

    #[test]
    fn test_no_edge_cases() {
        let synthesizer = ExampleSynthesizer::new().with_edge_cases(false);
        let result = synthesizer.generate_examples("add numbers");

        // Should not have zero inputs when edge cases disabled
        let has_zero = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.as_i64() == Some(0)));

        assert!(!has_zero, "Should not include zero edge case when disabled");
    }

    #[test]
    fn test_consistency_validation() {
        let synthesizer = ExampleSynthesizer::new();
        let examples = vec![
            Example {
                inputs: vec![json!(2), json!(3)],
                expected: json!(5),
                explanation: None,
            },
            Example {
                inputs: vec![json!(2), json!(3)], // Same inputs
                expected: json!(7),               // Different output!
                explanation: None,
            },
        ];

        let errors = synthesizer.validate_consistency(&examples);
        assert!(!errors.is_empty(), "Should detect inconsistency");
        assert!(
            errors[0].contains("Inconsistency"),
            "Error should mention inconsistency"
        );
    }

    #[test]
    fn test_coverage_stats() {
        let synthesizer = ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse list");

        assert_eq!(result.coverage.total_examples, result.examples.len());
        assert!(
            result.coverage.unique_patterns > 0,
            "Should track unique patterns"
        );
    }

    #[test]
    fn test_problem_type_inference() {
        let synthesizer = ExampleSynthesizer::new();

        // Test addition inference
        let add_type = synthesizer.infer_problem_type("add two numbers together");
        assert_eq!(add_type, ProblemType::Arithmetic(ArithmeticKind::Addition));

        // Test counting inference
        let count_type = synthesizer.infer_problem_type("count the elements");
        assert_eq!(count_type, ProblemType::Counting);

        // Test reverse inference
        let rev_type = synthesizer.infer_problem_type("reverse the list");
        assert_eq!(rev_type, ProblemType::ListTransformation(ListKind::Reverse));
    }

    #[test]
    fn test_from_requirements() {
        let synthesizer = ExampleSynthesizer::new();

        let requirements = ParsedRequirements {
            function_name: "add".to_string(),
            inputs: vec![
                InputSpec {
                    name: "a".to_string(),
                    type_: "int".to_string(),
                    description: None,
                },
                InputSpec {
                    name: "b".to_string(),
                    type_: "int".to_string(),
                    description: None,
                },
            ],
            output: OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "Add two numbers".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let result = synthesizer.from_requirements(&requirements);
        assert!(
            !result.examples.is_empty(),
            "Should generate from requirements"
        );
    }

    #[test]
    fn test_existing_examples_validation() {
        let synthesizer = ExampleSynthesizer::new();

        let requirements = ParsedRequirements {
            function_name: "add".to_string(),
            inputs: vec![InputSpec {
                name: "a".to_string(),
                type_: "int".to_string(),
                description: None,
            }],
            output: OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "Test function".to_string(),
            examples: vec![Example {
                inputs: vec![json!(5)],
                expected: json!(5),
                explanation: None,
            }],
            constraints: vec![],
        };

        let result = synthesizer.from_requirements(&requirements);
        assert_eq!(
            result.examples.len(),
            1,
            "Should preserve existing examples"
        );
        assert!(
            result.validation_errors.is_empty(),
            "Valid example should pass"
        );
    }
}
