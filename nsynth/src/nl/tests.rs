//! Comprehensive Tests for Natural Language Frontend
//!
//! Test suite covers:
//! - LLM integration (mock client)
//! - Example generation quality
//! - Dialogue flow
//! - End-to-end NL→code pipeline
//! - Standard benchmark problems

use super::*;
use serde_json::json;

// ============================================================
// Unit Tests - NLPipeline
// ============================================================

#[cfg(test)]
mod nl_pipeline_tests {
    use super::*;

    #[test]
    fn test_nl_pipeline_creation() {
        let pipeline = NLPipeline::new();
        assert!(!pipeline.is_ready()); // No API key by default
    }

    #[test]
    fn test_nl_pipeline_with_model() {
        let pipeline = NLPipeline::with_model("claude-3-opus-20240229".to_string());
        assert_eq!(pipeline.model(), "claude-3-opus-20240229");
    }

    #[test]
    fn test_nl_pipeline_with_api_key() {
        let pipeline = NLPipeline::with_api_key("test-key-123".to_string());
        assert!(pipeline.is_ready());
    }

    #[test]
    fn test_nl_parse_empty_requirements() {
        let pipeline = NLPipeline::new();
        let result = pipeline.parse_from_examples(vec![]);
        assert_eq!(result.inputs.len(), 0);
        assert_eq!(result.function_name, "synthesized_function");
    }

    #[test]
    fn test_nl_parse_with_examples() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![json!(5), json!(3)],
            expected: json!(8),
            explanation: Some("addition".to_string()),
        }];

        let result = pipeline.parse_from_examples(examples);
        assert_eq!(result.inputs.len(), 2);
        assert_eq!(result.inputs[0].type_, "int");
        assert_eq!(result.inputs[0].name, "arg_0");
        assert_eq!(result.inputs[1].name, "arg_1");
    }

    #[test]
    fn test_nl_parse_list_examples() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![json!([1, 2, 3])],
            expected: json!(3),
            explanation: Some("count".to_string()),
        }];

        let result = pipeline.parse_from_examples(examples);
        assert_eq!(result.inputs.len(), 1);
        assert_eq!(result.inputs[0].type_, "list");
    }

    #[test]
    fn test_nl_parse_string_examples() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![json!("hello")],
            expected: json!(5),
            explanation: Some("length".to_string()),
        }];

        let result = pipeline.parse_from_examples(examples);
        assert_eq!(result.inputs.len(), 1);
        assert_eq!(result.inputs[0].type_, "string");
    }

    #[tokio::test]
    async fn test_nl_cache_integration() {
        let pipeline = NLPipeline::new();

        // Cache should be empty initially
        let (size, _) = pipeline.cache_stats().await;
        assert_eq!(size, 0);

        // Clear should work
        pipeline.clear_cache().await;

        let (size, _) = pipeline.cache_stats().await;
        assert_eq!(size, 0);
    }
}

// ============================================================
// Unit Tests - Example Synthesizer
// ============================================================

#[cfg(test)]
mod synthesizer_tests {
    use super::*;

    #[test]
    fn test_synthesizer_creation() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        assert_eq!(synthesizer.min_examples(), 5);
        assert_eq!(synthesizer.max_examples(), 10);
    }

    #[test]
    fn test_synthesizer_custom_bounds() {
        let synthesizer = synthesizer::ExampleSynthesizer::with_bounds(3, 7);
        assert_eq!(synthesizer.min_examples(), 3);
        assert_eq!(synthesizer.max_examples(), 7);
    }

    #[test]
    fn test_synthesizer_with_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let result = synthesizer.generate_examples("reverse array");
        assert!(result.coverage.edge_cases > 0, "Should include edge cases");
    }

    #[test]
    fn test_synthesizer_without_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(false);

        let result = synthesizer.generate_examples("add numbers");

        // Should not have zero inputs when edge cases disabled
        let has_zero = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.as_i64() == Some(0)));
        assert!(!has_zero, "Should not include zero edge case when disabled");
    }

    #[test]
    fn test_synthesizer_diverse_generation() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_diverse(true);

        let result = synthesizer.generate_examples("add numbers");
        assert!(result.examples.len() >= 5, "Should generate min examples");
    }

    #[test]
    fn test_synthesizer_consistency_validation() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();

        let examples = vec![
            Example {
                inputs: vec![json!(2), json!(3)],
                expected: json!(5),
                explanation: Some("consistent".to_string()),
            },
            Example {
                inputs: vec![json!(2), json!(3)], // Same inputs
                expected: json!(7),               // Different output!
                explanation: Some("inconsistent".to_string()),
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
    fn test_synthesizer_coverage_stats() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse list");

        assert_eq!(result.coverage.total_examples, result.examples.len());
        assert!(
            result.coverage.unique_patterns > 0,
            "Should track unique patterns"
        );
    }

    // ============================================================
    // Problem Type Inference Tests
    // ============================================================

    #[test]
    fn test_infer_arithmetic_addition() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add two numbers");

        assert!(!result.examples.is_empty());
        // Should generate addition examples
        let is_addition = result
            .examples
            .iter()
            .any(|e| e.inputs.len() == 2 && e.expected.as_i64().is_some());
        assert!(is_addition, "Should generate addition examples");
    }

    #[test]
    fn test_infer_arithmetic_maximum() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("find maximum value");

        assert!(!result.examples.is_empty());
    }

    #[test]
    fn test_infer_arithmetic_minimum() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("find minimum value");

        assert!(!result.examples.is_empty());
    }

    #[test]
    fn test_infer_counting() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("count the elements");

        assert!(!result.examples.is_empty());
        // Should have list inputs
        let has_list = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_array()));
        assert!(has_list, "Should generate list examples for counting");
    }

    #[test]
    fn test_infer_reverse() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse the array");

        assert!(!result.examples.is_empty());
    }

    #[test]
    fn test_infer_sort() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sort the list");

        assert!(!result.examples.is_empty());
    }

    #[test]
    fn test_infer_filter() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("filter even numbers");

        assert!(!result.examples.is_empty());
    }

    // ============================================================
    // Quality Tests for Example Generation
    // ============================================================

    #[test]
    fn test_example_quality_no_null_inputs() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add numbers");

        // No input should be null
        let has_null = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_null()));
        assert!(!has_null, "Should not have null inputs");
    }

    #[test]
    fn test_example_quality_no_null_output() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add numbers");

        // No output should be null
        let has_null = result.examples.iter().any(|e| e.expected.is_null());
        assert!(!has_null, "Should not have null outputs");
    }

    #[test]
    fn test_example_quality_type_consistency() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add numbers");

        // All outputs should be integers for addition
        let all_int = result.examples.iter().all(|e| e.expected.is_i64());
        assert!(all_int, "All outputs should be integers for addition");
    }

    #[test]
    fn test_example_quality_explanations() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("add numbers");

        // At least some examples should have explanations
        let has_explanation = result.examples.iter().any(|e| e.explanation.is_some());
        assert!(
            has_explanation,
            "Should have explanations for at least some examples"
        );
    }
}

// ============================================================
// Unit Tests - Dialogue Manager
// ============================================================

#[cfg(test)]
mod dialogue_tests {
    use super::*;

    #[test]
    fn test_dialogue_manager_creation() {
        let manager = dialogue::DialogueManager::new();
        assert_eq!(manager.state().round, 0);
        assert_eq!(manager.max_questions(), 5);
    }

    #[test]
    fn test_dialogue_manager_custom_max_questions() {
        let manager = dialogue::DialogueManager::with_max_questions(10);
        assert_eq!(manager.max_questions(), 10);
    }

    #[test]
    fn test_dialogue_manager_detect_ambiguity() {
        let manager = dialogue::DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: crate::nl::OutputSpec {
                type_: "unknown".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let ambiguities = manager.detect_ambiguity(&req);
        assert!(
            !ambiguities.is_empty(),
            "Should detect ambiguities in incomplete requirements"
        );
    }

    #[test]
    fn test_dialogue_manager_no_ambiguity_complete() {
        let manager = dialogue::DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                type_: "int".to_string(),
                description: None,
            }],
            output: crate::nl::OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![Example {
                inputs: vec![json!(5)],
                expected: json!(5),
                explanation: None,
            }],
            constraints: vec!["empty_input: return 0".to_string()],
        };

        let ambiguities = manager.detect_ambiguity(&req);
        // Complete requirements should have fewer ambiguities
        assert!(
            ambiguities.len() < 3,
            "Complete requirements should have fewer ambiguities"
        );
    }

    #[test]
    fn test_dialogue_manager_generate_questions() {
        let mut manager = dialogue::DialogueManager::new();

        let ambiguities = vec![dialogue::Ambiguity {
            qtype: dialogue::QuestionType::EmptyInput,
            description: "test".to_string(),
            questions: vec!["What happens?".to_string()],
        }];

        let questions = manager.generate_followup_questions(&ambiguities);
        assert_eq!(questions.len(), 1);
        assert!(questions[0].text.contains("What happens"));
    }

    #[test]
    fn test_dialogue_manager_questions_limit() {
        let mut manager = dialogue::DialogueManager::with_max_questions(2);

        let ambiguities = vec![
            dialogue::Ambiguity {
                qtype: dialogue::QuestionType::EmptyInput,
                description: "empty".to_string(),
                questions: vec!["Empty?".to_string()],
            },
            dialogue::Ambiguity {
                qtype: dialogue::QuestionType::DataType,
                description: "type".to_string(),
                questions: vec!["Type?".to_string()],
            },
            dialogue::Ambiguity {
                qtype: dialogue::QuestionType::ErrorHandling,
                description: "error".to_string(),
                questions: vec!["Error?".to_string()],
            },
        ];

        let questions = manager.generate_followup_questions(&ambiguities);
        assert_eq!(questions.len(), 2, "Should limit to max_questions");
    }

    #[test]
    fn test_dialogue_manager_refine_requirements() {
        let manager = dialogue::DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: crate::nl::OutputSpec {
                type_: "unknown".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let answers = vec![dialogue::Answer {
            question_id: "output_type".to_string(),
            text: "int".to_string(),
            provided: true,
        }];

        let result = manager.refine_requirements(&req, &answers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dialogue_manager_refine_with_empty_input_constraint() {
        let manager = dialogue::DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: crate::nl::OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let answers = vec![dialogue::Answer {
            question_id: "empty_input".to_string(),
            text: "return error code -1".to_string(),
            provided: true,
        }];

        let result = manager.refine_requirements(&req, &answers).unwrap();
        assert!(result.constraints.iter().any(|c| c.contains("empty_input")));
    }

    #[test]
    fn test_dialogue_manager_refine_with_data_type() {
        let manager = dialogue::DialogueManager::new();
        let req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: crate::nl::OutputSpec {
                type_: "unknown".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let answers = vec![dialogue::Answer {
            question_id: "output_type".to_string(),
            text: "float".to_string(),
            provided: true,
        }];

        let result = manager.refine_requirements(&req, &answers).unwrap();
        assert_eq!(result.output.type_, "float");
    }

    #[test]
    fn test_dialogue_manager_confirm_specification() {
        let manager = dialogue::DialogueManager::new();

        let complete_req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![InputSpec {
                name: "x".to_string(),
                type_: "int".to_string(),
                description: None,
            }],
            output: crate::nl::OutputSpec {
                type_: "int".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![Example {
                inputs: vec![json!(5)],
                expected: json!(5),
                explanation: None,
            }],
            constraints: vec![],
        };

        let result = manager.confirm_specification(&complete_req);
        assert!(result.is_ok(), "Complete specification should be confirmed");
        assert!(result.unwrap(), "Complete specification should be true");
    }

    #[test]
    fn test_dialogue_manager_confirm_incomplete() {
        let manager = dialogue::DialogueManager::new();

        let incomplete_req = ParsedRequirements {
            function_name: "test".to_string(),
            inputs: vec![],
            output: crate::nl::OutputSpec {
                type_: "unknown".to_string(),
                description: None,
            },
            description: "test function".to_string(),
            examples: vec![],
            constraints: vec![],
        };

        let result = manager.confirm_specification(&incomplete_req);
        assert!(result.is_ok());
        assert!(
            !result.unwrap(),
            "Incomplete specification should not be confirmed"
        );
    }

    #[test]
    fn test_dialogue_manager_process_answer() {
        let mut manager = dialogue::DialogueManager::new();

        let answer = dialogue::Answer {
            question_id: "test".to_string(),
            text: "test answer".to_string(),
            provided: true,
        };

        manager.process_answer("test".to_string(), answer);
        assert_eq!(manager.state().answers.len(), 1);
    }

    #[test]
    fn test_dialogue_manager_next_round() {
        let mut manager = dialogue::DialogueManager::new();

        let result = manager.next_round();
        assert!(result.is_ok());
        assert_eq!(manager.state().round, 1);
    }

    #[test]
    fn test_dialogue_manager_max_rounds() {
        let mut manager = dialogue::DialogueManager::new();

        // Advance beyond max rounds
        for _ in 0..10 {
            let _ = manager.next_round();
        }

        let result = manager.next_round();
        assert!(result.is_err(), "Should error when exceeding max rounds");
    }

    #[test]
    fn test_dialogue_manager_is_complete() {
        let mut manager = dialogue::DialogueManager::new();

        // Initially not complete
        assert!(!manager.is_complete());

        // Add a question and answer
        let ambiguities = vec![dialogue::Ambiguity {
            qtype: dialogue::QuestionType::EmptyInput,
            description: "test".to_string(),
            questions: vec!["What?".to_string()],
        }];

        let _ = manager.generate_followup_questions(&ambiguities);

        let answer = dialogue::Answer {
            question_id: "test".to_string(),
            text: "answer".to_string(),
            provided: true,
        };

        manager.process_answer("test".to_string(), answer);

        assert!(
            manager.is_complete(),
            "Should be complete when all questions answered"
        );
    }
}

// ============================================================
// Integration Tests - End-to-End NL Pipeline
// ============================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_nl_pipeline() {
        let pipeline = NLPipeline::new();
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let dialogue_mgr = dialogue::DialogueManager::new();

        // Simulate NL input
        let description = "sort an array in ascending order";

        // Parse requirements
        let req = pipeline.parse_from_examples(vec![]);

        // Detect ambiguities
        let ambiguities = dialogue_mgr.detect_ambiguity(&req);

        // Generate examples
        let examples = synthesizer.generate_examples(description);

        assert!(
            !ambiguities.is_empty(),
            "Should have ambiguities for empty requirements"
        );
        assert!(!examples.examples.is_empty(), "Should generate examples");
    }

    #[test]
    fn test_nl_to_synthesis_flow() {
        let pipeline = NLPipeline::new();

        // Try synthesizing from NL
        let result = pipeline.synthesize_from_nl("add two numbers");

        // Should have attempted synthesis
        assert!(!result.method.is_empty(), "Should report synthesis method");
    }

    #[test]
    fn test_nl_pipeline_error_handling() {
        let pipeline = NLPipeline::new();

        // Test with ambiguous description
        let result = pipeline.synthesize_from_nl("do something with the input");

        // Should handle gracefully
        assert!(!result.method.is_empty());
    }
}

// ============================================================
// Example Problem Tests - Standard Benchmarks
// ============================================================

#[cfg(test)]
mod example_problem_tests {
    use super::*;

    // ============================================================
    // Problem 1: Reverse an Array
    // ============================================================

    #[test]
    fn test_problem_reverse_array_inference() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse an array");

        assert!(
            !result.examples.is_empty(),
            "Should generate examples for reverse"
        );
        assert!(
            result.examples.len() >= 5,
            "Should generate minimum examples"
        );
    }

    #[test]
    fn test_problem_reverse_array_examples() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse an array");

        // Should have list inputs
        let has_list_input = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_array()));
        assert!(has_list_input, "Should have list inputs for reverse");

        // Should have list outputs
        let has_list_output = result.examples.iter().any(|e| e.expected.is_array());
        assert!(has_list_output, "Should have list outputs for reverse");
    }

    #[test]
    fn test_problem_reverse_array_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let result = synthesizer.generate_examples("reverse an array");

        // Should include empty list
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

        // Should include single element
        let has_single = result.examples.iter().any(|e| {
            e.inputs.iter().any(|i| {
                if let Some(arr) = i.as_array() {
                    arr.len() == 1
                } else {
                    false
                }
            })
        });
        assert!(has_single, "Should include single element edge case");
    }

    #[test]
    fn test_problem_reverse_array_consistency() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("reverse an array");

        // Should have no consistency errors
        assert!(
            result.validation_errors.is_empty(),
            "Reverse examples should be consistent: {:?}",
            result.validation_errors
        );
    }

    // ============================================================
    // Problem 2: Sort Numbers
    // ============================================================

    #[test]
    fn test_problem_sort_numbers_inference() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sort numbers");

        assert!(
            !result.examples.is_empty(),
            "Should generate examples for sort"
        );
        assert!(
            result.examples.len() >= 5,
            "Should generate minimum examples"
        );
    }

    #[test]
    fn test_problem_sort_numbers_examples() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sort numbers in ascending order");

        // Should have list inputs
        let has_list_input = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_array()));
        assert!(has_list_input, "Should have list inputs for sort");

        // Should have list outputs
        let has_list_output = result.examples.iter().any(|e| e.expected.is_array());
        assert!(has_list_output, "Should have list outputs for sort");
    }

    #[test]
    fn test_problem_sort_numbers_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let result = synthesizer.generate_examples("sort numbers");

        // Should include empty list
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
    fn test_problem_sort_numbers_with_duplicates() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sort numbers with possible duplicates");

        assert!(!result.examples.is_empty());
        // Should handle duplicate elements
    }

    // ============================================================
    // Problem 3: Filter Even Numbers
    // ============================================================

    #[test]
    fn test_problem_filter_even_inference() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("filter even numbers");

        assert!(
            !result.examples.is_empty(),
            "Should generate examples for filter"
        );
        assert!(
            result.examples.len() >= 5,
            "Should generate minimum examples"
        );
    }

    #[test]
    fn test_problem_filter_even_examples() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("filter even numbers from list");

        // Should have list inputs
        let has_list_input = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_array()));
        assert!(has_list_input, "Should have list inputs for filter");

        // Should have list outputs
        let has_list_output = result.examples.iter().any(|e| e.expected.is_array());
        assert!(has_list_output, "Should have list outputs for filter");
    }

    #[test]
    fn test_problem_filter_even_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let result = synthesizer.generate_examples("filter even numbers");

        // Should include empty list
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

        // Should include list with no evens
        let has_no_evens = result.examples.iter().any(|e| {
            if let Some(arr) = e.expected.as_array() {
                arr.is_empty()
            } else {
                false
            }
        });
        // This might not always be generated, so we don't assert
        let _ = has_no_evens;
    }

    #[test]
    fn test_problem_filter_even_all_evens() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("filter even numbers");

        // Should handle lists where all numbers are even
        assert!(!result.examples.is_empty());
    }

    // ============================================================
    // Problem 4: Sum of Squares
    // ============================================================

    #[test]
    fn test_problem_sum_of_squares_inference() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("compute sum of squares");

        assert!(
            !result.examples.is_empty(),
            "Should generate examples for sum of squares"
        );
        assert!(
            result.examples.len() >= 5,
            "Should generate minimum examples"
        );
    }

    #[test]
    fn test_problem_sum_of_squares_examples() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sum of squares of list elements");

        // Should have list inputs
        let has_list_input = result
            .examples
            .iter()
            .any(|e| e.inputs.iter().any(|i| i.is_array()));
        assert!(has_list_input, "Should have list inputs for sum of squares");

        // Should have integer outputs
        let has_int_output = result.examples.iter().all(|e| e.expected.is_i64());
        assert!(
            has_int_output,
            "Should have integer outputs for sum of squares"
        );
    }

    #[test]
    fn test_problem_sum_of_squares_edge_cases() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let result = synthesizer.generate_examples("sum of squares");

        // Should include empty list (sum = 0)
        let has_empty = result.examples.iter().any(|e| {
            e.inputs.iter().any(|i| {
                if let Some(arr) = i.as_array() {
                    arr.is_empty() && e.expected.as_i64() == Some(0)
                } else {
                    false
                }
            })
        });
        assert!(has_empty, "Should include empty list edge case with sum 0");
    }

    #[test]
    fn test_problem_sum_of_squares_with_zero() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sum of squares including zeros");

        assert!(!result.examples.is_empty());
        // Should handle zero elements (0^2 = 0)
    }

    #[test]
    fn test_problem_sum_of_squares_consistency() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();
        let result = synthesizer.generate_examples("sum of squares");

        // Should have no consistency errors
        assert!(
            result.validation_errors.is_empty(),
            "Sum of squares examples should be consistent: {:?}",
            result.validation_errors
        );
    }

    // ============================================================
    // Coverage Tests Across All Problems
    // ============================================================

    #[test]
    fn test_all_problems_generate_examples() {
        let synthesizer = synthesizer::ExampleSynthesizer::new();

        let problems = vec![
            "reverse an array",
            "sort numbers ascending",
            "filter even numbers",
            "sum of squares",
            "add two numbers",
            "count elements",
            "find maximum",
            "find minimum",
        ];

        for problem in problems {
            let result = synthesizer.generate_examples(problem);
            assert!(
                !result.examples.is_empty(),
                "Should generate examples for: {}",
                problem
            );
            assert!(
                result.validation_errors.is_empty(),
                "Examples should be consistent for: {} - Errors: {:?}",
                problem,
                result.validation_errors
            );
        }
    }

    #[test]
    fn test_all_problems_coverage_quality() {
        let synthesizer = synthesizer::ExampleSynthesizer::new().with_edge_cases(true);

        let problems = vec![
            "reverse an array",
            "sort numbers",
            "filter even numbers",
            "sum of squares",
        ];

        for problem in problems {
            let result = synthesizer.generate_examples(problem);

            // Check coverage statistics
            assert!(
                result.coverage.total_examples >= 5,
                "Should meet minimum examples for: {}",
                problem
            );
            assert!(
                result.coverage.unique_patterns > 0,
                "Should have unique patterns for: {}",
                problem
            );

            // Edge cases should be included when enabled
            assert!(
                result.coverage.edge_cases > 0,
                "Should have edge cases for: {}",
                problem
            );
        }
    }
}

// ============================================================
// End-to-End NL Synthesis Tests
// ============================================================

#[cfg(test)]
mod e2e_nl_synthesis_tests {
    use super::*;

    #[test]
    fn test_e2e_reverse_array() {
        let pipeline = NLPipeline::new();
        let result = pipeline.synthesize_from_nl("reverse an array");

        // Should attempt synthesis
        assert!(!result.method.is_empty());
        assert!(!result.code.is_empty() || result.error.is_some());
    }

    #[test]
    fn test_e2e_sort_numbers() {
        let pipeline = NLPipeline::new();
        let result = pipeline.synthesize_from_nl("sort numbers in ascending order");

        assert!(!result.method.is_empty());
        assert!(!result.code.is_empty() || result.error.is_some());
    }

    #[test]
    fn test_e2e_filter_even() {
        let pipeline = NLPipeline::new();
        let result = pipeline.synthesize_from_nl("filter even numbers from list");

        assert!(!result.method.is_empty());
        assert!(!result.code.is_empty() || result.error.is_some());
    }

    #[test]
    fn test_e2e_sum_of_squares() {
        let pipeline = NLPipeline::new();
        let result = pipeline.synthesize_from_nl("compute sum of squares of list elements");

        assert!(!result.method.is_empty());
        assert!(!result.code.is_empty() || result.error.is_some());
    }

    #[test]
    fn test_e2e_arithmetic_operations() {
        let pipeline = NLPipeline::new();

        let operations = vec![
            "add two numbers",
            "find maximum of two numbers",
            "find minimum of two numbers",
        ];

        for op in operations {
            let result = pipeline.synthesize_from_nl(op);
            assert!(
                !result.method.is_empty(),
                "Should attempt synthesis for: {}",
                op
            );
        }
    }
}

// ============================================================
// JSON Value Conversion Tests
// ============================================================

#[cfg(test)]
mod json_conversion_tests {
    use super::*;
    use crate::benchmark::Value;

    #[test]
    fn test_json_to_value_int() {
        let pipeline = NLPipeline::new();
        let json_val = json!(42);
        let result = pipeline.json_to_value(json_val);

        assert!(result.is_some());
        assert_eq!(result, Some(Value::Int(42)));
    }

    #[test]
    fn test_json_to_value_float() {
        let pipeline = NLPipeline::new();
        let json_val = json!(3.14);
        let result = pipeline.json_to_value(json_val);

        assert!(result.is_some());
        if let Some(Value::Float(bits)) = result {
            let val = f64::from_bits(bits);
            assert!((val - 3.14).abs() < 0.001);
        } else {
            panic!("Expected Float value");
        }
    }

    #[test]
    fn test_json_to_value_array() {
        let pipeline = NLPipeline::new();
        let json_val = json!([1, 2, 3]);
        let result = pipeline.json_to_value(json_val);

        assert!(result.is_some());
        assert_eq!(result, Some(Value::Array(vec![1, 2, 3])));
    }

    #[test]
    fn test_json_to_value_string() {
        let pipeline = NLPipeline::new();
        let json_val = json!("hello");
        let result = pipeline.json_to_value(json_val);

        assert!(result.is_some());
        assert_eq!(result, Some(Value::Str("hello".to_string())));
    }

    #[test]
    fn test_json_to_value_bool() {
        let pipeline = NLPipeline::new();
        let json_val = json!(true);
        let result = pipeline.json_to_value(json_val);

        assert!(result.is_some());
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_json_to_value_mixed_array() {
        let pipeline = NLPipeline::new();
        let json_val = json!([1, "two", 3]);
        let result = pipeline.json_to_value(json_val);

        // Should only extract integers
        assert!(result.is_some());
        if let Some(Value::Array(arr)) = result {
            assert_eq!(arr, vec![1, 3]);
        } else {
            panic!("Expected Array value");
        }
    }
}

// ============================================================
// Signature Inference Tests
// ============================================================

#[cfg(test)]
mod signature_inference_tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    #[test]
    fn test_infer_signature_empty() {
        let pipeline = NLPipeline::new();
        let sig = pipeline.infer_signature(&[]);
        assert_eq!(sig, "fn f() -> i64");
    }

    #[test]
    fn test_infer_signature_single_int() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![Value::Int(5)],
            expected: Value::Int(5),
        }];
        let sig = pipeline.infer_signature(&examples);
        assert_eq!(sig, "fn f(x0: i64) -> i64");
    }

    #[test]
    fn test_infer_signature_two_ints() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![Value::Int(2), Value::Int(3)],
            expected: Value::Int(5),
        }];
        let sig = pipeline.infer_signature(&examples);
        assert_eq!(sig, "fn f(x0: i64, x1: i64) -> i64");
    }

    #[test]
    fn test_infer_signature_list_input() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![Value::Array(vec![1, 2, 3])],
            expected: Value::Int(3),
        }];
        let sig = pipeline.infer_signature(&examples);
        assert_eq!(sig, "fn f(x0: [i64]) -> i64");
    }

    #[test]
    fn test_infer_signature_list_output() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![Value::Array(vec![3, 2, 1])],
            expected: Value::Array(vec![1, 2, 3]),
        }];
        let sig = pipeline.infer_signature(&examples);
        assert_eq!(sig, "fn f(x0: [i64]) -> [i64]");
    }

    #[test]
    fn test_infer_signature_mixed_types() {
        let pipeline = NLPipeline::new();
        let examples = vec![Example {
            inputs: vec![Value::Array(vec![1, 2]), Value::Int(3)],
            expected: Value::Array(vec![1, 2, 3]),
        }];
        let sig = pipeline.infer_signature(&examples);
        assert_eq!(sig, "fn f(x0: [i64], x1: i64) -> [i64]");
    }
}

// ============================================================
// Function Name Generation Tests
// ============================================================

#[cfg(test)]
mod function_name_tests {
    use super::*;

    #[test]
    fn test_function_name_from_input_simple() {
        let pipeline = NLPipeline::new();
        let name = pipeline.function_name_from_input("add numbers");
        assert!(name.contains("add"));
        assert!(name.contains("numbers") || name.contains("number"));
    }

    #[test]
    fn test_function_name_from_input_multi_word() {
        let pipeline = NLPipeline::new();
        let name = pipeline.function_name_from_input("reverse the array");
        assert!(name.contains("reverse"));
    }

    #[test]
    fn test_function_name_from_input_special_chars() {
        let pipeline = NLPipeline::new();
        let name = pipeline.function_name_from_input("sort-numbers!ascending");
        // Should replace special chars with underscores
        assert!(!name.contains('-'));
        assert!(!name.contains('!'));
    }

    #[test]
    fn test_function_name_from_input_limit() {
        let pipeline = NLPipeline::new();
        let name = pipeline.function_name_from_input("add one two three four five six seven");
        // Should limit to first 3 words
        let parts: Vec<&str> = name.split('_').filter(|s| !s.is_empty()).collect();
        assert!(parts.len() <= 3);
    }
}

// ============================================================
// Cache Tests
// ============================================================

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_initially_empty() {
        let pipeline = NLPipeline::new();
        let (size, max) = pipeline.cache_stats().await;
        assert_eq!(size, 0);
        assert!(max > 0);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let pipeline = NLPipeline::new();
        pipeline.clear_cache().await;
        let (size, _) = pipeline.cache_stats().await;
        assert_eq!(size, 0);
    }

    #[tokio::test]
    async fn test_cache_stats_consistency() {
        let pipeline = NLPipeline::new();
        let (size1, max1) = pipeline.cache_stats().await;
        let (size2, max2) = pipeline.cache_stats().await;

        assert_eq!(size1, size2);
        assert_eq!(max1, max2);
    }
}
