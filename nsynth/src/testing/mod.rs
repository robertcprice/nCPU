//! Test generation and coverage analysis for nSynth
//!
//! This module provides automated test generation capabilities including:
//! - Normal case generation from function signatures
//! - Edge case detection (empty, null, boundaries)
//! - Property-based test generation
//! - Fuzzing input generation
//! - Coverage tracking and reporting

pub mod generation;
pub mod coverage;

use crate::benchmark::{Value, Example, TreeNode};
use std::collections::HashMap;

/// A single test case with metadata
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub inputs: Vec<Value>,
    pub expected: Value,
    pub description: String,
    pub category: TestCategory,
}

/// Categories of test cases for coverage tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TestCategory {
    /// Normal/typical inputs
    Normal,
    /// Edge cases (empty, single element, null-like)
    EdgeCase,
    /// Boundary values (max/min, overflow boundaries)
    Boundary,
    /// Fuzzing-generated inputs
    Fuzzing,
    /// Property-based tests (invariants)
    Property,
}

/// A test suite for a single function or problem
#[derive(Debug, Clone)]
pub struct TestSuite {
    pub name: String,
    pub tests: Vec<TestCase>,
    pub coverage_target: f64,
}

impl TestSuite {
    /// Create a new test suite
    pub fn new(name: String, coverage_target: f64) -> Self {
        Self {
            name,
            tests: Vec::new(),
            coverage_target,
        }
    }

    /// Add a test case to the suite
    pub fn add_test(&mut self, test: TestCase) {
        self.tests.push(test);
    }

    /// Get the number of tests in each category
    pub fn category_counts(&self) -> HashMap<TestCategory, usize> {
        let mut counts = HashMap::new();
        for test in &self.tests {
            *counts.entry(test.category).or_insert(0) += 1;
        }
        counts
    }

    /// Get tests by category
    pub fn tests_by_category(&self, category: TestCategory) -> Vec<&TestCase> {
        self.tests.iter()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Convert test suite to examples for benchmark format
    pub fn to_examples(&self) -> Vec<Example> {
        self.tests.iter()
            .map(|t| Example {
                inputs: t.inputs.clone(),
                expected: t.expected.clone(),
            })
            .collect()
    }
}

/// Code path identified for coverage analysis
#[derive(Debug, Clone)]
pub struct CodePath {
    pub id: String,
    pub description: String,
    pub conditions: Vec<PathCondition>,
}

/// Conditions that activate a code path
#[derive(Debug, Clone)]
pub enum PathCondition {
    /// Equality condition (x == value)
    Equals(String, i64),
    /// Inequality condition (x != value)
    NotEquals(String, i64),
    /// Less than (x < value)
    LessThan(String, i64),
    /// Greater than (x > value)
    GreaterThan(String, i64),
    /// Range condition (min <= x <= max)
    Range(String, i64, i64),
    /// Array length condition
    ArrayLen(usize),
    /// String contains substring
    Contains(String),
    /// Custom condition
    Custom(String),
}

impl TestCase {
    /// Create a new test case
    pub fn new(
        name: String,
        inputs: Vec<Value>,
        expected: Value,
        description: String,
        category: TestCategory,
    ) -> Self {
        Self {
            name,
            inputs,
            expected,
            description,
            category,
        }
    }

    /// Create a normal test case
    pub fn normal(name: String, inputs: Vec<Value>, expected: Value, description: String) -> Self {
        Self::new(name, inputs, expected, description, TestCategory::Normal)
    }

    /// Create an edge case test
    pub fn edge(name: String, inputs: Vec<Value>, expected: Value, description: String) -> Self {
        Self::new(name, inputs, expected, description, TestCategory::EdgeCase)
    }

    /// Create a boundary test
    pub fn boundary(name: String, inputs: Vec<Value>, expected: Value, description: String) -> Self {
        Self::new(name, inputs, expected, description, TestCategory::Boundary)
    }

    /// Create a property test
    pub fn property(name: String, inputs: Vec<Value>, expected: Value, description: String) -> Self {
        Self::new(name, inputs, expected, description, TestCategory::Property)
    }

    /// Create a fuzzing test
    pub fn fuzzing(name: String, inputs: Vec<Value>, expected: Value, description: String) -> Self {
        Self::new(name, inputs, expected, description, TestCategory::Fuzzing)
    }
}

/// Helper to create integer value
pub fn int(v: i64) -> Value {
    Value::Int(v)
}

/// Helper to create string value
pub fn string(v: &str) -> Value {
    Value::Str(v.to_string())
}

/// Helper to create array value
pub fn array(v: &[i64]) -> Value {
    Value::Array(v.to_vec())
}

/// Helper to create pair value
pub fn pair(a: i64, b: i64) -> Value {
    Value::Pair(a, b)
}

/// Helper to create quad value
pub fn quad(a: i64, b: i64, c: i64, d: i64) -> Value {
    Value::Quad(a, b, c, d)
}

/// Helper to create tree value
pub fn tree(nodes: Vec<TreeNode>) -> Value {
    Value::Tree(nodes)
}

/// Helper to create tree node
pub fn tree_node(value: i64, left: i32, right: i32) -> TreeNode {
    TreeNode { value, left, right }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_case_creation() {
        let test = TestCase::normal(
            "test_add".to_string(),
            vec![int(2), int(3)],
            int(5),
            "Basic addition test".to_string(),
        );
        assert_eq!(test.name, "test_add");
        assert_eq!(test.category, TestCategory::Normal);
    }

    #[test]
    fn test_test_suite() {
        let mut suite = TestSuite::new("add_tests".to_string(), 0.9);
        suite.add_test(TestCase::normal(
            "test1".to_string(),
            vec![int(1), int(2)],
            int(3),
            "Basic test".to_string(),
        ));
        suite.add_test(TestCase::edge(
            "test2".to_string(),
            vec![int(0), int(0)],
            int(0),
            "Zero edge case".to_string(),
        ));

        assert_eq!(suite.tests.len(), 2);
        let counts = suite.category_counts();
        assert_eq!(*counts.get(&TestCategory::Normal).unwrap(), 1);
        assert_eq!(*counts.get(&TestCategory::EdgeCase).unwrap(), 1);
    }

    #[test]
    fn test_to_examples() {
        let mut suite = TestSuite::new("test".to_string(), 0.8);
        suite.add_test(TestCase::normal(
            "t1".to_string(),
            vec![int(5)],
            int(25),
            "Square".to_string(),
        ));

        let examples = suite.to_examples();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].inputs.len(), 1);
        assert_eq!(examples[0].expected, int(25));
    }
}
