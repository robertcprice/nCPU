//! Coverage analysis for test suites
//!
//! Provides coverage tracking and analysis for generated test suites,
//! including path coverage, edge case coverage, and boundary coverage.

use crate::benchmark::Problem;
use crate::testing::{CodePath, PathCondition, TestCase, TestCategory, TestSuite};
use std::collections::HashSet;

/// Coverage metrics for a test suite
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    pub path_coverage: f64,
    pub edge_coverage: f64,
    pub boundary_coverage: f64,
    pub overall_coverage: f64,
    pub covered_paths: HashSet<String>,
    pub total_paths: usize,
}

impl CoverageMetrics {
    /// Create new coverage metrics
    pub fn new() -> Self {
        Self {
            path_coverage: 0.0,
            edge_coverage: 0.0,
            boundary_coverage: 0.0,
            overall_coverage: 0.0,
            covered_paths: HashSet::new(),
            total_paths: 0,
        }
    }

    /// Calculate overall coverage from individual metrics
    pub fn calculate_overall(&mut self) {
        self.overall_coverage =
            (self.path_coverage + self.edge_coverage + self.boundary_coverage) / 3.0;
    }
}

/// Coverage analyzer for test suites
pub struct CoverageAnalyzer {
    /// Identified code paths for the problem
    code_paths: Vec<CodePath>,
    /// Coverage target (0.0 to 1.0)
    target: f64,
}

impl CoverageAnalyzer {
    /// Create a new coverage analyzer
    pub fn new(target: f64) -> Self {
        Self {
            code_paths: Vec::new(),
            target,
        }
    }

    /// Analyze coverage of a test suite against a problem
    pub fn analyze_suite(&self, suite: &TestSuite, problem: &Problem) -> CoverageMetrics {
        let mut metrics = CoverageMetrics::new();

        // Identify code paths from the problem's reference code
        let paths = self.identify_paths(problem);
        metrics.total_paths = paths.len();

        // Track which paths are covered
        for test in &suite.tests {
            if let Some(path_id) = self.check_path_coverage(&test, &paths) {
                metrics.covered_paths.insert(path_id);
            }
        }

        // Calculate path coverage
        metrics.path_coverage = if metrics.total_paths > 0 {
            metrics.covered_paths.len() as f64 / metrics.total_paths as f64
        } else {
            0.0
        };

        // Calculate edge coverage
        let edge_tests = suite.tests_by_category(TestCategory::EdgeCase);
        let required_edges = self.count_required_edges(problem);
        metrics.edge_coverage = if required_edges > 0 {
            (edge_tests.len() as f64 / required_edges as f64).min(1.0)
        } else {
            0.0
        };

        // Calculate boundary coverage
        let boundary_tests = suite.tests_by_category(TestCategory::Boundary);
        let required_boundaries = self.count_required_boundaries(problem);
        metrics.boundary_coverage = if required_boundaries > 0 {
            (boundary_tests.len() as f64 / required_boundaries as f64).min(1.0)
        } else {
            0.0
        };

        metrics.calculate_overall();
        metrics
    }

    /// Identify code paths from problem signature and reference code
    fn identify_paths(&self, problem: &Problem) -> Vec<CodePath> {
        let mut paths = Vec::new();

        // Analyze signature for type-based paths
        let sig = problem.signature;

        // Integer operations paths
        if sig.contains("i64") || sig.contains("i32") {
            paths.extend(self.integer_paths());
        }

        // Array operations paths
        if sig.contains("[i64]") || sig.contains("[") {
            paths.extend(self.array_paths());
        }

        // String operations paths
        if sig.contains("string") {
            paths.extend(self.string_paths());
        }

        // Control flow paths from reference code
        paths.extend(self.control_flow_paths(problem.reference_code));

        paths
    }

    /// Integer operation code paths
    fn integer_paths(&self) -> Vec<CodePath> {
        vec![
            CodePath {
                id: "int_zero".to_string(),
                description: "Input is zero".to_string(),
                conditions: vec![PathCondition::Equals("x".to_string(), 0)],
            },
            CodePath {
                id: "int_positive".to_string(),
                description: "Input is positive".to_string(),
                conditions: vec![PathCondition::GreaterThan("x".to_string(), 0)],
            },
            CodePath {
                id: "int_negative".to_string(),
                description: "Input is negative".to_string(),
                conditions: vec![PathCondition::LessThan("x".to_string(), 0)],
            },
            CodePath {
                id: "int_overflow_pos".to_string(),
                description: "Positive boundary (near overflow)".to_string(),
                conditions: vec![PathCondition::Range(
                    "x".to_string(),
                    i64::MAX - 10,
                    i64::MAX,
                )],
            },
            CodePath {
                id: "int_overflow_neg".to_string(),
                description: "Negative boundary (near overflow)".to_string(),
                conditions: vec![PathCondition::Range(
                    "x".to_string(),
                    i64::MIN,
                    i64::MIN + 10,
                )],
            },
        ]
    }

    /// Array operation code paths
    fn array_paths(&self) -> Vec<CodePath> {
        vec![
            CodePath {
                id: "array_empty".to_string(),
                description: "Empty array".to_string(),
                conditions: vec![PathCondition::ArrayLen(0)],
            },
            CodePath {
                id: "array_single".to_string(),
                description: "Single element array".to_string(),
                conditions: vec![PathCondition::ArrayLen(1)],
            },
            CodePath {
                id: "array_two".to_string(),
                description: "Two element array".to_string(),
                conditions: vec![PathCondition::ArrayLen(2)],
            },
            CodePath {
                id: "array_large".to_string(),
                description: "Large array (>10 elements)".to_string(),
                conditions: vec![PathCondition::Custom("array.len > 10".to_string())],
            },
        ]
    }

    /// String operation code paths
    fn string_paths(&self) -> Vec<CodePath> {
        vec![
            CodePath {
                id: "string_empty".to_string(),
                description: "Empty string".to_string(),
                conditions: vec![PathCondition::Custom("s.is_empty()".to_string())],
            },
            CodePath {
                id: "string_single_char".to_string(),
                description: "Single character string".to_string(),
                conditions: vec![PathCondition::Custom("s.len() == 1".to_string())],
            },
            CodePath {
                id: "string_whitespace".to_string(),
                description: "Whitespace-only string".to_string(),
                conditions: vec![PathCondition::Custom("s.trim().is_empty()".to_string())],
            },
            CodePath {
                id: "string_with_spaces".to_string(),
                description: "String with leading/trailing spaces".to_string(),
                conditions: vec![PathCondition::Custom(
                    "s.starts_with(' ') || s.ends_with(' ')".to_string(),
                )],
            },
        ]
    }

    /// Extract control flow paths from reference code
    fn control_flow_paths(&self, code: &str) -> Vec<CodePath> {
        let mut paths = Vec::new();
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let line = line.trim();

            // Detect if statements
            if line.starts_with("if ") {
                let condition = line
                    .strip_prefix("if ")
                    .and_then(|s| s.split('{').next())
                    .unwrap_or("")
                    .trim();

                if !condition.is_empty() {
                    paths.push(CodePath {
                        id: format!("if_branch_{}", i),
                        description: format!("If condition: {}", condition),
                        conditions: vec![PathCondition::Custom(condition.to_string())],
                    });
                }
            }

            // Detect while loops
            if line.starts_with("while ") {
                let condition = line
                    .strip_prefix("while ")
                    .and_then(|s| s.split('{').next())
                    .unwrap_or("")
                    .trim();

                if !condition.is_empty() {
                    paths.push(CodePath {
                        id: format!("while_{}", i),
                        description: format!("While loop condition: {}", condition),
                        conditions: vec![PathCondition::Custom(condition.to_string())],
                    });
                }
            }

            // Detect match statements
            if line.contains("match ") {
                paths.push(CodePath {
                    id: format!("match_{}", i),
                    description: "Match expression".to_string(),
                    conditions: vec![PathCondition::Custom("match".to_string())],
                });
            }
        }

        paths
    }

    /// Check if a test covers any code paths
    fn check_path_coverage(&self, test: &TestCase, paths: &[CodePath]) -> Option<String> {
        for path in paths {
            if self.test_matches_path(test, path) {
                return Some(path.id.clone());
            }
        }
        None
    }

    /// Check if a test matches a code path
    fn test_matches_path(&self, test: &TestCase, path: &CodePath) -> bool {
        for condition in &path.conditions {
            match condition {
                PathCondition::Equals(var, val) => {
                    if let Some(&crate::benchmark::Value::Int(v)) = test.inputs.first() {
                        if var == "x" && v == *val {
                            return true;
                        }
                    }
                }
                PathCondition::GreaterThan(var, val) => {
                    if let Some(&crate::benchmark::Value::Int(v)) = test.inputs.first() {
                        if var == "x" && v > *val {
                            return true;
                        }
                    }
                }
                PathCondition::LessThan(var, val) => {
                    if let Some(&crate::benchmark::Value::Int(v)) = test.inputs.first() {
                        if var == "x" && v < *val {
                            return true;
                        }
                    }
                }
                PathCondition::ArrayLen(len) => {
                    if let Some(crate::benchmark::Value::Array(arr)) = test.inputs.first() {
                        if arr.len() == *len {
                            return true;
                        }
                    }
                }
                PathCondition::Custom(cond) => {
                    // Check for common patterns in custom conditions
                    if cond.contains("== 0") && test.category == TestCategory::EdgeCase {
                        return true;
                    }
                    if cond.contains("is_empty()") {
                        if let Some(crate::benchmark::Value::Array(arr)) = test.inputs.first() {
                            if arr.is_empty() {
                                return true;
                            }
                        }
                        if let Some(crate::benchmark::Value::Str(s)) = test.inputs.first() {
                            if s.is_empty() {
                                return true;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Count required edge cases for a problem
    fn count_required_edges(&self, problem: &Problem) -> usize {
        let mut count = 0;
        let sig = problem.signature;

        // Zero edge case
        if sig.contains("i64") {
            count += 1;
        }

        // Empty array/string
        if sig.contains("[") || sig.contains("string") {
            count += 1;
        }

        // Null/None cases
        if sig.contains("Option") || sig.contains("Result") {
            count += 1;
        }

        count.max(1)
    }

    /// Count required boundary cases for a problem
    fn count_required_boundaries(&self, problem: &Problem) -> usize {
        let mut count = 0;
        let sig = problem.signature;

        // Numeric boundaries
        if sig.contains("i64") || sig.contains("i32") {
            count += 2; // positive and negative boundaries
        }

        // Array size boundaries
        if sig.contains("[") {
            count += 1;
        }

        count.max(1)
    }

    /// Generate coverage report
    pub fn coverage_report(&self, metrics: &CoverageMetrics) -> String {
        format!(
            "Coverage Report:\n\
             - Path Coverage: {:.1}%\n\
             - Edge Coverage: {:.1}%\n\
             - Boundary Coverage: {:.1}%\n\
             - Overall Coverage: {:.1}%\n\
             - Covered Paths: {}/{}\n\
             - Target: {:.1}%",
            metrics.path_coverage * 100.0,
            metrics.edge_coverage * 100.0,
            metrics.boundary_coverage * 100.0,
            metrics.overall_coverage * 100.0,
            metrics.covered_paths.len(),
            metrics.total_paths,
            self.target * 100.0
        )
    }

    /// Check if coverage meets target
    pub fn meets_target(&self, metrics: &CoverageMetrics) -> bool {
        metrics.overall_coverage >= self.target
    }

    /// Suggest additional tests to improve coverage
    pub fn suggest_improvements(
        &self,
        suite: &TestSuite,
        metrics: &CoverageMetrics,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Check edge coverage
        let edge_tests = suite.tests_by_category(TestCategory::EdgeCase);
        if edge_tests.is_empty() {
            suggestions.push("Add edge case tests (zero, empty, null values)".to_string());
        }

        // Check boundary coverage
        let boundary_tests = suite.tests_by_category(TestCategory::Boundary);
        if boundary_tests.is_empty() {
            suggestions.push("Add boundary tests (min/max values)".to_string());
        }

        // Check overall coverage
        if metrics.overall_coverage < self.target {
            suggestions.push(format!(
                "Increase coverage from {:.1}% to target {:.1}%",
                metrics.overall_coverage * 100.0,
                self.target * 100.0
            ));
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{array, int, string};

    #[test]
    fn test_coverage_metrics() {
        let mut metrics = CoverageMetrics::new();
        metrics.path_coverage = 0.8;
        metrics.edge_coverage = 0.6;
        metrics.boundary_coverage = 1.0;
        metrics.calculate_overall();

        assert!((metrics.overall_coverage - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = CoverageAnalyzer::new(0.9);
        assert_eq!(analyzer.target, 0.9);
    }

    #[test]
    fn test_integer_paths() {
        let analyzer = CoverageAnalyzer::new(0.8);
        let paths = analyzer.integer_paths();
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.id == "int_zero"));
    }

    #[test]
    fn test_array_paths() {
        let analyzer = CoverageAnalyzer::new(0.8);
        let paths = analyzer.array_paths();
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.id == "array_empty"));
    }

    #[test]
    fn test_string_paths() {
        let analyzer = CoverageAnalyzer::new(0.8);
        let paths = analyzer.string_paths();
        assert!(!paths.is_empty());
        assert!(paths.iter().any(|p| p.id == "string_empty"));
    }
}
