//! Validation pipeline for synthesized programs
//!
//! Multi-stage validation system that checks synthesized code for correctness,
//! security issues, performance problems, and style consistency.

pub mod pipeline;
pub mod stages;

use std::collections::HashMap;

/// Result of a validation operation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed (no critical/high issues)
    pub passed: bool,
    /// All issues found during validation
    pub issues: Vec<Issue>,
    /// Warnings that don't fail validation
    pub warnings: Vec<Warning>,
    /// Overall quality score (0.0 to 1.0)
    pub score: f64,
    /// Per-category issue counts
    pub category_counts: HashMap<IssueCategory, usize>,
}

impl ValidationResult {
    /// Create a new successful validation result
    pub fn success() -> Self {
        Self {
            passed: true,
            issues: Vec::new(),
            warnings: Vec::new(),
            score: 1.0,
            category_counts: HashMap::new(),
        }
    }

    /// Create a failed validation result with issues
    pub fn failed(issues: Vec<Issue>) -> Self {
        let passed = issues.iter().all(|i| i.severity != Severity::Critical && i.severity != Severity::High);
        let score = Self::calculate_score(&issues, &[]);
        let category_counts = Self::count_categories(&issues);

        Self {
            passed,
            issues,
            warnings: Vec::new(),
            score,
            category_counts,
        }
    }

    /// Add a warning to this result
    pub fn with_warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Merge multiple validation results
    pub fn merge(results: impl IntoIterator<Item = ValidationResult>) -> Self {
        let mut all_issues = Vec::new();
        let mut all_warnings = Vec::new();
        let mut all_counts = HashMap::new();

        for result in results {
            all_issues.extend(result.issues);
            all_warnings.extend(result.warnings);
            for (cat, count) in result.category_counts {
                *all_counts.entry(cat).or_insert(0) += count;
            }
        }

        let passed = all_issues.iter().all(|i| i.severity != Severity::Critical && i.severity != Severity::High);
        let score = Self::calculate_score(&all_issues, &all_warnings);

        Self {
            passed,
            issues: all_issues,
            warnings: all_warnings,
            score,
            category_counts: all_counts,
        }
    }

    /// Calculate quality score from issues and warnings
    fn calculate_score(issues: &[Issue], warnings: &[Warning]) -> f64 {
        let critical = issues.iter().filter(|i| i.severity == Severity::Critical).count();
        let high = issues.iter().filter(|i| i.severity == Severity::High).count();
        let medium = issues.iter().filter(|i| i.severity == Severity::Medium).count();
        let low = issues.iter().filter(|i| i.severity == Severity::Low).count();
        let info = issues.iter().filter(|i| i.severity == Severity::Info).count();
        let warnings_count = warnings.len();

        let penalty = (critical * 100) + (high * 50) + (medium * 10) + (low * 5) + (info * 1) + (warnings_count * 2);
        let max_score = 100.0;

        ((max_score - penalty as f64) / max_score).max(0.0)
    }

    /// Count issues by category
    fn count_categories(issues: &[Issue]) -> HashMap<IssueCategory, usize> {
        let mut counts = HashMap::new();
        for issue in issues {
            *counts.entry(issue.category.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get summary message
    pub fn summary(&self) -> String {
        let critical = self.issues.iter().filter(|i| i.severity == Severity::Critical).count();
        let high = self.issues.iter().filter(|i| i.severity == Severity::High).count();
        let medium = self.issues.iter().filter(|i| i.severity == Severity::Medium).count();

        format!(
            "Validation: {} - Score: {:.2}% - Issues: {} critical, {} high, {} medium, {} warnings",
            if self.passed { "PASSED" } else { "FAILED" },
            self.score * 100.0,
            critical,
            high,
            medium,
            self.warnings.len()
        )
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::success()
    }
}

/// An issue found during validation
#[derive(Debug, Clone)]
pub struct Issue {
    /// Severity level of the issue
    pub severity: Severity,
    /// Category of the issue
    pub category: IssueCategory,
    /// Human-readable description
    pub message: String,
    /// Source location (if available)
    pub location: Option<Location>,
    /// Suggested fix (if available)
    pub suggested_fix: Option<String>,
}

impl Issue {
    /// Create a new issue
    pub fn new(
        severity: Severity,
        category: IssueCategory,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            category,
            message: message.into(),
            location: None,
            suggested_fix: None,
        }
    }

    /// Add a location to this issue
    pub fn with_location(mut self, file: impl Into<String>, line: usize, column: usize) -> Self {
        self.location = Some(Location {
            file: file.into(),
            line,
            column,
        });
        self
    }

    /// Add a suggested fix
    pub fn with_fix(mut self, fix: impl Into<String>) -> Self {
        self.suggested_fix = Some(fix.into());
        self
    }

    /// Create a critical issue
    pub fn critical(category: IssueCategory, message: impl Into<String>) -> Self {
        Self::new(Severity::Critical, category, message)
    }

    /// Create a high severity issue
    pub fn high(category: IssueCategory, message: impl Into<String>) -> Self {
        Self::new(Severity::High, category, message)
    }

    /// Create a medium severity issue
    pub fn medium(category: IssueCategory, message: impl Into<String>) -> Self {
        Self::new(Severity::Medium, category, message)
    }

    /// Create a low severity issue
    pub fn low(category: IssueCategory, message: impl Into<String>) -> Self {
        Self::new(Severity::Low, category, message)
    }
}

/// Severity level of an issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Critical - must fix before using
    Critical = 5,
    /// High - should fix
    High = 4,
    /// Medium - consider fixing
    Medium = 3,
    /// Low - minor issue
    Low = 2,
    /// Info - informational only
    Info = 1,
}

impl Severity {
    /// Get the display name
    pub fn name(&self) -> &str {
        match self {
            Severity::Critical => "critical",
            Severity::High => "high",
            Severity::Medium => "medium",
            Severity::Low => "low",
            Severity::Info => "info",
        }
    }
}

/// Category of validation issue
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IssueCategory {
    /// Syntax errors or invalid constructs
    Syntax,
    /// Type mismatches or invalid types
    Type,
    /// Security vulnerabilities
    Security,
    /// Performance issues
    Performance,
    /// Logic/correctness problems
    Correctness,
    /// Style and formatting
    Style,
}

impl IssueCategory {
    /// Get the display name
    pub fn name(&self) -> &str {
        match self {
            IssueCategory::Syntax => "syntax",
            IssueCategory::Type => "type",
            IssueCategory::Security => "security",
            IssueCategory::Performance => "performance",
            IssueCategory::Correctness => "correctness",
            IssueCategory::Style => "style",
        }
    }
}

/// Source location in code
#[derive(Debug, Clone)]
pub struct Location {
    /// File path or identifier
    pub file: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
}

impl Location {
    /// Create a new location
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }

    /// Format location for display
    pub fn display(&self) -> String {
        format!("{}:{}:{}", self.file, self.line, self.column)
    }
}

/// A warning that doesn't fail validation
#[derive(Debug, Clone)]
pub struct Warning {
    /// Warning message
    pub message: String,
    /// Source location (if available)
    pub location: Option<Location>,
}

impl Warning {
    /// Create a new warning
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            location: None,
        }
    }

    /// Add a location to this warning
    pub fn with_location(mut self, file: impl Into<String>, line: usize, column: usize) -> Self {
        self.location = Some(Location::new(file, line, column));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success();
        assert!(result.passed);
        assert_eq!(result.score, 1.0);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_validation_result_failed() {
        let issues = vec![
            Issue::high(IssueCategory::Type, "Type mismatch"),
            Issue::medium(IssueCategory::Style, "Inconsistent naming"),
        ];
        let result = ValidationResult::failed(issues);

        assert!(!result.passed);
        assert!(result.score < 1.0);
        assert_eq!(result.issues.len(), 2);
    }

    #[test]
    fn test_score_calculation() {
        let issues = vec![
            Issue::critical(IssueCategory::Security, "Critical issue"),
            Issue::high(IssueCategory::Performance, "High issue"),
        ];
        let result = ValidationResult::failed(issues);

        // 150 penalty out of 100 = 0.0
        assert_eq!(result.score, 0.0);
    }

    #[test]
    fn test_issue_builder() {
        let issue = Issue::critical(IssueCategory::Syntax, "Missing semicolon")
            .with_location("test.rs", 42, 10)
            .with_fix("Add semicolon");

        assert_eq!(issue.severity, Severity::Critical);
        assert_eq!(issue.category, IssueCategory::Syntax);
        assert!(issue.location.is_some());
        assert!(issue.suggested_fix.is_some());
    }

    #[test]
    fn test_merge_results() {
        let result1 = ValidationResult::failed(vec![
            Issue::medium(IssueCategory::Style, "Style issue 1"),
        ]);
        let result2 = ValidationResult::failed(vec![
            Issue::medium(IssueCategory::Style, "Style issue 2"),
        ]);

        let merged = ValidationResult::merge(vec![result1, result2]);

        assert_eq!(merged.issues.len(), 2);
        assert_eq!(merged.score, 0.8); // 20 penalty
    }
}
