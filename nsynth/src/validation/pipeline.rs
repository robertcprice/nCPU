//! Validation pipeline orchestration
//!
//! Multi-stage validation pipeline that runs validation stages in sequence
//! and aggregates results.

use super::{stages::*, Issue, Severity, ValidationResult};
use crate::bidirectional::parser::AST;

/// Main validation pipeline
pub struct ValidationPipeline {
    stages: Vec<Box<dyn ValidationStage>>,
    stop_on_critical: bool,
    max_issues_per_stage: usize,
}

/// Trait for validation stages
pub trait ValidationStage {
    /// Get the stage name
    fn name(&self) -> &str;

    /// Validate the program and return results
    fn validate(&self, program: &AST) -> ValidationResult;

    /// Attempt to fix issues found during validation
    fn fix(&self, _program: &AST, _issues: Vec<Issue>) -> Option<AST> {
        None
    }

    /// Whether this stage can auto-fix issues
    fn can_fix(&self) -> bool {
        false
    }
}

impl ValidationPipeline {
    /// Create a new validation pipeline with default stages
    pub fn new() -> Self {
        Self {
            stages: vec![
                Box::new(SyntaxValidationStage::new()),
                Box::new(TypeValidationStage::new()),
                Box::new(SecurityValidationStage::new()),
                Box::new(PerformanceValidationStage::new()),
                Box::new(CorrectnessValidationStage::new()),
                Box::new(StyleValidationStage::new()),
            ],
            stop_on_critical: true,
            max_issues_per_stage: 100,
        }
    }

    /// Create a pipeline with custom stages
    pub fn with_stages(stages: Vec<Box<dyn ValidationStage>>) -> Self {
        Self {
            stages,
            stop_on_critical: true,
            max_issues_per_stage: 100,
        }
    }

    /// Set whether to stop on critical issues
    pub fn stop_on_critical(mut self, stop: bool) -> Self {
        self.stop_on_critical = stop;
        self
    }

    /// Set maximum issues per stage
    pub fn max_issues_per_stage(mut self, max: usize) -> Self {
        self.max_issues_per_stage = max;
        self
    }

    /// Validate a program through all stages
    pub fn validate(&self, program: &AST) -> ValidationResult {
        let mut all_issues = Vec::new();
        let mut all_warnings = Vec::new();
        let mut stage_results = Vec::new();
        let mut passed = true;

        for (_stage_idx, stage) in self.stages.iter().enumerate() {
            let result = stage.validate(program);
            stage_results.push((stage.name(), result.clone()));

            if !result.passed {
                passed = false;
            }

            // Limit issues per stage
            let issues = if result.issues.len() > self.max_issues_per_stage {
                let mut truncated = result.issues;
                truncate_mark(&mut truncated, self.max_issues_per_stage);
                truncated
            } else {
                result.issues
            };

            all_issues.extend(issues);
            all_warnings.extend(result.warnings);

            // Check for critical issues
            if all_issues.iter().any(|i| i.severity == Severity::Critical) {
                if self.stop_on_critical {
                    break;
                }
            }
        }

        let score = self.calculate_score(&all_issues, &all_warnings);
        let category_counts = ValidationResult::count_categories(&all_issues);

        ValidationResult {
            passed,
            issues: all_issues,
            warnings: all_warnings,
            score,
            category_counts,
        }
    }

    /// Attempt to auto-fix issues
    pub fn fix(&self, program: &AST) -> (ValidationResult, Option<AST>) {
        let initial_result = self.validate(program);

        if initial_result.passed {
            return (initial_result, Some(program.clone()));
        }

        let mut current_program = program.clone();
        let mut all_fixed = Vec::new();

        for stage in &self.stages {
            if !stage.can_fix() {
                continue;
            }

            let stage_result = stage.validate(&current_program);
            let fixable_issues: Vec<_> = stage_result
                .issues
                .into_iter()
                .filter(|i| i.suggested_fix.is_some())
                .collect();

            if !fixable_issues.is_empty() {
                if let Some(fixed_program) = stage.fix(&current_program, fixable_issues.clone()) {
                    current_program = fixed_program;
                    all_fixed.extend(fixable_issues);
                }
            }
        }

        let final_result = self.validate(&current_program);
        let fixed_count = all_fixed.len();

        // Add info about how many issues were fixed
        let final_result = if fixed_count > 0 {
            ValidationResult {
                warnings: vec![super::Warning::new(format!(
                    "Auto-fixed {} issues",
                    fixed_count
                ))],
                ..final_result
            }
        } else {
            final_result
        };

        (final_result, Some(current_program))
    }

    /// Get list of stage names
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name().to_string()).collect()
    }

    /// Calculate overall score from issues and warnings
    fn calculate_score(&self, issues: &[Issue], warnings: &[super::Warning]) -> f64 {
        ValidationResult::calculate_score(issues, warnings)
    }

    /// Run validation on multiple programs
    pub fn validate_batch(&self, programs: &[AST]) -> Vec<ValidationResult> {
        programs.iter().map(|p| self.validate(p)).collect()
    }

    /// Generate a validation report
    pub fn generate_report(&self, result: &ValidationResult) -> ValidationReport {
        let by_severity = {
            let mut map = std::collections::HashMap::new();
            for issue in &result.issues {
                *map.entry(issue.severity).or_insert(0) += 1;
            }
            map
        };

        let by_category = {
            let mut map = std::collections::HashMap::new();
            for issue in &result.issues {
                *map.entry(issue.category.clone()).or_insert(0) += 1;
            }
            map
        };

        ValidationReport {
            passed: result.passed,
            score: result.score,
            total_issues: result.issues.len(),
            total_warnings: result.warnings.len(),
            by_severity,
            by_category,
            stage_names: self.stage_names(),
        }
    }
}

/// Validation report summary
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub passed: bool,
    pub score: f64,
    pub total_issues: usize,
    pub total_warnings: usize,
    pub by_severity: std::collections::HashMap<Severity, usize>,
    pub by_category: std::collections::HashMap<super::IssueCategory, usize>,
    pub stage_names: Vec<String>,
}

impl ValidationReport {
    /// Format report as text
    pub fn to_text(&self) -> String {
        let mut output = String::new();
        output.push_str("╔════════════════════════════════════════════════════════╗\n");
        output.push_str("║           VALIDATION REPORT                          ║\n");
        output.push_str("╠════════════════════════════════════════════════════════╣\n");

        // Overall status
        let status = if self.passed {
            "✓ PASSED"
        } else {
            "✗ FAILED"
        };
        output.push_str(&format!("║ Status: {:<50} ║\n", status));
        output.push_str(&format!(
            "║ Score:  {:.1}% / 100.0%{:>38} ║\n",
            self.score * 100.0,
            ""
        ));

        output.push_str("╠════════════════════════════════════════════════════════╣\n");
        output.push_str("║ ISSUES BY SEVERITY                                    ║\n");
        output.push_str("╠────────────────────────────────────────────────────────╣\n");

        let severities = [
            (Severity::Critical, "Critical"),
            (Severity::High, "High"),
            (Severity::Medium, "Medium"),
            (Severity::Low, "Low"),
            (Severity::Info, "Info"),
        ];

        for (severity, name) in severities {
            let count = *self.by_severity.get(&severity).unwrap_or(&0);
            if count > 0 {
                output.push_str(&format!("║   {:<12}: {:>40} ║\n", name, count));
            }
        }

        if self.total_warnings > 0 {
            output.push_str("╠────────────────────────────────────────────────────────╣\n");
            output.push_str(&format!("║   Warnings:   {:>40} ║\n", self.total_warnings));
        }

        output.push_str("╚════════════════════════════════════════════════════════╝\n");

        output
    }
}

impl Default for ValidationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to truncate issues and add a "truncated" marker
fn truncate_mark(issues: &mut Vec<Issue>, max: usize) {
    if issues.len() <= max {
        return;
    }

    let remaining = issues.len() - max;
    issues.truncate(max);

    issues.push(Issue::new(
        Severity::Info,
        super::IssueCategory::Style,
        format!("... {} more issues truncated", remaining),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidirectional::parser::{Function, AST};

    #[test]
    fn test_pipeline_creation() {
        let pipeline = ValidationPipeline::new();
        assert_eq!(pipeline.stages.len(), 6);
    }

    #[test]
    fn test_empty_ast() {
        let pipeline = ValidationPipeline::new();
        let ast = AST {
            functions: Vec::new(),
            structs: Vec::new(),
            imports: Vec::new(),
        };

        let result = pipeline.validate(&ast);
        assert!(result.passed);
    }

    #[test]
    fn test_stop_on_critical() {
        let pipeline = ValidationPipeline::new().stop_on_critical(true);
        // Test that pipeline stops on critical issues
    }

    #[test]
    fn test_max_issues_per_stage() {
        let pipeline = ValidationPipeline::new().max_issues_per_stage(10);
        assert_eq!(pipeline.max_issues_per_stage, 10);
    }

    #[test]
    fn test_report_generation() {
        let pipeline = ValidationPipeline::new();
        let ast = AST {
            functions: vec![Function {
                name: "test".to_string(),
                params: Vec::new(),
                return_type: "()".to_string(),
                body: Vec::new(),
                attributes: Vec::new(),
            }],
            structs: Vec::new(),
            imports: Vec::new(),
        };

        let result = pipeline.validate(&ast);
        let report = pipeline.generate_report(&result);

        assert!(!report.to_text().is_empty());
    }
}
