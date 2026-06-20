//! Individual validation stage implementations
//!
//! Each stage performs a specific category of validation.

use super::{Issue, IssueCategory, ValidationResult, Warning};
use crate::bidirectional::parser::{BinOp, Expression, Function, Statement, AST};
use crate::validation::pipeline::ValidationStage;

/// Syntax validation stage
#[derive(Debug, Clone, Default)]
pub struct SyntaxValidationStage {
    check_unused_vars: bool,
    check_dead_code: bool,
}

impl SyntaxValidationStage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_unused_check(mut self, check: bool) -> Self {
        self.check_unused_vars = check;
        self
    }

    pub fn with_dead_code_check(mut self, check: bool) -> Self {
        self.check_dead_code = check;
        self
    }

    /// Check for unreachable code
    fn check_unreachable_code(&self, stmts: &[Statement]) -> Vec<Issue> {
        let mut issues = Vec::new();

        for (idx, stmt) in stmts.iter().enumerate() {
            match stmt {
                Statement::Return(_) => {
                    if idx < stmts.len() - 1 {
                        issues.push(
                            Issue::medium(
                                IssueCategory::Syntax,
                                "Unreachable code after return or panic",
                            )
                            .with_location("generated", idx + 1, 0)
                            .with_fix("Remove unreachable statements"),
                        );
                    }
                }
                Statement::Expr(expr) if matches!(&**expr, Expression::Call { func, .. } if func == "panic" || func == "unreachable") => {
                    if idx < stmts.len() - 1 {
                        issues.push(
                            Issue::medium(
                                IssueCategory::Syntax,
                                "Unreachable code after return or panic",
                            )
                            .with_location("generated", idx + 1, 0)
                            .with_fix("Remove unreachable statements"),
                        );
                    }
                }
                _ => {}
            }
        }

        issues
    }

    /// Check for suspicious empty blocks
    fn check_empty_blocks(&self, stmts: &[Statement]) -> Vec<Issue> {
        let mut issues = Vec::new();

        for (idx, stmt) in stmts.iter().enumerate() {
            match stmt {
                Statement::If { then_block, .. } if then_block.is_empty() => {
                    issues.push(
                        Issue::low(IssueCategory::Syntax, "Empty if block")
                            .with_location("generated", idx + 1, 0)
                            .with_fix("Add logic or remove empty if"),
                    );
                }
                Statement::Loop { body }
                | Statement::While { body, .. }
                | Statement::For { body, .. }
                    if body.is_empty() =>
                {
                    issues.push(
                        Issue::medium(
                            IssueCategory::Syntax,
                            "Empty loop body - possible infinite loop or no-op",
                        )
                        .with_location("generated", idx + 1, 0)
                        .with_fix("Add loop body or remove loop"),
                    );
                }
                _ => {}
            }
        }

        issues
    }
}

/// Type validation stage
#[derive(Debug, Clone, Default)]
pub struct TypeValidationStage;

impl TypeValidationStage {
    pub fn new() -> Self {
        Self
    }

    /// Check for potentially missing type annotations
    fn check_implicit_types(&self, function: &Function) -> Vec<Issue> {
        let mut issues = Vec::new();

        for param in &function.params {
            if param.type_.is_empty() || param.type_ == "_" {
                issues.push(
                    Issue::high(
                        IssueCategory::Type,
                        format!("Parameter '{}' has no type annotation", param.name),
                    )
                    .with_location("generated", 0, 0)
                    .with_fix("Add explicit type annotation"),
                );
            }
        }

        issues
    }

    /// Check return type consistency
    fn check_return_type(&self, function: &Function) -> Vec<Issue> {
        let mut issues = Vec::new();

        let has_return = function
            .body
            .iter()
            .any(|s| matches!(s, Statement::Return(_)));

        if has_return && function.return_type == "()" {
            issues.push(
                Issue::medium(
                    IssueCategory::Type,
                    format!(
                        "Function '{}' returns value but declares unit return type",
                        function.name
                    ),
                )
                .with_location("generated", 0, 0)
                .with_fix("Update return type annotation"),
            );
        }

        issues
    }
}

/// Security validation stage
#[derive(Debug, Clone, Default)]
pub struct SecurityValidationStage {
    check_unsafe_blocks: bool,
    check_panic_calls: bool,
}

impl SecurityValidationStage {
    pub fn new() -> Self {
        Self {
            check_unsafe_blocks: true,
            check_panic_calls: true,
        }
    }

    /// Check for potential integer overflow
    fn check_overflow_risk(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::BinOp { op, left, right } => {
                match op {
                    BinOp::Mul | BinOp::Add => {
                        // Check if operands are potentially large
                        if let Expression::Call { func, .. } = &**left {
                            if func.contains("len") || func.contains("size") {
                                issues.push(
                                    Issue::high(
                                        IssueCategory::Security,
                                        "Potential integer overflow from size-based operation",
                                    )
                                    .with_fix("Use checked_mul or saturating operations"),
                                );
                            }
                        }
                    }
                    _ => {}
                }

                // Recursively check operands
                issues.extend(self.check_overflow_risk(left));
                issues.extend(self.check_overflow_risk(right));
            }
            _ => {}
        }

        issues
    }

    /// Check for panic calls that could crash
    fn check_panics(&self, stmt: &Statement) -> Vec<Issue> {
        let mut issues = Vec::new();

        match stmt {
            Statement::Expr(expr) if matches!(&**expr, Expression::Call { func, .. } if func == "panic" || func == "unreachable") =>
            {
                issues.push(
                    Issue::medium(
                        IssueCategory::Security,
                        "Direct panic call may cause program termination",
                    )
                    .with_fix("Use Result or Option for error handling"),
                );
            }
            _ => {}
        }

        issues
    }

    /// Check for index access without bounds checking
    fn check_unsafe_index(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::Index { index, .. } => {
                // Check for potential out-of-bounds access
                if let Expression::Int(n) = &**index {
                    if *n < 0 {
                        issues.push(
                            Issue::high(IssueCategory::Security, "Negative array index")
                                .with_fix("Ensure index is non-negative"),
                        );
                    }
                }
            }
            _ => {}
        }

        issues
    }
}

/// Performance validation stage
#[derive(Debug, Clone, Default)]
pub struct PerformanceValidationStage {
    check_complexity: bool,
    check_allocations: bool,
}

impl PerformanceValidationStage {
    pub fn new() -> Self {
        Self {
            check_complexity: true,
            check_allocations: true,
        }
    }

    /// Check for nested loops (potential O(n²) complexity)
    fn check_nested_loops(&self, stmts: &[Statement]) -> Vec<Issue> {
        let mut issues = Vec::new();
        let mut loop_depth = 0;

        for stmt in stmts {
            match stmt {
                Statement::Loop { .. } | Statement::While { .. } | Statement::For { .. } => {
                    loop_depth += 1;
                    if loop_depth > 2 {
                        issues.push(
                            Issue::medium(
                                IssueCategory::Performance,
                                format!(
                                    "Nested loops at depth {} - potential O(n^{}) complexity",
                                    loop_depth, loop_depth
                                ),
                            )
                            .with_fix("Consider algorithmic optimization"),
                        );
                    }
                }
                Statement::Block(inner) => {
                    issues.extend(self.check_nested_loops(inner));
                }
                Statement::If {
                    condition: _,
                    then_block,
                    else_block,
                } => {
                    issues.extend(self.check_nested_loops(then_block));
                    if let Some(else_block) = else_block {
                        issues.extend(self.check_nested_loops(else_block));
                    }
                }
                _ => {
                    // Reset loop depth after non-loop statement
                    loop_depth = 0;
                }
            }
        }

        issues
    }

    /// Check for inefficient string operations
    fn check_string_operations(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::BinOp { op, left, .. } => {
                if matches!(op, BinOp::Add) {
                    if let Expression::Call { func, .. } = &**left {
                        if func.contains("to_string") || func.contains("format") {
                            issues.push(
                                Issue::low(
                                    IssueCategory::Performance,
                                    "String concatenation in loop may allocate repeatedly",
                                )
                                .with_fix("Use String::with_capacity or format!"),
                            );
                        }
                    }
                }
            }
            _ => {}
        }

        issues
    }

    /// Check for redundant computations
    fn check_redundant_computation(&self, stmts: &[Statement]) -> Vec<Issue> {
        let mut issues = Vec::new();
        let mut call_counts = std::collections::HashMap::new();

        for stmt in stmts {
            if let Statement::Expr(expr) = stmt {
                if let Expression::Call { func, .. } = &**expr {
                    *call_counts.entry(func.clone()).or_insert(0) += 1;
                }
            }
        }

        for (func, count) in call_counts {
            if count > 3 && func.contains("len") {
                issues.push(
                    Issue::low(
                        IssueCategory::Performance,
                        format!("'{}' called {} times - consider caching", func, count),
                    )
                    .with_fix("Store result in a variable"),
                );
            }
        }

        issues
    }
}

/// Correctness validation stage
#[derive(Debug, Clone, Default)]
pub struct CorrectnessValidationStage {
    check_logic_errors: bool,
    check_edge_cases: bool,
}

impl CorrectnessValidationStage {
    pub fn new() -> Self {
        Self {
            check_logic_errors: true,
            check_edge_cases: true,
        }
    }

    /// Check for suspicious comparisons
    fn check_suspicious_comparisons(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::BinOp { op, left, right } => {
                match op {
                    BinOp::Eq | BinOp::Ne => {
                        // Check for float comparison
                        if let (Expression::Float(_), _) = (&**left, &**right) {
                            issues.push(
                                Issue::medium(
                                    IssueCategory::Correctness,
                                    "Direct float comparison may fail due to precision",
                                )
                                .with_fix("Use epsilon-based comparison"),
                            );
                        }

                        // Check for comparison with literal that might be typo
                        if let Expression::Int(n) = &**right {
                            if *n == 1 && matches!(op, BinOp::Eq) {
                                issues.push(
                                    Issue::low(
                                        IssueCategory::Correctness,
                                        "Comparison with 1 might be a typo for assignment",
                                    )
                                    .with_fix("Verify intended comparison"),
                                );
                            }
                        }
                    }
                    _ => {}
                }

                issues.extend(self.check_suspicious_comparisons(left));
                issues.extend(self.check_suspicious_comparisons(right));
            }
            _ => {}
        }

        issues
    }

    /// Check for division by zero risk
    fn check_division_by_zero(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::BinOp { op, right, .. } if matches!(op, BinOp::Div | BinOp::Mod) => {
                if let Expression::Int(0) = &**right {
                    issues.push(
                        Issue::critical(IssueCategory::Correctness, "Division by zero")
                            .with_fix("Add guards or check for zero divisor"),
                    );
                }
            }
            _ => {}
        }

        issues
    }

    /// Check for off-by-one errors
    fn check_off_by_one(&self, expr: &Expression) -> Vec<Issue> {
        let mut issues = Vec::new();

        match expr {
            Expression::BinOp { op, .. } => {
                if matches!(op, BinOp::Le | BinOp::Lt) {
                    // Look for patterns like i < len where i might need to be <=
                    issues.extend(std::iter::once(
                        Issue::low(
                            IssueCategory::Correctness,
                            "Potential off-by-one error in comparison",
                        )
                        .with_fix("Verify boundary conditions"),
                    ));
                }
            }
            _ => {}
        }

        issues
    }

    /// Check for unused variables
    fn check_unused_variables(&self, function: &Function) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Simple heuristic: if parameter appears nowhere in body, mark as unused
        // This is a simplified check - real implementation would need full symbol analysis
        for param in &function.params {
            let param_str = format!("&{}", param.name);
            let is_used = function.body.iter().any(|stmt| {
                let stmt_string = format!("{:?}", stmt);
                stmt_string.contains(&param.name) || stmt_string.contains(&param_str)
            });

            if !is_used && !param.name.starts_with('_') {
                issues.push(
                    Issue::low(
                        IssueCategory::Correctness,
                        format!("Parameter '{}' is never used", param.name),
                    )
                    .with_fix("Prefix with '_' or remove parameter"),
                );
            }
        }

        issues
    }
}

/// Style validation stage
#[derive(Debug, Clone, Default)]
pub struct StyleValidationStage {
    check_naming: bool,
    check_line_length: bool,
}

impl StyleValidationStage {
    pub fn new() -> Self {
        Self {
            check_naming: true,
            check_line_length: true,
        }
    }

    /// Check naming conventions
    fn check_naming_conventions(&self, function: &Function) -> Vec<Issue> {
        let mut issues = Vec::new();

        // Function names should be snake_case
        if !function.name.is_empty() && function.name.chars().next().unwrap().is_uppercase() {
            issues.push(
                Issue::low(
                    IssueCategory::Style,
                    format!("Function '{}' should use snake_case naming", function.name),
                )
                .with_fix("Rename to snake_case"),
            );
        }

        // Parameter names should be snake_case
        for param in &function.params {
            if !param.name.is_empty() && param.name.chars().next().unwrap().is_uppercase() {
                issues.push(
                    Issue::low(
                        IssueCategory::Style,
                        format!("Parameter '{}' should use snake_case naming", param.name),
                    )
                    .with_fix("Rename to snake_case"),
                );
            }
        }

        issues
    }

    /// Check for overly complex functions
    fn check_function_complexity(&self, function: &Function) -> Vec<Issue> {
        let mut issues = Vec::new();

        let statement_count = function.body.len();
        if statement_count > 50 {
            issues.push(
                Issue::medium(
                    IssueCategory::Style,
                    format!(
                        "Function '{}' has {} statements - consider splitting",
                        function.name, statement_count
                    ),
                )
                .with_fix("Extract helper functions"),
            );
        }

        issues
    }
}

// Implement ValidationStage trait for all stages

impl super::pipeline::ValidationStage for SyntaxValidationStage {
    fn name(&self) -> &str {
        "Syntax Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings: Vec<Warning> = Vec::new();

        for function in &program.functions {
            if self.check_dead_code {
                issues.extend(self.check_unreachable_code(&function.body));
            }
            issues.extend(self.check_empty_blocks(&function.body));
        }

        if issues.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failed(issues)
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

impl super::pipeline::ValidationStage for TypeValidationStage {
    fn name(&self) -> &str {
        "Type Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();

        for function in &program.functions {
            issues.extend(self.check_implicit_types(function));
            issues.extend(self.check_return_type(function));
        }

        if issues.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failed(issues)
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

impl super::pipeline::ValidationStage for SecurityValidationStage {
    fn name(&self) -> &str {
        "Security Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();

        for function in &program.functions {
            for stmt in &function.body {
                if self.check_panic_calls {
                    issues.extend(self.check_panics(stmt));
                }

                // Recursively check expressions
                if let Statement::Expr(expr) = stmt {
                    issues.extend(self.check_overflow_risk(expr));
                    issues.extend(self.check_unsafe_index(expr));
                }
            }
        }

        if issues.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failed(issues)
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

impl super::pipeline::ValidationStage for PerformanceValidationStage {
    fn name(&self) -> &str {
        "Performance Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();
        let mut warnings: Vec<Warning> = Vec::new();

        for function in &program.functions {
            if self.check_complexity {
                issues.extend(self.check_nested_loops(&function.body));
                issues.extend(self.check_redundant_computation(&function.body));
            }

            for stmt in &function.body {
                if let Statement::Expr(expr) = stmt {
                    issues.extend(self.check_string_operations(expr));
                }
            }
        }

        if issues.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failed(issues)
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

impl super::pipeline::ValidationStage for CorrectnessValidationStage {
    fn name(&self) -> &str {
        "Correctness Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();

        for function in &program.functions {
            if self.check_logic_errors {
                issues.extend(self.check_unused_variables(function));
            }

            for stmt in &function.body {
                if let Statement::Expr(expr) = stmt {
                    if self.check_edge_cases {
                        issues.extend(self.check_division_by_zero(expr));
                        issues.extend(self.check_off_by_one(expr));
                        issues.extend(self.check_suspicious_comparisons(expr));
                    }
                }
            }
        }

        if issues.is_empty() {
            ValidationResult::success()
        } else {
            ValidationResult::failed(issues)
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

impl super::pipeline::ValidationStage for StyleValidationStage {
    fn name(&self) -> &str {
        "Style Validation"
    }

    fn validate(&self, program: &AST) -> ValidationResult {
        let mut issues = Vec::new();

        for function in &program.functions {
            if self.check_naming {
                issues.extend(self.check_naming_conventions(function));
            }
            issues.extend(self.check_function_complexity(function));
        }

        // Style issues don't fail validation
        let warnings: Vec<_> = issues
            .into_iter()
            .map(|i| {
                let file = i
                    .location
                    .as_ref()
                    .map(|l| l.file.as_str())
                    .unwrap_or("generated");
                let line = i.location.as_ref().map(|l| l.line).unwrap_or(0);
                let column = i.location.as_ref().map(|l| l.column).unwrap_or(0);
                Warning::new(i.message).with_location(file, line, column)
            })
            .collect();

        ValidationResult {
            passed: true,
            issues: Vec::new(),
            warnings,
            score: 1.0,
            category_counts: Default::default(),
        }
    }

    fn can_fix(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidirectional::parser::{BinOp, Expression, Function, Parameter, Statement, AST};

    #[test]
    fn test_syntax_validation() {
        let stage = SyntaxValidationStage::new();
        let ast = AST {
            functions: vec![Function {
                name: "test".to_string(),
                params: vec![],
                return_type: "()".to_string(),
                body: vec![
                    Statement::Return(Box::new(Expression::Int(0))),
                    Statement::Expr(Box::new(Expression::Int(42))), // Unreachable
                ],
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        assert!(!result.passed);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_type_validation() {
        let stage = TypeValidationStage::new();
        let ast = AST {
            functions: vec![Function {
                name: "test".to_string(),
                params: vec![Parameter {
                    name: "x".to_string(),
                    type_: "".to_string(),
                }],
                return_type: "()".to_string(),
                body: vec![Statement::Return(Box::new(Expression::Int(0)))],
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        assert!(!result.passed);
    }

    #[test]
    fn test_security_validation() {
        let stage = SecurityValidationStage::new();
        let ast = AST {
            functions: vec![Function {
                name: "test".to_string(),
                params: vec![],
                return_type: "()".to_string(),
                body: vec![Statement::Expr(Box::new(Expression::BinOp {
                    op: BinOp::Div,
                    left: Box::new(Expression::Int(10)),
                    right: Box::new(Expression::Int(0)),
                }))],
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        assert!(!result.passed);
    }

    #[test]
    fn test_performance_validation() {
        let stage = PerformanceValidationStage::new();

        let nested_loop = vec![Statement::For {
            var: "i".to_string(),
            iter: Box::new(Expression::Variable("items".to_string())),
            body: vec![Statement::For {
                var: "j".to_string(),
                iter: Box::new(Expression::Variable("inner".to_string())),
                body: vec![Statement::For {
                    var: "k".to_string(),
                    iter: Box::new(Expression::Variable("deep".to_string())),
                    body: vec![],
                }],
            }],
        }];

        let ast = AST {
            functions: vec![Function {
                name: "nested_loops".to_string(),
                params: vec![],
                return_type: "()".to_string(),
                body: nested_loop,
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        assert!(!result.passed);
    }

    #[test]
    fn test_correctness_validation() {
        let stage = CorrectnessValidationStage::new();
        let ast = AST {
            functions: vec![Function {
                name: "test".to_string(),
                params: vec![Parameter {
                    name: "unused".to_string(),
                    type_: "i64".to_string(),
                }],
                return_type: "()".to_string(),
                body: vec![Statement::Return(Box::new(Expression::Int(0)))],
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        // Should pass even with unused var (just a low severity issue)
        assert!(result.passed);
    }

    #[test]
    fn test_style_validation() {
        let stage = StyleValidationStage::new();
        let ast = AST {
            functions: vec![Function {
                name: "BadName".to_string(), // Violates snake_case
                params: vec![],
                return_type: "()".to_string(),
                body: vec![],
                attributes: vec![],
            }],
            structs: vec![],
            imports: vec![],
        };

        let result = stage.validate(&ast);
        // Style validation should always pass (warnings only)
        assert!(result.passed);
        assert!(!result.warnings.is_empty());
    }
}
