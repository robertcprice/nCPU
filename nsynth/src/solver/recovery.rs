//! Error recovery system for the solver.
//!
//! Provides intelligent error classification, strategy selection, and recovery
//! mechanisms for handling various solver failures with graceful degradation.

use crate::solver::{SolveResult, SolverError};
use std::collections::HashMap;

/// Recovery strategies for handling solver errors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecoveryStrategy {
    /// Retry the solving process with a different internal route/algorithm.
    RetryWithDifferentRoute,
    /// Backtrack to the last successful intermediate state.
    BacktrackToLastSuccess,
    /// Simplify the problem (reduce examples, relax constraints).
    SimplifyProblem,
    /// Request human guidance for complex failures.
    RequestHumanGuidance,
    /// Apply a known fix pattern for common errors.
    ApplyKnownFix,
    /// Abort solving after exhausting all options.
    Abort,
}

/// A recovery plan with metadata for execution.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// The strategy to employ.
    pub strategy: RecoveryStrategy,
    /// Human-readable reasoning for this choice.
    pub reasoning: String,
    /// Estimated success probability (0.0 - 1.0).
    pub estimated_success: f64,
    /// Maximum attempts to make with this strategy.
    pub max_attempts: usize,
}

/// Context information about an error that occurred.
#[derive(Debug)]
pub struct ErrorContext {
    /// The error that occurred.
    pub error: SolverError,
    /// The solving phase when error occurred (e.g., "search", "gradient").
    pub phase: String,
    /// Any partial result obtained before failure.
    pub partial_result: Option<SolveResult>,
    /// Number of recovery attempts already made.
    pub attempts_so_far: usize,
    /// Additional context metadata.
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context.
    pub fn new(error: SolverError, phase: impl Into<String>) -> Self {
        Self {
            error,
            phase: phase.into(),
            partial_result: None,
            attempts_so_far: 0,
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the partial result.
    pub fn with_partial_result(mut self, result: SolveResult) -> Self {
        self.partial_result = Some(result);
        self
    }
}

/// Classification of error types for strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorType {
    /// Search space exhausted without finding solution.
    SearchExhausted,
    /// Performance-related (timeout, resource limits).
    Performance,
    /// Syntax/parse errors in input or generated code.
    Syntax,
    /// Type mismatches in generated code.
    TypeMismatch,
    /// Verification/semantic errors.
    Verification,
    /// Configuration-related errors.
    Configuration,
    /// Unknown/unclassified error.
    Unknown,
}

/// The recovery engine that orchestrates error recovery.
pub struct RecoveryEngine {
    max_total_attempts: usize,
    strategy_selector: Box<dyn StrategySelector>,
    history: Vec<RecoveryAttempt>,
}

/// Record of a recovery attempt.
#[derive(Debug, Clone)]
struct RecoveryAttempt {
    strategy: RecoveryStrategy,
    succeeded: bool,
    duration_ms: u64,
}

impl RecoveryEngine {
    /// Create a new recovery engine with default settings.
    pub fn new() -> Self {
        Self {
            max_total_attempts: 5,
            strategy_selector: Box::new(DefaultSelector::new()),
            history: Vec::new(),
        }
    }

    /// Create a recovery engine with a custom strategy selector.
    pub fn with_selector(mut self, selector: Box<dyn StrategySelector>) -> Self {
        self.strategy_selector = selector;
        self
    }

    /// Set the maximum total recovery attempts.
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_total_attempts = max;
        self
    }

    /// Attempt to recover from an error based on context.
    pub fn attempt_recovery(&self, ctx: &ErrorContext) -> Result<RecoveryPlan, SolverError> {
        // Classify error type
        let error_type = self.classify_error(&ctx.error);

        // Check if we should continue attempting
        if ctx.attempts_so_far >= self.max_total_attempts {
            return Ok(RecoveryPlan {
                strategy: RecoveryStrategy::Abort,
                reasoning: format!(
                    "Max total attempts ({} exceeded for error type: {:?}",
                    self.max_total_attempts, error_type
                ),
                estimated_success: 0.0,
                max_attempts: 0,
            });
        }

        // Select appropriate strategy based on error and context
        let plan = (self.strategy_selector).select_strategy(ctx, error_type);

        // Validate plan
        if plan.max_attempts == 0 {
            return Ok(RecoveryPlan {
                strategy: RecoveryStrategy::Abort,
                reasoning: "Strategy has zero remaining attempts".to_string(),
                estimated_success: 0.0,
                max_attempts: 0,
            });
        }

        Ok(plan)
    }

    /// Record a recovery attempt outcome.
    pub fn record_attempt(
        &mut self,
        strategy: RecoveryStrategy,
        succeeded: bool,
        duration_ms: u64,
    ) {
        self.history.push(RecoveryAttempt {
            strategy,
            succeeded,
            duration_ms,
        });
    }

    /// Get recovery statistics.
    pub fn stats(&self) -> RecoveryStats {
        let total_attempts = self.history.len();
        let successful_attempts = self.history.iter().filter(|a| a.succeeded).count();
        let success_rate = if total_attempts > 0 {
            successful_attempts as f64 / total_attempts as f64
        } else {
            0.0
        };

        let mut strategy_counts: HashMap<RecoveryStrategy, usize> = HashMap::new();
        for attempt in &self.history {
            *strategy_counts.entry(attempt.strategy.clone()).or_insert(0) += 1;
        }

        RecoveryStats {
            total_attempts,
            successful_attempts,
            success_rate,
            strategy_counts,
        }
    }

    /// Classify a SolverError into an ErrorType for strategy selection.
    fn classify_error(&self, error: &SolverError) -> ErrorType {
        match error {
            SolverError::NoSolutionFound(msg) => {
                if msg.contains("timeout") || msg.contains("exceeded") {
                    ErrorType::Performance
                } else if msg.contains("search") || msg.contains("exhausted") {
                    ErrorType::SearchExhausted
                } else {
                    ErrorType::Unknown
                }
            }
            SolverError::Timeout(_) => ErrorType::Performance,
            SolverError::ParseError(_) => ErrorType::Syntax,
            SolverError::VerificationFailed(_) => ErrorType::Verification,
            SolverError::ConfigurationError(_) => ErrorType::Configuration,
            SolverError::CommunicationError(_) => ErrorType::Unknown,
            SolverError::JoinError(_) => ErrorType::Unknown,
            SolverError::IoError(_) => ErrorType::Unknown,
            SolverError::Other(msg) => {
                if msg.contains("type") || msg.contains("mismatch") {
                    ErrorType::TypeMismatch
                } else {
                    ErrorType::Unknown
                }
            }
        }
    }
}

impl Default for RecoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about recovery attempts.
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_attempts: usize,
    pub successful_attempts: usize,
    pub success_rate: f64,
    pub strategy_counts: HashMap<RecoveryStrategy, usize>,
}

/// Trait for selecting recovery strategies based on error context.
pub trait StrategySelector: Send + Sync {
    /// Select a recovery strategy based on error context.
    fn select_strategy(&self, ctx: &ErrorContext, error_type: ErrorType) -> RecoveryPlan;
}

/// Default strategy selector with rule-based decision making.
pub struct DefaultSelector {
    /// Success threshold for continuing retry strategies.
    retry_threshold: f64,
}

impl DefaultSelector {
    /// Create a new default selector.
    pub fn new() -> Self {
        Self {
            retry_threshold: 0.3,
        }
    }

    /// Set the retry threshold (strategies below this success rate won't retry).
    pub fn with_retry_threshold(mut self, threshold: f64) -> Self {
        self.retry_threshold = threshold;
        self
    }
}

impl Default for DefaultSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategySelector for DefaultSelector {
    fn select_strategy(&self, ctx: &ErrorContext, error_type: ErrorType) -> RecoveryPlan {
        let (strategy, reasoning, success, attempts): (RecoveryStrategy, String, f64, usize) =
            match error_type {
                ErrorType::SearchExhausted => {
                    // For search exhaustion, try different routes first
                    if ctx.attempts_so_far < 2 {
                        (
                            RecoveryStrategy::RetryWithDifferentRoute,
                            "Search space exhausted, try alternative route".to_string(),
                            0.6,
                            2,
                        )
                    } else if ctx.partial_result.is_some() {
                        (
                            RecoveryStrategy::BacktrackToLastSuccess,
                            "Multiple routes failed, backtrack to partial solution".to_string(),
                            0.4,
                            1,
                        )
                    } else {
                        (
                            RecoveryStrategy::SimplifyProblem,
                            "No partial solution, try simplified problem".to_string(),
                            0.5,
                            1,
                        )
                    }
                }
                ErrorType::Performance => {
                    // For performance issues, simplify or abort
                    if ctx.attempts_so_far == 0 {
                        (
                            RecoveryStrategy::SimplifyProblem,
                            "Performance issue, try simplified version".to_string(),
                            0.5,
                            1,
                        )
                    } else {
                        (
                            RecoveryStrategy::Abort,
                            "Repeated performance issues, aborting".to_string(),
                            0.0,
                            0,
                        )
                    }
                }
                ErrorType::Syntax => {
                    // Syntax errors often need human guidance or known fixes
                    if ctx.attempts_so_far == 0 {
                        (
                            RecoveryStrategy::ApplyKnownFix,
                            "Syntax error, try common fix patterns".to_string(),
                            0.4,
                            1,
                        )
                    } else {
                        (
                            RecoveryStrategy::RequestHumanGuidance,
                            "Syntax fix failed, requires human review".to_string(),
                            0.2,
                            1,
                        )
                    }
                }
                ErrorType::TypeMismatch => {
                    // Type mismatches may need backtracking or simplification
                    (
                        RecoveryStrategy::BacktrackToLastSuccess,
                        "Type mismatch detected, backtrack to valid state".to_string(),
                        0.3,
                        1,
                    )
                }
                ErrorType::Verification => {
                    // Verification failures suggest semantic issues
                    if ctx.attempts_so_far < 2 {
                        (
                            RecoveryStrategy::RetryWithDifferentRoute,
                            "Verification failed, alternative route may work".to_string(),
                            0.5,
                            2,
                        )
                    } else {
                        (
                            RecoveryStrategy::SimplifyProblem,
                            "Multiple verification failures, simplify".to_string(),
                            0.4,
                            1,
                        )
                    }
                }
                ErrorType::Configuration => {
                    // Configuration errors should abort quickly
                    (
                        RecoveryStrategy::Abort,
                        "Configuration error cannot be recovered".to_string(),
                        0.0,
                        0,
                    )
                }
                ErrorType::Unknown => {
                    // Unknown errors get conservative backtracking
                    (
                        RecoveryStrategy::BacktrackToLastSuccess,
                        "Unknown error, conservative backtrack".to_string(),
                        0.3,
                        1,
                    )
                }
            };

        RecoveryPlan {
            strategy,
            reasoning,
            estimated_success: success,
            max_attempts: attempts.saturating_sub(ctx.attempts_so_far),
        }
    }
}

/// Adaptive strategy selector that learns from past attempts.
pub struct AdaptiveSelector {
    /// Success rates observed for each strategy.
    success_rates: HashMap<(ErrorType, RecoveryStrategy), f64>,
    /// Minimum samples before trusting success rates.
    min_samples: usize,
    /// Default fallback selector.
    fallback: DefaultSelector,
}

impl AdaptiveSelector {
    /// Create a new adaptive selector.
    pub fn new() -> Self {
        Self {
            success_rates: HashMap::new(),
            min_samples: 3,
            fallback: DefaultSelector::new(),
        }
    }

    /// Update success rates based on an attempt outcome.
    pub fn record_outcome(
        &mut self,
        error_type: ErrorType,
        strategy: RecoveryStrategy,
        succeeded: bool,
    ) {
        let key = (error_type, strategy);
        let rate = self.success_rates.entry(key).or_insert(0.5);

        // Simple exponential moving average
        *rate = if *rate == 0.5 {
            if succeeded {
                1.0
            } else {
                0.0
            }
        } else {
            *rate * 0.8 + if succeeded { 0.2 } else { 0.0 }
        };
    }
}

impl Default for AdaptiveSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategySelector for AdaptiveSelector {
    fn select_strategy(&self, ctx: &ErrorContext, error_type: ErrorType) -> RecoveryPlan {
        // Try to find best strategy from learned rates
        let mut best_strategy = None;
        let mut best_rate = 0.0;

        for ((et, strategy), rate) in self.success_rates.iter() {
            if *et == error_type && *rate > best_rate {
                best_rate = *rate;
                best_strategy = Some(strategy.clone());
            }
        }

        if let Some(strategy) = best_strategy {
            return RecoveryPlan {
                strategy,
                reasoning: format!(
                    "Adaptive selection based on learned success rate {:.2}",
                    best_rate
                ),
                estimated_success: best_rate,
                max_attempts: 2_usize.saturating_sub(ctx.attempts_so_far),
            };
        }

        // Fall back to default selector
        self.fallback.select_strategy(ctx, error_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let engine = RecoveryEngine::new();

        assert_eq!(
            engine.classify_error(&SolverError::Timeout("test".to_string())),
            ErrorType::Performance
        );
        assert_eq!(
            engine.classify_error(&SolverError::ParseError("test".to_string())),
            ErrorType::Syntax
        );
        assert_eq!(
            engine.classify_error(&SolverError::VerificationFailed("test".to_string())),
            ErrorType::Verification
        );
        assert_eq!(
            engine.classify_error(&SolverError::ConfigurationError("test".to_string())),
            ErrorType::Configuration
        );
    }

    #[test]
    fn test_recovery_plan_max_attempts() {
        let engine = RecoveryEngine::new().with_max_attempts(3);
        let selector = DefaultSelector::new();

        let ctx = ErrorContext::new(
            SolverError::NoSolutionFound("test".to_string()),
            "test_phase",
        );

        // First attempt should get full attempts
        let plan = selector.select_strategy(&ctx, ErrorType::SearchExhausted);
        assert_eq!(plan.max_attempts, 2);

        // After 2 attempts, should have fewer
        let ctx = ErrorContext {
            attempts_so_far: 2,
            ..ctx
        };
        let plan = selector.select_strategy(&ctx, ErrorType::SearchExhausted);
        // attempts_so_far=2: SearchExhausted's <2 branch is false, no partial_result
        // → SimplifyProblem with attempts=1, max_attempts = 1.saturating_sub(2) = 0
        assert_eq!(plan.max_attempts, 0);
    }

    #[test]
    fn test_engine_abort_on_max_attempts() {
        let engine = RecoveryEngine::new().with_max_attempts(2);

        let ctx = ErrorContext {
            attempts_so_far: 2,
            ..ErrorContext::new(SolverError::Timeout("test".to_string()), "test")
        };

        let plan = engine.attempt_recovery(&ctx).unwrap();
        assert_eq!(plan.strategy, RecoveryStrategy::Abort);
    }

    #[test]
    fn test_context_builder() {
        let ctx = ErrorContext::new(SolverError::ParseError("test".to_string()), "parsing")
            .with_metadata("key", "value")
            .with_partial_result(SolveResult {
                success: false,
                code: String::new(),
                method: "test".to_string(),
                error: None,
                metadata: crate::differentiable::DifferentiableMetadata::default(),
            });

        assert_eq!(ctx.phase, "parsing");
        assert!(ctx.partial_result.is_some());
        assert_eq!(ctx.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_adaptive_selector_learning() {
        let mut selector = AdaptiveSelector::new();

        let error_type = ErrorType::SearchExhausted;
        let strategy = RecoveryStrategy::RetryWithDifferentRoute;

        // Record successful outcomes
        selector.record_outcome(error_type, strategy.clone(), true);
        selector.record_outcome(error_type, strategy.clone(), true);
        selector.record_outcome(error_type, strategy.clone(), false);

        let ctx = ErrorContext::new(SolverError::NoSolutionFound("test".to_string()), "search");

        let plan = selector.select_strategy(&ctx, error_type);
        assert_eq!(plan.strategy, strategy);
        assert!(plan.estimated_success > 0.5);
    }
}
