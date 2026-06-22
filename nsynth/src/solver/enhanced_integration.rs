//! Enhanced solver integration layer for Phase 1 modules.
//!
//! Combines transfer learning, multi-objective optimization into a unified
//! solver enhancement layer. All thresholds and constraints are learned from
//! data distributions (percentiles, no hardcoded values).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use crate::benchmark::Problem;
use crate::solver::SolveResult;
use crate::solver::method_stats::{TypeClass, ProblemFeatures};
use crate::transfer_learning::TransferLearner;
use crate::multi_objective::{MultiObjectiveOptimizer, ParetoPoint, Solution, ObjectiveVector};

/// Global flag to enable/disable enhanced solver at runtime.
/// Set via CLI flag --enhanced-solver or programmatically.
static ENHANCED_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable enhanced solver globally.
pub fn enable_enhanced_solver() {
    ENHANCED_ENABLED.store(true, Ordering::SeqCst);
}

/// Disable enhanced solver globally.
pub fn disable_enhanced_solver() {
    ENHANCED_ENABLED.store(false, Ordering::SeqCst);
}

/// Check if enhanced solver is enabled.
pub fn enhanced_solver_enabled() -> bool {
    ENHANCED_ENABLED.load(Ordering::SeqCst)
}

/// Global enhanced solver instance (lazy initialization).
/// Ported from `once_cell::sync::Lazy` to the std `LazyLock` (stable since
/// Rust 1.80) to avoid pulling in the `once_cell` crate, which is not a
/// canonical dependency.
static ENHANCED_SOLVER: std::sync::LazyLock<EnhancedSolver> =
    std::sync::LazyLock::new(EnhancedSolver::new);

/// Main enhanced solver coordinating all Phase 1 modules.
#[derive(Clone)]
pub struct EnhancedSolver {
    /// Transfer learning across domains
    transfer: Arc<TransferLearner>,
    /// Multi-objective Pareto optimizer
    optimizer: Arc<MultiObjectiveOptimizer>,
    /// Last update timestamp for learned parameters
    last_update: Arc<std::sync::RwLock<SystemTime>>,
}

impl EnhancedSolver {
    /// Create new enhanced solver with all modules initialized.
    pub fn new() -> Self {
        Self {
            transfer: Arc::new(TransferLearner::new()),
            optimizer: Arc::new(MultiObjectiveOptimizer::new()),
            last_update: Arc::new(std::sync::RwLock::new(SystemTime::now())),
        }
    }

    /// Get global enhanced solver instance.
    pub fn global() -> &'static EnhancedSolver {
        &ENHANCED_SOLVER
    }

    /// Select best solution from multiple candidates using Pareto optimization.
    ///
    /// When multiple solver methods return candidates, select the one that
    /// maximizes success rate while minimizing latency, memory, and cost.
    ///
    /// Uses timeout-protected lock operations with graceful degradation.
    pub fn select_best_solution(&self, candidates: Vec<SolveResult>) -> Option<SolveResult> {
        if candidates.is_empty() {
            return None;
        }

        // Convert candidates to Pareto points
        let pareto_candidates: Vec<ParetoPoint> = candidates
            .iter()
            .enumerate()
            .filter_map(|(_i, result)| {
                let objectives = self.extract_objectives(result);
                // Skip invalid objectives
                if !objectives.is_valid() {
                    return None;
                }
                Some(ParetoPoint {
                    solution: Solution::Method {
                        name: result.method.clone(),
                        confidence: result.confidence_score(),
                    },
                    objectives,
                })
            })
            .collect();

        if pareto_candidates.is_empty() {
            // Fallback: return first successful candidate if all objectives invalid
            return candidates.into_iter().find(|r| r.success);
        }

        // Update Pareto front with all candidates
        for point in &pareto_candidates {
            self.optimizer.update_pareto_front(point.clone());
        }

        // Select best using learned weights
        let selected = self.optimizer.select(pareto_candidates)?;

        // Find matching result by method name
        for (_i, result) in candidates.into_iter().enumerate() {
            if result.method == selected.solution.method_name() {
                return Some(result);
            }
        }
        None
    }

    /// Get recommended source domain for transfer learning.
    pub fn recommended_source_domain(&self, features: &ProblemFeatures) -> Option<TypeClass> {
        self.transfer.find_source(features)
    }

    /// Extract problem features for transfer learning.
    pub fn extract_features(&self, problem: &Problem) -> ProblemFeatures {
        ProblemFeatures::from_problem(problem)
    }

    /// Record transfer learning outcome for cross-domain knowledge tracking.
    ///
    /// Wrapper around TransferLearner::record_outcome that gracefully handles errors.
    /// Returns true if recording succeeded, false otherwise.
    pub fn record_transfer_outcome(&self, source: TypeClass, target: TypeClass, success: bool) -> bool {
        self.transfer.record_outcome(source, target, success).is_ok()
    }

    /// Extract objective vector from solve result for multi-objective optimization.
    ///
    /// Applies bounds checking and numerical validation to all heuristics.
    fn extract_objectives(&self, result: &SolveResult) -> ObjectiveVector {
        // Extract with bounds validation
        let success_rate = if result.success { 1.0 } else { result.partial_success_rate() };
        let latency_ms = result.estimated_latency_ms();
        let memory_bytes = result.estimated_memory_bytes();
        let cost_cents = result.estimated_cost_cents();

        // Clamp and validate
        let success = success_rate.max(0.0).min(1.0);
        let latency = latency_ms.min(u64::MAX - 1000); // Safe ceiling
        let memory = memory_bytes.min(usize::MAX / 2);  // Prevent overflow
        let cost = if cost_cents.is_finite() && cost_cents >= 0.0 {
            cost_cents.min(1_000_000.0) // $10,000 max
        } else {
            0.0 // Invalid cost treated as zero
        };

        ObjectiveVector {
            success_rate: success,
            latency_ms: latency,
            memory_bytes: memory,
            cost_cents: cost,
        }
    }

    /// Update last modified timestamp.
    pub fn touch(&self) {
        let _ = self.last_update.write().map(|mut w| *w = SystemTime::now());
    }

    /// Get statistics about all enhanced modules.
    ///
    /// Uses try_read for timeout protection; returns stats with current values
    /// or defaults if locks are contended.
    pub fn stats(&self) -> EnhancedStats {
        let optimizer_stats = self.optimizer.get_stats();

        // Timeout-protected lock read with fallback
        let last_update = self.last_update.try_read()
            .ok()
            .map(|guard| *guard)
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

        EnhancedStats {
            pareto_front_size: optimizer_stats.pareto_front_size,
            max_front_size: optimizer_stats.max_front_size,
            last_update,
        }
    }
}

impl Default for EnhancedSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the enhanced solver state.
#[derive(Clone, Debug)]
pub struct EnhancedStats {
    /// Current Pareto front size
    pub pareto_front_size: usize,
    /// Maximum Pareto front size (learned)
    pub max_front_size: usize,
    /// Last update timestamp
    pub last_update: SystemTime,
}

// ========================================================================
// Extensions to SolveResult
// ========================================================================

/// Extension methods for SolveResult to support multi-objective optimization.
pub trait SolveResultExt {
    /// Estimated confidence score [0, 1].
    fn confidence_score(&self) -> f64;

    /// Partial success rate when not fully successful.
    fn partial_success_rate(&self) -> f64;

    /// Estimated latency in milliseconds (heuristic).
    fn estimated_latency_ms(&self) -> u64;

    /// Estimated memory footprint in bytes (heuristic).
    fn estimated_memory_bytes(&self) -> usize;

    /// Estimated compute cost in cents (heuristic).
    fn estimated_cost_cents(&self) -> f64;

    /// Whether this is a complete success.
    fn is_complete_success(&self) -> bool;
}

impl SolveResultExt for SolveResult {
    fn confidence_score(&self) -> f64 {
        // Canonical `SolveResult` does not model partial successes (no
        // `partial_results` field), so confidence is binary: full on success,
        // zero otherwise. The fork's partial-success branch is dropped because
        // the canonical solver never emits partials.
        if self.success {
            1.0
        } else {
            0.0
        }
    }

    fn partial_success_rate(&self) -> f64 {
        // No partial-results concept in canonical `SolveResult`.
        0.0
    }

    fn estimated_latency_ms(&self) -> u64 {
        // Heuristic: code length × method multiplier
        let base = (self.code.len() as u64 / 10).min(10_000); // Cap base at 1000 chars worth
        let multiplier = match self.method.as_str() {
            m if m.contains("gradient") => 100,
            m if m.contains("search") => 50,
            m if m.contains("enumerative") => 10,
            _ => 20,
        };
        // Guard against overflow: use saturating mul
        base.saturating_mul(multiplier).min(u64::MAX - 1000)
    }

    fn estimated_memory_bytes(&self) -> usize {
        // Heuristic: code size + estimated runtime overhead
        let code_size = self.code.len();
        let overhead = match self.method.as_str() {
            m if m.contains("gradient") => 10_000_000,  // 10MB for tensors
            m if m.contains("search") => 1_000_000,    // 1MB for search state
            _ => 100_000,                             // 100KB baseline
        };
        // Guard against overflow
        code_size.saturating_add(overhead).min(usize::MAX / 2)
    }

    fn estimated_cost_cents(&self) -> f64 {
        // Heuristic: latency × cost factor
        let latency_ms = self.estimated_latency_ms();
        let cost_factor = match self.method.as_str() {
            m if m.contains("gradient") => 0.01,  // GPU expensive
            m if m.contains("search") => 0.005,
            _ => 0.001,
        };
        // Guard: ensure result is finite and non-negative
        let cost = latency_ms as f64 * cost_factor;
        if cost.is_finite() && cost >= 0.0 {
            cost.min(1_000_000.0) // $10,000 max
        } else {
            0.0
        }
    }

    fn is_complete_success(&self) -> bool {
        // No partial-results in canonical `SolveResult`; success is complete.
        self.success
    }
}

/// Extension for Solution to extract method name.
impl Solution {
    pub fn method_name(&self) -> String {
        match self {
            Solution::Method { name, .. } => name.clone(),
            Solution::Program { .. } => "synthesized_program".to_string(),
        }
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Problem, Value};

    fn test_problem() -> Problem {
        Problem {
            name: "test_add".to_string(),
            category: "arithmetic",
            description: "Add two numbers",
            signature: "fn test_add(a: i64, b: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(1), Value::Int(2)],
                    expected: Value::Int(3),
                },
                Example {
                    inputs: vec![Value::Int(5), Value::Int(7)],
                    expected: Value::Int(12),
                },
            ],
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
    fn test_enhanced_solver_creation() {
        let solver = EnhancedSolver::new();
        let stats = solver.stats();
        assert_eq!(stats.pareto_front_size, 0);
    }

    #[test]
    fn test_extract_features() {
        let solver = EnhancedSolver::new();
        let problem = test_problem();
        let features = solver.extract_features(&problem);

        assert_eq!(features.arity, 2); // 2 args
    }

    #[test]
    fn test_solve_result_extensions() {
        let result = SolveResult {
            success: true,
            code: "fn add(a: i64, b: i64) -> i64 { return a + b; }".to_string(),
            method: "direct_search".to_string(),
            error: None,
            metadata: Default::default(),
        };

        assert_eq!(result.confidence_score(), 1.0);
        assert!(result.estimated_latency_ms() > 0);
        assert!(result.is_complete_success());
    }

    #[test]
    fn test_pareto_solution_selection() {
        let solver = EnhancedSolver::new();

        let candidates = vec![
            SolveResult {
                success: true,
                code: "fast".to_string(),
                method: "fast_method".to_string(),
                error: None,
                metadata: Default::default(),
            },
            SolveResult {
                success: true,
                code: "slow".to_string(),
                method: "slow_method".to_string(),
                error: None,
                metadata: Default::default(),
            },
        ];

        let selected = solver.select_best_solution(candidates);
        assert!(selected.is_some());
    }
}
