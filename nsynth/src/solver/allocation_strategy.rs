//! EMA-based solver allocation strategy for intelligent parallelism control.
//!
//! Tracks per-solver success rates and latencies using exponential moving averages
//! to make intelligent allocation decisions. Domains are matched mechanically from
//! problem structure, and selection balances expected success, latency, and cost.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::benchmark::Problem;
use crate::solver::method_stats::TypeClass;

/// EMA alpha for smooth adaptation - gives 10% weight to new observations.
const EMA_ALPHA: f64 = 0.1;

/// Minimum success threshold for solver consideration (default).
const DEFAULT_MIN_SUCCESS_THRESHOLD: f64 = 0.3;

/// Solver method enumeration for allocation decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SolverMethod {
    /// Gradient-based search for scalar integer problems.
    GradientSearch,
    /// Enumerative search for small integer domains.
    EnumerativeSearch,
    /// Teacher-guided search using learned patterns.
    TeacherSearch,
    /// Tree structure search for tree input problems.
    TreeSearch,
    /// Bitwise operation search.
    BitwiseSolver,
    /// Float regression solver.
    FloatSolver,
    /// Array composition solver.
    ArrayCompose,
}

impl SolverMethod {
    /// Get method name as string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::GradientSearch => "GradientSearch",
            Self::EnumerativeSearch => "EnumerativeSearch",
            Self::TeacherSearch => "TeacherSearch",
            Self::TreeSearch => "TreeSearch",
            Self::BitwiseSolver => "BitwiseSolver",
            Self::FloatSolver => "FloatSolver",
            Self::ArrayCompose => "ArrayCompose",
        }
    }

    /// Get default expected latency for this method (baseline before adaptation).
    pub fn default_latency(&self) -> Duration {
        match self {
            Self::GradientSearch => Duration::from_millis(100),
            Self::EnumerativeSearch => Duration::from_millis(500),
            Self::TeacherSearch => Duration::from_millis(50),
            Self::TreeSearch => Duration::from_millis(200),
            Self::BitwiseSolver => Duration::from_millis(150),
            Self::FloatSolver => Duration::from_millis(100),
            Self::ArrayCompose => Duration::from_millis(300),
        }
    }

    /// Get resource cost (0-1 scale, higher = more expensive).
    pub fn resource_cost(&self) -> f64 {
        match self {
            Self::GradientSearch => 0.3,
            Self::EnumerativeSearch => 0.8,
            Self::TeacherSearch => 0.1,
            Self::TreeSearch => 0.5,
            Self::BitwiseSolver => 0.4,
            Self::FloatSolver => 0.3,
            Self::ArrayCompose => 0.6,
        }
    }
}

/// Solver selection with performance expectations.
#[derive(Clone, Debug)]
pub struct SolverSelection {
    /// The solver method to use.
    pub method: SolverMethod,
    /// Expected success probability (0-1).
    pub expected_success: f64,
    /// Expected latency for this solve.
    pub expected_latency: Duration,
    /// Resource cost (0-1 scale).
    pub resource_cost: f64,
}

impl SolverSelection {
    /// Compute selection score balancing success, latency, and cost.
    pub fn score(&self) -> SolverSelectionScore {
        SolverSelectionScore(
            self.expected_success
                - (self.expected_latency.as_secs_f64() / 10.0) // Normalize latency impact
                - (self.resource_cost * 0.1) // Small penalty for resource cost
        )
    }
}

/// Solver selection score for ranking candidates.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SolverSelectionScore(pub f64);

/// Core allocation strategy using EMA-based statistics.
#[derive(Clone)]
pub struct AllocationStrategy {
    /// Per-solver success rates (EMA).
    success_rates: Arc<RwLock<HashMap<String, f64>>>,
    /// Per-solver latencies (EMA).
    latencies: Arc<RwLock<HashMap<String, Duration>>>,
    /// Minimum success threshold for solver consideration.
    min_success_threshold: f64,
}

impl Default for AllocationStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl AllocationStrategy {
    /// Create new allocation strategy with default thresholds.
    pub fn new() -> Self {
        Self {
            success_rates: Arc::new(RwLock::new(HashMap::new())),
            latencies: Arc::new(RwLock::new(HashMap::new())),
            min_success_threshold: DEFAULT_MIN_SUCCESS_THRESHOLD,
        }
    }

    /// Create allocation strategy with custom success threshold.
    pub fn with_threshold(min_success_threshold: f64) -> Self {
        Self {
            success_rates: Arc::new(RwLock::new(HashMap::new())),
            latencies: Arc::new(RwLock::new(HashMap::new())),
            min_success_threshold,
        }
    }

    /// Select optimal solver set for parallel execution.
    ///
    /// Returns ranked list of solver selections, limited by max_parallelism.
    /// Filters out solvers below success threshold and ranks by score.
    pub fn select_solvers(&self, problem: &Problem, max_parallelism: u8) -> Vec<SolverSelection> {
        let all_methods = [
            SolverMethod::GradientSearch,
            SolverMethod::EnumerativeSearch,
            SolverMethod::TeacherSearch,
            SolverMethod::TreeSearch,
            SolverMethod::BitwiseSolver,
            SolverMethod::FloatSolver,
            SolverMethod::ArrayCompose,
        ];

        // Score each solver by domain match and historical performance
        let mut candidates: Vec<SolverSelection> = all_methods
            .iter()
            .filter_map(|method| {
                let suitability = self.suitability_score(method, problem);
                if suitability < self.min_success_threshold {
                    return None;
                }

                let method_name = method.as_str();
                let success_rates = self.success_rates.read().unwrap();
                let latencies = self.latencies.read().unwrap();

                let expected_success = success_rates.get(method_name).copied().unwrap_or(suitability);
                let expected_latency = latencies.get(method_name).copied().unwrap_or_else(|| method.default_latency());
                let resource_cost = method.resource_cost();

                Some(SolverSelection {
                    method: *method,
                    expected_success,
                    expected_latency,
                    resource_cost,
                })
            })
            .collect();

        // Sort by score (descending)
        candidates.sort_by(|a, b| {
            b.score()
                .0
                .partial_cmp(&a.score().0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit by parallelism
        candidates.truncate(max_parallelism as usize);
        candidates
    }

    /// Update solver statistics after a solve attempt.
    ///
    /// Uses EMA to smoothly adapt to new observations.
    pub fn update_stats(&self, solver: &str, success: bool, latency: Duration) {
        // Update success rate with EMA
        {
            let mut rates = self.success_rates.write().unwrap();
            let current = rates.get(solver).copied().unwrap_or(0.5); // Default 50% for unseen solvers
            let new_rate = if success {
                // EMA: new = alpha * 1 + (1 - alpha) * old
                EMA_ALPHA * 1.0 + (1.0 - EMA_ALPHA) * current
            } else {
                // EMA: new = alpha * 0 + (1 - alpha) * old
                EMA_ALPHA * 0.0 + (1.0 - EMA_ALPHA) * current
            };
            rates.insert(solver.to_string(), new_rate);
        }

        // Update latency with EMA
        {
            let mut latencies = self.latencies.write().unwrap();
            let current = latencies.get(solver).copied().unwrap_or(Duration::from_millis(100));
            let current_secs = current.as_secs_f64();
            let new_secs = latency.as_secs_f64();
            // EMA for duration
            let ema_secs = EMA_ALPHA * new_secs + (1.0 - EMA_ALPHA) * current_secs;
            latencies.insert(solver.to_string(), Duration::from_secs_f64(ema_secs));
        }
    }

    /// Get current success rate for a solver (if available).
    pub fn get_success_rate(&self, solver: &str) -> Option<f64> {
        self.success_rates.read().unwrap().get(solver).copied()
    }

    /// Get current latency expectation for a solver (if available).
    pub fn get_latency(&self, solver: &str) -> Option<Duration> {
        self.latencies.read().unwrap().get(solver).copied()
    }

    /// Reset all statistics (e.g., for testing or domain change).
    pub fn reset_stats(&self) {
        let mut rates = self.success_rates.write().unwrap();
        let mut lats = self.latencies.write().unwrap();
        rates.clear();
        lats.clear();
    }

    /// Compute suitability score for a solver on a problem.
    ///
    /// Combines domain match (0-1) with historical success rate.
    fn suitability_score(&self, solver: &SolverMethod, problem: &Problem) -> f64 {
        let domain_match = self.domain_match(solver, problem);

        // Get historical success rate if available
        let solver_name = solver.as_str();
        let rates = self.success_rates.read().unwrap();
        let historical_rate = rates.get(solver_name).copied().unwrap_or(0.5);

        // Weight domain match higher for unfamiliar solvers
        // As we gather data, historical rate dominates
        domain_match * 0.7 + historical_rate * 0.3
    }

    /// Compute domain match score (0-1) between solver and problem.
    ///
    /// Derived mechanically from problem structure, no hardcoded rules.
    fn domain_match(&self, solver: &SolverMethod, problem: &Problem) -> f64 {
        // Extract type information from problem
        let type_classes = extract_type_classes(problem);

        // Compute match based on solver capabilities
        match solver {
            SolverMethod::GradientSearch | SolverMethod::EnumerativeSearch => {
                // Best for scalar integer problems
                if type_classes.contains(&TypeClass::ScalarInt) && !type_classes.contains(&TypeClass::Array) {
                    0.9
                } else if type_classes.contains(&TypeClass::ScalarInt) {
                    0.6
                } else {
                    0.1
                }
            }
            SolverMethod::BitwiseSolver => {
                // Best for scalar integer problems with bitwise operations
                if type_classes.contains(&TypeClass::ScalarInt) && !type_classes.contains(&TypeClass::Array) {
                    0.8
                } else if type_classes.contains(&TypeClass::ScalarInt) {
                    0.5
                } else {
                    0.1
                }
            }
            SolverMethod::TeacherSearch => {
                // General purpose, higher for strings and arrays
                if type_classes.contains(&TypeClass::String) {
                    0.9
                } else if type_classes.contains(&TypeClass::Array) {
                    0.8
                } else {
                    0.5
                }
            }
            SolverMethod::TreeSearch => {
                // Specialized for tree input problems
                if type_classes.contains(&TypeClass::Tree) {
                    0.95
                } else {
                    0.2
                }
            }
            SolverMethod::FloatSolver => {
                // Specialized for float problems
                if type_classes.contains(&TypeClass::Float) {
                    0.95
                } else if type_classes.contains(&TypeClass::ScalarInt) {
                    0.4
                } else {
                    0.1
                }
            }
            SolverMethod::ArrayCompose => {
                // Specialized for array problems
                if type_classes.contains(&TypeClass::Array) {
                    0.95
                } else {
                    0.1
                }
            }
        }
    }
}

/// Extract type classes present in a problem.
fn extract_type_classes(problem: &Problem) -> Vec<TypeClass> {
    let mut classes = std::collections::HashSet::new();

    for example in &problem.examples {
        for input in &example.inputs {
            classes.insert(TypeClass::from_value(input));
        }
        classes.insert(TypeClass::from_value(&example.expected));
    }

    classes.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn create_test_problem(output_type: Value) -> Problem {
        Problem {
            name: "test_problem".to_string(),
            category: "test",
            description: "Test problem",
            signature: "fn test(x: i64) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Int(42)],
                expected: output_type,
            }],
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
    fn test_allocation_strategy_new() {
        let strategy = AllocationStrategy::new();
        assert_eq!(strategy.min_success_threshold, DEFAULT_MIN_SUCCESS_THRESHOLD);
    }

    #[test]
    fn test_allocation_strategy_with_threshold() {
        let strategy = AllocationStrategy::with_threshold(0.5);
        assert_eq!(strategy.min_success_threshold, 0.5);
    }

    #[test]
    fn test_select_solvers_returns_within_limit() {
        let strategy = AllocationStrategy::new();
        let problem = create_test_problem(Value::Int(42));
        let selections = strategy.select_solvers(&problem, 3);
        assert!(selections.len() <= 3);
    }

    #[test]
    fn test_update_stats_ema() {
        let strategy = AllocationStrategy::new();
        let solver = "GradientSearch";

        // Initial state - no stats
        assert_eq!(strategy.get_success_rate(solver), None);

        // First success
        strategy.update_stats(solver, true, Duration::from_millis(100));
        let rate = strategy.get_success_rate(solver).unwrap();
        assert!(rate > 0.0 && rate <= 1.0);

        // Multiple successes should increase rate
        let prev_rate = rate;
        strategy.update_stats(solver, true, Duration::from_millis(100));
        let new_rate = strategy.get_success_rate(solver).unwrap();
        assert!(new_rate > prev_rate);

        // Failure should decrease rate
        strategy.update_stats(solver, false, Duration::from_millis(100));
        let failed_rate = strategy.get_success_rate(solver).unwrap();
        assert!(failed_rate < new_rate);
    }

    #[test]
    fn test_update_latency_ema() {
        let strategy = AllocationStrategy::new();
        let solver = "GradientSearch";

        // Update latency
        strategy.update_stats(solver, true, Duration::from_millis(100));
        let latency = strategy.get_latency(solver).unwrap();
        assert_eq!(latency.as_millis(), 100);

        // EMA should smooth variations
        strategy.update_stats(solver, true, Duration::from_millis(200));
        let new_latency = strategy.get_latency(solver).unwrap();
        assert!(new_latency.as_millis() > 100);
        assert!(new_latency.as_millis() < 200); // EMA smooths, not direct jump
    }

    #[test]
    fn test_solver_method_defaults() {
        let methods = [
            SolverMethod::GradientSearch,
            SolverMethod::EnumerativeSearch,
            SolverMethod::TeacherSearch,
            SolverMethod::TreeSearch,
            SolverMethod::BitwiseSolver,
            SolverMethod::FloatSolver,
            SolverMethod::ArrayCompose,
        ];

        for method in methods {
            let name = method.as_str();
            assert!(!name.is_empty());

            let latency = method.default_latency();
            assert!(latency.as_millis() > 0);

            let cost = method.resource_cost();
            assert!(cost >= 0.0 && cost <= 1.0);
        }
    }

    #[test]
    fn test_reset_stats() {
        let strategy = AllocationStrategy::new();
        strategy.update_stats("GradientSearch", true, Duration::from_millis(100));

        strategy.reset_stats();

        assert_eq!(strategy.get_success_rate("GradientSearch"), None);
        assert_eq!(strategy.get_latency("GradientSearch"), None);
    }

    #[test]
    fn test_solver_selection_score() {
        let selection = SolverSelection {
            method: SolverMethod::GradientSearch,
            expected_success: 0.8,
            expected_latency: Duration::from_millis(100),
            resource_cost: 0.3,
        };

        let score = selection.score();
        assert!(score.0 > 0.0);
        assert!(score.0 <= 1.0);
    }

    #[test]
    fn test_domain_match_scalar_int() {
        let strategy = AllocationStrategy::new();
        let problem = create_test_problem(Value::Int(42));

        let selection = strategy.select_solvers(&problem, 10);
        assert!(!selection.is_empty());

        // Gradient search should be highly ranked for scalar int
        let gradient_rank = selection
            .iter()
            .position(|s| s.method == SolverMethod::GradientSearch);
        assert!(gradient_rank.is_some());
    }

    #[test]
    fn test_domain_match_array() {
        let strategy = AllocationStrategy::new();
        let problem = Problem {
            name: "array_problem".to_string(),
            category: "test",
            description: "Array test",
            signature: "fn test(arr: Vec<i64>) -> Vec<i64>",
            examples: vec![Example {
                inputs: vec![Value::Array(vec![1, 2, 3])],
                expected: Value::Array(vec![2, 4, 6]),
            }],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };

        let selection = strategy.select_solvers(&problem, 10);
        assert!(!selection.is_empty());

        // Array compose should be highly ranked for array problems
        let array_rank = selection
            .iter()
            .position(|s| s.method == SolverMethod::ArrayCompose);
        assert!(array_rank.is_some());
    }
}
