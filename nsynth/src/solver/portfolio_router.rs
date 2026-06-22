//! Portfolio router for parallel solver execution with adaptive timeouts.
//!
//! Coordinates multiple solver strategies in parallel, intelligently allocating
//! resources based on problem characteristics and historical performance data.
//! Implements portfolio optimization theory to maximize success probability
//! while minimizing resource usage and latency.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::benchmark::Problem;
use crate::solver::enhanced_integration::EnhancedSolver;
use crate::solver::allocation_strategy::{AllocationStrategy, SolverSelection};
use crate::solver::parallel_executor::{
    AdaptiveTimeout, ExecutionResult, ExecutionTask, ParallelExecutor,
};
use crate::solver::SolveResult;

/// Global flag to enable/disable portfolio router at runtime.
/// Set via CLI flag --portfolio or programmatically.
static PORTFOLIO_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable portfolio router globally.
pub fn enable_portfolio_router() {
    PORTFOLIO_ENABLED.store(true, Ordering::SeqCst);
}

/// Disable portfolio router globally.
pub fn disable_portfolio_router() {
    PORTFOLIO_ENABLED.store(false, Ordering::SeqCst);
}

/// Check if portfolio router is enabled.
pub fn portfolio_router_enabled() -> bool {
    PORTFOLIO_ENABLED.load(Ordering::SeqCst)
}

/// Default timeout configuration for solver execution.
#[derive(Clone, Debug)]
pub struct TimeoutConfig {
    /// Base timeout for simple problems
    pub base_timeout: Duration,
    /// Multiplier for problem complexity
    pub complexity_multiplier: f64,
    /// Maximum timeout cap
    pub max_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            base_timeout: Duration::from_secs(30),
            complexity_multiplier: 1.5,
            max_timeout: Duration::from_secs(300),
        }
    }
}

impl TimeoutConfig {
    /// Create new timeout config with custom values.
    pub fn new(base_timeout: Duration, complexity_multiplier: f64, max_timeout: Duration) -> Self {
        Self {
            base_timeout,
            complexity_multiplier,
            max_timeout,
        }
    }

    /// Create with custom base timeout.
    pub fn with_base_timeout(base_timeout: Duration) -> Self {
        Self {
            base_timeout,
            ..Default::default()
        }
    }

    /// Create with custom max timeout.
    pub fn with_max_timeout(max_timeout: Duration) -> Self {
        Self {
            max_timeout,
            ..Default::default()
        }
    }
}

/// Portfolio allocation result specifying which solvers to run.
#[derive(Clone, Debug)]
pub struct PortfolioAllocation {
    /// Selected solvers with their performance expectations
    pub solvers: Vec<SolverSelection>,
    /// Computed timeouts for each solver
    pub timeouts: Vec<Duration>,
    /// Maximum parallelism for this allocation
    pub parallelism: u8,
}

impl PortfolioAllocation {
    /// Create empty allocation.
    pub fn empty() -> Self {
        Self {
            solvers: Vec::new(),
            timeouts: Vec::new(),
            parallelism: 0,
        }
    }

    /// Check if allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.solvers.is_empty()
    }

    /// Get number of solvers in allocation.
    pub fn len(&self) -> usize {
        self.solvers.len()
    }

    /// Create execution tasks from this allocation.
    pub fn to_tasks(&self, problem: &Problem) -> Vec<ExecutionTask> {
        self.solvers
            .iter()
            .zip(self.timeouts.iter())
            .map(|(selection, timeout)| {
                ExecutionTask::new(selection.method, problem.clone(), *timeout)
            })
            .collect()
    }
}

/// Result from portfolio execution with best candidate selection.
#[derive(Clone, Debug)]
pub struct AllocationResult {
    /// Individual solver results
    pub results: Vec<ExecutionResult>,
    /// Best candidate selected by Pareto optimization
    pub best_candidate: Option<SolveResult>,
    /// Total execution time
    pub total_duration: Duration,
}

impl AllocationResult {
    /// Create empty result.
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            best_candidate: None,
            total_duration: Duration::ZERO,
        }
    }

    /// Check if any solver succeeded.
    pub fn has_success(&self) -> bool {
        self.results.iter().any(|r| r.is_success())
    }

    /// Get count of successful solvers.
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_success()).count()
    }

    /// Get count of timed out solvers.
    pub fn timeout_count(&self) -> usize {
        self.results.iter().filter(|r| r.timed_out).count()
    }
}

/// Portfolio router coordinating parallel solver execution.
///
/// Uses portfolio optimization theory to intelligently allocate resources
/// across multiple solver strategies based on problem characteristics and
/// historical performance data.
pub struct PortfolioRouter {
    /// Parallel execution engine
    executor: Arc<ParallelExecutor>,
    /// Allocation strategy for solver selection
    allocator: Arc<AllocationStrategy>,
    /// Timeout configuration
    timeout_config: TimeoutConfig,
    /// Enhanced solver for Pareto-optimal selection
    enhanced_solver: Arc<EnhancedSolver>,
}

impl Default for PortfolioRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl PortfolioRouter {
    /// Create new portfolio router with default configuration.
    pub fn new() -> Self {
        Self {
            executor: Arc::new(
                ParallelExecutor::new()
                    .with_timeout_enforcement(crate::solver::parallel_executor::TimeoutEnforcement::Adaptive)
                    .with_adaptive_timeout(AdaptiveTimeout::new())
                    .with_base_timeout(Duration::from_secs(30))
            ),
            allocator: Arc::new(AllocationStrategy::new()),
            timeout_config: TimeoutConfig::default(),
            enhanced_solver: Arc::new(EnhancedSolver::new()),
        }
    }

    /// Create with custom timeout configuration.
    pub fn with_timeout_config(mut self, config: TimeoutConfig) -> Self {
        // Extract base timeout before moving config
        let base_timeout = config.base_timeout;
        self.timeout_config = config;
        // Update executor base timeout
        let executor = Arc::unwrap_or_clone(Arc::clone(&self.executor))
            .with_base_timeout(base_timeout);
        self.executor = Arc::new(executor);
        self
    }

    /// Create with custom parallel executor.
    pub fn with_executor(mut self, executor: ParallelExecutor) -> Self {
        self.executor = Arc::new(executor);
        self
    }

    /// Create with custom allocation strategy.
    pub fn with_allocator(mut self, allocator: AllocationStrategy) -> Self {
        self.allocator = Arc::new(allocator);
        self
    }

    /// Get global portfolio router instance (lazy initialization).
    /// Ported from `once_cell::sync::Lazy` to the std `LazyLock` (no `once_cell`
    /// dependency in the canonical crate).
    pub fn global() -> &'static Self {
        use std::sync::LazyLock;
        static GLOBAL_ROUTER: LazyLock<PortfolioRouter> = LazyLock::new(PortfolioRouter::new);
        &GLOBAL_ROUTER
    }

    /// Route a problem to the optimal solver portfolio.
    ///
    /// Returns None if allocation fails or no solvers are available.
    /// Otherwise returns the best solution found using Pareto optimization.
    pub fn route(&self, problem: &Problem) -> Option<SolveResult> {
        // Compute optimal allocation
        let allocation = self.compute_allocation(problem);
        if allocation.is_empty() {
            return None;
        }

        // Execute portfolio
        let execution_result = self.execute_portfolio(problem, &allocation);
        if execution_result.results.is_empty() {
            return None;
        }

        // Update allocation stats with outcomes
        self.update_allocation_stats(&execution_result);

        // Return best candidate selected by Pareto optimization
        execution_result.best_candidate
    }

    /// Compute optimal solver allocation for a problem.
    ///
    /// Analyzes problem characteristics and selects the best solver portfolio
    /// with appropriate timeouts based on complexity and historical data.
    fn compute_allocation(&self, problem: &Problem) -> PortfolioAllocation {
        // Determine max parallelism based on problem complexity
        let parallelism = self.compute_parallelism(problem);

        // Select solvers using allocation strategy
        let solvers = self.allocator.select_solvers(problem, parallelism);
        if solvers.is_empty() {
            return PortfolioAllocation::empty();
        }

        // Compute timeouts for each solver
        let timeouts: Vec<Duration> = solvers
            .iter()
            .map(|selection| self.compute_timeout(&selection.method, problem))
            .collect();

        PortfolioAllocation {
            solvers,
            timeouts,
            parallelism,
        }
    }

    /// Execute solver portfolio and collect results.
    ///
    /// Runs selected solvers in parallel with adaptive timeouts,
    /// then applies Pareto optimization to select the best solution.
    fn execute_portfolio(&self, problem: &Problem, allocation: &PortfolioAllocation) -> AllocationResult {
        let start = std::time::Instant::now();

        // Convert allocation to execution tasks
        let tasks = allocation.to_tasks(problem);

        // Execute in parallel
        let results = if tasks.len() <= 1 {
            // Single task - execute directly
            tasks
                .into_iter()
                .map(|task| {
                    self.executor.execute_solver(task.solver, &task.problem, task.timeout)
                })
                .collect()
        } else {
            // Multiple tasks - parallel execution
            self.executor.execute_parallel(tasks)
        };

        let total_duration = start.elapsed();

        // Select best solution using Pareto optimization
        let successful_results: Vec<SolveResult> = results
            .iter()
            .filter_map(|r| r.result.clone())
            .collect();

        let best_candidate = if !successful_results.is_empty() {
            self.enhanced_solver.select_best_solution(successful_results)
        } else {
            None
        };

        AllocationResult {
            results,
            best_candidate,
            total_duration,
        }
    }

    /// Compute timeout for a solver on a specific problem.
    ///
    /// Timeout is computed as:
    /// base_timeout × complexity_factor × solver_multiplier
    /// Capped at max_timeout
    fn compute_timeout(&self, solver: &crate::solver::allocation_strategy::SolverMethod, problem: &Problem) -> Duration {
        // Start with base timeout
        let mut timeout_secs = self.timeout_config.base_timeout.as_secs_f64();

        // Apply complexity factor based on problem characteristics
        let complexity_factor = self.compute_complexity_factor(problem);
        timeout_secs *= complexity_factor;

        // Apply solver-specific multiplier
        let solver_multiplier = self.solver_timeout_multiplier(solver);
        timeout_secs *= solver_multiplier;

        // Apply complexity multiplier from config
        timeout_secs *= self.timeout_config.complexity_multiplier;

        // Cap at max timeout
        let timeout_secs = timeout_secs.min(self.timeout_config.max_timeout.as_secs_f64());

        Duration::from_secs_f64(timeout_secs)
    }

    /// Compute problem complexity factor (1.0 = simple, higher = more complex).
    fn compute_complexity_factor(&self, problem: &Problem) -> f64 {
        let mut factor = 1.0;

        // Factor in arity (more inputs = more complex)
        let max_arity = problem.examples
            .iter()
            .map(|e| e.inputs.len())
            .max()
            .unwrap_or(0);
        factor *= 1.0 + (max_arity as f64 * 0.1);

        // Factor in example count (more examples = more validation needed)
        let example_count = problem.examples.len();
        if example_count > 5 {
            factor *= 1.0 + ((example_count - 5) as f64 * 0.05);
        }

        // Factor in problem size (code length indicator)
        if problem.reference_code.len() > 500 {
            factor *= 1.2;
        }

        // Recursive problems are more complex
        if problem.recursive_allowed {
            factor *= 1.5;
        }

        // Tree input problems are more complex
        if problem.tree_input {
            factor *= 1.3;
        }

        factor
    }

    /// Get solver-specific timeout multiplier.
    ///
    /// Search-based solvers get more time, enumerative get less.
    fn solver_timeout_multiplier(&self, solver: &crate::solver::allocation_strategy::SolverMethod) -> f64 {
        match solver {
            crate::solver::allocation_strategy::SolverMethod::GradientSearch => 2.0,
            crate::solver::allocation_strategy::SolverMethod::EnumerativeSearch => 1.0,
            crate::solver::allocation_strategy::SolverMethod::TeacherSearch => 1.5,
            crate::solver::allocation_strategy::SolverMethod::TreeSearch => 2.0,
            crate::solver::allocation_strategy::SolverMethod::BitwiseSolver => 1.5,
            crate::solver::allocation_strategy::SolverMethod::FloatSolver => 1.5,
            crate::solver::allocation_strategy::SolverMethod::ArrayCompose => 1.8,
        }
    }

    /// Compute maximum parallelism for a problem.
    ///
    /// More complex problems get more parallelism.
    fn compute_parallelism(&self, problem: &Problem) -> u8 {
        let base_parallelism = 2u8;

        // Increase parallelism for complex problems
        if problem.recursive_allowed || problem.tree_input {
            base_parallelism + 2
        } else if problem.examples.len() > 5 {
            base_parallelism + 1
        } else {
            base_parallelism
        }.min(4) // Cap at 4 for resource management
    }

    /// Update allocation statistics with execution outcomes.
    ///
    /// Records success/failure and timing data for adaptive improvement.
    fn update_allocation_stats(&self, result: &AllocationResult) {
        for execution_result in &result.results {
            let solver_name = execution_result.solver.as_str();
            let success = execution_result.is_success();
            let duration = execution_result.duration;

            // Update allocation strategy stats
            self.allocator.update_stats(solver_name, success, duration);

            // Update timeout stats
            self.executor.update_timeout_stats(solver_name, duration);
        }
    }

    /// Get current timeout configuration.
    pub fn timeout_config(&self) -> &TimeoutConfig {
        &self.timeout_config
    }

    /// Get reference to allocation strategy for inspection.
    pub fn allocator(&self) -> &AllocationStrategy {
        &self.allocator
    }

    /// Get reference to parallel executor for inspection.
    pub fn executor(&self) -> &ParallelExecutor {
        &self.executor
    }

    /// Get reference to enhanced solver for inspection.
    pub fn enhanced_solver(&self) -> &EnhancedSolver {
        &self.enhanced_solver
    }

    /// Reset all learned statistics (success rates, timeouts, etc.).
    pub fn reset_stats(&self) {
        self.allocator.reset_stats();
        self.executor.reset_timeout_stats();
    }

    /// Route with fallback to sequential execution on failure.
    ///
    /// If parallel execution fails, falls back to sequential execution
    /// of solvers in order of expected success.
    pub fn route_with_fallback(&self, problem: &Problem) -> Option<SolveResult> {
        match self.route(problem) {
            Some(result) => Some(result),
            None => {
                // Fallback: try solvers sequentially
                let allocation = self.compute_allocation(problem);
                for (solver, timeout) in allocation.solvers.iter().zip(allocation.timeouts.iter()) {
                    let result = self.executor.execute_solver(solver.method, problem, *timeout);
                    if result.is_success() {
                        return result.result;
                    }
                }
                None
            }
        }
    }
}

// Helper trait for string access on SolverMethod
trait SolverMethodName {
    fn as_str(&self) -> &'static str;
}

impl SolverMethodName for crate::solver::allocation_strategy::SolverMethod {
    fn as_str(&self) -> &'static str {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn create_test_problem(arity: usize, recursive: bool) -> Problem {
        let inputs: Vec<Value> = (0..arity).map(|_| Value::Int(42)).collect();
        Problem {
            name: "test_problem".to_string(),
            category: "test",
            description: "Test problem",
            signature: "fn test(x: i64) -> i64",
            examples: vec![Example {
                inputs,
                expected: Value::Int(84),
            }],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: recursive,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn test_timeout_config_default() {
        let config = TimeoutConfig::default();
        assert_eq!(config.base_timeout, Duration::from_secs(30));
        assert_eq!(config.complexity_multiplier, 1.5);
        assert_eq!(config.max_timeout, Duration::from_secs(300));
    }

    #[test]
    fn test_timeout_config_custom() {
        let config = TimeoutConfig::new(
            Duration::from_secs(60),
            2.0,
            Duration::from_secs(600),
        );
        assert_eq!(config.base_timeout, Duration::from_secs(60));
        assert_eq!(config.complexity_multiplier, 2.0);
        assert_eq!(config.max_timeout, Duration::from_secs(600));
    }

    #[test]
    fn test_portfolio_allocation_empty() {
        let allocation = PortfolioAllocation::empty();
        assert!(allocation.is_empty());
        assert_eq!(allocation.len(), 0);
    }

    #[test]
    fn test_allocation_result_empty() {
        let result = AllocationResult::empty();
        assert!(!result.has_success());
        assert_eq!(result.success_count(), 0);
    }

    #[test]
    fn test_portfolio_router_new() {
        let router = PortfolioRouter::new();
        assert_eq!(router.timeout_config.base_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_portfolio_router_with_timeout_config() {
        let config = TimeoutConfig::with_base_timeout(Duration::from_secs(60));
        let router = PortfolioRouter::new().with_timeout_config(config);
        assert_eq!(router.timeout_config.base_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_compute_complexity_factor() {
        let router = PortfolioRouter::new();

        // Simple problem
        let simple = create_test_problem(1, false);
        let simple_factor = router.compute_complexity_factor(&simple);
        assert!(simple_factor >= 1.0);

        // Complex problem
        let complex = create_test_problem(3, true);
        let complex_factor = router.compute_complexity_factor(&complex);
        assert!(complex_factor > simple_factor);
    }

    #[test]
    fn test_compute_timeout() {
        let router = PortfolioRouter::new();
        let problem = create_test_problem(1, false);

        let timeout = router.compute_timeout(
            &crate::solver::allocation_strategy::SolverMethod::GradientSearch,
            &problem,
        );

        assert!(timeout >= Duration::from_secs(30));
        assert!(timeout <= Duration::from_secs(300));
    }

    #[test]
    fn test_solver_timeout_multiplier() {
        let router = PortfolioRouter::new();

        let gradient_mult = router.solver_timeout_multiplier(
            &crate::solver::allocation_strategy::SolverMethod::GradientSearch
        );
        assert_eq!(gradient_mult, 2.0);

        let enumerative_mult = router.solver_timeout_multiplier(
            &crate::solver::allocation_strategy::SolverMethod::EnumerativeSearch
        );
        assert_eq!(enumerative_mult, 1.0);
    }

    #[test]
    fn test_compute_parallelism() {
        let router = PortfolioRouter::new();

        // Simple problem
        let simple = create_test_problem(1, false);
        let simple_parallelism = router.compute_parallelism(&simple);
        assert!(simple_parallelism >= 2);

        // Complex problem
        let complex = create_test_problem(1, true);
        let complex_parallelism = router.compute_parallelism(&complex);
        assert!(complex_parallelism >= simple_parallelism);
    }

    #[test]
    fn test_compute_allocation() {
        let router = PortfolioRouter::new();
        let problem = create_test_problem(1, false);

        let allocation = router.compute_allocation(&problem);
        assert!(!allocation.is_empty());
        assert_eq!(allocation.solvers.len(), allocation.timeouts.len());
    }

    #[test]
    fn test_global_router() {
        let router = PortfolioRouter::global();
        assert_eq!(router.timeout_config.base_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_reset_stats() {
        let router = PortfolioRouter::new();
        router.allocator.update_stats("test", true, Duration::from_millis(100));

        router.reset_stats();

        assert_eq!(router.allocator.get_success_rate("test"), None);
    }

    #[test]
    fn test_allocation_to_tasks() {
        let router = PortfolioRouter::new();
        let problem = create_test_problem(1, false);
        let allocation = router.compute_allocation(&problem);

        let tasks = allocation.to_tasks(&problem);
        assert_eq!(tasks.len(), allocation.solvers.len());
    }
}
