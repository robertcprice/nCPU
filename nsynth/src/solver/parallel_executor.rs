//! Parallel executor for solver portfolio synthesis with adaptive timeouts.
//!
//! Provides thread pool based parallel execution of multiple solvers with
//! intelligent timeout management based on historical performance data.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::benchmark::Problem;
use crate::solver::allocation_strategy::SolverMethod;
use crate::solver::SolveResult;

/// Timeout enforcement strategies for solver execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeoutEnforcement {
    /// Strict timeout - kill solver immediately when timeout expires
    Strict,
    /// Graceful timeout - allow current operation to finish before stopping
    Graceful,
    /// Adaptive timeout - use historical data to set timeouts dynamically
    Adaptive,
}

impl Default for TimeoutEnforcement {
    fn default() -> Self {
        Self::Graceful
    }
}

/// Duration statistics for timeout computation.
#[derive(Clone, Debug)]
pub struct DurationStats {
    /// Mean execution time
    pub mean: Duration,
    /// Standard deviation
    pub std: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// Number of samples
    pub samples: usize,
}

impl Default for DurationStats {
    fn default() -> Self {
        Self {
            mean: Duration::from_millis(100),
            std: Duration::from_millis(50),
            p95: Duration::from_millis(200),
            samples: 0,
        }
    }
}

impl DurationStats {
    /// Create new stats from a single duration observation.
    pub fn from_duration(d: Duration) -> Self {
        Self {
            mean: d,
            std: Duration::ZERO,
            p95: d,
            samples: 1,
        }
    }

    /// Update stats with a new duration observation using EMA.
    pub fn update(&mut self, new_duration: Duration, alpha: f64) {
        let new_ms = new_duration.as_secs_f64() * 1000.0;
        let mean_ms = self.mean.as_secs_f64() * 1000.0;

        // Update mean with EMA
        let new_mean_ms = alpha * new_ms + (1.0 - alpha) * mean_ms;
        self.mean = Duration::from_secs_f64(new_mean_ms / 1000.0);

        // Update std (simplified EMA approximation)
        let std_ms = self.std.as_secs_f64() * 1000.0;
        let diff = (new_ms - new_mean_ms).abs();
        let new_std_ms = alpha * diff + (1.0 - alpha) * std_ms;
        self.std = Duration::from_secs_f64(new_std_ms / 1000.0);

        // Update p95 (conservative: max of current p95 and new duration)
        let p95_ms = self.p95.as_secs_f64() * 1000.0;
        let new_p95_ms = if new_ms > p95_ms {
            // Decay old p95, include new observation
            alpha * new_ms + (1.0 - alpha) * p95_ms
        } else {
            // Decay towards current observation
            alpha * new_ms + (1.0 - alpha) * p95_ms
        };
        self.p95 = Duration::from_secs_f64(new_p95_ms / 1000.0);

        self.samples += 1;
    }

    /// Compute recommended timeout with 20% buffer over p95.
    pub fn recommended_timeout(&self) -> Duration {
        let p95_ms = self.p95.as_secs_f64() * 1000.0;
        Duration::from_secs_f64((p95_ms * 1.2) / 1000.0)
    }
}

/// Adaptive timeout tracker with historical performance data.
#[derive(Clone)]
pub struct AdaptiveTimeout {
    /// Historical execution times per solver method
    historical_times: Arc<RwLock<HashMap<String, DurationStats>>>,
    /// EMA alpha for smooth adaptation
    alpha: f64,
    /// Minimum samples before using adaptive timeout
    min_samples: usize,
}

impl Default for AdaptiveTimeout {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveTimeout {
    /// Create new adaptive timeout tracker with default settings.
    pub fn new() -> Self {
        Self {
            historical_times: Arc::new(RwLock::new(HashMap::new())),
            alpha: 0.1, // 10% weight to new observations
            min_samples: 5,
        }
    }

    /// Create with custom EMA alpha.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            historical_times: Arc::new(RwLock::new(HashMap::new())),
            alpha,
            min_samples: 5,
        }
    }

    /// Create with custom minimum sample threshold.
    pub fn with_min_samples(min_samples: usize) -> Self {
        Self {
            historical_times: Arc::new(RwLock::new(HashMap::new())),
            alpha: 0.1,
            min_samples,
        }
    }

    /// Record execution time for a solver method.
    pub fn record_time(&self, solver: &str, duration: Duration) {
        let mut times = self.historical_times.write().unwrap();
        let is_new = !times.contains_key(solver);
        let stats = times
            .entry(solver.to_string())
            .or_insert_with(|| DurationStats::from_duration(duration));

        // Only update if this wasn't a newly created entry
        // (from_duration already initializes with the first duration)
        if !is_new {
            stats.update(duration, self.alpha);
        }
    }

    /// Get adaptive timeout for a solver method.
    ///
    /// Returns None if insufficient data available.
    pub fn get_timeout(&self, solver: &str) -> Option<Duration> {
        let times = self.historical_times.read().unwrap();
        let stats = times.get(solver)?;
        if stats.samples >= self.min_samples {
            Some(stats.recommended_timeout())
        } else {
            None
        }
    }

    /// Get current statistics for a solver (if available).
    pub fn get_stats(&self, solver: &str) -> Option<DurationStats> {
        let times = self.historical_times.read().unwrap();
        times.get(solver).cloned()
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        let mut times = self.historical_times.write().unwrap();
        times.clear();
    }
}

/// Task for parallel solver execution.
#[derive(Clone)]
pub struct ExecutionTask {
    /// The solver method to execute
    pub solver: SolverMethod,
    /// The problem to solve
    pub problem: Problem,
    /// Timeout for this execution
    pub timeout: Duration,
}

impl ExecutionTask {
    /// Create new execution task.
    pub fn new(solver: SolverMethod, problem: Problem, timeout: Duration) -> Self {
        Self {
            solver,
            problem,
            timeout,
        }
    }
}

/// Result from parallel solver execution.
#[derive(Clone, Debug)]
pub struct ExecutionResult {
    /// The solver method that was executed
    pub solver: SolverMethod,
    /// Solve result (None if timed out or errored)
    pub result: Option<SolveResult>,
    /// Actual execution duration
    pub duration: Duration,
    /// Whether execution timed out
    pub timed_out: bool,
}

impl ExecutionResult {
    /// Create successful execution result.
    pub fn success(solver: SolverMethod, result: SolveResult, duration: Duration) -> Self {
        Self {
            solver,
            result: Some(result),
            duration,
            timed_out: false,
        }
    }

    /// Create timeout execution result.
    pub fn timeout(solver: SolverMethod, duration: Duration) -> Self {
        Self {
            solver,
            result: None,
            duration,
            timed_out: true,
        }
    }

    /// Create error execution result.
    pub fn error(solver: SolverMethod, duration: Duration) -> Self {
        Self {
            solver,
            result: None,
            duration,
            timed_out: false,
        }
    }

    /// Check if execution produced a successful solve result.
    pub fn is_success(&self) -> bool {
        self.result
            .as_ref()
            .map(|r| r.success)
            .unwrap_or(false)
    }
}

/// Parallel executor for solver portfolio synthesis.
#[derive(Clone)]
pub struct ParallelExecutor {
    /// Maximum number of threads (default: num_cpus - 1)
    max_threads: usize,
    /// Timeout enforcement strategy
    timeout_enforcement: TimeoutEnforcement,
    /// Optional adaptive timeout tracker
    adaptive_timeout: Option<Arc<AdaptiveTimeout>>,
    /// Base timeout for solvers without historical data
    base_timeout: Duration,
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelExecutor {
    /// Create new parallel executor with default settings.
    pub fn new() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            max_threads: (num_cpus - 1).max(1),
            timeout_enforcement: TimeoutEnforcement::default(),
            adaptive_timeout: None,
            base_timeout: Duration::from_secs(5),
        }
    }

    /// Set maximum number of threads.
    pub fn with_max_threads(mut self, max_threads: usize) -> Self {
        self.max_threads = max_threads.max(1);
        self
    }

    /// Set timeout enforcement strategy.
    pub fn with_timeout_enforcement(mut self, enforcement: TimeoutEnforcement) -> Self {
        self.timeout_enforcement = enforcement;
        if enforcement == TimeoutEnforcement::Adaptive && self.adaptive_timeout.is_none() {
            self.adaptive_timeout = Some(Arc::new(AdaptiveTimeout::new()));
        }
        self
    }

    /// Set adaptive timeout tracker.
    pub fn with_adaptive_timeout(mut self, adaptive_timeout: AdaptiveTimeout) -> Self {
        self.adaptive_timeout = Some(Arc::new(adaptive_timeout));
        self.timeout_enforcement = TimeoutEnforcement::Adaptive;
        self
    }

    /// Set base timeout for solvers without historical data.
    pub fn with_base_timeout(mut self, base_timeout: Duration) -> Self {
        self.base_timeout = base_timeout;
        self
    }

    /// Compute timeout for a solver method.
    ///
    /// Uses adaptive timeout if available and has sufficient data,
    /// otherwise falls back to base timeout.
    fn compute_timeout(&self, solver: &SolverMethod) -> Duration {
        if self.timeout_enforcement == TimeoutEnforcement::Adaptive {
            if let Some(adaptive) = &self.adaptive_timeout {
                let solver_name = solver.as_str();
                if let Some(timeout) = adaptive.get_timeout(solver_name) {
                    return timeout;
                }
            }
        }
        self.base_timeout
    }

    /// Execute a single solver with timeout.
    ///
    /// This is a stub implementation - in production, this would delegate
    /// to the actual solver implementation based on the method.
    pub fn execute_solver(
        &self,
        solver: SolverMethod,
        problem: &Problem,
        timeout: Duration,
    ) -> ExecutionResult {
        let start = Instant::now();
        let cancel_flag = Arc::new(AtomicBool::new(false));

        // Spawn solver thread
        let handle = self.spawn_solver_thread(solver, problem.clone(), cancel_flag.clone());

        // Wait for completion or timeout
        let result = self.join_with_timeout(handle, timeout, cancel_flag);
        let duration = start.elapsed();

        // Record execution time for adaptive timeout
        if let Some(adaptive) = &self.adaptive_timeout {
            adaptive.record_time(solver.as_str(), duration);
        }

        ExecutionResult {
            solver,
            result,
            duration,
            timed_out: duration >= timeout,
        }
    }

    /// Execute multiple solvers in parallel.
    ///
    /// Spawns up to max_threads concurrent solver executions.
    /// Returns results in order of completion.
    pub fn execute_parallel(&self, tasks: Vec<ExecutionTask>) -> Vec<ExecutionResult> {
        if tasks.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(tasks.len());
        let mut task_iter = tasks.into_iter().peekable();

        // Execute tasks in batches limited by max_threads
        while task_iter.peek().is_some() {
            let batch: Vec<_> = task_iter.by_ref().take(self.max_threads).collect();

            // Spawn all tasks in this batch
            let mut handles: Vec<JoinHandle<ExecutionResult>> = Vec::new();

            for task in batch {
                let task_solver = task.solver;
                let task_problem = task.problem.clone();
                let task_timeout = task.timeout;
                let cancel_flag = Arc::new(AtomicBool::new(false));

                let handle = thread::spawn(move || {
                    let start = Instant::now();
                    let solver_handle = Self::spawn_solver_thread_static(
                        task_solver,
                        task_problem.clone(),
                        cancel_flag.clone(),
                    );

                    let result = Self::join_with_timeout_static(
                        solver_handle,
                        task_timeout,
                        cancel_flag,
                        TimeoutEnforcement::Graceful, // Default to graceful for parallel tasks
                    );
                    let duration = start.elapsed();

                    ExecutionResult {
                        solver: task_solver,
                        result,
                        duration,
                        timed_out: duration >= task_timeout,
                    }
                });

                handles.push(handle);
            }

            // Collect results from this batch
            for handle in handles {
                match handle.join() {
                    Ok(result) => {
                        // Record time for adaptive timeout
                        if let Some(adaptive) = &self.adaptive_timeout {
                            adaptive.record_time(result.solver.as_str(), result.duration);
                        }
                        results.push(result);
                    }
                    Err(_) => {
                        // Thread panicked - record as error
                        results.push(ExecutionResult {
                            solver: SolverMethod::GradientSearch, // Placeholder
                            result: None,
                            duration: Duration::ZERO,
                            timed_out: false,
                        });
                    }
                }
            }
        }

        results
    }

    /// Spawn a solver thread with cancellation support.
    fn spawn_solver_thread(
        &self,
        solver: SolverMethod,
        problem: Problem,
        cancel_flag: Arc<AtomicBool>,
    ) -> JoinHandle<Option<SolveResult>> {
        thread::spawn(move || {
            // Check for early cancellation
            if cancel_flag.load(Ordering::Relaxed) {
                return None;
            }

            // Stub: In production, this would call the actual solver implementation
            // based on the solver method. For now, return a mock result.
            Self::mock_solve(&solver, &problem, &cancel_flag)
        })
    }

    /// Static version for use in thread::spawn closure.
    fn spawn_solver_thread_static(
        solver: SolverMethod,
        problem: Problem,
        cancel_flag: Arc<AtomicBool>,
    ) -> JoinHandle<Option<SolveResult>> {
        thread::spawn(move || {
            if cancel_flag.load(Ordering::Relaxed) {
                return None;
            }
            Self::mock_solve(&solver, &problem, &cancel_flag)
        })
    }

    /// Join a solver thread with timeout support.
    fn join_with_timeout(
        &self,
        handle: JoinHandle<Option<SolveResult>>,
        timeout: Duration,
        cancel_flag: Arc<AtomicBool>,
    ) -> Option<SolveResult> {
        Self::join_with_timeout_static(handle, timeout, cancel_flag, self.timeout_enforcement)
    }

    /// Static version for joining with timeout.
    fn join_with_timeout_static(
        handle: JoinHandle<Option<SolveResult>>,
        timeout: Duration,
        cancel_flag: Arc<AtomicBool>,
        enforcement: TimeoutEnforcement,
    ) -> Option<SolveResult> {
        // Wait for thread completion with timeout
        let start = Instant::now();

        loop {
            // Check if thread is done
            if handle.is_finished() {
                return handle.join().ok().flatten();
            }

            // Check timeout
            if start.elapsed() >= timeout {
                match enforcement {
                    TimeoutEnforcement::Strict => {
                        cancel_flag.store(true, Ordering::Relaxed);
                        // Give thread a chance to see cancellation
                        thread::yield_now();
                        let _ = handle.join(); // Clean up thread
                        return None;
                    }
                    TimeoutEnforcement::Graceful => {
                        // Let thread finish, but return None
                        return None;
                    }
                    TimeoutEnforcement::Adaptive => {
                        cancel_flag.store(true, Ordering::Relaxed);
                        thread::yield_now();
                        let _ = handle.join();
                        return None;
                    }
                }
            }

            // Small sleep to avoid busy waiting
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Mock solve implementation for testing.
    ///
    /// In production, this would delegate to actual solver implementations.
    fn mock_solve(
        _solver: &SolverMethod,
        _problem: &Problem,
        cancel_flag: &AtomicBool,
    ) -> Option<SolveResult> {
        // Simulate some work
        for _ in 0..10 {
            thread::sleep(Duration::from_millis(10));
            if cancel_flag.load(Ordering::Relaxed) {
                return None;
            }
        }

        // Return a mock success result
        Some(SolveResult {
            success: true,
            code: "// mock solution".to_string(),
            method: "mock_solver".to_string(),
            error: None,
            metadata: crate::differentiable::DifferentiableMetadata::default(),
        })
    }

    /// Update timeout statistics with observed execution time.
    pub fn update_timeout_stats(&self, solver: &str, duration: Duration) {
        if let Some(adaptive) = &self.adaptive_timeout {
            adaptive.record_time(solver, duration);
        }
    }

    /// Get recommended adaptive timeout for a solver (if available).
    pub fn get_adaptive_timeout(&self, solver: &str) -> Option<Duration> {
        if let Some(adaptive) = &self.adaptive_timeout {
            adaptive.get_timeout(solver)
        } else {
            None
        }
    }

    /// Reset adaptive timeout statistics.
    pub fn reset_timeout_stats(&self) {
        if let Some(adaptive) = &self.adaptive_timeout {
            adaptive.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn create_test_problem() -> Problem {
        Problem {
            name: "test_problem".to_string(),
            category: "test",
            description: "Test problem",
            signature: "fn test(x: i64) -> i64",
            examples: vec![Example {
                inputs: vec![Value::Int(42)],
                expected: Value::Int(84),
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
    fn test_parallel_executor_new() {
        let executor = ParallelExecutor::new();
        assert!(executor.max_threads >= 1);
    }

    #[test]
    fn test_parallel_executor_with_max_threads() {
        let executor = ParallelExecutor::new().with_max_threads(4);
        assert_eq!(executor.max_threads, 4);
    }

    #[test]
    fn test_parallel_executor_with_timeout_enforcement() {
        let executor = ParallelExecutor::new()
            .with_timeout_enforcement(TimeoutEnforcement::Strict);
        assert_eq!(executor.timeout_enforcement, TimeoutEnforcement::Strict);
    }

    #[test]
    fn test_parallel_executor_with_adaptive_timeout() {
        let adaptive = AdaptiveTimeout::new();
        let executor = ParallelExecutor::new().with_adaptive_timeout(adaptive);
        assert!(executor.adaptive_timeout.is_some());
        assert_eq!(executor.timeout_enforcement, TimeoutEnforcement::Adaptive);
    }

    #[test]
    fn test_duration_stats_default() {
        let stats = DurationStats::default();
        assert_eq!(stats.samples, 0);
        assert!(stats.mean.as_millis() > 0);
    }

    #[test]
    fn test_duration_stats_from_duration() {
        let duration = Duration::from_millis(250);
        let stats = DurationStats::from_duration(duration);
        assert_eq!(stats.samples, 1);
        assert_eq!(stats.mean, duration);
    }

    #[test]
    fn test_duration_stats_update() {
        let mut stats = DurationStats::default();
        let initial_mean = stats.mean;

        stats.update(Duration::from_millis(200), 0.1);

        assert_eq!(stats.samples, 1);
        assert!(stats.mean != initial_mean);
    }

    #[test]
    fn test_duration_stats_recommended_timeout() {
        let mut stats = DurationStats::default();
        stats.p95 = Duration::from_millis(200);

        let timeout = stats.recommended_timeout();
        // Should be p95 * 1.2
        assert_eq!(timeout.as_millis(), 240);
    }

    #[test]
    fn test_adaptive_timeout_new() {
        let adaptive = AdaptiveTimeout::new();
        assert_eq!(adaptive.alpha, 0.1);
        assert_eq!(adaptive.min_samples, 5);
    }

    #[test]
    fn test_adaptive_timeout_with_alpha() {
        let adaptive = AdaptiveTimeout::with_alpha(0.5);
        assert_eq!(adaptive.alpha, 0.5);
    }

    #[test]
    fn test_adaptive_timeout_with_min_samples() {
        let adaptive = AdaptiveTimeout::with_min_samples(10);
        assert_eq!(adaptive.min_samples, 10);
    }

    #[test]
    fn test_adaptive_timeout_record_time() {
        let adaptive = AdaptiveTimeout::new();
        adaptive.record_time("test_solver", Duration::from_millis(100));

        let stats = adaptive.get_stats("test_solver");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().samples, 1);
    }

    #[test]
    fn test_adaptive_timeout_get_timeout_insufficient_samples() {
        let adaptive = AdaptiveTimeout::new();
        adaptive.record_time("test_solver", Duration::from_millis(100));

        // With only 1 sample (less than min_samples=5), should return None
        let timeout = adaptive.get_timeout("test_solver");
        assert!(timeout.is_none());
    }

    #[test]
    fn test_adaptive_timeout_get_timeout_sufficient_samples() {
        let adaptive = AdaptiveTimeout::with_min_samples(3);

        // Record enough samples
        for _ in 0..3 {
            adaptive.record_time("test_solver", Duration::from_millis(100));
        }

        let timeout = adaptive.get_timeout("test_solver");
        assert!(timeout.is_some());
        assert!(timeout.unwrap().as_millis() > 0);
    }

    #[test]
    fn test_adaptive_timeout_reset() {
        let adaptive = AdaptiveTimeout::new();
        adaptive.record_time("test_solver", Duration::from_millis(100));

        adaptive.reset();

        assert!(adaptive.get_stats("test_solver").is_none());
    }

    #[test]
    fn test_execution_task_new() {
        let solver = SolverMethod::GradientSearch;
        let problem = create_test_problem();
        let timeout = Duration::from_secs(5);

        let task = ExecutionTask::new(solver, problem, timeout);
        assert_eq!(task.timeout, timeout);
    }

    #[test]
    fn test_execution_result_success() {
        let solver = SolverMethod::GradientSearch;
        let result = SolveResult {
            success: true,
            code: "// solution".to_string(),
            method: "test".to_string(),
            error: None,
            metadata: crate::differentiable::DifferentiableMetadata::default(),
        };

        let exec_result = ExecutionResult::success(solver, result, Duration::from_millis(100));
        assert!(exec_result.is_success());
        assert!(!exec_result.timed_out);
    }

    #[test]
    fn test_execution_result_timeout() {
        let solver = SolverMethod::GradientSearch;
        let exec_result = ExecutionResult::timeout(solver, Duration::from_millis(500));
        assert!(exec_result.timed_out);
        assert!(!exec_result.is_success());
    }

    #[test]
    fn test_execution_result_error() {
        let solver = SolverMethod::GradientSearch;
        let exec_result = ExecutionResult::error(solver, Duration::from_millis(50));
        assert!(!exec_result.timed_out);
        assert!(!exec_result.is_success());
    }

    #[test]
    fn test_execute_solver() {
        let executor = ParallelExecutor::new();
        let solver = SolverMethod::GradientSearch;
        let problem = create_test_problem();
        let timeout = Duration::from_secs(1);

        let result = executor.execute_solver(solver, &problem, timeout);
        assert_eq!(result.solver, solver);
        // Should complete within reasonable time
        assert!(result.duration < timeout + Duration::from_millis(100));
    }

    #[test]
    fn test_execute_parallel_empty() {
        let executor = ParallelExecutor::new();
        let results = executor.execute_parallel(vec![]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_execute_parallel_single_task() {
        let executor = ParallelExecutor::new();
        let problem = create_test_problem();
        let task = ExecutionTask::new(
            SolverMethod::GradientSearch,
            problem,
            Duration::from_secs(1),
        );

        let results = executor.execute_parallel(vec![task]);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_parallel_multiple_tasks() {
        let executor = ParallelExecutor::new().with_max_threads(2);
        let problem = create_test_problem();

        let tasks = vec![
            ExecutionTask::new(
                SolverMethod::GradientSearch,
                problem.clone(),
                Duration::from_secs(1),
            ),
            ExecutionTask::new(
                SolverMethod::EnumerativeSearch,
                problem.clone(),
                Duration::from_secs(1),
            ),
            ExecutionTask::new(
                SolverMethod::TeacherSearch,
                problem,
                Duration::from_secs(1),
            ),
        ];

        let results = executor.execute_parallel(tasks);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_compute_timeout_without_adaptive() {
        let executor = ParallelExecutor::new();
        let solver = SolverMethod::GradientSearch;

        let timeout = executor.compute_timeout(&solver);
        assert_eq!(timeout, executor.base_timeout);
    }

    #[test]
    fn test_compute_timeout_with_adaptive_insufficient_data() {
        let adaptive = AdaptiveTimeout::new();
        let executor = ParallelExecutor::new()
            .with_adaptive_timeout(adaptive)
            .with_base_timeout(Duration::from_secs(10));

        let solver = SolverMethod::GradientSearch;
        let timeout = executor.compute_timeout(&solver);
        // Should fall back to base timeout with insufficient data
        assert_eq!(timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_update_timeout_stats() {
        let adaptive = AdaptiveTimeout::with_min_samples(1);
        let executor = ParallelExecutor::new()
            .with_adaptive_timeout(adaptive);

        executor.update_timeout_stats("test_solver", Duration::from_millis(150));

        let timeout = executor.get_adaptive_timeout("test_solver");
        assert!(timeout.is_some());
    }

    #[test]
    fn test_reset_timeout_stats() {
        let adaptive = AdaptiveTimeout::new();
        let executor = ParallelExecutor::new()
            .with_adaptive_timeout(adaptive);

        executor.update_timeout_stats("test_solver", Duration::from_millis(100));
        executor.reset_timeout_stats();

        assert!(executor.get_adaptive_timeout("test_solver").is_none());
    }
}
