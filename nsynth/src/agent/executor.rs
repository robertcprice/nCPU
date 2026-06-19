// Plan Executor
// Executes tasks in dependency order with progress tracking, rollback, and error recovery

use crate::solver::SolverError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::task::JoinSet;

/// Unique identifier for tasks within a plan
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Individual task definition within a plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Human-readable task name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Tasks that must complete before this one can start
    pub dependencies: Vec<TaskId>,
    /// Whether this task can be executed in parallel with independent tasks
    pub parallel_safe: bool,
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Timeout for task execution
    pub timeout: Duration,
    /// Task-specific metadata
    pub metadata: serde_json::Value,
}

impl Task {
    /// Create a new task with default settings
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: TaskId::new(id),
            name: name.into(),
            description: None,
            dependencies: Vec::new(),
            parallel_safe: true,
            max_retries: 3,
            timeout: Duration::from_secs(300),
            metadata: serde_json::Value::Null,
        }
    }

    /// Add a dependency to this task
    pub fn with_dependency(mut self, dep: TaskId) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Add multiple dependencies to this task
    pub fn with_dependencies(mut self, deps: Vec<TaskId>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Set whether this task is safe for parallel execution
    pub fn with_parallel_safe(mut self, safe: bool) -> Self {
        self.parallel_safe = safe;
        self
    }

    /// Set the maximum retry attempts
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the execution timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Execution status of a task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently executing
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed and can be retried
    FailedRetryable,
    /// Task failed permanently
    FailedPermanent,
    /// Task was skipped
    Skipped,
    /// Task was cancelled
    Cancelled,
}

/// Result of a task execution attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: TaskId,
    /// Execution status
    pub status: TaskStatus,
    /// Number of attempts made
    pub attempts: usize,
    /// Output or error message
    pub output: String,
    /// Execution duration
    pub duration: Duration,
    /// Timestamp of completion
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

/// Milestone marker for tracking progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// Milestone identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Tasks that must complete for this milestone
    pub required_tasks: HashSet<TaskId>,
    /// Whether this milestone has been reached
    pub reached: bool,
    /// Timestamp when milestone was reached
    pub reached_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Rollback action for reverting task effects
#[derive(Debug, Clone)]
pub enum RollbackAction {
    /// Delete a file
    DeleteFile(PathBuf),
    /// Restore a file from backup
    RestoreFile { original: PathBuf, backup: PathBuf },
    /// Execute a custom rollback command
    CustomCommand(String),
    /// No rollback needed
    None,
}

/// Plan containing multiple tasks with dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    /// Unique plan identifier
    pub id: String,
    /// Human-readable plan name
    pub name: String,
    /// Plan description
    pub description: Option<String>,
    /// Tasks in the plan
    pub tasks: Vec<Task>,
    /// Milestones for tracking progress
    pub milestones: Vec<Milestone>,
    /// Plan creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl Plan {
    /// Create a new plan
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            tasks: Vec::new(),
            milestones: Vec::new(),
            created_at: now,
        }
    }

    /// Add a task to the plan
    pub fn with_task(mut self, task: Task) -> Self {
        self.tasks.push(task);
        self
    }

    /// Add multiple tasks to the plan
    pub fn with_tasks(mut self, tasks: Vec<Task>) -> Self {
        self.tasks.extend(tasks);
        self
    }

    /// Add a milestone to the plan
    pub fn with_milestone(mut self, milestone: Milestone) -> Self {
        self.milestones.push(milestone);
        self
    }

    /// Set the plan description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Validate the plan for circular dependencies
    pub fn validate(&self) -> Result<(), SolverError> {
        // Build adjacency list
        let mut graph: HashMap<&TaskId, Vec<&TaskId>> = HashMap::new();
        for task in &self.tasks {
            graph.insert(&task.id, Vec::new());
        }
        for task in &self.tasks {
            for dep in &task.dependencies {
                if let Some(deps) = graph.get_mut(&task.id) {
                    deps.push(dep);
                }
            }
        }

        // Detect cycles using DFS
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        fn detect_cycle(
            task_id: &TaskId,
            graph: &HashMap<&TaskId, Vec<&TaskId>>,
            visited: &mut HashSet<TaskId>,
            recursion_stack: &mut HashSet<TaskId>,
        ) -> bool {
            visited.insert(task_id.clone());
            recursion_stack.insert(task_id.clone());

            if let Some(deps) = graph.get(task_id) {
                for dep in deps.iter() {
                    if !visited.contains(dep) {
                        if detect_cycle(dep, graph, visited, recursion_stack) {
                            return true;
                        }
                    } else if recursion_stack.contains(dep) {
                        return true;
                    }
                }
            }

            recursion_stack.remove(task_id);
            false
        }

        for task in &self.tasks {
            if !visited.contains(&task.id) {
                if detect_cycle(&task.id, &graph, &mut visited, &mut recursion_stack) {
                    return Err(SolverError::ConfigurationError(format!(
                        "Circular dependency detected involving task: {}",
                        task.id
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get tasks in topological order (dependency order)
    pub fn execution_order(&self) -> Result<Vec<TaskId>, SolverError> {
        self.validate()?;

        let mut in_degree: HashMap<TaskId, usize> = HashMap::new();
        let mut adj_list: HashMap<TaskId, Vec<TaskId>> = HashMap::new();

        // Initialize
        for task in &self.tasks {
            in_degree.insert(task.id.clone(), 0);
            adj_list.insert(task.id.clone(), Vec::new());
        }

        // Build graph
        for task in &self.tasks {
            for dep in &task.dependencies {
                (*in_degree.entry(task.id.clone()).or_insert(0)) += 1;
                adj_list
                    .entry(dep.clone())
                    .or_default()
                    .push(task.id.clone());
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<TaskId> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut result = Vec::new();

        while let Some(task_id) = queue.pop() {
            result.push(task_id.clone());

            if let Some(dependents) = adj_list.get(&task_id) {
                for dep in dependents {
                    if let Some(degree) = in_degree.get_mut(dep) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(dep.clone());
                        }
                    }
                }
            }
        }

        Ok(result)
    }
}

/// Progress tracking during plan execution
#[derive(Debug, Clone)]
pub struct ExecutionProgress {
    /// Current milestone
    pub current_milestone: Option<String>,
    /// Completed tasks
    pub completed_tasks: HashSet<TaskId>,
    /// Failed tasks
    pub failed_tasks: HashSet<TaskId>,
    /// Running tasks
    pub running_tasks: HashSet<TaskId>,
    /// Task results indexed by task ID
    pub task_results: HashMap<TaskId, TaskResult>,
    /// Start time
    pub start_time: Instant,
    /// Last update time
    pub last_update: Instant,
}

impl ExecutionProgress {
    pub fn new() -> Self {
        Self {
            current_milestone: None,
            completed_tasks: HashSet::new(),
            failed_tasks: HashSet::new(),
            running_tasks: HashSet::new(),
            task_results: HashMap::new(),
            start_time: Instant::now(),
            last_update: Instant::now(),
        }
    }

    /// Calculate overall completion percentage
    pub fn completion_percentage(&self, total_tasks: usize) -> f64 {
        if total_tasks == 0 {
            return 100.0;
        }
        (self.completed_tasks.len() as f64 / total_tasks as f64) * 100.0
    }

    /// Get estimated remaining time
    pub fn estimated_remaining(&self, total_tasks: usize) -> Option<Duration> {
        let completed = self.completed_tasks.len();
        if completed == 0 {
            return None;
        }

        let elapsed = self.start_time.elapsed();
        let avg_time_per_task = elapsed / completed as u32;
        let remaining = total_tasks.saturating_sub(completed);

        Some(avg_time_per_task * remaining as u32)
    }
}

/// Executor configuration
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum number of parallel tasks
    pub max_parallel_tasks: usize,
    /// Default retry attempts for tasks
    pub default_max_retries: usize,
    /// Default timeout for tasks
    pub default_timeout: Duration,
    /// Enable rollback on failure
    pub enable_rollback: bool,
    /// Directory for storing execution state
    pub state_dir: Option<PathBuf>,
    /// Enable progress persistence
    pub persist_progress: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: 4,
            default_max_retries: 3,
            default_timeout: Duration::from_secs(300),
            enable_rollback: true,
            state_dir: None,
            persist_progress: false,
        }
    }
}

/// Task executor function type - simplified to use a direct function pointer
type TaskExecutorFn = Arc<
    dyn Fn(&Task) -> Pin<Box<dyn Future<Output = Result<String, SolverError>> + Send>>
        + Send
        + Sync,
>;

/// Main plan executor
pub struct Executor {
    config: ExecutorConfig,
    executor_fn: Mutex<Option<TaskExecutorFn>>,
    rollback_actions: Arc<Mutex<HashMap<TaskId, RollbackAction>>>,
    progress: Arc<RwLock<ExecutionProgress>>,
}

impl Executor {
    /// Create a new executor with default configuration
    pub fn new() -> Self {
        Self::with_config(ExecutorConfig::default())
    }

    /// Create a new executor with custom configuration
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self {
            config,
            executor_fn: Mutex::new(None),
            rollback_actions: Arc::new(Mutex::new(HashMap::new())),
            progress: Arc::new(RwLock::new(ExecutionProgress::new())),
        }
    }

    /// Set the task executor function
    pub async fn with_executor<F, Fut>(&self, f: F)
    where
        F: Fn(&Task) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, SolverError>> + Send + 'static,
    {
        let wrapped: Arc<
            dyn Fn(&Task) -> Pin<Box<dyn Future<Output = Result<String, SolverError>> + Send>>
                + Send
                + Sync,
        > = Arc::new(move |task: &Task| {
            let result = f(task);
            Box::pin(async move { result.await })
        });
        *self.executor_fn.lock().await = Some(wrapped);
    }

    /// Register a rollback action for a task
    pub async fn register_rollback(&self, task_id: TaskId, action: RollbackAction) {
        self.rollback_actions.lock().await.insert(task_id, action);
    }

    /// Execute a plan
    pub async fn execute(&self, plan: &Plan) -> Result<ExecutionSummary, SolverError> {
        // Validate plan
        plan.validate()?;

        // Initialize progress
        {
            let mut progress = self.progress.write().await;
            progress.start_time = Instant::now();
            progress.last_update = Instant::now();
        }

        // Get execution order
        let execution_order = plan.execution_order()?;

        // Create task lookup with owned tasks
        let task_map: HashMap<TaskId, Task> = plan
            .tasks
            .iter()
            .map(|task| (task.id.clone(), task.clone()))
            .collect();

        // Track completed tasks for dependency resolution
        let completed = Arc::new(Mutex::new(HashSet::new()));

        // Create semaphore for parallel execution
        let semaphore = Arc::new(Semaphore::new(self.config.max_parallel_tasks));

        // Create join set for managing parallel tasks
        let mut join_set = JoinSet::new();

        // Spawn tasks as dependencies become available
        for task_id in execution_order {
            let task = task_map
                .get(&task_id)
                .ok_or_else(|| {
                    SolverError::ConfigurationError(format!("Task not found: {}", task_id))
                })?
                .clone();

            // Check if dependencies are satisfied
            let completed_clone = completed.clone();
            let deps = task.dependencies.clone();

            // Wait for dependencies. Harvest running tasks while waiting so a
            // failed prerequisite is propagated instead of causing an infinite wait.
            while !deps.is_empty() {
                let completed_guard = completed_clone.lock().await;
                let all_deps_complete = deps.iter().all(|dep| completed_guard.contains(dep));
                if all_deps_complete {
                    drop(completed_guard);
                    break;
                }
                drop(completed_guard);

                match join_set.join_next().await {
                    Some(result) => {
                        result.map_err(|e| SolverError::JoinError(e.to_string()))??;
                    }
                    None => {
                        return Err(SolverError::ConfigurationError(format!(
                            "Task {} has unsatisfied dependencies with no runnable prerequisite",
                            task.id
                        )));
                    }
                }
            }

            // Check if task is parallel-safe
            if task.parallel_safe {
                // Spawn parallel task
                let sem = semaphore.clone();
                let task_clone = task.clone();
                let progress_clone = self.progress.clone();
                let executor_fn = self.executor_fn.lock().await.clone();
                let rollback_actions = self.rollback_actions.clone();
                let completed_clone = completed.clone();

                join_set.spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    Self::execute_single_task(
                        &task_clone,
                        progress_clone,
                        executor_fn,
                        rollback_actions,
                        completed_clone,
                    )
                    .await
                });
            } else {
                // Execute sequentially
                let task_clone = task.clone();
                let progress_clone = self.progress.clone();
                let executor_fn = self.executor_fn.lock().await.clone();
                let rollback_actions = self.rollback_actions.clone();
                let completed_clone = completed.clone();

                Self::execute_single_task(
                    &task_clone,
                    progress_clone,
                    executor_fn,
                    rollback_actions,
                    completed_clone,
                )
                .await?;
            }

            // Check for milestone completion
            self.check_milestones(plan).await?;
        }

        // Wait for all parallel tasks to complete
        while let Some(result) = join_set.join_next().await {
            result.map_err(|e| SolverError::JoinError(e.to_string()))??;
        }

        // Generate summary
        self.generate_summary(plan).await
    }

    /// Execute a single task with retry logic
    async fn execute_single_task(
        task: &Task,
        progress: Arc<RwLock<ExecutionProgress>>,
        executor_fn: Option<TaskExecutorFn>,
        rollback_actions: Arc<Mutex<HashMap<TaskId, RollbackAction>>>,
        completed: Arc<Mutex<HashSet<TaskId>>>,
    ) -> Result<(), SolverError> {
        let task_id = task.id.clone();

        // Update progress to running
        {
            let mut prog = progress.write().await;
            prog.running_tasks.insert(task_id.clone());
            prog.last_update = Instant::now();
        }

        // Get executor function
        let executor = executor_fn.as_ref().ok_or_else(|| {
            SolverError::ConfigurationError("No executor function set".to_string())
        })?;

        // Execute with retries
        for attempt in 0..=task.max_retries {
            let start = Instant::now();

            // Execute with timeout
            let result = tokio::time::timeout(task.timeout, executor(&task)).await;

            let duration = start.elapsed();
            let status;
            let output;

            match result {
                Ok(Ok(success_output)) => {
                    status = TaskStatus::Completed;
                    output = success_output;

                    // Mark as completed
                    {
                        let mut prog = progress.write().await;
                        prog.completed_tasks.insert(task_id.clone());
                        prog.running_tasks.remove(&task_id);
                        prog.task_results.insert(
                            task_id.clone(),
                            TaskResult {
                                task_id: task_id.clone(),
                                status,
                                attempts: attempt + 1,
                                output: output.clone(),
                                duration,
                                completed_at: chrono::Utc::now(),
                            },
                        );
                        prog.last_update = Instant::now();
                    }

                    // Mark dependencies as satisfied
                    completed.lock().await.insert(task_id.clone());

                    return Ok(());
                }
                Ok(Err(e)) => {
                    output = e.to_string();

                    if attempt < task.max_retries {
                        // Retryable failure - continue loop
                        tokio::time::sleep(Duration::from_millis(100 * (attempt + 1) as u64)).await;
                        continue;
                    } else {
                        // Permanent failure
                        let status = TaskStatus::FailedPermanent;

                        // Update progress
                        {
                            let mut prog = progress.write().await;
                            prog.failed_tasks.insert(task_id.clone());
                            prog.running_tasks.remove(&task_id);
                            prog.task_results.insert(
                                task_id.clone(),
                                TaskResult {
                                    task_id: task_id.clone(),
                                    status,
                                    attempts: attempt + 1,
                                    output: output.clone(),
                                    duration,
                                    completed_at: chrono::Utc::now(),
                                },
                            );
                            prog.last_update = Instant::now();
                        }

                        // Perform rollback if enabled
                        if let Some(action) = rollback_actions.lock().await.get(&task_id) {
                            Self::perform_rollback(action).await?;
                        }

                        return Err(e);
                    }
                }
                Err(_) => {
                    let timeout_err = SolverError::Timeout(format!(
                        "Task {} exceeded timeout of {:?}",
                        task_id, task.timeout
                    ));
                    output = timeout_err.to_string();
                    let status = TaskStatus::FailedPermanent;

                    // Update progress
                    {
                        let mut prog = progress.write().await;
                        prog.failed_tasks.insert(task_id.clone());
                        prog.running_tasks.remove(&task_id);
                        prog.task_results.insert(
                            task_id.clone(),
                            TaskResult {
                                task_id: task_id.clone(),
                                status,
                                attempts: attempt + 1,
                                output: output.clone(),
                                duration,
                                completed_at: chrono::Utc::now(),
                            },
                        );
                        prog.last_update = Instant::now();
                    }

                    return Err(timeout_err);
                }
            }
        }

        Ok(())
    }

    /// Perform a rollback action

    /// Perform a rollback action
    async fn perform_rollback(action: &RollbackAction) -> Result<(), SolverError> {
        match action {
            RollbackAction::DeleteFile(path) => {
                if path.exists() {
                    fs::remove_file(path).await.map_err(|e| {
                        SolverError::IoError(format!("Failed to delete file: {}", e))
                    })?;
                }
            }
            RollbackAction::RestoreFile { original, backup } => {
                if backup.exists() {
                    fs::copy(backup, original).await.map_err(|e| {
                        SolverError::IoError(format!("Failed to restore file: {}", e))
                    })?;
                }
            }
            RollbackAction::CustomCommand(cmd) => {
                // Execute custom rollback command
                let output = tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(cmd)
                    .output()
                    .await
                    .map_err(|e| {
                        SolverError::IoError(format!("Failed to execute rollback: {}", e))
                    })?;

                if !output.status.success() {
                    return Err(SolverError::Other(format!(
                        "Rollback command failed: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )));
                }
            }
            RollbackAction::None => {}
        }
        Ok(())
    }

    /// Check and update milestones
    async fn check_milestones(&self, plan: &Plan) -> Result<(), SolverError> {
        let progress = self.progress.read().await;
        let mut current_milestone: Option<String> = None;

        for milestone in &plan.milestones {
            if milestone.reached {
                continue;
            }

            // Check if all required tasks are complete
            let all_complete = milestone
                .required_tasks
                .iter()
                .all(|task_id| progress.completed_tasks.contains(task_id));

            if all_complete {
                current_milestone = Some(milestone.id.clone());
                break;
            }
        }

        // Update current milestone
        if current_milestone.is_some() {
            drop(progress);
            let mut prog = self.progress.write().await;
            prog.current_milestone = current_milestone;
        }

        Ok(())
    }

    /// Generate execution summary
    async fn generate_summary(&self, plan: &Plan) -> Result<ExecutionSummary, SolverError> {
        let progress = self.progress.read().await;

        let total_tasks = plan.tasks.len();
        let completed = progress.completed_tasks.len();
        let failed = progress.failed_tasks.len();
        let duration = progress.start_time.elapsed();

        Ok(ExecutionSummary {
            plan_id: plan.id.clone(),
            plan_name: plan.name.clone(),
            total_tasks,
            completed_tasks: completed,
            failed_tasks: failed,
            skipped_tasks: 0,
            duration,
            success: failed == 0,
            task_results: progress.task_results.clone(),
            milestones_reached: self.count_reached_milestones(plan).await,
            total_milestones: plan.milestones.len(),
        })
    }

    /// Count reached milestones
    async fn count_reached_milestones(&self, plan: &Plan) -> usize {
        let progress = self.progress.read().await;
        plan.milestones
            .iter()
            .filter(|m| {
                m.reached
                    || m.required_tasks
                        .iter()
                        .all(|t| progress.completed_tasks.contains(t))
            })
            .count()
    }

    /// Get current execution progress
    pub async fn progress(&self) -> ExecutionProgress {
        self.progress.read().await.clone()
    }

    /// Reset the executor state
    pub async fn reset(&self) {
        let mut progress = self.progress.write().await;
        *progress = ExecutionProgress::new();
        self.rollback_actions.lock().await.clear();
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of plan execution
#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    /// Plan identifier
    pub plan_id: String,
    /// Plan name
    pub plan_name: String,
    /// Total number of tasks
    pub total_tasks: usize,
    /// Number of completed tasks
    pub completed_tasks: usize,
    /// Number of failed tasks
    pub failed_tasks: usize,
    /// Number of skipped tasks
    pub skipped_tasks: usize,
    /// Total execution duration
    pub duration: Duration,
    /// Whether execution was successful
    pub success: bool,
    /// Individual task results
    pub task_results: HashMap<TaskId, TaskResult>,
    /// Number of milestones reached
    pub milestones_reached: usize,
    /// Total number of milestones
    pub total_milestones: usize,
}

impl ExecutionSummary {
    /// Get success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_tasks == 0 {
            return 100.0;
        }
        (self.completed_tasks as f64 / self.total_tasks as f64) * 100.0
    }

    /// Get a summary message
    pub fn summary_message(&self) -> String {
        if self.success {
            format!(
                "Plan '{}' completed successfully: {}/{} tasks in {:.2}s",
                self.plan_name,
                self.completed_tasks,
                self.total_tasks,
                self.duration.as_secs_f64()
            )
        } else {
            format!(
                "Plan '{}' partially completed: {}/{} tasks succeeded, {} failed in {:.2}s",
                self.plan_name,
                self.completed_tasks,
                self.total_tasks,
                self.failed_tasks,
                self.duration.as_secs_f64()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_plan() -> Plan {
        Plan::new("test-plan", "Test Plan")
            .with_description("A test plan for unit tests")
            .with_tasks(vec![
                Task::new("task-1", "First Task")
                    .with_description("This is the first task")
                    .with_timeout(Duration::from_secs(10)),
                Task::new("task-2", "Second Task")
                    .with_dependency(TaskId::new("task-1"))
                    .with_parallel_safe(true),
                Task::new("task-3", "Third Task")
                    .with_dependency(TaskId::new("task-1"))
                    .with_parallel_safe(true),
                Task::new("task-4", "Fourth Task")
                    .with_dependencies(vec![TaskId::new("task-2"), TaskId::new("task-3")])
                    .with_parallel_safe(false),
            ])
            .with_milestone(Milestone {
                id: "milestone-1".to_string(),
                name: "First Tasks Complete".to_string(),
                required_tasks: HashSet::from_iter(vec![
                    TaskId::new("task-1"),
                    TaskId::new("task-2"),
                    TaskId::new("task-3"),
                ]),
                reached: false,
                reached_at: None,
            })
    }

    #[tokio::test]
    async fn test_task_creation() {
        let task = Task::new("test", "Test Task")
            .with_description("A test task")
            .with_dependency(TaskId::new("dep-1"))
            .with_parallel_safe(false)
            .with_max_retries(5)
            .with_timeout(Duration::from_secs(60));

        assert_eq!(task.id.as_str(), "test");
        assert_eq!(task.name, "Test Task");
        assert_eq!(task.description, Some("A test task".to_string()));
        assert_eq!(task.dependencies.len(), 1);
        assert!(!task.parallel_safe);
        assert_eq!(task.max_retries, 5);
        assert_eq!(task.timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_plan_validation() {
        let plan = create_test_plan();
        assert!(plan.validate().is_ok());
    }

    #[tokio::test]
    async fn test_circular_dependency_detection() {
        let plan = Plan::new("circular", "Circular Plan").with_tasks(vec![
            Task::new("a", "Task A").with_dependency(TaskId::new("c")),
            Task::new("b", "Task B").with_dependency(TaskId::new("a")),
            Task::new("c", "Task C").with_dependency(TaskId::new("b")),
        ]);

        assert!(plan.validate().is_err());
    }

    #[tokio::test]
    async fn test_execution_order() {
        let plan = create_test_plan();
        let order = plan.execution_order().unwrap();

        // task-1 should come first (no dependencies)
        assert_eq!(order[0].as_str(), "task-1");

        // task-4 should come last (depends on task-2 and task-3)
        assert_eq!(order[3].as_str(), "task-4");

        // task-2 and task-3 should both come after task-1
        let idx_1 = order.iter().position(|id| id.as_str() == "task-1").unwrap();
        let idx_2 = order.iter().position(|id| id.as_str() == "task-2").unwrap();
        let idx_3 = order.iter().position(|id| id.as_str() == "task-3").unwrap();

        assert!(idx_2 > idx_1);
        assert!(idx_3 > idx_1);
    }

    #[tokio::test]
    async fn test_executor_creation() {
        let executor = Executor::new();
        assert_eq!(executor.config.max_parallel_tasks, 4);

        let custom_executor = Executor::with_config(ExecutorConfig {
            max_parallel_tasks: 8,
            ..Default::default()
        });
        assert_eq!(custom_executor.config.max_parallel_tasks, 8);
    }

    #[tokio::test]
    async fn test_execution_with_mock_executor() {
        let executor = Executor::with_config(ExecutorConfig {
            max_parallel_tasks: 2,
            ..Default::default()
        });

        // Set up mock executor
        executor
            .with_executor(|task: &Task| {
                let task_name = task.name.clone();
                async move {
                    // Simulate work
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(format!("Completed: {}", task_name))
                }
            })
            .await;

        let plan = create_test_plan();
        let summary = executor.execute(&plan).await.unwrap();

        assert!(summary.success);
        assert_eq!(summary.completed_tasks, 4);
        assert_eq!(summary.failed_tasks, 0);
    }

    #[tokio::test]
    async fn test_execution_with_failure() {
        let executor = Executor::new();

        // Set up executor that fails on task-3
        executor
            .with_executor(|task: &Task| {
                let task_id = task.id.clone();
                let task_name = task.name.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;

                    if task_id.as_str() == "task-3" {
                        return Err(SolverError::Other("Task 3 failed".to_string()));
                    }

                    Ok(format!("Completed: {}", task_name))
                }
            })
            .await;

        let plan = create_test_plan();
        let result = executor.execute(&plan).await;

        assert!(result.is_err());

        let summary = result.unwrap_err();
        match summary {
            SolverError::Other(msg) => assert!(msg.contains("Task 3 failed")),
            _ => panic!("Expected Other error"),
        }
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let executor = Executor::new();

        executor
            .with_executor(|task: &Task| {
                let task_name = task.name.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(format!("Completed: {}", task_name))
                }
            })
            .await;

        let plan = create_test_plan();

        // Spawn execution in background
        let executor_arc = Arc::new(executor);
        let executor_clone = executor_arc.clone();
        let plan_clone = plan.clone();

        let handle = tokio::spawn(async move { executor_clone.execute(&plan_clone).await });

        // Check progress during execution
        tokio::time::sleep(Duration::from_millis(150)).await;

        let progress = executor_arc.progress().await;
        assert!(progress.completed_tasks.len() > 0 || progress.running_tasks.len() > 0);

        // Wait for completion
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_rollback_registration() {
        let executor = Executor::new();

        let task_id = TaskId::new("rollback-test");
        let action = RollbackAction::DeleteFile(PathBuf::from("/tmp/test.txt"));

        executor.register_rollback(task_id.clone(), action).await;

        let actions = executor.rollback_actions.lock().await;
        assert!(actions.contains_key(&task_id));
    }

    #[tokio::test]
    async fn test_rollback_execution() {
        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("executor_rollback_test.txt");
        fs::write(&test_file, b"test content").await.unwrap();

        assert!(test_file.exists());

        // Perform rollback
        let action = RollbackAction::DeleteFile(test_file.clone());
        Executor::perform_rollback(&action).await.unwrap();

        assert!(!test_file.exists());
    }

    #[tokio::test]
    async fn test_milestone_tracking() {
        let executor = Executor::new();

        executor
            .with_executor(|task: &Task| {
                let task_name = task.name.clone();
                async move {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(format!("Completed: {}", task_name))
                }
            })
            .await;

        let plan = create_test_plan();
        let summary = executor.execute(&plan).await.unwrap();

        assert_eq!(summary.milestones_reached, 1);
        assert_eq!(summary.total_milestones, 1);
    }

    #[tokio::test]
    async fn test_execution_summary() {
        let mut progress = ExecutionProgress::new();
        progress.completed_tasks.insert(TaskId::new("task-1"));
        progress.completed_tasks.insert(TaskId::new("task-2"));
        progress.failed_tasks.insert(TaskId::new("task-3"));

        assert_eq!(progress.completion_percentage(5), 40.0);

        let summary = ExecutionSummary {
            plan_id: "test".to_string(),
            plan_name: "Test".to_string(),
            total_tasks: 3,
            completed_tasks: 2,
            failed_tasks: 1,
            skipped_tasks: 0,
            duration: Duration::from_secs(10),
            success: false,
            task_results: HashMap::new(),
            milestones_reached: 1,
            total_milestones: 2,
        };

        assert!((summary.success_rate() - 66.66).abs() < 0.1);
        assert!(!summary.success);
    }

    #[tokio::test]
    async fn test_parallel_execution_limits() {
        let executor = Executor::with_config(ExecutorConfig {
            max_parallel_tasks: 2,
            ..Default::default()
        });

        let running_count = Arc::new(Mutex::new(0usize));
        let max_concurrent = Arc::new(Mutex::new(0usize));

        executor
            .with_executor({
                let running_count = running_count.clone();
                let max_concurrent = max_concurrent.clone();
                move |task: &Task| {
                    let task_name = task.name.clone();
                    let running_count = running_count.clone();
                    let max_concurrent = max_concurrent.clone();
                    async move {
                        // Increment running count
                        {
                            let mut running = running_count.lock().await;
                            *running += 1;
                            let current = *running;

                            // Update max if needed
                            let mut max = max_concurrent.lock().await;
                            if current > *max {
                                *max = current;
                            }
                        }

                        // Do some work
                        tokio::time::sleep(Duration::from_millis(100)).await;

                        // Decrement running count
                        {
                            let mut running = running_count.lock().await;
                            *running -= 1;
                        }

                        Ok(format!("Completed: {}", task_name))
                    }
                }
            })
            .await;

        let plan = Plan::new("parallel-test", "Parallel Test").with_tasks(vec![
            Task::new("task-1", "Task 1").with_timeout(Duration::from_secs(5)),
            Task::new("task-2", "Task 2").with_timeout(Duration::from_secs(5)),
            Task::new("task-3", "Task 3").with_timeout(Duration::from_secs(5)),
            Task::new("task-4", "Task 4").with_timeout(Duration::from_secs(5)),
        ]);

        executor.execute(&plan).await.unwrap();

        // Check that we never exceeded max_parallel_tasks
        let max = *max_concurrent.lock().await;
        assert!(max <= 2);
    }

    #[tokio::test]
    async fn test_task_timeout() {
        let executor = Executor::new();

        executor
            .with_executor(|task: &Task| {
                async move {
                    // Sleep longer than timeout
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    Ok(format!("Should not complete"))
                }
            })
            .await;

        let plan = Plan::new("timeout-test", "Timeout Test").with_tasks(vec![Task::new(
            "slow-task",
            "Slow Task",
        )
        .with_timeout(Duration::from_millis(100))]);

        let result = executor.execute(&plan).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            SolverError::Timeout(_) => {}
            _ => panic!("Expected timeout error"),
        }
    }

    #[tokio::test]
    async fn test_retry_logic() {
        let executor = Executor::new();
        let attempt_count = Arc::new(Mutex::new(0));

        executor
            .with_executor({
                let attempt_count = attempt_count.clone();
                move |task: &Task| {
                    let attempt_count = attempt_count.clone();
                    async move {
                        let mut count = attempt_count.lock().await;
                        *count += 1;
                        let attempts = *count;
                        drop(count);

                        if attempts < 3 {
                            Err(SolverError::Other("Temporary failure".to_string()))
                        } else {
                            Ok(format!("Succeeded on attempt {}", attempts))
                        }
                    }
                }
            })
            .await;

        let plan = Plan::new("retry-test", "Retry Test").with_tasks(vec![Task::new(
            "flakey-task",
            "Flakey Task",
        )
        .with_max_retries(5)
        .with_timeout(Duration::from_secs(10))]);

        let result = executor.execute(&plan).await;
        assert!(result.is_ok());

        let final_attempts = *attempt_count.lock().await;
        assert!(final_attempts >= 3);
    }

    #[tokio::test]
    async fn test_executor_reset() {
        let executor = Executor::new();

        executor
            .with_executor(|task: &Task| {
                let task_name = task.name.clone();
                async move { Ok(format!("Completed: {}", task_name)) }
            })
            .await;

        let plan = create_test_plan();
        executor.execute(&plan).await.unwrap();

        let progress = executor.progress().await;
        assert_eq!(progress.completed_tasks.len(), 4);

        executor.reset().await;

        let progress = executor.progress().await;
        assert_eq!(progress.completed_tasks.len(), 0);
        assert_eq!(progress.running_tasks.len(), 0);
    }

    #[tokio::test]
    async fn test_complex_dependency_graph() {
        let executor = Executor::new();

        let execution_order = Arc::new(Mutex::new(Vec::new()));

        executor
            .with_executor({
                let execution_order = execution_order.clone();
                move |task: &Task| {
                    let execution_order = execution_order.clone();
                    let task_id = task.id.clone();
                    let task_name = task.name.clone();
                    async move {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        execution_order.lock().await.push(task_id.clone());
                        Ok(format!("Completed: {}", task_name))
                    }
                }
            })
            .await;

        // Create a diamond dependency pattern
        //     A
        //    / \
        //   B   C
        //    \ /
        //     D
        let plan = Plan::new("diamond", "Diamond Dependencies").with_tasks(vec![
            Task::new("a", "Task A").with_parallel_safe(true),
            Task::new("b", "Task B")
                .with_dependency(TaskId::new("a"))
                .with_parallel_safe(true),
            Task::new("c", "Task C")
                .with_dependency(TaskId::new("a"))
                .with_parallel_safe(true),
            Task::new("d", "Task D")
                .with_dependencies(vec![TaskId::new("b"), TaskId::new("c")])
                .with_parallel_safe(true),
        ]);

        executor.execute(&plan).await.unwrap();

        let order = execution_order.lock().await;
        let pos_a = order.iter().position(|id| id.as_str() == "a").unwrap();
        let pos_b = order.iter().position(|id| id.as_str() == "b").unwrap();
        let pos_c = order.iter().position(|id| id.as_str() == "c").unwrap();
        let pos_d = order.iter().position(|id| id.as_str() == "d").unwrap();

        // A must come first
        assert_eq!(pos_a, 0);

        // B and C must come after A
        assert!(pos_b > pos_a);
        assert!(pos_c > pos_a);

        // D must come after both B and C
        assert!(pos_d > pos_b);
        assert!(pos_d > pos_c);
    }
}
