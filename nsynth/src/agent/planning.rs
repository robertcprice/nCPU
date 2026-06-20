// Task Decomposition and Planning Module for Multi-Agent Synthesis
//
// This module provides intelligent task decomposition capabilities for the
// multi-agent orchestrator, enabling high-level goals to be broken down into
// executable subtasks with proper dependency tracking and optimization.

use crate::benchmark::{Problem, Value};
use crate::solver::{SolveResult, SolverError};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for tasks within the planning system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TaskId(u64);

impl TaskId {
    pub fn new(id: u64) -> Self {
        TaskId(id)
    }

    pub fn next(&self) -> Self {
        TaskId(self.0 + 1)
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// Strategy selection for task decomposition
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecompositionStrategy {
    /// Break down by synthesis method family (gradient, search, enumerative, etc.)
    BySynthesisFamily,
    /// Decompose by input/output type characteristics
    ByTypeCharacteristics,
    /// Hierarchical breakdown with subtask refinement
    Hierarchical,
    /// Dependency-driven decomposition based on problem constraints
    DependencyDriven,
    /// Adaptive strategy that selects the best approach based on problem analysis
    Adaptive,
}

impl DecompositionStrategy {
    /// Select the best strategy based on problem characteristics
    pub fn select_for_problem(problem: &Problem) -> Self {
        // Count examples and analyze input types
        let example_count = problem.examples.len();
        let has_arrays = problem
            .examples
            .iter()
            .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))));
        let has_strings = problem
            .examples
            .iter()
            .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Str(_))));
        let is_recursive = problem.recursive_allowed;
        let is_tree_input = problem.tree_input;
        let arg_count = problem
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0);

        // Adaptive selection logic
        match (
            has_arrays,
            has_strings,
            is_recursive,
            is_tree_input,
            example_count,
            arg_count,
        ) {
            // Complex recursive or tree problems benefit from hierarchical breakdown
            (_, _, true, _, _, _) | (_, _, _, true, _, _) => DecompositionStrategy::Hierarchical,
            // Multi-argument problems with various types need dependency tracking
            (_, _, _, _, _, n) if n > 3 => DecompositionStrategy::DependencyDriven,
            // Simple scalar problems work well with family-based approach
            (false, false, false, false, _, _) => DecompositionStrategy::BySynthesisFamily,
            // Type-heavy problems benefit from characteristic-based decomposition
            _ => DecompositionStrategy::ByTypeCharacteristics,
        }
    }
}

/// Priority level for task execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl TaskPriority {
    /// Calculate priority based on task characteristics
    pub fn estimate(
        is_foundation: bool,
        estimated_complexity: usize,
        example_count: usize,
    ) -> Self {
        match (is_foundation, estimated_complexity, example_count) {
            (true, _, _) => TaskPriority::Critical,
            (_, c, _) if c > 100 => TaskPriority::Low,
            (_, c, e) if c > 50 && e < 5 => TaskPriority::Medium,
            (_, _, e) if e < 3 => TaskPriority::High,
            _ => TaskPriority::Medium,
        }
    }
}

/// Current status of a task in the execution pipeline
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task created but not yet ready to execute
    Pending,
    /// Ready to execute (dependencies satisfied)
    Ready,
    /// Currently being executed
    InProgress,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed(String),
    /// Skipped due to optimization or irrelevance
    Skipped(String),
    /// Blocked waiting for dependencies
    Blocked,
}

/// A decomposed task with metadata and dependencies
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub name: String,
    pub description: String,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub dependencies: HashSet<TaskId>,
    pub dependents: HashSet<TaskId>,
    pub estimated_complexity: usize,
    pub method_hint: Option<String>,
    pub subtasks: Vec<TaskId>,
    pub parent_task: Option<TaskId>,
    pub result: Option<TaskResult>,
    pub created_at: std::time::Instant,
    pub started_at: Option<std::time::Instant>,
    pub completed_at: Option<std::time::Instant>,
}

impl Task {
    /// Create a new task with the given name and description
    pub fn new(id: TaskId, name: String, description: String) -> Self {
        Task {
            id,
            name,
            description,
            status: TaskStatus::Pending,
            priority: TaskPriority::Medium,
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            estimated_complexity: 50,
            method_hint: None,
            subtasks: Vec::new(),
            parent_task: None,
            result: None,
            created_at: std::time::Instant::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// Add a dependency on another task
    pub fn add_dependency(&mut self, dep_id: TaskId) {
        self.dependencies.insert(dep_id);
    }

    /// Add a dependent task (something that depends on this task)
    pub fn add_dependent(&mut self, dep_id: TaskId) {
        self.dependents.insert(dep_id);
    }

    /// Check if all dependencies are satisfied
    pub fn dependencies_satisfied(&self, completed_tasks: &HashSet<TaskId>) -> bool {
        self.dependencies
            .iter()
            .all(|id| completed_tasks.contains(id))
    }

    /// Mark the task as started
    pub fn start(&mut self) {
        self.status = TaskStatus::InProgress;
        self.started_at = Some(std::time::Instant::now());
    }

    /// Mark the task as completed with a result
    pub fn complete(&mut self, result: TaskResult) {
        self.status = TaskStatus::Completed;
        self.result = Some(result);
        self.completed_at = Some(std::time::Instant::now());
    }

    /// Mark the task as failed
    pub fn fail(&mut self, error: String) {
        self.status = TaskStatus::Failed(error);
        self.completed_at = Some(std::time::Instant::now());
    }

    /// Mark the task as skipped
    pub fn skip(&mut self, reason: String) {
        self.status = TaskStatus::Skipped(reason);
        self.completed_at = Some(std::time::Instant::now());
    }

    /// Get the duration from start to completion
    pub fn duration(&self) -> Option<std::time::Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }
}

/// Result of a task execution
#[derive(Debug, Clone)]
pub struct TaskResult {
    pub success: bool,
    pub output: String,
    pub metadata: TaskMetadata,
    pub error: Option<String>,
}

impl TaskResult {
    /// Create a successful result
    pub fn success(output: String, metadata: TaskMetadata) -> Self {
        TaskResult {
            success: true,
            output,
            metadata,
            error: None,
        }
    }

    /// Create a failed result
    pub fn failure(error: String) -> Self {
        TaskResult {
            success: false,
            output: String::new(),
            metadata: TaskMetadata::default(),
            error: Some(error),
        }
    }

    /// Convert from solver result
    pub fn from_solver_result(result: &SolveResult) -> Self {
        TaskResult {
            success: result.success,
            output: result.code.clone(),
            metadata: TaskMetadata {
                method: result.method.clone(),
                complexity: 0, // DifferentiableMetadata doesn't have complexity
                iterations: 0, // DifferentiableMetadata doesn't have iterations
                additional: HashMap::new(),
            },
            error: result.error.clone(),
        }
    }
}

/// Metadata about a task execution
#[derive(Debug, Clone, Default)]
pub struct TaskMetadata {
    pub method: String,
    pub complexity: usize,
    pub iterations: usize,
    pub additional: HashMap<String, String>,
}

/// Optimizations that can be applied to tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskOptimization {
    /// Skip duplicate tasks
    Deduplication,
    /// Parallelize independent tasks
    Parallelization,
    /// Cache intermediate results
    Caching,
    /// Prune unnecessary tasks
    Pruning,
    /// Merge similar tasks
    Merging,
    /// Reorder tasks for better efficiency
    Reordering,
}

/// Main task decomposer struct
pub struct TaskDecomposer {
    next_task_id: TaskId,
    tasks: HashMap<TaskId, Task>,
    strategy: DecompositionStrategy,
    completed_tasks: HashSet<TaskId>,
    optimizations: HashSet<TaskOptimization>,
    cache: HashMap<String, TaskResult>,
}

impl TaskDecomposer {
    /// Create a new task decomposer with the given strategy
    pub fn new(strategy: DecompositionStrategy) -> Self {
        TaskDecomposer {
            next_task_id: TaskId::new(1),
            tasks: HashMap::new(),
            strategy,
            completed_tasks: HashSet::new(),
            optimizations: HashSet::new(),
            cache: HashMap::new(),
        }
    }

    /// Create a decomposer with adaptive strategy selection
    pub fn adaptive() -> Self {
        TaskDecomposer {
            next_task_id: TaskId::new(1),
            tasks: HashMap::new(),
            strategy: DecompositionStrategy::Adaptive,
            completed_tasks: HashSet::new(),
            optimizations: HashSet::new(),
            cache: HashMap::new(),
        }
    }

    /// Enable specific optimizations
    pub fn enable_optimization(&mut self, opt: TaskOptimization) {
        self.optimizations.insert(opt);
    }

    /// Disable specific optimizations
    pub fn disable_optimization(&mut self, opt: &TaskOptimization) {
        self.optimizations.remove(opt);
    }

    /// Decompose a high-level problem into executable tasks
    pub fn decompose(&mut self, problem: &Problem) -> Result<Vec<TaskId>, SolverError> {
        let strategy = if matches!(self.strategy, DecompositionStrategy::Adaptive) {
            DecompositionStrategy::select_for_problem(problem)
        } else {
            self.strategy.clone()
        };

        match strategy {
            DecompositionStrategy::BySynthesisFamily => self.decompose_by_family(problem),
            DecompositionStrategy::ByTypeCharacteristics => self.decompose_by_type(problem),
            DecompositionStrategy::Hierarchical => self.decompose_hierarchical(problem),
            DecompositionStrategy::DependencyDriven => self.decompose_by_dependency(problem),
            DecompositionStrategy::Adaptive => {
                // Should have been resolved above, but fallback to family-based
                self.decompose_by_family(problem)
            }
        }
    }

    /// Decompose by synthesis method family (gradient, search, enumerative, etc.)
    fn decompose_by_family(&mut self, problem: &Problem) -> Result<Vec<TaskId>, SolverError> {
        let mut root_tasks = Vec::new();

        // Analyze problem characteristics
        let is_scalar = problem
            .examples
            .iter()
            .all(|ex| ex.inputs.iter().all(|v| matches!(v, Value::Int(_))));
        let has_array = problem
            .examples
            .iter()
            .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Array(_))));
        let has_string = problem.examples.iter().any(|ex| {
            ex.inputs.iter().any(|v| matches!(v, Value::Str(_)))
                || matches!(ex.expected, Value::Str(_))
        });
        let is_float = problem.signature.contains("-> f64");

        // Float problems get their own lane
        if is_float {
            let task = self.create_task(
                "Float Regression".to_string(),
                "Solve using float regression methods".to_string(),
            );
            self.tasks.get_mut(&task).unwrap().method_hint = Some("float_regression".to_string());
            root_tasks.push(task);
            return Ok(root_tasks);
        }

        // String synthesis path
        if has_string {
            let string_task = self.create_task(
                "String Synthesis".to_string(),
                "Synthesize string manipulation program".to_string(),
            );
            self.tasks.get_mut(&string_task).unwrap().method_hint =
                Some("string_synthesis".to_string());
            root_tasks.push(string_task);

            // Try morphology first for single-arg cases
            let single_arg = problem
                .examples
                .first()
                .map(|e| e.inputs.len() == 1)
                .unwrap_or(false);
            if single_arg {
                let morph_task = self.create_task(
                    "Morphology Analysis".to_string(),
                    "Fast morphology specialist for single-arg string transduction".to_string(),
                );
                self.tasks.get_mut(&morph_task).unwrap().method_hint =
                    Some("morph_transduction".to_string());
                self.tasks
                    .get_mut(&string_task)
                    .unwrap()
                    .add_dependency(morph_task);
                self.tasks
                    .get_mut(&morph_task)
                    .unwrap()
                    .add_dependent(string_task);
                root_tasks.push(morph_task);
            }
        }

        // Scalar gradient path (for integer-only problems)
        if is_scalar && !has_string {
            let gradient_task = self.create_task(
                "Scalar Gradient Synthesis".to_string(),
                "Differentiable synthesis using gradient-based methods".to_string(),
            );
            self.tasks.get_mut(&gradient_task).unwrap().method_hint =
                Some("gradient_only".to_string());
            root_tasks.push(gradient_task);
        }

        // Array input handling
        if has_array {
            let array_task = self.create_task(
                "Array Synthesis".to_string(),
                "Synthesize array manipulation program using specialized methods".to_string(),
            );
            self.tasks.get_mut(&array_task).unwrap().method_hint =
                Some("array_gradient".to_string());
            root_tasks.push(array_task);
        }

        // Expression-only synthesis (faster fallback)
        if is_scalar && !has_array && !has_string {
            let expr_task = self.create_task(
                "Expression Synthesis".to_string(),
                "Pure expression-based synthesis without recursion".to_string(),
            );
            self.tasks.get_mut(&expr_task).unwrap().method_hint = Some("expr_only".to_string());

            // Expression synthesis can run in parallel with gradient
            root_tasks.push(expr_task);
        }

        // Search-based synthesis (general fallback)
        let search_task = self.create_task(
            "Search-Based Synthesis".to_string(),
            "Enumerative search for program synthesis".to_string(),
        );
        self.tasks.get_mut(&search_task).unwrap().method_hint = Some("search_teacher".to_string());
        self.tasks.get_mut(&search_task).unwrap().priority = TaskPriority::Low; // Search is expensive
        root_tasks.push(search_task);

        // If we have string or scalar tasks, make search depend on them (as fallback)
        for &task_id in &root_tasks {
            if task_id != search_task {
                self.tasks
                    .get_mut(&search_task)
                    .unwrap()
                    .add_dependency(task_id);
                self.tasks
                    .get_mut(&task_id)
                    .unwrap()
                    .add_dependent(search_task);
            }
        }

        Ok(root_tasks)
    }

    /// Decompose by type characteristics (scalar, array, string, tree, etc.)
    fn decompose_by_type(&mut self, problem: &Problem) -> Result<Vec<TaskId>, SolverError> {
        let mut tasks = Vec::new();

        // Analyze input/output types across all examples
        let mut input_types = HashSet::new();
        let mut output_types = HashSet::new();

        for ex in &problem.examples {
            for input in &ex.inputs {
                match input {
                    Value::Int(_) => {
                        input_types.insert("int");
                    }
                    Value::Float(_) => {
                        input_types.insert("float");
                    }
                    Value::Str(_) => {
                        input_types.insert("string");
                    }
                    Value::Bool(_) => {
                        input_types.insert("bool");
                    }
                    Value::Array(_) => {
                        input_types.insert("array");
                    }
                    Value::Pair(_, _) => {
                        input_types.insert("pair");
                    }
                    Value::Quad(_, _, _, _) => {
                        input_types.insert("quad");
                    }
                    Value::Tree(_) => {
                        input_types.insert("tree");
                    }
                }
            }
            match &ex.expected {
                Value::Int(_) => {
                    output_types.insert("int");
                }
                Value::Float(_) => {
                    output_types.insert("float");
                }
                Value::Str(_) => {
                    output_types.insert("string");
                }
                Value::Bool(_) => {
                    output_types.insert("bool");
                }
                Value::Array(_) => {
                    output_types.insert("array");
                }
                Value::Pair(_, _) => {
                    output_types.insert("pair");
                }
                Value::Quad(_, _, _, _) => {
                    output_types.insert("quad");
                }
                Value::Tree(_) => {
                    output_types.insert("tree");
                }
            }
        }

        // Create tasks based on type analysis
        if input_types.contains("float") || output_types.contains("float") {
            let task = self.create_task(
                "Float Processing".to_string(),
                "Handle floating-point values using regression".to_string(),
            );
            tasks.push(task);
        }

        if input_types.contains("array") {
            let task = self.create_task(
                "Array Processing".to_string(),
                "Handle array inputs and operations".to_string(),
            );
            tasks.push(task);
        }

        if input_types.contains("string") || output_types.contains("string") {
            let task = self.create_task(
                "String Processing".to_string(),
                "Handle string operations and transformations".to_string(),
            );
            tasks.push(task);
        }

        if input_types.contains("tree") || output_types.contains("tree") {
            let task = self.create_task(
                "Tree Processing".to_string(),
                "Handle tree structure operations".to_string(),
            );
            tasks.push(task);
        }

        if input_types.contains("pair") || input_types.contains("quad") {
            let task = self.create_task(
                "Structured Type Processing".to_string(),
                "Handle pairs and quads as structured values".to_string(),
            );
            tasks.push(task);
        }

        // Pure scalar integer case
        if input_types.iter().all(|&t| t == "int") && output_types.contains("int") {
            let task = self.create_task(
                "Scalar Integer Synthesis".to_string(),
                "Pure integer arithmetic synthesis".to_string(),
            );
            tasks.push(task);
        }

        // Fallback task
        let fallback = self.create_task(
            "General Synthesis".to_string(),
            "Fallback synthesis for mixed or complex types".to_string(),
        );

        // Make fallback depend on specific type handlers
        for &task_id in &tasks {
            self.tasks
                .get_mut(&fallback)
                .unwrap()
                .add_dependency(task_id);
            self.tasks
                .get_mut(&task_id)
                .unwrap()
                .add_dependent(fallback);
        }
        tasks.push(fallback);

        Ok(tasks)
    }

    /// Hierarchical decomposition with subtask refinement
    fn decompose_hierarchical(&mut self, problem: &Problem) -> Result<Vec<TaskId>, SolverError> {
        let mut root_tasks = Vec::new();

        // Create analysis task
        let analysis_task = self.create_task(
            "Problem Analysis".to_string(),
            "Analyze problem structure and select optimal strategy".to_string(),
        );
        self.tasks.get_mut(&analysis_task).unwrap().priority = TaskPriority::Critical;
        root_tasks.push(analysis_task);

        // Create synthesis strategy selection task
        let strategy_task = self.create_task(
            "Strategy Selection".to_string(),
            "Select and configure synthesis strategy".to_string(),
        );
        self.tasks
            .get_mut(&strategy_task)
            .unwrap()
            .add_dependency(analysis_task);
        self.tasks
            .get_mut(&analysis_task)
            .unwrap()
            .add_dependent(strategy_task);
        self.tasks.get_mut(&strategy_task).unwrap().parent_task = Some(analysis_task);
        root_tasks.push(strategy_task);

        // Create execution phase task
        let execution_task = self.create_task(
            "Synthesis Execution".to_string(),
            "Execute selected synthesis strategy".to_string(),
        );
        self.tasks
            .get_mut(&execution_task)
            .unwrap()
            .add_dependency(strategy_task);
        self.tasks
            .get_mut(&strategy_task)
            .unwrap()
            .add_dependent(execution_task);
        self.tasks.get_mut(&execution_task).unwrap().parent_task = Some(strategy_task);
        root_tasks.push(execution_task);

        // Create verification task
        let verification_task = self.create_task(
            "Solution Verification".to_string(),
            "Verify synthesized solution against examples".to_string(),
        );
        self.tasks
            .get_mut(&verification_task)
            .unwrap()
            .add_dependency(execution_task);
        self.tasks
            .get_mut(&execution_task)
            .unwrap()
            .add_dependent(verification_task);
        self.tasks.get_mut(&verification_task).unwrap().parent_task = Some(execution_task);
        root_tasks.push(verification_task);

        // Create optimization task (optional, depends on verification)
        let optimization_task = self.create_task(
            "Solution Optimization".to_string(),
            "Optimize successful solution if needed".to_string(),
        );
        self.tasks
            .get_mut(&optimization_task)
            .unwrap()
            .add_dependency(verification_task);
        self.tasks
            .get_mut(&verification_task)
            .unwrap()
            .add_dependent(optimization_task);
        self.tasks.get_mut(&optimization_task).unwrap().parent_task = Some(verification_task);
        root_tasks.push(optimization_task);

        Ok(root_tasks)
    }

    /// Dependency-driven decomposition based on constraints
    fn decompose_by_dependency(&mut self, problem: &Problem) -> Result<Vec<TaskId>, SolverError> {
        let mut tasks = Vec::new();

        // Input validation task
        let validation_task = self.create_task(
            "Input Validation".to_string(),
            "Validate and normalize input examples".to_string(),
        );
        self.tasks.get_mut(&validation_task).unwrap().priority = TaskPriority::Critical;
        tasks.push(validation_task);

        // Signature analysis
        let signature_task = self.create_task(
            "Signature Analysis".to_string(),
            "Analyze function signature and parameter types".to_string(),
        );
        self.tasks
            .get_mut(&signature_task)
            .unwrap()
            .add_dependency(validation_task);
        self.tasks
            .get_mut(&validation_task)
            .unwrap()
            .add_dependent(signature_task);
        tasks.push(signature_task);

        // Pattern discovery
        let pattern_task = self.create_task(
            "Pattern Discovery".to_string(),
            "Discover patterns in example inputs/outputs".to_string(),
        );
        self.tasks
            .get_mut(&pattern_task)
            .unwrap()
            .add_dependency(signature_task);
        self.tasks
            .get_mut(&signature_task)
            .unwrap()
            .add_dependent(pattern_task);
        tasks.push(pattern_task);

        // Method selection
        let method_task = self.create_task(
            "Method Selection".to_string(),
            "Select best synthesis method based on analysis".to_string(),
        );
        self.tasks
            .get_mut(&method_task)
            .unwrap()
            .add_dependency(pattern_task);
        self.tasks
            .get_mut(&pattern_task)
            .unwrap()
            .add_dependent(method_task);
        tasks.push(method_task);

        // Synthesis execution
        let synthesis_task = self.create_task(
            "Synthesis Execution".to_string(),
            "Execute synthesis with selected method".to_string(),
        );
        self.tasks
            .get_mut(&synthesis_task)
            .unwrap()
            .add_dependency(method_task);
        self.tasks
            .get_mut(&method_task)
            .unwrap()
            .add_dependent(synthesis_task);
        tasks.push(synthesis_task);

        Ok(tasks)
    }

    /// Create a new task with auto-incremented ID
    fn create_task(&mut self, name: String, description: String) -> TaskId {
        let id = self.next_task_id;
        self.next_task_id = id.next();

        let mut task = Task::new(id, name, description);

        // Set priority based on task characteristics
        let is_foundation = task.name.contains("Analysis")
            || task.name.contains("Validation")
            || task.name.contains("Signature");
        task.priority = TaskPriority::estimate(is_foundation, task.estimated_complexity, 1);

        self.tasks.insert(id, task);
        id
    }

    /// Get a task by ID
    pub fn get_task(&self, id: TaskId) -> Option<&Task> {
        self.tasks.get(&id)
    }

    /// Get a mutable reference to a task
    pub fn get_task_mut(&mut self, id: TaskId) -> Option<&mut Task> {
        self.tasks.get_mut(&id)
    }

    /// Get all tasks
    pub fn all_tasks(&self) -> &HashMap<TaskId, Task> {
        &self.tasks
    }

    /// Get ready tasks (dependencies satisfied)
    pub fn ready_tasks(&self) -> Vec<TaskId> {
        self.tasks
            .iter()
            .filter(|(_, task)| {
                task.status == TaskStatus::Pending
                    && task.dependencies_satisfied(&self.completed_tasks)
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Record the real result produced by a bound executor for a ready task.
    ///
    /// `TaskDecomposer` deliberately does not invent execution. Callers must run
    /// an actual solver/tool and supply its evidence-bearing result here.
    pub fn record_task_result(
        &mut self,
        id: TaskId,
        result: TaskResult,
    ) -> Result<&Task, SolverError> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or_else(|| SolverError::ConfigurationError(format!("Task {} not found", id)))?;

        if task.status != TaskStatus::Pending {
            return Err(SolverError::ConfigurationError(format!(
                "Task {} is not pending (current status: {:?})",
                id, task.status
            )));
        }

        if !task.dependencies_satisfied(&self.completed_tasks) {
            return Err(SolverError::ConfigurationError(format!(
                "Task {} has unsatisfied dependencies",
                id
            )));
        }

        task.start();

        if result.success && self.optimizations.contains(&TaskOptimization::Caching) {
            if let Some(hint) = &task.method_hint {
                let cache_key = format!("{}:{}", hint, task.name);
                self.cache.insert(cache_key, result.clone());
            }
        }

        if result.success {
            task.complete(result.clone());
            self.completed_tasks.insert(id);
        } else {
            task.fail(result.error.unwrap_or_else(|| "Unknown error".to_string()));
        }

        Ok(self.tasks.get(&id).unwrap())
    }

    /// Reject the old no-executor API instead of fabricating task completion.
    #[deprecated(note = "bind a real executor and call record_task_result")]
    pub fn execute_task(&mut self, id: TaskId) -> Result<&Task, SolverError> {
        Err(SolverError::ConfigurationError(format!(
            "task {id} has no bound executor; call record_task_result with real execution evidence"
        )))
    }

    /// Optimize the task graph by pruning unnecessary tasks
    pub fn optimize_prune(&mut self) -> usize {
        if !self.optimizations.contains(&TaskOptimization::Pruning) {
            return 0;
        }

        let mut to_prune = Vec::new();

        // Find tasks that are skipped or have no dependents and aren't root tasks
        for (id, task) in &self.tasks {
            if task.dependents.is_empty()
                && task.parent_task.is_some()
                && task.status == TaskStatus::Pending
            {
                to_prune.push(*id);
            }
        }

        // Mark pruned tasks as skipped
        for id in to_prune.iter() {
            if let Some(task) = self.tasks.get_mut(id) {
                task.skip("Pruned during optimization".to_string());
            }
        }

        to_prune.len()
    }

    /// Optimize the task graph by reordering for better parallelization
    pub fn optimize_reorder(&mut self) -> Vec<TaskId> {
        if !self.optimizations.contains(&TaskOptimization::Reordering) {
            return Vec::new();
        }

        // Get ready tasks sorted by priority
        let mut ready = self.ready_tasks();
        ready.sort_by(|a, b| {
            let task_a = self.tasks.get(a);
            let task_b = self.tasks.get(b);
            match (task_a, task_b) {
                (Some(ta), Some(tb)) => tb
                    .priority
                    .cmp(&ta.priority)
                    .then_with(|| ta.id.cmp(&tb.id)),
                _ => std::cmp::Ordering::Equal,
            }
        });

        ready
    }

    /// Integrate with existing solver by converting problem to tasks
    pub fn integrate_solver(
        &mut self,
        problem: &Problem,
        solver_fn: impl Fn(&Problem) -> SolveResult,
    ) -> Result<SolveResult, SolverError> {
        // Decompose first so planning state remains observable even if synthesis fails.
        let task_ids = self.decompose(problem)?;

        // Invoke the existing solver exactly once. The decomposer is an orchestration
        // layer around the solver, not a replacement that fabricates a solution.
        let solve_result = solver_fn(problem);
        self.record_integrated_solver_outcome(&task_ids, &solve_result)?;
        if !solve_result.success {
            return Err(SolverError::NoSolutionFound(
                solve_result
                    .error
                    .clone()
                    .unwrap_or_else(|| "Existing solver returned no solution".to_string()),
            ));
        }

        Ok(solve_result)
    }

    /// Attach one real solver outcome to the matching plan node and explicitly
    /// skip every node that was not independently executed. This preserves an
    /// honest trace instead of marking a decorative plan as completed.
    fn record_integrated_solver_outcome(
        &mut self,
        task_ids: &[TaskId],
        solve_result: &SolveResult,
    ) -> Result<(), SolverError> {
        let executed_id = task_ids
            .iter()
            .copied()
            .find(|id| {
                self.tasks.get(id).is_some_and(|task| {
                    task.method_hint.as_deref() == Some(solve_result.method.as_str())
                        || (task.method_hint.as_deref() == Some("search_teacher")
                            && solve_result.method.starts_with("search_"))
                })
            })
            .or_else(|| {
                task_ids.iter().copied().find(|id| {
                    self.tasks
                        .get(id)
                        .is_some_and(|task| task.name == "Synthesis Execution")
                })
            })
            .or_else(|| task_ids.last().copied())
            .ok_or_else(|| SolverError::ConfigurationError("empty task plan".to_string()))?;

        for id in task_ids {
            let task = self
                .tasks
                .get_mut(id)
                .ok_or_else(|| SolverError::ConfigurationError(format!("Task {} not found", id)))?;
            if *id == executed_id {
                task.start();
                if solve_result.success {
                    task.complete(TaskResult::from_solver_result(solve_result));
                    self.completed_tasks.insert(*id);
                } else {
                    task.fail(
                        solve_result
                            .error
                            .clone()
                            .unwrap_or_else(|| "solver returned no solution".to_string()),
                    );
                }
            } else {
                task.skip("not independently executed by integrated solver".to_string());
            }
        }

        Ok(())
    }

    /// Reset the decomposer state for a new problem
    pub fn reset(&mut self) {
        self.next_task_id = TaskId::new(1);
        self.tasks.clear();
        self.completed_tasks.clear();
    }

    /// Get statistics about the current decomposition
    pub fn stats(&self) -> DecomposerStats {
        let total = self.tasks.len();
        let completed = self.completed_tasks.len();
        let pending = self
            .tasks
            .values()
            .filter(|task| {
                matches!(
                    task.status,
                    TaskStatus::Pending
                        | TaskStatus::Ready
                        | TaskStatus::InProgress
                        | TaskStatus::Blocked
                )
            })
            .count();
        let failed = self
            .tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Failed(_)))
            .count();
        let skipped = self
            .tasks
            .values()
            .filter(|t| matches!(t.status, TaskStatus::Skipped(_)))
            .count();

        DecomposerStats {
            total_tasks: total,
            completed_tasks: completed,
            pending_tasks: pending,
            failed_tasks: failed,
            skipped_tasks: skipped,
            cache_hits: self.cache.len(),
        }
    }
}

/// Statistics about the decomposer state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecomposerStats {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub pending_tasks: usize,
    pub failed_tasks: usize,
    pub skipped_tasks: usize,
    pub cache_hits: usize,
}

impl fmt::Display for DecomposerStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Decomposer Statistics:")?;
        writeln!(f, "  Total tasks: {}", self.total_tasks)?;
        writeln!(f, "  Completed: {}", self.completed_tasks)?;
        writeln!(f, "  Pending: {}", self.pending_tasks)?;
        writeln!(f, "  Failed: {}", self.failed_tasks)?;
        writeln!(f, "  Skipped: {}", self.skipped_tasks)?;
        writeln!(f, "  Cache entries: {}", self.cache_hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    fn create_test_problem(name: &str, examples: Vec<Example>) -> Problem {
        Problem {
            name: name.to_string(),
            category: "test",
            description: "",
            signature: "fn test(x: i64) -> i64",
            examples,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        }
    }

    fn create_scalar_example(input: i64, output: i64) -> Example {
        Example {
            inputs: vec![Value::Int(input)],
            expected: Value::Int(output),
        }
    }

    fn create_array_example(input: Vec<i64>, output: i64) -> Example {
        Example {
            inputs: vec![Value::Array(input)],
            expected: Value::Int(output),
        }
    }

    fn observed_task_result(method: &str) -> TaskResult {
        TaskResult::success(
            "real executor evidence".to_string(),
            TaskMetadata {
                method: method.to_string(),
                complexity: 1,
                iterations: 1,
                additional: HashMap::from([("evidence".to_string(), "test-run".to_string())]),
            },
        )
    }

    #[test]
    fn test_task_id_increment() {
        let id1 = TaskId::new(1);
        let id2 = id1.next();
        assert_eq!(id2.0, 2);
    }

    #[test]
    fn test_task_lifecycle() {
        let mut task = Task::new(
            TaskId::new(1),
            "Test Task".to_string(),
            "A test task".to_string(),
        );

        assert!(matches!(task.status, TaskStatus::Pending));

        let mut completed = HashSet::new();
        assert!(task.dependencies_satisfied(&completed));

        task.start();
        assert!(matches!(task.status, TaskStatus::InProgress));
        assert!(task.started_at.is_some());

        let result = TaskResult::success("output".to_string(), TaskMetadata::default());
        task.complete(result);
        assert!(matches!(task.status, TaskStatus::Completed));
        assert!(task.completed_at.is_some());
        assert!(task.duration().is_some());
    }

    #[test]
    fn test_task_dependencies() {
        let mut task1 = Task::new(
            TaskId::new(1),
            "Task 1".to_string(),
            "First task".to_string(),
        );
        let mut task2 = Task::new(
            TaskId::new(2),
            "Task 2".to_string(),
            "Second task".to_string(),
        );

        task2.add_dependency(task1.id);
        task1.add_dependent(task2.id);

        let mut completed = HashSet::new();
        assert!(!task2.dependencies_satisfied(&completed));

        completed.insert(task1.id);
        assert!(task2.dependencies_satisfied(&completed));
    }

    #[test]
    fn test_task_priority_estimation() {
        let critical = TaskPriority::estimate(true, 50, 5);
        assert_eq!(critical, TaskPriority::Critical);

        let low = TaskPriority::estimate(false, 150, 10);
        assert_eq!(low, TaskPriority::Low);

        let high = TaskPriority::estimate(false, 30, 2);
        assert_eq!(high, TaskPriority::High);
    }

    #[test]
    fn test_decomposer_creation() {
        let decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);
        assert_eq!(decomposer.next_task_id.0, 1);
        assert!(decomposer.tasks.is_empty());
    }

    #[test]
    fn test_decompose_by_family_scalar() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);
        let examples = vec![create_scalar_example(1, 2), create_scalar_example(2, 4)];
        let problem = create_test_problem("double", examples);

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();
        assert!(!task_ids.is_empty());

        // Should have gradient, expression, and search tasks
        assert!(task_ids.len() >= 3);

        // Check that tasks were created
        assert!(!decomposer.all_tasks().is_empty());
    }

    #[test]
    fn test_decompose_by_family_array() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);
        let examples = vec![
            create_array_example(vec![1, 2, 3], 6),
            create_array_example(vec![4, 5], 9),
        ];
        let problem = create_test_problem("sum_array", examples);

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();
        assert!(!task_ids.is_empty());

        // Should have array-specific tasks
        let has_array_task = task_ids.iter().any(|&id| {
            decomposer
                .get_task(id)
                .map(|t| t.name.contains("Array"))
                .unwrap_or(false)
        });
        assert!(has_array_task);
    }

    #[test]
    fn test_decompose_by_type() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::ByTypeCharacteristics);
        let examples = vec![create_scalar_example(1, 2), create_scalar_example(3, 6)];
        let problem = create_test_problem("scalar_test", examples);

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();
        assert!(!task_ids.is_empty());

        // Should have scalar integer task
        let has_scalar = task_ids.iter().any(|&id| {
            decomposer
                .get_task(id)
                .map(|t| t.name.contains("Scalar"))
                .unwrap_or(false)
        });
        assert!(has_scalar);
    }

    #[test]
    fn test_hierarchical_decomposition() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::Hierarchical);
        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("hierarchical_test", examples);

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();
        // Should have analysis, strategy, execution, verification, optimization
        assert!(task_ids.len() >= 5);

        // Check dependency chain
        let analysis = decomposer.get_task(task_ids[0]).unwrap();
        let strategy = decomposer.get_task(task_ids[1]).unwrap();

        assert!(strategy.dependencies.contains(&analysis.id));
        assert!(analysis.dependents.contains(&strategy.id));
        assert_eq!(strategy.parent_task, Some(analysis.id));
    }

    #[test]
    fn test_dependency_driven_decomposition() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::DependencyDriven);
        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("dep_test", examples);

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();
        // Should have validation, signature, pattern, method, synthesis
        assert!(task_ids.len() >= 5);

        // Verify dependency chain
        let mut current_id = None;
        for task_id in &task_ids {
            let task = decomposer.get_task(*task_id).unwrap();
            if let Some(prev_id) = current_id {
                // Each subsequent task should depend on the previous
                assert!(task.dependencies.contains(&prev_id));
            }
            current_id = Some(*task_id);
        }
    }

    #[test]
    fn test_adaptive_strategy_selection() {
        // Test scalar problem selection
        let scalar_examples = vec![create_scalar_example(1, 2)];
        let mut scalar_problem = create_test_problem("scalar", scalar_examples);
        scalar_problem.recursive_allowed = false;

        let strategy = DecompositionStrategy::select_for_problem(&scalar_problem);
        assert_eq!(strategy, DecompositionStrategy::BySynthesisFamily);

        // Test recursive problem selection
        let mut recursive_examples = vec![create_scalar_example(1, 2)];
        let mut recursive_problem = create_test_problem("recursive", recursive_examples);
        recursive_problem.recursive_allowed = true;

        let strategy = DecompositionStrategy::select_for_problem(&recursive_problem);
        assert_eq!(strategy, DecompositionStrategy::Hierarchical);
    }

    #[test]
    fn test_ready_tasks() {
        let mut decomposer = TaskDecomposer::adaptive();
        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("ready_test", examples);

        decomposer.decompose(&problem).unwrap();

        // Initially, root tasks should be ready
        let ready = decomposer.ready_tasks();
        assert!(!ready.is_empty());

        // After completing a task, dependents might become ready
        if let Some(&first_id) = ready.first() {
            decomposer
                .record_task_result(first_id, observed_task_result("test_executor"))
                .unwrap();

            let new_ready = decomposer.ready_tasks();
            // Should have different ready tasks now
            assert!(new_ready != ready || new_ready.is_empty());
        }
    }

    #[test]
    fn test_optimization_pruning() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::Hierarchical);
        decomposer.enable_optimization(TaskOptimization::Pruning);

        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("prune_test", examples);
        decomposer.decompose(&problem).unwrap();

        // Mark some tasks as completed
        for _ in 0..2 {
            if let Some(task_id) = decomposer.ready_tasks().into_iter().next() {
                decomposer
                    .record_task_result(task_id, observed_task_result("test_executor"))
                    .unwrap();
            }
        }

        // Run pruning
        let pruned = decomposer.optimize_prune();
        // Pruning should handle cases where tasks have no dependents
        assert!(pruned <= decomposer.all_tasks().len());
    }

    #[test]
    fn test_optimization_reordering() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);
        decomposer.enable_optimization(TaskOptimization::Reordering);

        let examples = vec![create_scalar_example(1, 2), create_scalar_example(3, 4)];
        let problem = create_test_problem("reorder_test", examples);
        decomposer.decompose(&problem).unwrap();

        let reordered = decomposer.optimize_reorder();
        // Should return ready tasks ordered by priority
        assert!(!reordered.is_empty());

        // Check that critical tasks come first
        let mut priorities: Vec<_> = reordered
            .iter()
            .filter_map(|&id| decomposer.get_task(id))
            .map(|t| t.priority)
            .collect();

        // Priorities should be in descending order (critical first)
        if priorities.len() > 1 {
            for i in 1..priorities.len() {
                assert!(priorities[i - 1] >= priorities[i]);
            }
        }
    }

    #[test]
    fn test_cache_optimization() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);
        decomposer.enable_optimization(TaskOptimization::Caching);

        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("cache_test", examples);
        decomposer.decompose(&problem).unwrap();

        let task_ids = decomposer.ready_tasks();
        if let Some(&first_id) = task_ids.first() {
            // Execute task (should cache result)
            decomposer
                .record_task_result(first_id, observed_task_result("test_executor"))
                .unwrap();

            // Check cache has entries
            assert!(decomposer.cache.len() > 0);
        }
    }

    #[test]
    fn test_decomposer_stats() {
        let mut decomposer = TaskDecomposer::adaptive();
        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("stats_test", examples);

        decomposer.decompose(&problem).unwrap();

        let stats = decomposer.stats();
        assert_eq!(stats.total_tasks, decomposer.tasks.len());
        assert_eq!(stats.completed_tasks, 0);

        // Execute a task
        let ready = decomposer.ready_tasks();
        if let Some(&first_id) = ready.first() {
            decomposer
                .record_task_result(first_id, observed_task_result("test_executor"))
                .unwrap();

            let new_stats = decomposer.stats();
            assert_eq!(new_stats.completed_tasks, 1);
        }
    }

    #[test]
    fn test_decomposer_reset() {
        let mut decomposer = TaskDecomposer::adaptive();
        let examples = vec![create_scalar_example(1, 2)];
        let problem = create_test_problem("reset_test", examples);

        decomposer.decompose(&problem).unwrap();
        assert!(!decomposer.tasks.is_empty());

        decomposer.reset();
        assert!(decomposer.tasks.is_empty());
        assert!(decomposer.completed_tasks.is_empty());
        assert_eq!(decomposer.next_task_id.0, 1);
    }

    #[test]
    fn test_task_result_conversion() {
        let solver_result = SolveResult {
            success: true,
            code: "fn test() { 42 }".to_string(),
            method: "search".to_string(),
            error: None,
            metadata: crate::differentiable::DifferentiableMetadata::default(),
        };

        let task_result = TaskResult::from_solver_result(&solver_result);
        assert!(task_result.success);
        assert_eq!(task_result.output, "fn test() { 42 }");
        assert_eq!(task_result.metadata.method, "search");
        // complexity and iterations are now 0 since DifferentiableMetadata doesn't have them
        assert_eq!(task_result.metadata.complexity, 0);
        assert_eq!(task_result.metadata.iterations, 0);
    }

    #[test]
    fn test_decompose_multi_arg_problem() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::Adaptive);

        // Create multi-argument problem
        let examples = vec![Example {
            inputs: vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)],
            expected: Value::Int(10),
        }];
        let mut problem = create_test_problem("multi_arg", examples);
        problem.signature = "fn multi_arg(a: i64, b: i64, c: i64, d: i64) -> i64";

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        // Multi-arg problems should use dependency-driven strategy
        let task_ids = result.unwrap();
        assert!(!task_ids.is_empty());
    }

    #[test]
    fn test_decompose_recursive_problem() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::Adaptive);

        let examples = vec![create_scalar_example(5, 120)]; // 5! = 120
        let mut problem = create_test_problem("factorial", examples);
        problem.recursive_allowed = true;

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        // Recursive problems should use hierarchical strategy
        let task_ids = result.unwrap();

        // Verify hierarchical structure exists
        let has_analysis = task_ids.iter().any(|&id| {
            decomposer
                .get_task(id)
                .map(|t| t.name.contains("Analysis"))
                .unwrap_or(false)
        });
        assert!(has_analysis);
    }

    #[test]
    fn test_task_metadata() {
        let mut metadata = TaskMetadata::default();
        metadata.method = "test_method".to_string();
        metadata.complexity = 42;
        metadata.iterations = 100;
        metadata
            .additional
            .insert("key".to_string(), "value".to_string());

        assert_eq!(metadata.method, "test_method");
        assert_eq!(metadata.complexity, 42);
        assert_eq!(metadata.iterations, 100);
        assert_eq!(metadata.additional.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_string_output_problem() {
        let mut decomposer = TaskDecomposer::new(DecompositionStrategy::BySynthesisFamily);

        let examples = vec![Example {
            inputs: vec![Value::Str("hello".to_string())],
            expected: Value::Str("HELLO".to_string()),
        }];
        let mut problem = create_test_problem("uppercase", examples);
        problem.signature = "fn uppercase(s: string) -> string";

        let result = decomposer.decompose(&problem);
        assert!(result.is_ok());

        let task_ids = result.unwrap();

        // Should have string synthesis task
        let has_string = task_ids.iter().any(|&id| {
            decomposer
                .get_task(id)
                .map(|t| t.name.contains("String"))
                .unwrap_or(false)
        });
        assert!(has_string);
    }

    #[test]
    fn test_integrate_solver_invokes_real_solver_once() {
        use std::cell::Cell;

        let mut decomposer = TaskDecomposer::adaptive();
        let problem = create_test_problem("integrated", vec![create_scalar_example(2, 4)]);
        let calls = Cell::new(0usize);

        let result = decomposer
            .integrate_solver(&problem, |_| {
                calls.set(calls.get() + 1);
                SolveResult {
                    success: true,
                    code: "fn integrated(x: i64) -> i64 { x * 2 }".to_string(),
                    method: "search_affine".to_string(),
                    error: None,
                    metadata: crate::differentiable::DifferentiableMetadata::default(),
                }
            })
            .unwrap();

        assert_eq!(calls.get(), 1);
        assert_eq!(result.method, "search_affine");
        assert!(result.code.contains("x * 2"));
        let stats = decomposer.stats();
        assert_eq!(stats.completed_tasks, 1);
        assert_eq!(stats.skipped_tasks, stats.total_tasks - 1);
        assert_eq!(stats.pending_tasks, 0);
    }

    #[test]
    fn test_integrate_solver_propagates_failure() {
        let mut decomposer = TaskDecomposer::adaptive();
        let problem = create_test_problem("unsolved", vec![create_scalar_example(2, 4)]);

        let error = decomposer
            .integrate_solver(&problem, |_| SolveResult {
                success: false,
                code: String::new(),
                method: "none".to_string(),
                error: Some("portfolio exhausted".to_string()),
                metadata: crate::differentiable::DifferentiableMetadata::default(),
            })
            .unwrap_err();

        assert!(
            matches!(error, SolverError::NoSolutionFound(message) if message == "portfolio exhausted")
        );
        assert_eq!(decomposer.stats().completed_tasks, 0);
        assert_eq!(decomposer.stats().failed_tasks, 1);
        assert_eq!(decomposer.stats().pending_tasks, 0);
    }

    #[test]
    #[allow(deprecated)]
    fn execute_task_without_executor_is_rejected() {
        let mut decomposer = TaskDecomposer::adaptive();
        let problem = create_test_problem("no_executor", vec![create_scalar_example(2, 4)]);
        let task_id = decomposer.decompose(&problem).unwrap()[0];

        let error = decomposer.execute_task(task_id).unwrap_err();
        assert!(
            matches!(error, SolverError::ConfigurationError(message) if message.contains("no bound executor"))
        );
        assert_eq!(
            decomposer.get_task(task_id).unwrap().status,
            TaskStatus::Pending
        );
    }
}
