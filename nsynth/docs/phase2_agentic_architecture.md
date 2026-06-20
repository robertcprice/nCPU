# Phase 2 Agentic Enhancement Architecture

**Complete Hierarchical Planning, Tool Use, Memory, and Multi-Agent System for nCPU Synthesis**

---

## Executive Summary

Phase 2 transforms nCPU from a **single-pass cascading synthesizer** into an **adaptive multi-agent system** with hierarchical planning, tool-using capabilities, persistent memory, and collaborative problem-solving. The architecture enables synthesis of complex programs through decomposition, strategic composition, and learned meta-reasoning.

### Key Innovations

1. **Hierarchical Planning System**: Breaks "build web app" into modules → functions → actions
2. **Tool Use Framework**: Git, file system, shell, HTTP, database access for real-world operations
3. **Memory System**: Synthesis caching, pattern learning, context management with cross-session persistence
4. **Multi-Agent Coordination**: Planner, Synthesizer, Validator, Optimizer agents with supervision
5. **Integration**: Seamless connection to existing solver pipeline without disruption

## Implementation Status (2026-06-19)

The **planning foundation** described by this document is implemented and verified under
`src/agent/`:

- `planning.rs` — adaptive decomposition plus the real-solver bridge
- `hierarchy.rs` — thread-safe goal trees, priorities, parent/child links, and cycle checks
- `dependencies.rs` — incremental dependency graphs, validation, and executable ordering
- `executor.rs` — asynchronous execution, bounded concurrency, retries, rollback, and progress

The existing `debate.rs` and `orchestrator.rs` modules are also exported through
`src/agent/mod.rs`. The complete agent suite currently passes 98 tests and the release library
build succeeds.

The tool framework, persistent memory subsystem, and additional specialized-agent modules later
in this document remain **planned architecture**, not completed implementation. This distinction
is intentional so the roadmap does not imply those later Phase 2 components already ship.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [File Structure](#2-file-structure)
3. [Core Types and Relationships](#3-core-types-and-relationships)
4. [Hierarchical Planning System](#4-hierarchical-planning-system)
5. [Tool Use Framework](#5-tool-use-framework)
6. [Memory System](#6-memory-system)
7. [Multi-Agent Coordination](#7-multi-agent-coordination)
8. [Integration Points](#8-integration-points)
9. [Data Flow](#9-data-flow)
10. [Communication Protocols](#10-communication-protocols)
11. [Error Handling Strategy](#11-error-handling-strategy)
12. [Performance Considerations](#12-performance-considerations)

---

## 1. Architecture Overview

### 1.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Phase 2 Agentic System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐     ┌─────────────────┐     ┌──────────────────┐       │
│  │   NL/Code     │────▶│  Planner Agent  │────▶│  Plan Hierarchy   │       │
│  │    Input      │     │  (Meta-Reasoner)│     │  (Task Tree)      │       │
│  └───────────────┘     └─────────────────┘     └──────────────────┘       │
│                                │                                             │
│                                │ decomposes                                   │
│                                ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    Task Queue & Coordinator                         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                │                                             │
│              ┌─────────────────┼─────────────────┐                        │
│              │                 │                 │                        │
│              ▼                 ▼                 ▼                        │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────────┐             │
│  │ Synthesizer   │  │  Validator   │  │    Optimizer      │             │
│  │    Agent      │  │    Agent     │  │     Agent        │             │
│  └───────┬───────┘  └──────┬───────┘  └────────┬─────────┘             │
│          │                  │                  │                        │
│          │                  │                  │                        │
│          ▼                  ▼                  ▼                        │
│  ┌───────────────────────────────────────────────────────────┐        │
│  │                   Tool Use Framework                      │        │
│  │  ┌─────┐ ┌──────┐ ┌──────┐ ┌─────┐ ┌────────┐ ┌──────┐  │        │
│  │  │ Git │ │ File │ │Shell │ │ HTTP │ │Database│ │ ...  │  │        │
│  │  └─────┘ └──────┘ └──────┘ └─────┘ └────────┘ └──────┘  │        │
│  └───────────────────────────────────────────────────────────┘        │
│                                │                                             │
│                                │ writes                                       │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────┐        │
│  │                    Memory System                           │        │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │        │
│  │  │Synthesis│ │ Pattern │ │ Context │ │ Experience DB   │ │        │
│  │  │ Cache   │ │ Memory  │ │ Memory  │ │                 │ │        │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘ │        │
│  └───────────────────────────────────────────────────────────┘        │
│                                │                                             │
│                                │ uses                                         │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────┐        │
│  │              Existing Solver Pipeline                      │        │
│  │  (Search Teachers, Differentiable, Enumerative, etc.)      │        │
│  └───────────────────────────────────────────────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Non-Breaking Integration**: All Phase 2 components layer atop existing pipeline
2. **Graceful Degradation**: Agents fall back to direct pipeline if unavailable
3. **Learned Optimization**: Routing improves with cross-session experience
4. **Parallel Processing**: Independent agents work concurrently
5. **Observability**: All decisions logged and introspectable

---

## 2. File Structure

```
src/
├── agent/                          # Existing agent module (debate, orchestrator)
│   ├── mod.rs
│   ├── debate.rs
│   └── orchestrator.rs
│
├── planning/                       # NEW: Hierarchical planning system
│   ├── mod.rs
│   ├── planner.rs                 # Main planner agent
│   ├── task.rs                    # Task hierarchy types
│   ├── decomposition.rs           # Problem decomposition logic
│   ├── strategy.rs                # Strategy selection and composition
│   └── scheduler.rs               # Task scheduling and dependencies
│
├── tools/                         # NEW: Tool use framework
│   ├── mod.rs
│   ├── registry.rs                # Tool registration and dispatch
│   ├── git.rs                     # Git operations tool
│   ├── filesystem.rs              # File system operations
│   ├── shell.rs                   # Shell command execution
│   ├── http.rs                    # HTTP client tool
│   ├── database.rs                # Database access tool
│   ├── search.rs                  # Code/search tool
│   └── sandbox.rs                 # Sandboxed execution environment
│
├── memory/                        # NEW: Enhanced memory system
│   ├── mod.rs
│   ├── synthesis_cache.rs         # Fast synthesis result caching
│   ├── pattern_memory.rs          # Learned pattern storage
│   ├── context_memory.rs          # Context-aware memory management
│   ├── experience_db.rs           # Extended experience database
│   ├── analogy.rs                 # Cross-domain analogy retrieval
│   └── consolidation.rs           # Memory consolidation and pruning
│
├── agents/                        # NEW: Specialized agents
│   ├── mod.rs
│   ├── supervisor.rs              # Meta-supervisor agent
│   ├── planner_agent.rs           # Planner specialization
│   ├── synthesizer_agent.rs       # Synthesizer specialization
│   ├── validator_agent.rs         # Validator specialization
│   ├── optimizer_agent.rs         # Optimizer specialization
│   ├── debugger_agent.rs          # Debugger specialization
│   ├── learning_agent.rs           # Learning and pattern extraction
│   └── communication.rs            # Inter-agent messaging
│
├── coordination/                  # NEW: Multi-agent coordination
│   ├── mod.rs
│   ├── coordinator.rs             # Central coordinator
│   ├── task_queue.rs              # Distributed task queue
│   ├── voting.rs                  # Consensus mechanisms
│   ├── negotiation.rs             # Resource negotiation
│   └── supervision.rs             # Supervision and oversight
│
├── solver/                        # EXISTING: Core solver (unchanged)
│   ├── mod.rs
│   ├── pipeline.rs
│   ├── search.rs
│   ├── strategy.rs
│   └── ...
│
├── benchmark.rs                    # EXISTING: Problem types
├── orchestrator.rs                # EXISTING: Orchestrator
├── strategy.rs                    # EXISTING: Strategy registry
├── solved_cache.rs               # EXISTING: Solved cache
├── method_router.rs              # EXISTING: Method routing
├── learning/experience.rs        # EXISTING: Experience tracking
└── lib.rs                         # Module exports
```

---

## 3. Core Types and Relationships

### 3.1 Task Hierarchy Types

```rust
// src/planning/task.rs

/// Unique identifier for tasks in the hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

/// A task in the planning hierarchy
#[derive(Debug, Clone)]
pub struct Task {
    pub id: TaskId,
    pub parent: Option<TaskId>,
    pub level: TaskLevel,
    pub description: String,
    pub status: TaskStatus,
    pub dependencies: Vec<TaskId>,
    pub children: Vec<TaskId>,
    pub estimate: Option<WorkEstimate>,
    pub context: TaskContext,
}

/// Level in the task hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskLevel {
    /// Module-level: "build authentication system"
    Module,
    /// Function-level: "implement password hashing"
    Function,
    /// Action-level: "call bcrypt library"
    Action,
}

/// Current status of a task
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
    Skipped,
}

/// Estimated work for a task
#[derive(Debug, Clone, Copy)]
pub struct WorkEstimate {
    pub complexity: Complexity,
    pub time_estimate: Duration,
    pub resource_requirements: Resources,
}

/// Task execution context
#[derive(Debug, Clone)]
pub struct TaskContext {
    pub problem_context: ProblemContext,
    pub available_tools: Vec<ToolId>,
    pub memory_context: MemoryContext,
    pub synthesis_hints: Vec<SynthesisHint>,
}
```

### 3.2 Planning Types

```rust
// src/planning/planner.rs

/// Planning result with hierarchical task breakdown
#[derive(Debug, Clone)]
pub struct Plan {
    pub root_task: TaskId,
    pub tasks: HashMap<TaskId, Task>,
    pub strategy: PlanStrategy,
    pub metadata: PlanMetadata,
}

/// Planning strategy used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanStrategy {
    /// Direct synthesis (existing behavior)
    Direct,
    /// Decomposed into sub-problems
    Decomposed,
    /// Composed from known patterns
    Compositional,
    /// Learned from similar problems
    Analogical,
}

/// Plan metadata
#[derive(Debug, Clone)]
pub struct PlanMetadata {
    pub confidence: f64,
    pub rationale: String,
    pub alternatives: Vec<Plan>,
    pub risk_factors: Vec<RiskFactor>,
}
```

### 3.3 Tool Types

```rust
// src/tools/mod.rs

/// Unique identifier for tools
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolId(u64);

/// A tool that agents can use
#[derive(Debug, Clone)]
pub struct Tool {
    pub id: ToolId,
    pub name: String,
    pub category: ToolCategory,
    pub description: String,
    pub capabilities: Vec<Capability>,
    pub safety: SafetyLevel,
    pub cost: ResourceCost,
}

/// Tool categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    Git,
    FileSystem,
    Shell,
    Network,
    Database,
    Search,
    Synthesis,
}

/// Tool safety level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    Safe,           // Read-only, no side effects
    Bounded,        // Limited side effects (e.g., temp files)
    Dangerous,      // Can modify state (requires approval)
    Critical,       // Irreversible changes (requires multiple approvals)
}

/// Tool execution result
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_id: ToolId,
    pub success: bool,
    pub output: ToolOutput,
    pub duration: Duration,
    pub cost: ResourceCost,
    pub side_effects: Vec<SideEffect>,
}

/// Tool output types
#[derive(Debug, Clone)]
pub enum ToolOutput {
    Text(String),
    Binary(Vec<u8>),
    Structured(serde_json::Value),
    Stream(Receiver<u8>),
}
```

### 3.4 Memory Types

```rust
// src/memory/mod.rs

/// Memory system entry point
pub struct MemorySystem {
    synthesis_cache: Arc<SynthesisCache>,
    pattern_memory: Arc<PatternMemory>,
    context_memory: Arc<ContextMemory>,
    experience_db: Arc<ExperienceDB>,
    analogy_index: Arc<AnalogyIndex>,
}

/// Cached synthesis result
#[derive(Debug, Clone)]
pub struct CachedSynthesis {
    pub problem_hash: ProblemHash,
    pub result: SolveResult,
    pub timestamp: SystemTime,
    pub hit_count: u64,
    pub confidence: f64,
}

/// Learned pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: PatternId,
    pub pattern_type: PatternType,
    pub abstraction: ProgramAbstraction,
    pub effectiveness: EffectivenessStats,
    pub source_problems: Vec<ProblemHash>,
}

/// Context-aware memory entry
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub context_id: ContextId,
    pub task_context: TaskContext,
    pub relevant_patterns: Vec<PatternId>,
    pub relevant_experience: Vec<ExperienceId>,
    pub temporal_context: TemporalContext,
}
```

### 3.5 Agent Types

```rust
// src/agents/mod.rs

/// Base agent trait
pub trait Agent: Send + Sync {
    fn agent_id(&self) -> AgentId;
    fn capabilities(&self) -> &[Capability];
    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse;
    fn status(&self) -> AgentStatus;
}

/// Specialized agent types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentId {
    Supervisor,          // Meta-supervisor
    Planner,            // Planning specialist
    Synthesizer,        // Synthesis specialist
    Validator,          // Validation specialist
    Optimizer,          // Optimization specialist
    Debugger,           // Debugging specialist
    Learning,           // Pattern extraction and learning
}

/// Inter-agent messages
#[derive(Debug, Clone)]
pub enum AgentMessage {
    TaskAssignment { task: TaskId, priority: Priority },
    ProgressUpdate { task: TaskId, progress: f64 },
    SolutionProposal { task: TaskId, solution: Solution },
    ValidationRequest { solution: Solution },
    OptimizationRequest { solution: Solution },
    ErrorReport { task: TaskId, error: AgentError },
    MemoryQuery { query: MemoryQuery },
    ToolUseRequest { tool: ToolId, args: ToolArgs },
}

/// Agent responses
#[derive(Debug, Clone)]
pub enum AgentResponse {
    TaskAccepted { task: TaskId },
    TaskCompleted { task: TaskId, result: TaskResult },
    SolutionValidated { validation: ValidationResult },
    SolutionOptimized { solution: Solution },
    MemoryResponse { data: MemoryData },
    ToolResult { result: ToolResult },
    Error { error: AgentError },
}
```

### 3.6 Type Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Type Relationships                                │
└─────────────────────────────────────────────────────────────────────────────┘

    Plan                Agent             Tool              Memory
     │                   │                 │                  │
     ├─ Task ────────┬──┤             ToolResult         CachedEntry
     │                │  │                 │                  │
     ├─ TaskLevel      │  │             ToolOutput        Pattern
     │                │  │                 │                  │
     ├─ TaskStatus     │  │             SideEffect        ContextEntry
     │                │  │                 │                  │
     └─ WorkEstimate  │  │             SafetyLevel       Experience
                      │  │                                    │
                      │  └─ AgentMessage ─────────────────────┤
                      │                                       │
                      └─ AgentResponse ◄───────────────────────┘
```

---

## 4. Hierarchical Planning System

### 4.1 Overview

The planning system transforms high-level requests ("build web app") into executable task hierarchies:

```
"Build authentication system"
    │
    ├─ Module: User Management
    │   ├─ Function: user_registration
    │   │   ├─ Action: hash_password (call bcrypt)
    │   │   ├─ Action: store_user (database write)
    │   │   └─ Action: send_confirmation (HTTP request)
    │   └─ Function: user_login
    │       ├─ Action: verify_password (bcrypt compare)
    │       ├─ Action: generate_token (crypto operation)
    │       └─ Action: create_session (database write)
    └─ Module: Session Management
        └─ Function: validate_session
            └─ Action: check_token (crypto verify)
```

### 4.2 Planner Architecture

```rust
// src/planning/planner.rs

pub struct Planner {
    memory: Arc<MemorySystem>,
    tools: Arc<ToolRegistry>,
    strategy_selector: StrategySelector,
    decomposition_engine: DecompositionEngine,
}

impl Planner {
    /// Create a plan from a natural language or code-based request
    pub async fn plan(&self, request: &PlanningRequest) -> Result<Plan> {
        // 1. Analyze request type
        let request_type = self.analyze_request(request)?;

        // 2. Select planning strategy
        let strategy = self.strategy_selector.select(&request_type);

        // 3. Generate initial plan
        let mut plan = match strategy {
            PlanStrategy::Direct => self.plan_direct(request),
            PlanStrategy::Decomposed => self.plan_decomposed(request),
            PlanStrategy::Compositional => self.plan_compositional(request),
            PlanStrategy::Analogical => self.plan_analogical(request),
        }?;

        // 4. Validate plan completeness
        self.validate_plan(&mut plan)?;

        // 5. Estimate work for each task
        self.estimate_work(&mut plan);

        // 6. Add context and hints
        self.annotate_context(&mut plan);

        Ok(plan)
    }

    /// Decompose a complex task into sub-tasks
    fn plan_decomposed(&self, request: &PlanningRequest) -> Result<Plan> {
        let root = Task {
            id: TaskId::new(),
            parent: None,
            level: TaskLevel::Module,
            description: request.description.clone(),
            status: TaskStatus::Pending,
            dependencies: vec![],
            children: vec![],
            estimate: None,
            context: self.build_context(request),
        };

        // Identify sub-problems using pattern matching
        let sub_problems = self.decomposition_engine.decompose(request)?;

        let mut tasks = HashMap::new();
        tasks.insert(root.id, root);

        for sub_problem in sub_problems {
            let sub_task = self.create_task_from_problem(sub_problem, root.id)?;
            root.children.push(sub_task.id);
            tasks.insert(sub_task.id, sub_task);
        }

        Ok(Plan {
            root_task: root.id,
            tasks,
            strategy: PlanStrategy::Decomposed,
            metadata: PlanMetadata::default(),
        })
    }
}
```

### 4.3 Problem Decomposition

```rust
// src/planning/decomposition.rs

pub struct DecompositionEngine {
    pattern_matcher: PatternMatcher,
    complexity_analyzer: ComplexityAnalyzer,
}

impl DecompositionEngine {
    /// Decompose a problem into sub-problems
    pub fn decompose(&self, request: &PlanningRequest) -> Result<Vec<SubProblem>> {
        let mut sub_problems = Vec::new();

        // 1. Identify independent components
        let components = self.identify_components(request)?;

        // 2. Analyze dependencies between components
        let dependencies = self.analyze_dependencies(&components)?;

        // 3. Group components into solvable units
        let groups = self.group_components(components, dependencies)?;

        // 4. Create sub-problems for each group
        for group in groups {
            let sub_problem = SubProblem {
                id: SubProblemId::new(),
                description: group.description,
                examples: self.generate_examples(&group)?,
                complexity: self.complexity_analyzer.estimate(&group),
                dependencies: group.dependencies,
            };
            sub_problems.push(sub_problem);
        }

        Ok(sub_problems)
    }

    /// Identify independent components in a problem
    fn identify_components(&self, request: &PlanningRequest) -> Result<Vec<Component>> {
        let mut components = Vec::new();

        // Parse input/output signatures
        let signature = parse_signature(&request.signature)?;

        // Identify separate concerns based on patterns
        if let Some(validation_logic) = self.extract_validation(&signature) {
            components.push(Component {
                type_: ComponentType::Validation,
                signature: validation_logic,
            });
        }

        if let Some(core_logic) = self.extract_core_logic(&signature) {
            components.push(Component {
                type_: ComponentType::CoreLogic,
                signature: core_logic,
            });
        }

        if let Some(output_format) = self.extract_output_format(&signature) {
            components.push(Component {
                type_: ComponentType::OutputFormatting,
                signature: output_format,
            });
        }

        Ok(components)
    }
}
```

### 4.4 Strategy Selection

```rust
// src/planning/strategy.rs

pub struct StrategySelector {
    router: MethodRouter,
    experience_db: Arc<ExperienceDB>,
    pattern_memory: Arc<PatternMemory>,
}

impl StrategySelector {
    /// Select the best planning strategy for a request
    pub fn select(&self, request: &PlanningRequest) -> PlanStrategy {
        // 1. Check for direct applicability (simple problems)
        if self.is_trivial(request) {
            return PlanStrategy::Direct;
        }

        // 2. Check for known patterns (compositional)
        if let Some(pattern) = self.pattern_memory.find_match(request) {
            if pattern.effectiveness.confidence > 0.8 {
                return PlanStrategy::Compositional;
            }
        }

        // 3. Check for analogous problems
        if let Some(analogy) = self.experience_db.find_analogy(request) {
            if analogy.similarity > 0.75 {
                return PlanStrategy::Analogical;
            }
        }

        // 4. Default to decomposition for complex problems
        PlanStrategy::Decomposed
    }

    /// Check if a problem is trivial enough for direct synthesis
    fn is_trivial(&self, request: &PlanningRequest) -> bool {
        // Low arity, simple types, few examples
        request.arity <= 2
            && request.complexity == Complexity::Trivial
            && request.examples.len() <= 3
    }
}
```

### 4.5 Task Scheduling

```rust
// src/planning/scheduler.rs

pub struct TaskScheduler {
    queue: TaskQueue,
    executor: TaskExecutor,
    dependency_tracker: DependencyTracker,
}

impl TaskScheduler {
    /// Execute a plan by scheduling tasks
    pub async fn execute_plan(&self, plan: Plan) -> Result<ExecutionResult> {
        let mut execution = Execution::new(plan);

        // 1. Schedule root task
        self.queue.schedule(execution.plan.root_task, Priority::High);

        // 2. Process tasks as dependencies resolve
        while !execution.is_complete() {
            // Get next ready task
            let task_id = self.queue.next_ready().await?;

            // Execute task
            let result = self.executor.execute_task(task_id, &execution).await?;

            // Update execution state
            execution.record_result(task_id, result);

            // Resolve dependencies
            self.dependency_tracker.resolve(task_id);

            // Schedule newly ready tasks
            for child in execution.get_children(task_id) {
                if self.dependency_tracker.is_ready(child) {
                    self.queue.schedule(child, Priority::Normal);
                }
            }
        }

        Ok(execution.result())
    }
}
```

---

## 5. Tool Use Framework

### 5.1 Tool Registry

```rust
// src/tools/registry.rs

pub struct ToolRegistry {
    tools: HashMap<ToolId, Tool>,
    by_name: HashMap<String, ToolId>,
    by_category: HashMap<ToolCategory, Vec<ToolId>>,
    safety_policy: SafetyPolicy,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Tool) -> Result<()> {
        // Validate tool safety
        self.safety_policy.validate(&tool)?;

        // Register by ID
        self.tools.insert(tool.id, tool.clone());

        // Register by name
        self.by_name.insert(tool.name.clone(), tool.id);

        // Register by category
        self.by_category
            .entry(tool.category)
            .or_insert_with(Vec::new)
            .push(tool.id);

        Ok(())
    }

    pub async fn execute(&self, tool_id: ToolId, args: ToolArgs) -> Result<ToolResult> {
        let tool = self.get_tool(tool_id)?;

        // Check safety policy
        self.safety_policy.check_authorization(&tool, &args)?;

        // Execute with timeout
        let result = tokio::time::timeout(
            Duration::from_secs(60),
            tool.execute(args)
        ).await??;

        // Log execution
        self.log_execution(tool_id, &result);

        Ok(result)
    }
}
```

### 5.2 Git Tool

```rust
// src/tools/git.rs

pub struct GitTool {
    repo_path: PathBuf,
    safety: SafetyLevel,
}

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: GitOperation = args.deserialize()?;

        match operation {
            GitOperation::Clone { url, path } => {
                self.clone(url, path).await
            }
            GitOperation::Commit { message, files } => {
                self.commit(message, files).await
            }
            GitOperation::Branch { name, base } => {
                self.create_branch(name, base).await
            }
            GitOperation::Diff { ref_ } => {
                self.diff(ref_).await
            }
            GitOperation::Status => {
                self.status().await
            }
        }
    }
}

impl GitTool {
    async fn clone(&self, url: String, path: PathBuf) -> Result<ToolResult> {
        let output = Command::new("git")
            .arg("clone")
            .arg(url)
            .arg(&path)
            .output()
            .await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: output.status.success(),
            output: ToolOutput::Text(String::from_utf8_lossy(&output.stdout).to_string()),
            duration: elapsed,
            cost: ResourceCost::minimal(),
            side_effects: vec![SideEffect::FileSystemWrite],
        })
    }

    async fn commit(&self, message: String, files: Vec<PathBuf>) -> Result<ToolResult> {
        // Stage files
        for file in files {
            Command::new("git")
                .arg("add")
                .arg(&file)
                .status()
                .await?;
        }

        // Commit
        let output = Command::new("git")
            .arg("commit")
            .arg("-m")
            .arg(&message)
            .output()
            .await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: output.status.success(),
            output: ToolOutput::Text(String::from_utf8_lossy(&output.stdout).to_string()),
            duration: elapsed,
            cost: ResourceCost::minimal(),
            side_effects: vec![SideEffect::GitCommit],
        })
    }
}
```

### 5.3 File System Tool

```rust
// src/tools/filesystem.rs

pub struct FileSystemTool {
    allowed_paths: Vec<PathBuf>,
    read_only: bool,
}

impl Tool for FileSystemTool {
    fn name(&self) -> &str {
        "filesystem"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: FsOperation = args.deserialize()?;

        match operation {
            FsOperation::Read { path } => {
                self.read(path).await
            }
            FsOperation::Write { path, content } => {
                self.write(path, content).await
            }
            FsOperation::List { path, recursive } => {
                self.list(path, recursive).await
            }
            FsOperation::Delete { path } => {
                self.delete(path).await
            }
            FsOperation::Mkdir { path } => {
                self.mkdir(path).await
            }
        }
    }
}

impl FileSystemTool {
    async fn read(&self, path: PathBuf) -> Result<ToolResult> {
        self.check_allowed(&path)?;

        let content = tokio::fs::read_to_string(&path).await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: true,
            output: ToolOutput::Text(content),
            duration: elapsed,
            cost: ResourceCost::minimal(),
            side_effects: vec![],
        })
    }

    async fn write(&self, path: PathBuf, content: String) -> Result<ToolResult> {
        if self.read_only {
            return Err(Error::WriteDenied);
        }

        self.check_allowed(&path)?;

        tokio::fs::write(&path, content).await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: true,
            output: ToolOutput::Text(format!("Written to {}", path.display())),
            duration: elapsed,
            cost: ResourceCost::minimal(),
            side_effects: vec![SideEffect::FileSystemWrite],
        })
    }
}
```

### 5.4 Shell Tool

```rust
// src/tools/shell.rs

pub struct ShellTool {
    allowed_commands: HashSet<String>,
    timeout: Duration,
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: ShellOperation = args.deserialize()?;

        match operation {
            ShellOperation::Execute { command, args: cmd_args } => {
                self.execute_command(command, cmd_args).await
            }
            ShellOperation::ExecuteScript { script } => {
                self.execute_script(script).await
            }
        }
    }
}

impl ShellTool {
    async fn execute_command(&self, command: String, args: Vec<String>) -> Result<ToolResult> {
        // Check command is allowed
        if !self.allowed_commands.contains(&command) {
            return Err(Error::CommandNotAllowed(command));
        }

        // Execute with timeout
        let output = tokio::time::timeout(
            self.timeout,
            Command::new(&command)
                .args(&args)
                .output()
        ).await???;

        Ok(ToolResult {
            tool_id: self.id(),
            success: output.status.success(),
            output: ToolOutput::Text(String::from_utf8_lossy(&output.stdout).to_string()),
            duration: elapsed,
            cost: ResourceCost::minimal(),
            side_effects: vec![],
        })
    }
}
```

### 5.5 HTTP Tool

```rust
// src/tools/http.rs

pub struct HttpTool {
    client: reqwest::Client,
    allowed_domains: HashSet<String>,
}

impl Tool for HttpTool {
    fn name(&self) -> &str {
        "http"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: HttpOperation = args.deserialize()?;

        match operation {
            HttpOperation::Get { url, headers } => {
                self.get(url, headers).await
            }
            HttpOperation::Post { url, body, headers } => {
                self.post(url, body, headers).await
            }
            HttpOperation::Put { url, body, headers } => {
                self.put(url, body, headers).await
            }
        }
    }
}

impl HttpTool {
    async fn get(&self, url: String, headers: HashMap<String, String>) -> Result<ToolResult> {
        // Check domain is allowed
        let domain = extract_domain(&url)?;
        if !self.allowed_domains.contains(&domain) {
            return Err(Error::DomainNotAllowed(domain));
        }

        // Build request
        let mut request = self.client.get(&url);
        for (key, value) in headers {
            request = request.header(key, value);
        }

        // Send request
        let response = request.send().await?;
        let text = response.text().await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: response.status().is_success(),
            output: ToolOutput::Text(text),
            duration: elapsed,
            cost: ResourceCost::network(),
            side_effects: vec![],
        })
    }
}
```

### 5.6 Database Tool

```rust
// src/tools/database.rs

pub struct DatabaseTool {
    pool: SqlPool,
    allowed_queries: Vec<QueryPattern>,
    read_only: bool,
}

impl Tool for DatabaseTool {
    fn name(&self) -> &str {
        "database"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: DbOperation = args.deserialize()?;

        match operation {
            DbOperation::Query { sql, params } => {
                self.query(sql, params).await
            }
            DbOperation::Execute { sql, params } => {
                self.execute(sql, params).await
            }
            DbOperation::Schema { table } => {
                self.schema(table).await
            }
        }
    }
}

impl DatabaseTool {
    async fn query(&self, sql: String, params: Vec<Value>) -> Result<ToolResult> {
        // Check query is allowed
        self.check_query_allowed(&sql)?;

        // Execute query
        let rows = sqlx::query_as::<_, JsonRow>(&sql)
            .bind_all(params)
            .fetch_all(&self.pool)
            .await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: true,
            output: ToolOutput::Structured(serde_json::to_value(rows)?),
            duration: elapsed,
            cost: ResourceCost::database(),
            side_effects: vec![],
        })
    }
}
```

### 5.7 Sandbox Tool

```rust
// src/tools/sandbox.rs

pub struct SandboxTool {
    runtime: SandboxRuntime,
    resource_limits: ResourceLimits,
}

impl Tool for SandboxTool {
    fn name(&self) -> &str {
        "sandbox"
    }

    async fn execute(&self, args: ToolArgs) -> Result<ToolResult> {
        let operation: SandboxOperation = args.deserialize()?;

        match operation {
            SandboxOperation::ExecuteCode { code, language } => {
                self.execute_code(code, language).await
            }
            SandboxOperation::Verify { code, examples } => {
                self.verify(code, examples).await
            }
        }
    }
}

impl SandboxTool {
    async fn execute_code(&self, code: String, language: Language) -> Result<ToolResult> {
        // Create sandboxed environment
        let sandbox = self.runtime.create(self.resource_limits).await?;

        // Execute code
        let result = sandbox.execute(code, language).await?;

        Ok(ToolResult {
            tool_id: self.id(),
            success: result.success,
            output: ToolOutput::Text(result.output),
            duration: elapsed,
            cost: ResourceCost::computation(result.cpu_time),
            side_effects: vec![],
        })
    }
}
```

---

## 6. Memory System

### 6.1 Memory System Architecture

```rust
// src/memory/mod.rs

pub struct MemorySystem {
    synthesis_cache: Arc<SynthesisCache>,
    pattern_memory: Arc<PatternMemory>,
    context_memory: Arc<ContextMemory>,
    experience_db: Arc<ExperienceDB>,
    analogy_index: Arc<AnalogyIndex>,
    consolidation: ConsolidationService,
}

impl MemorySystem {
    pub async fn query(&self, query: MemoryQuery) -> Result<MemoryResponse> {
        match query {
            MemoryQuery::Synthesis { problem } => {
                self.query_synthesis(problem).await
            }
            MemoryQuery::Pattern { pattern } => {
                self.query_pattern(pattern).await
            }
            MemoryQuery::Context { context } => {
                self.query_context(context).await
            }
            MemoryQuery::Experience { features } => {
                self.query_experience(features).await
            }
            MemoryQuery::Analogy { problem } => {
                self.query_analogy(problem).await
            }
        }
    }

    pub async fn store(&self, entry: MemoryEntry) -> Result<()> {
        match entry {
            MemoryEntry::Synthesis { problem, result } => {
                self.synthesis_cache.store(problem, result).await
            }
            MemoryEntry::Pattern { pattern } => {
                self.pattern_memory.store(pattern).await
            }
            MemoryEntry::Experience { experience } => {
                self.experience_db.store(experience).await
            }
            MemoryEntry::Context { context } => {
                self.context_memory.store(context).await
            }
        }
    }

    pub async fn consolidate(&self) -> Result<ConsolidationReport> {
        self.consolidation.run().await
    }
}
```

### 6.2 Synthesis Cache

```rust
// src/memory/synthesis_cache.rs

pub struct SynthesisCache {
    cache: Arc<RwLock<HashMap<ProblemHash, CachedSynthesis>>>,
    lru_index: Arc<RwLock<LruIndex>>,
    stats: Arc<CacheStats>,
}

impl SynthesisCache {
    pub async fn get(&self, hash: ProblemHash) -> Option<CachedSynthesis> {
        let cache = self.cache.read().await;
        let entry = cache.get(&hash)?;

        // Update LRU
        let mut lru = self.lru_index.write().await;
        lru.touch(hash);

        // Update stats
        self.stats.record_hit();

        Some(entry.clone())
    }

    pub async fn store(&self, hash: ProblemHash, result: SolveResult) {
        let entry = CachedSynthesis {
            problem_hash: hash,
            result,
            timestamp: SystemTime::now(),
            hit_count: 0,
            confidence: 1.0,
        };

        let mut cache = self.cache.write().await;
        cache.insert(hash, entry);

        // Update stats
        self.stats.record_write();
    }

    pub async fn invalidate(&self, hash: ProblemHash) {
        let mut cache = self.cache.write().await;
        cache.remove(&hash);

        let mut lru = self.lru_index.write().await;
        lru.remove(hash);
    }

    pub async fn flush(&self) -> Result<()> {
        let cache = self.cache.read().await;
        let path = self.cache_path();

        let serialized = serde_json::to_string_pretty(&*cache)?;
        tokio::fs::write(path, serialized).await?;

        Ok(())
    }
}
```

### 6.3 Pattern Memory

```rust
// src/memory/pattern_memory.rs

pub struct PatternMemory {
    patterns: Arc<RwLock<HashMap<PatternId, Pattern>>>,
    by_type: Arc<RwLock<HashMap<PatternType, Vec<PatternId>>>>,
    effectiveness: EffectivenessTracker,
}

impl PatternMemory {
    pub async fn store(&self, pattern: Pattern) -> Result<()> {
        let id = pattern.id;

        // Store pattern
        let mut patterns = self.patterns.write().await;
        patterns.insert(id, pattern.clone());

        // Index by type
        let mut by_type = self.by_type.write().await;
        by_type.entry(pattern.pattern_type)
            .or_insert_with(Vec::new)
            .push(id);

        Ok(())
    }

    pub async fn find_match(&self, problem: &Problem) -> Option<Pattern> {
        // Extract features from problem
        let features = self.extract_features(problem)?;

        // Search for matching patterns
        let patterns = self.patterns.read().await;
        let matches = patterns.values()
            .filter(|p| self.matches(&p.abstraction, &features))
            .collect::<Vec<_>>();

        // Return best match by effectiveness
        matches.into_iter()
            .max_by_key(|p| self.effectiveness.score(p.id))
            .cloned()
    }

    pub async fn update_effectiveness(&self, pattern_id: PatternId, success: bool) {
        self.effectiveness.record_outcome(pattern_id, success);
    }
}
```

### 6.4 Context Memory

```rust
// src/memory/context_memory.rs

pub struct ContextMemory {
    contexts: Arc<RwLock<HashMap<ContextId, ContextEntry>>>,
    temporal_index: TemporalIndex,
}

impl ContextMemory {
    pub async fn store(&self, context: ContextEntry) -> Result<()> {
        let id = context.context_id;
        let mut contexts = self.contexts.write().await;
        contexts.insert(id, context.clone());

        // Update temporal index
        self.temporal_index.insert(id, context.temporal_context);

        Ok(())
    }

    pub async fn query_relevant(&self, query: ContextQuery) -> Vec<ContextEntry> {
        let contexts = self.contexts.read().await;

        contexts.values()
            .filter(|c| self.is_relevant(c, &query))
            .cloned()
            .collect()
    }

    fn is_relevant(&self, context: &ContextEntry, query: &ContextQuery) -> bool {
        // Check task type match
        if query.task_type != context.task_context.task_type {
            return false;
        }

        // Check temporal proximity
        if !self.temporal_index.is_proximity(
            context.context_id,
            query.temporal_window,
        ) {
            return false;
        }

        // Check feature overlap
        let query_features: HashSet<_> = query.features.iter().collect();
        let context_features: HashSet<_> = context.task_context.features.iter().collect();

        let overlap = query_features.intersection(&context_features).count();
        overlap >= query.min_feature_overlap
    }
}
```

### 6.5 Experience Database

```rust
// src/memory/experience_db.rs

pub struct ExperienceDB {
    experiences: Arc<RwLock<HashMap<ExperienceId, Experience>>>,
    by_problem: Arc<RwLock<HashMap<ProblemHash, Vec<ExperienceId>>>>,
    by_pattern: Arc<RwLock<HashMap<PatternId, Vec<ExperienceId>>>>,
    similarity_index: SimilarityIndex,
}

impl ExperienceDB {
    pub async fn store(&self, experience: Experience) -> Result<()> {
        let id = experience.id;
        let problem_hash = experience.problem.hash();

        // Store experience
        let mut experiences = self.experiences.write().await;
        experiences.insert(id, experience.clone());

        // Index by problem
        let mut by_problem = self.by_problem.write().await;
        by_problem.entry(problem_hash)
            .or_insert_with(Vec::new)
            .push(id);

        // Index by pattern
        for lesson in &experience.lessons {
            if let Some(pattern_id) = lesson.pattern_id {
                let mut by_pattern = self.by_pattern.write().await;
                by_pattern.entry(pattern_id)
                    .or_insert_with(Vec::new)
                    .push(id);
            }
        }

        // Update similarity index
        self.similarity_index.insert(id, &experience.problem);

        Ok(())
    }

    pub async fn find_analogy(&self, problem: &Problem) -> Option<Analogy> {
        let problem_hash = problem.hash();

        // Find similar problems
        let similar = self.similarity_index.find_similar(
            problem,
            min_similarity = 0.5,
            max_results = 10,
        ).await?;

        // Calculate analogy strength
        let best = similar.into_iter()
            .max_by_key(|a| a.similarity)?;

        Some(Analogy {
            source_problem: best.problem_hash,
            target_problem: problem_hash,
            similarity: best.similarity,
            analogy_type: best.analogy_type,
            transferable_lessons: best.lessons,
        })
    }

    pub async fn get_successful_methods(&self, problem: &Problem) -> Vec<MethodStat> {
        let problem_hash = problem.hash();

        let by_problem = self.by_problem.read().await;
        let experience_ids = by_problem.get(&problem_hash)?;

        let experiences = self.experiences.read().await;
        let mut methods = HashMap::new();

        for id in experience_ids {
            if let Some(exp) = experiences.get(id) {
                if exp.outcome == SolveOutcome::Success {
                    *methods.entry(exp.solution.method.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        methods.into_iter()
            .map(|(method, count)| MethodStat { method, count })
            .collect()
    }
}
```

### 6.6 Analogy Index

```rust
// src/memory/analogy.rs

pub struct AnalogyIndex {
    problem_vectors: HashMap<ProblemHash, ProblemVector>,
    similarity_cache: Arc<RwLock<SimilarityCache>>,
    embedding_model: EmbeddingModel,
}

impl AnalogyIndex {
    pub async fn find_similar(
        &self,
        problem: &Problem,
        min_similarity: f64,
        max_results: usize,
    ) -> Result<Vec<SimilarProblem>> {
        // Generate problem vector
        let target_vector = self.embedding_model.embed(problem).await?;

        let mut results = Vec::new();

        for (hash, vector) in &self.problem_vectors {
            // Check cache first
            let similarity = if let Some(cached) = self.similarity_cache.read().await.get(&(*hash, problem.hash())) {
                *cached
            } else {
                // Calculate cosine similarity
                let sim = cosine_similarity(&target_vector, vector);

                // Cache result
                self.similarity_cache.write().await.insert((*hash, problem.hash()), sim);

                sim
            };

            if similarity >= min_similarity {
                results.push(SimilarProblem {
                    problem_hash: *hash,
                    similarity,
                    analogy_type: self.classify_analogy(similarity),
                    lessons: vec![],
                });
            }
        }

        // Sort by similarity and limit
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(max_results);

        Ok(results)
    }

    fn classify_analogy(&self, similarity: f64) -> AnalogyType {
        if similarity >= 0.9 {
            AnalogyType::Identical
        } else if similarity >= 0.75 {
            AnalogyType::Structural
        } else if similarity >= 0.6 {
            AnalogyType::Domain
        } else {
            AnalogyType::Weak
        }
    }
}
```

### 6.7 Memory Consolidation

```rust
// src/memory/consolidation.rs

pub struct ConsolidationService {
    memory: Arc<MemorySystem>,
    config: ConsolidationConfig,
}

impl ConsolidationService {
    pub async fn run(&self) -> Result<ConsolidationReport> {
        let mut report = ConsolidationReport::default();

        // 1. Prune stale cache entries
        report.cache_pruned = self.prune_cache().await?;

        // 2. Consolidate similar patterns
        report.patterns_consolidated = self.consolidate_patterns().await?;

        // 3. Extract new patterns from experiences
        report.new_patterns = self.extract_patterns().await?;

        // 4. Update effectiveness scores
        report.effectiveness_updated = self.update_effectiveness().await?;

        // 5. Rebuild similarity index
        report.index_rebuilt = self.rebuild_index().await?;

        Ok(report)
    }

    async fn prune_cache(&self) -> Result<usize> {
        let cache = &self.memory.synthesis_cache;
        let mut pruned = 0;

        // Get low-confidence entries
        let entries = cache.entries().await;
        for (hash, entry) in entries {
            if entry.confidence < self.config.min_confidence {
                cache.invalidate(hash).await;
                pruned += 1;
            }
        }

        Ok(pruned)
    }

    async fn consolidate_patterns(&self) -> Result<usize> {
        let pattern_memory = &self.memory.pattern_memory;
        let mut consolidated = 0;

        // Find similar patterns
        let patterns = pattern_memory.all_patterns().await;
        let groups = self.group_similar_patterns(patterns)?;

        // Merge each group
        for group in groups {
            if group.len() > 1 {
                let merged = self.merge_patterns(group)?;
                pattern_memory.store(merged).await?;
                consolidated += group.len() - 1;
            }
        }

        Ok(consolidated)
    }
}
```

---

## 7. Multi-Agent Coordination

### 7.1 Supervisor Agent

```rust
// src/agents/supervisor.rs

pub struct SupervisorAgent {
    id: AgentId,
    coordinator: Arc<Coordinator>,
    memory: Arc<MemorySystem>,
    tools: Arc<ToolRegistry>,
}

impl Agent for SupervisorAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Supervisor
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::TaskAssignment { task, priority } => {
                self.delegate_task(task, priority).await
            }
            AgentMessage::ProgressUpdate { task, progress } => {
                self.handle_progress(task, progress).await
            }
            AgentMessage::SolutionProposal { task, solution } => {
                self.handle_proposal(task, solution).await
            }
            AgentMessage::ErrorReport { task, error } => {
                self.handle_error(task, error).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl SupervisorAgent {
    async fn delegate_task(&self, task: TaskId, priority: Priority) -> AgentResponse {
        // Get task details
        let task_details = self.coordinator.get_task(task)?;

        // Select best agent for this task
        let agent = self.select_agent(&task_details)?;

        // Delegate task
        let response = self.coordinator.send_to_agent(
            agent,
            AgentMessage::TaskAssignment { task, priority },
        ).await?;

        Ok(response)
    }

    fn select_agent(&self, task: &Task) -> Result<AgentId> {
        match task.level {
            TaskLevel::Module => Ok(AgentId::Planner),
            TaskLevel::Function => Ok(AgentId::Synthesizer),
            TaskLevel::Action => Ok(AgentId::Synthesizer),
        }
    }

    async fn handle_proposal(&self, task: TaskId, solution: Solution) -> AgentResponse {
        // Request validation
        let validation = self.coordinator.send_to_agent(
            AgentId::Validator,
            AgentMessage::ValidationRequest { solution: solution.clone() },
        ).await?;

        match validation {
            AgentResponse::SolutionValidated { validation } => {
                if validation.is_valid {
                    // Proposal is valid, request optimization
                    let optimization = self.coordinator.send_to_agent(
                        AgentId::Optimizer,
                        AgentMessage::OptimizationRequest { solution },
                    ).await?;

                    optimization
                } else {
                    // Proposal invalid, return error
                    AgentResponse::Error {
                        error: AgentError::ValidationFailed(validation.errors),
                    }
                }
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedResponse },
        }
    }
}
```

### 7.2 Planner Agent

```rust
// src/agents/planner_agent.rs

pub struct PlannerAgent {
    id: AgentId,
    planner: Arc<Planner>,
    memory: Arc<MemorySystem>,
}

impl Agent for PlannerAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Planner
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::TaskAssignment { task, priority } => {
                self.plan_task(task, priority).await
            }
            AgentMessage::MemoryQuery { query } => {
                self.query_memory(query).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl PlannerAgent {
    async fn plan_task(&self, task: TaskId, priority: Priority) -> AgentResponse {
        // Get task context
        let task_context = self.get_task_context(task)?;

        // Create planning request
        let request = PlanningRequest {
            description: task_context.description,
            signature: task_context.signature,
            examples: task_context.examples,
            constraints: task_context.constraints,
        };

        // Generate plan
        let plan = self.planner.plan(&request).await?;

        // Report plan
        AgentResponse::PlanGenerated { plan }
    }

    async fn query_memory(&self, query: MemoryQuery) -> AgentResponse {
        let response = self.memory.query(query).await?;
        AgentResponse::MemoryResponse { data: response }
    }
}
```

### 7.3 Synthesizer Agent

```rust
// src/agents/synthesizer_agent.rs

pub struct SynthesizerAgent {
    id: AgentId,
    tools: Arc<ToolRegistry>,
    memory: Arc<MemorySystem>,
    solver: Arc<Solver>,
}

impl Agent for SynthesizerAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Synthesizer
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::TaskAssignment { task, priority } => {
                self.synthesize(task).await
            }
            AgentMessage::ToolUseRequest { tool, args } => {
                self.use_tool(tool, args).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl SynthesizerAgent {
    async fn synthesize(&self, task: TaskId) -> AgentResponse {
        // Get task context
        let task_context = self.get_task_context(task)?;

        // Check synthesis cache
        let cache_key = task_context.problem_hash();
        if let Some(cached) = self.memory.synthesis_cache.get(cache_key).await {
            return AgentResponse::SolutionProposal {
                task,
                solution: Solution {
                    code: cached.result.code,
                    method: cached.result.method,
                    confidence: cached.confidence,
                    metadata: cached.metadata,
                },
            };
        }

        // Try synthesis strategies
        let result = self.solver.solve(&task_context.problem).await?;

        if result.success {
            // Cache result
            self.memory.synthesis_cache.store(cache_key, result.clone()).await;

            AgentResponse::SolutionProposal {
                task,
                solution: Solution {
                    code: result.code,
                    method: result.method,
                    confidence: 1.0,
                    metadata: result.metadata,
                },
            }
        } else {
            AgentResponse::Error {
                error: AgentError::SynthesisFailed(result.error.unwrap_or_default()),
            }
        }
    }

    async fn use_tool(&self, tool: ToolId, args: ToolArgs) -> AgentResponse {
        let result = self.tools.execute(tool, args).await?;
        AgentResponse::ToolResult { result }
    }
}
```

### 7.4 Validator Agent

```rust
// src/agents/validator_agent.rs

pub struct ValidatorAgent {
    id: AgentId,
    tools: Arc<ToolRegistry>,
    memory: Arc<MemorySystem>,
}

impl Agent for ValidatorAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Validator
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::ValidationRequest { solution } => {
                self.validate(solution).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl ValidatorAgent {
    async fn validate(&self, solution: Solution) -> AgentResponse {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // 1. Syntax validation
        if let Err(e) = self.validate_syntax(&solution.code) {
            errors.push(ValidationError::Syntax(e));
        }

        // 2. Semantic validation
        if let Err(e) = self.validate_semantics(&solution.code).await {
            errors.push(ValidationError::Semantic(e));
        }

        // 3. Example verification
        if let Err(e) = self.verify_examples(&solution).await {
            errors.push(ValidationError::VerificationFailed(e));
        }

        // 4. Edge case validation
        let edge_warnings = self.validate_edge_cases(&solution).await?;
        warnings.extend(edge_warnings);

        AgentResponse::SolutionValidated {
            validation: ValidationResult {
                is_valid: errors.is_empty(),
                errors,
                warnings,
                confidence: self.calculate_confidence(&errors, &warnings),
            }
        }
    }

    async fn verify_examples(&self, solution: &Solution) -> Result<()> {
        // Use sandbox tool for verification
        let tool_result = self.tools.execute(
            ToolId::Sandbox,
            ToolArgs::Sandbox(SandboxOperation::Verify {
                code: solution.code.clone(),
                examples: solution.examples.clone(),
            }),
        ).await?;

        if tool_result.success {
            Ok(())
        } else {
            Err(Error::VerificationFailed(tool_result.output))
        }
    }
}
```

### 7.5 Optimizer Agent

```rust
// src/agents/optimizer_agent.rs

pub struct OptimizerAgent {
    id: AgentId,
    tools: Arc<ToolRegistry>,
    memory: Arc<MemorySystem>,
}

impl Agent for OptimizerAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Optimizer
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::OptimizationRequest { solution } => {
                self.optimize(solution).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl OptimizerAgent {
    async fn optimize(&self, solution: Solution) -> AgentResponse {
        let mut optimized = solution.clone();

        // 1. Performance optimization
        optimized = self.optimize_performance(optimized).await?;

        // 2. Code size optimization
        optimized = self.optimize_size(optimized).await?;

        // 3. Readability optimization
        optimized = self.optimize_readability(optimized).await?;

        // 4. Verify optimized solution
        let validation = self.verify_optimized(&solution, &optimized).await?;

        if validation.is_valid {
            AgentResponse::SolutionOptimized { solution: optimized }
        } else {
            // Optimization broke something, return original
            AgentResponse::SolutionOptimized { solution }
        }
    }

    async fn optimize_performance(&self, solution: Solution) -> Result<Solution> {
        // Use patterns from memory for optimization
        let patterns = self.memory.pattern_memory.find_optimization_patterns(
            &solution.code,
        ).await?;

        let mut optimized_code = solution.code;

        for pattern in patterns {
            optimized_code = self.apply_optimization_pattern(optimized_code, pattern)?;
        }

        Ok(Solution { code: optimized_code, ..solution })
    }
}
```

### 7.6 Learning Agent

```rust
// src/agents/learning_agent.rs

pub struct LearningAgent {
    id: AgentId,
    memory: Arc<MemorySystem>,
    pattern_extractor: PatternExtractor,
}

impl Agent for LearningAgent {
    fn agent_id(&self) -> AgentId {
        AgentId::Learning
    }

    fn handle_message(&mut self, msg: AgentMessage) -> AgentResponse {
        match msg {
            AgentMessage::ExperienceRecord { experience } => {
                self.process_experience(experience).await
            }
            AgentMessage::MemoryQuery { query } => {
                self.query_memory(query).await
            }
            _ => AgentResponse::Error { error: AgentError::UnexpectedMessage },
        }
    }
}

impl LearningAgent {
    async fn process_experience(&self, experience: Experience) -> AgentResponse {
        // 1. Store experience
        self.memory.experience_db.store(experience.clone()).await?;

        // 2. Extract patterns
        let patterns = self.pattern_extractor.extract(&experience)?;

        // 3. Store patterns
        for pattern in patterns {
            self.memory.pattern_memory.store(pattern).await?;
        }

        // 4. Update effectiveness scores
        if let Some(method) = experience.solution.method {
            self.memory.update_effectiveness(
                experience.problem.hash(),
                method,
                experience.outcome == SolveOutcome::Success,
            ).await?;
        }

        AgentResponse::ExperienceProcessed {
            patterns_extracted: patterns.len(),
        }
    }
}
```

### 7.7 Communication System

```rust
// src/agents/communication.rs

pub struct CommunicationSystem {
    channels: HashMap<AgentId, mpsc::Sender<AgentMessage>>,
    response_channels: HashMap<TaskId, oneshot::Sender<AgentResponse>>,
}

impl CommunicationSystem {
    pub async fn send_to_agent(
        &self,
        agent: AgentId,
        message: AgentMessage,
    ) -> Result<AgentResponse> {
        let channel = self.channels.get(&agent)
            .ok_or(Error::AgentNotFound(agent))?;

        let (tx, rx) = oneshot::channel();

        // Store response channel
        if let AgentMessage::TaskAssignment { task, .. } = &message {
            self.response_channels.insert(*task, tx);
        }

        // Send message
        channel.send(message).await
            .map_err(|_| Error::AgentDisconnected(agent))?;

        // Wait for response
        rx.await
            .map_err(|_| Error::ResponseTimeout)?
    }

    pub async fn broadcast(&self, message: AgentMessage) -> Vec<Result<AgentResponse>> {
        let mut responses = Vec::new();

        for (&agent, channel) in &self.channels {
            let result = channel.send(message.clone()).await;
            responses.push(result);
        }

        responses
    }
}
```

---

## 8. Integration Points

### 8.1 Solver Pipeline Integration

```rust
// src/solver/integration.rs

pub struct AgenticSolver {
    base_solver: Arc<Solver>,
    planner: Arc<Planner>,
    coordinator: Arc<Coordinator>,
    memory: Arc<MemorySystem>,
}

impl AgenticSolver {
    pub async fn solve(&self, problem: &Problem) -> Result<SolveResult> {
        // 1. Check cache first (fast path)
        let cache_key = problem.hash();
        if let Some(cached) = self.memory.synthesis_cache.get(cache_key).await {
            return Ok(cached.result);
        }

        // 2. Create planning request
        let request = PlanningRequest::from_problem(problem);

        // 3. Generate plan
        let plan = self.planner.plan(&request).await?;

        // 4. Execute plan through coordinator
        let result = self.coordinator.execute_plan(plan).await?;

        // 5. Cache result
        self.memory.synthesis_cache.store(cache_key, result.clone()).await;

        Ok(result)
    }

    pub async fn solve_legacy(&self, problem: &Problem) -> Result<SolveResult> {
        // Fall back to base solver for non-agentic path
        Ok(self.base_solver.solve(problem)?)
    }
}
```

### 8.2 CLI Integration

```rust
// src/main.rs (additions)

pub struct Cli {
    #[arg(long, default_value_t = false)]
    agentic: bool,

    #[arg(long, default_value_t = false)]
    plan_only: bool,

    #[arg(long)]
    plan_file: Option<PathBuf>,
}

impl Cli {
    pub async fn run(self) -> Result<()> {
        if self.agentic {
            self.run_agentic().await
        } else {
            self.run_legacy().await
        }
    }

    async fn run_agentic(self) -> Result<()> {
        // Load problem
        let problem = self.load_problem()?;

        // Create agentic solver
        let solver = AgenticSolver::new(self.config).await?;

        // Solve
        let result = solver.solve(&problem).await?;

        // Output result
        self.print_result(&result);

        Ok(())
    }

    async fn run_legacy(self) -> Result<()> {
        // Use existing solver
        let result = solve_problem(&problem)?;

        self.print_result(&result);

        Ok(())
    }
}
```

### 8.3 Library API Integration

```rust
// src/lib.rs (additions)

pub use planning::{Planner, Plan, PlanStrategy};
pub use tools::{ToolRegistry, Tool, ToolId};
pub use memory::{MemorySystem, MemoryQuery, MemoryEntry};
pub use agents::{Agent, AgentId, AgentMessage, AgentResponse};
pub use coordination::{Coordinator, CoordinatorConfig};

/// Agentic solver interface
pub async fn solve_problem_agentic(
    problem: &Problem,
    config: AgenticConfig,
) -> Result<SolveResult> {
    let solver = AgenticSolver::new(config).await?;
    solver.solve(problem).await
}

/// Legacy solver interface (unchanged)
pub fn solve_problem(problem: &Problem) -> SolveResult {
    solver::solve_problem(problem)
}
```

---

## 9. Data Flow

### 9.1 End-to-End Flow

```
User Input (NL/Code)
        │
        ▼
┌──────────────────┐
│  Parser /        │
│  Analyzer        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Planner Agent   │◄────┐
│  - Analyze       │     │
│  - Decompose     │     │
│  - Generate Plan │     │
└────────┬─────────┘     │
         │               │
         ▼               │
┌──────────────────┐    │
│  Coordinator     │    │
│  - Schedule      │    │
│  - Delegate      │────┘
└────────┬─────────┘
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│  Synthesizer     │                  │  Validator      │
│  Agent           │                  │  Agent           │
│  - Use Tools     │─────────────────▶│  - Verify       │
│  - Generate Code │                  │  - Check        │
└────────┬─────────┘                  └────────┬─────────┘
         │                                      │
         │                                      │
         ▼                                      ▼
┌──────────────────┐                  ┌──────────────────┐
│  Optimizer       │                  │  Learning        │
│  Agent           │                  │  Agent           │
│  - Improve       │                  │  - Extract       │
│  - Refine        │                  │  - Store         │
└────────┬─────────┘                  └────────┬─────────┘
         │                                      │
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Memory System   │
              │  - Cache         │
              │  - Patterns      │
              │  - Experience    │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Final Result    │
              │  - Code           │
              │  - Metadata       │
              └──────────────────┘
```

### 9.2 Tool Use Flow

```
Agent Request
        │
        ▼
┌──────────────────┐
│  Safety Check    │
│  - Authorization │
│  - Validation    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tool Execution  │
│  - Setup         │
│  - Execute       │
│  - Cleanup       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Result Capture  │
│  - Output        │
│  - Duration      │
│  - Side Effects │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Logging         │
│  - Audit Trail   │
│  - Metrics       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Response        │
│  - Success/Error │
│  - Data          │
└──────────────────┘
```

### 9.3 Memory Access Flow

```
Memory Query
        │
        ▼
┌──────────────────┐
│  Query Routing   │
│  - Type Check    │
│  - Routing       │
└────────┬─────────┘
         │
         ├───────────┬─────────────┬──────────────┐
         ▼           ▼             ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Synthesis│ │ Pattern  │ │ Context  │ │Experience│
│ Cache    │ │ Memory   │ │ Memory   │ │ DB       │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │            │
     └────────┬──┴────────────┴────────────┘
              │
              ▼
┌──────────────────┐
│  Response Merge  │
│  - Combine       │
│  - Rank          │
│  - Format        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Response        │
│  - Results       │
│  - Metadata      │
└──────────────────┘
```

---

## 10. Communication Protocols

### 10.1 Agent Message Protocol

```rust
/// Agent message with guaranteed delivery confirmation
#[derive(Debug, Clone)]
pub struct GuaranteedMessage {
    pub id: MessageId,
    pub from: AgentId,
    pub to: AgentId,
    pub payload: AgentMessage,
    pub timestamp: SystemTime,
    pub requires_ack: bool,
}

impl GuaranteedMessage {
    pub async fn send(&self, channel: &mpsc::Sender<Self>) -> Result<Acknowledgment> {
        channel.send(self.clone()).await?;

        if self.requires_ack {
            // Wait for acknowledgment
            timeout(Duration::from_secs(5), self.wait_for_ack()).await?
        }

        Ok(Acknowledgment { message_id: self.id })
    }
}
```

### 10.2 Task Assignment Protocol

```rust
/// Task assignment with handshake
pub struct TaskAssignment {
    pub task_id: TaskId,
    pub assigned_to: AgentId,
    pub assigned_by: AgentId,
    pub priority: Priority,
    pub deadline: Option<SystemTime>,
    pub resources: Resources,
    pub context: TaskContext,
}

impl TaskAssignment {
    pub async fn assign(&self) -> Result<AssignmentConfirmation> {
        // 1. Check agent availability
        // 2. Check resource availability
        // 3. Assign task
        // 4. Wait for acceptance
        // 5. Return confirmation

        todo!()
    }
}
```

### 10.3 Coordination Protocol

```rust
/// Coordination message for multi-agent collaboration
#[derive(Debug, Clone)]
pub enum CoordinationMessage {
    /// Request to join a collaboration
    JoinRequest { task_id: TaskId, agent: AgentId },

    /// Acceptance into collaboration
    JoinAccepted { task_id: TaskId, role: Role },

    /// Status update within collaboration
    StatusUpdate { task_id: TaskId, agent: AgentId, status: AgentStatus },

    /// Proposal for collaboration decision
    Proposal { task_id: TaskId, proposal: Proposal },

    /// Vote on proposal
    Vote { task_id: TaskId, agent: AgentId, vote: Vote },

    /// Consensus reached
    Consensus { task_id: TaskId, decision: Decision },

    /// Collaboration complete
    Complete { task_id: TaskId, result: CollaborationResult },
}
```

---

## 11. Error Handling Strategy

### 11.1 Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum AgenticError {
    #[error("Agent not found: {0}")]
    AgentNotFound(AgentId),

    #[error("Agent disconnected: {0}")]
    AgentDisconnected(AgentId),

    #[error("Tool not found: {0}")]
    ToolNotFound(ToolId),

    #[error("Tool execution failed: {0}")]
    ToolExecutionFailed(String),

    #[error("Memory access failed: {0}")]
    MemoryAccessFailed(String),

    #[error("Planning failed: {0}")]
    PlanningFailed(String),

    #[error("Synthesis failed: {0}")]
    SynthesisFailed(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Timeout exceeded: {0}s")]
    Timeout(u64),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
}
```

### 11.2 Error Recovery Strategies

```rust
pub struct ErrorRecovery {
    max_retries: u32,
    backoff_strategy: BackoffStrategy,
    fallback_strategies: HashMap<ErrorType, FallbackStrategy>,
}

impl ErrorRecovery {
    pub async fn recover(&self, error: AgenticError) -> Result<()> {
        match error {
            AgenticError::Timeout(_) => {
                self.handle_timeout().await
            }
            AgenticError::AgentDisconnected(agent) => {
                self.handle_disconnection(agent).await
            }
            AgenticError::ToolExecutionFailed(_) => {
                self.handle_tool_failure().await
            }
            _ => Err(error),
        }
    }

    async fn handle_timeout(&self) -> Result<()> {
        // Retry with backoff
        for attempt in 0..self.max_retries {
            tokio::time::sleep(self.backoff_strategy.delay(attempt)).await;

            // Retry operation
            if self.retry_operation().await.is_ok() {
                return Ok(());
            }
        }

        // Fall back to alternative strategy
        self.fallback_strategy(ErrorType::Timeout)
    }

    async fn handle_disconnection(&self, agent: AgentId) -> Result<()> {
        // Attempt to reconnect
        if self.reconnect_agent(agent).await? {
            return Ok(());
        }

        // Spawn replacement agent
        self.spawn_replacement_agent(agent).await
    }
}
```

### 11.3 Error Propagation

```rust
/// Error context for tracking error chain
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub error: AgenticError,
    pub source: ErrorSource,
    pub chain: Vec<AgenticError>,
    pub timestamp: SystemTime,
    pub correlation_id: Uuid,
}

impl ErrorContext {
    pub fn propagate(&self) -> AgenticError {
        // Add context and propagate up
        self.error.clone()
    }

    pub async fn log(&self) {
        // Log with full context
        tracing::error!(
            error = ?self.error,
            source = ?self.source,
            chain = ?self.chain,
            correlation_id = %self.correlation_id,
            "Agentic error occurred"
        );
    }
}
```

---

## 12. Performance Considerations

### 12.1 Caching Strategy

```rust
pub struct CacheManager {
    synthesis_cache: Arc<SynthesisCache>,
    pattern_cache: Arc<PatternCache>,
    lru_policy: LruPolicy,
    cache_size: usize,
}

impl CacheManager {
    pub fn configure(&mut self, config: CacheConfig) {
        self.cache_size = config.max_size;
        self.lru_policy = config.eviction_policy;
    }

    pub async fn get_or_compute<T>(
        &self,
        key: CacheKey,
        compute: impl Future<Output = Result<T>>,
    ) -> Result<T> {
        // Try cache first
        if let Some(cached) = self.get(&key).await {
            return Ok(cached);
        }

        // Compute result
        let result = compute.await?;

        // Cache result
        self.store(key, result.clone()).await;

        Ok(result)
    }
}
```

### 12.2 Parallel Execution

```rust
pub struct ParallelExecutor {
    max_parallel: usize,
    semaphore: Arc<Semaphore>,
}

impl ParallelExecutor {
    pub async fn execute_parallel<T, F>(
        &self,
        tasks: Vec<F>,
    ) -> Vec<Result<T>>
    where
        F: Future<Output = Result<T>>,
    {
        let mut results = Vec::new();
        let mut tasks = tokio::task::JoinSet::new();

        for task in tasks {
            let permit = self.semaphore.clone().acquire_owned().await?;

            tasks.spawn(async move {
                let _permit = permit;
                task.await
            });
        }

        while let Some(result) = tasks.join_next().await {
            results.push(result.unwrap());
        }

        results
    }
}
```

### 12.3 Resource Management

```rust
pub struct ResourceManager {
    cpu_pool: ResourcePool<Cpu>,
    memory_pool: ResourcePool<Memory>,
    gpu_pool: ResourcePool<Gpu>,
}

impl ResourceManager {
    pub async fn allocate(&self, request: ResourceRequest) -> Result<Allocation> {
        match request.resource_type {
            ResourceType::Cpu => {
                self.cpu_pool.allocate(request.amount).await
            }
            ResourceType::Memory => {
                self.memory_pool.allocate(request.amount).await
            }
            ResourceType::Gpu => {
                self.gpu_pool.allocate(request.amount).await
            }
        }
    }

    pub async fn release(&self, allocation: Allocation) {
        match allocation.resource_type {
            ResourceType::Cpu => {
                self.cpu_pool.release(allocation).await
            }
            ResourceType::Memory => {
                self.memory_pool.release(allocation).await
            }
            ResourceType::Gpu => {
                self.gpu_pool.release(allocation).await
            }
        }
    }
}
```

---

## Implementation Phases

### Phase 2.1: Core Infrastructure (Weeks 1-2)
- Task hierarchy types and planning basics
- Tool registry and core tools (Git, File, Shell)
- Memory system foundation (cache, patterns)
- Basic agent traits and messaging

### Phase 2.2: Planning System (Weeks 3-4)
- Planner agent implementation
- Problem decomposition
- Strategy selection
- Task scheduling

### Phase 2.3: Multi-Agent System (Weeks 5-6)
- Specialized agents (Synthesizer, Validator, Optimizer)
- Coordinator implementation
- Communication protocols
- Error handling and recovery

### Phase 2.4: Integration (Weeks 7-8)
- Solver pipeline integration
- CLI integration
- Library API
- Testing and validation

### Phase 2.5: Optimization (Weeks 9-10)
- Performance optimization
- Memory consolidation
- Advanced tools (HTTP, Database)
- Documentation

---

## Testing Strategy

### Unit Tests
- Each agent in isolation
- Tool execution
- Memory operations
- Planning logic

### Integration Tests
- Agent coordination
- End-to-end synthesis
- Tool chain execution
- Memory persistence

### Performance Tests
- Cache hit rates
- Parallel execution efficiency
- Resource utilization
- Scalability

### Validation Tests
- Solution correctness
- Tool safety
- Memory integrity
- Error recovery

---

## Conclusion

This architecture provides a comprehensive foundation for agentic enhancement of the nCPU synthesis system. The hierarchical planning, tool use, memory, and multi-agent coordination layers work together to enable synthesis of increasingly complex programs through intelligent decomposition, strategic composition, and learned optimization.

The non-breaking integration ensures existing functionality continues to work while new agentic capabilities are progressively added. The modular design allows each component to be developed, tested, and deployed independently while contributing to the overall system intelligence.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-18
**Status**: Design Complete - Ready for Implementation
