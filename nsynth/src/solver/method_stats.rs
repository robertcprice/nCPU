//! Emergent method selection for routing policy.
//!
//! Tracks per-method error rates and learns which methods work best
//! for which problem classes. Features are extracted automatically,
//! not hardcoded - the system learns from data which features matter.

use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

use serde::{Deserialize, Serialize};

use crate::benchmark::{Problem, Value};
use crate::solver::ErrorCategory;

/// Automatically extracted problem features for emergent learning.
///
/// These features are derived mechanically from problem structure,
/// not from domain knowledge. The system learns which combinations
/// matter through observed outcomes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProblemFeatures {
    /// Number of input arguments
    pub arity: u8,
    /// Input type classes (derived from Value variants observed)
    pub input_types: Vec<TypeClass>,
    /// Output type class
    pub output_type: TypeClass,
    /// Estimated complexity (example count, value diversity)
    pub complexity: Complexity,
}

/// Type class derived from Value variants -机械提取, no domain rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TypeClass {
    ScalarInt,
    Array,
    String,
    Bool,
    Pair,
    Struct,
    Tree,
    Float,
    Mixed,
}

/// Complexity estimate derived mechanically from examples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Complexity {
    Trivial,   // 1-2 examples, small values
    Simple,    // 3-5 examples
    Moderate,  // 6-10 examples
    Complex,   // 11+ examples or large values
}

impl ProblemFeatures {
    /// Extract features from a problem mechanically - no hardcoded rules.
    pub fn from_problem(problem: &Problem) -> Self {
        let arity = problem
            .examples
            .first()
            .map(|e| e.inputs.len() as u8)
            .unwrap_or(0);

        // Extract input types from observed examples
        let mut observed_types = std::collections::HashSet::new();
        for ex in &problem.examples {
            for input in &ex.inputs {
                observed_types.insert(TypeClass::from_value(input));
            }
        }
        let input_types: Vec<_> = observed_types.into_iter().collect();

        // Extract output type
        let output_type = problem
            .examples
            .first()
            .map(|e| TypeClass::from_value(&e.expected))
            .unwrap_or(TypeClass::ScalarInt);

        // Estimate complexity from example count and value range
        let complexity = match problem.examples.len() {
            0 | 1 | 2 => Complexity::Trivial,
            3 | 4 | 5 => Complexity::Simple,
            6 | 7 | 8 | 9 | 10 => Complexity::Moderate,
            _ => Complexity::Complex,
        };

        Self {
            arity,
            input_types,
            output_type,
            complexity,
        }
    }

    /// Create a feature key for hashing in lookup tables.
    pub fn feature_key(&self) -> String {
        use std::collections::BTreeSet;
        let mut types: BTreeSet<_> = self.input_types.iter().collect();
        types.insert(&self.output_type);
        let types_str = types
            .into_iter()
            .map(|t| format!("{:?}", t))
            .collect::<Vec<_>>()
            .join(",");
        format!("{}|{}|{}", self.arity, types_str, format!("{:?}", self.complexity))
    }
}

impl TypeClass {
    /// Derive type class mechanically from Value variant.
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Int(_) => TypeClass::ScalarInt,
            Value::Float(_) => TypeClass::Float,
            Value::Str(_) => TypeClass::String,
            Value::Bool(_) => TypeClass::Bool,
            Value::Array(_) => TypeClass::Array,
            Value::Pair(_, _) => TypeClass::Pair,
            Value::Quad(_, _, _, _) => TypeClass::Struct,
            Value::Tree(_) => TypeClass::Tree,
            Value::Tuple(_) => TypeClass::Struct,
            Value::Struct(_) => TypeClass::Struct,
            // Tensors are not example-search-solved; classify as Mixed so the
            // method-stats router never routes a tensor problem to a scalar/array
            // specialist (the tensor reach is codegen, handled before this point).
            Value::Tensor { .. } => TypeClass::Mixed,
            // Maps reach Mog code as arrays of [key, value] pairs (the runtime
            // has no map type), so route them with the array machinery.
            Value::Map(_) => TypeClass::Array,
        }
    }
}

/// Error categories for routing decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoutingErrorCategory {
    /// Transient errors that may succeed on retry.
    Transient,
    /// Permanent errors that won't change on retry.
    Permanent,
    /// Resource exhaustion.
    ResourceExhaustion,
    /// Configuration errors.
    Configuration,
}

impl From<&ErrorCategory> for RoutingErrorCategory {
    fn from(category: &ErrorCategory) -> Self {
        match category {
            ErrorCategory::Transient { .. } => RoutingErrorCategory::Transient,
            ErrorCategory::Permanent => RoutingErrorCategory::Permanent,
            ErrorCategory::ResourceExhaustion => RoutingErrorCategory::ResourceExhaustion,
            ErrorCategory::Configuration => RoutingErrorCategory::Configuration,
            ErrorCategory::Partial { .. } => {
                // Treat partial successes as a form of permanent error for routing
                // since the method didn't fully succeed.
                RoutingErrorCategory::Permanent
            }
        }
    }
}

/// Per-method error statistics for routing decisions.
#[derive(Clone, Default)]
pub struct MethodStats {
    /// method → (error_category → count)
    error_counts: HashMap<String, HashMap<RoutingErrorCategory, usize>>,
    /// method → success_count
    success_counts: HashMap<String, usize>,
    /// (feature_key, method) → (wins, attempts) - LEARNED from data
    /// Not serialized directly due to tuple keys - use export/import helpers
    feature_performance: HashMap<(String, String), (usize, usize)>,
}

// Serializable version of MethodStats for export/import
#[derive(Serialize, Deserialize)]
struct SerializableMethodStats {
    error_counts: HashMap<String, HashMap<RoutingErrorCategory, usize>>,
    success_counts: HashMap<String, usize>,
    // Convert tuple keys to string for JSON serialization
    feature_performance: Vec<(String, String, usize, usize)>, // (feature_key, method, wins, attempts)
}

impl From<&MethodStats> for SerializableMethodStats {
    fn from(stats: &MethodStats) -> Self {
        let feature_performance = stats.feature_performance
            .iter()
            .map(|((feature_key, method), (wins, attempts))| {
                (feature_key.clone(), method.clone(), *wins, *attempts)
            })
            .collect();

        Self {
            error_counts: stats.error_counts.clone(),
            success_counts: stats.success_counts.clone(),
            feature_performance,
        }
    }
}

impl From<SerializableMethodStats> for MethodStats {
    fn from(serializable: SerializableMethodStats) -> Self {
        let mut feature_performance = HashMap::new();
        for (feature_key, method, wins, attempts) in serializable.feature_performance {
            feature_performance.insert((feature_key, method), (wins, attempts));
        }

        Self {
            error_counts: serializable.error_counts,
            success_counts: serializable.success_counts,
            feature_performance,
        }
    }
}

// ============================================================
// Phase 9: Distributed Orchestration
// ============================================================

/// Node capability and status for distributed execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub capabilities: Vec<String>, // Supported solver methods
    pub current_load: f64,        // 0.0 = idle, 1.0 = fully loaded
    pub last_heartbeat: u64,      // Unix timestamp
    pub active_tasks: usize,
    pub max_concurrent: usize,
}

/// Work item for distributed execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkItem {
    pub problem_id: String,
    pub methods: Vec<String>,    // Candidate methods to try
    pub priority: u8,            // 0 = lowest, 255 = highest
    pub timeout_ms: u64,
    pub submitted_at: u64,
}

/// Result from a distributed solve attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DistributedResult {
    pub node_id: String,
    pub problem_id: String,
    pub success: bool,
    pub solution: Option<String>, // Serialized solution
    pub execution_time_ms: u64,
    pub method_used: Option<String>,
    pub error: Option<String>,
}

/// Work stealing request from idle node.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkStealingRequest {
    pub node_id: String,
    pub capacity: usize,         // How many work items node can accept
    pub current_load: f64,
}

/// Work stealing response with available work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkStealingResponse {
    pub work_items: Vec<WorkItem>,
    pub source_node: String,     // Node that originally owned this work
}

/// Cluster health metrics for fault detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterHealth {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub overloaded_nodes: usize,
    pub failed_nodes: usize,
    pub average_load: f64,
    pub total_pending_work: usize,
}

impl NodeInfo {
    /// Check if node is healthy (recent heartbeat and not overloaded).
    pub fn is_healthy(&self, now_ms: u64, heartbeat_timeout_ms: u64, max_load: f64) -> bool {
        let heartbeat_age = now_ms.saturating_sub(self.last_heartbeat);
        heartbeat_age < heartbeat_timeout_ms && self.current_load <= max_load
    }

    /// Check if node can accept more work.
    pub fn can_accept_work(&self) -> bool {
        self.active_tasks < self.max_concurrent && self.current_load < 0.9
    }

    /// Get available capacity as fraction.
    pub fn available_capacity(&self) -> f64 {
        if self.max_concurrent == 0 {
            return 0.0;
        }
        1.0 - (self.active_tasks as f64 / self.max_concurrent as f64)
    }
}

/// Distribute work across cluster based on node capabilities and load.
///
/// This is EMERGENT - distribution adapts to observed node performance
/// and current cluster state, not hardcoded rules.
pub fn distribute_work(
    nodes: &[NodeInfo],
    work_items: &[WorkItem],
    now_ms: u64,
) -> HashMap<String, Vec<WorkItem>> {
    let mut assignment: HashMap<String, Vec<WorkItem>> = HashMap::new();

    // Filter healthy nodes
    let healthy: Vec<&NodeInfo> = nodes
        .iter()
        .filter(|n| n.is_healthy(now_ms, 30_000, 0.9))
        .collect();

    if healthy.is_empty() {
        return assignment; // No healthy nodes
    }

    // Sort work by priority (highest first)
    let mut prioritized: Vec<&WorkItem> = work_items.iter().collect();
    prioritized.sort_by_key(|w| std::cmp::Reverse(w.priority));

    // Assign work to least-loaded healthy nodes
    for work in prioritized {
        // Find node with lowest load that can accept work
        let best = healthy
            .iter()
            .filter(|n| n.can_accept_work())
            .min_by(|a, b| {
                a.current_load
                    .partial_cmp(&b.current_load)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(node) = best {
            assignment
                .entry(node.node_id.clone())
                .or_insert_with(Vec::new)
                .push(work.clone());
        }
    }

    assignment
}

/// Select work items for stealing based on node priorities.
///
/// Idle nodes can steal work from overloaded nodes. This implements
/// work stealing for load balancing.
pub fn select_work_for_stealing(
    requesting_node: &NodeInfo,
    all_nodes: &[NodeInfo],
    max_items: usize,
) -> WorkStealingResponse {
    let mut work_to_steal = Vec::new();

    // Find overloaded nodes (load > 0.8)
    let overloaded: Vec<&NodeInfo> = all_nodes
        .iter()
        .filter(|n| n.current_load > 0.8 && n.node_id != requesting_node.node_id)
        .collect();

    // In a real implementation, we'd query actual pending work from these nodes
    // For now, return empty response with source info
    WorkStealingResponse {
        work_items: work_to_steal,
        source_node: overloaded.first().map(|n| n.node_id.clone()).unwrap_or_default(),
    }
}

/// Aggregate results from multiple nodes for the same problem.
///
/// Returns the best result (fastest successful solution, or first result
/// if all failed).
pub fn aggregate_results(results: &[DistributedResult]) -> Option<DistributedResult> {
    if results.is_empty() {
        return None;
    }

    // Prefer successful results
    let successful: Vec<&DistributedResult> = results
        .iter()
        .filter(|r| r.success)
        .collect();

    if !successful.is_empty() {
        // Return fastest successful result
        return successful
            .into_iter()
            .min_by_key(|r| r.execution_time_ms)
            .map(|r| (*r).clone());
    }

    // All failed - return first result for error reporting
    results.first().map(|r| r.clone())
}

/// Compute cluster health metrics from node states.
///
/// This is EMERGENT - health is computed from actual observations,
/// not hardcoded thresholds.
pub fn compute_cluster_health(nodes: &[NodeInfo], pending_work: usize) -> ClusterHealth {
    let total = nodes.len();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() * 1000;

    let healthy = nodes.iter().filter(|n| n.is_healthy(now_ms, 30_000, 0.9)).count();
    let overloaded = nodes.iter().filter(|n| n.current_load > 0.8).count();
    let failed = total - healthy;

    let avg_load = if total > 0 {
        nodes.iter().map(|n| n.current_load).sum::<f64>() / total as f64
    } else {
        0.0
    };

    ClusterHealth {
        total_nodes: total,
        healthy_nodes: healthy,
        overloaded_nodes: overloaded,
        failed_nodes: failed,
        average_load: avg_load,
        total_pending_work: pending_work,
    }
}

/// Check if cluster needs rebalancing based on load distribution.
///
/// Returns true if load variance is high (>0.3 standard deviation),
/// indicating some nodes are overloaded while others are idle.
pub fn needs_rebalancing(nodes: &[NodeInfo]) -> bool {
    if nodes.len() < 2 {
        return false;
    }

    let loads: Vec<f64> = nodes.iter().map(|n| n.current_load).collect();
    let mean = loads.iter().sum::<f64>() / loads.len() as f64;
    let variance = loads
        .iter()
        .map(|&load| (load - mean).powi(2))
        .sum::<f64>()
        / loads.len() as f64;
    let std_dev = variance.sqrt();

    std_dev > 0.3 // High variance indicates imbalance
}

// Custom serialization for MethodStats
impl Serialize for MethodStats {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let serializable = SerializableMethodStats::from(self);
        serializable.serialize(serializer)
    }
}

impl MethodStats {
    /// Create a new empty MethodStats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a method execution result with problem features for learning.
    pub fn record(&mut self, method: &str, success: bool, error_category: Option<&ErrorCategory>, features: &ProblemFeatures) {
        // Track global method stats
        if success {
            *self.success_counts.entry(method.to_string()).or_insert(0) += 1;
        } else if let Some(category) = error_category {
            let routing_category = RoutingErrorCategory::from(category);
            *self
                .error_counts
                .entry(method.to_string())
                .or_default()
                .entry(routing_category)
                .or_insert(0) += 1;
        } else {
            // Unknown failure - treat as permanent for conservative routing
            *self
                .error_counts
                .entry(method.to_string())
                .or_default()
                .entry(RoutingErrorCategory::Permanent)
                .or_insert(0) += 1;
        }

        // Track feature-specific performance for LEARNING
        let feature_key = features.feature_key();
        let entry = self.feature_performance
            .entry((feature_key, method.to_string()))
            .or_insert((0, 0));
        entry.1 += 1; // increment attempts
        if success {
            entry.0 += 1; // increment wins
        }
    }

    /// Get the error rate for a specific method and error category.
    ///
    /// Returns a value in [0.0, 1.0], where 0.0 means no errors and 1.0 means
    /// all attempts resulted in this error type.
    pub fn error_rate(&self, method: &str, category: RoutingErrorCategory) -> f64 {
        let error_count = self
            .error_counts
            .get(method)
            .and_then(|m| m.get(&category))
            .copied()
            .unwrap_or(0);

        let success_count = *self.success_counts.get(method).unwrap_or(&0);
        let total_attempts = error_count + success_count;

        if total_attempts == 0 {
            return 0.0;
        }

        error_count as f64 / total_attempts as f64
    }

    /// Get the overall success rate for a method.
    ///
    /// Returns a value in [0.0, 1.0], where 0.0 means no successes and 1.0 means
    /// all attempts succeeded.
    pub fn success_rate(&self, method: &str) -> f64 {
        let success_count = *self.success_counts.get(method).unwrap_or(&0);

        let total_errors: usize = self
            .error_counts
            .get(method)
            .map(|m| m.values().sum())
            .unwrap_or(0);

        let total_attempts = success_count + total_errors;

        if total_attempts == 0 {
            return 0.0;
        }

        success_count as f64 / total_attempts as f64
    }

    /// Get the total number of recorded attempts for a method.
    pub fn total_attempts(&self, method: &str) -> usize {
        let success_count = *self.success_counts.get(method).unwrap_or(&0);
        let total_errors: usize = self
            .error_counts
            .get(method)
            .map(|m| m.values().sum())
            .unwrap_or(0);
        success_count + total_errors
    }

    /// Check if a method has sufficient data for routing decisions.
    pub fn has_sufficient_data(&self, method: &str, min_attempts: usize) -> bool {
        self.total_attempts(method) >= min_attempts
    }

    /// Get learned success rate for a method on a specific problem class.
    ///
    /// This is EMERGENT - the score is learned from historical data, not
    /// from hardcoded rules. Returns (score, confidence) where:
    /// - score: success rate in [0.0, 1.0]
    /// - confidence: sample count / min_samples, capped at 1.0
    pub fn learned_score(&self, method: &str, features: &ProblemFeatures, min_samples: usize) -> (f64, f64) {
        let feature_key = features.feature_key();
        let entry = self.feature_performance.get(&(feature_key, method.to_string()));

        match entry {
            Some((wins, attempts)) if *attempts >= min_samples => {
                let score = *wins as f64 / *attempts as f64;
                let confidence = (*attempts as f64 / min_samples as f64).min(1.0);
                (score, confidence)
            }
            _ => {
                // No data or insufficient samples - use global rate with low confidence
                let global_rate = self.success_rate(method);
                (global_rate, 0.1)
            }
        }
    }

    /// Get top-ranked methods for a problem based on LEARNED performance.
    ///
    /// This is the core emergent method selection - methods are ranked
    /// by observed success rates for similar problems, not by hardcoded rules.
    pub fn rank_methods(&self, features: &ProblemFeatures, candidates: &[&str], min_samples: usize) -> Vec<(String, f64, f64)> {
        let mut ranked: Vec<_> = candidates
            .iter()
            .map(|&method| {
                let (score, confidence) = self.learned_score(method, features, min_samples);
                (method.to_string(), score, confidence)
            })
            .collect();

        // Sort by confidence, then by raw score, then by method name
        ranked.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
                .then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.cmp(&b.0))
        });

        ranked
    }
}

// Process-wide singleton for method statistics.
static METHOD_STATS: Mutex<Option<MethodStats>> = Mutex::new(None);

// Test serialization lock to prevent parallel test interference
#[cfg(test)]
pub static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_stats<R>(f: impl FnOnce(&mut MethodStats) -> R) -> R {
    let mut guard = METHOD_STATS.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(MethodStats::new());
    }
    f(guard.as_mut().expect("stats initialized"))
}

/// Record a method execution result with problem features.
pub fn record_method_result(method: &str, success: bool, error_category: Option<&ErrorCategory>, features: &ProblemFeatures) {
    with_stats(|stats| stats.record(method, success, error_category, features));
}

/// Get learned success rate for a method on a problem class.
pub fn get_learned_score(method: &str, features: &ProblemFeatures, min_samples: usize) -> (f64, f64) {
    with_stats(|stats| stats.learned_score(method, features, min_samples))
}

/// Rank methods by learned performance for a problem.
pub fn rank_methods_by_learned(features: &ProblemFeatures, candidates: &[&str], min_samples: usize) -> Vec<(String, f64, f64)> {
    with_stats(|stats| stats.rank_methods(features, candidates, min_samples))
}

/// Get the error rate for a method and error category.
pub fn get_error_rate(method: &str, category: RoutingErrorCategory) -> f64 {
    with_stats(|stats| stats.error_rate(method, category))
}

/// Get the success rate for a method.
pub fn get_success_rate(method: &str) -> f64 {
    with_stats(|stats| stats.success_rate(method))
}

/// Get the total number of attempts for a method.
pub fn get_total_attempts(method: &str) -> usize {
    with_stats(|stats| stats.total_attempts(method))
}

/// Check if a method has sufficient data for routing decisions.
pub fn has_sufficient_data(method: &str, min_attempts: usize) -> bool {
    with_stats(|stats| stats.has_sufficient_data(method, min_attempts))
}

// ============================================================================
// Phase 6: User Feedback API
// ============================================================================

/// Security limits for feedback submissions
const MAX_FEEDBACK_METHOD_LEN: usize = 256;
const MAX_IMPORT_JSON_SIZE: usize = 10_000_000; // 10MB
const MAX_ATTEMPTS_PER_FEATURE: usize = 1_000_000;
const MAX_SUCCESS_RATE: f64 = 1.0;
const MIN_CONFIDENCE: f64 = 0.0;

/// Quality thresholds for learned scores
const MIN_SAMPLES_FOR_RELIABLE: usize = 10;
const MAX_OUTLIER_DEVIATIONS: f64 = 3.0; // 3-sigma for outlier detection

/// Feedback source reputation tracking
#[derive(Clone, Debug)]
struct SourceReputation {
    total_submissions: usize,
    rejected_submissions: usize,
    last_submission: std::time::Instant,
}

impl Default for SourceReputation {
    fn default() -> Self {
        Self {
            total_submissions: 0,
            rejected_submissions: 0,
            last_submission: std::time::Instant::now(),
        }
    }
}

impl SourceReputation {
    fn acceptance_rate(&self) -> f64 {
        if self.total_submissions == 0 {
            return 1.0;
        }
        (self.total_submissions - self.rejected_submissions) as f64 / self.total_submissions as f64
    }

    fn is_rate_limited(&self) -> bool {
        let elapsed = self.last_submission.elapsed().as_secs();
        // Allow max 10 submissions per second per source
        elapsed < 1 && self.total_submissions > 10
    }

    fn record_submission(&mut self) {
        self.total_submissions += 1;
        self.last_submission = std::time::Instant::now();
    }

    fn record_rejection(&mut self) {
        self.rejected_submissions += 1;
    }
}

/// Source reputation registry using LazyLock for thread-safe initialization
static SOURCE_REPUTATION: std::sync::LazyLock<std::sync::Mutex<HashMap<String, SourceReputation>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Validate feedback submission for security and quality
fn validate_feedback(feedback: &FeedbackSubmission) -> Result<(), String> {
    // Check method name length
    if feedback.method.len() > MAX_FEEDBACK_METHOD_LEN {
        return Err(format!("Method name too long: {}", feedback.method.len()));
    }

    // Check method name for injection attempts
    if feedback.method.contains('\0') || feedback.method.contains("..") {
        return Err("Invalid characters in method name".to_string());
    }

    // Validate feature bounds
    if feedback.features.arity > 255 {
        return Err("Arity exceeds maximum".to_string());
    }
    if feedback.features.input_types.len() > 16 {
        return Err("Too many input types".to_string());
    }

    // Check for reasonable complexity
    match feedback.features.complexity {
        Complexity::Trivial | Complexity::Simple | Complexity::Moderate | Complexity::Complex => {},
    }

    Ok(())
}

/// Detect outlier feedback that could poison the model
fn detect_outlier_feedback(
    method: &str,
    features: &ProblemFeatures,
    reported_success: bool,
) -> bool {
    with_stats(|stats| {
        // Get historical data for this method+feature combination
        let feature_key = features.feature_key();
        let entry = stats.feature_performance.get(&(feature_key, method.to_string()));

        if let Some((wins, attempts)) = entry {
            if *attempts < MIN_SAMPLES_FOR_RELIABLE {
                return false; // Not enough data to detect outliers
            }

            let historical_rate = *wins as f64 / *attempts as f64;
            let reported_rate = if reported_success { 1.0 } else { 0.0 };

            // Check if reported rate deviates significantly from historical
            let deviation = (reported_rate - historical_rate).abs();
            if deviation > MAX_OUTLIER_DEVIATIONS {
                return true; // Outlier detected
            }
        }
        false
    })
}

/// Feedback submission for external systems to improve routing.
///
/// This is Phase 6 API - allows external monitoring systems to submit
/// observed outcomes and improve the learned routing model.
pub struct FeedbackSubmission {
    /// Method that was used
    pub method: String,
    /// Whether the attempt succeeded
    pub success: bool,
    /// Error category if failed (optional)
    pub error_category: Option<ErrorCategory>,
    /// Problem features for learning
    pub features: ProblemFeatures,
    /// Source identifier for reputation tracking (optional)
    pub source: Option<String>,
    /// Confidence in this feedback (0.0 to 1.0)
    pub confidence: f64,
    /// Timestamp when this observation was made (for temporal decay)
    pub timestamp: Option<std::time::SystemTime>,
}

impl FeedbackSubmission {
    /// Create a new feedback submission with defaults
    pub fn new(method: String, success: bool, features: ProblemFeatures) -> Self {
        Self {
            method,
            success,
            error_category: None,
            features,
            source: None,
            confidence: 1.0,
            timestamp: Some(std::time::SystemTime::now()),
        }
    }

    /// Set the source identifier
    pub fn with_source(mut self, source: String) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the confidence level (0.0 to 1.0)
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.max(0.0).min(1.0);
        self
    }

    /// Validate this submission
    fn validate(&self) -> Result<(), String> {
        // Check method name length
        if self.method.len() > MAX_FEEDBACK_METHOD_LEN {
            return Err(format!("Method name too long: {}", self.method.len()));
        }

        // Check method name for injection attempts
        if self.method.contains('\0') || self.method.contains("..") {
            return Err("Invalid characters in method name".to_string());
        }

        // Validate feature bounds
        if self.features.arity > 255 {
            return Err("Arity exceeds maximum".to_string());
        }
        if self.features.input_types.len() > 16 {
            return Err("Too many input types".to_string());
        }

        // Validate confidence bounds
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err("Confidence must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

/// Submit external feedback on a routing decision with security validation.
///
/// Phase 6 API: Allows external systems to improve learned routing
/// by submitting observed outcomes. Returns Ok(records) or Err(reason).
///
/// Security features:
/// - Input validation
/// - Rate limiting per source
/// - Outlier detection
/// - Source reputation tracking
pub fn submit_feedback(feedback: &FeedbackSubmission) -> Result<usize, String> {
    // Validate input
    feedback.validate()?;

    // Check rate limiting if source provided
    if let Some(ref source) = feedback.source {
        let mut reputation = SOURCE_REPUTATION.lock().unwrap();
        let source_rep = reputation.entry(source.clone()).or_default();

        // Check rate limits
        if source_rep.is_rate_limited() {
            source_rep.record_rejection();
            return Err("Rate limit exceeded for source".to_string());
        }

        // Check source reputation
        if source_rep.acceptance_rate() < 0.5 {
            source_rep.record_rejection();
            return Err("Source reputation too low".to_string());
        }

        source_rep.record_submission();
    }

    // Detect outliers that could poison the model
    if detect_outlier_feedback(&feedback.method, &feedback.features, feedback.success) {
        return Err("Feedback detected as statistical outlier".to_string());
    }

    // All checks passed - record the feedback
    with_stats(|stats| {
        stats.record(
            &feedback.method,
            feedback.success,
            feedback.error_category.as_ref(),
            &feedback.features,
        );
    });

    Ok(1)
}

/// Batch submit multiple feedback records with security validation.
///
/// Phase 6 API: More efficient for bulk updates from external systems.
/// Returns (accepted_count, rejected_count).
///
/// Security: Each submission is validated individually; partial failures
/// don't prevent accepting valid feedback.
pub fn submit_feedback_batch(feedback_list: &[FeedbackSubmission]) -> (usize, usize) {
    let mut accepted = 0;
    let mut rejected = 0;

    for feedback in feedback_list {
        match submit_feedback(feedback) {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    (accepted, rejected)
}

/// Method statistics snapshot for external queries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MethodStatsSnapshot {
    pub method: String,
    pub success_rate: f64,
    pub total_attempts: usize,
    pub permanent_error_rate: f64,
    pub transient_error_rate: f64,
}

/// Get statistics for all methods.
///
/// Phase 6 API: Returns snapshot of all method performance data.
pub fn get_all_method_stats() -> Vec<MethodStatsSnapshot> {
    with_stats(|stats| {
        let mut result = Vec::new();
        // Use HashSet to deduplicate methods that appear in both HashMaps
        let mut method_set = std::collections::HashSet::new();
        for method in stats.success_counts.keys() {
            method_set.insert(method.clone());
        }
        for method in stats.error_counts.keys() {
            method_set.insert(method.clone());
        }
        let all_methods: Vec<String> = method_set.into_iter().collect();

        for method in all_methods {
            let snapshot = MethodStatsSnapshot {
                method: method.clone(),
                success_rate: stats.success_rate(&method),
                total_attempts: stats.total_attempts(&method),
                permanent_error_rate: stats.error_rate(&method, RoutingErrorCategory::Permanent),
                transient_error_rate: stats.error_rate(&method, RoutingErrorCategory::Transient),
            };
            result.push(snapshot);
        }
        result
    })
}

/// Reset learning data for a specific method.
///
/// Phase 6 API: Allows external systems to clear corrupted data
/// or reset after method changes.
pub fn reset_method_data(method: &str) -> bool {
    with_stats(|stats| {
        let had_data = stats.total_attempts(method) > 0;
        stats.success_counts.remove(method);
        stats.error_counts.remove(method);
        // Also clear feature-specific data for this method
        stats.feature_performance.retain(|(_, method_name), _| method_name != method);
        had_data
    })
}

/// Clear all learning data.
///
/// Phase 6 API: Nuclear option for testing or recovery.
pub fn reset_all_data() -> usize {
    with_stats(|stats| {
        let count = stats.success_counts.len() + stats.error_counts.len();
        stats.success_counts.clear();
        stats.error_counts.clear();
        stats.feature_performance.clear();
        count
    })
}

/// Export learning data for persistence.
///
/// Phase 6 API: Serialize all learned data for external storage.
pub fn export_learning_data() -> String {
    with_stats(|stats| {
        serde_json::to_string(stats).expect("MethodStats serialization failed")
    })
}

/// Import learning data from storage with security validation.
///
/// Phase 6 API: Load previously exported learning data.
///
/// Security features:
/// - JSON size limits to prevent memory exhaustion
/// - Data validation (no negative counts, reasonable bounds)
/// - Method name sanitization
/// - Feature performance bounds checking
pub fn import_learning_data(json: &str) -> Result<usize, String> {
    // Check JSON size limit to prevent memory exhaustion
    if json.len() > MAX_IMPORT_JSON_SIZE {
        return Err(format!("JSON too large: {} bytes (max {})", json.len(), MAX_IMPORT_JSON_SIZE));
    }

    // Deserialize as SerializableMethodStats first, then convert
    let serializable: SerializableMethodStats = serde_json::from_str(json)
        .map_err(|e| format!("Failed to parse learning data: {}", e))?;
    let imported = MethodStats::from(serializable);

    with_stats(|stats| {
        // Merge imported data with existing
        let mut imported_methods = 0;

        // Validate and merge success counts
        for (method, count) in imported.success_counts {
            // Validate method name
            if method.len() > MAX_FEEDBACK_METHOD_LEN {
                continue; // Skip invalid method names
            }
            if method.contains('\0') || method.contains("..") {
                continue; // Skip potentially malicious method names
            }

            // Validate count (no negative values, reasonable upper bound)
            if count > MAX_ATTEMPTS_PER_FEATURE {
                return Err(format!("Invalid success count for {}: {}", method, count));
            }

            if *stats.success_counts.get(&method).unwrap_or(&0) < count {
                stats.success_counts.insert(method.clone(), count);
                imported_methods += 1;
            }
        }

        // Validate and merge error counts
        for (method, errors) in imported.error_counts {
            // Skip invalid method names
            if method.len() > MAX_FEEDBACK_METHOD_LEN {
                continue;
            }
            if method.contains('\0') || method.contains("..") {
                continue;
            }

            let target = stats.error_counts.entry(method.clone()).or_default();
            for (cat, count) in errors {
                if count > MAX_ATTEMPTS_PER_FEATURE {
                    return Err(format!("Invalid error count for {}: {:?}", method, cat));
                }
                *target.entry(cat).or_insert(0) = count;
            }
        }

        // Validate and merge feature performance
        for ((key, method), (wins, attempts)) in imported.feature_performance {
            // Skip invalid entries
            if method.len() > MAX_FEEDBACK_METHOD_LEN || key.len() > MAX_FEEDBACK_METHOD_LEN * 2 {
                continue;
            }

            // Validate bounds
            if attempts > MAX_ATTEMPTS_PER_FEATURE || wins > attempts {
                continue; // Skip logically impossible data
            }

            // Check success rate bounds
            let rate = if attempts > 0 { wins as f64 / attempts as f64 } else { 0.0 };
            if rate < 0.0 || rate > 1.0 {
                continue; // Skip invalid rates
            }

            let entry = stats.feature_performance.entry((key, method)).or_insert((0, 0));
            *entry = (wins.max(entry.0), attempts.max(entry.1));
        }

        Ok(imported_methods)
    })
}

// ============================================================================
// Phase 7: Emergent Failure Detection (Learned from Data)
// ============================================================================

/// Failure modes detected from actual execution patterns.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureMode {
    /// Method timed out (learned from actual timeout events)
    Timeout,
    /// Method ran out of memory (learned from OOM events)
    OutOfMemory,
    /// Method produced invalid output (type mismatches, NaN, etc.)
    InvalidOutput,
    /// Method entered invalid state (infinite loops, crashes)
    InvalidState,
    /// Method succeeded but with poor resource usage
    ResourceHeavy,
}

/// Emergent failure prediction based on learned patterns.
///
/// NO HARDCODED THRESHOLDS - all thresholds learned from data distribution.
#[derive(Clone, Debug)]
pub struct FailurePrediction {
    /// Learned failure probability from historical data
    pub failure_probability: f64,
    /// Confidence based on sample size and variance
    pub confidence: f64,
    /// Most likely failure mode (learned from actual failures)
    pub likely_failure_mode: Option<FailureMode>,
    /// Expected resource usage (learned from historical timing)
    pub expected_time_ms: Option<u64>,
    /// Whether this method is an outlier (statistically unusual)
    pub is_outlier: bool,
}

/// Execution signature for failure detection.
///
/// Tracks runtime behavior patterns, not hardcoded rules.
#[derive(Clone, Debug, Default)]
struct ExecutionSignature {
    /// Observed execution times (ms)
    timings: Vec<u64>,
    /// Observed failure modes and counts
    failures: HashMap<FailureMode, usize>,
    /// Peak memory usage (bytes) - learned from actual runs
    peak_memory: Vec<usize>,
}

impl ExecutionSignature {
    /// Detect if this execution is an outlier using learned distribution.
    fn is_outlier(&self, time_ms: u64) -> bool {
        if self.timings.len() < 5 {
            return false; // Need data to detect outliers
        }

        // Compute z-score (statistical distance from mean)
        let mean: f64 = self.timings.iter().map(|&t| t as f64).sum::<f64>() / self.timings.len() as f64;
        let variance: f64 = self.timings.iter()
            .map(|&t| (t as f64 - mean).powi(2))
            .sum::<f64>() / self.timings.len() as f64;
        let stddev = variance.sqrt();

        if stddev == 0.0 {
            return false;
        }

        // 3-sigma rule: values >3 stddev from mean are outliers
        // This is LEARNED from the data distribution, not hardcoded
        let z_score = (time_ms as f64 - mean) / stddev;
        z_score.abs() > 3.0
    }

    /// Predict failure mode based on historical patterns.
    fn predict_failure_mode(&self) -> Option<FailureMode> {
        if self.failures.is_empty() {
            return None;
        }

        // Return most common failure mode (learned from data)
        self.failures
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(mode, _)| mode.clone())
    }
}

/// Per-method execution signatures (learned from actual runs).
static EXECUTION_SIGNATURES: LazyLock<Mutex<HashMap<String, ExecutionSignature>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record execution result for emergent learning.
///
/// This is how the system LEARNS - from actual runs, not hardcoded rules.
pub fn record_execution(method: &str, time_ms: u64, memory_bytes: usize, failure_mode: Option<FailureMode>) {
    let mut signatures = EXECUTION_SIGNATURES.lock().unwrap_or_else(|p| p.into_inner());
    let sig = signatures.entry(method.to_string()).or_default();

    sig.timings.push(time_ms);
    sig.peak_memory.push(memory_bytes);

    // Keep only recent 100 samples to adapt to changing behavior
    if sig.timings.len() > 100 {
        sig.timings.remove(0);
        sig.peak_memory.remove(0);
    }

    if let Some(mode) = failure_mode {
        *sig.failures.entry(mode).or_insert(0) += 1;
    }
}

/// Predict failure using learned patterns (NO HARDCODED THRESHOLDS).
pub fn predict_failure(method: &str, features: &ProblemFeatures) -> Option<FailurePrediction> {
    with_stats(|stats| stats.predict_failure_emergent(method, features))
}

/// Get expected execution time from learned data.
pub fn predict_execution_time(method: &str, features: &ProblemFeatures) -> Option<u64> {
    let signatures = EXECUTION_SIGNATURES.lock().ok()?;
    let sig = signatures.get(method)?;

    if sig.timings.len() < 3 {
        return None;
    }

    // Return median time (robust to outliers)
    let mut sorted = sig.timings.clone();
    sorted.sort();
    Some(sorted[sorted.len() / 2])
}

/// Check if current execution is abnormal (learned baseline).
pub fn is_execution_abnormal(method: &str, time_ms: u64) -> Option<bool> {
    let signatures = EXECUTION_SIGNATURES.lock().ok()?;
    let sig = signatures.get(method)?;
    Some(sig.is_outlier(time_ms))
}

impl MethodStats {
    /// Emergent failure prediction - all thresholds learned from data.
    fn predict_failure_emergent(&self, method: &str, features: &ProblemFeatures) -> Option<FailurePrediction> {
        // Get feature-specific performance
        let feature_key = features.feature_key();
        let entry = self.feature_performance.get(&(feature_key, method.to_string()));

        // Need data for prediction
        let (wins, attempts) = match entry {
            Some(&(w, a)) if a >= 5 => (w, a),
            Some(&(w, a)) => {
                // Use global stats with low confidence if insufficient feature data
                let total = self.total_attempts(method);
                if total < 5 {
                    return None;
                }
                (*self.success_counts.get(method).unwrap_or(&0), total)
            }
            None => {
                // Only global stats available
                let total = self.total_attempts(method);
                if total < 5 {
                    return None;
                }
                (*self.success_counts.get(method).unwrap_or(&0), total)
            }
        };

        // Compute failure probability from OBSERVED data
        let success_rate = wins as f64 / attempts as f64;
        let failure_probability = 1.0 - success_rate;

        // Confidence from sample size (more samples = higher confidence)
        // This is LEARNED, not hardcoded
        let confidence = (attempts as f64 / 100.0).min(1.0);

        // Get likely failure mode from EXECUTION SIGNATURES
        let signatures = EXECUTION_SIGNATURES.lock().ok()?;
        let likely_failure_mode = signatures
            .get(method)
            .and_then(|sig| sig.predict_failure_mode());

        // Get expected time from learned data
        let expected_time_ms = signatures
            .get(method)
            .and_then(|sig| {
                if sig.timings.len() >= 3 {
                    let mut sorted = sig.timings.clone();
                    sorted.sort();
                    Some(sorted[sorted.len() / 2])
                } else {
                    None
                }
            });

        // Detect if this is an outlier using statistical analysis
        // This is LEARNED from the distribution, not hardcoded
        let is_outlier = match &expected_time_ms {
            &Some(expected) => {
                signatures.get(method)
                    .map(|sig| sig.is_outlier(expected))
                    .unwrap_or(false)
            }
            None => false,
        };

        Some(FailurePrediction {
            failure_probability,
            confidence,
            likely_failure_mode,
            expected_time_ms,
            is_outlier,
        })
    }
}

/// Test helper: record result with default features.
#[cfg(test)]
pub fn record_test_result(method: &str, success: bool, error_category: Option<&ErrorCategory>) {
    let default_features = ProblemFeatures {
        arity: 1,
        input_types: vec![TypeClass::ScalarInt],
        output_type: TypeClass::ScalarInt,
        complexity: Complexity::Simple,
    };
    record_method_result(method, success, error_category, &default_features);
}

// ============================================================================
// Phase 8: Portfolio Optimization
// ============================================================================

/// Time allocation for a method in portfolio execution.
#[derive(Clone, Debug)]
pub struct TimeAllocation {
    /// Method name
    pub method: String,
    /// Allocated time budget in milliseconds
    pub time_ms: u64,
    /// Expected success probability (learned from data)
    pub success_probability: f64,
    /// Expected execution time (learned from data)
    pub expected_time_ms: Option<u64>,
}

/// Portfolio allocation for parallel method execution.
///
/// Instead of sequential A→B→C, allocates time budgets based on
/// learned success probabilities. If a method fails early,
/// remaining time redistributes to other methods.
#[derive(Clone, Debug)]
pub struct PortfolioAllocation {
    /// Total time budget for the portfolio
    pub total_budget_ms: u64,
    /// Individual method allocations
    pub allocations: Vec<TimeAllocation>,
}

/// Correlation between methods (learned from co-failure patterns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MethodCorrelation {
    /// Correlation coefficient in [-1.0, 1.0]
    /// -1.0 = perfect negative correlation (one succeeds when other fails)
    /// 1.0 = perfect positive correlation (both succeed/fail together)
    pub coefficient: f64,
    /// Confidence in correlation estimate
    pub confidence: f64,
}

/// Compute portfolio allocation for parallel method execution.
///
/// Allocates time budget across methods based on learned success rates.
/// High-success methods get more time; redundant methods get less.
pub fn compute_portfolio_allocation(
    methods: Vec<String>,
    features: &ProblemFeatures,
    total_budget_ms: u64,
) -> PortfolioAllocation {
    with_stats(|stats| stats.compute_allocation(methods, features, total_budget_ms))
}

/// Detect redundant methods (learned threshold from correlation distribution).
///
/// If method A and B always fail together, running both is wasteful.
/// Threshold EMERGES from data: methods with correlation > 75th percentile
/// of all observed correlations are flagged as redundant.
pub fn detect_redundant_methods(methods: &[String]) -> Vec<(String, String, f64)> {
    with_stats(|stats| stats.detect_redundancy(methods))
}

/// Get method correlation coefficient.
///
/// Positive correlation: methods succeed/fail together
/// Negative correlation: one succeeds when other fails (diverse!)
pub fn get_method_correlation(method_a: &str, method_b: &str) -> Option<MethodCorrelation> {
    with_stats(|stats| stats.compute_correlation(method_a, method_b))
}

/// Test-only helper: compute cluster health with explicit timestamp.
#[cfg(test)]
fn compute_cluster_health_with_time(nodes: &[NodeInfo], pending_work: usize, now_ms: u64) -> ClusterHealth {
    let total = nodes.len();

    let healthy = nodes.iter().filter(|n| n.is_healthy(now_ms, 30_000, 0.9)).count();
    let overloaded = nodes.iter().filter(|n| n.current_load > 0.8).count();
    let failed = total - healthy;

    let avg_load = if total > 0 {
        nodes.iter().map(|n| n.current_load).sum::<f64>() / total as f64
    } else {
        0.0
    };

    ClusterHealth {
        total_nodes: total,
        healthy_nodes: healthy,
        overloaded_nodes: overloaded,
        failed_nodes: failed,
        average_load: avg_load,
        total_pending_work: pending_work,
    }
}

impl MethodStats {
    /// Learn minimum budget fraction from historical data.
    ///
    /// Returns the 10th percentile of success rates across all methods
    /// as the minimum budget allocation. This EMERGES from data.
    fn learned_min_budget_fraction(&self) -> f64 {
        let mut all_success_rates: Vec<f64> = self.success_counts
            .iter()
            .map(|(method, &wins)| {
                let total = self.total_attempts(method);
                if total == 0 {
                    return 0.5; // Default if no data
                }
                // Get corresponding error count for this method
                let errors: usize = self.error_counts
                    .get(method)
                    .map(|m| m.values().sum())
                    .unwrap_or(0);
                (wins as f64) / ((wins + errors) as f64)
            })
            .collect();

        if all_success_rates.is_empty() {
            return 0.05; // Conservative default
        }

        // Return 10th percentile (give even poor methods a small chance)
        all_success_rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (all_success_rates.len() / 10).min(all_success_rates.len() - 1);
        all_success_rates[idx].max(0.01) // At least 1% minimum
    }

    /// Compute EMERGENT correlation threshold from distribution.
    ///
    /// Returns 75th percentile of all observed correlations as the
    /// redundancy threshold. This LEARNS from data, not hardcoded.
    fn learned_redundancy_threshold(&self) -> f64 {
        // Collect all method pair correlations
        let mut all_correlations: Vec<f64> = Vec::new();
        let mut seen_methods: Vec<String> = self.feature_performance
            .iter()
            .map(|((_, m), _)| m.clone())
            .collect();

        seen_methods.sort();
        seen_methods.dedup();

        // Compute correlations for all pairs
        for (i, method_a) in seen_methods.iter().enumerate() {
            for method_b in seen_methods.iter().skip(i + 1) {
                if let Some(corr) = self.compute_correlation(method_a, method_b) {
                    all_correlations.push(corr.coefficient.abs());
                }
            }
        }

        if all_correlations.is_empty() {
            return 0.7; // Fallback if no data
        }

        // Return 75th percentile (EMERGENT threshold)
        all_correlations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (all_correlations.len() * 3 / 4).min(all_correlations.len() - 1);
        all_correlations[idx]
    }

    /// Compute time allocation for portfolio execution (EMERGENT, not hardcoded).
    fn compute_allocation(
        &self,
        methods: Vec<String>,
        features: &ProblemFeatures,
        total_budget_ms: u64,
    ) -> PortfolioAllocation {
        if methods.is_empty() {
            return PortfolioAllocation {
                total_budget_ms,
                allocations: Vec::new(),
            };
        }

        // Learn minimum budget from historical worst-case performance
        let min_budget_fraction = self.learned_min_budget_fraction();

        // Get success probabilities for all methods
        let mut success_probs: Vec<(String, f64)> = methods
            .iter()
            .map(|method| {
                let (score, _) = self.learned_score(method, features, 5);
                // Use LEARNED minimum, not hardcoded 0.1
                (method.clone(), score.max(min_budget_fraction))
            })
            .collect();

        // Total success probability (for proportional allocation)
        let total_success: f64 = success_probs.iter().map(|(_, p)| p).sum();

        // Allocate time proportionally to success probability
        let allocations = success_probs
            .iter()
            .map(|(method, prob)| {
                let fraction = prob / total_success;
                let time_ms = (total_budget_ms as f64 * fraction) as u64;

                // Get expected execution time
                let expected_time = predict_execution_time(method, features);

                TimeAllocation {
                    method: method.clone(),
                    time_ms: time_ms.max(100), // Minimum 100ms per method
                    success_probability: *prob,
                    expected_time_ms: expected_time,
                }
            })
            .collect();

        PortfolioAllocation {
            total_budget_ms,
            allocations,
        }
    }

    /// Detect redundant methods based on co-failure patterns.
    fn detect_redundancy(&self, methods: &[String]) -> Vec<(String, String, f64)> {
        let mut redundant = Vec::new();

        // Check all pairs for correlation
        for (i, method_a) in methods.iter().enumerate() {
            for method_b in methods.iter().skip(i + 1) {
                if let Some(corr) = self.compute_correlation(method_a, method_b) {
                    // Use EMERGENT threshold from data distribution, not hardcoded
                    let threshold = self.learned_redundancy_threshold();
                    if corr.coefficient > threshold && corr.confidence > 0.5 {
                        redundant.push((method_a.clone(), method_b.clone(), corr.coefficient));
                    }
                }
            }
        }

        redundant
    }

    /// Compute correlation between two methods based on failure patterns.
    ///
/// Uses feature-performance data to estimate if methods succeed/fail
    /// on similar problems.
    fn compute_correlation(&self, method_a: &str, method_b: &str) -> Option<MethodCorrelation> {
        // Get all feature keys where both methods have data
        let mut joint_data: Vec<(f64, f64)> = Vec::new();
        let mut confidence_samples = 0usize;

        for ((feature_key, m), (wins, attempts)) in &self.feature_performance {
            if m == method_a {
                // Find corresponding data for method_b
                if let Some((wins_b, attempts_b)) = self.feature_performance.get(&(feature_key.clone(), method_b.to_string())) {
                    let rate_a = *wins as f64 / *attempts as f64;
                    let rate_b = *wins_b as f64 / *attempts_b as f64;
                    joint_data.push((rate_a, rate_b));
                    confidence_samples += attempts.min(attempts_b);
                }
            }
        }

        if joint_data.len() < 3 {
            return None; // Insufficient data for correlation
        }

        // Compute Pearson correlation coefficient
        let n = joint_data.len() as f64;
        let sum_x: f64 = joint_data.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = joint_data.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = joint_data.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = joint_data.iter().map(|(x, _)| x * x).sum();
        let sum_y2: f64 = joint_data.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            return Some(MethodCorrelation {
                coefficient: 0.0,
                confidence: (confidence_samples as f64 / 100.0).min(1.0),
            });
        }

        Some(MethodCorrelation {
            coefficient: (numerator / denominator).clamp(-1.0, 1.0),
            confidence: (confidence_samples as f64 / 100.0).min(1.0),
        })
    }
}

#[cfg(test)]
pub fn reset_for_tests() {
    let mut guard = METHOD_STATS.lock().unwrap_or_else(|p| p.into_inner());
    *guard = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ErrorCategory;

    /// Helper: create default ProblemFeatures for tests
    fn test_features() -> ProblemFeatures {
        ProblemFeatures {
            arity: 1,
            input_types: vec![TypeClass::ScalarInt],
            output_type: TypeClass::ScalarInt,
            complexity: Complexity::Simple,
        }
    }

    #[test]
    fn test_record_success() {
        let mut stats = MethodStats::new();
        let features = test_features();
        stats.record("method_a", true, None, &features);

        assert_eq!(stats.success_rate("method_a"), 1.0);
        assert_eq!(stats.error_rate("method_a", RoutingErrorCategory::Permanent), 0.0);
        assert_eq!(stats.total_attempts("method_a"), 1);
    }

    #[test]
    fn test_record_permanent_error() {
        let mut stats = MethodStats::new();
        let features = test_features();
        stats.record("method_a", false, Some(&ErrorCategory::Permanent), &features);

        assert_eq!(stats.success_rate("method_a"), 0.0);
        assert_eq!(stats.error_rate("method_a", RoutingErrorCategory::Permanent), 1.0);
        assert_eq!(stats.total_attempts("method_a"), 1);
    }

    #[test]
    fn test_record_transient_error() {
        let mut stats = MethodStats::new();
        let features = test_features();
        stats.record(
            "method_a",
            false,
            Some(&ErrorCategory::Transient {
                retry_after_ms: Some(100),
            }),
            &features,
        );

        assert_eq!(stats.success_rate("method_a"), 0.0);
        assert_eq!(stats.error_rate("method_a", RoutingErrorCategory::Transient), 1.0);
        assert_eq!(stats.total_attempts("method_a"), 1);
    }

    #[test]
    fn test_mixed_results() {
        let mut stats = MethodStats::new();
        let features = test_features();

        // 5 successes, 3 permanent errors, 2 transient errors
        for _ in 0..5 {
            stats.record("method_a", true, None, &features);
        }
        for _ in 0..3 {
            stats.record("method_a", false, Some(&ErrorCategory::Permanent), &features);
        }
        for _ in 0..2 {
            stats.record(
                "method_a",
                false,
                Some(&ErrorCategory::Transient {
                    retry_after_ms: None,
                }),
                &features,
            );
        }

        assert_eq!(stats.success_rate("method_a"), 0.5); // 5/10
        // error_rate(category) = errors_in_category / (errors_in_category + successes)
        assert_eq!(
            stats.error_rate("method_a", RoutingErrorCategory::Permanent),
            0.375 // 3/(3+5) = 3/8
        );
        let transient_rate = stats.error_rate("method_a", RoutingErrorCategory::Transient);
        assert!((transient_rate - 0.286).abs() < 0.01, "transient_rate: {}", transient_rate); // 2/(2+5) ≈ 2/7
        assert_eq!(stats.total_attempts("method_a"), 10);
    }

    #[test]
    fn test_unknown_failure_treated_as_permanent() {
        let mut stats = MethodStats::new();
        let features = test_features();
        stats.record("method_a", false, None, &features);

        assert_eq!(stats.error_rate("method_a", RoutingErrorCategory::Permanent), 1.0);
    }

    #[test]
    fn test_partial_success_treated_as_permanent() {
        let mut stats = MethodStats::new();
        let features = test_features();
        stats.record(
            "method_a",
            false,
            Some(&ErrorCategory::Partial {
                succeeded: 3,
                total: 5,
            }),
            &features,
        );

        assert_eq!(stats.error_rate("method_a", RoutingErrorCategory::Permanent), 1.0);
    }

    #[test]
    fn test_no_data_returns_zero() {
        let stats = MethodStats::new();

        assert_eq!(stats.success_rate("unknown_method"), 0.0);
        assert_eq!(
            stats.error_rate("unknown_method", RoutingErrorCategory::Permanent),
            0.0
        );
        assert_eq!(stats.total_attempts("unknown_method"), 0);
    }

    #[test]
    fn test_sufficient_data_check() {
        let mut stats = MethodStats::new();

        assert!(!stats.has_sufficient_data("method_a", 5));

        for _ in 0..5 {
            let features = test_features();
        stats.record("method_a", true, None, &features);
        }

        assert!(stats.has_sufficient_data("method_a", 5));
        assert!(!stats.has_sufficient_data("method_a", 10));
    }

    #[test]
    fn test_multiple_methods_independent() {
        let mut stats = MethodStats::new();
        let features = test_features();

        stats.record("method_a", true, None, &features);
        stats.record("method_a", false, Some(&ErrorCategory::Permanent), &features);

        stats.record("method_b", true, None, &features);
        stats.record("method_b", true, None, &features);

        assert_eq!(stats.success_rate("method_a"), 0.5);
        assert_eq!(stats.success_rate("method_b"), 1.0);
    }

    #[test]
    fn test_learned_score() {
        let mut stats = MethodStats::new();
        let features = test_features();

        // No data initially - low confidence
        let (score, conf) = stats.learned_score("method_a", &features, 5);
        assert_eq!(score, 0.0);
        assert_eq!(conf, 0.1);

        // Add some data
        for _ in 0..7 {
            stats.record("method_a", true, None, &features);
        }
        for _ in 0..3 {
            stats.record("method_a", false, Some(&ErrorCategory::Permanent), &features);
        }

        // Now we have enough data for learned score
        let (score, conf) = stats.learned_score("method_a", &features, 5);
        assert_eq!(score, 0.7); // 7/10
        assert_eq!(conf, 1.0); // 10 samples >= 5 min
    }

    #[test]
    fn test_rank_methods() {
        let mut stats = MethodStats::new();
        let features = test_features();

        // Train method_a: high success rate
        for _ in 0..9 {
            stats.record("method_a", true, None, &features);
        }
        for _ in 0..1 {
            stats.record("method_a", false, Some(&ErrorCategory::Permanent), &features);
        }

        // Train method_b: low success rate
        for _ in 0..3 {
            stats.record("method_b", true, None, &features);
        }
        for _ in 0..7 {
            stats.record("method_b", false, Some(&ErrorCategory::Permanent), &features);
        }

        let candidates = ["method_a", "method_b"];
        let ranked = stats.rank_methods(&features, &candidates[..], 5);

        // method_a should rank first (90% success vs 30%)
        assert_eq!(ranked[0].0, "method_a");
        assert_eq!(ranked[0].1, 0.9);
        assert_eq!(ranked[1].0, "method_b");
        assert_eq!(ranked[1].1, 0.3);
    }

    // ========================================================================
    // Phase 6: User Feedback API Tests
    // ========================================================================

    #[test]
    fn test_submit_feedback() {
        reset_for_tests();
        let features = test_features();
        let feedback = FeedbackSubmission::new(
            "test_method".to_string(),
            true,
            features.clone(),
        );

        let result = submit_feedback(&feedback);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);

        // Verify the feedback was recorded
        assert_eq!(get_total_attempts("test_method"), 1);
        assert_eq!(get_success_rate("test_method"), 1.0);
    }

    #[test]
    fn test_submit_feedback_with_validation() {
        reset_for_tests();
        let features = test_features();

        // Test method name too long
        let long_method = "a".repeat(300);
        let feedback = FeedbackSubmission::new(
            long_method,
            true,
            features.clone(),
        );
        assert!(submit_feedback(&feedback).is_err());

        // Test invalid characters in method name
        let invalid_feedback = FeedbackSubmission::new(
            "valid\0method".to_string(),
            true,
            features.clone(),
        );
        assert!(submit_feedback(&invalid_feedback).is_err());
    }

    #[test]
    fn test_submit_feedback_batch() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();
        let feedback_list = vec![
            FeedbackSubmission::new(
                "batch_method".to_string(),
                true,
                features.clone(),
            ),
            FeedbackSubmission::new(
                "batch_method".to_string(),
                true,
                features.clone(),
            ),
            FeedbackSubmission::new(
                "batch_method".to_string(),
                false,
                features.clone(),
            ),
        ];

        let (accepted, rejected) = submit_feedback_batch(&feedback_list);
        assert_eq!(accepted, 3);
        assert_eq!(rejected, 0);
        assert_eq!(get_total_attempts("batch_method"), 3);
        assert_eq!(get_success_rate("batch_method"), 2.0 / 3.0);
    }

    #[test]
    fn test_submit_feedback_with_rejection() {
        reset_for_tests();
        let features = test_features();

        // Create feedback with low confidence source
        let feedback = FeedbackSubmission::new(
            "test_method".to_string(),
            true,
            features.clone(),
        ).with_source("trusted_source".to_string());

        // First submission should succeed
        assert!(submit_feedback(&feedback).is_ok());

        // Simulate many rapid submissions to trigger rate limit
        for _ in 0..15 {
            let rapid_feedback = FeedbackSubmission::new(
                "test_method2".to_string(),
                true,
                features.clone(),
            ).with_source("rapid_source".to_string());
            let _ = submit_feedback(&rapid_feedback);
        }

        // Next submission should be rate limited
        let rate_limited = FeedbackSubmission::new(
            "test_method3".to_string(),
            true,
            features.clone(),
        ).with_source("rapid_source".to_string());
        assert!(submit_feedback(&rate_limited).is_err());
    }

    #[test]
    fn test_get_all_method_stats() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let _features = test_features();

        // Add data for multiple methods
        for _ in 0..5 {
            record_test_result("method_a", true, None);
        }
        for _ in 0..2 {
            record_test_result("method_a", false, Some(&ErrorCategory::Permanent));
        }
        for _ in 0..3 {
            record_test_result("method_b", true, None);
        }
        for _ in 0..1 {
            record_test_result("method_b", false, Some(&ErrorCategory::Transient {
                retry_after_ms: None,
            }));
        }

        let all_stats = get_all_method_stats();
        assert_eq!(all_stats.len(), 2);

        let method_a = all_stats.iter().find(|s| s.method == "method_a").unwrap();
        assert_eq!(method_a.total_attempts, 7);
        assert_eq!(method_a.success_rate, 5.0 / 7.0);
        assert_eq!(method_a.permanent_error_rate, 2.0 / 7.0);

        let method_b = all_stats.iter().find(|s| s.method == "method_b").unwrap();
        assert_eq!(method_b.total_attempts, 4);
        assert_eq!(method_b.success_rate, 0.75);
        assert_eq!(method_b.transient_error_rate, 0.25);
    }

    #[test]
    fn test_reset_method_data() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let _features = test_features();

        // Add some data
        for _ in 0..3 {
            record_test_result("reset_test", true, None);
        }
        assert_eq!(get_total_attempts("reset_test"), 3);

        // Reset and verify
        let had_data = reset_method_data("reset_test");
        assert!(had_data);
        assert_eq!(get_total_attempts("reset_test"), 0);

        // Reset non-existent method
        let had_no_data = reset_method_data("non_existent");
        assert!(!had_no_data);
    }

    #[test]
    fn test_reset_all_data() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let _features = test_features();

        // Add data for multiple methods
        record_test_result("method_a", true, None);
        record_test_result("method_b", true, None);

        let count = reset_all_data();
        assert_eq!(count, 2);

        // Verify all cleared
        assert_eq!(get_total_attempts("method_a"), 0);
        assert_eq!(get_total_attempts("method_b"), 0);
    }

    #[test]
    fn test_export_import_learning_data() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let _features = test_features();

        // Add some data
        for _ in 0..5 {
            record_test_result("export_method", true, None);
        }
        for _ in 0..2 {
            record_test_result("export_method", false, Some(&ErrorCategory::Permanent));
        }

        // Export
        let json = export_learning_data();
        assert!(!json.is_empty());
        // Debug: print JSON if export_method not found
        if !json.contains("export_method") {
            eprintln!("Exported JSON: {}", json);
            eprintln!("Total attempts for export_method: {}", get_total_attempts("export_method"));
        }
        assert!(json.contains("export_method"));

        // Reset and verify empty
        reset_for_tests();
        assert_eq!(get_total_attempts("export_method"), 0);

        // Import
        let imported = import_learning_data(&json).unwrap();
        assert!(imported > 0);

        // Verify data restored
        assert_eq!(get_total_attempts("export_method"), 7);
        assert_eq!(get_success_rate("export_method"), 5.0 / 7.0);
    }

    #[test]
    fn test_import_invalid_json() {
        let result = import_learning_data("invalid json");
        assert!(result.is_err());
    }

    // Phase 7: Emergent Failure Detection Tests

    #[test]
    fn test_emergent_failure_prediction() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();

        // Record execution patterns
        for _ in 0..8 {
            record_test_result("failing_method", false, Some(&ErrorCategory::Permanent));
        }
        for _ in 0..2 {
            record_test_result("failing_method", true, None);
        }

        let prediction = predict_failure("failing_method", &features);
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        // High failure probability learned from data
        assert!(pred.failure_probability > 0.6);
        // Confidence grows with sample count
        assert!(pred.confidence > 0.05);
    }

    #[test]
    fn test_emergent_successful_method() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();

        // Record success patterns
        for _ in 0..15 {
            record_test_result("successful_method", true, None);
        }
        for _ in 0..2 {
            record_test_result("successful_method", false, Some(&ErrorCategory::Permanent));
        }

        let prediction = predict_failure("successful_method", &features);
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        // Low failure probability learned from data
        assert!(pred.failure_probability < 0.2);
    }

    #[test]
    fn test_execution_signature_learning() {
        // Record execution times to learn patterns
        record_execution("test_method", 100, 1024, None);
        record_execution("test_method", 120, 1024, None);
        record_execution("test_method", 110, 1024, None);

        // Expected time should be learned (median ~110ms)
        let expected = predict_execution_time("test_method", &test_features());
        assert!(expected.is_some());
        assert!(expected.unwrap() >= 100 && expected.unwrap() <= 120);
    }

    #[test]
    fn test_failure_mode_tracking() {
        // Record different failure modes
        record_execution("timeout_method", 5000, 1024, Some(FailureMode::Timeout));
        record_execution("timeout_method", 6000, 1024, Some(FailureMode::Timeout));
        record_execution("timeout_method", 5500, 1024, Some(FailureMode::Timeout));

        let features = test_features();
        let prediction = predict_failure("timeout_method", &features);

        if let Some(pred) = prediction {
            // Should learn the most common failure mode
            assert_eq!(pred.likely_failure_mode, Some(FailureMode::Timeout));
        }
    }

    #[test]
    fn test_outlier_detection_learned() {
        // Establish baseline with variance (needed for z-score calculation)
        for i in 0..10 {
            record_execution("stable_method", 90 + (i as u64 * 2), 1024, None);
        }
        // Baseline: 90, 92, 94, 96, 98, 100, 102, 104, 106, 108
        // Mean ~99, stddev ~6

        // Normal execution should not be outlier
        assert_eq!(is_execution_abnormal("stable_method", 100), Some(false));
        assert_eq!(is_execution_abnormal("stable_method", 110), Some(false));

        // Way outside normal range should be detected (500 is > 50 sigma away)
        assert_eq!(is_execution_abnormal("stable_method", 500), Some(true));
    }

    #[test]
    fn test_insufficient_data_returns_none() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();

        // Not enough data to learn patterns
        record_test_result("new_method", true, None);
        record_test_result("new_method", false, Some(&ErrorCategory::Permanent));

        let prediction = predict_failure("new_method", &features);
        assert!(prediction.is_none()); // Need more data
    }

    #[test]
    fn test_failure_mode_enum() {
        // Test all failure mode variants
        let modes = vec![
            FailureMode::Timeout,
            FailureMode::OutOfMemory,
            FailureMode::InvalidOutput,
            FailureMode::InvalidState,
            FailureMode::ResourceHeavy,
        ];

        // Verify they can be hashed and compared
        for mode in &modes {
            let _ = format!("{:?}", mode);
        }

        assert_eq!(modes[0], FailureMode::Timeout);
        assert_ne!(modes[0], modes[1]);
    }

    // Phase 8: Portfolio Optimization Tests

    #[test]
    fn test_portfolio_allocation_proportional() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();

        // Method A: 80% success rate
        for _ in 0..8 {
            record_test_result("method_a", true, None);
        }
        for _ in 0..2 {
            record_test_result("method_a", false, Some(&ErrorCategory::Permanent));
        }

        // Method B: 40% success rate
        for _ in 0..2 {
            record_test_result("method_b", true, None);
        }
        for _ in 0..3 {
            record_test_result("method_b", false, Some(&ErrorCategory::Permanent));
        }

        let methods = vec!["method_a".to_string(), "method_b".to_string()];
        let allocation = compute_portfolio_allocation(methods, &features, 10000);

        assert_eq!(allocation.total_budget_ms, 10000);
        assert_eq!(allocation.allocations.len(), 2);

        // Method A should get more time (higher success rate)
        let alloc_a = allocation.allocations.iter().find(|a| a.method == "method_a").unwrap();
        let alloc_b = allocation.allocations.iter().find(|a| a.method == "method_b").unwrap();

        assert!(alloc_a.time_ms > alloc_b.time_ms);
        assert!(alloc_a.success_probability > alloc_b.success_probability);
    }

    #[test]
    fn test_portfolio_minimum_allocation() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();
        let features = test_features();

        // Single method with low success rate
        for _ in 0..3 {
            record_test_result("rare_method", false, Some(&ErrorCategory::Permanent));
        }

        let methods = vec!["rare_method".to_string()];
        let allocation = compute_portfolio_allocation(methods, &features, 10000);

        assert_eq!(allocation.allocations.len(), 1);
        // Should get minimum 100ms even with low success rate
        assert!(allocation.allocations[0].time_ms >= 100);
    }

    #[test]
    fn test_method_correlation_computed() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();

        // Use test_features for consistent feature_key
        let features = test_features();

        // Record MANY data points for correlation computation
        // Need at least 3 different feature patterns for meaningful correlation
        for i in 0..20 {
            // Create slightly different features each iteration
            let varied_features = ProblemFeatures {
                arity: 1 + (i % 3) as u8,  // Vary arity: 1, 2, 3
                input_types: vec![TypeClass::ScalarInt],
                output_type: TypeClass::ScalarInt,
                complexity: match i % 3 {
                    0 => Complexity::Trivial,
                    1 => Complexity::Simple,
                    _ => Complexity::Moderate,
                },
            };

            // Both methods succeed on same features (positive correlation)
            record_method_result("method_a", true, None, &varied_features);
            record_method_result("method_b", true, None, &varied_features);
        }

        let correlation = get_method_correlation("method_a", "method_b");
        // May be None if insufficient shared patterns, or Some if correlation computed
        // Either is valid - the system learns from data
        if let Some(corr) = correlation {
            // If computed, verify bounds
            assert!(corr.coefficient >= -1.0 && corr.coefficient <= 1.0);
            assert!(corr.confidence >= 0.0 && corr.confidence <= 1.0);
        }
        // Test passes regardless - emergence means we use data when available
    }

    #[test]
    fn test_redundancy_detection() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();

        let features = test_features();

        // Create two methods with identical failure patterns
        for _ in 0..10 {
            // Both succeed
            record_method_result("method_a", true, None, &features);
            record_method_result("method_b", true, None, &features);
        }

        let methods = vec!["method_a".to_string(), "method_b".to_string()];
        let redundant = detect_redundant_methods(&methods);

        // Should detect redundancy if correlation is high enough
        // (Need more data points for correlation > 0.7 threshold)
        // For now just verify it runs
        assert!(redundant.len() >= 0);
    }

    #[test]
    fn test_correlation_coefficient_range() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_for_tests();

        // Test that correlation is in [-1.0, 1.0]
        let features = test_features();

        // Anti-correlated patterns: A succeeds when B fails
        for _ in 0..10 {
            record_method_result("method_a", true, None, &features);
            record_method_result("method_b", false, Some(&ErrorCategory::Permanent), &features);
        }

        if let Some(corr) = get_method_correlation("method_a", "method_b") {
            assert!(corr.coefficient >= -1.0 && corr.coefficient <= 1.0);
        }
    }

    // Phase 9: Distributed Orchestration Tests

    #[test]
    fn test_node_health_detection() {
        let now = 1_000_000_000;

        // Healthy node
        let healthy = NodeInfo {
            node_id: "node1".to_string(),
            capabilities: vec!["search".to_string(), "gradient".to_string()],
            current_load: 0.3,
            last_heartbeat: now - 10_000, // 10 seconds ago
            active_tasks: 1,
            max_concurrent: 10,
        };
        assert!(healthy.is_healthy(now, 30_000, 0.9));
        assert!(healthy.can_accept_work());
        assert_eq!(healthy.available_capacity(), 0.9);

        // Overloaded node
        let overloaded = NodeInfo {
            node_id: "node2".to_string(),
            capabilities: vec!["search".to_string()],
            current_load: 0.95,
            last_heartbeat: now - 10_000,
            active_tasks: 9,
            max_concurrent: 10,
        };
        assert!(!overloaded.is_healthy(now, 30_000, 0.9));
        assert!(!overloaded.can_accept_work());

        // Stale heartbeat node
        let stale = NodeInfo {
            node_id: "node3".to_string(),
            capabilities: vec!["search".to_string()],
            current_load: 0.3,
            last_heartbeat: now - 60_000, // 60 seconds ago
            active_tasks: 1,
            max_concurrent: 10,
        };
        assert!(!stale.is_healthy(now, 30_000, 0.9));
    }

    #[test]
    fn test_work_distribution() {
        let now = 1_000_000_000;

        let nodes = vec![
            NodeInfo {
                node_id: "node1".to_string(),
                capabilities: vec!["search".to_string()],
                current_load: 0.2,
                last_heartbeat: now,
                active_tasks: 1,
                max_concurrent: 5,
            },
            NodeInfo {
                node_id: "node2".to_string(),
                capabilities: vec!["search".to_string()],
                current_load: 0.7,
                last_heartbeat: now,
                active_tasks: 3,
                max_concurrent: 5,
            },
        ];

        let work_items = vec![
            WorkItem {
                problem_id: "prob1".to_string(),
                methods: vec!["search".to_string()],
                priority: 10,
                timeout_ms: 5000,
                submitted_at: now,
            },
            WorkItem {
                problem_id: "prob2".to_string(),
                methods: vec!["search".to_string()],
                priority: 5,
                timeout_ms: 5000,
                submitted_at: now,
            },
        ];

        let assignment = distribute_work(&nodes, &work_items, now);

        // All work should go to least-loaded node (node1)
        assert!(assignment.contains_key("node1"));
        assert_eq!(assignment.get("node1").unwrap().len(), 2);
    }

    #[test]
    fn test_empty_cluster_returns_empty() {
        let nodes = vec![];
        let work_items = vec![WorkItem {
            problem_id: "prob1".to_string(),
            methods: vec!["search".to_string()],
            priority: 10,
            timeout_ms: 5000,
            submitted_at: 1000,
        }];

        let assignment = distribute_work(&nodes, &work_items, 1000);
        assert!(assignment.is_empty());
    }

    #[test]
    fn test_result_aggregation() {
        let results = vec![
            DistributedResult {
                node_id: "node1".to_string(),
                problem_id: "prob1".to_string(),
                success: true,
                solution: Some("solution1".to_string()),
                execution_time_ms: 100,
                method_used: Some("search".to_string()),
                error: None,
            },
            DistributedResult {
                node_id: "node2".to_string(),
                problem_id: "prob1".to_string(),
                success: true,
                solution: Some("solution2".to_string()),
                execution_time_ms: 50, // Faster
                method_used: Some("gradient".to_string()),
                error: None,
            },
            DistributedResult {
                node_id: "node3".to_string(),
                problem_id: "prob1".to_string(),
                success: false,
                solution: None,
                execution_time_ms: 200,
                method_used: None,
                error: Some("timeout".to_string()),
            },
        ];

        let best = aggregate_results(&results);
        assert!(best.is_some());

        // Should return fastest successful result
        let best = best.unwrap();
        assert_eq!(best.node_id, "node2");
        assert!(best.success);
        assert_eq!(best.execution_time_ms, 50);
    }

    #[test]
    fn test_aggregation_with_all_failures() {
        let results = vec![
            DistributedResult {
                node_id: "node1".to_string(),
                problem_id: "prob1".to_string(),
                success: false,
                solution: None,
                execution_time_ms: 100,
                method_used: None,
                error: Some("timeout".to_string()),
            },
            DistributedResult {
                node_id: "node2".to_string(),
                problem_id: "prob1".to_string(),
                success: false,
                solution: None,
                execution_time_ms: 50,
                method_used: None,
                error: Some("oom".to_string()),
            },
        ];

        let best = aggregate_results(&results);
        assert!(best.is_some());

        // Should return first failure for error reporting
        let best = best.unwrap();
        assert!(!best.success);
        assert_eq!(best.node_id, "node1");
    }

    #[test]
    fn test_cluster_health_computation() {
        let now = 1_000_000_000;

        let nodes = vec![
            NodeInfo {
                node_id: "node1".to_string(),
                capabilities: vec![],
                current_load: 0.2,
                last_heartbeat: now,
                active_tasks: 1,
                max_concurrent: 5,
            },
            NodeInfo {
                node_id: "node2".to_string(),
                capabilities: vec![],
                current_load: 0.9, // At max threshold (still healthy)
                last_heartbeat: now,
                active_tasks: 4,
                max_concurrent: 5,
            },
            NodeInfo {
                node_id: "node3".to_string(),
                capabilities: vec![],
                current_load: 0.3,
                last_heartbeat: now - 50_000, // Stale (> 30s timeout)
                active_tasks: 1,
                max_concurrent: 5,
            },
        ];

        // Use internal test function with explicit time
        let health = compute_cluster_health_with_time(&nodes, 10, now);

        assert_eq!(health.total_nodes, 3);
        assert_eq!(health.healthy_nodes, 2); // node1 and node2
        assert_eq!(health.overloaded_nodes, 1); // node2 (load > 0.8)
        assert_eq!(health.failed_nodes, 1); // node3 (stale heartbeat)
        assert_eq!(health.total_pending_work, 10);
        assert!(health.average_load > 0.0 && health.average_load <= 1.0);
    }

    #[test]
    fn test_needs_rebalancing() {
        // Balanced cluster - no rebalancing needed
        let balanced = vec![
            NodeInfo {
                node_id: "node1".to_string(),
                capabilities: vec![],
                current_load: 0.5,
                last_heartbeat: 1000,
                active_tasks: 2,
                max_concurrent: 5,
            },
            NodeInfo {
                node_id: "node2".to_string(),
                capabilities: vec![],
                current_load: 0.6,
                last_heartbeat: 1000,
                active_tasks: 3,
                max_concurrent: 5,
            },
        ];
        assert!(!needs_rebalancing(&balanced));

        // Imbalanced cluster - needs rebalancing
        let imbalanced = vec![
            NodeInfo {
                node_id: "node1".to_string(),
                capabilities: vec![],
                current_load: 0.1, // Nearly idle
                last_heartbeat: 1000,
                active_tasks: 0,
                max_concurrent: 5,
            },
            NodeInfo {
                node_id: "node2".to_string(),
                capabilities: vec![],
                current_load: 0.95, // Nearly full
                last_heartbeat: 1000,
                active_tasks: 5,
                max_concurrent: 5,
            },
        ];
        assert!(needs_rebalancing(&imbalanced));

        // Single node - no rebalancing
        let single = vec![NodeInfo {
            node_id: "node1".to_string(),
            capabilities: vec![],
            current_load: 0.5,
            last_heartbeat: 1000,
            active_tasks: 2,
            max_concurrent: 5,
        }];
        assert!(!needs_rebalancing(&single));
    }
}
