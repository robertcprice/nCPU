// Dependency Resolution System for Agent Orchestration
// Handles circular dependency detection, topological sorting, and incremental updates

use crate::solver::SolverError;
use petgraph::algo::toposort;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::prelude::*;
use petgraph::visit::IntoNeighbors;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Unique identifier for dependencies in the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DependencyId(pub u64);

impl DependencyId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for DependencyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dep({})", self.0)
    }
}

impl fmt::Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for DependencyEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.edge_type)
    }
}

/// Type of dependency relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Standard required dependency - must be satisfied first
    Required,
    /// Optional dependency - can be skipped if unavailable
    Optional,
    /// Weak dependency - used for optimization but not required
    Weak,
    /// Conflicting dependency - cannot coexist
    Conflict,
}

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyType::Required => write!(f, "required"),
            DependencyType::Optional => write!(f, "optional"),
            DependencyType::Weak => write!(f, "weak"),
            DependencyType::Conflict => write!(f, "conflict"),
        }
    }
}

/// A single dependency relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Unique identifier
    pub id: DependencyId,
    /// Name/label of the dependency
    pub name: String,
    /// Type of dependency relationship
    pub dep_type: DependencyType,
    /// Metadata associated with this dependency
    pub metadata: serde_json::Value,
}

impl Dependency {
    pub fn new(id: DependencyId, name: impl Into<String>, dep_type: DependencyType) -> Self {
        Self {
            id,
            name: name.into(),
            dep_type,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Edge in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Type of relationship
    pub edge_type: DependencyType,
    /// Priority for ordering (higher = earlier)
    pub priority: i32,
    /// Whether this edge is active (for incremental updates)
    pub active: bool,
    /// Timestamp when this edge was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl DependencyEdge {
    pub fn new(edge_type: DependencyType) -> Self {
        Self {
            edge_type,
            priority: 0,
            active: true,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// Result of circular dependency detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleDetectionResult {
    /// Whether cycles were found
    pub has_cycles: bool,
    /// List of detected cycles (each is a vector of dependency IDs)
    pub cycles: Vec<Vec<DependencyId>>,
    /// Total number of cycles detected
    pub cycle_count: usize,
}

impl CycleDetectionResult {
    pub fn none() -> Self {
        Self {
            has_cycles: false,
            cycles: Vec::new(),
            cycle_count: 0,
        }
    }

    pub fn with_cycles(cycles: Vec<Vec<DependencyId>>) -> Self {
        Self {
            has_cycles: !cycles.is_empty(),
            cycle_count: cycles.len(),
            cycles,
        }
    }
}

/// Result of topological sort operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalOrder {
    /// Ordered list of dependency IDs
    pub order: Vec<DependencyId>,
    /// Whether the sort was successful (false if cycles detected)
    pub success: bool,
    /// Any cycles that prevented sorting
    pub blocking_cycles: Vec<Vec<DependencyId>>,
}

impl TopologicalOrder {
    pub fn success(order: Vec<DependencyId>) -> Self {
        Self {
            order,
            success: true,
            blocking_cycles: Vec::new(),
        }
    }

    pub fn blocked(cycles: Vec<Vec<DependencyId>>) -> Self {
        Self {
            order: Vec::new(),
            success: false,
            blocking_cycles: cycles,
        }
    }
}

/// Validation result for dependency relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<ValidationError>,
    /// List of validation warnings
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn invalid(errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    pub fn with_warning(mut self, warning: ValidationWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    pub fn with_warnings(mut self, warnings: Vec<ValidationWarning>) -> Self {
        self.warnings.extend(warnings);
        self
    }
}

/// A validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error message
    pub message: String,
    /// Dependencies involved in the error
    pub involved: Vec<DependencyId>,
    /// Error type
    pub error_type: ValidationErrorType,
}

impl ValidationError {
    pub fn new(message: impl Into<String>, error_type: ValidationErrorType) -> Self {
        Self {
            message: message.into(),
            involved: Vec::new(),
            error_type,
        }
    }

    pub fn with_involved(mut self, deps: Vec<DependencyId>) -> Self {
        self.involved = deps;
        self
    }
}

/// Type of validation error
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationErrorType {
    /// Circular dependency detected
    CircularDependency,
    /// Missing required dependency
    MissingRequired,
    /// Conflicting dependencies
    Conflict,
    /// Self-dependency
    SelfDependency,
    /// Unknown dependency reference
    UnknownReference,
}

/// A validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Dependencies involved
    pub involved: Vec<DependencyId>,
    /// Warning type
    pub warning_type: ValidationWarningType,
}

impl ValidationWarning {
    pub fn new(message: impl Into<String>, warning_type: ValidationWarningType) -> Self {
        Self {
            message: message.into(),
            involved: Vec::new(),
            warning_type,
        }
    }

    pub fn with_involved(mut self, involved: Vec<DependencyId>) -> Self {
        self.involved = involved;
        self
    }
}

/// Type of validation warning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationWarningType {
    /// Optional dependency unavailable
    OptionalUnavailable,
    /// Weak dependency not satisfied
    WeakNotSatisfied,
    /// Deprecated dependency usage
    DeprecatedUsage,
}

/// Incremental update operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyUpdate {
    /// Add a new dependency
    AddDependency(Dependency),
    /// Remove an existing dependency
    RemoveDependency(DependencyId),
    /// Add an edge between dependencies
    AddEdge {
        from: DependencyId,
        to: DependencyId,
        edge: DependencyEdge,
    },
    /// Remove an edge
    RemoveEdge {
        from: DependencyId,
        to: DependencyId,
    },
    /// Update an edge
    UpdateEdge {
        from: DependencyId,
        to: DependencyId,
        edge: DependencyEdge,
    },
    /// Batch updates
    Batch(Vec<DependencyUpdate>),
}

/// Result of an incremental update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    /// Whether the update was successful
    pub success: bool,
    /// Dependencies that were affected
    pub affected: Vec<DependencyId>,
    /// New cycles introduced by the update
    pub new_cycles: Vec<Vec<DependencyId>>,
    /// Whether re-sorting is needed
    pub needs_resort: bool,
}

impl UpdateResult {
    pub fn success(affected: Vec<DependencyId>) -> Self {
        Self {
            success: true,
            affected,
            new_cycles: Vec::new(),
            needs_resort: true,
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self {
            success: false,
            affected: Vec::new(),
            new_cycles: Vec::new(),
            needs_resort: false,
        }
    }

    pub fn with_cycles(mut self, cycles: Vec<Vec<DependencyId>>) -> Self {
        let has_cycles = !cycles.is_empty();
        self.new_cycles = cycles;
        self.needs_resort = has_cycles;
        self
    }
}

/// Visual representation options for the dependency graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationFormat {
    /// DOT format (Graphviz)
    Dot,
    /// JSON format
    Json,
    /// Mermaid format
    Mermaid,
}

/// Main dependency resolution engine
pub struct DependencyResolver {
    /// The dependency graph
    graph: DiGraph<Dependency, DependencyEdge>,
    /// Map from DependencyId to NodeIndex
    id_to_node: HashMap<DependencyId, NodeIndex>,
    /// Map from NodeIndex to DependencyId
    node_to_id: HashMap<NodeIndex, DependencyId>,
    /// Cached topological order
    cached_order: Option<Vec<DependencyId>>,
    /// Whether the cache is valid
    cache_valid: bool,
}

impl DependencyResolver {
    /// Create a new empty dependency resolver
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            id_to_node: HashMap::new(),
            node_to_id: HashMap::new(),
            cached_order: None,
            cache_valid: true,
        }
    }

    /// Create a resolver with pre-allocated capacity
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(nodes, edges),
            id_to_node: HashMap::with_capacity(nodes),
            node_to_id: HashMap::with_capacity(nodes),
            cached_order: None,
            cache_valid: true,
        }
    }

    /// Add a dependency node to the graph
    pub fn add_dependency(&mut self, dependency: Dependency) -> Result<(), SolverError> {
        if self.id_to_node.contains_key(&dependency.id) {
            return Err(SolverError::ConfigurationError(format!(
                "Dependency ID {:?} already exists",
                dependency.id
            )));
        }

        let id = dependency.id;
        let node_idx = self.graph.add_node(dependency);
        self.id_to_node.insert(id, node_idx);
        self.node_to_id.insert(node_idx, id);
        self.invalidate_cache();

        Ok(())
    }

    /// Remove a dependency from the graph
    pub fn remove_dependency(&mut self, id: DependencyId) -> Result<(), SolverError> {
        let node_idx = *self.id_to_node.get(&id).ok_or_else(|| {
            SolverError::ConfigurationError(format!("Dependency ID {:?} not found", id))
        })?;

        self.graph.remove_node(node_idx);
        self.id_to_node.remove(&id);
        self.node_to_id.remove(&node_idx);
        self.invalidate_cache();

        Ok(())
    }

    /// Add a dependency edge (relationship) between two dependencies
    pub fn add_edge(
        &mut self,
        from: DependencyId,
        to: DependencyId,
        edge: DependencyEdge,
    ) -> Result<(), SolverError> {
        if from == to {
            return Err(SolverError::ConfigurationError(
                "Cannot add self-dependency edge".to_string(),
            ));
        }

        let from_node = *self.id_to_node.get(&from).ok_or_else(|| {
            SolverError::ConfigurationError(format!("Source dependency {:?} not found", from))
        })?;
        let to_node = *self.id_to_node.get(&to).ok_or_else(|| {
            SolverError::ConfigurationError(format!("Target dependency {:?} not found", to))
        })?;

        self.graph.add_edge(from_node, to_node, edge);
        self.invalidate_cache();

        Ok(())
    }

    /// Remove an edge between two dependencies
    pub fn remove_edge(&mut self, from: DependencyId, to: DependencyId) -> Result<(), SolverError> {
        let from_node = *self.id_to_node.get(&from).ok_or_else(|| {
            SolverError::ConfigurationError(format!("Source dependency {:?} not found", from))
        })?;
        let to_node = *self.id_to_node.get(&to).ok_or_else(|| {
            SolverError::ConfigurationError(format!("Target dependency {:?} not found", to))
        })?;

        if let Some(edge_idx) = self.graph.find_edge(from_node, to_node) {
            self.graph.remove_edge(edge_idx);
            self.invalidate_cache();
        }

        Ok(())
    }

    /// Detect circular dependencies in the graph
    pub fn detect_cycles(&mut self) -> CycleDetectionResult {
        let cycles = self.find_all_cycles();
        CycleDetectionResult::with_cycles(cycles)
    }

    /// Find all cycles in the dependency graph
    fn find_all_cycles(&mut self) -> Vec<Vec<DependencyId>> {
        let mut cycles = Vec::new();
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut rec_stack: Vec<NodeIndex> = Vec::new();
        let mut path: Vec<DependencyId> = Vec::new();

        for node in self.graph.node_indices() {
            if !visited.contains(&node) {
                if let Some(cycle) =
                    self.dfs_find_cycle(node, &mut visited, &mut rec_stack, &mut path)
                {
                    cycles.push(cycle);
                }
            }
        }

        cycles
    }

    /// DFS-based cycle detection that returns the actual cycle path
    fn dfs_find_cycle(
        &self,
        current: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        rec_stack: &mut Vec<NodeIndex>,
        path: &mut Vec<DependencyId>,
    ) -> Option<Vec<DependencyId>> {
        visited.insert(current);
        rec_stack.push(current);

        if let Some(&dep_id) = self.node_to_id.get(&current) {
            path.push(dep_id);
        }

        for neighbor in self.graph.neighbors(current) {
            if !visited.contains(&neighbor) {
                if let Some(cycle) = self.dfs_find_cycle(neighbor, visited, rec_stack, path) {
                    return Some(cycle);
                }
            } else {
                // Check if neighbor is in recursion stack (back edge found)
                if let Some(stack_idx) = rec_stack.iter().rposition(|&n| n == neighbor) {
                    // Found a cycle - extract the portion of path from when neighbor was first visited
                    let cycle_start = stack_idx;
                    let cycle: Vec<DependencyId> = rec_stack[cycle_start..]
                        .iter()
                        .filter_map(|node| self.node_to_id.get(node).copied())
                        .collect();

                    // Add the closing edge back to the start
                    if cycle.len() > 1 {
                        return Some(cycle);
                    }
                }
            }
        }

        rec_stack.pop();
        if let Some(&dep_id) = self.node_to_id.get(&current) {
            path.pop();
        }
        None
    }

    /// Perform topological sort on the dependency graph
    pub fn topological_sort(&mut self) -> TopologicalOrder {
        // Check for cycles first
        let cycle_result = self.detect_cycles();
        if cycle_result.has_cycles {
            return TopologicalOrder::blocked(cycle_result.cycles);
        }

        // Perform topological sort
        match toposort(&self.graph, None) {
            Ok(mut node_order) => {
                // Edges point from a dependent to its prerequisite, so petgraph's
                // source-first order must be reversed for executable dependency order.
                node_order.reverse();
                let id_order: Vec<DependencyId> = node_order
                    .iter()
                    .filter_map(|node| self.node_to_id.get(node).copied())
                    .collect();

                self.cached_order = Some(id_order.clone());
                self.cache_valid = true;

                TopologicalOrder::success(id_order)
            }
            Err(_) => {
                // Shouldn't happen if cycle detection passed, but handle it
                TopologicalOrder::blocked(cycle_result.cycles)
            }
        }
    }

    /// Validate the dependency graph
    pub fn validate(&mut self) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for cycles
        let cycle_result = self.detect_cycles();
        if cycle_result.has_cycles {
            for cycle in &cycle_result.cycles {
                errors.push(
                    ValidationError::new(
                        format!("Circular dependency detected: {:?}", cycle),
                        ValidationErrorType::CircularDependency,
                    )
                    .with_involved(cycle.clone()),
                );
            }
        }

        // Check for self-dependencies
        for node in self.graph.node_indices() {
            if let Some(edge) = self.graph.find_edge(node, node) {
                let dep_id = self.node_to_id.get(&node).copied();
                errors.push(
                    ValidationError::new(
                        "Self-dependency detected",
                        ValidationErrorType::SelfDependency,
                    )
                    .with_involved(dep_id.into_iter().collect()),
                );
            }
        }

        // Check for isolated nodes (potential missing required dependencies)
        for node in self.graph.node_indices() {
            let has_incoming = self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .next()
                .is_some();
            let has_outgoing = self
                .graph
                .edges_directed(node, petgraph::Direction::Outgoing)
                .next()
                .is_some();

            if !has_incoming && !has_outgoing {
                if let Some(&dep_id) = self.node_to_id.get(&node) {
                    warnings.push(
                        ValidationWarning::new(
                            format!("Isolated dependency: {:?}", dep_id),
                            ValidationWarningType::OptionalUnavailable,
                        )
                        .with_involved(vec![dep_id]),
                    );
                }
            }
        }

        if errors.is_empty() {
            ValidationResult::valid().with_warnings(warnings)
        } else {
            ValidationResult::invalid(errors)
        }
    }

    /// Apply incremental updates to the graph
    pub fn apply_updates(&mut self, updates: Vec<DependencyUpdate>) -> UpdateResult {
        let mut affected = Vec::new();
        let mut all_success = true;

        for update in updates {
            match update {
                DependencyUpdate::AddDependency(dep) => {
                    if let Err(e) = self.add_dependency(dep.clone()) {
                        all_success = false;
                        log::error!("Failed to add dependency {:?}: {}", dep.id, e);
                    } else {
                        affected.push(dep.id);
                    }
                }
                DependencyUpdate::RemoveDependency(id) => {
                    if let Err(e) = self.remove_dependency(id) {
                        all_success = false;
                        log::error!("Failed to remove dependency {:?}: {}", id, e);
                    } else {
                        affected.push(id);
                    }
                }
                DependencyUpdate::AddEdge { from, to, edge } => {
                    if let Err(e) = self.add_edge(from, to, edge) {
                        all_success = false;
                        log::error!("Failed to add edge {:?} -> {:?}: {}", from, to, e);
                    } else {
                        affected.extend([from, to]);
                    }
                }
                DependencyUpdate::RemoveEdge { from, to } => {
                    if let Err(e) = self.remove_edge(from, to) {
                        all_success = false;
                        log::error!("Failed to remove edge {:?} -> {:?}: {}", from, to, e);
                    } else {
                        affected.extend([from, to]);
                    }
                }
                DependencyUpdate::UpdateEdge { from, to, edge } => {
                    self.remove_edge(from, to).ok();
                    if let Err(e) = self.add_edge(from, to, edge) {
                        all_success = false;
                        log::error!("Failed to update edge {:?} -> {:?}: {}", from, to, e);
                    } else {
                        affected.extend([from, to]);
                    }
                }
                DependencyUpdate::Batch(inner_updates) => {
                    let batch_result = self.apply_updates(inner_updates);
                    affected.extend(batch_result.affected);
                    if !batch_result.success {
                        all_success = false;
                    }
                }
            }
        }

        // Check for new cycles
        let new_cycles = self.detect_cycles().cycles;
        let needs_resort = !affected.is_empty();

        UpdateResult {
            success: all_success,
            affected,
            new_cycles,
            needs_resort,
        }
    }

    /// Get the current topological order (cached if available)
    pub fn get_order(&mut self) -> Vec<DependencyId> {
        if !self.cache_valid || self.cached_order.is_none() {
            let result = self.topological_sort();
            if result.success {
                self.cached_order = Some(result.order);
            }
        }

        self.cached_order.clone().unwrap_or_default()
    }

    /// Check if a specific dependency exists
    pub fn contains(&self, id: DependencyId) -> bool {
        self.id_to_node.contains_key(&id)
    }

    /// Get a dependency by ID
    pub fn get(&self, id: DependencyId) -> Option<&Dependency> {
        self.id_to_node
            .get(&id)
            .and_then(|node| self.graph.node_weight(*node))
    }

    /// Get dependencies that depend on the given one (reverse dependencies)
    pub fn get_reverse_dependencies(&self, id: DependencyId) -> Vec<DependencyId> {
        let node_idx = match self.id_to_node.get(&id) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        self.graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .filter_map(|node| self.node_to_id.get(&node).copied())
            .collect()
    }

    /// Get dependencies that the given one depends on
    pub fn get_forward_dependencies(&self, id: DependencyId) -> Vec<DependencyId> {
        let node_idx = match self.id_to_node.get(&id) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        self.graph
            .neighbors(node_idx)
            .filter_map(|node| self.node_to_id.get(&node).copied())
            .collect()
    }

    /// Generate a visual representation of the dependency graph
    pub fn visualize(&self, format: VisualizationFormat) -> String {
        match format {
            VisualizationFormat::Dot => {
                use std::fmt::Write;
                let mut output = String::new();
                if let Err(_) = write!(
                    output,
                    "{}",
                    Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
                ) {
                    String::new()
                } else {
                    output
                }
            }
            VisualizationFormat::Json => {
                serde_json::to_string_pretty(&self.graph_to_json()).unwrap_or_default()
            }
            VisualizationFormat::Mermaid => self.to_mermaid(),
        }
    }

    /// Convert the graph to Mermaid flowchart format
    fn to_mermaid(&self) -> String {
        let mut lines = vec!["flowchart TD".to_string()];

        for node in self.graph.node_indices() {
            if let Some(dep) = self.graph.node_weight(node) {
                lines.push(format!("    {}[\"{}\"]", dep.id.raw(), dep.name));
            }
        }

        for edge in self.graph.edge_indices() {
            if let Some((source, target)) = self.graph.edge_endpoints(edge) {
                if let (Some(source_id), Some(target_id)) =
                    (self.node_to_id.get(&source), self.node_to_id.get(&target))
                {
                    if let Some(edge_weight) = self.graph.edge_weight(edge) {
                        let label = match edge_weight.edge_type {
                            DependencyType::Required => "|required|",
                            DependencyType::Optional => "|optional|",
                            DependencyType::Weak => "|weak|",
                            DependencyType::Conflict => "-x|conflict|-x",
                        };
                        lines.push(format!(
                            "    {} {} {}",
                            source_id.raw(),
                            label,
                            target_id.raw()
                        ));
                    }
                }
            }
        }

        lines.join("\n")
    }

    /// Convert the graph to a JSON-serializable structure
    fn graph_to_json(&self) -> serde_json::Value {
        let nodes: Vec<serde_json::Value> = self
            .graph
            .node_indices()
            .filter_map(|node| {
                self.graph
                    .node_weight(node)
                    .and_then(|dep| serde_json::to_value(dep).ok())
            })
            .collect();

        let edges: Vec<serde_json::Value> = self
            .graph
            .edge_indices()
            .filter_map(|edge| {
                let (source, target) = self.graph.edge_endpoints(edge)?;
                let source_id = self.node_to_id.get(&source)?.raw();
                let target_id = self.node_to_id.get(&target)?.raw();
                let edge_weight = self.graph.edge_weight(edge)?;

                Some(serde_json::json!({
                    "from": source_id,
                    "to": target_id,
                    "type": edge_weight.edge_type,
                    "priority": edge_weight.priority,
                    "active": edge_weight.active,
                }))
            })
            .collect();

        serde_json::json!({
            "nodes": nodes,
            "edges": edges,
            "node_count": self.graph.node_count(),
            "edge_count": self.graph.edge_count(),
        })
    }

    /// Get statistics about the dependency graph
    pub fn stats(&self) -> DependencyStats {
        DependencyStats {
            total_dependencies: self.graph.node_count(),
            total_edges: self.graph.edge_count(),
            isolated_nodes: self.count_isolated(),
            max_depth: self.calculate_max_depth(),
            avg_branching_factor: self.calculate_avg_branching(),
        }
    }

    /// Count isolated nodes (no edges)
    fn count_isolated(&self) -> usize {
        self.graph
            .node_indices()
            .filter(|&node| {
                self.graph.edges(node).next().is_none()
                    && self
                        .graph
                        .edges_directed(node, petgraph::Direction::Incoming)
                        .next()
                        .is_none()
            })
            .count()
    }

    /// Calculate the maximum depth of the dependency tree
    fn calculate_max_depth(&self) -> usize {
        let mut max_depth = 0;

        // Find root nodes (no incoming edges)
        let roots: Vec<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|&node| {
                self.graph
                    .edges_directed(node, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .collect();

        // Calculate depth from each root
        for root in roots {
            let depth = self.depth_from_node(root);
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    /// Calculate depth starting from a specific node
    fn depth_from_node(&self, node: NodeIndex) -> usize {
        let mut max_child_depth = 0;

        for child in self.graph.neighbors(node) {
            let child_depth = self.depth_from_node(child);
            max_child_depth = max_child_depth.max(child_depth);
        }

        max_child_depth + 1
    }

    /// Calculate average branching factor
    fn calculate_avg_branching(&self) -> f64 {
        if self.graph.node_count() == 0 {
            return 0.0;
        }

        let total_out_edges: usize = self
            .graph
            .node_indices()
            .map(|node| self.graph.neighbors(node).count())
            .sum();

        total_out_edges as f64 / self.graph.node_count() as f64
    }

    /// Invalidate the cached topological order
    fn invalidate_cache(&mut self) {
        self.cache_valid = false;
    }

    /// Get the number of dependencies
    pub fn len(&self) -> usize {
        self.graph.node_count()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.graph.node_count() == 0
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyStats {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Total number of edges (relationships)
    pub total_edges: usize,
    /// Number of isolated nodes
    pub isolated_nodes: usize,
    /// Maximum depth of the dependency tree
    pub max_depth: usize,
    /// Average branching factor
    pub avg_branching_factor: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_resolver() -> DependencyResolver {
        let mut resolver = DependencyResolver::with_capacity(10, 20);

        // Add test dependencies
        let deps = vec![
            Dependency::new(DependencyId::new(1), "base", DependencyType::Required),
            Dependency::new(DependencyId::new(2), "auth", DependencyType::Required),
            Dependency::new(DependencyId::new(3), "database", DependencyType::Required),
            Dependency::new(DependencyId::new(4), "api", DependencyType::Optional),
            Dependency::new(DependencyId::new(5), "cache", DependencyType::Weak),
        ];

        for dep in deps {
            resolver.add_dependency(dep).unwrap();
        }

        // Add edges: api -> auth -> database, cache -> database
        resolver
            .add_edge(
                DependencyId::new(4), // api
                DependencyId::new(2), // auth
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(2), // auth
                DependencyId::new(3), // database
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(5), // cache
                DependencyId::new(3), // database
                DependencyEdge::new(DependencyType::Weak),
            )
            .unwrap();

        resolver
    }

    #[test]
    fn test_dependency_creation() {
        let dep = Dependency::new(DependencyId::new(1), "test_dep", DependencyType::Required);

        assert_eq!(dep.id, DependencyId::new(1));
        assert_eq!(dep.name, "test_dep");
        assert_eq!(dep.dep_type, DependencyType::Required);
    }

    #[test]
    fn test_resolver_creation() {
        let resolver = DependencyResolver::new();
        assert!(resolver.is_empty());
        assert_eq!(resolver.len(), 0);
    }

    #[test]
    fn test_add_dependency() {
        let mut resolver = DependencyResolver::new();
        let dep = Dependency::new(DependencyId::new(1), "test", DependencyType::Required);

        assert!(resolver.add_dependency(dep).is_ok());
        assert_eq!(resolver.len(), 1);
        assert!(resolver.contains(DependencyId::new(1)));
    }

    #[test]
    fn test_duplicate_dependency() {
        let mut resolver = DependencyResolver::new();
        let dep = Dependency::new(DependencyId::new(1), "test", DependencyType::Required);

        resolver.add_dependency(dep.clone()).unwrap();
        assert!(resolver.add_dependency(dep).is_err());
    }

    #[test]
    fn test_add_edge() {
        let mut resolver = DependencyResolver::with_capacity(2, 2);

        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(2),
                "b",
                DependencyType::Required,
            ))
            .unwrap();

        assert!(resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Required),
            )
            .is_ok());
    }

    #[test]
    fn test_self_dependency_rejected() {
        let mut resolver = DependencyResolver::with_capacity(1, 1);

        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            ))
            .unwrap();

        assert!(resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(1),
                DependencyEdge::new(DependencyType::Required),
            )
            .is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let mut resolver = DependencyResolver::with_capacity(3, 3);

        // Create a cycle: a -> b -> c -> a
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(2),
                "b",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(3),
                "c",
                DependencyType::Required,
            ))
            .unwrap();

        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(2),
                DependencyId::new(3),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(3),
                DependencyId::new(1),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();

        let result = resolver.detect_cycles();
        assert!(result.has_cycles);
        assert_eq!(result.cycle_count, 1);
        assert!(!result.cycles.is_empty());
    }

    #[test]
    fn test_topological_sort_success() {
        let mut resolver = create_test_resolver();

        let result = resolver.topological_sort();
        assert!(result.success);
        assert!(!result.order.is_empty());

        // Verify order respects dependencies
        // api depends on auth, auth depends on database
        let order = &result.order;
        let api_pos = order.iter().position(|&id| id == DependencyId::new(4));
        let auth_pos = order.iter().position(|&id| id == DependencyId::new(2));
        let db_pos = order.iter().position(|&id| id == DependencyId::new(3));

        // api should come after auth, auth after database
        if let (Some(api), Some(auth), Some(db)) = (api_pos, auth_pos, db_pos) {
            assert!(api > auth, "api should come after auth");
            assert!(auth > db, "auth should come after database");
        }
    }

    #[test]
    fn test_topological_sort_with_cycles() {
        let mut resolver = DependencyResolver::with_capacity(3, 3);

        // Create a cycle
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(2),
                "b",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(3),
                "c",
                DependencyType::Required,
            ))
            .unwrap();

        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(2),
                DependencyId::new(3),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(3),
                DependencyId::new(1),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();

        let result = resolver.topological_sort();
        assert!(!result.success);
        assert!(!result.blocking_cycles.is_empty());
    }

    #[test]
    fn test_validation_valid_graph() {
        let mut resolver = create_test_resolver();
        let result = resolver.validate();

        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_with_cycle() {
        let mut resolver = DependencyResolver::with_capacity(3, 3);

        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(2),
                "b",
                DependencyType::Required,
            ))
            .unwrap();

        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(2),
                DependencyId::new(1),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();

        let result = resolver.validate();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_incremental_updates() {
        let mut resolver = create_test_resolver();

        let updates = vec![
            DependencyUpdate::AddDependency(Dependency::new(
                DependencyId::new(6),
                "logging",
                DependencyType::Optional,
            )),
            DependencyUpdate::AddEdge {
                from: DependencyId::new(4),
                to: DependencyId::new(6),
                edge: DependencyEdge::new(DependencyType::Optional),
            },
        ];

        let result = resolver.apply_updates(updates);
        assert!(result.success);
        assert!(!result.affected.is_empty());
        assert_eq!(resolver.len(), 6);
    }

    #[test]
    fn test_get_forward_dependencies() {
        let resolver = create_test_resolver();

        let deps = resolver.get_forward_dependencies(DependencyId::new(2)); // auth
        assert_eq!(deps.len(), 1);
        assert!(deps.contains(&DependencyId::new(3))); // database
    }

    #[test]
    fn test_get_reverse_dependencies() {
        let resolver = create_test_resolver();

        let deps = resolver.get_reverse_dependencies(DependencyId::new(3)); // database
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&DependencyId::new(2))); // auth
        assert!(deps.contains(&DependencyId::new(5))); // cache
    }

    #[test]
    fn test_visualize_dot() {
        let resolver = create_test_resolver();
        let dot = resolver.visualize(VisualizationFormat::Dot);

        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_visualize_mermaid() {
        let resolver = create_test_resolver();
        let mermaid = resolver.visualize(VisualizationFormat::Mermaid);

        assert!(mermaid.contains("flowchart TD"));
        assert!(mermaid.contains("auth"));
        assert!(mermaid.contains("database"));
    }

    #[test]
    fn test_stats() {
        let resolver = create_test_resolver();
        let stats = resolver.stats();

        assert_eq!(stats.total_dependencies, 5);
        assert_eq!(stats.total_edges, 3);
        assert!(stats.max_depth > 0);
    }

    #[test]
    fn test_remove_dependency() {
        let mut resolver = create_test_resolver();

        resolver.remove_dependency(DependencyId::new(5)).unwrap();
        assert_eq!(resolver.len(), 4);
        assert!(!resolver.contains(DependencyId::new(5)));
    }

    #[test]
    fn test_remove_edge() {
        let mut resolver = create_test_resolver();

        resolver
            .remove_edge(DependencyId::new(5), DependencyId::new(3))
            .unwrap();
        let deps = resolver.get_reverse_dependencies(DependencyId::new(3));
        assert_eq!(deps.len(), 1);
    }

    #[test]
    fn test_batch_updates() {
        let mut resolver = DependencyResolver::with_capacity(5, 10);

        let batch = vec![
            DependencyUpdate::AddDependency(Dependency::new(
                DependencyId::new(1),
                "a",
                DependencyType::Required,
            )),
            DependencyUpdate::AddDependency(Dependency::new(
                DependencyId::new(2),
                "b",
                DependencyType::Required,
            )),
            DependencyUpdate::AddEdge {
                from: DependencyId::new(1),
                to: DependencyId::new(2),
                edge: DependencyEdge::new(DependencyType::Required),
            },
        ];

        let result = resolver.apply_updates(batch);
        assert!(result.success);
        assert_eq!(resolver.len(), 2);
    }

    #[test]
    fn test_cache_invalidation() {
        let mut resolver = create_test_resolver();

        // Get initial order
        let order1 = resolver.get_order();
        assert!(!order1.is_empty());

        // Add new dependency
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(10),
                "new",
                DependencyType::Required,
            ))
            .unwrap();

        // Cache should be invalidated, new order should include new dependency
        let order2 = resolver.get_order();
        assert_eq!(order2.len(), order1.len() + 1);
    }

    #[test]
    fn test_dependency_id_display() {
        let id = DependencyId::new(42);
        assert_eq!(format!("{}", id), "Dep(42)");
    }

    #[test]
    fn test_dependency_type_display() {
        assert_eq!(format!("{}", DependencyType::Required), "required");
        assert_eq!(format!("{}", DependencyType::Optional), "optional");
        assert_eq!(format!("{}", DependencyType::Weak), "weak");
        assert_eq!(format!("{}", DependencyType::Conflict), "conflict");
    }

    #[test]
    fn test_edge_with_priority() {
        let edge = DependencyEdge::new(DependencyType::Required).with_priority(10);

        assert_eq!(edge.priority, 10);
    }

    #[test]
    fn test_visualization_format_json() {
        let resolver = create_test_resolver();
        let json = resolver.visualize(VisualizationFormat::Json);

        assert!(json.contains("nodes"));
        assert!(json.contains("edges"));
    }

    #[test]
    fn test_complex_topological_order() {
        let mut resolver = DependencyResolver::with_capacity(10, 20);

        // Create a diamond dependency graph. Edges point from a dependent
        // to its prerequisite, so e is executable first and a last.
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        //     |
        //     e
        for id in 1..=5 {
            resolver
                .add_dependency(Dependency::new(
                    DependencyId::new(id),
                    format!("dep_{}", id),
                    DependencyType::Required,
                ))
                .unwrap();
        }

        // a -> b, a -> c
        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(3),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        // b -> d, c -> d
        resolver
            .add_edge(
                DependencyId::new(2),
                DependencyId::new(4),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(3),
                DependencyId::new(4),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();
        // d -> e
        resolver
            .add_edge(
                DependencyId::new(4),
                DependencyId::new(5),
                DependencyEdge::new(DependencyType::Required),
            )
            .unwrap();

        let result = resolver.topological_sort();
        assert!(result.success);

        let order = result.order;
        let pos = |id| order.iter().position(|&x| x == DependencyId::new(id));

        // Verify ordering constraints
        let (pos_a, pos_b, pos_c, pos_d, pos_e) = (pos(1), pos(2), pos(3), pos(4), pos(5));

        if let (Some(pa), Some(pb), Some(pc), Some(pd), Some(pe)) =
            (pos_a, pos_b, pos_c, pos_d, pos_e)
        {
            assert!(pe < pd, "e should come before d");
            assert!(pd < pb, "d should come before b");
            assert!(pd < pc, "d should come before c");
            assert!(pb < pa, "b should come before a");
            assert!(pc < pa, "c should come before a");
        }
    }

    #[test]
    fn test_weak_dependencies() {
        let mut resolver = DependencyResolver::with_capacity(3, 2);

        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(1),
                "main",
                DependencyType::Required,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(2),
                "optional",
                DependencyType::Optional,
            ))
            .unwrap();
        resolver
            .add_dependency(Dependency::new(
                DependencyId::new(3),
                "weak",
                DependencyType::Weak,
            ))
            .unwrap();

        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(2),
                DependencyEdge::new(DependencyType::Optional),
            )
            .unwrap();
        resolver
            .add_edge(
                DependencyId::new(1),
                DependencyId::new(3),
                DependencyEdge::new(DependencyType::Weak),
            )
            .unwrap();

        let validation = resolver.validate();
        assert!(validation.is_valid);
    }
}
