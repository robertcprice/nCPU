//! Goal tree management for hierarchical task planning.
//!
//! This module provides a thread-safe goal tree structure with:
//! - Parent-child relationships for hierarchical goals
//! - Dependency tracking between goals
//! - Priority-based goal selection
//! - Goal completion tracking
//! - Topological sorting for execution order

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Unique identifier for a goal
pub type GoalId = u64;

/// Priority level for goal selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Status of a goal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Blocked,
}

/// A single goal in the goal tree
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: GoalId,
    pub name: String,
    pub description: Option<String>,
    pub priority: Priority,
    pub status: GoalStatus,
    pub parent_id: Option<GoalId>,
    pub children: Vec<GoalId>,
    pub dependencies: HashSet<GoalId>,
    pub dependents: HashSet<GoalId>,
    pub metadata: HashMap<String, String>,
}

impl Goal {
    /// Create a new goal with the given ID and name
    pub fn new(id: GoalId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: None,
            priority: Priority::Medium,
            status: GoalStatus::Pending,
            parent_id: None,
            children: Vec::new(),
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            metadata: HashMap::new(),
        }
    }

    /// Builder pattern for description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder pattern for priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Builder pattern for parent
    pub fn with_parent(mut self, parent_id: GoalId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Add a dependency to this goal
    pub fn add_dependency(&mut self, dep_id: GoalId) {
        self.dependencies.insert(dep_id);
    }

    /// Remove a dependency from this goal
    pub fn remove_dependency(&mut self, dep_id: &GoalId) {
        self.dependencies.remove(dep_id);
    }

    /// Add a child to this goal
    pub fn add_child(&mut self, child_id: GoalId) {
        self.children.push(child_id);
    }

    /// Add a dependent (goal that depends on this one)
    pub fn add_dependent(&mut self, dependent_id: GoalId) {
        self.dependents.insert(dependent_id);
    }

    /// Remove a dependent
    pub fn remove_dependent(&mut self, dependent_id: &GoalId) {
        self.dependents.remove(dependent_id);
    }

    /// Check if this goal is ready to execute (all dependencies completed)
    pub fn is_ready(&self, tree: &GoalTree) -> bool {
        self.status == GoalStatus::Pending
            && self
                .dependencies
                .iter()
                .all(|dep_id| match tree.get_goal(*dep_id) {
                    Some(goal) => goal.status == GoalStatus::Completed,
                    None => false,
                })
    }

    /// Check if this goal is blocked (any dependency failed)
    pub fn is_blocked(&self, tree: &GoalTree) -> bool {
        self.dependencies
            .iter()
            .any(|dep_id| match tree.get_goal(*dep_id) {
                Some(goal) => goal.status == GoalStatus::Failed,
                None => false,
            })
    }

    /// Check if this goal is a descendant of the given goal ID
    pub fn is_descendant_of(&self, tree: &GoalTree, ancestor_id: GoalId) -> bool {
        let mut current_id = self.parent_id;
        while let Some(parent_id) = current_id {
            if parent_id == ancestor_id {
                return true;
            }
            current_id = tree.get_goal(parent_id).and_then(|g| g.parent_id);
        }
        false
    }

    /// Check if this goal is an ancestor of the given goal ID
    pub fn is_ancestor_of(&self, tree: &GoalTree, descendant_id: GoalId) -> bool {
        if let Some(descendant) = tree.get_goal(descendant_id) {
            descendant.is_descendant_of(tree, self.id)
        } else {
            false
        }
    }
}

/// Thread-safe goal tree for hierarchical task planning
#[derive(Debug, Clone)]
pub struct GoalTree {
    goals: Arc<RwLock<HashMap<GoalId, Goal>>>,
    next_id: Arc<RwLock<GoalId>>,
}

impl Default for GoalTree {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalTree {
    /// Create a new empty goal tree
    pub fn new() -> Self {
        Self {
            goals: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Add a new goal to the tree
    pub fn add_goal(&self, goal: Goal) -> Result<GoalId, String> {
        let mut goals = self.goals.write().map_err(|e| e.to_string())?;

        // Check if goal ID already exists
        if goals.contains_key(&goal.id) {
            return Err(format!("Goal ID {} already exists", goal.id));
        }

        let goal_id = goal.id;

        // Update parent's children if this goal has a parent
        if let Some(parent_id) = goal.parent_id {
            if let Some(parent) = goals.get_mut(&parent_id) {
                parent.add_child(goal_id);
            }
        }

        // Update dependencies' dependents
        for &dep_id in &goal.dependencies {
            if let Some(dep_goal) = goals.get_mut(&dep_id) {
                dep_goal.add_dependent(goal_id);
            }
        }

        goals.insert(goal_id, goal);
        Ok(goal_id)
    }

    /// Create a new goal with an auto-generated ID
    pub fn create_goal(&self, name: impl Into<String>) -> Result<GoalId, String> {
        let mut next_id = self.next_id.write().map_err(|e| e.to_string())?;
        let id = *next_id;
        *next_id = id + 1;
        drop(next_id);

        let goal = Goal::new(id, name);
        self.add_goal(goal)
    }

    /// Get a goal by ID
    pub fn get_goal(&self, id: GoalId) -> Option<Goal> {
        let goals = self.goals.read().ok()?;
        goals.get(&id).cloned()
    }

    /// Get all goals
    pub fn get_all_goals(&self) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        goals.values().cloned().collect()
    }

    /// Get goals by status
    pub fn get_goals_by_status(&self, status: GoalStatus) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        goals
            .values()
            .filter(|g| g.status == status)
            .cloned()
            .collect()
    }

    /// Get children of a goal
    pub fn get_children(&self, parent_id: GoalId) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        if let Some(parent) = goals.get(&parent_id) {
            parent
                .children
                .iter()
                .filter_map(|child_id| goals.get(child_id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Update goal status
    pub fn update_status(&self, id: GoalId, status: GoalStatus) -> Result<(), String> {
        let mut goals = self.goals.write().map_err(|e| e.to_string())?;

        if let Some(goal) = goals.get_mut(&id) {
            goal.status = status;
            Ok(())
        } else {
            Err(format!("Goal {} not found", id))
        }
    }

    /// Add a dependency between goals
    pub fn add_dependency(&self, goal_id: GoalId, dep_id: GoalId) -> Result<(), String> {
        if goal_id == dep_id {
            return Err("Goal cannot depend on itself".to_string());
        }

        let mut goals = self.goals.write().map_err(|e| e.to_string())?;

        // Check for circular dependencies
        if self.would_create_cycle(&goals, goal_id, dep_id)? {
            return Err("Would create circular dependency".to_string());
        }

        if let Some(goal) = goals.get_mut(&goal_id) {
            goal.add_dependency(dep_id);
        }

        if let Some(dep_goal) = goals.get_mut(&dep_id) {
            dep_goal.add_dependent(goal_id);
        }

        Ok(())
    }

    /// Remove a dependency between goals
    pub fn remove_dependency(&self, goal_id: GoalId, dep_id: GoalId) -> Result<(), String> {
        let mut goals = self.goals.write().map_err(|e| e.to_string())?;

        if let Some(goal) = goals.get_mut(&goal_id) {
            goal.remove_dependency(&dep_id);
        }

        if let Some(dep_goal) = goals.get_mut(&dep_id) {
            dep_goal.remove_dependent(&goal_id);
        }

        Ok(())
    }

    /// Set parent for a goal (creates parent-child relationship)
    pub fn set_parent(&self, child_id: GoalId, parent_id: GoalId) -> Result<(), String> {
        if child_id == parent_id {
            return Err("Goal cannot be its own parent".to_string());
        }

        let mut goals = self.goals.write().map_err(|e| e.to_string())?;

        // Check for circular parent relationship
        if self.would_create_parent_cycle(&goals, child_id, parent_id)? {
            return Err("Would create circular parent relationship".to_string());
        }

        // Remove from old parent if exists
        let old_parent_id = goals.get(&child_id).and_then(|g| g.parent_id);
        if let Some(old_parent) = old_parent_id {
            if let Some(old_parent_goal) = goals.get_mut(&old_parent) {
                old_parent_goal.children.retain(|&id| id != child_id);
            }
        }

        // Add to new parent
        if let Some(child) = goals.get_mut(&child_id) {
            child.parent_id = Some(parent_id);
        }

        if let Some(parent) = goals.get_mut(&parent_id) {
            parent.add_child(child_id);
        }

        Ok(())
    }

    /// Get the next ready goal based on priority
    pub fn get_next_ready_goal(&self) -> Option<Goal> {
        let goals = self.goals.read().ok()?;
        let mut ready_goals: Vec<_> = goals.values().filter(|g| g.is_ready(self)).collect();

        // Sort by priority (descending), then by ID (ascending for determinism)
        ready_goals.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));

        ready_goals.first().cloned().cloned()
    }

    /// Get all ready goals sorted by priority
    pub fn get_ready_goals(&self) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        let mut ready_goals: Vec<_> = goals
            .values()
            .filter(|g| g.is_ready(self))
            .cloned()
            .collect();

        ready_goals.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));

        ready_goals
    }

    /// Get goals in topological order for execution
    pub fn topological_sort(&self) -> Result<Vec<Goal>, String> {
        let goals = self.goals.read().map_err(|e| e.to_string())?;
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        let goal_ids: Vec<_> = goals.keys().copied().collect();

        for &id in &goal_ids {
            if !visited.contains(&id) {
                self.visit(id, &goals, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        Ok(result)
    }

    /// Helper for topological sort using DFS
    fn visit(
        &self,
        id: GoalId,
        goals: &HashMap<GoalId, Goal>,
        visited: &mut HashSet<GoalId>,
        temp_visited: &mut HashSet<GoalId>,
        result: &mut Vec<Goal>,
    ) -> Result<(), String> {
        if temp_visited.contains(&id) {
            return Err("Circular dependency detected".to_string());
        }

        if visited.contains(&id) {
            return Ok(());
        }

        temp_visited.insert(id);

        if let Some(goal) = goals.get(&id) {
            for &dep_id in &goal.dependencies {
                self.visit(dep_id, goals, visited, temp_visited, result)?;
            }
            result.push(goal.clone());
        }

        temp_visited.remove(&id);
        visited.insert(id);

        Ok(())
    }

    /// Check if adding a dependency would create a cycle
    fn would_create_cycle(
        &self,
        goals: &HashMap<GoalId, Goal>,
        from: GoalId,
        to: GoalId,
    ) -> Result<bool, String> {
        let mut visited = HashSet::new();
        Ok(self.has_path_from(to, from, goals, &mut visited)?)
    }

    /// Check if there's a path from `from` to `to` following dependencies
    fn has_path_from(
        &self,
        from: GoalId,
        to: GoalId,
        goals: &HashMap<GoalId, Goal>,
        visited: &mut HashSet<GoalId>,
    ) -> Result<bool, String> {
        if from == to {
            return Ok(true);
        }

        if visited.contains(&from) {
            return Ok(false);
        }

        visited.insert(from);

        if let Some(goal) = goals.get(&from) {
            for &dep_id in &goal.dependencies {
                if self.has_path_from(dep_id, to, goals, visited)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Check if setting a parent would create a cycle
    fn would_create_parent_cycle(
        &self,
        goals: &HashMap<GoalId, Goal>,
        child: GoalId,
        parent: GoalId,
    ) -> Result<bool, String> {
        let mut visited = HashSet::new();
        Ok(self.has_parent_path_from(child, parent, goals, &mut visited)?)
    }

    /// Check if there's a parent path from `from` to `to`
    fn has_parent_path_from(
        &self,
        from: GoalId,
        to: GoalId,
        goals: &HashMap<GoalId, Goal>,
        visited: &mut HashSet<GoalId>,
    ) -> Result<bool, String> {
        if from == to {
            return Ok(true);
        }

        if visited.contains(&from) {
            return Ok(false);
        }

        visited.insert(from);

        if let Some(goal) = goals.get(&from) {
            // Check children
            for &child_id in &goal.children {
                if self.has_parent_path_from(child_id, to, goals, visited)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get root goals (goals with no parent)
    pub fn get_root_goals(&self) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        let mut roots: Vec<_> = goals
            .values()
            .filter(|g| g.parent_id.is_none())
            .cloned()
            .collect();
        roots.sort_by_key(|goal| goal.id);
        roots
    }

    /// Get leaf goals (goals with no children)
    pub fn get_leaf_goals(&self) -> Vec<Goal> {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        goals
            .values()
            .filter(|g| g.children.is_empty())
            .cloned()
            .collect()
    }

    /// Get the subtree rooted at the given goal ID
    pub fn get_subtree(&self, root_id: GoalId) -> Vec<Goal> {
        let mut subtree = Vec::new();
        let mut to_visit = vec![root_id];
        let mut visited = HashSet::new();

        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();

        while let Some(id) = to_visit.pop() {
            if visited.contains(&id) {
                continue;
            }

            visited.insert(id);

            if let Some(goal) = goals.get(&id) {
                subtree.push(goal.clone());
                to_visit.extend(goal.children.iter().copied());
            }
        }

        subtree
    }

    /// Clear all goals from the tree
    pub fn clear(&self) -> Result<(), String> {
        let mut goals = self.goals.write().map_err(|e| e.to_string())?;
        goals.clear();
        Ok(())
    }

    /// Get the total number of goals
    pub fn len(&self) -> usize {
        self.goals.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Check if the tree is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clone the goal tree (creates a new independent tree)
    pub fn clone_tree(&self) -> Self {
        let goals = self.goals.read().map_err(|e| e.to_string()).unwrap();
        Self {
            goals: Arc::new(RwLock::new(goals.clone())),
            next_id: Arc::clone(&self.next_id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tree() -> GoalTree {
        let tree = GoalTree::new();

        // Create goals
        let root1 = tree.create_goal("Root 1").unwrap();
        let root2 = tree.create_goal("Root 2").unwrap();
        let child1 = tree.create_goal("Child 1").unwrap();
        let child2 = tree.create_goal("Child 2").unwrap();
        let grandchild1 = tree.create_goal("Grandchild 1").unwrap();

        // Build hierarchy
        tree.set_parent(child1, root1).unwrap();
        tree.set_parent(child2, root1).unwrap();
        tree.set_parent(grandchild1, child1).unwrap();

        // Add dependencies
        tree.add_dependency(child2, child1).unwrap();

        // Update priorities
        {
            let mut goals = tree.goals.write().unwrap();
            if let Some(goal) = goals.get_mut(&root2) {
                goal.priority = Priority::High;
            }
        }

        tree
    }

    #[test]
    fn test_goal_creation() {
        let tree = GoalTree::new();
        let id = tree.create_goal("Test Goal").unwrap();

        let goal = tree.get_goal(id).unwrap();
        assert_eq!(goal.name, "Test Goal");
        assert_eq!(goal.status, GoalStatus::Pending);
        assert_eq!(goal.priority, Priority::Medium);
    }

    #[test]
    fn test_goal_with_builder() {
        let tree = GoalTree::new();
        let goal = Goal::new(1, "Builder Test")
            .with_description("Test description")
            .with_priority(Priority::High);

        tree.add_goal(goal).unwrap();

        let retrieved = tree.get_goal(1).unwrap();
        assert_eq!(retrieved.description, Some("Test description".to_string()));
        assert_eq!(retrieved.priority, Priority::High);
    }

    #[test]
    fn test_parent_child_relationships() {
        let tree = create_test_tree();

        // Check root goals
        let roots = tree.get_root_goals();
        assert_eq!(roots.len(), 2);

        // Check children
        let children = tree.get_children(roots[0].id);
        assert_eq!(children.len(), 2);

        // Check grandchildren
        let grandchildren = tree.get_children(children[0].id);
        assert_eq!(grandchildren.len(), 1);
    }

    #[test]
    fn test_dependency_tracking() {
        let tree = create_test_tree();

        let child1 = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Child 1")
            .unwrap();
        let child2 = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Child 2")
            .unwrap();

        assert!(child2.dependencies.contains(&child1.id));
        assert!(child1.dependents.contains(&child2.id));
    }

    #[test]
    fn test_status_updates() {
        let tree = create_test_tree();

        let goal_id = tree.get_all_goals()[0].id;
        tree.update_status(goal_id, GoalStatus::InProgress).unwrap();

        let goal = tree.get_goal(goal_id).unwrap();
        assert_eq!(goal.status, GoalStatus::InProgress);
    }

    #[test]
    fn test_ready_goals() {
        let tree = create_test_tree();

        // Initially, root goals should be ready
        let ready = tree.get_ready_goals();
        assert!(ready.len() >= 2);

        // Grandchild depends on child completion via hierarchy
        let root = tree.get_root_goals()[0].clone();
        tree.update_status(root.id, GoalStatus::Completed).unwrap();

        let ready_after = tree.get_ready_goals();
        assert!(!ready_after.iter().any(|g| g.id == root.id));
    }

    #[test]
    fn test_priority_selection() {
        let tree = create_test_tree();

        let next = tree.get_next_ready_goal();
        assert!(next.is_some());

        // Should be a root goal
        let next_goal = next.unwrap();
        assert!(next_goal.parent_id.is_none());
    }

    #[test]
    fn test_topological_sort() {
        let tree = create_test_tree();

        let sorted = tree.topological_sort().unwrap();

        // Child 1 should come before Child 2 (dependency)
        let child1_idx = sorted.iter().position(|g| g.name == "Child 1").unwrap();
        let child2_idx = sorted.iter().position(|g| g.name == "Child 2").unwrap();
        assert!(child1_idx < child2_idx);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let tree = GoalTree::new();

        let goal1 = tree.create_goal("Goal 1").unwrap();
        let goal2 = tree.create_goal("Goal 2").unwrap();

        tree.add_dependency(goal1, goal2).unwrap();

        // This should fail (would create cycle)
        let result = tree.add_dependency(goal2, goal1);
        assert!(result.is_err());
    }

    #[test]
    fn test_circular_parent_detection() {
        let tree = GoalTree::new();

        let parent = tree.create_goal("Parent").unwrap();
        let child = tree.create_goal("Child").unwrap();

        tree.set_parent(child, parent).unwrap();

        // This should fail (would create cycle)
        let result = tree.set_parent(parent, child);
        assert!(result.is_err());
    }

    #[test]
    fn test_self_dependency_prevention() {
        let tree = GoalTree::new();

        let goal = tree.create_goal("Self Dep").unwrap();

        let result = tree.add_dependency(goal, goal);
        assert!(result.is_err());
    }

    #[test]
    fn test_self_parent_prevention() {
        let tree = GoalTree::new();

        let goal = tree.create_goal("Self Parent").unwrap();

        let result = tree.set_parent(goal, goal);
        assert!(result.is_err());
    }

    #[test]
    fn test_subtree_extraction() {
        let tree = create_test_tree();

        let root1 = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Root 1")
            .unwrap();
        let subtree = tree.get_subtree(root1.id);

        assert_eq!(subtree.len(), 4); // Root 1, Child 1, Child 2, Grandchild 1
    }

    #[test]
    fn test_leaf_goals() {
        let tree = create_test_tree();

        let leaves = tree.get_leaf_goals();
        assert!(leaves.len() >= 2); // At least Root 2 and Grandchild 1
    }

    #[test]
    fn test_is_descendant() {
        let tree = create_test_tree();

        let root1 = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Root 1")
            .unwrap();
        let grandchild = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Grandchild 1")
            .unwrap();

        assert!(grandchild.is_descendant_of(&tree, root1.id));
        assert!(!root1.is_descendant_of(&tree, grandchild.id));
    }

    #[test]
    fn test_is_ancestor() {
        let tree = create_test_tree();

        let root1 = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Root 1")
            .unwrap();
        let grandchild = tree
            .get_all_goals()
            .into_iter()
            .find(|g| g.name == "Grandchild 1")
            .unwrap();

        assert!(root1.is_ancestor_of(&tree, grandchild.id));
        assert!(!grandchild.is_ancestor_of(&tree, root1.id));
    }

    #[test]
    fn test_clear_tree() {
        let tree = create_test_tree();

        assert!(tree.len() > 0);
        tree.clear().unwrap();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_clone_tree() {
        let tree = create_test_tree();

        let cloned = tree.clone_tree();
        assert_eq!(cloned.len(), tree.len());

        // Modify original
        let goal_id = tree.get_all_goals()[0].id;
        tree.update_status(goal_id, GoalStatus::Completed).unwrap();

        // Clone should be unchanged
        let cloned_goal = cloned.get_goal(goal_id).unwrap();
        assert_eq!(cloned_goal.status, GoalStatus::Pending);
    }

    #[test]
    fn test_remove_dependency() {
        let tree = create_test_tree();

        let goals = tree.get_all_goals();
        let child1 = goals.iter().find(|g| g.name == "Child 1").unwrap();
        let child2 = goals.iter().find(|g| g.name == "Child 2").unwrap();

        tree.remove_dependency(child2.id, child1.id).unwrap();

        let updated_child2 = tree.get_goal(child2.id).unwrap();
        assert!(!updated_child2.dependencies.contains(&child1.id));

        let updated_child1 = tree.get_goal(child1.id).unwrap();
        assert!(!updated_child1.dependents.contains(&child2.id));
    }

    #[test]
    fn test_blocked_goals() {
        let tree = GoalTree::new();

        let goal1 = tree.create_goal("Goal 1").unwrap();
        let goal2 = tree.create_goal("Goal 2").unwrap();

        tree.add_dependency(goal2, goal1).unwrap();
        tree.update_status(goal1, GoalStatus::Failed).unwrap();

        let goal2_ref = tree.get_goal(goal2).unwrap();
        assert!(goal2_ref.is_blocked(&tree));
    }

    #[test]
    fn test_metadata() {
        let tree = GoalTree::new();

        let mut goal = Goal::new(1, "Meta Test");
        goal.metadata.insert("key".to_string(), "value".to_string());

        tree.add_goal(goal).unwrap();

        let retrieved = tree.get_goal(1).unwrap();
        assert_eq!(retrieved.metadata.get("key"), Some(&"value".to_string()));
    }
}
