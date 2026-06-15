//! Stage 5 search teachers for binary tree traversal problems.
//!
//! Trees are represented as flat arrays of TreeNode structures, enabling O(1) random access
//! during iterative (non-recursive) tree traversal. These teachers recognize common tree
//! patterns (count_nodes, sum_values, max_value, tree_height, leaf_count) and emit
//! explicit-stack DFS traversal code.

use crate::benchmark::{Example, Problem, TreeNode, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::solver::SolveResult;

/// Check if this problem takes exactly one tree input and returns an i64.
fn is_tree_problem(problem: &Problem) -> bool {
    // Must have examples
    if problem.examples.is_empty() {
        return false;
    }

    // Check that at least one input is a tree
    for ex in &problem.examples {
        for inp in &ex.inputs {
            if matches!(inp, Value::Tree(_)) {
                return true;
            }
        }
    }
    false
}

/// Extract the tree from a problem's first example (if it exists).
fn extract_tree(problem: &Problem) -> Option<&[TreeNode]> {
    problem.examples[0]
        .inputs
        .iter()
        .find_map(|v| match v {
            Value::Tree(nodes) => Some(nodes.as_slice()),
            _ => None,
        })
}

/// Verify that all examples have the same tree as input (structural consistency check).
fn tree_consistent_across_examples(problem: &Problem) -> bool {
    if problem.examples.len() <= 1 {
        return true;
    }
    let first_tree = extract_tree(problem);
    problem.examples.iter().all(|ex| {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes.as_slice()),
                _ => None,
            });
        tree == first_tree
    })
}

/// Compute tree height recursively (for pattern matching).
fn compute_tree_height(tree: &[TreeNode]) -> i64 {
    if tree.is_empty() {
        return -1;
    }

    fn recurse(idx: i32, nodes: &[TreeNode]) -> i64 {
        if idx < 0 {
            return -1;
        }
        let node = &nodes[idx as usize];
        1 + std::cmp::max(
            recurse(node.left, nodes),
            recurse(node.right, nodes),
        )
    }

    recurse(0, tree)
}

/// Count total nodes in tree (including root).
fn count_nodes_recursive(idx: i32, tree: &[TreeNode]) -> i64 {
    if idx < 0 {
        return 0;
    }
    let node = &tree[idx as usize];
    1 + count_nodes_recursive(node.left, tree) + count_nodes_recursive(node.right, tree)
}

/// Sum all node values in tree.
fn sum_nodes_recursive(idx: i32, tree: &[TreeNode]) -> i64 {
    if idx < 0 {
        return 0;
    }
    let node = &tree[idx as usize];
    node.value + sum_nodes_recursive(node.left, tree) + sum_nodes_recursive(node.right, tree)
}

/// Find max value in tree.
fn max_value_recursive(idx: i32, tree: &[TreeNode]) -> Option<i64> {
    if idx < 0 {
        return None;
    }
    let node = &tree[idx as usize];
    let left_max = max_value_recursive(node.left, tree);
    let right_max = max_value_recursive(node.right, tree);

    let mut candidates = vec![node.value];
    if let Some(lm) = left_max {
        candidates.push(lm);
    }
    if let Some(rm) = right_max {
        candidates.push(rm);
    }

    Some(*candidates.iter().max().unwrap())
}

/// Count leaf nodes (nodes with both children == -1).
fn count_leaves_recursive(idx: i32, tree: &[TreeNode]) -> i64 {
    if idx < 0 {
        return 0;
    }
    let node = &tree[idx as usize];
    if node.left < 0 && node.right < 0 {
        1
    } else {
        count_leaves_recursive(node.left, tree) + count_leaves_recursive(node.right, tree)
    }
}

/// Teacher: Count nodes in tree.
pub(super) fn search_tree_count_nodes(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must be a tree problem with consistent examples
    if !is_tree_problem(problem) || !tree_consistent_across_examples(problem) {
        return None;
    }

    let tree = extract_tree(problem)?;
    let expected = problem.examples[0].expected_int();
    let actual_count = tree.len() as i64;

    // Verify on first example
    if expected != actual_count {
        return None;
    }

    // Verify on all examples (all should have same tree)
    for ex in &problem.examples {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes),
                _ => None,
            })?;

        if ex.expected_int() != tree.len() as i64 {
            return None;
        }
    }

    let code = format!(
        "fn {fn_name}(tree: Tree) -> i64 {{\n    \
         stack: [i32; 1000] = [];\n    \
         sp: i32 = 0;\n    \
         count: i64 = 0;\n    \
         \n    \
         if tree.nodes.length > 0 {{\n        \
         stack[0] = 0;\n        \
         sp = 1;\n    \
         }}\n    \
         \n    \
         while sp > 0 {{\n        \
         sp = sp - 1;\n        \
         node_idx: i32 = stack[sp];\n        \
         \n        \
         if node_idx < 0 {{ continue; }}\n        \
         \n        \
         node: TreeNode = tree.nodes[node_idx];\n        \
         count = count + 1;\n        \
         \n        \
         if node.right >= 0 {{\n          \
         stack[sp] = node.right;\n          \
         sp = sp + 1;\n        \
         }}\n        \
         if node.left >= 0 {{\n          \
         stack[sp] = node.left;\n          \
         sp = sp + 1;\n        \
         }}\n    \
         }}\n    \
         \n    \
         return count;\n\
         }}"
    );

    Some(SolveResult {
        success: true,
        code,
        method: "search_tree_count_nodes".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

/// Teacher: Sum values in tree.
pub(super) fn search_tree_sum_values(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must be a tree problem
    if !is_tree_problem(problem) || !tree_consistent_across_examples(problem) {
        return None;
    }

    let tree = extract_tree(problem)?;
    let expected = problem.examples[0].expected_int();
    let actual_sum: i64 = tree.iter().map(|n| n.value).sum();

    // Verify on first example
    if expected != actual_sum {
        return None;
    }

    // Verify on all examples
    for ex in &problem.examples {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes),
                _ => None,
            })?;

        let sum: i64 = tree.iter().map(|n| n.value).sum();
        if ex.expected_int() != sum {
            return None;
        }
    }

    let code = format!(
        "fn {fn_name}(tree: Tree) -> i64 {{\n    \
         stack: [i32; 1000] = [];\n    \
         sp: i32 = 0;\n    \
         sum: i64 = 0;\n    \
         \n    \
         if tree.nodes.length > 0 {{\n        \
         stack[0] = 0;\n        \
         sp = 1;\n    \
         }}\n    \
         \n    \
         while sp > 0 {{\n        \
         sp = sp - 1;\n        \
         node_idx: i32 = stack[sp];\n        \
         \n        \
         if node_idx < 0 {{ continue; }}\n        \
         \n        \
         node: TreeNode = tree.nodes[node_idx];\n        \
         sum = sum + node.value;\n        \
         \n        \
         if node.right >= 0 {{\n          \
         stack[sp] = node.right;\n          \
         sp = sp + 1;\n        \
         }}\n        \
         if node.left >= 0 {{\n          \
         stack[sp] = node.left;\n          \
         sp = sp + 1;\n        \
         }}\n    \
         }}\n    \
         \n    \
         return sum;\n\
         }}"
    );

    Some(SolveResult {
        success: true,
        code,
        method: "search_tree_sum_values".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

/// Teacher: Find maximum value in tree.
pub(super) fn search_tree_max_value(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must be a tree problem
    if !is_tree_problem(problem) || !tree_consistent_across_examples(problem) {
        return None;
    }

    let tree = extract_tree(problem)?;
    if tree.is_empty() {
        return None;
    }

    let expected = problem.examples[0].expected_int();
    let actual_max = max_value_recursive(0, tree)?;

    // Verify on first example
    if expected != actual_max {
        return None;
    }

    // Verify on all examples
    for ex in &problem.examples {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes),
                _ => None,
            })?;

        if tree.is_empty() {
            continue;
        }

        if let Some(max_val) = max_value_recursive(0, tree) {
            if ex.expected_int() != max_val {
                return None;
            }
        } else {
            return None;
        }
    }

    let code = format!(
        "fn {fn_name}(tree: Tree) -> i64 {{\n    \
         stack: [i32; 1000] = [];\n    \
         sp: i32 = 0;\n    \
         max_val: i64 = -9223372036854775808;\n    \
         \n    \
         if tree.nodes.length > 0 {{\n        \
         stack[0] = 0;\n        \
         sp = 1;\n    \
         }}\n    \
         \n    \
         while sp > 0 {{\n        \
         sp = sp - 1;\n        \
         node_idx: i32 = stack[sp];\n        \
         \n        \
         if node_idx < 0 {{ continue; }}\n        \
         \n        \
         node: TreeNode = tree.nodes[node_idx];\n        \
         if node.value > max_val {{\n          \
         max_val = node.value;\n        \
         }}\n        \
         \n        \
         if node.right >= 0 {{\n          \
         stack[sp] = node.right;\n          \
         sp = sp + 1;\n        \
         }}\n        \
         if node.left >= 0 {{\n          \
         stack[sp] = node.left;\n          \
         sp = sp + 1;\n        \
         }}\n    \
         }}\n    \
         \n    \
         return max_val;\n\
         }}"
    );

    Some(SolveResult {
        success: true,
        code,
        method: "search_tree_max_value".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

/// Teacher: Compute height of tree.
pub(super) fn search_tree_height(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must be a tree problem
    if !is_tree_problem(problem) || !tree_consistent_across_examples(problem) {
        return None;
    }

    let tree = extract_tree(problem)?;
    let expected = problem.examples[0].expected_int();
    let actual_height = compute_tree_height(tree);

    // Verify on first example
    if expected != actual_height {
        return None;
    }

    // Verify on all examples
    for ex in &problem.examples {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes),
                _ => None,
            })?;

        if compute_tree_height(tree) != ex.expected_int() {
            return None;
        }
    }

    // Height computation: use explicit stack with (index, depth) pairs stored
    // in separate arrays (simulating tuple packing).
    let code = format!(
        "fn {fn_name}(tree: Tree) -> i64 {{\n    \
         stack_idx: [i32; 1000] = [];\n    \
         stack_depth: [i64; 1000] = [];\n    \
         sp: i32 = 0;\n    \
         max_depth: i64 = -1;\n    \
         \n    \
         if tree.nodes.length > 0 {{\n        \
         stack_idx[0] = 0;\n        \
         stack_depth[0] = 0;\n        \
         sp = 1;\n    \
         }}\n    \
         \n    \
         while sp > 0 {{\n        \
         sp = sp - 1;\n        \
         node_idx: i32 = stack_idx[sp];\n        \
         depth: i64 = stack_depth[sp];\n        \
         \n        \
         if node_idx < 0 {{ continue; }}\n        \
         \n        \
         node: TreeNode = tree.nodes[node_idx];\n        \
         if depth > max_depth {{\n          \
         max_depth = depth;\n        \
         }}\n        \
         \n        \
         if node.right >= 0 {{\n          \
         stack_idx[sp] = node.right;\n          \
         stack_depth[sp] = depth + 1;\n          \
         sp = sp + 1;\n        \
         }}\n        \
         if node.left >= 0 {{\n          \
         stack_idx[sp] = node.left;\n          \
         stack_depth[sp] = depth + 1;\n          \
         sp = sp + 1;\n        \
         }}\n    \
         }}\n    \
         \n    \
         return max_depth;\n\
         }}"
    );

    Some(SolveResult {
        success: true,
        code,
        method: "search_tree_height".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

/// Teacher: Count leaf nodes (nodes with no children).
pub(super) fn search_tree_leaf_count(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must be a tree problem
    if !is_tree_problem(problem) || !tree_consistent_across_examples(problem) {
        return None;
    }

    let tree = extract_tree(problem)?;
    let expected = problem.examples[0].expected_int();
    let actual_leaves = count_leaves_recursive(0, tree);

    // Verify on first example
    if expected != actual_leaves {
        return None;
    }

    // Verify on all examples
    for ex in &problem.examples {
        let tree = ex
            .inputs
            .iter()
            .find_map(|v| match v {
                Value::Tree(nodes) => Some(nodes),
                _ => None,
            })?;

        if count_leaves_recursive(0, tree) != ex.expected_int() {
            return None;
        }
    }

    let code = format!(
        "fn {fn_name}(tree: Tree) -> i64 {{\n    \
         stack: [i32; 1000] = [];\n    \
         sp: i32 = 0;\n    \
         leaves: i64 = 0;\n    \
         \n    \
         if tree.nodes.length > 0 {{\n        \
         stack[0] = 0;\n        \
         sp = 1;\n    \
         }}\n    \
         \n    \
         while sp > 0 {{\n        \
         sp = sp - 1;\n        \
         node_idx: i32 = stack[sp];\n        \
         \n        \
         if node_idx < 0 {{ continue; }}\n        \
         \n        \
         node: TreeNode = tree.nodes[node_idx];\n        \
         if node.left < 0 && node.right < 0 {{\n          \
         leaves = leaves + 1;\n        \
         }}\n        \
         \n        \
         if node.right >= 0 {{\n          \
         stack[sp] = node.right;\n          \
         sp = sp + 1;\n        \
         }}\n        \
         if node.left >= 0 {{\n          \
         stack[sp] = node.left;\n          \
         sp = sp + 1;\n        \
         }}\n    \
         }}\n    \
         \n    \
         return leaves;\n\
         }}"
    );

    Some(SolveResult {
        success: true,
        code,
        method: "search_tree_leaf_count".to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

// TODO: Tree teacher tests incomplete - Problem struct initialization needs all fields.
// Tree teachers are functional but tests deferred to Stage 5 integration test suite.
// #[cfg(test)]
// mod tests {
//     use super::*;
// }
