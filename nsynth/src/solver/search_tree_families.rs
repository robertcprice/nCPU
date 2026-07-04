//! Stage 5 search teachers for binary tree traversal problems.
//!
//! Trees are represented as flat arrays of TreeNode structures, enabling O(1) random access
//! during iterative (non-recursive) tree traversal. These teachers recognize common tree
//! patterns (count_nodes, sum_values, max_value, tree_height, leaf_count) and emit
//! explicit-stack DFS traversal code.

use crate::benchmark::{Example, Problem, TreeNode, Value};
use crate::differentiable::DifferentiableMetadata;
use crate::solver::SolveResult;

/// Build a verified tree-family result. The tree searchers compute the answer in
/// Rust to RECOGNIZE the family, then emit a hand-written traversal program — but
/// recognizing the family is NOT the same as the EMITTED program being correct. A
/// bug in a template would otherwise return `success: true` UNVERIFIED (a latent
/// false-accept). So strict-verify the emitted code over the whole problem
/// (examples + holdout/robustness floor) and only succeed if it actually runs
/// right; a failing template degrades to `None`, not a fake "verified".
fn verified_tree_result(problem: &Problem, code: String, method: &str) -> Option<SolveResult> {
    crate::runtime::verify_problem_code_strict(problem, &code).ok()?;
    Some(SolveResult {
        success: true,
        code,
        method: method.to_string(),
        error: None,
        metadata: DifferentiableMetadata::default(),
    })
}

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
    problem.examples[0].inputs.iter().find_map(|v| match v {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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
        1 + std::cmp::max(recurse(node.left, nodes), recurse(node.right, nodes))
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
pub(super) fn search_tree_count_nodes(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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

    verified_tree_result(problem, code, "search_tree_count_nodes")
}

/// Teacher: Sum values in tree.
pub(super) fn search_tree_sum_values(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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

    verified_tree_result(problem, code, "search_tree_sum_values")
}

/// Teacher: Find maximum value in tree.
pub(super) fn search_tree_max_value(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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

    verified_tree_result(problem, code, "search_tree_max_value")
}

/// Teacher: Compute height of tree.
pub(super) fn search_tree_height(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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

    verified_tree_result(problem, code, "search_tree_height")
}

/// Teacher: Count leaf nodes (nodes with no children).
pub(super) fn search_tree_leaf_count(problem: &Problem, fn_name: &str) -> Option<SolveResult> {
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
        let tree = ex.inputs.iter().find_map(|v| match v {
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

    verified_tree_result(problem, code, "search_tree_leaf_count")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Source-level invariant. A 2026-06-30 audit claimed these five families
    /// BYPASS the verifier. That is refuted: every family returns solely via
    /// `verified_tree_result`, which calls `verify_problem_code_strict`. This
    /// tripwire keeps it that way. Scans only the pre-test portion of the file
    /// so the assertion strings here cannot skew the counts.
    #[test]
    fn tree_families_return_only_via_strict_verifier() {
        let full = include_str!("search_tree_families.rs");
        let code = full.split("#[cfg(test)]").next().unwrap();
        // Match the struct FIELD (trailing comma) so the doc comment's prose
        // mention of `success: true` doesn't count.
        assert_eq!(
            code.matches("success: true,").count(),
            1,
            "the ONLY SolveResult construction must be inside verified_tree_result"
        );
        assert!(
            code.contains("crate::runtime::verify_problem_code_strict(problem, &code)"),
            "verified_tree_result must call the strict oracle"
        );
        assert_eq!(
            code.matches("verified_tree_result(problem,").count(),
            5,
            "all five tree families must return via verified_tree_result"
        );
    }

    /// Fail-closed proof: these families emit a STALE tree representation
    /// (`tree.nodes.length`, `[i32; 1000]`) the interpreter no longer supports,
    /// so `verify_problem_code_strict` rejects the emitted code and the family
    /// returns `None` — dead, but SOUND (never a fake "verified"). If someone
    /// later fixes the codegen this test flips to a solve+verify; until then it
    /// documents the honest state instead of asserting a phantom capability.
    #[test]
    fn tree_count_nodes_is_dead_but_fail_closed() {
        let tree = vec![
            TreeNode { value: 1, left: 1, right: 2 },
            TreeNode { value: 2, left: -1, right: -1 },
            TreeNode { value: 3, left: -1, right: -1 },
        ];
        let mut p = Problem::default();
        p.name = "tree_count_v0".to_string();
        p.signature = "fn tree_count_v0(t: Tree) -> i64";
        p.examples = vec![Example {
            inputs: vec![Value::Tree(tree)],
            expected: Value::Int(3),
        }];
        // The Rust recognizer matches (3 nodes == expected 3), so any result it
        // returned would be gated by strict verify. It returns None because the
        // emitted Mog cannot pass that gate — proving there is no bypass path.
        assert!(
            search_tree_count_nodes(&p, "tree_count_v0").is_none(),
            "stale-codegen family must fail closed to None, never a fake verified"
        );
    }
}
