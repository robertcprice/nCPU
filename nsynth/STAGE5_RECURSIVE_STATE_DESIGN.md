# Stage 5: Recursive State (Tree Traversal Synthesis)

## Overview

Design Stage 5 to synthesize **iterative tree traversal functions** using an explicit stack. Rather than supporting true recursion (which requires a call stack Mog doesn't have), we'll recognize tree-structured I/O and synthesize explicit-stack traversal patterns.

Stage 5 enables synthesis of programs that operate over **tree-shaped data** (binary trees, expression trees) and return a summary value (count, sum, height, validation).

---

## 1. Data Type Extensions

### 1.1 Add `Tree<T>` Type to `Value`

**File**: `src/benchmark.rs`

Add a new `Value` variant for binary tree nodes:

```rust
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Value {
    // ... existing variants ...
    Int(i64),
    Str(String),
    Array(Vec<i64>),
    Pair(i64, i64),
    Quad(i64, i64, i64, i64),
    
    /// Binary tree node: (value, left_idx, right_idx) where indices point
    /// into a flat Vec<TreeNode> array. -1 means null/leaf.
    /// Enables O(1) access without heap indirection during execution.
    Tree(Vec<TreeNode>),
}

/// A single node in a tree representation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct TreeNode {
    pub value: i64,        // Node data
    pub left: i32,         // Index of left child (-1 for null)
    pub right: i32,        // Index of right child (-1 for null)
}

impl TreeNode {
    pub fn new(value: i64, left: i32, right: i32) -> Self {
        TreeNode { value, left, right }
    }
    
    pub fn leaf(value: i64) -> Self {
        TreeNode { value, left: -1, right: -1 }
    }
}
```

### 1.2 Update `Value::Display`, Serialization

```rust
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ... existing arms ...
            Value::Tree(nodes) => {
                // Display as: Tree([v1(L:0,R:1), v2(L:-1,R:-1), ...])
                let node_strs: Vec<_> = nodes
                    .iter()
                    .map(|n| format!("{}(L:{},R:{})", n.value, n.left, n.right))
                    .collect();
                write!(f, "Tree([{}])", node_strs.join(", "))
            }
        }
    }
}
```

### 1.3 Update Example::expected_int()

Tree problems return an `Int` summary (count, sum, height, etc.), so:

```rust
pub fn expected_int(&self) -> i64 {
    match &self.expected {
        Value::Int(i) => *i,
        Value::Pair(a, _) => *a,
        Value::Quad(a, _, _, _) => *a,
        Value::Tree(_) => 0,  // Tree is an input, not output; expected is Int
        _ => 0,
    }
}
```

---

## 2. Problem Extensions

### 2.1 Add Fields to `Problem` Struct

**File**: `src/benchmark.rs`

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub name: String,
    pub category: &'static str,
    pub description: &'static str,
    pub signature: &'static str,
    pub examples: Vec<Example>,
    pub holdouts: Vec<Example>,
    pub reference_code: &'static str,
    
    // NEW: Stage 5 fields
    pub recursive_allowed: bool,  // If true, function may recurse or use loops
    pub tree_input: bool,         // If true, one input is a Tree
    pub explicit_stack: bool,     // If true, expects explicit stack simulation
}
```

### 2.2 Helper Methods

```rust
impl Problem {
    /// Check if this problem takes a tree as input
    pub fn has_tree_input(&self) -> bool {
        self.examples
            .iter()
            .any(|ex| ex.inputs.iter().any(|v| matches!(v, Value::Tree(_))))
    }
    
    /// Check if all examples have matching tree structure
    pub fn tree_input_consistent(&self) -> bool {
        if self.examples.is_empty() {
            return true;
        }
        let first_tree = self.examples[0].inputs.iter()
            .find(|v| matches!(v, Value::Tree(_)));
        
        self.examples.iter().all(|ex| {
            ex.inputs.iter()
                .find(|v| matches!(v, Value::Tree(_)))
                .eq(&first_tree)
        })
    }
}
```

---

## 3. Codegen for Iterative Tree Traversal

### 3.1 Template Patterns

Stage 5 will emit **explicit-stack traversal code** (DFS/BFS patterns):

#### Pattern A: Count Nodes (DFS Iterative)
```rust
fn count_nodes(tree: Tree) -> i64 {
    stack: [i32; 1000] = ...;  // Explicit stack (array-based)
    sp: i32 = 0;               // Stack pointer
    count: i64 = 0;
    
    if tree.root >= 0 {
        stack[0] = tree.root;
        sp = 1;
    }
    
    while sp > 0 {
        sp = sp - 1;
        node_idx: i32 = stack[sp];
        
        if node_idx < 0 {
            continue;  // Skip null
        }
        
        node: TreeNode = tree.nodes[node_idx];
        count = count + 1;
        
        // Push children (reverse order for correct traversal)
        if node.right >= 0 {
            stack[sp] = node.right;
            sp = sp + 1;
        }
        if node.left >= 0 {
            stack[sp] = node.left;
            sp = sp + 1;
        }
    }
    
    return count;
}
```

#### Pattern B: Sum Values (DFS Iterative)
```rust
fn sum_tree(tree: Tree) -> i64 {
    stack: [i32; 1000] = ...;
    sp: i32 = 0;
    sum: i64 = 0;
    
    if tree.root >= 0 {
        stack[0] = tree.root;
        sp = 1;
    }
    
    while sp > 0 {
        sp = sp - 1;
        node_idx: i32 = stack[sp];
        
        if node_idx < 0 { continue; }
        
        node: TreeNode = tree.nodes[node_idx];
        sum = sum + node.value;
        
        if node.right >= 0 {
            stack[sp] = node.right;
            sp = sp + 1;
        }
        if node.left >= 0 {
            stack[sp] = node.left;
            sp = sp + 1;
        }
    }
    
    return sum;
}
```

#### Pattern C: Find Max Value
```rust
fn max_value(tree: Tree) -> i64 {
    stack: [i32; 1000] = ...;
    sp: i32 = 0;
    max_val: i64 = <MIN_I64>;
    
    if tree.root >= 0 {
        stack[0] = tree.root;
        sp = 1;
    }
    
    while sp > 0 {
        sp = sp - 1;
        node_idx: i32 = stack[sp];
        
        if node_idx < 0 { continue; }
        
        node: TreeNode = tree.nodes[node_idx];
        if node.value > max_val {
            max_val = node.value;
        }
        
        if node.right >= 0 {
            stack[sp] = node.right;
            sp = sp + 1;
        }
        if node.left >= 0 {
            stack[sp] = node.left;
            sp = sp + 1;
        }
    }
    
    return max_val;
}
```

#### Pattern D: Check Property (e.g., is_bst)
```rust
fn is_bst(tree: Tree) -> i64 {
    // Returns 1 if all nodes satisfy: left.value < node.value < right.value
    stack: [i32; 1000] = ...;
    sp: i32 = 0;
    valid: i64 = 1;
    
    if tree.root >= 0 {
        stack[0] = tree.root;
        sp = 1;
    }
    
    while sp > 0 && valid > 0 {
        sp = sp - 1;
        node_idx: i32 = stack[sp];
        
        if node_idx < 0 { continue; }
        
        node: TreeNode = tree.nodes[node_idx];
        
        // Check left child
        if node.left >= 0 {
            left: TreeNode = tree.nodes[node.left];
            if left.value >= node.value {
                valid = 0;
                break;
            }
            stack[sp] = node.left;
            sp = sp + 1;
        }
        
        // Check right child
        if node.right >= 0 {
            right: TreeNode = tree.nodes[node.right];
            if right.value <= node.value {
                valid = 0;
                break;
            }
            stack[sp] = node.right;
            sp = sp + 1;
        }
    }
    
    return valid;
}
```

---

## 4. Search Teachers for Tree Patterns

### 4.1 Teacher: `search_tree_count_nodes`

**File**: `src/solver/search_tree_families.rs`

```rust
pub(super) fn search_tree_count_nodes(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // Guard: Must have exactly one tree input, single i64 output
    if !problem.has_tree_input() {
        return None;
    }
    
    // Extract first example
    let first_ex = &problem.examples[0];
    let tree_input = first_ex.inputs.iter()
        .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
    
    // Check if output matches node count
    let expected = first_ex.expected_int();
    let node_count = tree_input.len() as i64;
    
    if expected != node_count {
        return None;  // Not a count_nodes pattern
    }
    
    // Verify on all examples
    for ex in &problem.examples {
        let tree = ex.inputs.iter()
            .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
        
        if ex.expected_int() != tree.len() as i64 {
            return None;
        }
    }
    
    // Generate code
    let code = format!(
        r#"fn {fn_name}(tree: Tree) -> i64 {{
    stack: [i32; 1000] = ...;
    sp: i32 = 0;
    count: i64 = 0;
    
    if tree.root >= 0 {{
        stack[0] = tree.root;
        sp = 1;
    }}
    
    while sp > 0 {{
        sp = sp - 1;
        node_idx: i32 = stack[sp];
        
        if node_idx < 0 {{ continue; }}
        
        count = count + 1;
        node: TreeNode = tree.nodes[node_idx];
        
        if node.right >= 0 {{
            stack[sp] = node.right;
            sp = sp + 1;
        }}
        if node.left >= 0 {{
            stack[sp] = node.left;
            sp = sp + 1;
        }}
    }}
    
    return count;
}}"#,
        fn_name = fn_name
    );
    
    Some(SolveResult {
        method: "search_tree_count_nodes".to_string(),
        code: code.clone(),
        time_ms: 1,
        found_by: fn_name.to_string(),
    })
}
```

### 4.2 Teacher: `search_tree_sum_values`

Similar to count, but accumulate `node.value`:

```rust
pub(super) fn search_tree_sum_values(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if !problem.has_tree_input() {
        return None;
    }
    
    // Extract tree and sum all node values
    let first_ex = &problem.examples[0];
    let tree_input = first_ex.inputs.iter()
        .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
    
    let expected = first_ex.expected_int();
    let actual_sum: i64 = tree_input.iter().map(|n| n.value).sum();
    
    if expected != actual_sum {
        return None;
    }
    
    // Verify all examples
    for ex in &problem.examples {
        let tree = ex.inputs.iter()
            .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
        let sum: i64 = tree.iter().map(|n| n.value).sum();
        if ex.expected_int() != sum {
            return None;
        }
    }
    
    // Generate code (similar to count, but accumulate node.value)
    let code = format!(
        r#"fn {fn_name}(tree: Tree) -> i64 {{
    // ... [similar stack-based traversal, but sum += node.value] ...
}}"#,
        fn_name = fn_name
    );
    
    Some(SolveResult { /* ... */ })
}
```

### 4.3 Teacher: `search_tree_max_value`

```rust
pub(super) fn search_tree_max_value(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if !problem.has_tree_input() {
        return None;
    }
    
    let first_ex = &problem.examples[0];
    let tree_input = first_ex.inputs.iter()
        .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
    
    if tree_input.is_empty() {
        return None;
    }
    
    let expected = first_ex.expected_int();
    let actual_max = tree_input.iter().map(|n| n.value).max().unwrap_or(i64::MIN);
    
    if expected != actual_max {
        return None;
    }
    
    // Verify all examples
    for ex in &problem.examples {
        let tree = ex.inputs.iter()
            .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
        if tree.is_empty() {
            continue;
        }
        let max_val = tree.iter().map(|n| n.value).max().unwrap();
        if ex.expected_int() != max_val {
            return None;
        }
    }
    
    // Generate code
    let code = format!(
        r#"fn {fn_name}(tree: Tree) -> i64 {{
    // ... [similar stack-based traversal, track max] ...
}}"#,
        fn_name = fn_name
    );
    
    Some(SolveResult { /* ... */ })
}
```

### 4.4 Teacher: `search_tree_height`

```rust
pub(super) fn search_tree_height(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if !problem.has_tree_input() {
        return None;
    }
    
    let first_ex = &problem.examples[0];
    let tree_input = first_ex.inputs.iter()
        .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
    
    let expected = first_ex.expected_int();
    let actual_height = compute_tree_height(tree_input);
    
    if expected != actual_height {
        return None;
    }
    
    // Verify all examples
    for ex in &problem.examples {
        let tree = ex.inputs.iter()
            .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
        if compute_tree_height(tree) != ex.expected_int() {
            return None;
        }
    }
    
    // Generate code (needs depth tracking in stack)
    let code = format!(
        r#"fn {fn_name}(tree: Tree) -> i64 {{
    // ... [stack holds (node_idx, depth) pairs] ...
}}"#,
        fn_name = fn_name
    );
    
    Some(SolveResult { /* ... */ })
}

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
    recurse(0, tree)  // Assume root is at index 0
}
```

### 4.5 Teacher: `search_tree_leaf_count`

Count only the leaf nodes (nodes with both children -1):

```rust
pub(super) fn search_tree_leaf_count(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if !problem.has_tree_input() {
        return None;
    }
    
    let first_ex = &problem.examples[0];
    let tree_input = first_ex.inputs.iter()
        .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
    
    let expected = first_ex.expected_int();
    let actual_leaves = tree_input.iter()
        .filter(|n| n.left < 0 && n.right < 0)
        .count() as i64;
    
    if expected != actual_leaves {
        return None;
    }
    
    // Verify all examples
    for ex in &problem.examples {
        let tree = ex.inputs.iter()
            .find_map(|v| match v { Value::Tree(nodes) => Some(nodes), _ => None })?;
        let leaves = tree.iter()
            .filter(|n| n.left < 0 && n.right < 0)
            .count() as i64;
        if ex.expected_int() != leaves {
            return None;
        }
    }
    
    // Generate code
    let code = format!(
        r#"fn {fn_name}(tree: Tree) -> i64 {{
    // ... [count nodes where left < 0 && right < 0] ...
}}"#,
        fn_name = fn_name
    );
    
    Some(SolveResult { /* ... */ })
}
```

---

## 5. Benchmark Problems (Stage 5 Tree Suite)

Add 5 tree problems to the benchmark:

### Problem 1: `count_tree_nodes`
- **Input**: Binary tree with 5-10 nodes
- **Output**: Total node count
- **Examples**: `tree{(5, [1,2], [3,4], ...)} -> 7`

### Problem 2: `sum_tree_values`
- **Input**: Binary tree with integer values
- **Output**: Sum of all node values
- **Examples**: `tree{(5, [10, -1], [3, -1], ...)} -> 18`

### Problem 3: `tree_max_value`
- **Input**: Binary tree with integer values
- **Output**: Maximum value in tree
- **Examples**: `tree{(5, [10, -1], [3, -1], ...)} -> 10`

### Problem 4: `tree_height`
- **Input**: Binary tree
- **Output**: Height of tree (path length from root to deepest leaf)
- **Examples**: `tree{...} -> 3`

### Problem 5: `tree_leaf_count`
- **Input**: Binary tree
- **Output**: Count of leaf nodes (nodes with no children)
- **Examples**: `tree{...} -> 4`

---

## 6. Integration Points

### 6.1 Module Structure

Create new file: `src/solver/search_tree_families.rs`

```rust
// Five teachers: search_tree_count_nodes, search_tree_sum_values,
// search_tree_max_value, search_tree_height, search_tree_leaf_count

mod search_tree_families {
    use super::*;
    
    pub fn search_tree_count_nodes(...) -> Option<SolveResult> { ... }
    pub fn search_tree_sum_values(...) -> Option<SolveResult> { ... }
    pub fn search_tree_max_value(...) -> Option<SolveResult> { ... }
    pub fn search_tree_height(...) -> Option<SolveResult> { ... }
    pub fn search_tree_leaf_count(...) -> Option<SolveResult> { ... }
}
```

### 6.2 Register in Solver

Update `src/solver/solver.rs` to add tree teachers to the search roster:

```rust
// In solve_problem() or the search candidate list:
SearchCandidate {
    key: "search_tree_count_nodes",
    func: search_tree_families::search_tree_count_nodes,
},
SearchCandidate {
    key: "search_tree_sum_values",
    func: search_tree_families::search_tree_sum_values,
},
SearchCandidate {
    key: "search_tree_max_value",
    func: search_tree_families::search_tree_max_value,
},
SearchCandidate {
    key: "search_tree_height",
    func: search_tree_families::search_tree_height,
},
SearchCandidate {
    key: "search_tree_leaf_count",
    func: search_tree_families::search_tree_leaf_count,
},
```

---

## 7. Testing & Validation

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tree_count_nodes() { ... }
    
    #[test]
    fn test_tree_sum_values() { ... }
    
    #[test]
    fn test_tree_max_value() { ... }
    
    #[test]
    fn test_tree_height() { ... }
    
    #[test]
    fn test_tree_leaf_count() { ... }
}
```

### 7.2 Integration Tests

Run benchmarks and verify:
- All 5 tree problems solve
- No regression on existing problems
- Teachers are called in correct order (most specific first)
- Code verification passes (parse + execute)

---

## 8. Implementation Roadmap

### Phase 1: Data Types (1-2 hours)
- [ ] Add `TreeNode` struct and `Value::Tree` variant
- [ ] Update `Display`, serialization, helper methods
- [ ] Update `Problem` struct with `recursive_allowed`, `tree_input`, `explicit_stack` flags

### Phase 2: Search Teachers (3-4 hours)
- [ ] Create `src/solver/search_tree_families.rs`
- [ ] Implement 5 search teachers (count, sum, max, height, leaf_count)
- [ ] Add unit tests for each teacher

### Phase 3: Benchmarks (2-3 hours)
- [ ] Add 5 tree problems to benchmark suite
- [ ] Ensure examples are valid tree structures
- [ ] Verify reference code works

### Phase 4: Integration & Testing (2-3 hours)
- [ ] Register teachers in solver
- [ ] Run full benchmark suite
- [ ] Verify no regressions
- [ ] Document code

### **Estimated Total**: 8–12 hours

---

## 9. Success Criteria

✅ All 5 tree problems solve via explicit tree teachers
✅ No regression on existing (Stage 1-4) problems
✅ Tree pattern recognition works for various tree shapes
✅ Code generation produces syntactically valid Mog
✅ Verification passes on all examples (incl. holdouts)
✅ Unit tests cover all teachers + edge cases
✅ Documentation explains iterative stack simulation vs true recursion

---

## 10. Future Extensions (Post-Stage 5)

1. **Recursive codegen** (true call stack when Mog supports it)
2. **Tree transformation** (map, fold over tree values)
3. **Graph traversal** (DAGs, cycles with visited set)
4. **Balanced tree validation** (AVL, red-black properties)
5. **Binary search tree operations** (insert, delete, search)
