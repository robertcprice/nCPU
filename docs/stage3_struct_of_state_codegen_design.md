# Stage 3: Struct-of-State Codegen Design for nsynth

**Status**: Design document (not yet implemented)  
**Target**: Extends nsynth scalable stateful synthesis from scalar-state (Stage 1.5) to multi-field struct state (Stage 3)  
**Integration**: Parallel to tensor_codegen.rs; called from `solver::solve_problem` for struct-input problems

---

## Executive Summary

**What We're Building**: A codegen framework that synthesizes Mog functions with **multi-field struct state**, where each field independently evolves via reducer operations over array inputs.

**Why It Matters**: 
- Unlocks stateful algorithms on heterogeneous state (count, sum, max, min, flags, etc. all evolving in parallel)
- Enables window-based feature aggregation (running statistics, rolling sums, momentum)
- Bridges synthesis from scalar accumulators (105 benchmarks solved) to realistic stateful computations (10-15 new problems per struct shape)

**Example Problem** (solve via struct):
```mog
fn aggregate(state: State, arr: [i64]) -> State {
  count_new = state.count + len(arr);
  sum_new = state.sum + sum(arr);
  max_new = max(state.max, max(arr));
  return { count: count_new, sum: sum_new, max: max_new };
}
```

---

## Architecture Overview

### 1. Core Data Structures

#### StructField
```rust
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StructField {
    pub field_name: String,      // "count", "sum", "max"
    pub field_type: String,       // "i64" (may extend to "f64", "bool")
    pub init_value: i64,          // initial state for this field (0, 1, i64::MIN, i64::MAX)
}
```

#### StructStateType
```rust
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StructStateType {
    pub struct_name: String,      // "State", "Accumulator", etc.
    pub fields: Vec<StructField>, // ordered list of fields
}

impl StructStateType {
    pub fn mog_struct_def(&self) -> String {
        // Emits: struct State { count: i64, sum: i64, max: i64 }
    }
    
    pub fn mog_constructor(&self, init_vals: &[i64]) -> String {
        // Emits: { count: 0, sum: 0, max: <min>, ... }
    }
}
```

#### StructStateCodegen
```rust
#[derive(Debug, Clone)]
pub struct StructStateCodegen {
    struct_type: StructStateType,
    field_transitions: HashMap<String, FieldTransitionRule>,
    var_counter: usize,
}

impl StructStateCodegen {
    pub fn new(struct_type: StructStateType) -> Self;
    
    pub fn emit_function(
        &mut self,
        fn_name: &str,
        problem: &Problem,
    ) -> String;
}
```

### 2. Field Transition Patterns (5-10 Templates)

Each template models one field's evolution rule.

#### Pattern 1: **Sum Accumulation**
```
field_new = field_prev + sum(arr)
Codegen:  let {field}_new = {field_prev} + sum({arr});
Validators: all examples match this pattern
```

#### Pattern 2: **Max/Min Tracking**
```
field_new = max(field_prev, max(arr))
field_new = min(field_prev, min(arr))
Codegen:  let {field}_new = max({field_prev}, max({arr}));
```

#### Pattern 3: **Count with Predicate**
```
field_new = field_prev + count_positive(arr)
field_new = field_prev + count_zero(arr)
field_new = field_prev + count_negative(arr)
Codegen:  let {field}_new = {field_prev} + count_positive({arr});
```

#### Pattern 4: **Multiplicative Update**
```
field_new = field_prev * product(arr)
Codegen:  let {field}_new = {field_prev} * product({arr});
Validators: product reduction (multiply all elements)
```

#### Pattern 5: **Delta Accumulation (Cross-Field)**
```
field_new = field_prev + (reducer_a(arr) - reducer_b(arr))
Codegen:  let {field}_new = {field_prev} + (sum({arr}) - count({arr}));
Validators: matches composite reducer patterns from Stage 1.5
```

#### Pattern 6: **Binary Combine (Two Array Inputs)**
```
field_new = field_prev + reducer_a(arr1) + reducer_b(arr2)
Codegen:  let {field}_new = {field_prev} + sum({arr1}) + max({arr2});
Validators: extends dual-reducer logic from `search_stateful_reducer_dual`
```

#### Pattern 7: **Event-Modulated (Conditional)**
```
field_new = if event > 0 { field_prev + sum(arr) } else { field_prev }
Codegen:  let {field}_new = if {event} > 0 { {field_prev} + sum({arr}) } else { {field_prev} };
Validators: gated reducer from `search_stateful_reducer_event`
```

#### Pattern 8: **Min/Max with Clamp**
```
field_new = clamp(field_prev + sum(arr), lower, upper)
Codegen:  let {field}_new = max({lower}, min({field_prev} + sum({arr}), {upper}));
Validators: extract bounds from examples, validate all respect them
```

#### Pattern 9: **Running Average (Implicit)**
```
field_new = (field_prev * count_prev + sum(arr)) / (count_prev + len(arr))
Codegen:  let count_new = count_prev + len(arr);
          let sum_new = sum_prev + sum(arr);
          let avg_new = sum_new / count_new;  (if avg field requested)
Validators: requires careful pattern recognition to avoid false positives
```

#### Pattern 10: **Identity Pass-Through (Unchanged Field)**
```
field_new = field_prev
Codegen:  let {field}_new = {field_prev};
Validators: all examples show this field unchanged
```

---

## Implementation Strategy

### Phase A: Core Codegen Functions (220–280 lines)

```rust
// nsynth/src/solver/struct_of_state_codegen.rs

pub(super) fn codegen_struct_field_transition(
    field: &StructField,
    transition_rule: &FieldTransitionRule,
    field_prev_name: &str,
    arr_arg: &str,
    other_args: Option<&[&str]>,  // for event, second array, etc.
) -> String {
    // Returns: "    let {field}_new = {expr};"
    // Handles all 10 pattern templates above
    // ~30 lines per pattern, switch statement
}

pub(super) fn codegen_struct_return_statement(
    struct_type: &StructStateType,
    field_new_names: &[&str],
) -> String {
    // Returns: "    return { count: count_new, sum: sum_new, max: max_new };"
    // ~15 lines
}

pub fn emit_struct_of_state_function(
    fn_name: &str,
    struct_type: &StructStateType,
    rules: Vec<FieldTransitionRule>,
    param_names: &[&str],  // [state_prev, arr, ...]
) -> String {
    // Orchestrates function body:
    // 1. Unpack state fields into local vars (state_prev.count → count_prev)
    // 2. Emit each field's transition codegen
    // 3. Construct and return new struct
    // ~80 lines
}

pub(super) fn reducer_transition(
    reducer_kind: &str,
    arr_arg: &str,
    field_prev: &str,
    op: &str,
) -> String {
    // Reducer-only transitions: sum, max, min, count_*, product
    // Reuses logic from search_codegen.rs::reducer_body
    // ~20 lines
}
```

### Phase B: Search Teacher (180–240 lines)

```rust
// Added to nsynth/src/solver/search_families.rs

pub(super) fn search_struct_of_state(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    // 1. Parse problem signature: struct-input, struct-output
    let struct_type = infer_struct_type_from_problem(problem)?;
    
    // 2. For each field, enumerate matching transition patterns
    let mut field_rules = Vec::new();
    for field in &struct_type.fields {
        if let Some(rule) = find_matching_transition_rule(
            problem,
            &struct_type,
            field,
        ) {
            field_rules.push(rule);
        }
    }
    
    // 3. If all fields matched, emit and verify
    if field_rules.len() == struct_type.fields.len() {
        let code = emit_struct_of_state_function(
            fn_name,
            &struct_type,
            field_rules,
            extract_param_names(&problem.signature),
        );
        return verified_result(problem, code, "search_struct_of_state");
    }
    None
}

fn find_matching_transition_rule(
    problem: &Problem,
    struct_type: &StructStateType,
    field: &StructField,
) -> Option<FieldTransitionRule> {
    // For each field, try 10 templates in priority order
    // (identity first, then simple reducers, then compounds)
    // Return first that validates across all examples
    // ~100 lines
}

fn infer_struct_type_from_problem(problem: &Problem) -> Option<StructStateType> {
    // From problem.signature, extract struct name and field types
    // Example: "fn f(state: State, arr: [i64]) -> State"
    // Call nsynth's type parser for "State" → field list
    // ~30 lines
}
```

### Phase C: Integration Hook (50–80 lines)

```rust
// In nsynth/src/solver/search.rs::solve_problem (existing)

fn solve_problem(problem: &Problem) -> SolveResult {
    // ... existing stages ...
    
    // NEW: Stage 3 (between Stage 2 tensor and Stage 4 fallback)
    if has_struct_input_output(problem) {
        if let Some(result) = search_struct_of_state(problem, fn_name) {
            return result;
        }
    }
    
    // ... continue to Stage 4 ...
}

fn has_struct_input_output(problem: &Problem) -> bool {
    // Check if signature contains custom struct types
    // Exclude built-in array/scalar/string/tensor types
}
```

---

## Field Transition Rule Data Model

```rust
#[derive(Debug, Clone)]
pub struct FieldTransitionRule {
    pub field_name: String,
    pub pattern_kind: TransitionPatternKind,
    pub mog_expr: String,  // the actual code to emit
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TransitionPatternKind {
    Identity,                            // field_new = field_prev
    SimpleReducer(String, String),       // (reducer, op) e.g. ("sum", "+")
    CompoundReducer(String, String, String, String),  // (r_a, op1, r_b, op2)
    DualArrayReducer(String, String, String, String), // two arrays
    EventModulated(String, String, String),  // (combine_kind, op, gate_kind)
    Clamp(i64, i64),                    // (lower, upper)
    Multiplicative(String),              // (reducer)
    Custom(String),                      // fallback for hand-coded patterns
}
```

---

## Validation & Test Strategy

### Unit Tests (60–90 lines)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_field_pattern_identity() {
        let field = StructField { 
            field_name: "count".to_string(), 
            field_type: "i64".to_string(),
            init_value: 0,
        };
        let rule = FieldTransitionRule {
            field_name: "count".to_string(),
            pattern_kind: TransitionPatternKind::Identity,
            mog_expr: "count_prev".to_string(),
        };
        let code = codegen_struct_field_transition(&field, &rule, "count_prev", "arr", None);
        assert!(code.contains("let count_new = count_prev;"));
    }

    #[test]
    fn test_struct_of_state_full_example() {
        // Problem: aggregate(state: State, arr: [i64]) -> State
        // Expected: { count: count_prev + len(arr), sum: sum_prev + sum(arr), ... }
        // Verify emitted code validates on 5-10 examples
    }

    #[test]
    fn test_struct_field_pattern_dual_reducer() {
        // Pattern 5: cross-field delta
        // Verify (sum - count) and similar compounds
    }

    #[test]
    fn test_event_modulated_struct_transition() {
        // Pattern 7: gated reducer
        // Verify if-else codegen correctness
    }
}
```

### Integration Points

1. **Call from `solver.rs::solve_problem`**: Add Stage 3 check between tensor (Stage 2) and fallback (Stage 4)
2. **Verify via existing `verify_problem_code_strict`**: All emitted code runs through standard validation
3. **Compose with dual/event teachers**: Reuse reducer logic from `search_stateful_reducer_dual` and `search_stateful_reducer_event`

---

## Expected Problem Coverage

### 10-15 New Benchmark Classes

| Class | Example | Struct Fields | Patterns Used |
|-------|---------|---------------|---------------|
| **Running Statistics** | aggregate_metrics(state, arr) | count, sum, max, min | Patterns 1, 2, 3 |
| **Delta Accumulation** | track_changes(state, arr1, arr2) | total, delta | Patterns 5, 6 |
| **Gated Aggregation** | conditional_sum(state, event, arr) | sum, flag | Patterns 7, 10 |
| **Momentum** | exponential_moving_avg(state, arr) | sum, count, avg | Patterns 9, 1, 3 |
| **Windowing** | sliding_feature(state, arr) | recent_sum, recent_max, recent_min | Patterns 1, 2 |
| **Multi-Modal** | classify_and_count(state, arr) | positives, zeros, negatives | Patterns 3, 10 |
| **Clamped Range** | bounded_accumulator(state, arr) | value | Pattern 8 |
| **Product Tracking** | running_product(state, arr) | product | Pattern 4 |

---

## Key Design Decisions

### Decision 1: Field Ordering
**Choice**: Preserve field order from struct definition (deterministic, matches problem signature)  
**Rationale**: Simplifies validation, matches user intent, no ambiguity in return struct construction

### Decision 2: Init Value Inference
**Choice**: Infer from first example's state snapshot (e.g., if count=0, assume init=0)  
**Rationale**: Most problems start from a clean state; examples reveal the ground truth

### Decision 3: Pattern Priority
**Choice**: Try identity first, then simple reducers (1, 2, 3), then compounds (5, 6, 7), then special cases (8, 9)  
**Rationale**: Identity is cheapest check; simple reducers cover 70% of problems; compounds handle remainder

### Decision 4: No Implicit Field Coupling
**Choice**: Each field transition is independent; no cross-field expressions (yet)  
**Rationale**: Simplifies validation and reduces false positives; can extend in Stage 4 if needed

### Decision 5: Struct Name Handling
**Choice**: Infer from problem signature; if ambiguous, assume "State"  
**Rationale**: Mog type system requires explicit struct names; problems usually follow naming convention

---

## Estimated Scope

| Component | Lines of Code | Confidence |
|-----------|---------------|-----------|
| Core codegen functions | 220–280 | 0.85 |
| Search teacher | 180–240 | 0.80 |
| Data structures + helpers | 80–120 | 0.90 |
| Unit tests | 60–90 | 0.85 |
| Integration hook | 50–80 | 0.95 |
| **Total** | **590–810** | **0.87** |

### Confidence Notes
- **High (0.90+)**: Data structures, integration hooks, basic codegen (reuse existing patterns)
- **Medium (0.80–0.85)**: Search teacher (pattern matching can be subtle), unit tests (need good coverage)
- **Medium-Low (0.75–0.80)**: Init value inference (edge cases in struct inference), event-modulated patterns

---

## Integration with Existing Codebase

### Reuse Opportunities

1. **`search_codegen.rs::reducer_body()`** → Extract for Pattern 1–6 transition emission
2. **`search_families.rs::reducer_fn()`** → Reuse reducer ground-truth evaluators
3. **`tensor_codegen.rs::TensorCodegen` pattern** → Model StructStateCodegen similarly (stateful builder)
4. **`verify_problem_code_strict()`** → No changes; use as-is for validation

### New Dependencies

- Parse struct types from Mog signatures (extend existing type parser)
- Handle struct construction syntax `{ field: value, ... }` in emitted code

---

## Next Steps

1. **Implement Phase A** (core codegen): 1–2 days
   - Codegen for each pattern template
   - Test on synthetic struct problems

2. **Implement Phase B** (search teacher): 1–2 days
   - Pattern matching for each field
   - Validation loop

3. **Implement Phase C** (integration): 0.5 days
   - Hook into solver.rs
   - End-to-end test on benchmark suite

4. **Extend to Stage 3.5** (struct composition): future
   - Multiple nested structs
   - Field arithmetic (cross-field expressions)
   - Composite reducers over struct arrays

---

## Appendix: Example Full Codegen Output

**Problem Input**:
```mog
fn aggregate(state: State, arr: [i64]) -> State {
  // state has fields: count (i64), sum (i64), max (i64)
  // examples show:
  // - count increments by len(arr)
  // - sum increments by sum(arr)
  // - max updates to max(max, max(arr))
}
```

**Emitted Code**:
```mog
fn aggregate(state: State, arr: [i64]) -> State {
    count_prev: i64 = state.count;
    sum_prev: i64 = state.sum;
    max_prev: i64 = state.max;
    
    arr_sum: i64 = 0;
    for v in arr {
        arr_sum = arr_sum + v;
    }
    arr_max: i64 = arr[0];
    for v in arr {
        if v > arr_max { arr_max = v; }
    }
    
    count_new: i64 = count_prev + len(arr);
    sum_new: i64 = sum_prev + arr_sum;
    max_new: i64 = count_prev;
    if arr_max > max_prev { max_new = arr_max; }
    
    return { count: count_new, sum: sum_new, max: max_new };
}
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-06-15  
**Author**: nsynth Design  
