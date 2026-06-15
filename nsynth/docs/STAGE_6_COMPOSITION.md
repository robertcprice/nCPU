# Stage 6: Module Composition & Function Pipeline Synthesis

**Goal**: Enable nsynth to synthesize programs that CALL other synthesized functions in sequence, composing behavior through function composition rather than writing monolithic functions.

**Status**: Design phase (not yet implemented)

---

## 1. High-Level Vision

Currently, each synthesized function is a **monolithic block** — all computation happens in a single function body with no abstraction boundaries. Stage 6 introduces **function-level composition**, where:

1. A library of reusable functions exists (filter, map, reduce, scan, etc.)
2. Problems can be solved by **chaining calls** to library functions
3. The solver recognizes common patterns and emits function call sequences
4. Output of one function becomes the input to the next

### Example

**Problem**: `filter_sum_positive([1, -2, 3, -4, 5]) -> 9`

**Current (monolithic)**:
```mog
fn filter_sum_positive(arr: [i64]) -> i64 {
  result: i64 = 0
  i: i64 = 0
  while i < len(arr) {
    if arr[i] > 0 {
      result = result + arr[i]
    }
    i = i + 1
  }
  return result
}
```

**Stage 6 (composed)**:
```mog
fn filter(arr: [i64], predicate: fn(i64)->bool) -> [i64] { ... }
fn sum(arr: [i64]) -> i64 { ... }
fn filter_sum_positive(arr: [i64]) -> i64 {
  filtered: [i64] = filter(arr, fn(x: i64)->bool { return x > 0 })
  return sum(filtered)
}
```

---

## 2. Architecture Overview

### 2.1 Composition Library (Hard-Coded Templates)

**File**: `nsynth/src/synthesis/composition_lib.rs` (new)

```rust
pub struct CompositionTemplate {
    pub name: &'static str,
    pub signature: &'static str,
    pub code: &'static str,
    pub pattern: CompositionPattern,  // What it detects in I/O
}

pub enum CompositionPattern {
    Filter,        // Input array → smaller output array
    Map,           // Input array → output array (same size)
    Reduce,        // Input array → scalar
    Scan,          // Input array → array (cumulative)
    Sort,          // Input array → sorted array
    Zip,           // Two arrays → array of pairs
    Transpose,     // Array of pairs → two arrays
    FindAll,       // Predicate search → indices
}
```

**Initial Library** (hand-coded):
- `filter(arr, predicate) -> [i64]` — removes elements not matching predicate
- `map(arr, transform) -> [i64]` — applies transformation to each element
- `reduce(arr, init, op) -> i64` — folds array to scalar
- `scan(arr, init, op) -> [i64]` — cumulative fold
- `sum(arr) -> i64` — special case of reduce with +
- `product(arr) -> i64` — special case of reduce with *
- `reverse(arr) -> [i64]` — reverses array
- `sorted(arr) -> [i64]` — returns sorted array
- `count_matching(arr, predicate) -> i64` — counts elements matching predicate

### 2.2 Problem Metadata Extension

**File**: `nsynth/src/benchmark.rs` (extend `Problem`)

```rust
pub struct Problem {
    // ... existing fields ...
    
    /// If true, solver may emit function calls to library functions
    pub composition_allowed: bool,
    
    /// Hint for which composition patterns might work (empty = unknown)
    pub composition_hints: Vec<CompositionPattern>,
}
```

### 2.3 Solver Extension

**File**: `nsynth/src/synthesis/composition.rs` (new)

```rust
pub fn try_composition_solve(
    problem: &Problem,
    fn_name: &str,
) -> Option<SolveResult> {
    if !problem.composition_allowed {
        return None;
    }
    
    // Phase 1: Pattern recognition on I/O examples
    let patterns = detect_composition_patterns(problem);
    
    if patterns.is_empty() {
        return None;
    }
    
    // Phase 2: Try each pattern
    for pattern in patterns {
        if let Some(code) = emit_composed_function(fn_name, &pattern, problem) {
            if verify_problem_code_strict(problem, &code).is_ok() {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "composition".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                });
            }
        }
    }
    
    None
}

fn detect_composition_patterns(problem: &Problem) -> Vec<CompositionPattern> {
    // Analyze examples to determine which patterns fit
    // e.g., "input array shrinks → likely filter"
    //       "input/output same size → likely map"
    //       "input array → scalar → likely reduce"
    vec![]  // TODO
}

fn emit_composed_function(
    fn_name: &str,
    pattern: &CompositionPattern,
    problem: &Problem,
) -> Option<String> {
    // Generate function call chain based on pattern
    None  // TODO
}
```

### 2.4 Search Teacher Integration

**File**: `nsynth/src/solver/search.rs` (extend `SEARCH_CANDIDATES`)

Add `search_composition` teacher AFTER enumerative but BEFORE gradient:
```rust
pub const SEARCH_CANDIDATES: &[SearchTeacher] = &[
    // ... existing teachers ...
    search_composition,  // NEW: composition pattern recognizer
    // ... more teachers ...
];
```

---

## 3. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goal**: Get the infrastructure in place, start with **INLINE composition** (flatten all calls into one function body).

1. **Composition library** (`composition_lib.rs`)
   - Define `CompositionTemplate` struct
   - Hand-code 9 library templates (filter, map, reduce, etc.)
   - Each template includes Mog code (verified independently)

2. **Problem metadata** (`benchmark.rs`)
   - Add `composition_allowed: bool` field
   - Add `composition_hints: Vec<CompositionPattern>` field
   - Update all factory functions to set `composition_allowed = true` where sensible

3. **Pattern recognizer** (`composition.rs`)
   - Implement `detect_composition_patterns(examples)`:
     - **Filter**: Input array size > output array size
     - **Map**: Input size == output size
     - **Reduce**: Input array → scalar output
     - **Scan**: Input array → same-size output array
     - **Sort**: Output is sorted version of input
   - Return ordered list of candidates (best first)

4. **Inlined codegen** (`composition.rs`)
   - `emit_inline_filter(arr, predicate)` → inlined filter loop
   - `emit_inline_map(arr, transform)` → inlined map loop
   - `emit_inline_reduce(arr, op)` → inlined accumulation loop
   - NO separate function calls yet — just inline the patterns

5. **Teacher integration** (`search.rs`)
   - Add `search_composition` to `SEARCH_CANDIDATES`
   - Gate on `problem.composition_allowed`
   - Measure impact on existing benchmarks (should be neutral)

**Deliverable**: 5-8 composition benchmarks pass (filter_sum, map_double, reduce_min, etc.)

### Phase 2: Call Chains (Week 1.5)

**Goal**: Emit actual function calls (not inlined), building on inlined foundation.

1. **Codegen with calls** (`composition.rs`)
   - `emit_composed_function_with_calls()`
   - Generates helper functions for each stage
   - Main function calls them in sequence
   - Each helper has a clear signature

2. **Verification of composed code**
   - Call `verify_problem_code_strict()` on the generated multi-function code
   - Test that the runtime can execute helper function calls

3. **Composition benchmarks** (extend `benchmark.rs`)
   - `double_and_max(arr) -> i64`: map(double) → reduce(max)
   - `filter_sum_positive(arr) -> i64`: filter(>0) → sum
   - `count_unique(arr) -> i64`: sort → scan for changes → count
   - `deduplicate_then_count(arr) -> i64`: unique filter → count

**Deliverable**: 8+ benchmarks with multi-function calls verified and passing

### Phase 3: Advanced Patterns (Week 2)

**Goal**: Support more complex compositions, cross-problem inference.

1. **Nested composition**
   - `filter(map(arr, transform), predicate)` → three-stage chain
   - Codegen handles arbitrary depth

2. **Higher-order patterns**
   - `scan(arr, op) + take(n)` → partial cumulative result
   - `group_by(arr, key_fn)` → grouped iteration

3. **Inference from reference code**
   - Extract composition patterns from `problem.reference_code`
   - Match against library templates
   - Use as hint for solver

4. **Cross-problem learned biases** (optional)
   - Record which compositions work for which problem shapes
   - Replay successful compositions for new problems

**Deliverable**: 12+ composition benchmarks, at least 3 nested calls per composition

### Phase 4: Integration & Optimization (Week 2.5)

**Goal**: Full integration, cost modeling, search space pruning.

1. **Cost model**
   - Track composition cost (pattern recognition, codegen, verify)
   - Compare vs. monolithic synthesis time
   - Only pursue composition if time budget allows

2. **Search pruning**
   - Skip composition if no pattern detected with >75% confidence
   - Cache pattern detection results
   - Estimate verify time before trying composition

3. **Regression suite**
   - Ensure existing 105 factories still solve (composition_allowed=false)
   - New benchmarks all pass with composition
   - No performance regression on non-composition problems

4. **Paper section** (optional)
   - "Compositional Program Synthesis via Verified Function Libraries"
   - Show composition solves problems 2-3x faster than monolithic synthesis
   - Demonstrate generalization: one problem's composition works for similar problems

**Deliverable**: All tests green, composition integrated into main solver pipeline, 2-5 page paper section

---

## 4. Challenge Analysis

### Challenge 1: **Predicate Functions & Higher-Order Logic**

**Problem**: `filter(arr, predicate)` requires representing the predicate itself as a synthesizable parameter.

**Solution Options**:
- **Option A (Inlining)**: Don't emit predicates as separate functions; inline them as conditions.
  - Simplest, no higher-order synthesis needed
  - Limited reuse (predicate tied to filter call)
  - **Recommended for Phase 1**

- **Option B (Template Specialization)**: Generate specialized versions of filter for specific predicates.
  - `filter_positive(arr) -> [i64]` baked in, not parameterized
  - More code generation, more verification
  - Better for performance

- **Option C (Closure Synthesis)**: Synthesize the predicate as a separate closure/function.
  - Most flexible, allows arbitrary predicates
  - Requires synthesizer to reason about closures
  - **Defer to Phase 3+**

**Recommendation**: Use Option A for Phase 1 (inline predicates), upgrade to Option B (specialization) in Phase 2.

### Challenge 2: **Function Signature Matching**

**Problem**: How does the synthesizer know which library functions can be called in sequence?

**Solution**: Maintain a signature registry.
```rust
// Each composition template declares input/output types
struct LibraryFunction {
    input_types: Vec<Type>,    // e.g. [Array, i64] for filter
    output_type: Type,         // Array or Scalar
    pattern: CompositionPattern,
}

// Before emitting a call, verify output_type of f1 matches input_type of f2
fn signatures_compatible(f1: &LibraryFunction, f2: &LibraryFunction) -> bool { ... }
```

### Challenge 3: **Verification of Multi-Function Programs**

**Problem**: The runtime may not support nested function calls or may require special handling.

**Solution**:
- Test the runtime with simple call chains first
- Add a `supports_function_calls` capability flag
- Fall back to inlining if runtime doesn't support calls
- Phase 2 should verify this works end-to-end

### Challenge 4: **Search Space Explosion**

**Problem**: With M library functions, and N composition depths, there are M^N possible chains to try.

**Solution**:
1. **Early termination**: Stop after finding first verified composition
2. **Pattern-guided pruning**: Only consider chains that match detected patterns
3. **Signature matching**: Filter chains that have type mismatches
4. **Confidence-based ordering**: Try highest-confidence patterns first

**Budget**: Composition search should take <5s total per problem (vs. 10-30s for gradient).

---

## 5. Benchmark Specifications

### Phase 1 Benchmarks (Inline Composition)

```
fn filter_sum_positive(arr: [i64]) -> i64 {
  // Sum of positive elements
  Examples: [1,-2,3,-4,5] -> 9
}

fn filter_count_even(arr: [i64]) -> i64 {
  // Count even elements
  Examples: [1,2,3,4,5,6] -> 3
}

fn map_double_sum(arr: [i64]) -> i64 {
  // Sum of doubled elements
  Examples: [1,2,3] -> 12
}

fn map_square_max(arr: [i64]) -> i64 {
  // Max of squared elements
  Examples: [1,-3,2] -> 9
}

fn reduce_product(arr: [i64]) -> i64 {
  // Product of elements
  Examples: [2,3,4] -> 24
}

fn reduce_min(arr: [i64]) -> i64 {
  // Minimum element
  Examples: [5,2,8,1] -> 1
}

fn reduce_gcd_all(arr: [i64]) -> i64 {
  // GCD of all elements
  Examples: [12,18,24] -> 6
}

fn scan_cumsum(arr: [i64]) -> [i64] {
  // Cumulative sum
  Examples: [1,2,3] -> [1,3,6]
}
```

### Phase 2 Benchmarks (Function Calls)

```
fn double_and_max(arr: [i64]) -> i64 {
  // Max of doubled elements (via map + reduce)
  Examples: [1,2,3] -> 6
}

fn filter_sum_positive(arr: [i64]) -> i64 {
  // Sum of positive elements (via filter + reduce)
  Examples: [1,-2,3,-4,5] -> 9
}

fn count_unique(arr: [i64]) -> i64 {
  // Count unique elements (via sort + scan)
  Examples: [1,2,1,3,2] -> 3
}

fn deduplicate_then_count(arr: [i64]) -> i64 {
  // Count distinct elements (multi-call)
  Examples: [1,2,1,3] -> 3
}
```

---

## 6. Integration Checklist

- [ ] Composition library templates defined (`composition_lib.rs`)
- [ ] Problem metadata extended (`composition_allowed`, `composition_hints`)
- [ ] Pattern recognizer implemented (detect filter/map/reduce)
- [ ] Inlined codegen working (no function calls yet)
- [ ] `search_composition` teacher added to solver pipeline
- [ ] Phase 1 benchmarks (8) all passing
- [ ] Multi-function call codegen implemented
- [ ] Phase 2 benchmarks (4+) all passing
- [ ] Regression tests confirm no impact on existing 105 factories
- [ ] Composition cost model measured
- [ ] Documentation updated (this file + code comments)
- [ ] Paper section drafted (optional)

---

## 7. Open Questions

1. **Should composition be enabled by default or opt-in?**
   - Default: yes, `composition_allowed = true` for all factories
   - Reason: Cost is low if no pattern detected; benefit is high when it works

2. **Should library functions themselves be synthesized, or always hand-coded?**
   - Hand-coded is safer (verified once, reused many times)
   - Defer synthesis of library functions to future work

3. **How do we handle edge cases (empty arrays, division by zero) in composed calls?**
   - Library templates must handle all cases
   - Verification ensures safety before returning

4. **Can composition help with the "template slowdown" problem?**
   - Currently template fallback takes 343s for 3 problems
   - If composition can solve some of those, it wins
   - Measure in Phase 4

---

## 8. Estimated Effort & Timeline

| Phase | Duration | Effort | Risk |
|-------|----------|--------|------|
| 1: Foundation | 1 week | 40h | Medium (verification) |
| 2: Call chains | 1.5 days | 12h | Low (builds on Phase 1) |
| 3: Advanced patterns | 1 week | 35h | High (complex codegen) |
| 4: Integration | 2.5 days | 20h | Low (mostly testing) |
| **Total** | **2.5 weeks** | **107h** | **Medium** |

**Critical Path**: Phase 1 (infrastructure) → Phase 2 (verification) → then Phase 3/4 in parallel.

**Success Criteria**:
- Phase 1: 8 benchmarks solve via inline composition
- Phase 2: 4+ benchmarks solve via function calls
- Phase 4: Zero regression, 105 existing factories still pass, composition solves 3-5% of new problems faster

---

## 9. References & Related Work

- **FlashFill** (Gulwani 2011): string-by-example via syntax-directed templates → motivation for composition library
- **Program synthesis via library functions** (Ellis et al. 2015): DreamCoder → uses library to reduce search space
- **Higher-order synthesis**: lambda calculus, church encodings → deferred to Phase 3+
- **Function call optimization**: tail-call elimination, inlining heuristics → Phase 4 optimization

---

## Appendix: Pseudocode for Phase 1

### Pattern Detection

```
fn detect_composition_patterns(problem: Problem) -> Vec<Pattern> {
  patterns = []
  examples = problem.examples
  
  // Check for filter (array shrinkage)
  if input_is_array(examples[0]) {
    output = expected_output(examples[0])
    if output_is_array(output) {
      input_size = array_size(input)
      output_size = array_size(output)
      if output_size < input_size {
        patterns.push(Filter)
      }
    }
  }
  
  // Check for reduce (array → scalar)
  if input_is_array(examples[0]) {
    output = expected_output(examples[0])
    if output_is_scalar(output) {
      patterns.push(Reduce)
    }
  }
  
  // Check for map (array same size)
  if input_is_array(examples[0]) {
    output = expected_output(examples[0])
    if output_is_array(output) {
      if array_size(input) == array_size(output) {
        patterns.push(Map)
      }
    }
  }
  
  return patterns
}
```

### Inline Codegen for Filter

```
fn emit_inline_filter(arr_name: &str, fn_name: &str, predicate: &str) -> String {
  format!(r#"
    fn {fn_name}(arr: [i64]) -> i64 {{
      result: i64 = 0
      i: i64 = 0
      while i < len(arr) {{
        if {predicate}(arr[i]) {{
          result = result + arr[i]
        }}
        i = i + 1
      }}
      return result
    }}
  "#)
}
```

---

**Author**: Bobby Price  
**Date**: June 15, 2026  
**Version**: 1.0 (Design phase)
