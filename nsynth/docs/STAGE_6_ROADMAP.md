# Stage 6 Implementation Roadmap

**Objective**: Implement function composition synthesis in nsynth, breaking monolithic function synthesis into reusable library-based pipelines.

**Timeline**: 2.5 weeks (estimated 107 hours)

---

## Sprint Structure

### Sprint 1: Foundation (Days 1-5)

#### Day 1-2: Composition Library & Metadata

**Goal**: Define the library of reusable functions and extend Problem struct.

**Deliverables**:
1. `nsynth/src/synthesis/composition_lib.rs` (new file, ~300 lines)
   - `CompositionTemplate` struct with (name, signature, code, pattern)
   - `CompositionPattern` enum (Filter, Map, Reduce, Scan, Sort, Zip, Transpose, FindAll)
   - Hard-coded templates for 9 library functions
   - Unit tests verifying each template compiles and passes basic I/O

2. `nsynth/src/benchmark.rs` (extend existing)
   - Add `composition_allowed: bool` field to `Problem` struct
   - Add `composition_hints: Vec<CompositionPattern>` field
   - Audit all 105 factory functions:
     - Set `composition_allowed = true` for array/reduce problems
     - Set `composition_allowed = false` for string/struct/external problems
   - Update `Default` impl or builder functions

3. `nsynth/src/synthesis/mod.rs` (extend)
   - Add module declaration: `pub mod composition_lib;`
   - Expose `CompositionTemplate`, `CompositionPattern`

4. **Regression Check**
   - Existing benchmark still 105/105 (composition_allowed=false defaults to skipping)
   - No solver behavior changes yet

**Testing**:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn composition_templates_compile() {
        for template in COMPOSITION_TEMPLATES {
            let _ = parse_mog(&template.code);  // or verify_syntax
        }
    }
    
    #[test]
    fn filter_template_correct() {
        // Run filter template on simple I/O
        let arr = vec![1, -2, 3, -4];
        let result = /* execute template */ ;
        assert_eq!(result, vec![1, 3]); // filtered positive
    }
}
```

**Effort**: 12 hours

---

#### Day 3: Pattern Recognizer

**Goal**: Detect which composition patterns fit a problem's I/O examples.

**Deliverables**:
1. `nsynth/src/synthesis/composition.rs` (new file, ~400 lines)
   - `fn detect_composition_patterns(problem: &Problem) -> Vec<PatternMatch>`
   - `struct PatternMatch { pattern: CompositionPattern, confidence: f32 }`
   - Implementation:
     - **Filter**: count(output_array) < count(input_array) → confidence 0.8
     - **Map**: count(output_array) == count(input_array) AND all outputs transform input element → 0.7
     - **Reduce**: input_array, output_scalar → 0.9
     - **Scan**: input_array, output_array (same size), cumulative property → 0.8
     - **Sort**: output_array is sorted(input_array) → 0.85
   - Return patterns ordered by confidence (best first)

2. Unit tests
   - `test_detect_filter_pattern()` — [1,-2,3] → [1,3], expect Filter
   - `test_detect_map_pattern()` — [1,2,3] → [2,4,6], expect Map
   - `test_detect_reduce_pattern()` — [1,2,3,4] → 10, expect Reduce
   - `test_no_pattern_detected()` — complex logic, expect empty list

**Effort**: 10 hours

---

#### Day 4: Inline Codegen (Phase 1)

**Goal**: Generate verified code with inlined composition patterns (no function calls yet).

**Deliverables**:
1. `nsynth/src/synthesis/composition.rs` (extend)
   - `fn emit_inline_filter(filter_logic: &str) -> String`
     - Returns Mog function with inlined filter loop
   - `fn emit_inline_map(transform_logic: &str) -> String`
     - Returns Mog function with inlined map loop
   - `fn emit_inline_reduce(init: i64, op: &str) -> String`
     - Returns Mog function with inlined reduce loop
   - `fn emit_inline_scan(init: i64, op: &str) -> String`
     - Returns Mog function with inlined scan loop
   - Each function returns fully verified Mog code string

2. `fn try_composition_solve_inline(problem: &Problem) -> Option<SolveResult>`
   - Call `detect_composition_patterns()`
   - For top pattern, call appropriate emit_inline_* function
   - Verify code with `verify_problem_code_strict(problem, &code)`
   - Return SolveResult if verification passes

3. Unit tests
   - `test_emit_inline_filter()` — check generated code syntax
   - `test_emit_inline_reduce()` — verify behavior on simple arrays
   - `test_composition_solve_filter_sum()` — end-to-end on actual problem

**Effort**: 15 hours

---

#### Day 5: Teacher Integration & Benchmarks

**Goal**: Wire composition into solver pipeline, add Phase 1 benchmarks.

**Deliverables**:
1. `nsynth/src/solver/search.rs` (extend)
   - `pub fn search_composition(problem: &Problem) -> Option<SolveResult>`
   - Wrapper calling `try_composition_solve_inline()`
   - Add to `SEARCH_CANDIDATES` between enumerative and gradient stages
   - Gate on `problem.composition_allowed`

2. `nsynth/src/benchmark.rs` (add factories)
   - 8 Phase 1 benchmark factories:
     - `filter_sum_positive` — sum positive elements
     - `filter_count_even` — count even elements
     - `map_double_sum` — sum doubled elements
     - `map_square_max` — max of squared elements
     - `reduce_product` — product of all elements
     - `reduce_min` — minimum element
     - `scan_cumsum` — cumulative sum
     - `count_abs_greater_five` — count elements > 5 in absolute value

3. Benchmark runner
   - `cargo test --lib benchmark::composition` — runs only composition factories
   - Expected: 8/8 passing via `search_composition` teacher

4. Regression suite
   - Existing 95/95 factories still pass
   - `composition_allowed = false` for non-array problems (strings, structs, external)
   - Measure solve time per problem — should be near-identical

**Effort**: 13 hours

---

**Sprint 1 Total**: 50 hours

---

### Sprint 2: Multi-Function Calls (Days 6-8)

#### Day 6: Call-Chain Codegen

**Goal**: Emit separate functions and call them in sequence (no inlining).

**Deliverables**:
1. `nsynth/src/synthesis/composition.rs` (extend)
   - `struct ComposedProgram { functions: Vec<FunctionDef>, main_fn: String }`
   - `fn emit_composed_function_with_calls(pattern: &PatternMatch, problem: &Problem) -> Option<String>`
     - For filter+reduce: emit `filter_helper()` + `reduce_helper()` + main calls both
     - Each helper is a standalone Mog function with full signature
     - Main function calls them in sequence, assigning to temp variables
     - Example:
       ```mog
       fn _filter_helper(arr: [i64]) -> [i64] { ... }
       fn _reduce_helper(arr: [i64]) -> i64 { ... }
       fn filter_sum_positive(arr: [i64]) -> i64 {
         filtered: [i64] = _filter_helper(arr)
         return _reduce_helper(filtered)
       }
       ```

2. Unit tests
   - `test_emit_composed_filter_reduce()` — check structure and syntax
   - `test_composed_code_verifies()` — run through verify_problem_code_strict

**Effort**: 8 hours

---

#### Day 7: Runtime Support & Verification

**Goal**: Ensure the Mog runtime can execute multi-function code; add Phase 2 benchmarks.

**Deliverables**:
1. `nsynth/src/runtime.rs` (audit + extend if needed)
   - Verify function calls work end-to-end
   - If runtime doesn't support nested calls: fall back to inlining automatically
   - Add `supports_function_calls()` capability check

2. `nsynth/src/synthesis/composition.rs` (extend)
   - `fn try_composition_solve_with_calls(problem: &Problem) -> Option<SolveResult>`
   - Call `emit_composed_function_with_calls()`
   - Verify and return result
   - Update `try_composition_solve_inline()` as fallback if calls don't work

3. `nsynth/src/benchmark.rs` (add Phase 2 factories)
   - 4 Phase 2 benchmarks:
     - `double_and_max` — map(double) + reduce(max)
     - `filter_sum_positive` — filter(>0) + reduce(sum)
     - `count_unique` — sort + scan-count-changes
     - `deduplicate_then_count` — dedup + count

4. Regression
   - All Phase 1 benchmarks still pass
   - All existing 95 factories still pass

**Effort**: 10 hours

---

#### Day 8: Integration & Testing

**Goal**: Full integration into solver, comprehensive regression testing.

**Deliverables**:
1. `nsynth/src/solver/search.rs` (refine)
   - Update `search_composition()` to try both inline AND calls
   - Prefer calls if they verify, fall back to inline
   - Measure time: should be <2s per problem

2. `nsynth/src/synthesis/mod.rs`
   - Add module declaration: `pub mod composition;`
   - Ensure no duplicate modules

3. Comprehensive tests
   - `test_composition_benchmarks` — all 12 Phase 1+2 benchmarks pass
   - `test_no_regression_existing_factories` — 95 existing still pass
   - `test_composition_allowed_false_skips` — non-array problems skip composition
   - `test_composition_cost_under_budget` — each composition solve <5s

4. Documentation update
   - Inline code comments for pattern detection logic
   - Update `docs/ARCHITECTURE.md` to mention `composition` teacher

**Effort**: 8 hours

---

**Sprint 2 Total**: 26 hours

---

### Sprint 3: Advanced Patterns (Days 9-12) [Optional / Phase 3]

#### Day 9-10: Nested Composition & Type Checking

**Goal**: Support 3+ function call chains, ensure type safety.

**Deliverables**:
1. `nsynth/src/synthesis/composition.rs` (extend)
   - `struct FunctionSignature { inputs: Vec<Type>, output: Type }`
   - `fn signatures_compatible(f1: &Signature, f2: &Signature) -> bool`
   - `fn emit_composed_chain(patterns: Vec<Pattern>) -> Option<String>`
     - Validate type compatibility between consecutive functions
     - Emit multi-step chain (3-4 functions calling each other)

2. Unit tests
   - `test_type_check_filter_to_reduce()` — Array → Array → i64 compatible
   - `test_type_check_incompatible()` — Scalar → Array invalid
   - `test_nested_triple_chain()` — map + filter + reduce works

**Effort**: 12 hours (deferred if time is tight)

---

#### Day 11: Learned Bias Acceleration

**Goal**: Cache successful compositions to speed up similar problems.

**Deliverables**:
1. `nsynth/src/synthesis/learned_biases.rs` (extend to include composition)
   - New bias type: `CompositionBias { problem_shape_hash: u64, pattern: Vec<Pattern>, code: String }`
   - When a composition succeeds, record it
   - On future similar problems, replay the successful pattern

2. Unit tests
   - `test_composition_bias_replay()` — two similar problems use same pattern

**Effort**: 10 hours (optional)

---

#### Day 12: Paper Section & Polish

**Goal**: Write up the composition approach, final regression testing.

**Deliverables**:
1. `docs/STAGE_6_COMPOSITION_RESULTS.md` (new)
   - Overview of composition approach
   - Benchmark results (12+ problems, solve times, coverage)
   - Comparison to monolithic synthesis
   - Examples of composed vs inlined code

2. Paper section
   - "Module Composition via Verified Function Libraries"
   - 3-5 pages
   - Figures: pattern detection flowchart, codegen example, benchmark table

3. Final regression
   - All 105+ benchmarks pass (105 existing + 12 composition)
   - Verify no performance regression
   - Ensure composition_allowed defaults correctly

**Effort**: 15 hours

---

**Sprint 3 Total**: 37 hours (optional; can defer to future)

---

**Overall Effort Estimate**:
- **Must-have (Sprints 1-2)**: 76 hours (1.9 weeks)
- **Nice-to-have (Sprint 3)**: +37 hours (total 2.5 weeks)

---

## Detailed Work Breakdown

### File-by-File Checklist

#### New Files to Create:
- [ ] `nsynth/src/synthesis/composition_lib.rs` — Library templates (300 LOC)
- [ ] `nsynth/src/synthesis/composition.rs` — Pattern detection & codegen (600 LOC)
- [ ] `tests/composition_tests.rs` (optional) — Standalone composition unit tests (200 LOC)

#### Files to Modify:
- [ ] `nsynth/src/benchmark.rs`
  - Add `composition_allowed` field to `Problem`
  - Add `composition_hints` field
  - Add 12 factory functions for composition benchmarks
  - Update existing ~105 factories to set `composition_allowed`

- [ ] `nsynth/src/synthesis/mod.rs`
  - Add `pub mod composition_lib;`
  - Add `pub mod composition;`

- [ ] `nsynth/src/solver/search.rs`
  - Add `pub fn search_composition()` function
  - Add to `SEARCH_CANDIDATES` between stage 3 and gradient
  - Update tests to verify composition teacher ordering

- [ ] `nsynth/src/runtime.rs`
  - Audit function call support
  - Add `supports_function_calls()` if needed

- [ ] `docs/ARCHITECTURE.md`
  - Add section: "Composition synthesis (`search_composition` teacher)"
  - Link to STAGE_6_COMPOSITION.md

#### Test Updates:
- [ ] `nsynth/src/synthesis/mod.rs` — add composition module tests
- [ ] `nsynth/src/solver/tests/` — add composition regression tests
- [ ] Create `tests/composition_integration.rs` — end-to-end tests

---

## Milestone Checklist

### Milestone 1: Metadata & Library (Day 2 EOD)
- [ ] `composition_lib.rs` compiles, 9 templates defined
- [ ] `Problem` struct extended, all factories updated
- [ ] No regressions, 95/95 existing benchmarks pass

### Milestone 2: Pattern Detection (Day 3 EOD)
- [ ] Pattern detection working on sample problems
- [ ] Unit tests for filter/map/reduce/scan patterns
- [ ] Confidence scoring correct

### Milestone 3: Inline Codegen (Day 5 EOD)
- [ ] 8 Phase 1 benchmarks added
- [ ] Inline composition solving all 8 benchmarks
- [ ] No regressions on existing 95

### Milestone 4: Multi-Function Calls (Day 8 EOD)
- [ ] 4 Phase 2 benchmarks added
- [ ] Function call codegen working
- [ ] All 12 composition benchmarks pass via `search_composition`
- [ ] Existing 95 benchmarks still pass

### Milestone 5: Regression & Integration (Sprint 2 End)
- [ ] All 105+ benchmarks pass
- [ ] Composition cost <5s per problem
- [ ] No existing performance regression

### Milestone 6: Advanced Patterns (Sprint 3, Optional)
- [ ] Nested compositions working (3+ function chains)
- [ ] Type checking correct
- [ ] 4+ additional benchmarks added

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Runtime doesn't support function calls | Medium | High | Fallback to inlining; test early (Day 7) |
| Pattern detection overfits to training set | Low | Medium | Validate on holdout examples; require >0.7 confidence |
| Composition search space explodes | Medium | Medium | Limit patterns to top 2-3; early termination |
| Regression on existing benchmarks | Low | High | Run full benchmark suite after each change |
| Type checking too strict | Medium | Low | Start permissive; tighten incrementally |
| Verify timeout on multi-function code | Low | High | Add timeout of 10s per verify call; bail if exceeded |

---

## Success Criteria

### Hard Criteria (Must Have):
1. All 105 existing benchmarks still pass (no regression)
2. All 12 Phase 1+2 composition benchmarks pass
3. Composition cost <5s per problem average
4. Code compiles with zero compiler warnings
5. All unit tests pass

### Soft Criteria (Nice to Have):
1. Composition solves 3-5% of problems faster than monolithic synthesis
2. Learned bias acceleration working (future replay)
3. Paper section written and integrated
4. Advanced patterns (nested chains) working
5. Type checking comprehensive

---

## Timeline Visualization

```
Sprint 1 (Days 1-5):  HHHHH (50 hours)
  Day 1-2: Lib + metadata
  Day 3: Pattern detection
  Day 4: Inline codegen
  Day 5: Teacher + benchmarks

Sprint 2 (Days 6-8):  HHH (26 hours)
  Day 6: Call-chain codegen
  Day 7: Runtime verification
  Day 8: Integration

Sprint 3 (Days 9-12): HHHH (37 hours, optional)
  Day 9-10: Nested composition
  Day 11: Learned bias
  Day 12: Paper + polish

Total: 113 hours ≈ 2.5 weeks (assuming 40-50 hrs/week)
```

---

## Post-Implementation Tasks

After both sprints complete:

1. **Performance profiling**
   - Which composition patterns are fastest?
   - Which are slowest (bottleneck)?
   - Optimize hot paths

2. **Generalization study**
   - Can a composition learned for problem A help with problem B?
   - Build learned-bias bank for future uses

3. **Paper preparation**
   - Results section: 12+ benchmarks, solve time comparison
   - Ablation: inline vs. calls, with/without type checking
   - Generalization on library function reuse

4. **Future: Function synthesis**
   - Phase 3 → synthesize custom library functions?
   - Phase 4 → user-defined composition patterns?

---

**Owner**: Bobby Price  
**Last Updated**: June 15, 2026  
**Status**: Ready for implementation
