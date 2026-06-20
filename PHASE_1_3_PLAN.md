# Phase 1.3: Probabilistic Synthesis Integration - Detailed Plan

## Current State Analysis

### ✅ What's Working
1. **linguigenesis-core**: Compiles after cfg fix (phonology-dependent modules behind feature gate)
2. **Lemma Registry**: linguigenesis uses `lemma` field in Entity + `lemma_index` in Registry - nsynth bridge already compatible
3. **Multi-Language CLI**: `--target` flag works for basic Rust→target conversion
4. **Prob Module**: Complete distribution sampling, MCMC inference, variational inference

### ❓ What Needs Investigation
1. **Solver Pipeline Structure**: How to add probabilistic teacher alongside existing teachers
2. **Problem Detection**: How to identify when probabilistic synthesis is appropriate
3. **Integration Points**: Where in solve_problem() to invoke probabilistic path

---

## Phase 1.3 Tasks

### Task 1.3.1: Investigate Solver Pipeline (2 hours)
**Goal**: Understand current pipeline structure and identify integration points

**Actions**:
1. Map `solve_problem_inner()` stages
2. Identify where teachers are invoked (search_families.rs)
3. Check if there's already a "probabilistic" detection mechanism
4. Understand how `--probabilistic` flag should route

**Files to Read**:
- `src/solver/pipeline.rs` - main solve logic
- `src/solver/search_families.rs` - teacher invocation
- `src/solver/search.rs` - search execution
- `src/main.rs` - flag parsing

**Deliverable**: Integration point recommendation

---

### Task 1.3.2: Detect Probabilistic Problems (3 hours)
**Goal**: Identify when examples suggest probabilistic synthesis

**Approach**:
Probabilistic problems have these characteristics:
- **Uncertainty**: Examples show non-deterministic patterns
- **Distributions**: I/O suggests random processes (coin flips, measurements with noise)
- **Inference**: Multiple valid hypotheses consistent with data

**Detection Heuristics**:
```rust
fn is_probabilistic_problem(problem: &Problem) -> bool {
    // Check 1: Examples have conflicting I/O (non-deterministic)
    let has_conflicts = has_conflicting_examples(&problem.examples);

    // Check 2: Signature suggests probability (return f64, no clear function)
    let has_uncertain_output = problem.signature.contains("f64") &&
                              !has_exact_pattern(&problem.examples);

    // Check 3: Examples suggest sampling (many similar inputs, varied outputs)
    let suggests_sampling = suggests_sampling_process(&problem.examples);

    has_conflicts || has_uncertain_output || suggests_sampling
}
```

**Files to Create**:
- `src/solver/probabilistic.rs` - detection and routing

**Tests to Create**:
- Coin flip detection: `[true, false, true]` → probabilistic
- Noisy measurements: `[(1.0, 1.01), (1.0, 0.99)]` → probabilistic
- Deterministic: `[1, 2, 3]` → `1, 2, 3` → NOT probabilistic

---

### Task 1.3.3: Implement Probabilistic Teacher (4 hours)
**Goal**: Create teacher that generates probabilistic programs

**Approach**:
The probabilistic teacher should:
1. Detect distribution type from examples
2. Synthesize program with `sample()` and `observe()` calls
3. Use MCMC to learn parameters
4. Generate executable Rust code

**Implementation**:
```rust
// src/solver/probabilistic.rs

pub struct ProbabilisticTeacher {
    config: ProbConfig,
}

impl ProbabilisticTeacher {
    pub fn teach(problem: &Problem) -> SolveResult {
        // 1. Analyze examples to infer distribution
        let dist_type = infer_distribution_type(&problem.examples);

        // 2. Create probabilistic model
        let model = create_probabilistic_model(dist_type, &problem.examples);

        // 3. Run inference to learn parameters
        let params = run_mcmc_inference(&model);

        // 4. Generate code
        let code = generate_probabilistic_code(&model, params);
        ...
    }
}
```

**Distribution Detection**:
- Bernoulli: Bool outputs, binary pattern
- Normal: Float outputs, bell-shaped distribution
- Categorical: Int outputs in range [0, k]
- Poisson: Int outputs, count data

**Files to Modify**:
- `src/solver/probabilistic.rs` - create new module
- `src/solver/mod.rs` - export probabilistic module

---

### Task 1.3.4: Wire into Pipeline (2 hours)
**Goal**: Add probabilistic path to solve_problem_inner()

**Changes**:
1. Add `--probabilistic` flag to main.rs
2. Add detection call in solve_problem_inner()
3. Route to probabilistic teacher when appropriate
4. Fallback to normal pipeline on failure

**Pipeline Integration**:
```rust
// src/solver/pipeline.rs

fn solve_problem_inner(problem: &Problem) -> SolveResult {
    // ... existing checks ...

    // NEW: Probabilistic path
    if should_try_probabilistic(problem) {
        eprintln!("[solve] trying probabilistic synthesis");
        let prob_result = super::probabilistic::solve_probabilistic_problem(problem);
        if prob_result.success {
            return prob_result;
        }
        eprintln!("[solve] probabilistic failed, continuing to normal pipeline");
    }

    // ... existing enumeration and search ...
}
```

**Files to Modify**:
- `src/main.rs` - add `--probabilistic` flag
- `src/solver/pipeline.rs` - add probabilistic routing
- `src/solver/mod.rs` - export solve_probabilistic_problem

---

### Task 1.3.5: Code Generation (3 hours)
**Goal**: Generate executable Rust code with probabilistic primitives

**Code Template**:
```rust
// Example: Coin flip with learned bias
fn coin_flip() -> bool {
    // Learned bias: 0.55
    let p = 0.55;
    let u: f64 = rand::random();
    u < p
}

// Example: Noisy measurement
fn measure(value: f64) -> f64 {
    // Learned noise: N(0, 0.1)
    let noise = rand::distributions::Normal::new(0.0, 0.1).sample(&mut rand::thread_rng());
    value + noise
}
```

**Implementation**:
```rust
// src/prob/codegen.rs

pub fn generate_probabilistic_code(
    dist: ProbDistribution,
    params: &[f64],
    fn_name: &str,
) -> String {
    match dist {
        ProbDistribution::Bernoulli { .. } => {
            format!(r#"use rand::Rng;

fn {fn_name}() -> bool {{
    let p = {p};
    let mut rng = rand::thread_rng();
    let u: f64 = rng.gen();
    u < p
}}"#, p=params[0])
        }
        // ... other distributions
    }
}
```

**Files to Create**:
- `src/prob/codegen.rs` - code generation module
- `src/prob/mod.rs` - export codegen

---

### Task 1.3.6: Testing & Validation (2 hours)
**Goal**: Verify probabilistic synthesis works end-to-end

**Test Cases**:
1. **Coin Flip**: Examples `[true, false, true, true, false]` → Learn bias → Sample
2. **Noisy Sensor**: Inputs `[1.0, 2.0, 3.0]`, Outputs `[1.1, 1.9, 3.05]` → Learn noise
3. **Dice Roll**: Examples `[1, 6, 3, 5, 2]` → Uniform[1,6]
4. **Count Data**: Examples `[3, 5, 4, 6, 2]` → Poisson(λ=4)

**Integration Tests**:
```rust
#[test]
fn test_probabilistic_coin_flip() {
    let examples = vec![
        Example { inputs: vec![], expected: Value::Bool(true) },
        Example { inputs: vec![], expected: Value::Bool(false) },
        // ... more examples
    ];
    let result = solve_probabilistic_problem(&problem);
    assert!(result.success);
    assert!(result.code.contains("rand::"));
}
```

**Files to Modify**:
- `src/prob/tests.rs` - integration tests
- `tests/integration_probabilistic.rs` - end-to-end tests

---

## Success Criteria

1. ✅ `--probabilistic` flag recognized in CLI
2. ✅ Probabilistic problems detected automatically
3. ✅ Probabilistic teacher generates working code
4. ✅ MCMC inference learns correct parameters
5. ✅ Generated code compiles and runs
6. ✅ Integration tests pass

---

## Dependencies & Risks

### Dependencies
- `src/prob` module already complete (distributions, MCMC, variational)
- Solver pipeline structure understood
- Detection heuristics need validation

### Risks
1. **False Positives**: Non-probabilistic problems routed to probabilistic path
   - Mitigation: Strong detection heuristics, fallback to normal pipeline
2. **MCMC Performance**: Inference too slow for interactive use
   - Mitigation: Configurable iterations, timeout guards
3. **Code Generation**: Generated code may not compile
   - Mitigation: Template-based generation with validation

---

## Estimated Timeline

| Task | Effort | Dependencies |
|------|--------|--------------|
| 1.3.1 Investigate Pipeline | 2h | None |
| 1.3.2 Detect Problems | 3h | 1.3.1 |
| 1.3.3 Implement Teacher | 4h | 1.3.1 |
| 1.3.4 Wire to Pipeline | 2h | 1.3.2, 1.3.3 |
| 1.3.5 Code Generation | 3h | 1.3.3 |
| 1.3.6 Testing | 2h | All above |
| **Total** | **16h** | |

---

## Next Steps

1. Start with Task 1.3.1: Read and map pipeline.rs
2. Create detection module (Task 1.3.2)
3. Implement teacher (Task 1.3.3)
4. Wire and test (Tasks 1.3.4-1.3.6)
