# Stage 4: Synthetic Time Argument Design

## Overview
Stage 4 extends nsynth to synthesize functions with **synthetic time arguments**: parameters injected by the synthesizer (not present in examples) that represent an external iteration index or recursion depth. This enables:
- **Loop-free closed forms**: `f(t) = t² + 2t + 1` instead of `for i in 0..n { ... }`
- **Sequences as functions**: Fibonacci `fib(t)` = value at index t
- **Timed behaviors**: Programs that depend on execution step/phase

## Problem Space

### Current Limitation
Today: `Problem { signature: "fn fibonacci(n: i64) -> i64", examples: [...], ... }`
- Examples tie n to the return value directly
- Loop-based solvers must unroll recursion inline (expensive, limited by depth)
- Closed-form sequences hard to discover without explicit "find formula" solvers

### Proposed Extension
New: `Problem { signature: "...", synthetic_args: ["t"], examples: [...], inferred_t_values: [0, 1, 2, ...], ... }`
- Same examples (no t column in input)
- Solver **infers** which t value produced each output
- Synthesis can emit `fn fib(t: i64) -> i64` and optimize for speed

### Use Cases

1. **Arithmetic/Fibonacci/Triangular**: Direct formula discovery
   - `triangular(t) → t*(t+1)/2`
   - `fib(t) → ⌊φᵗ / √5⌋` (Binet's formula)
   
2. **Simulation State**: Time-dependent behavior
   - `simulate_gravity(t) → position_at_step_t`
   - `game_state(t) → score_after_t_turns`
   
3. **Recurrence Relations**: Loop unrolling without explicit loop
   - `lucas(t) → L(t) = L(t-1) + L(t-2), L(0)=2, L(1)=1`

## Architecture

### 1. Problem Struct Extension

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
    // === NEW ===
    pub synthetic_args: Vec<String>,  // e.g. ["t"] for time argument
    // Optional: hints for t-value inference (if not inferrable from signature)
    pub synthetic_arg_ranges: Option<Vec<(i64, i64)>>, // e.g. [(0, 100)] for t ∈ [0, 100)
}
```

**Migration**:
- All existing `Problem` instances get `synthetic_args: vec![]` (no synthetic args)
- Backward compatible: examples/holdouts unchanged, no t column
- For Stage 4 problems: `synthetic_args: vec!["t".to_string()]`

### 2. Example Struct: No Change

**File**: `src/benchmark.rs`

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub inputs: Vec<Value>,
    pub expected: Value,
    // t values inferred elsewhere (not stored here to avoid bloat)
}
```

**Inference Strategy**:
- Given a problem with `synthetic_args: ["t"]`, solver builds a **t-mapping table**:
  ```
  Example 0: inputs=[], expected=0 → infer t=0
  Example 1: inputs=[], expected=1 → infer t=1
  Example 2: inputs=[], expected=1 → infer t=2 (ambiguous; Fibonacci)
  Example 3: inputs=[], expected=2 → infer t=3
  ```
- Two strategies (see §2.1–2.2):
  1. **Positional (default)**: t = index in examples
  2. **Discovery-based**: solve t from examples using reference code or heuristics

### 2.1 Positional Inference (Fast)

**When**: Synthetic arg appears "time-like" (no real inputs, strictly increasing/monotonic outputs)

```rust
fn infer_time_values_positional(problem: &Problem) -> Option<Vec<i64>> {
    if problem.synthetic_args.is_empty() {
        return None;
    }
    if !problem.examples.is_empty() && problem.examples[0].inputs.is_empty() {
        // All examples have no real inputs → time is the only parameter
        return Some((0..problem.examples.len() as i64).collect());
    }
    None
}
```

**Example**: Fibonacci
```
Example 0: () → 0   ⟹ t=0, fib(0)=0
Example 1: () → 1   ⟹ t=1, fib(1)=1
Example 2: () → 1   ⟹ t=2, fib(2)=1
Example 3: () → 2   ⟹ t=3, fib(3)=2
```

### 2.2 Discovery-Based Inference (Fallback)

**When**: Examples have real inputs + synthetic args (e.g., `problem(a, t) → result`)

```rust
fn infer_time_values_from_reference(
    problem: &Problem,
    reference_impl: &str,
) -> Result<Vec<i64>, String> {
    // Parse reference code
    // For each example, find t such that reference_impl(inputs, t) ≈ expected
    // Use binary search or parameter sweep
    todo!("Stage 4b: implement if mixed inputs + synthetic args needed")
}
```

**Deferred**: This is complex and not needed for initial use cases (pure sequences).

### 3. Signature Extension

**File**: `src/solver/signature.rs`

Current function signature: `fn fibonacci(n: i64) -> i64`
Stage 4 signature: `fn fibonacci(t: i64) -> i64`

The signature **declares the synthetic args**, allowing code generation to reference them:

```rust
pub(super) fn parse_synthetic_args(signature: &str) -> Vec<String> {
    // Extract param names that are declared synthetic
    // Strategy: look for params that match Problem.synthetic_args
    // For now, assume problem.synthetic_args is authoritative
    todo!("Match signature params against problem.synthetic_args")
}
```

**Code Generation Convention**:
- Real inputs first (a, b, c, ...)
- Synthetic inputs last (t, u, ...)
- Example: `fn mixed(a: i64, b: [i64], t: i64) -> i64` (t is synthetic)

### 4. Mog AST: Minimal Extension

**File**: `~/projects/mog/compiler/src/ast.rs` (if needed; likely unchanged)

**Strategy**: Don't add new AST nodes. Treat `t` as a regular parameter:
- `t` is declared in `FunctionDeclaration.params`
- `t` appears in expressions like normal: `t + 1`, `t * t`, `fib(t-1)`
- No special AST markers needed; synthesis rules enforce that t is only bound at entry

**Mog Functions with Time Args** (examples):

```mog
// Closed form
fn triangular(t: i64) -> i64 {
    return (t * (t + 1)) / 2;
}

// Loop over t (if closed form not found)
fn fib(t: i64) -> i64 {
    if t == 0 { return 0; }
    if t == 1 { return 1; }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= t {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}

// Closed form via discovered pattern
fn lucas(t: i64) -> i64 {
    // Discovered: L(t) = φᵗ + ψᵗ (where φ,ψ = golden ratio roots)
    // Approximated as closed form
    return ...;
}
```

### 5. Synthesis Stages & Routing

**File**: `src/solver/solver.rs` (Stage 4 added to pipeline)

```rust
pub fn solve_problem(problem: &Problem, ...) -> SolveResult {
    // Existing stages 1-3
    if let Some(result) = stage_1_exact_scalar(&problem) { return result; }
    if let Some(result) = stage_2_search(&problem) { return result; }
    if let Some(result) = stage_3_gradient(&problem) { return result; }
    
    // === NEW: Stage 4 ===
    if !problem.synthetic_args.is_empty() {
        if let Some(result) = stage_4_synthetic_time(&problem) { return result; }
    }
    
    // Fallback
    return solve_default(problem);
}

fn stage_4_synthetic_time(problem: &Problem) -> Option<SolveResult> {
    let t_values = infer_time_values(problem)?;
    
    // Sub-stages:
    // 4a. Closed-form formula discovery (polynomial regression, lookup tables)
    if let Some(code) = synthesize_closed_form(problem, &t_values) {
        return Some(verify_and_return(code, "stage_4_closed_form"));
    }
    
    // 4b. Loop-based sequence solver (fallback to unrolled loop over t)
    if let Some(code) = synthesize_loop_sequence(problem, &t_values) {
        return Some(verify_and_return(code, "stage_4_loop_sequence"));
    }
    
    None
}
```

### 5.1 Stage 4a: Closed-Form Formula Discovery

**File**: `src/solver/search_synthetic_time.rs` (new)

**Goal**: Discover polynomial/exponential/closed-form `f(t) → result`

**Approach**:
```rust
pub(super) fn synthesize_closed_form(
    problem: &Problem,
    t_values: &[i64],
) -> Option<String> {
    let outputs: Vec<i64> = problem.examples
        .iter()
        .map(|ex| ex.expected_int())
        .collect();
    
    // Try polynomial regression up to degree 4
    for degree in 1..=4 {
        if let Some((coeffs, loss)) = fit_polynomial(degree, &t_values, &outputs) {
            if loss < THRESHOLD {
                return Some(emit_polynomial_formula(
                    problem.function_name(),
                    degree,
                    &coeffs,
                ));
            }
        }
    }
    
    // Try known sequences (Fibonacci, triangular, etc.)
    if let Some(formula) = match_known_sequences(&outputs, t_values) {
        return Some(formula);
    }
    
    // Try constant-offset / exponential patterns
    if let Some(formula) = match_exponential(&outputs, t_values) {
        return Some(formula);
    }
    
    None
}

fn fit_polynomial(degree: usize, x: &[i64], y: &[i64]) -> Option<(Vec<f64>, f64)> {
    // Least-squares fit: y ≈ a₀ + a₁x + a₂x² + ...
    // Use Gaussian elimination to solve Vandermonde system
    todo!("Implement numerical linear algebra")
}

fn emit_polynomial_formula(fn_name: &str, degree: usize, coeffs: &[f64]) -> String {
    // Convert floats → exact i64 if possible
    // Emit: fn foo(t: i64) -> i64 { return c0 + c1*t + c2*t*t + ...; }
    format!("fn {fn_name}(t: i64) -> i64 {{\n    return ...;\n}}\n")
}
```

**Known Sequences Database**:
```rust
const KNOWN_SEQUENCES: &[(
    &str,                            // name (for logging)
    fn(&[i64]) -> Option<String>,   // recognizer
)] = &[
    ("fibonacci", recognize_fibonacci),
    ("triangular", recognize_triangular),
    ("square", recognize_square),
    ("factorial", recognize_factorial),
    ("lucas", recognize_lucas),
    ("catalan", recognize_catalan),
];

fn recognize_fibonacci(outputs: &[i64]) -> Option<String> {
    // Check: each element is sum of previous two
    if outputs.len() < 3 {
        return None;
    }
    for i in 2..outputs.len() {
        if outputs[i] != outputs[i-1] + outputs[i-2] {
            return None;
        }
    }
    // Verified; emit Fibonacci formula
    Some("fn fib(t: i64) -> i64 { ... }".to_string())
}

fn recognize_triangular(outputs: &[i64]) -> Option<String> {
    // Check: outputs[i] == i*(i+1)/2
    for (i, &output) in outputs.iter().enumerate() {
        if output != (i as i64) * (i as i64 + 1) / 2 {
            return None;
        }
    }
    Some("fn triangular(t: i64) -> i64 { return (t * (t + 1)) / 2; }".to_string())
}
```

### 5.2 Stage 4b: Loop-Based Sequence Solver

**File**: `src/solver/search_synthetic_time.rs`

**Goal**: If closed form fails, emit a loop that computes values for t ∈ [0, T]

**Approach**:
```rust
pub(super) fn synthesize_loop_sequence(
    problem: &Problem,
    t_values: &[i64],
) -> Option<String> {
    let outputs: Vec<i64> = problem.examples
        .iter()
        .map(|ex| ex.expected_int())
        .collect();
    
    let max_t = t_values.iter().max().copied().unwrap_or(0);
    
    // Try recurrence relations
    if let Some(code) = detect_recurrence_and_emit(problem, &outputs, max_t) {
        return Some(code);
    }
    
    // Fallback: direct computation loop (if reference code available)
    // This requires running reference code inside a loop
    None
}

fn detect_recurrence_and_emit(
    problem: &Problem,
    outputs: &[i64],
    max_t: i64,
) -> Option<String> {
    // Check for: outputs[i] = a*outputs[i-1] + b*outputs[i-2] (Fibonacci-like)
    if outputs.len() < 3 {
        return None;
    }
    
    // Solve for a, b via first two gaps
    let (a, b) = solve_linear_recurrence_2(&outputs)?;
    
    let fn_name = problem.function_name();
    let code = format!(
        "fn {fn_name}(t: i64) -> i64 {{\n\
         if t == 0 {{ return {}; }}\n\
         if t == 1 {{ return {}; }}\n\
         a: i64 = {};\n\
         b: i64 = {};\n\
         i: i64 = 2;\n\
         while i <= t {{\n\
             tmp: i64 = {} * a + {} * b;\n\
             a = b;\n\
             b = tmp;\n\
             i = i + 1;\n\
         }}\n\
         return b;\n\
         }}\n",
        outputs[0], outputs[1], outputs[0], outputs[1], a, b
    );
    
    Some(code)
}

fn solve_linear_recurrence_2(outputs: &[i64]) -> Option<(i64, i64)> {
    // Solve: outputs[i] = a*outputs[i-1] + b*outputs[i-2]
    // Using outputs[2] and outputs[3] as constraints
    // This is a 2x2 linear system
    todo!("Implement Cramer's rule for 2x2 system")
}
```

### 6. Verification & Codegen Integration

**File**: `src/runtime.rs`

**Current verification**:
```rust
pub fn verify_problem_code_strict(problem: &Problem, code: &str) -> Result<(), String> {
    // Runs code on all examples, checks against expected
}
```

**Stage 4 verification**:
```rust
pub fn verify_problem_code_stage4(
    problem: &Problem,
    code: &str,
    t_values: &[i64],
) -> Result<(), String> {
    let fn_name = problem.function_name();
    
    // Build t bindings for each example
    for (example, &t) in problem.examples.iter().zip(t_values.iter()) {
        let mut args = example.inputs.clone();
        
        // Inject synthetic args at the end
        for (i, synthetic_arg) in problem.synthetic_args.iter().enumerate() {
            // Map synthetic_arg name to its value
            // For Stage 4, we only have "t" → t_values[i]
            args.push(Value::Int(t));
        }
        
        let value = execute_function_for_problem(code, fn_name, &args, problem)?;
        if !output_matches(&value, &example.expected) {
            return Err(format!(
                "stage4 verification failed: {} with t={}, expected {}, got {:?}",
                problem.name, t, example.expected, value
            ));
        }
    }
    
    Ok(())
}
```

### 7. Codegen Routing

**File**: `src/solver/search_codegen.rs` (extend)

**Current**: Hand-written code generators for specific patterns (quadratic, fib, etc.)

**Stage 4**: Parameterized generators

```rust
pub(super) fn code_polynomial_formula(fn_name: &str, degree: usize, coeffs: &[i64]) -> String {
    // Build AST-like Mog code for polynomial evaluation
    let mut code = format!("fn {fn_name}(t: i64) -> i64 {{\n    return ");
    
    // Horner's method for efficiency: a₀ + t*(a₁ + t*(a₂ + ...))
    let mut first = true;
    for (i, &c) in coeffs.iter().enumerate() {
        if !first {
            code.push_str(" + ");
        }
        code.push_str(&format!("({} * t^{})", c, i));
        first = false;
    }
    
    code.push_str(";\n}\n");
    code
}

pub(super) fn code_recurrence_formula(
    fn_name: &str,
    init_vals: &[i64],
    coeffs: &[i64],
) -> String {
    // Emit: f(0)=init[0], f(1)=init[1], f(t) = c[0]*f(t-1) + c[1]*f(t-2) + ...
    format!("fn {fn_name}(t: i64) -> i64 {{\n\
         if t == 0 {{ return {}; }}\n\
         if t == 1 {{ return {}; }}\n\
         ... loop ...\n\
         }}\n",
        init_vals[0], init_vals[1]
    )
}
```

## Type Inference Strategy

### Signature Parsing

**File**: `src/solver/signature.rs` (extend)

```rust
pub(super) fn extract_synthetic_args(
    signature: &str,
    declared_synthetic: &[String],
) -> Result<Vec<String>, String> {
    let param_types = parse_param_types(signature);
    let param_names = extract_param_names(signature)?;
    
    // Validate that declared_synthetic names match actual params
    for syn_name in declared_synthetic {
        if !param_names.contains(syn_name) {
            return Err(format!(
                "synthetic arg '{}' not found in signature",
                syn_name
            ));
        }
    }
    
    Ok(declared_synthetic.to_vec())
}
```

### Type Inference Flow

1. **Parse signature**: Extract param names & types
2. **Check Problem.synthetic_args**: Which params are synthetic?
3. **Infer t-values**: Positional or discovery-based
4. **Build execution context**: Map t-values to param indices
5. **Generate code**: Emit params in order (real first, synthetic last)

## AST Signature Changes Summary

| Component | Change | Rationale |
|-----------|--------|-----------|
| `Problem` | Add `synthetic_args: Vec<String>`, `synthetic_arg_ranges: Option<Vec<(i64, i64)>>` | Declare which params are synthetic and their ranges |
| `Example` | No change | T-values inferred separately; examples stay immutable |
| Mog AST | No change | T treated as regular parameter; synthesis rules constrain usage |
| `SolveResult` | Add `stage: String` (e.g., "stage_4_closed_form") | Already exists; no change needed |
| Verification | Add `verify_problem_code_stage4()` | Handle t-value binding during execution |
| Codegen | Add parameterized generators | Polynomial, recurrence, formula emitters |

## Codegen Routing Decision Tree

```
solve_problem(problem)
├─ stage_1_exact_scalar
├─ stage_2_search
├─ stage_3_gradient
└─ stage_4_synthetic_time (if synthetic_args != empty)
   ├─ infer_time_values
   ├─ stage_4a_closed_form
   │  ├─ fit_polynomial(degree=1..4)
   │  ├─ recognize_known_sequences (fib, triangular, etc.)
   │  └─ emit_polynomial_formula or match_formula_code
   └─ stage_4b_loop_sequence (fallback)
      ├─ detect_recurrence_2 (linear 2nd-order)
      └─ emit_recurrence_loop
```

## Type Inference Strategy: Polynomial Fit

**Algorithm**: Least-squares fit via normal equations

**Input**: 
- `t_values: &[i64]` (e.g., `[0, 1, 2, 3, ...]`)
- `outputs: &[i64]` (corresponding function values)

**Output**: 
- Polynomial degree + coefficients
- Loss (MSE)

**Steps**:
1. Build Vandermonde matrix: `V[i, j] = t_values[i]^j`
2. Form normal equations: `(V^T V) * coeffs = V^T * outputs`
3. Solve via Gaussian elimination
4. Compute residuals; if MSE < threshold, return coeffs
5. Else try next degree

**Threshold**: MSE < 0.01 (ensures exact integer results up to rounding)

## Estimated Development Hours

### Phase 1: Core Infrastructure (8–10 hours)
- Extend `Problem` struct + migration: 1–2 hrs
- `infer_time_values()` positional: 0.5 hr
- Signature parsing / extraction: 1 hr
- Stage 4 solver stub + routing: 1.5 hrs
- Verification hooks: 1.5 hrs
- Tests (10 unit tests): 2–3 hrs

### Phase 2: Closed-Form Discovery (10–12 hours)
- Polynomial fit implementation: 3–4 hrs
- Known sequences recognizers (5–6 patterns): 3–4 hrs
- Codegen for polynomial + recurrence: 2 hrs
- Integration + end-to-end tests: 2 hrs

### Phase 3: Fallback Sequence Solver (4–6 hours)
- Recurrence detection (2nd-order linear): 2 hrs
- Loop code generation: 1.5 hrs
- Tests: 1–2 hrs

### Phase 4: Benchmarking & Optimization (4–5 hours)
- Create Stage 4 benchmark set (fibonacci, triangular, etc.): 2 hrs
- Performance profiling: 1 hr
- Refinements: 1–2 hrs

### **Total: 26–33 hours**

- **Low estimate (26 hrs)**: Focus on polynomial fit + 3 known sequences, minimal benchmarking
- **Medium estimate (30 hrs)**: All phases as outlined, solid test coverage
- **High estimate (33 hrs)**: Add discovery-based t-inference, numerical stability improvements

## Deferred (Stage 4b+)

1. **Mixed real + synthetic inputs**: `f(a, b, t) → result`
   - Requires discovery-based t-inference
   - Complicates verification (enumerate over t × real inputs)

2. **Recursion with synthetic t**: `fib(t) = fib(t-1) + fib(t-2)`
   - Requires recursion safety checks
   - Codegen for recursive Mog (not yet in compiler)

3. **Multi-step synthetic time**: `f(t, u) → result` (2D grid)
   - Extends reasoning to 2D polynomial fit
   - Verification & codegen complexity

4. **Differential equation solving**: Time-continuous vs. discrete
   - Requires numerical ODE solver
   - Out of scope for initial Stage 4

## References

### Known Sequences Patterns
- **Fibonacci**: `F(n) = F(n-1) + F(n-2)` with `F(0)=0, F(1)=1`
  - Closed form (Binet): `F(n) = ⌊φⁿ / √5 + 0.5⌋` (used for very large n)
  - Loop (simple): unroll iteration
  
- **Triangular**: `T(n) = n(n+1)/2` (sum of 1..n)
  
- **Square**: `S(n) = n²`
  
- **Factorial**: `n!` (non-polynomial growth; falls to recurrence or loop)
  
- **Lucas**: `L(n) = L(n-1) + L(n-2)` with `L(0)=2, L(1)=1`
  - Closed form: eigenvalue sum formula (similar to Fibonacci)

### Numerical Methods
- **Vandermonde system**: O(n³) exact solve; O(n²) via FFT for large n
- **Least-squares fit**: Normal equations, Cholesky factorization
- **Horner's method**: Efficient polynomial evaluation (O(degree) ops)

### Papers & References
- "Program Synthesis from Examples": Gulwani et al. (Sketch, FlashFill)
- "Bottom-Up Synthesis": Alur et al. (Bit-blasting, piecewise linear)
- "Polynomial Program Synthesis": Matsubara et al. (curve fitting for sequences)
