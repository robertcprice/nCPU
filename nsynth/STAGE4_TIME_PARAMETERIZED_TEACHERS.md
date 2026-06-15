# Stage 4 Time-Parameterized Synthesis Teachers

## Overview
Design search teachers (Stage 4) for time-parameterized synthesis problems where output is a function of a single time/index parameter `t: i64` and produces `-> i64` output. These recognize mathematical sequences and closed-form patterns without looping.

## 1. Teacher: `search_polynomial_time`

### Signature
```rust
pub(super) fn search_polynomial_time(problem: &Problem, fn_name: &str) -> Option<SolveResult>
```

### Input Guard
- **Arity**: Single integer input `t: i64`
- **Examples**: At least 5 points (t, output) pairs with t >= 0
- **Pattern**: Output grows polynomially with t (no exponential or factorial jumps)

### Pattern Recognition Heuristics
```
1. Extract (t, output) pairs from examples
2. Sort by t ascending
3. Check polynomial degree:
   - Degree 1 (linear):      Δ¹ is constant     → y = a·t + b
   - Degree 2 (quadratic):   Δ² is constant    → y = a·t² + b·t + c
   - Degree 3 (cubic):       Δ³ is constant    → y = a·t³ + b·t² + c·t + d
4. Guard against overfitting:
   - Require >= 2 points per degree (min 5 points total)
   - Check that fitted polynomial matches ALL examples exactly
   - Reject if any coefficient > 100 or < -100 (implausibly large)
5. Accept only if polynomial fits perfectly with reasonable coefficients
```

### Codegen Template
```rust
fn {fn_name}(t: i64) -> i64 {
    return {a}*t*t + {b}*t + {c};  // degree varies: 1, 2, 3
}
```

### Examples
- `(0→1), (1→3), (2→7), (3→13)` → Δ¹=[2,4,6] → linear, y = 2t + 1
- `(0→0), (1→1), (2→4), (3→9)` → Δ²=[1,2,3] then [1,1] → quadratic, y = t²
- `(0→2), (1→5), (2→10), (3→17)` → Δ²=[1,2,3] then [1,1] → quadratic, y = t² + 2t

### Implementation Details
```rust
// Finite difference method
fn detect_polynomial_degree(pts: &[(i64, i64)]) -> Option<(usize, Vec<i64>)> {
    // pts sorted by t
    // Compute Δ¹ (first differences)
    // Compute Δ² (second differences)
    // Compute Δ³ (third differences)
    // Return (degree, coefficients) when differences stabilize
}

// Fit polynomial via system of equations or Lagrange interpolation
fn fit_polynomial(pts: &[(i64, i64)], degree: usize) -> Option<Vec<i64>> {
    // Use 3-point Gaussian elimination for degree 1-3
    // Return [a, b, c, d] coefficients
}
```

---

## 2. Teacher: `search_exponential_time`

### Signature
```rust
pub(super) fn search_exponential_time(problem: &Problem, fn_name: &str) -> Option<SolveResult>
```

### Input Guard
- **Arity**: Single integer input `t: i64`
- **Examples**: At least 4 points (t, output) with t in 0..8 (to avoid overflow)
- **Pattern**: Rapid exponential growth (2x, 3x, φx per step) OR Fibonacci pattern

### Pattern Recognition Heuristics
```
1. Extract (t, output) pairs, sort by t
2. Check growth rate:
   - ratio[i] = output[i+1] / output[i]
   - If all ratios ≈ 2.0 → try powers of 2 (2^t)
   - If all ratios ≈ 3.0 → try powers of 3 (3^t)
   - If ratios follow [1, 2, 1.5, 1.67, 1.6, ...] (harmonic mean ~φ) → try Fibonacci
   - If output grows but ratio decays → try triangular (not exponential, reject)
3. Guards:
   - Require max_t <= 62 to prevent overflow (2^63 > i64::MAX)
   - Check exact match on all examples
   - Reject if any example violates growth pattern consistency
4. Accept best match (power-of-2 > power-of-3 > Fibonacci by preference)
```

### Codegen Templates

#### Powers of 2
```rust
fn {fn_name}(t: i64) -> i64 {
    if t < 0 { return 1; }
    if t >= 63 { return -1; }  // overflow sentinel
    acc: i64 = 1;
    i: i64 = 0;
    while i < t {
        acc = acc * 2;
        i = i + 1;
    }
    return acc;
}
```

#### Powers of 3
```rust
fn {fn_name}(t: i64) -> i64 {
    if t < 0 { return 1; }
    if t >= 40 { return -1; }
    acc: i64 = 1;
    i: i64 = 0;
    while i < t {
        acc = acc * 3;
        i = i + 1;
    }
    return acc;
}
```

#### Fibonacci (closed-form via Binet or iteration)
```rust
fn {fn_name}(n: i64) -> i64 {
    if n == 0 { return 0; }
    if n == 1 { return 1; }
    a: i64 = 0;
    b: i64 = 1;
    i: i64 = 2;
    while i <= n {
        tmp: i64 = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}
```

#### Triangular Numbers (fallback for rejected exponential)
```rust
fn {fn_name}(t: i64) -> i64 {
    if t < 0 { return 0; }
    return (t * (t + 1)) / 2;
}
```

### Examples
- `(0→1), (1→2), (2→4), (3→8), (4→16)` → ratio=[2,2,2,2] → 2^t
- `(0→1), (1→3), (2→9), (3→27)` → ratio=[3,3,3] → 3^t
- `(0→0), (1→1), (2→1), (3→2), (4→3), (5→5)` → Fibonacci(n)

### Implementation Details
```rust
fn detect_exponential_growth(pts: &[(i64, i64)]) -> Option<ExponentialKind> {
    // Compute ratios output[i+1] / output[i]
    // Check consistency of ratios
    // Match to base (2, 3, or fibonacci)
}

enum ExponentialKind {
    PowerOf2,
    PowerOf3,
    Fibonacci,
}
```

---

## 3. Teacher: `search_factorial_time`

### Signature
```rust
pub(super) fn search_factorial_time(problem: &Problem, fn_name: &str) -> Option<SolveResult>
```

### Input Guard
- **Arity**: Single integer input `t: i64`
- **Examples**: At least 4 points matching factorial sequence (0→1, 1→1, 2→2, 3→6, 4→24, ...)
- **Pattern**: output = t! (factorial growth)

### Pattern Recognition Heuristics
```
1. Extract (t, output) pairs, sort by t
2. Check against factorial:
   - Compute t! for t in 0..min(t_max, 20)  [20! > i64::MAX]
   - Match all examples against factorial table
   - Require NO mismatches (all examples exactly match t!)
3. Guards:
   - Require all t in [0, 20] (21! overflows i64)
   - Reject if any output doesn't match t!
4. Accept if exact match found
```

### Codegen Template
```rust
fn {fn_name}(n: i64) -> i64 {
    if n < 0 { return 1; }
    if n > 20 { return -1; }  // overflow sentinel
    r: i64 = 1;
    i: i64 = 1;
    while i <= n {
        r = r * i;
        i = i + 1;
    }
    return r;
}
```

### Examples
- `(0→1), (1→1), (2→2), (3→6), (4→24), (5→120)` → factorial(n)

### Implementation Details
```rust
const FACTORIAL_TABLE: &[i64] = &[
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,
    3628800, 39916800, 479001600, 6227020800, 87178291200,
    1307674368000, 20922789888000, 355687428096000,
    6402373705728000, 121645100408832000, 2432902008176640000,
];

fn is_factorial_pattern(pts: &[(i64, i64)]) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t as usize >= FACTORIAL_TABLE.len() {
            return false;
        }
        FACTORIAL_TABLE[t as usize] == output
    })
}
```

---

## 4. Teacher: `search_triangular_time`

### Signature
```rust
pub(super) fn search_triangular_time(problem: &Problem, fn_name: &str) -> Option<SolveResult>
```

### Input Guard
- **Arity**: Single integer input `t: i64`
- **Examples**: At least 4 points matching triangular sequence (0→0, 1→1, 2→3, 3→6, 4→10, ...)
- **Pattern**: output = t*(t+1)/2 (triangular number, sum 1..t)

### Pattern Recognition Heuristics
```
1. Extract (t, output) pairs, sort by t
2. Check triangular pattern:
   - Compute tri(t) = t*(t+1)/2 for each example t
   - Match all examples against triangular formula
   - Require NO mismatches
3. Guards:
   - Require all t in [0, 1000000] (no practical overflow risk for typical inputs)
   - Check for cumsum pattern:
     - Δ¹[i] = i+1 (differences are 1, 2, 3, 4, ...)
     - Δ² is constant = 1
4. Accept if exact match found
```

### Codegen Template
```rust
fn {fn_name}(t: i64) -> i64 {
    if t < 0 { return 0; }
    return (t * (t + 1)) / 2;
}
```

### Examples
- `(0→0), (1→1), (2→3), (3→6), (4→10), (5→15)` → triangular(t)
- Related: cumsum pattern (1, 1+2=3, 1+2+3=6, ...)

### Implementation Details
```rust
fn is_triangular_pattern(pts: &[(i64, i64)]) -> bool {
    pts.iter().all(|&(t, output)| {
        if t < 0 || t > 1000000 { return false; }
        let expected = (t * (t + 1)) / 2;
        expected == output
    })
}

// Alternative: detect from differences
fn triangular_from_differences(pts: &[(i64, i64)]) -> bool {
    // Compute Δ¹: should be [1, 2, 3, 4, ...]
    // If differences match 1..N exactly, it's triangular
    let mut expected_diff = 1i64;
    for i in 1..pts.len() {
        let actual_diff = pts[i].1 - pts[i-1].1;
        if actual_diff != expected_diff {
            return false;
        }
        expected_diff += 1;
    }
    true
}
```

---

## Integration into Stage 4 Search

### Placement in Search Roster
Add these four teachers to `SEARCH_CANDIDATES` in `src/solver/search.rs`:

```rust
SearchCandidate {
    key: "search_polynomial_time",
    func: search_polynomial_time,
},
SearchCandidate {
    key: "search_exponential_time",
    func: search_exponential_time,
},
SearchCandidate {
    key: "search_factorial_time",
    func: search_factorial_time,
},
SearchCandidate {
    key: "search_triangular_time",
    func: search_triangular_time,
},
```

### Ordering Strategy
Place **before** `search_scalar_expr` (which does brute-force expression search) but **after** specific structural solvers like `search_fib_iter_loop`. Order by pattern rarity:
1. `search_factorial_time` (rarest, most specific)
2. `search_exponential_time` (common in algorithms)
3. `search_triangular_time` (common in combinatorics)
4. `search_polynomial_time` (most general, last resort before expr search)

### Module Structure
- New file: `src/solver/search_time_families.rs`
- Helpers: Reuse `verify_problem_code_strict`, `code_quadratic_search`, etc. from existing modules
- Validators: Add specialized `validate_time_*` functions if needed

---

## Estimated Coverage

### Teachers Solve
| Teacher | Est. Problems | Est. % | Domains |
|---------|---------------|--------|---------|
| search_factorial_time | 2-3 | 2-3% | Sequence (factorial) |
| search_exponential_time | 8-12 | 8-12% | Sequence (powers, Fibonacci, growth) |
| search_triangular_time | 2-4 | 2-4% | Sequence (triangular, cumsum) |
| search_polynomial_time | 15-25 | 15-25% | Sequence (polynomial fit) |
| **Total (4 teachers)** | **27-44** | **27-44%** | Diverse sequences/time-parameterized |

### Non-Overlap with Existing Teachers
- `search_fib_iter_loop` handles Fibonacci hardcoded loop → `search_exponential_time` detects at closed-form level
- `search_polynomial_quadratic` handles degree-2 only via brute-force grid → `search_polynomial_time` generalizes to degrees 1-3 with solver
- `search_triangular_check_loop` checks membership (boolean) → `search_triangular_time` returns the value
- No overlap with array/string/branching solvers

### Fallthrough
- Patterns not matching any teacher (e.g., Catalan numbers, partition function) fall through to `search_scalar_expr` (differentiable search) or `template` family

---

## Key Design Decisions

### 1. **Single Scalar Input Only**
Time-parameterized teachers work on problems with a single `t: i64` input. Multi-input problems are outside scope (handled by composition or branching elsewhere).

### 2. **Closed-Form Preference**
All templates are closed-form (no loops except for exponentiation/factorial). This makes them fast (O(t) or O(1)) and cacheable for repeated calls.

### 3. **Exact Match Required**
Teachers are **high-precision**: they require all examples to match exactly, not approximately. This prevents false positives and ensures generalization.

### 4. **Overflow Guards**
- Factorial: max t=20
- Exponential (2^): max t=62
- Exponential (3^): max t=39
- Triangular: no risk for typical inputs
- All emit a sentinel (−1 or 0) on overflow to gracefully fail rather than wrap around.

### 5. **No Heuristic Guessing**
Teachers recognize patterns from the data itself (finite differences, growth ratios) rather than guessing. This is inspired by classical numerical methods (Newton forward difference, least-squares fit).

### 6. **Ordering by Specificity**
Teachers are tried in order of pattern specificity:
1. Factorial (most specific pattern)
2. Exponential (specific growth signature)
3. Triangular (specific difference pattern)
4. Polynomial (most general, catches linear/quadratic/cubic)

This ensures rare patterns are caught before general ones (no shadowing).

---

## Testing Strategy

### Unit Tests (proposed)
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_time_linear() { ... }
    
    #[test]
    fn test_polynomial_time_quadratic() { ... }
    
    #[test]
    fn test_exponential_time_power_of_2() { ... }
    
    #[test]
    fn test_exponential_time_fibonacci() { ... }
    
    #[test]
    fn test_factorial_time() { ... }
    
    #[test]
    fn test_triangular_time() { ... }
}
```

### Benchmark Integration
- Wrap each teacher with `verify_problem_code_strict()` to ensure correctness
- Measure per-teacher solve time (expect <10ms for pattern recognition)
- Track coverage: "X/105 problems solved by time-parameterized teachers"

### Regression Gates
- Ensure no existing solved problems regress (false negatives)
- Ensure time-parameterized teachers don't overshadow existing solvers (pattern specificity ordering)

---

## Summary

These four teachers recognize the most common closed-form sequences in programming competitions and academic curricula:

| Sequence | Teacher | Pattern | Codegen |
|----------|---------|---------|---------|
| Linear, quadratic, cubic | `search_polynomial_time` | Constant finite differences | y = a·t² + b·t + c |
| Powers of 2, 3, Fibonacci | `search_exponential_time` | Exponential growth rate | 2^t, 3^t, fib(t) |
| n! | `search_factorial_time` | Factorial growth (1,1,2,6,24,...) | loop: r *= i |
| Triangular, cumsum | `search_triangular_time` | Δ¹ = [1,2,3,...] | (t*(t+1))/2 |

**Estimated total coverage: 27–44 problems (27–44% of 105-problem benchmark)**, enabling Stage 4 search to compete with gradient/enumerative methods on sequence recognition without expensive brute-force expression search.
