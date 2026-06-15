# Stage 6: Technical Specification

**Purpose**: Detailed technical specification for function composition synthesis implementation.

---

## 1. Data Structures

### 1.1 Composition Pattern Enum

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompositionPattern {
    /// Input array → smaller output array (remove elements)
    Filter,
    
    /// Input array → output array (same size, transform each element)
    Map,
    
    /// Input array → scalar output (fold/accumulation)
    Reduce,
    
    /// Input array → output array (same size, cumulative fold)
    Scan,
    
    /// Input array → sorted output array
    Sort,
    
    /// Two arrays → array of pairs
    Zip,
    
    /// Array of pairs → two arrays
    Transpose,
    
    /// Predicate search → indices/count of matches
    FindAll,
}

impl CompositionPattern {
    pub fn name(&self) -> &'static str {
        match self {
            Filter => "filter",
            Map => "map",
            Reduce => "reduce",
            Scan => "scan",
            Sort => "sort",
            Zip => "zip",
            Transpose => "transpose",
            FindAll => "find_all",
        }
    }
}
```

### 1.2 Composition Template

```rust
pub struct CompositionTemplate {
    /// Library function name, e.g., "filter", "map", "reduce"
    pub name: &'static str,
    
    /// Full Mog function signature
    /// e.g., "fn filter(arr: [i64], threshold: i64) -> [i64]"
    pub signature: &'static str,
    
    /// Complete Mog function implementation
    /// Must be fully verified and self-contained
    pub code: &'static str,
    
    /// Which composition pattern this implements
    pub pattern: CompositionPattern,
    
    /// Input type signature (for type checking)
    pub input_types: &'static [Type],
    
    /// Output type (for type checking)
    pub output_type: Type,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    Array,
    Scalar,
    Pair,
}
```

### 1.3 Pattern Match Result

```rust
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// Which composition pattern(s) fit
    pub patterns: Vec<CompositionPattern>,
    
    /// Confidence score [0.0, 1.0]
    /// Used to order candidates (highest first)
    pub confidence: f32,
    
    /// Number of examples this pattern explains
    pub num_examples_explained: usize,
    
    /// Optional derived parameters
    /// e.g., ("threshold", "0") for filter
    pub inferred_params: Vec<(String, String)>,
}
```

### 1.4 Composed Program

```rust
pub struct ComposedProgram {
    /// Helper functions emitted first
    /// Each is a complete Mog function
    pub helper_functions: Vec<String>,
    
    /// Main function that calls helpers in sequence
    pub main_function: String,
    
    /// How the composition is structured (for diagnostics)
    pub composition_chain: Vec<CompositionPattern>,
    
    /// Code verification status
    pub verified: bool,
}
```

### 1.5 Problem Extension

```rust
impl Problem {
    // NEW FIELDS:
    /// If false, composition solver is skipped entirely
    pub composition_allowed: bool,
    
    /// Optional hints about which patterns might work
    /// Empty = no hint, solver must discover
    pub composition_hints: Vec<CompositionPattern>,
    
    // EXISTING FIELDS:
    pub name: String,
    pub category: &'static str,
    pub examples: Vec<Example>,
    // ... rest of struct ...
}
```

---

## 2. Pattern Detection Algorithm

### 2.1 High-Level Flow

```
Input: Problem {examples, ...}
Output: Vec<PatternMatch>

1. Extract input/output signatures from all examples
2. For each example pair (input, expected_output):
   a. Try detect_filter() → if ok, add to candidates
   b. Try detect_map() → if ok, add to candidates
   c. Try detect_reduce() → if ok, add to candidates
   d. Try detect_scan() → if ok, add to candidates
   e. Try detect_sort() → if ok, add to candidates
3. Aggregate results (voting, confidence scoring)
4. Filter out patterns with confidence < 0.5
5. Return sorted by confidence (descending)
```

### 2.2 Filter Detection

**Precondition**: input is array, output is array

```
def detect_filter(input: [i64], output: [i64]) -> Option<PatternMatch> {
  // Filter produces output ⊆ input (subset property)
  // All elements of output must exist in input
  
  if not all(x in input for x in output):
    return None  // Not a filter
  
  confidence = 1.0 - (|input| - |output|) / |input|  // Higher if subset is large
  
  // Try to infer predicate
  // e.g., if output = [1, 3, 5] and input = [1, 2, 3, 4, 5]
  // → likely "x is odd" or "x > 0"
  
  return PatternMatch {
    patterns: [Filter],
    confidence: 0.8 * confidence,  // Base 0.8, scaled by subset size
    num_examples_explained: 1,
    inferred_params: [...try to infer predicate...],
  }
}
```

### 2.3 Map Detection

**Precondition**: input is array, output is array, same size

```
def detect_map(input: [i64], output: [i64]) -> Option<PatternMatch> {
  if len(input) != len(output):
    return None
  
  // Map applies same transformation to each element
  // Check if output[i] = f(input[i]) for some consistent f
  
  if can_infer_unary_function(input, output):
    return PatternMatch {
      patterns: [Map],
      confidence: 0.75,
      num_examples_explained: 1,
      inferred_params: [("transform", "...inferred...")],
    }
  
  return None
}
```

### 2.4 Reduce Detection

**Precondition**: input is array, output is scalar

```
def detect_reduce(input: [i64], output: i64) -> Option<PatternMatch> {
  // Reduce folds array to scalar
  // Try to infer the operator
  
  if try_infer_reduce_operator(input, output, input[0]):
    // Try sum: output == sum(input)?
    // Try product: output == product(input)?
    // Try min: output == min(input)?
    // Try max: output == max(input)?
    
    return PatternMatch {
      patterns: [Reduce],
      confidence: 0.9,  // High confidence for reduce
      num_examples_explained: 1,
      inferred_params: [("operator", "...inferred...")],
    }
  
  return None
}
```

### 2.5 Scan Detection

**Precondition**: input is array, output is array, same size

```
def detect_scan(input: [i64], output: [i64]) -> Option<PatternMatch> {
  if len(input) != len(output):
    return None
  
  // Scan applies cumulative fold
  // output[i] = fold(input[0..=i], op)
  
  if is_cumulative_fold(input, output):
    return PatternMatch {
      patterns: [Scan],
      confidence: 0.85,
      num_examples_explained: 1,
      inferred_params: [("operator", "...inferred...")],
    }
  
  return None
}
```

### 2.6 Sort Detection

**Precondition**: input is array, output is array, same size

```
def detect_sort(input: [i64], output: [i64]) -> Option<PatternMatch> {
  if len(input) != len(output):
    return None
  
  if sorted(input) == output:
    return PatternMatch {
      patterns: [Sort],
      confidence: 0.95,
      num_examples_explained: 1,
      inferred_params: [],
    }
  
  return None
}
```

### 2.7 Aggregation

```rust
pub fn detect_composition_patterns(problem: &Problem) -> Vec<PatternMatch> {
    let mut all_matches: Vec<PatternMatch> = Vec::new();
    
    for example in &problem.examples {
        let (input_type, output_type) = classify_io(&example);
        
        match (input_type, output_type) {
            (InputType::Array, OutputType::Array) => {
                // Could be filter, map, scan, sort, or zip/transpose
                if let Some(m) = detect_filter(...) { all_matches.push(m); }
                if let Some(m) = detect_map(...) { all_matches.push(m); }
                if let Some(m) = detect_scan(...) { all_matches.push(m); }
                if let Some(m) = detect_sort(...) { all_matches.push(m); }
            }
            (InputType::Array, OutputType::Scalar) => {
                // Must be reduce or find_all
                if let Some(m) = detect_reduce(...) { all_matches.push(m); }
            }
            _ => {
                // Composition doesn't apply
            }
        }
    }
    
    // Aggregate by pattern (voting)
    let aggregated = aggregate_matches(all_matches);
    
    // Sort by confidence
    aggregated.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    
    // Filter minimum confidence threshold
    aggregated.into_iter()
        .filter(|m| m.confidence >= 0.5)
        .collect()
}

fn aggregate_matches(matches: Vec<PatternMatch>) -> Vec<PatternMatch> {
    // Group by pattern, sum confidence, count votes
    // Return merged results
    let mut grouped: HashMap<CompositionPattern, Vec<PatternMatch>> = HashMap::new();
    
    for m in matches {
        for p in m.patterns {
            grouped.entry(p).or_default().push(m.clone());
        }
    }
    
    grouped.into_iter().map(|(pattern, group)| {
        PatternMatch {
            patterns: vec![pattern],
            confidence: group.iter().map(|m| m.confidence).sum::<f32>() / group.len() as f32,
            num_examples_explained: group.len(),
            inferred_params: /* merge/vote on params */ vec![],
        }
    }).collect()
}
```

---

## 3. Inline Codegen

### 3.1 Filter Inline

```rust
pub fn emit_inline_filter(
    fn_name: &str,
    predicate_code: &str,  // e.g., "x > 0"
) -> String {
    format!(r#"
fn {fn_name}(arr: [i64]) -> i64 {{
    result: i64 = 0
    i: i64 = 0
    while i < len(arr) {{
        x: i64 = arr[i]
        if {predicate_code} {{
            result = result + x
        }}
        i = i + 1
    }}
    return result
}}
    "#)
}

// Specialized versions for common predicates:
pub fn emit_inline_filter_positive(fn_name: &str) -> String {
    emit_inline_filter(fn_name, "x > 0")
}

pub fn emit_inline_filter_even(fn_name: &str) -> String {
    emit_inline_filter(fn_name, "(x % 2) == 0")
}
```

### 3.2 Map Inline

```rust
pub fn emit_inline_map(
    fn_name: &str,
    transform_code: &str,  // e.g., "x * 2"
) -> String {
    // For array output (e.g., double_all)
    // OR for scalar output (e.g., sum of transformed)
    
    // Variant 1: Output is array
    format!(r#"
fn {fn_name}(arr: [i64]) -> [i64] {{
    result: [i64] = []
    i: i64 = 0
    while i < len(arr) {{
        x: i64 = arr[i]
        result[i] = {transform_code}
        i = i + 1
    }}
    return result
}}
    "#)
}

pub fn emit_inline_map_to_scalar(
    fn_name: &str,
    transform_code: &str,
    reduce_op: &str,  // "+" for sum, "*" for product
) -> String {
    // Variant 2: Map + Reduce combined (for efficiency)
    format!(r#"
fn {fn_name}(arr: [i64]) -> i64 {{
    result: i64 = 0
    i: i64 = 0
    while i < len(arr) {{
        x: i64 = arr[i]
        transformed: i64 = {transform_code}
        result = result {reduce_op} transformed
        i = i + 1
    }}
    return result
}}
    "#)
}
```

### 3.3 Reduce Inline

```rust
pub fn emit_inline_reduce(
    fn_name: &str,
    init: i64,
    op: &str,  // "+", "*", "min", "max"
) -> String {
    let op_code = match op {
        "+" => "result + arr[i]",
        "*" => "result * arr[i]",
        "min" => "if arr[i] < result { arr[i] } else { result }",
        "max" => "if arr[i] > result { arr[i] } else { result }",
        _ => "result + arr[i]",  // default: sum
    };
    
    format!(r#"
fn {fn_name}(arr: [i64]) -> i64 {{
    result: i64 = {init}
    i: i64 = 0
    while i < len(arr) {{
        result = {op_code}
        i = i + 1
    }}
    return result
}}
    "#)
}
```

### 3.4 Scan Inline

```rust
pub fn emit_inline_scan(
    fn_name: &str,
    init: i64,
    op: &str,  // "+" or "*"
) -> String {
    let op_code = match op {
        "+" => "result + arr[i]",
        "*" => "result * arr[i]",
        _ => "result + arr[i]",
    };
    
    format!(r#"
fn {fn_name}(arr: [i64]) -> [i64] {{
    result: [i64] = []
    acc: i64 = {init}
    i: i64 = 0
    while i < len(arr) {{
        acc = {op_code}
        result[i] = acc
        i = i + 1
    }}
    return result
}}
    "#)
}
```

---

## 4. Function Call Codegen

### 4.1 Multi-Function Structure

```rust
pub fn emit_composed_function_with_calls(
    fn_name: &str,
    patterns: Vec<CompositionPattern>,
    problem: &Problem,
) -> Option<String> {
    // For pattern chain [Filter, Reduce]:
    // 1. Emit filter_helper function
    // 2. Emit reduce_helper function
    // 3. Emit main function that calls both in sequence
    
    let mut code = String::new();
    
    // Helper functions
    code.push_str(&emit_filter_helper());  // fn _filter_helper(arr) -> [i64]
    code.push('\n');
    code.push_str(&emit_reduce_helper());  // fn _reduce_helper(arr) -> i64
    code.push('\n');
    
    // Main function that chains them
    code.push_str(&format!(
        r#"
fn {fn_name}(arr: [i64]) -> i64 {{
    filtered: [i64] = _filter_helper(arr)
    return _reduce_helper(filtered)
}}
        "#
    ));
    
    Some(code)
}
```

### 4.2 Helper Function Naming

```
_filter_helper_1
_map_helper_1
_reduce_helper_1
_scan_helper_1
_sort_helper_1

For nested:
_filter_map_helper_1
_map_reduce_helper_1
_filter_map_reduce_helper_1
```

---

## 5. Verification

### 5.1 Pre-Verification Checks

```rust
pub fn verify_composition_code(code: &str, problem: &Problem) -> Result<(), String> {
    // 1. Syntax check: parse as Mog
    parse_mog(&code)
        .map_err(|e| format!("Parse error: {}", e))?;
    
    // 2. Semantic check: all functions defined, no undeclared calls
    let functions = extract_function_names(&code);
    // ... verify no forward references ...
    
    // 3. Execution: run on examples, check outputs
    verify_problem_code_strict(problem, &code)
}
```

### 5.2 Post-Composition Verification

```rust
fn try_composition_solve(problem: &Problem) -> Option<SolveResult> {
    if !problem.composition_allowed {
        return None;
    }
    
    let patterns = detect_composition_patterns(problem)?;
    
    // Try inline first (simpler, more likely to verify)
    if let Some(code) = emit_inline_composition(&patterns, problem) {
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "composition_inline".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    
    // Try with function calls (more general, less likely to verify)
    if let Some(code) = emit_composition_with_calls(&patterns, problem) {
        if verify_problem_code_strict(problem, &code).is_ok() {
            return Some(SolveResult {
                success: true,
                code,
                method: "composition_calls".to_string(),
                error: None,
                metadata: DifferentiableMetadata::default(),
            });
        }
    }
    
    None
}
```

---

## 6. Composition Teacher in Solver Pipeline

### 6.1 Teacher Function Signature

```rust
pub fn search_composition(problem: &Problem) -> Option<SolveResult> {
    try_composition_solve(problem)
}
```

### 6.2 Placement in `SEARCH_CANDIDATES`

```rust
pub const SEARCH_CANDIDATES: &[SearchTeacher] = &[
    // Stage 1: Pre-enumeration teachers
    search_bitwise,
    search_text_families,
    search_runtime,
    
    // Stage 2: Enumerative-eligible teachers
    
    // NEW: Composition (between enum and gradient for efficiency)
    search_composition,
    
    // Stage 3: Gradient-preferred teachers
    search_scalar_families,
    // ... rest of teachers ...
];
```

### 6.3 Cost Budget

```
Per-problem composition budget: 5 seconds max
  - Pattern detection: 50-100ms
  - Inline codegen: 10-20ms
  - Function-call codegen: 20-50ms
  - Verification: 0-5 seconds (runtime dependent)
```

---

## 7. Composition Library Templates

### 7.1 Template Examples

```rust
pub const COMPOSITION_TEMPLATES: &[CompositionTemplate] = &[
    CompositionTemplate {
        name: "filter",
        signature: "fn filter(arr: [i64]) -> [i64]",
        code: r#"
fn filter(arr: [i64]) -> [i64] {
    result: [i64] = []
    i: i64 = 0
    while i < len(arr) {
        if arr[i] > 0 {
            result[len(result)] = arr[i]
        }
        i = i + 1
    }
    return result
}
        "#,
        pattern: CompositionPattern::Filter,
        input_types: &[Type::Array],
        output_type: Type::Array,
    },
    
    // ... more templates ...
];
```

### 7.2 Template Verification

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn verify_all_templates() {
        for template in COMPOSITION_TEMPLATES {
            // Parse template code
            let parsed = parse_mog(template.code);
            assert!(parsed.is_ok(), "Template {} failed to parse", template.name);
            
            // Execute on simple test case
            // e.g., for filter, run on [1, -2, 3] → [1, 3]
        }
    }
}
```

---

## 8. Integration Points

### 8.1 Solver (`src/solver.rs`)

- `try_composition_solve()` called after enumerative, before gradient
- Result is `Option<SolveResult>` with method = "composition"

### 8.2 Benchmarks (`src/benchmark.rs`)

- Add `composition_allowed` field
- Add 12 factory functions with `composition_allowed = true`
- Existing 105 factories default to `composition_allowed = false` for non-array problems

### 8.3 Runtime (`src/runtime.rs`)

- Ensure function call execution works
- No changes expected (runtime already supports calls)

### 8.4 Tests (`tests/...`)

- Add `test_composition_teacher_priority` — verify teacher ordering
- Add `test_composition_benchmarks_all_pass` — all 12 factories solve
- Add `test_no_regression_on_existing_factories` — 105 still pass

---

## 9. Error Handling

### 9.1 Graceful Fallback

```rust
impl SearchTeacher for search_composition {
    fn try_solve(problem: &Problem) -> Option<SolveResult> {
        // If composition is not allowed, return None
        if !problem.composition_allowed {
            return None;
        }
        
        // If pattern detection fails, return None (graceful)
        let patterns = match detect_composition_patterns(problem) {
            Some(p) => p,
            None => return None,  // No pattern found, skip
        };
        
        // If codegen fails, return None
        let code = match emit_inline_composition(&patterns, problem) {
            Some(c) => c,
            None => return None,  // Couldn't generate code
        };
        
        // If verification fails, return None
        match verify_problem_code_strict(problem, &code) {
            Ok(_) => {
                Some(SolveResult {
                    success: true,
                    code,
                    method: "composition".to_string(),
                    error: None,
                    metadata: DifferentiableMetadata::default(),
                })
            }
            Err(_) => {
                return None;  // Verification failed, let next teacher try
            }
        }
    }
}
```

### 9.2 Diagnostics

```rust
pub fn composition_diagnostics(problem: &Problem) -> String {
    let patterns = detect_composition_patterns(problem)
        .unwrap_or_default();
    
    format!(
        "Composition analysis for {}:\n  Patterns: {:?}\n  Allowed: {}",
        problem.name,
        patterns,
        problem.composition_allowed
    )
}
```

---

## 10. Configuration & Tuning

### 10.1 Environment Variables

```
NSYNTH_COMPOSITION_ENABLED=1           # Default: true
NSYNTH_COMPOSITION_TIMEOUT_MS=5000     # Budget: 5 seconds
NSYNTH_COMPOSITION_MIN_CONFIDENCE=0.5  # Min pattern confidence
```

### 10.2 Tuning Parameters

```rust
const COMPOSITION_CONFIDENCE_FILTER: f32 = 0.75;
const COMPOSITION_CONFIDENCE_MAP: f32 = 0.70;
const COMPOSITION_CONFIDENCE_REDUCE: f32 = 0.90;
const COMPOSITION_CONFIDENCE_SCAN: f32 = 0.85;
const COMPOSITION_CONFIDENCE_SORT: f32 = 0.95;

const COMPOSITION_TIME_BUDGET_MS: u64 = 5000;
const COMPOSITION_VERIFY_TIMEOUT_MS: u64 = 3000;
```

---

## Appendix: Example Walk-Through

### Problem: `filter_sum_positive`

```
Input: Problem {
    name: "filter_sum_positive",
    examples: [
        Example { inputs: [[1, -2, 3, -4, 5]], expected: 9 },
        Example { inputs: [[-1, 2, -3]], expected: 2 },
    ],
    composition_allowed: true,
    composition_hints: [Filter, Reduce],
}

Step 1: detect_composition_patterns()
  - Input: Array([1, -2, 3, -4, 5])
  - Output: Scalar(9)
  → Pattern: Reduce (0.9 confidence)
  
  - BUT also: subset {1, 3, 5} ⊂ input
  → Pattern: Filter (0.75 confidence)

  Result: [
    PatternMatch { patterns: [Reduce], confidence: 0.9, ... },
    PatternMatch { patterns: [Filter], confidence: 0.75, ... },
  ]

Step 2: Try inline codegen for Reduce first (highest confidence)
  emit_inline_reduce(fn_name="filter_sum_positive", init=0, op="+")
  → Code is just "sum all elements"
  → But examples show filtering, not summing all
  → Verification FAILS

Step 3: Try inline codegen for Filter next
  emit_inline_filter(fn_name="filter_sum_positive", predicate="x > 0")
  → Code is "sum elements where x > 0"
  → Verification: [1, -2, 3, -4, 5] → sum of [1, 3, 5] = 9 ✓
  → SUCCESS, return SolveResult { code, method: "composition_inline" }
```

---

**Version**: 1.0  
**Date**: June 15, 2026  
**Owner**: Bobby Price
