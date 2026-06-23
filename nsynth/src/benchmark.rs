/// Binary tree node for Stage 5 tree synthesis problems.
/// left/right are indices into a Vec<TreeNode> (negative means null).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct TreeNode {
    pub value: i64,
    pub left: i32,
    pub right: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Value {
    Int(i64),
    /// A float value kept as IEEE-754 bits so `Value` (and `Example`/`Problem`,
    /// which derive `Eq`/`Ord`/`Serialize`) stay derivable — `f64` is neither
    /// `Eq` nor `Ord`. Recover with `f64::from_bits`.
    Float(u64),
    Str(String),
    Bool(bool),
    /// A heterogeneous array. Historically the wire array was `Vec<i64>`; it now
    /// carries `Vec<Value>` so typed (string/float/bool), nested (array-of-array),
    /// and struct-element arrays can be expressed and verified end-to-end. The
    /// overwhelmingly common int-array case is built via the `array(&[i64])`
    /// helper (each element wrapped as `Value::Int`) so existing call sites and
    /// their rendered output are unchanged; typed/nested arrays use `array_of`.
    Array(Vec<Value>),
    Pair(i64, i64),
    /// 4-field struct for struct-of-state benchmarks (e.g., {a, b, c, d}).
    Quad(i64, i64, i64, i64),
    /// Binary tree for Stage 5 tree synthesis (index 0 is root, negative indices mean null).
    Tree(Vec<TreeNode>),
    /// A heterogeneous, arbitrary-arity tuple of values. Generalizes `Pair`/`Quad`
    /// beyond the 2-/4-int special cases so any positional composite (e.g. a
    /// `Some(x)` tag, a non-int pair, a 3-tuple) round-trips on the wire. `Pair`
    /// and `Quad` are kept as-is for the existing call sites; new structural
    /// shapes that don't fit them use `Tuple`.
    Tuple(Vec<Value>),
    /// A named struct with named, arbitrarily-typed fields, stored as
    /// (name, value) pairs in a CANONICAL (name-sorted) order so equality and
    /// serialization are deterministic. Generalizes the 2-/4-int struct cases
    /// that previously had to collapse onto `Pair`/`Quad`.
    Struct(Vec<(String, Value)>),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(b) => write!(f, "{}", f64::from_bits(*b)),
            Value::Str(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Array(a) => write!(
                f,
                "[{}]",
                a.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Pair(a, b) => write!(f, "({a}, {b})"),
            Value::Quad(a, b, c, d) => write!(f, "({a}, {b}, {c}, {d})"),
            Value::Tree(nodes) => {
                write!(f, "Tree[")?;
                for (i, node) in nodes.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "({}, {}, {})", node.value, node.left, node.right)?;
                }
                write!(f, "]")
            }
            Value::Tuple(vs) => write!(
                f,
                "({})",
                vs.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Value::Struct(fields) => write!(
                f,
                "{{{}}}",
                fields
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

impl Value {
    /// Back-compat accessor for the numeric solvers, which only operate on
    /// integer arrays. Returns an owned `Vec<i64>` iff *every* element is a
    /// `Value::Int` (it cannot borrow `&[i64]` out of a `Vec<Value>`). A typed,
    /// nested, or otherwise non-integer array yields `None`, so a numeric solver
    /// cleanly refuses rather than fabricating values from the wrong shape.
    pub fn as_i64_slice(&self) -> Option<Vec<i64>> {
        match self {
            Value::Array(elems) => elems
                .iter()
                .map(|e| match e {
                    Value::Int(v) => Some(*v),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    /// Borrow the element vector of an array value (`Vec<Value>`), or `None` for
    /// a non-array. Lets consumers that genuinely handle heterogeneous/nested
    /// arrays inspect the wire elements without copying.
    pub fn as_value_slice(&self) -> Option<&[Value]> {
        match self {
            Value::Array(elems) => Some(elems.as_slice()),
            _ => None,
        }
    }

    /// Construct an integer array value from a slice of `i64` (each wrapped as
    /// `Value::Int`). The companion of the module-private `array` helper for use
    /// outside this module.
    pub fn int_array(values: &[i64]) -> Value {
        Value::Array(values.iter().copied().map(Value::Int).collect())
    }

    /// Construct an array value from already-typed elements: typed scalars
    /// (`Str`/`Float`/`Bool`), nested arrays (`Array`), or struct/pair elements.
    /// This is the constructor for the typed/nested arrays the widened wire type
    /// now supports (the integer case has the dedicated [`Value::int_array`]).
    pub fn array_of(values: Vec<Value>) -> Value {
        Value::Array(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example {
    pub inputs: Vec<Value>,
    pub expected: Value,
}

impl Example {
    /// The expected output as an i64. Numeric solvers operate only on integer
    /// problems, so this returns the int payload (0 for non-int outputs, which
    /// those solvers never see).
    pub fn expected_int(&self) -> i64 {
        match &self.expected {
            Value::Int(i) => *i,
            Value::Pair(a, _) => *a,
            Value::Quad(a, _, _, _) => *a,
            _ => 0,
        }
    }

    /// The expected output as a bool. Returns `None` for non-bool expected
    /// outputs so the caller (typically the predicate-style classifier or a
    /// structural branch solver) can refuse cleanly instead of fabricating a
    /// default.
    pub fn expected_bool(&self) -> Option<bool> {
        match &self.expected {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The expected output as an f64 (the float-regression lane). An `Int`
    /// expected output is widened; a `Float` is recovered from its bits.
    pub fn expected_f64(&self) -> Option<f64> {
        match &self.expected {
            Value::Float(b) => Some(f64::from_bits(*b)),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// A single scalar input coerced to f64 (`Int` or `Float`), or None for
/// non-scalar inputs. Used by the float-regression solver.
pub fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(b) => Some(f64::from_bits(*b)),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// Definition of a single function within a multi-function program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionDef {
    pub name: String,
    pub signature: String,
    pub examples: Vec<Example>,
    pub entry_point: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Problem {
    pub name: String,
    pub category: &'static str,
    pub description: &'static str,
    pub signature: &'static str,
    pub examples: Vec<Example>,
    pub holdouts: Vec<Example>,
    pub reference_code: &'static str,
    pub synthetic_args: Vec<String>,
    pub synthetic_values: Vec<Vec<i64>>,
    pub recursive_allowed: bool,
    pub tree_input: bool,
    pub explicit_stack: bool,
    pub functions: Vec<FunctionDef>,
}

impl Problem {
    /// Return the information that a synthesizer is allowed to observe.
    ///
    /// Holdouts and reference implementations are evaluator-owned oracles. They
    /// must never influence candidate generation, routing, ranking, or acceptance.
    /// Keeping this boundary on the data type makes the public solver entrypoints
    /// safe even when callers pass a benchmark `Problem` directly.
    pub fn synthesis_view(&self) -> Self {
        let mut problem = self.clone();
        problem.holdouts.clear();
        problem.reference_code = "";
        problem
    }

    pub fn function_name(&self) -> &str {
        self.signature
            .split_once("fn ")
            .and_then(|(_, rest)| rest.split_once('('))
            .map(|(name, _)| name.trim())
            .unwrap_or("")
    }

    /// True when the function returns a string (used to pick the right print
    /// builtin and expected-output rendering). Case-insensitive so `-> String`
    /// and `-> string` both match.
    fn returns_string(&self) -> bool {
        self.signature
            .replace(' ', "")
            .to_ascii_lowercase()
            .contains("->string")
    }

    /// True when the function returns an array (`-> [i64]`), so the wrapper
    /// prints with the array-capable `println` rather than `println_i64`.
    fn returns_array(&self) -> bool {
        self.signature.replace(' ', "").contains("->[")
    }

    pub fn expected_stdout(&self) -> String {
        self.examples
            .iter()
            .map(|example| render_expected(&example.expected))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn build_wrapper(&self) -> Result<String, String> {
        let fn_name = self.function_name();
        // String- and array-returning functions print with the generic
        // `println` (which renders strings raw and arrays as "[a, b, c]");
        // integer functions print with `println_i64`.
        let print = if self.returns_string() || self.returns_array() {
            "println"
        } else {
            "println_i64"
        };
        let mut lines = vec!["fn main() -> i64 {".to_string()];
        for (example_idx, example) in self.examples.iter().enumerate() {
            let mut args = example
                .inputs
                .iter()
                .map(|value| render_value(self, value))
                .collect::<Result<Vec<_>, _>>()?;

            // Append synthetic args if present
            for arg_idx in 0..self.synthetic_args.len() {
                if let Some(val) = self.synthetic_arg_value(arg_idx, example_idx) {
                    args.push(val.to_string());
                }
            }

            let args_str = args.join(", ");
            lines.push(format!("    {print}({fn_name}({args_str}));"));
        }
        lines.push("    return 0;".to_string());
        lines.push("}".to_string());
        Ok(lines.join("\n"))
    }

    pub fn wrap_program(&self, generated_code: &str) -> Result<String, String> {
        Ok(format!(
            "{}\n\n{}\n",
            generated_code.trim_end(),
            self.build_wrapper()?
        ))
    }

    /// Check if this problem has synthetic args (e.g., time parameter).
    pub fn has_synthetic_args(&self) -> bool {
        !self.synthetic_args.is_empty()
    }

    /// Get synthetic arg value for a given example index.
    /// Returns None if no synthetic args are defined or index is out of bounds.
    pub fn synthetic_arg_value(&self, arg_index: usize, example_index: usize) -> Option<i64> {
        if arg_index < self.synthetic_values.len() {
            self.synthetic_values[arg_index].get(example_index).copied()
        } else {
            None
        }
    }
}

fn example(inputs: Vec<Value>, expected: i64) -> Example {
    Example {
        inputs,
        expected: Value::Int(expected),
    }
}

/// An example with a struct (Quad) expected output.
fn example_quad(inputs: Vec<Value>, expected: Value) -> Example {
    Example { inputs, expected }
}

/// An example with a string expected output (for string-returning problems).
#[allow(dead_code)]
fn example_str(inputs: Vec<Value>, expected: &str) -> Example {
    Example {
        inputs,
        expected: Value::Str(expected.to_string()),
    }
}

/// Render an expected output to match what the wrapper's print builtin emits.
fn render_expected(value: &Value) -> String {
    match value {
        Value::Int(v) => v.to_string(),
        Value::Float(b) => format!("{:.7}", f64::from_bits(*b)),
        Value::Str(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Array(a) => format!(
            "[{}]",
            a.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Pair(a, b) => format!("({a}, {b})"),
        Value::Quad(a, b, c, d) => format!("({a}, {b}, {c}, {d})"),
        Value::Tree(nodes) => {
            let node_strs: Vec<String> = nodes
                .iter()
                .map(|n| format!("({},{},{})", n.value, n.left, n.right))
                .collect();
            format!("Tree[{}]", node_strs.join(";"))
        }
        Value::Tuple(vs) => format!(
            "({})",
            vs.iter()
                .map(render_expected)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Struct(fields) => format!(
            "{{{}}}",
            fields
                .iter()
                .map(|(k, v)| format!("{k}: {}", render_expected(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn int(v: i64) -> Value {
    Value::Int(v)
}

fn string(v: &str) -> Value {
    Value::Str(v.to_string())
}

/// Build an integer array value. Signature is unchanged (`&[i64] -> Value`) so
/// the ~hundreds of in-module call sites are untouched; the widening to
/// `Vec<Value>` is absorbed here by wrapping each element as `Value::Int`.
fn array(v: &[i64]) -> Value {
    Value::Array(v.iter().copied().map(Value::Int).collect())
}

fn pair(a: i64, b: i64) -> Value {
    Value::Pair(a, b)
}

fn quad(a: i64, b: i64, c: i64, d: i64) -> Value {
    Value::Quad(a, b, c, d)
}

fn tree(nodes: Vec<TreeNode>) -> Value {
    Value::Tree(nodes)
}

fn tree_node(value: i64, left: i32, right: i32) -> TreeNode {
    TreeNode { value, left, right }
}

fn tree_from_edges(edges: Vec<(i64, i32, i32)>) -> Value {
    let nodes = edges
        .into_iter()
        .map(|(value, left, right)| TreeNode { value, left, right })
        .collect();
    Value::Tree(nodes)
}

fn get_tree_root(value: &Value) -> Option<&[TreeNode]> {
    match value {
        Value::Tree(nodes) => {
            if nodes.is_empty() {
                None
            } else {
                Some(nodes)
            }
        }
        _ => None,
    }
}

fn tree_size(tree: &[TreeNode]) -> usize {
    tree.len()
}

fn render_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn render_value(problem: &Problem, value: &Value) -> Result<String, String> {
    match value {
        Value::Int(v) => Ok(v.to_string()),
        Value::Float(b) => Ok(format!("{:.7}", f64::from_bits(*b))),
        Value::Str(v) => Ok(render_string(v)),
        Value::Bool(b) => Ok(if *b {
            "true".to_string()
        } else {
            "false".to_string()
        }),
        Value::Array(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
        Value::Pair(a, b) => {
            if problem.signature.contains("Point") {
                Ok(format!("Point {{ x: {a}, y: {b} }}"))
            } else if problem.signature.contains("Rectangle") {
                Ok(format!("Rectangle {{ width: {a}, height: {b} }}"))
            } else {
                Err(format!(
                    "cannot render pair literal for {} with signature {}",
                    problem.name, problem.signature
                ))
            }
        }
        Value::Quad(a, b, c, d) => {
            if problem.signature.contains("DualTally") {
                Ok(format!(
                    "DualTally {{ pos_count: {a}, neg_count: {b}, zero_count: {c}, total: {d} }}"
                ))
            } else if problem.signature.contains("RateLimiter") {
                Ok(format!(
                    "RateLimiter {{ total: {a}, exceeded: {b}, count: {c}, limit_reached: {d} }}"
                ))
            } else if problem.signature.contains("RunningCorrelation") {
                Ok(format!(
                    "RunningCorrelation {{ sum_x: {a}, sum_y: {b}, sum_xy: {c}, count: {d} }}"
                ))
            } else if problem.signature.contains("MutualInfo") {
                Ok(format!(
                    "MutualInfo {{ joint_00: {a}, joint_01: {b}, joint_10: {c}, joint_11: {d} }}"
                ))
            } else if problem.signature.contains("ThresholdClassifier") {
                Ok(format!(
                    "ThresholdClassifier {{ below: {a}, between: {b}, above: {c}, total: {d} }}"
                ))
            } else if problem.signature.contains("PairedExtrema") {
                Ok(format!(
                    "PairedExtrema {{ min: {a}, min_idx: {b}, max: {c}, max_idx: {d} }}"
                ))
            } else {
                Err(format!(
                    "cannot render quad literal for {} with signature {}",
                    problem.name, problem.signature
                ))
            }
        }
        Value::Tree(_nodes) => Err(format!(
            "tree rendering not yet implemented for {}",
            problem.name
        )),
        // A positional tuple has no field names and no signature mapping, so we
        // cannot reconstruct a struct-literal the wrapper would print — mirror the
        // Pair/Quad "no known signature" path and the Tree arm by erroring.
        Value::Tuple(_) => Err(format!(
            "cannot render tuple literal for {} with signature {}",
            problem.name, problem.signature
        )),
        // A named struct carries its own field names, so render it generically as
        // `{ field: v, ... }` (the field values render recursively).
        Value::Struct(fields) => {
            let rendered = fields
                .iter()
                .map(|(k, v)| render_value(problem, v).map(|s| format!("{k}: {s}")))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("{{ {} }}", rendered.join(", ")))
        }
    }
}

fn problem(
    name: &str,
    variant: usize,
    category: &'static str,
    description: &'static str,
    signature: &'static str,
    examples: Vec<Example>,
    holdouts: Vec<Example>,
    reference_code: &'static str,
) -> Problem {
    Problem {
        name: format!("{name}_v{variant}"),
        category,
        description,
        signature,
        examples,
        holdouts,
        reference_code,
        synthetic_args: Vec::new(),
        synthetic_values: Vec::new(),
        recursive_allowed: false,
        tree_input: false,
        explicit_stack: false,
        functions: vec![],
    }
}

pub type Factory = fn(usize) -> Problem;

pub fn get_benchmark(variants_per_factory: usize) -> Vec<Problem> {
    let mut problems = Vec::new();
    for variant in 0..variants_per_factory {
        for factory in FACTORIES {
            problems.push(factory(variant));
        }
    }
    problems
}

pub fn factory_count() -> usize {
    FACTORIES.len()
}

/// String-output benchmark problems (solved by the main pipeline's string path).
pub fn get_string_benchmark(variants_per_factory: usize) -> Vec<Problem> {
    let mut problems = Vec::new();
    for variant in 0..variants_per_factory {
        for factory in STRING_FACTORIES {
            problems.push(factory(variant));
        }
    }
    problems
}

/// Parameter type for holdout input sampling. A local, minimal mirror of
/// `solver::signature::ParamType` (which is `pub(super)` and unreachable here);
/// duplicating these ~20 lines keeps `benchmark` decoupled from the solver's
/// private API rather than widening that API's surface.
#[derive(Clone, Debug, PartialEq, Eq)]
enum HoldoutParamType {
    I64,
    F64,
    ArrayI64,
    String,
    /// A type we cannot yet sample faithfully — forces a fallback to the
    /// hand-authored holdouts rather than emitting wrong inputs.
    Other,
}

/// Parse the comma-separated parameter types out of a `fn name(a: T, b: U) -> R`
/// signature. Mirrors `solver::signature::parse_param_types` for the four
/// sampleable types; everything else is `Other`.
fn holdout_param_types(signature: &str) -> Vec<HoldoutParamType> {
    let params = signature
        .split_once('(')
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(params, _)| params)
        .unwrap_or("")
        .trim();
    if params.is_empty() {
        return Vec::new();
    }
    params
        .split(',')
        .map(|param| {
            let ty = param
                .split_once(':')
                .map(|(_, ty)| ty.trim())
                .unwrap_or_default();
            match ty {
                "i64" => HoldoutParamType::I64,
                "f64" => HoldoutParamType::F64,
                "[i64]" => HoldoutParamType::ArrayI64,
                "string" => HoldoutParamType::String,
                _ => HoldoutParamType::Other,
            }
        })
        .collect()
}

/// Deterministic 64-bit seed from a problem name (FNV-1a). Stable across runs
/// and machines — the whole point is reproducible holdout inputs with no clock
/// and no RNG. A non-zero offset avoids an all-zero seed for the empty string.
fn holdout_seed(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^ 0x9e3779b97f4a7c15
}

/// Tiny deterministic LCG used to sample holdout inputs (same constants as
/// `program_trace::InputSampler`, kept local to avoid a benchmark<->program_trace
/// module cycle). No `std::time`, no `rand`.
struct HoldoutRng {
    state: u64,
}

impl HoldoutRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state >> 33
    }
    /// Next i64 in the inclusive range `[min, max]`.
    fn next_in(&mut self, min: i64, max: i64) -> i64 {
        let span = (max as i128 - min as i128 + 1).max(1) as u128;
        min + (self.next_u64() as u128 % span) as i64
    }
}

/// How many fresh holdout points to attempt per problem. Widened (was 12) so the
/// generalization probe actually reaches into the wider input range below.
const HOLDOUT_SAMPLES: usize = 24;
/// Input scalar range — widened (was [-12,24]) so a candidate that only agrees
/// with the reference on the narrow old window is exercised on values it has not
/// seen. Kept small enough that quadratic reference math (`x*x`) cannot overflow
/// i64: 64*64 = 4096, comfortably within range.
const HOLDOUT_MIN: i64 = -64;
const HOLDOUT_MAX: i64 = 64;
/// Max sampled array length (was 6).
const HOLDOUT_ARRAY_MAX_LEN: usize = 10;

/// Salt XOR'd into the seed used by [`problem_from_reference`] when sampling the
/// *visible* seed examples. `generated_holdouts` seeds from `holdout_seed(name)`
/// un-salted, so without a distinct salt the strict verifier's holdout probes
/// would draw the SAME inputs as the visible examples — proving nothing about
/// generalization. This makes the seed-example draws disjoint from the holdout
/// draws so the differential oracle is exercised on inputs the candidate has not
/// seen.
const SEED_EXAMPLE_SALT: u64 = 0x517cc1b727220a95;

/// Salt for the input draws used by [`verify_code_against_property`]. Distinct
/// from both `SEED_EXAMPLE_SALT` and the un-salted holdout seed so a property
/// candidate is probed on its own fresh inputs.
const PROPERTY_SAMPLE_SALT: u64 = 0x2545f4914f6cdd1d;

/// Build one sampled input vector matching `param_types`, or `None` if any
/// param is a type we cannot sample (caller then falls back to hand holdouts).
fn sample_holdout_inputs(rng: &mut HoldoutRng, param_types: &[HoldoutParamType]) -> Option<Vec<Value>> {
    let mut inputs = Vec::with_capacity(param_types.len());
    for ty in param_types {
        match ty {
            HoldoutParamType::I64 => inputs.push(Value::Int(rng.next_in(HOLDOUT_MIN, HOLDOUT_MAX))),
            HoldoutParamType::F64 => {
                // Sample a small rational so reference float math stays sane.
                let num = rng.next_in(HOLDOUT_MIN, HOLDOUT_MAX) as f64;
                inputs.push(Value::Float((num).to_bits()));
            }
            HoldoutParamType::ArrayI64 => {
                let len = rng.next_in(0, HOLDOUT_ARRAY_MAX_LEN as i64) as usize;
                let elems: Vec<Value> = (0..len)
                    .map(|_| Value::Int(rng.next_in(HOLDOUT_MIN, HOLDOUT_MAX)))
                    .collect();
                inputs.push(Value::Array(elems));
            }
            HoldoutParamType::String => {
                // Deterministic lowercase string of length 1..=5.
                let len = rng.next_in(1, 5) as usize;
                let s: String = (0..len)
                    .map(|_| (b'a' + (rng.next_in(0, 25) as u8)) as char)
                    .collect();
                inputs.push(Value::Str(s));
            }
            HoldoutParamType::Other => return None,
        }
    }
    Some(inputs)
}

/// Holdout examples for `problem`, used by the STRICT verifier to catch
/// candidates that overfit the visible examples.
///
/// Historically this returned `problem.holdouts.clone()` — a fixed handful of
/// hand-authored points, so a candidate that merely matched those passed. Now,
/// when the problem ships a runnable `reference_code`, we deterministically
/// sample FRESH inputs of the signature's types, run the REFERENCE over them,
/// and use the reference's outputs as the expected values. The candidate is
/// NEVER consulted here, so these are true generalization probes the candidate
/// has not seen.
///
/// Soundness fallbacks (never fabricate, never trust the candidate):
///   - empty `reference_code` → hand-authored holdouts (current behavior).
///   - a signature with an unsampleable (`Other`) parameter → hand holdouts.
///   - the reference erroring / returning an unrepresentable value on a sample
///     → that point is skipped (a reference error is NOT a candidate failure).
///   - if NO point survives sampling → hand holdouts (keeps the per-problem
///     "holdouts are non-empty" invariant the benchmark relies on).
pub fn generated_holdouts(problem: &Problem) -> Vec<Example> {
    generated_holdouts_with_source(problem).0
}

/// Where a problem's strict-verify holdouts came from.
///
/// `Generated` means the points were sampled fresh and labelled by RUNNING the
/// problem's runnable `reference_code` — a true generalization probe on inputs
/// the candidate has not seen. `HandFallback` means we degraded to the
/// hand-authored `problem.holdouts` (no reference, an unsampleable signature, or
/// nothing survived sampling); those are NOT differential generalization
/// evidence, so a "verified by generalization" metric must EXCLUDE them. A pass
/// over `HandFallback` holdouts is still a valid example-style pass — it just
/// must not be counted as strict-by-generalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldoutSource {
    /// Points labelled by running the reference over freshly sampled inputs.
    Generated,
    /// Degraded to the hand-authored holdouts (keyed on `reference_code.is_empty()`
    /// or an unsampleable signature / empty sampling result).
    HandFallback,
}

/// Like [`generated_holdouts`] but also reports the [`HoldoutSource`] so callers
/// can avoid silently counting a hand-fallback pass as a generalization pass.
/// The fallback stays keyed on `reference_code.is_empty()` (and the existing
/// unsampleable-signature guards); no input-type-based logic is introduced.
pub fn generated_holdouts_with_source(problem: &Problem) -> (Vec<Example>, HoldoutSource) {
    // No oracle to run → keep the hand-authored holdouts.
    if problem.reference_code.is_empty() {
        return (problem.holdouts.clone(), HoldoutSource::HandFallback);
    }
    let param_types = holdout_param_types(problem.signature);
    // Zero-arg functions or any unsampleable parameter → fall back.
    if param_types.is_empty() || param_types.contains(&HoldoutParamType::Other) {
        return (problem.holdouts.clone(), HoldoutSource::HandFallback);
    }
    let fn_name = problem.function_name();
    if fn_name.is_empty() {
        return (problem.holdouts.clone(), HoldoutSource::HandFallback);
    }

    let mut rng = HoldoutRng::new(holdout_seed(&problem.name));
    let mut generated = Vec::with_capacity(HOLDOUT_SAMPLES);
    for _ in 0..HOLDOUT_SAMPLES {
        let Some(inputs) = sample_holdout_inputs(&mut rng, &param_types) else {
            // Should not happen (Other already filtered), but stay safe.
            return (problem.holdouts.clone(), HoldoutSource::HandFallback);
        };
        // Run the REFERENCE — the ONLY source of expected values here.
        let out = match crate::runtime::execute_function_for_problem(
            problem.reference_code,
            fn_name,
            &inputs,
            problem,
        ) {
            Ok(v) => v,
            // A reference error on this input (e.g. arr[0] on an empty array) is
            // not a candidate failure — skip the point.
            Err(_) => continue,
        };
        let expected = match crate::runtime::benchmark_value_from_runtime(&out) {
            Ok(v) => v,
            Err(_) => continue,
        };
        generated.push(Example { inputs, expected });
    }

    // If sampling produced nothing runnable, fall back so the invariant that
    // every problem yields non-empty holdouts is preserved.
    if generated.is_empty() {
        return (problem.holdouts.clone(), HoldoutSource::HandFallback);
    }
    (generated, HoldoutSource::Generated)
}

/// REFERENCE-IMPLEMENTATION front door: build a solvable [`Problem`] from a
/// runnable reference implementation alone — no hand-authored I/O examples.
///
/// This closes the "examples-only spec front door" root cause: previously every
/// spec had to arrive as literal I/O examples or it failed. Here the caller
/// supplies a reference whose behavior IS the specification ("synthesize a
/// function equivalent to THIS code"). We:
///   1. parse the signature's parameter types (must all be sampleable);
///   2. deterministically sample a handful of inputs and RUN the reference over
///      them via the U1 safe execute path
///      ([`crate::runtime::execute_function_for_problem`], which sets
///      `verify_mode` + `run_isolated`) to manufacture the seed `io_examples`;
///   3. KEEP `reference_code` set on the returned problem so the existing strict
///      verifier ([`crate::runtime::verify_problem_code_strict`]) then runs
///      [`generated_holdouts`] for differential testing against the reference.
///
/// The seed examples drive `solve_problem`/`build_wrapper` exactly like
/// hand-authored examples, so the whole solve+verify pipeline works unchanged.
///
/// `signature` and `reference_code` are `&'static str`. Callers holding owned
/// `String`s should `Box::leak` them at the front door (the existing pattern in
/// `agent::coding_intent::CodingIntent::problem_from_reference`).
///
/// Errors (never fabricate a "solved" spec from an unrunnable reference):
///   - the signature has zero parameters or any unsampleable (`Other`) type;
///   - the function name cannot be parsed from the signature;
///   - the reference errors / returns unrepresentable values on *every* sample
///     (zero seed examples survive).
pub fn problem_from_reference(
    name: &str,
    signature: &'static str,
    reference_code: &'static str,
) -> Result<Problem, String> {
    let param_types = holdout_param_types(signature);
    if param_types.is_empty() {
        return Err(format!(
            "cannot sample a reference with zero parameters (signature: {signature:?})"
        ));
    }
    if param_types.contains(&HoldoutParamType::Other) {
        return Err(format!(
            "signature has an unsampleable parameter type, cannot manufacture seed examples \
             (signature: {signature:?})"
        ));
    }

    // Build a temporary problem so `execute_function_for_problem` carries the
    // right signature/reference metadata. `reference_code` is set here AND on
    // the returned problem (it is the verifier-owned oracle).
    let mut problem = Problem {
        name: name.to_string(),
        category: "reference",
        description: "synthesize a function equivalent to the provided reference implementation",
        signature,
        examples: Vec::new(),
        holdouts: Vec::new(),
        reference_code,
        ..Default::default()
    };

    let fn_name = problem.function_name();
    if fn_name.is_empty() {
        return Err(format!(
            "cannot parse function name from signature {signature:?}"
        ));
    }
    // `fn_name` borrows `problem.signature` ('static), so it outlives the
    // mutable borrow below — copy to a `&str` independent of `problem` to push
    // examples while `fn_name` is still alive.
    let fn_name: &str = fn_name;

    // Seed the visible-example RNG with a salt distinct from `generated_holdouts`
    // (which uses `holdout_seed(name)` un-salted) so the strict verifier's
    // holdout probes don't coincide with these visible examples.
    let mut rng = HoldoutRng::new(holdout_seed(name) ^ SEED_EXAMPLE_SALT);
    let mut examples = Vec::with_capacity(HOLDOUT_SAMPLES);
    for _ in 0..HOLDOUT_SAMPLES {
        let Some(inputs) = sample_holdout_inputs(&mut rng, &param_types) else {
            // `Other` already filtered above; stay safe.
            break;
        };
        // Run the REFERENCE on the safe verify path to get the expected output.
        let out = match crate::runtime::execute_function_for_problem(
            reference_code,
            fn_name,
            &inputs,
            &problem,
        ) {
            Ok(v) => v,
            // A reference error on this input (e.g. `arr[0]` on an empty array)
            // is not a usable seed — skip the point.
            Err(_) => continue,
        };
        let expected = match crate::runtime::benchmark_value_from_runtime(&out) {
            Ok(v) => v,
            Err(_) => continue,
        };
        examples.push(Example { inputs, expected });
    }

    if examples.is_empty() {
        return Err(format!(
            "reference produced no representable outputs over {HOLDOUT_SAMPLES} sampled inputs; \
             cannot manufacture seed examples for {name}"
        ));
    }

    problem.examples = examples;
    Ok(problem)
}

/// PROPERTY front door: verify a candidate against a Mog PREDICATE oracle
/// instead of fixed I/O examples.
///
/// This is the third spec front door (alongside hand examples and
/// [`problem_from_reference`]): the specification is a *predicate* the output
/// must satisfy, not a table of expected outputs. Concretely "synthesize a
/// function whose result satisfies THIS property" — e.g. a sort whose output
/// `is_sorted`, where no single expected output is pinned down.
///
/// The predicate is an ordinary Mog function whose parameters are the
/// candidate's parameters FOLLOWED BY one trailing parameter bound to the
/// candidate's output, returning a truthy `i64` (`0`/`1`) or `bool`. We:
///   1. parse the candidate signature's parameter types (must all be sampleable);
///   2. deterministically sample inputs (own salt → disjoint from example /
///      holdout draws) and RUN the candidate over them via the U1 safe execute
///      path ([`crate::runtime::execute_function_for_problem`]);
///   3. RUN the predicate on `(inputs.., output)` and require it to hold on
///      every sample.
///
/// A candidate is accepted iff the predicate holds on all sampled inputs; the
/// first violation (or a candidate/predicate error) is an `Err`. This is the
/// "Mog predicate as verify oracle" — the property arm's verification ceiling.
/// (It is a *verifier*; property-driven *search* — enumerating candidates to
/// satisfy the predicate — is a separate, larger step and is out of scope here.)
///
/// `candidate_signature` and `predicate_signature` are `&'static str` because
/// they are stored on the temporary [`Problem`]s used for argument coercion;
/// callers holding owned `String`s should `Box::leak` at the front door (the
/// existing pattern in `agent::coding_intent`).
pub fn verify_code_against_property(
    candidate_name: &str,
    candidate_signature: &'static str,
    candidate_code: &str,
    predicate_name: &str,
    predicate_signature: &'static str,
    predicate_code: &str,
) -> Result<(), String> {
    let param_types = holdout_param_types(candidate_signature);
    if param_types.is_empty() {
        return Err(format!(
            "cannot sample a property spec with zero candidate parameters \
             (signature: {candidate_signature:?})"
        ));
    }
    if param_types.contains(&HoldoutParamType::Other) {
        return Err(format!(
            "candidate signature has an unsampleable parameter type \
             (signature: {candidate_signature:?})"
        ));
    }

    // Temporary problems carry the signatures so argument coercion
    // (`runtime_value_from_problem_meta`) sees the right parameter types.
    let cand_problem = Problem {
        name: candidate_name.to_string(),
        category: "property",
        description: "candidate under a property predicate",
        signature: candidate_signature,
        ..Default::default()
    };
    let pred_problem = Problem {
        name: predicate_name.to_string(),
        category: "property",
        description: "property predicate oracle",
        signature: predicate_signature,
        ..Default::default()
    };

    let mut rng = HoldoutRng::new(holdout_seed(candidate_name) ^ PROPERTY_SAMPLE_SALT);
    let mut checked = 0usize;
    for _ in 0..HOLDOUT_SAMPLES {
        let Some(inputs) = sample_holdout_inputs(&mut rng, &param_types) else {
            break;
        };
        let out = crate::runtime::execute_function_for_problem(
            candidate_code,
            candidate_name,
            &inputs,
            &cand_problem,
        )
        .map_err(|e| format!("candidate errored on a sampled input: {e}"))?;
        let out_bench = crate::runtime::benchmark_value_from_runtime(&out)
            .map_err(|e| format!("candidate output not representable: {e}"))?;

        // Predicate sees the candidate's inputs followed by its output.
        let mut pred_inputs = inputs.clone();
        pred_inputs.push(out_bench);
        let verdict = crate::runtime::execute_function_for_problem(
            predicate_code,
            predicate_name,
            &pred_inputs,
            &pred_problem,
        )
        .map_err(|e| format!("property predicate errored: {e}"))?;

        let holds = match verdict {
            crate::runtime::Value::Int(n) => n != 0,
            crate::runtime::Value::Bool(b) => b,
            other => {
                return Err(format!(
                    "property predicate returned a non-boolean value: {other:?}"
                ))
            }
        };
        if !holds {
            return Err(format!(
                "candidate violates the property on sampled input {inputs:?} (output {out:?})"
            ));
        }
        checked += 1;
    }

    if checked == 0 {
        return Err(format!(
            "no property samples could be drawn for {candidate_name}"
        ));
    }
    Ok(())
}

fn make_add_two(variant: usize) -> Problem {
    problem(
        "add_two",
        variant,
        "arithmetic",
        "Return the sum of two i64 integers.",
        "fn add_two(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(2), int(3)], 5),
            example(vec![int(10), int(-4)], 6),
            example(vec![int(7), int(8)], 15),
            example(vec![int(-3), int(-2)], -5),
        ],
        vec![
            example(vec![int(100), int(-37)], 63),
            example(vec![int(-12), int(-8)], -20),
        ],
        "fn add_two(a: i64, b: i64) -> i64 {\n    return a + b;\n}\n",
    )
}

fn make_abs_diff(variant: usize) -> Problem {
    problem(
        "abs_diff",
        variant,
        "arithmetic",
        "Return the absolute difference between two integers.",
        "fn abs_diff(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(7)], 4),
            example(vec![int(7), int(3)], 4),
            example(vec![int(0), int(5)], 5),
            example(vec![int(-3), int(2)], 5),
        ],
        vec![
            example(vec![int(-10), int(7)], 17),
            example(vec![int(9), int(-4)], 13),
        ],
        "fn abs_diff(a: i64, b: i64) -> i64 {\n    if a > b {\n        return a - b;\n    } else {\n        return b - a;\n    }\n}\n",
    )
}

fn make_max2(variant: usize) -> Problem {
    problem(
        "max2",
        variant,
        "control_flow",
        "Return the larger of two integers.",
        "fn max2(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(2), int(3)], 3),
            example(vec![int(10), int(-4)], 10),
            example(vec![int(7), int(7)], 7),
            example(vec![int(-3), int(-2)], -2),
        ],
        vec![
            example(vec![int(-3), int(9)], 9),
            example(vec![int(12), int(12)], 12),
        ],
        "fn max2(a: i64, b: i64) -> i64 {\n    if a > b {\n        return a;\n    } else {\n        return b;\n    }\n}\n",
    )
}

fn make_clamp(variant: usize) -> Problem {
    problem(
        "clamp_0_100",
        variant,
        "control_flow",
        "Clamp x into the closed range [0, 100].",
        "fn clamp_0_100(x: i64) -> i64",
        vec![
            example(vec![int(-5)], 0),
            example(vec![int(0)], 0),
            example(vec![int(37)], 37),
            example(vec![int(140)], 100),
        ],
        vec![
            example(vec![int(-1)], 0),
            example(vec![int(101)], 100),
            example(vec![int(42)], 42),
        ],
        "fn clamp_0_100(x: i64) -> i64 {\n    if x < 0 {\n        return 0;\n    }\n    if x > 100 {\n        return 100;\n    }\n    return x;\n}\n",
    )
}

fn make_sign(variant: usize) -> Problem {
    problem(
        "sign",
        variant,
        "control_flow",
        "Return -1 for negative, 0 for zero, and 1 for positive.",
        "fn sign(x: i64) -> i64",
        vec![
            example(vec![int(-5)], -1),
            example(vec![int(0)], 0),
            example(vec![int(7)], 1),
            example(vec![int(3)], 1),
        ],
        vec![
            example(vec![int(-8)], -1),
            example(vec![int(0)], 0),
            example(vec![int(15)], 1),
        ],
        "fn sign(x: i64) -> i64 {\n    if x < 0 {\n        return -1;\n    }\n    if x > 0 {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_sum_to_n(variant: usize) -> Problem {
    problem(
        "sum_to_n",
        variant,
        "arithmetic",
        "Return 1 + 2 + ... + n. For n <= 0 return 0.",
        "fn sum_to_n(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(5)], 15),
            example(vec![int(10)], 55),
        ],
        vec![example(vec![int(7)], 28), example(vec![int(-3)], 0)],
        "fn sum_to_n(n: i64) -> i64 {\n    if n <= 0 {\n        return 0;\n    }\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_gcd(variant: usize) -> Problem {
    problem(
        "gcd",
        variant,
        "arithmetic",
        "Return the greatest common divisor of two positive integers.",
        "fn gcd(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(12), int(18)], 6),
            example(vec![int(21), int(14)], 7),
            example(vec![int(9), int(28)], 1),
            example(vec![int(48), int(18)], 6),
        ],
        vec![
            example(vec![int(270), int(192)], 6),
            example(vec![int(17), int(13)], 1),
        ],
        "fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n",
    )
}

fn make_lcm(variant: usize) -> Problem {
    problem(
        "lcm",
        variant,
        "arithmetic",
        "Return the least common multiple of two positive integers.",
        "fn lcm(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(4)], 12),
            example(vec![int(6), int(8)], 24),
            example(vec![int(5), int(10)], 10),
            example(vec![int(7), int(9)], 63),
        ],
        vec![
            example(vec![int(8), int(12)], 24),
            example(vec![int(9), int(6)], 18),
        ],
        "fn gcd_inner(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n\nfn lcm(a: i64, b: i64) -> i64 {\n    return (a * b) / gcd_inner(a, b);\n}\n",
    )
}

fn make_array_sum(variant: usize) -> Problem {
    problem(
        "array_sum",
        variant,
        "arrays",
        "Return the sum of all elements in an array of i64 values.",
        "fn array_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5])], 5),
            example(vec![array(&[4, 4])], 8),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn array_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_array_max(variant: usize) -> Problem {
    problem(
        "array_max",
        variant,
        "arrays",
        "Return the largest element in a non-empty array.",
        "fn array_max(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[5])], 5),
            example(vec![array(&[4, 9])], 9),
            example(vec![array(&[2, 7, 1, 0])], 7),
        ],
        vec![
            example(vec![array(&[-3, -9, -1])], -1),
            example(vec![array(&[10, 2, 10])], 10),
        ],
        "fn array_max(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_count_occurrences(variant: usize) -> Problem {
    problem(
        "count_occurrences",
        variant,
        "arrays",
        "Count how many times target appears in arr.",
        "fn count_occurrences(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 1]), int(1)], 2),
            example(vec![array(&[5]), int(5)], 1),
            example(vec![array(&[4, 9]), int(1)], 0),
            example(vec![array(&[2, 7, 2, 0]), int(2)], 2),
        ],
        vec![
            example(vec![array(&[4, 1, 4, 4]), int(4)], 3),
            example(vec![array(&[2, 3]), int(5)], 0),
        ],
        "fn count_occurrences(arr: [i64], target: i64) -> i64 {\n    count: i64 = 0;\n    for item in arr {\n        if item == target {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_trimmed_len(variant: usize) -> Problem {
    problem(
        "trimmed_len",
        variant,
        "strings",
        "Trim leading and trailing spaces and return the remaining length.",
        "fn trimmed_len(s: string) -> i64",
        vec![
            example(vec![string(" mog ")], 3),
            example(vec![string("  diffusion")], 9),
            example(vec![string("compiler  ")], 8),
            example(vec![string("  hello world  ")], 11),
        ],
        vec![
            example(vec![string("   hi there   ")], 8),
            example(vec![string("      ")], 0),
        ],
        "fn trimmed_len(s: string) -> i64 {\n    t := s.trim();\n    return t.len;\n}\n",
    )
}

fn make_vowel_count(variant: usize) -> Problem {
    problem(
        "vowel_count",
        variant,
        "strings",
        "Count vowels (a, e, i, o, u) in a lowercase ASCII string.",
        "fn vowel_count(s: string) -> i64",
        vec![
            example(vec![string("mog")], 1),
            example(vec![string("aeiou")], 5),
            example(vec![string("banana")], 3),
            example(vec![string("rhythm")], 0),
        ],
        vec![
            example(vec![string("queue")], 4),
            example(vec![string("sky")], 0),
        ],
        "fn vowel_count(s: string) -> i64 {\n    chars := s.split(\"\");\n    total: i64 = 0;\n    for ch in chars {\n        if ch == \"a\" { total = total + 1; }\n        if ch == \"e\" { total = total + 1; }\n        if ch == \"i\" { total = total + 1; }\n        if ch == \"o\" { total = total + 1; }\n        if ch == \"u\" { total = total + 1; }\n    }\n    return total;\n}\n",
    )
}

fn make_contains_cat(variant: usize) -> Problem {
    problem(
        "contains_cat",
        variant,
        "strings",
        "Return 1 if the string contains the substring 'cat', else 0.",
        "fn contains_cat(s: string) -> i64",
        vec![
            example(vec![string("cat")], 1),
            example(vec![string("scatter")], 1),
            example(vec![string("dog")], 0),
            example(vec![string("hello")], 0),
        ],
        vec![
            example(vec![string("bobcat")], 1),
            example(vec![string("atlas")], 0),
        ],
        "fn contains_cat(s: string) -> i64 {\n    if s.contains(\"cat\") {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_point_sum(variant: usize) -> Problem {
    problem(
        "point_sum",
        variant,
        "structs",
        "Define struct Point { x: i64, y: i64 } and return x + y.",
        "fn point_sum(p: Point) -> i64",
        vec![
            example(vec![pair(3, 4)], 7),
            example(vec![pair(-1, 2)], 1),
            example(vec![pair(0, 0)], 0),
            example(vec![pair(9, 8)], 17),
        ],
        vec![
            example(vec![pair(12, -5)], 7),
            example(vec![pair(-3, -4)], -7),
        ],
        "struct Point {\n    x: i64,\n    y: i64,\n}\n\nfn point_sum(p: Point) -> i64 {\n    return p.x + p.y;\n}\n",
    )
}

fn make_safe_div_or_neg1(variant: usize) -> Problem {
    problem(
        "safe_div_or_neg1",
        variant,
        "result_optional",
        "Divide a by b. If b is zero, return -1.",
        "fn safe_div_or_neg1(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(10), int(2)], 5),
            example(vec![int(7), int(0)], -1),
            example(vec![int(9), int(3)], 3),
            example(vec![int(5), int(0)], -1),
        ],
        vec![
            example(vec![int(9), int(0)], -1),
            example(vec![int(21), int(7)], 3),
        ],
        "fn helper_div(a: i64, b: i64) -> Result<i64> {\n    if b == 0 {\n        return err(\"division by zero\");\n    }\n    return ok(a / b);\n}\n\nfn safe_div_or_neg1(a: i64, b: i64) -> i64 {\n    r := helper_div(a, b);\n    out: i64 = match r {\n        ok(v) => v,\n        err(e) => -1,\n    };\n    return out;\n}\n",
    )
}

fn make_positive_or_default(variant: usize) -> Problem {
    problem(
        "positive_or_default",
        variant,
        "result_optional",
        "Return x if x is positive, otherwise 0.",
        "fn positive_or_default(x: i64) -> i64",
        vec![
            example(vec![int(10)], 10),
            example(vec![int(0)], 0),
            example(vec![int(-5)], 0),
            example(vec![int(3)], 3),
        ],
        vec![example(vec![int(-4)], 0), example(vec![int(19)], 19)],
        "fn maybe_positive(x: i64) -> ?i64 {\n    if x > 0 {\n        return some(x);\n    }\n    return none;\n}\n\nfn positive_or_default(x: i64) -> i64 {\n    r := maybe_positive(x);\n    out: i64 = match r {\n        some(v) => v,\n        none => 0,\n    };\n    return out;\n}\n",
    )
}

fn make_factorial(variant: usize) -> Problem {
    problem(
        "factorial",
        variant,
        "recursion",
        "Return n! recursively.",
        "fn factorial(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(5)], 120),
            example(vec![int(7)], 5040),
        ],
        vec![
            example(vec![int(3)], 6),
            example(vec![int(6)], 720),
        ],
        "fn factorial(n: i64) -> i64 {\n    if n <= 1 {\n        return 1;\n    }\n    return n * factorial(n - 1);\n}\n",
    )
}

fn make_fibonacci(variant: usize) -> Problem {
    problem(
        "fibonacci",
        variant,
        "recursion",
        "Return the nth Fibonacci number recursively or iteratively.",
        "fn fibonacci(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(7)], 13),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(5)], 5),
            example(vec![int(8)], 21),
        ],
        "fn fibonacci(n: i64) -> i64 {\n    if n <= 0 { return 0; }\n    if n == 1 { return 1; }\n    return fibonacci(n - 1) + fibonacci(n - 2);\n}\n",
    )
}

fn make_closure_map_sum(variant: usize) -> Problem {
    problem(
        "closure_map_sum",
        variant,
        "higher_order",
        "Double every array element with .map() and return the sum of the doubled values.",
        "fn closure_map_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 12),
            example(vec![array(&[2, 2, 2])], 12),
            example(vec![array(&[5, 1, 4])], 20),
            example(vec![array(&[3, 5, 1])], 18),
        ],
        vec![
            example(vec![array(&[4, 6])], 20),
            example(vec![array(&[1, 1, 1, 1])], 8),
        ],
        "fn closure_map_sum(arr: [i64]) -> i64 {\n    doubled := arr.map(fn(x: i64) -> i64 { x * 2 });\n    total: i64 = 0;\n    for item in doubled {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_count_positive(variant: usize) -> Problem {
    problem(
        "count_positive",
        variant,
        "arrays",
        "Count how many elements in the array are greater than zero.",
        "fn count_positive(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[-5])], 0),
            example(vec![array(&[4, -9])], 1),
            example(vec![array(&[2, 7, -1, 0])], 2),
        ],
        vec![
            example(vec![array(&[0, 5, 0])], 1),
            example(vec![array(&[-1, -2, 3, 4])], 2),
        ],
        "fn count_positive(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        if item > 0 {\n            total = total + 1;\n        }\n    }\n    return total;\n}\n",
    )
}

fn make_is_even(variant: usize) -> Problem {
    problem(
        "is_even",
        variant,
        "control_flow",
        "Return 1 if x is even, otherwise 0.",
        "fn is_even(x: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 0),
            example(vec![int(8)], 1),
            example(vec![int(11)], 0),
        ],
        vec![
            example(vec![int(4)], 1),
            example(vec![int(7)], 0),
        ],
        "fn is_even(x: i64) -> i64 {\n    if (x % 2) == 0 {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_digit_sum(variant: usize) -> Problem {
    problem(
        "digit_sum",
        variant,
        "arithmetic",
        "Return the sum of the decimal digits of n.",
        "fn digit_sum(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(7)], 7),
            example(vec![int(123)], 6),
            example(vec![int(9081)], 18),
        ],
        vec![
            example(vec![int(456)], 15),
            example(vec![int(100)], 1),
        ],
        "fn digit_sum(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    total: i64 = 0;\n    while x > 0 {\n        total = total + (x % 10);\n        x = x / 10;\n    }\n    return total;\n}\n",
    )
}

fn make_reverse_digits(variant: usize) -> Problem {
    problem(
        "reverse_digits",
        variant,
        "loops",
        "Reverse the decimal digits of n.",
        "fn reverse_digits(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(120)], 21),
            example(vec![int(907)], 709),
            example(vec![int(4005)], 5004),
        ],
        vec![
            example(vec![int(123)], 321),
            example(vec![int(100)], 1),
        ],
        "fn reverse_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = (acc * 10) + (x % 10);\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_digit_count(variant: usize) -> Problem {
    problem(
        "digit_count",
        variant,
        "loops",
        "Count how many decimal digits n contains.",
        "fn digit_count(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(7)], 1),
            example(vec![int(120)], 3),
            example(vec![int(4005)], 4),
        ],
        vec![
            example(vec![int(99)], 2),
            example(vec![int(1000)], 4),
        ],
        "fn digit_count(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    if x == 0 {\n        return 1;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + 1;\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_even_digits(variant: usize) -> Problem {
    problem(
        "count_even_digits",
        variant,
        "loops",
        "Count how many decimal digits of n are even.",
        "fn count_even_digits(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(7)], 0),
            example(vec![int(120)], 2),
            example(vec![int(4005)], 3),
        ],
        vec![
            example(vec![int(246)], 3),
            example(vec![int(135)], 0),
        ],
        "fn count_even_digits(n: i64) -> i64 {\n    x: i64 = n;\n    if x < 0 {\n        x = 0 - x;\n    }\n    if x == 0 {\n        return 1;\n    }\n    acc: i64 = 0;\n    while x > 0 {\n        if ((x % 10) % 2) == 0 {\n            acc = acc + 1;\n        }\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_starts_with_m(variant: usize) -> Problem {
    problem(
        "starts_with_m",
        variant,
        "strings",
        "Return 1 if s starts with the lowercase letter m, else 0.",
        "fn starts_with_m(s: string) -> i64",
        vec![
            example(vec![string("mog")], 1),
            example(vec![string("metal")], 1),
            example(vec![string("apple")], 0),
            example(vec![string("map")], 1),
        ],
        vec![
            example(vec![string("moon")], 1),
            example(vec![string("sun")], 0),
        ],
        "fn starts_with_m(s: string) -> i64 {\n    if s.starts_with(\"m\") {\n        return 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_rectangle_area(variant: usize) -> Problem {
    problem(
        "rectangle_area",
        variant,
        "structs",
        "Define struct Rectangle { width: i64, height: i64 } and return its area.",
        "fn rectangle_area(r: Rectangle) -> i64",
        vec![
            example(vec![pair(3, 4)], 12),
            example(vec![pair(1, 2)], 2),
            example(vec![pair(5, 5)], 25),
            example(vec![pair(9, 8)], 72),
        ],
        vec![
            example(vec![pair(7, 3)], 21),
            example(vec![pair(6, 6)], 36),
        ],
        "struct Rectangle {\n    width: i64,\n    height: i64,\n}\n\nfn rectangle_area(r: Rectangle) -> i64 {\n    return r.width * r.height;\n}\n",
    )
}

fn make_power(variant: usize) -> Problem {
    problem(
        "power",
        variant,
        "arithmetic",
        "Compute base raised to the power exp (non-negative).",
        "fn power(base: i64, exp: i64) -> i64",
        vec![
            example(vec![int(2), int(0)], 1),
            example(vec![int(2), int(3)], 8),
            example(vec![int(5), int(2)], 25),
            example(vec![int(3), int(4)], 81),
        ],
        vec![
            example(vec![int(4), int(3)], 64),
            example(vec![int(10), int(2)], 100),
        ],
        "fn power(base: i64, exp: i64) -> i64 {\n    if exp == 0 { return 1; }\n    result: i64 = 1;\n    i: i64 = 0;\n    while i < exp {\n        result = result * base;\n        i = i + 1;\n    }\n    return result;\n}\n",
    )
}

fn make_polynomial(variant: usize) -> Problem {
    problem(
        "polynomial",
        variant,
        "arithmetic",
        "Evaluate the polynomial 2*x*x + 3*x + 1.",
        "fn polynomial(x: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 6),
            example(vec![int(2)], 15),
            example(vec![int(5)], 66),
        ],
        vec![example(vec![int(3)], 28), example(vec![int(-1)], 0)],
        "fn polynomial(x: i64) -> i64 {\n    return 2 * x * x + 3 * x + 1;\n}\n",
    )
}

fn make_collatz_steps(variant: usize) -> Problem {
    problem(
        "collatz_steps",
        variant,
        "loops",
        "Count how many steps it takes for the Collatz sequence starting at n to reach 1.",
        "fn collatz_steps(n: i64) -> i64",
        vec![
            example(vec![int(1)], 0),
            example(vec![int(2)], 1),
            example(vec![int(3)], 7),
            example(vec![int(10)], 6),
        ],
        vec![
            example(vec![int(6)], 8),
            example(vec![int(7)], 16),
        ],
        "fn collatz_steps(n: i64) -> i64 {\n    x: i64 = n;\n    steps: i64 = 0;\n    while x > 1 {\n        if x % 2 == 0 {\n            x = x / 2;\n        } else {\n            x = 3 * x + 1;\n        }\n        steps = steps + 1;\n    }\n    return steps;\n}\n",
    )
}

fn make_min3(variant: usize) -> Problem {
    problem(
        "min3",
        variant,
        "control_flow",
        "Return the minimum of three integers.",
        "fn min3(a: i64, b: i64, c: i64) -> i64",
        vec![
            example(vec![int(2), int(3), int(4)], 2),
            example(vec![int(10), int(-4), int(8)], -4),
            example(vec![int(7), int(7), int(7)], 7),
            example(vec![int(-3), int(-2), int(-9)], -9),
        ],
        vec![
            example(vec![int(5), int(1), int(3)], 1),
            example(vec![int(0), int(-1), int(2)], -1),
        ],
        "fn min3(a: i64, b: i64, c: i64) -> i64 {\n    m: i64 = a;\n    if b < m { m = b; }\n    if c < m { m = c; }\n    return m;\n}\n",
    )
}

fn make_reverse_sum(variant: usize) -> Problem {
    problem(
        "reverse_sum",
        variant,
        "arrays",
        "Sum all elements of an array.",
        "fn reverse_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5, 1])], 6),
            example(vec![array(&[4, 4, 4])], 12),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn reverse_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

fn make_array_max_elem(variant: usize) -> Problem {
    problem(
        "array_max_elem",
        variant,
        "arrays",
        "Find the maximum element in an array.",
        "fn array_max_elem(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[5, 1])], 5),
            example(vec![array(&[4, 9])], 9),
            example(vec![array(&[2, 7, 1, 0])], 7),
        ],
        vec![
            example(vec![array(&[-3, -9, -1])], -1),
            example(vec![array(&[10, 2, 10])], 10),
        ],
        "fn array_max_elem(arr: [i64]) -> i64 {\n    best := arr[0];\n    for item in arr {\n        if item > best {\n            best = item;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_is_prime(variant: usize) -> Problem {
    problem(
        "is_prime",
        variant,
        "loops",
        "Return 1 if the number is prime, 0 otherwise.",
        "fn is_prime(n: i64) -> i64",
        vec![
            example(vec![int(2)], 1),
            example(vec![int(4)], 0),
            example(vec![int(11)], 1),
            example(vec![int(15)], 0),
        ],
        vec![
            example(vec![int(7)], 1),
            example(vec![int(9)], 0),
        ],
        "fn is_prime(n: i64) -> i64 {\n    if n < 2 { return 0; }\n    if n == 2 { return 1; }\n    if n % 2 == 0 { return 0; }\n    i: i64 = 3;\n    while i * i <= n {\n        if n % i == 0 { return 0; }\n        i = i + 2;\n    }\n    return 1;\n}\n",
    )
}

fn make_nth_triangle(variant: usize) -> Problem {
    problem(
        "nth_triangle",
        variant,
        "loops",
        "Return the nth triangular number: 1+2+...+n.",
        "fn nth_triangle(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(5)], 15),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(7)], 28),
            example(vec![int(4)], 10),
        ],
        "fn nth_triangle(n: i64) -> i64 {\n    if n <= 0 {\n        return 0;\n    }\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_fib_iter(variant: usize) -> Problem {
    problem(
        "fib_iter",
        variant,
        "loops",
        "Return the nth Fibonacci number using iterative multi-variable update.",
        "fn fib_iter(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(7)], 13),
            example(vec![int(10)], 55),
        ],
        vec![
            example(vec![int(5)], 5),
            example(vec![int(8)], 21),
        ],
        "fn fib_iter(n: i64) -> i64 {\n    if n == 0 { return 0; }\n    if n == 1 { return 1; }\n    a: i64 = 0;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }\n    return b;\n}\n",
    )
}

fn make_palindrome_check(variant: usize) -> Problem {
    problem(
        "palindrome_check",
        variant,
        "strings",
        "Return 1 if the string is a palindrome, 0 otherwise.",
        "fn palindrome_check(s: string) -> i64",
        vec![
            example(vec![string("racecar")], 1),
            example(vec![string("hello")], 0),
            example(vec![string("aba")], 1),
            example(vec![string("ab")], 0),
            example(vec![string("a")], 1),
            example(vec![string("")], 1),
        ],
        vec![
            example(vec![string("level")], 1),
            example(vec![string("world")], 0),
        ],
        "fn palindrome_check(s: string) -> i64 {\n    chars := s.split(\"\");\n    left: i64 = 0;\n    right: i64 = s.len - 1;\n    while left < right {\n        if chars[left] != chars[right] { return 0; }\n        left = left + 1;\n        right = right - 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_count_words(variant: usize) -> Problem {
    problem(
        "count_words",
        variant,
        "strings",
        "Count the number of space-separated words in a string.",
        "fn count_words(s: string) -> i64",
        vec![
            example(vec![string("hello world")], 2),
            example(vec![string("one")], 1),
            example(vec![string("a b c d")], 4),
            example(vec![string("  two words  ")], 2),
            example(vec![string("")], 0),
        ],
        vec![
            example(vec![string("foo bar baz")], 3),
            example(vec![string("  spaced  ")], 1),
        ],
        "fn count_words(s: string) -> i64 {\n    t := s.trim();\n    if t.len == 0 { return 0; }\n    parts := t.split(\" \");\n    count: i64 = 0;\n    for p in parts {\n        if p.len > 0 {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_euler_totient(variant: usize) -> Problem {
    problem(
        "euler_totient",
        variant,
        "algorithms",
        "Compute Euler's totient function phi(n).",
        "fn euler_totient(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(2)], 1),
            example(vec![int(9)], 6),
            example(vec![int(12)], 4),
        ],
        vec![
            example(vec![int(5)], 4),
            example(vec![int(10)], 4),
        ],
        "fn euler_totient(n: i64) -> i64 {\n    result: i64 = n;\n    p: i64 = 2;\n    temp: i64 = n;\n    while p * p <= temp {\n        if temp % p == 0 {\n            while temp % p == 0 {\n                temp = temp / p;\n            }\n            result = result - result / p;\n        }\n        p = p + 1;\n    }\n    if temp > 1 {\n        result = result - result / temp;\n    }\n    return result;\n}\n",
    )
}

fn make_sum_squares(variant: usize) -> Problem {
    problem(
        "sum_squares",
        variant,
        "loops",
        "Compute the sum of squares from 1 to n.",
        "fn sum_squares(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 1),
            example(vec![int(3)], 14),
            example(vec![int(5)], 55),
        ],
        vec![
            example(vec![int(4)], 30),
            example(vec![int(2)], 5),
        ],
        "fn sum_squares(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + i * i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_product_1_to_n(variant: usize) -> Problem {
    problem(
        "product_1_to_n",
        variant,
        "loops",
        "Compute the product of all integers from 1 to n.",
        "fn product_1_to_n(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(4)], 24),
            example(vec![int(6)], 720),
        ],
        vec![
            example(vec![int(3)], 6),
            example(vec![int(5)], 120),
        ],
        "fn product_1_to_n(n: i64) -> i64 {\n    if n == 0 { return 1; }\n    total: i64 = 1;\n    i: i64 = 1;\n    while i <= n {\n        total = total * i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_count_divisors(variant: usize) -> Problem {
    problem(
        "count_divisors",
        variant,
        "loops",
        "Count how many positive divisors n has.",
        "fn count_divisors(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(2)], 2),
            example(vec![int(6)], 4),
            example(vec![int(12)], 6),
        ],
        vec![
            example(vec![int(4)], 3),
            example(vec![int(9)], 3),
        ],
        "fn count_divisors(n: i64) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            count = count + 1;\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_triangular_check(variant: usize) -> Problem {
    problem(
        "triangular_check",
        variant,
        "algorithms",
        "Return 1 if n is a triangular number, 0 otherwise.",
        "fn triangular_check(n: i64) -> i64",
        vec![
            example(vec![int(0)], 1),
            example(vec![int(1)], 1),
            example(vec![int(3)], 1),
            example(vec![int(4)], 0),
        ],
        vec![
            example(vec![int(6)], 1),
            example(vec![int(5)], 0),
        ],
        "fn triangular_check(n: i64) -> i64 {\n    k: i64 = 0;\n    while k * (k + 1) / 2 <= n {\n        if k * (k + 1) / 2 == n { return 1; }\n        k = k + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_max_pair_diff(variant: usize) -> Problem {
    problem(
        "max_pair_diff",
        variant,
        "arrays",
        "Find the maximum absolute difference between consecutive elements.",
        "fn max_pair_diff(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 5, 3])], 4),
            example(vec![array(&[10, 3, 9, 2])], 7),
            example(vec![array(&[7, 7, 7])], 0),
            example(vec![array(&[2, 11, 4, 20])], 16),
        ],
        vec![
            example(vec![array(&[1, 10])], 9),
            example(vec![array(&[5, 5, 5, 5])], 0),
        ],
        "fn max_pair_diff(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        diff: i64 = arr[i] - arr[i - 1];\n        if diff < 0 { diff = 0 - diff; }\n        if diff > best { best = diff; }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_sum_negatives(variant: usize) -> Problem {
    problem(
        "sum_negatives",
        variant,
        "arrays",
        "Sum all negative numbers in the array.",
        "fn sum_negatives(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[-1, 2, -3])], -4),
            example(vec![array(&[5, 1, 4])], 0),
            example(vec![array(&[-4, -9])], -13),
            example(vec![array(&[2, -7, 1, 0])], -7),
        ],
        vec![
            example(vec![array(&[-2, -2, 2])], -4),
            example(vec![array(&[1, 2, 3])], 0),
        ],
        "fn sum_negatives(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        if item < 0 {\n            total = total + item;\n        }\n    }\n    return total;\n}\n",
    )
}

fn make_gcd_extended(variant: usize) -> Problem {
    problem(
        "gcd_extended",
        variant,
        "algorithms",
        "Compute the GCD of two non-negative integers using Euclidean algorithm with variable swap.",
        "fn gcd_extended(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(12), int(8)], 4),
            example(vec![int(35), int(14)], 7),
            example(vec![int(7), int(13)], 1),
            example(vec![int(100), int(75)], 25),
            example(vec![int(0), int(5)], 5),
            example(vec![int(6), int(0)], 6),
        ],
        vec![
            example(vec![int(18), int(12)], 6),
            example(vec![int(17), int(5)], 1),
        ],
        "fn gcd_extended(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n",
    )
}

fn make_harmonic_sum(variant: usize) -> Problem {
    problem(
        "harmonic_sum",
        variant,
        "loops",
        "Compute integer harmonic sum: sum of 1000/i for i from 1 to n.",
        "fn harmonic_sum(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1000),
            example(vec![int(2)], 1500),
            example(vec![int(5)], 2283),
            example(vec![int(10)], 2927),
        ],
        vec![
            example(vec![int(3)], 1833),
            example(vec![int(4)], 2083),
        ],
        "fn harmonic_sum(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        total = total + 1000 / i;\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_second_max(variant: usize) -> Problem {
    problem(
        "second_max",
        variant,
        "arrays",
        "Return the second largest element in the array.",
        "fn second_max(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5])], 4),
            example(vec![array(&[2, 8, 3])], 3),
            example(vec![array(&[7, 7, 2, 9])], 7),
            example(vec![array(&[1, 3])], 1),
        ],
        vec![
            example(vec![array(&[5, 10, 8])], 8),
            example(vec![array(&[4, 4, 4])], 4),
        ],
        "fn second_max(arr: [i64]) -> i64 {\n    first: i64 = arr[0];\n    second: i64 = arr[0];\n    for item in arr {\n        if item > first {\n            second = first;\n            first = item;\n        } else {\n            if item > second {\n                second = item;\n            }\n        }\n    }\n    return second;\n}\n",
    )
}

fn make_array_range(variant: usize) -> Problem {
    problem(
        "array_range",
        variant,
        "arrays",
        "Return max - min of the array.",
        "fn array_range(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 5])], 4),
            example(vec![array(&[7, 2, 9, 4])], 7),
            example(vec![array(&[5, 5])], 0),
            example(vec![array(&[1, 8, 3, 2])], 7),
        ],
        vec![
            example(vec![array(&[0, 10, 5])], 10),
            example(vec![array(&[-3, 3])], 6),
        ],
        "fn array_range(arr: [i64]) -> i64 {\n    lo: i64 = arr[0];\n    hi: i64 = arr[0];\n    for item in arr {\n        if item < lo {\n            lo = item;\n        }\n        if item > hi {\n            hi = item;\n        }\n    }\n    return hi - lo;\n}\n",
    )
}

fn make_sum_of_divisors(variant: usize) -> Problem {
    problem(
        "sum_of_divisors",
        variant,
        "loops",
        "Sum all positive divisors of n including 1 and n.",
        "fn sum_of_divisors(n: i64) -> i64",
        vec![
            example(vec![int(6)], 12),
            example(vec![int(12)], 28),
            example(vec![int(7)], 8),
            example(vec![int(1)], 1),
        ],
        vec![
            example(vec![int(4)], 7),
            example(vec![int(9)], 13),
        ],
        "fn sum_of_divisors(n: i64) -> i64 {\n    total: i64 = 0;\n    i: i64 = 1;\n    while i <= n {\n        if n % i == 0 {\n            total = total + i;\n        }\n        i = i + 1;\n    }\n    return total;\n}\n",
    )
}

fn make_sum_odd_digits(variant: usize) -> Problem {
    problem(
        "sum_odd_digits",
        variant,
        "loops",
        "Sum the digits of n that are odd (1,3,5,7,9).",
        "fn sum_odd_digits(n: i64) -> i64",
        vec![
            example(vec![int(135)], 9),
            example(vec![int(248)], 0),
            example(vec![int(19)], 10),
            example(vec![int(0)], 0),
        ],
        vec![
            example(vec![int(357)], 15),
            example(vec![int(246)], 0),
        ],
        "fn sum_odd_digits(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if (d % 2) == 1 {\n            acc = acc + d;\n        }\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_zeros(variant: usize) -> Problem {
    problem(
        "count_zeros",
        variant,
        "arrays",
        "Count the number of zeros in the array.",
        "fn count_zeros(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 0, 2, 0])], 2),
            example(vec![array(&[0])], 1),
            example(vec![array(&[3, 5, 1])], 0),
            example(vec![array(&[0, 0, 0, 1])], 3),
        ],
        vec![
            example(vec![array(&[0, 1, 0])], 2),
            example(vec![array(&[1, 2, 3])], 0),
        ],
        "fn count_zeros(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    for item in arr {\n        if item == 0 {\n            count = count + 1;\n        }\n    }\n    return count;\n}\n",
    )
}

fn make_max_consecutive_sum(variant: usize) -> Problem {
    problem(
        "max_consecutive_sum",
        variant,
        "arrays",
        "Return the maximum sum of any non-empty contiguous subarray (Kadane's algorithm).",
        "fn max_consecutive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], 3),
            example(vec![array(&[3, -1, 2])], 4),
            example(vec![array(&[-1, -2, -3])], -1),
            example(vec![array(&[2, 3, -1, 4])], 8),
        ],
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[-5, 4, -1, 2])], 5),
        ],
        "fn max_consecutive_sum(arr: [i64]) -> i64 {\n    current: i64 = 0;\n    best: i64 = arr[0];\n    for item in arr {\n        if current > 0 {\n            current = current + item;\n        } else {\n            current = item;\n        }\n        if current > best {\n            best = current;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_min_consecutive_sum(variant: usize) -> Problem {
    problem(
        "min_consecutive_sum",
        variant,
        "arrays",
        "Return the minimum sum of any non-empty contiguous subarray (anti-Kadane's algorithm).",
        "fn min_consecutive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], -2),
            example(vec![array(&[3, -1, -2, 5])], -3),
            example(vec![array(&[1, 2, 3, 4])], 1),
            example(vec![array(&[-2, -3, 1, -4])], -8),
        ],
        vec![
            example(vec![array(&[-1, -2, -3])], -6),
            example(vec![array(&[5, -1, 3])], -1),
        ],
        "fn min_consecutive_sum(arr: [i64]) -> i64 {\n    current: i64 = 0;\n    best: i64 = arr[0];\n    for item in arr {\n        if current < 0 {\n            current = current + item;\n        } else {\n            current = item;\n        }\n        if current < best {\n            best = current;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_interactive_sum(variant: usize) -> Problem {
    problem(
        "interactive_sum",
        variant,
        "arrays",
        "Return the sum of all integers in an array.",
        "fn interactive_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 6),
            example(vec![array(&[5, 1])], 6),
            example(vec![array(&[4, 4, 4])], 12),
            example(vec![array(&[2, 7, 1, 0])], 10),
        ],
        vec![
            example(vec![array(&[10, -5, 2])], 7),
            example(vec![array(&[1, 2, 3, 4])], 10),
        ],
        "fn interactive_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    for item in arr {\n        total = total + item;\n    }\n    return total;\n}\n",
    )
}

// ── String-output programs (first-class via the widened Value output) ────────

fn make_reverse_str(variant: usize) -> Problem {
    problem(
        "reverse_str",
        variant,
        "strings",
        "Reverse the input string.",
        "fn reverse_str(s: string) -> string",
        vec![
            example_str(vec![string("abc")], "cba"),
            example_str(vec![string("hello")], "olleh"),
            example_str(vec![string("x")], "x"),
            example_str(vec![string("ab")], "ba"),
        ],
        vec![
            example_str(vec![string("world")], "dlrow"),
            example_str(vec![string("rust")], "tsur"),
        ],
        "fn reverse_str(s: string) -> string {\n    return s.reverse();\n}\n",
    )
}

fn make_uppercase(variant: usize) -> Problem {
    problem(
        "uppercase",
        variant,
        "strings",
        "Return the string in uppercase.",
        "fn uppercase(s: string) -> string",
        vec![
            example_str(vec![string("abc")], "ABC"),
            example_str(vec![string("Hello")], "HELLO"),
            example_str(vec![string("mix")], "MIX"),
        ],
        vec![example_str(vec![string("done")], "DONE")],
        "fn uppercase(s: string) -> string {\n    return s.upper();\n}\n",
    )
}

fn make_capitalize(variant: usize) -> Problem {
    problem(
        "capitalize",
        variant,
        "strings",
        "Uppercase the first character, leave the rest unchanged.",
        "fn capitalize(s: string) -> string",
        vec![
            example_str(vec![string("cat")], "Cat"),
            example_str(vec![string("dog")], "Dog"),
            example_str(vec![string("bird")], "Bird"),
            example_str(vec![string("fish")], "Fish"),
        ],
        vec![
            example_str(vec![string("fox")], "Fox"),
            example_str(vec![string("hen")], "Hen"),
        ],
        "fn capitalize(s: string) -> string {\n    return s.slice(0, 1).upper() + s.slice(1, s.len);\n}\n",
    )
}

fn make_drop_last(variant: usize) -> Problem {
    problem(
        "drop_last",
        variant,
        "strings",
        "Drop the last character of the string.",
        "fn drop_last(s: string) -> string",
        vec![
            example_str(vec![string("cats")], "cat"),
            example_str(vec![string("dogs")], "dog"),
            example_str(vec![string("birds")], "bird"),
        ],
        vec![example_str(vec![string("foxes")], "foxe")],
        "fn drop_last(s: string) -> string {\n    return s.slice(0, s.len - 1);\n}\n",
    )
}

fn make_full_name(variant: usize) -> Problem {
    problem(
        "full_name",
        variant,
        "strings",
        "Join first and last name with a space.",
        "fn full_name(a: string, b: string) -> string",
        vec![
            example_str(vec![string("john"), string("smith")], "john smith"),
            example_str(vec![string("jane"), string("doe")], "jane doe"),
            example_str(vec![string("amy"), string("lee")], "amy lee"),
        ],
        vec![example_str(vec![string("max"), string("ray")], "max ray")],
        "fn full_name(a: string, b: string) -> string {\n    return (a + \" \") + b;\n}\n",
    )
}

fn make_last_first(variant: usize) -> Problem {
    problem(
        "last_first",
        variant,
        "strings",
        "Render as \"last, first\".",
        "fn last_first(a: string, b: string) -> string",
        vec![
            example_str(vec![string("john"), string("smith")], "smith, john"),
            example_str(vec![string("jane"), string("doe")], "doe, jane"),
            example_str(vec![string("amy"), string("lee")], "lee, amy"),
        ],
        vec![example_str(vec![string("max"), string("ray")], "ray, max")],
        "fn last_first(a: string, b: string) -> string {\n    return (b + \", \") + a;\n}\n",
    )
}

pub const STRING_FACTORIES: &[fn(usize) -> Problem; 6] = &[
    make_reverse_str,
    make_uppercase,
    make_capitalize,
    make_drop_last,
    make_full_name,
    make_last_first,
];

pub const FACTORIES: &[fn(usize) -> Problem; 140] = &[
    make_add_two,
    make_abs_diff,
    make_max2,
    make_clamp,
    make_sign,
    make_sum_to_n,
    make_gcd,
    make_lcm,
    make_array_sum,
    make_array_max,
    make_count_occurrences,
    make_trimmed_len,
    make_vowel_count,
    make_contains_cat,
    make_point_sum,
    make_safe_div_or_neg1,
    make_positive_or_default,
    make_factorial,
    make_fibonacci,
    make_closure_map_sum,
    make_count_positive,
    make_is_even,
    make_digit_sum,
    make_reverse_digits,
    make_digit_count,
    make_count_even_digits,
    make_starts_with_m,
    make_rectangle_area,
    make_power,
    make_polynomial,
    make_collatz_steps,
    make_min3,
    make_reverse_sum,
    make_array_max_elem,
    make_is_prime,
    make_nth_triangle,
    make_fib_iter,
    make_palindrome_check,
    make_count_words,
    make_euler_totient,
    make_sum_squares,
    make_product_1_to_n,
    make_count_divisors,
    make_triangular_check,
    make_max_pair_diff,
    make_sum_negatives,
    make_gcd_extended,
    make_harmonic_sum,
    make_interactive_sum,
    make_second_max,
    make_array_range,
    make_sum_of_divisors,
    make_sum_odd_digits,
    make_count_zeros,
    make_max_consecutive_sum,
    make_min_consecutive_sum,
    make_kth_smallest,
    make_max_stock_profit,
    make_is_sorted,
    make_longest_increasing_run,
    make_digital_root,
    make_two_sum_exists,
    make_count_distinct,
    make_binary_search,
    make_strictly_increasing,
    make_has_strictly_increasing_run,
    make_first_index_of,
    make_last_index_of,
    make_is_anagram,
    make_longest_run,
    make_intersects,
    make_longest_plateau,
    make_prefix_max_sum,
    make_cube,
    make_square_plus_n,
    make_bilinear3,
    make_scaled_sum,
    make_product_offset,
    make_arr_sum_squares,
    make_min_element,
    make_sum_absolute,
    make_count_evens,
    make_sum_positives,
    make_sum_at_even_indices,
    make_kth_from_end,
    make_max_abs,
    make_digit_product,
    make_max_digit,
    make_min_positive,
    make_lucas_number,
    make_celsius_to_fahrenheit,
    make_is_perfect_square,
    make_next_power_of_2,
    make_count_peaks,
    make_alternating_sum,
    make_count_greater_than,
    make_dot_product,
    make_leading_digit,
    make_popcount,
    make_prefix_sum_k,
    make_is_palindrome_arr,
    make_sum_odd_indexed,
    // Tier-2: game-adjacent problems (April 2026)
    make_score_tracker,
    make_game_tick,
    make_tensor_3d_per_frame,
    make_ema_state,
    make_memory_cell,
    make_vending_change,
    make_combat_resolve,
    make_traffic_light_phase,
    make_run_length_decode_sum,
    make_stateful_reducer,
    make_count_adjacent_diff,
    make_priority_pop,
    make_turn_order_rotate,
    make_grid_bounds_check,
    make_simulate_gravity,
    // Stage 2 starter: event-modulated stateful benchmarks (June 2026).
    // These exercise the `search_stateful_reducer_event` teacher
    // for the (state, event, arr) -> state signature — per-tick
    // game-loop / physics patterns where the event scalar gates or
    // modulates the array contribution to the state. See
    // `docs/stateful_synthesis_status.md` Stage 2 section.
    make_physics_step_1d,
    make_brake_accumulator,
    make_boost_modulated,
    make_turn_counter_gated,
    make_damage_with_event,
    make_delta_accumulator,
    make_signed_count_delta,
    make_cross_range_state,
    make_boost_positive,
    make_running_max,
    make_running_min,
    make_flip_on_positive,
    make_increment_on_positive,
    make_reset_on_negative,
    make_loss_accumulator,
    make_inventory_total,
    // Stage 4: time-lane stateful benchmarks (June 2026)
    make_aging_state,
    make_time_decay,
    make_rate_accumulator,
    make_first_rate,
    make_count_rate,
    make_max_rate,
    make_tick_every_2,
    // Stage 2: tensor benchmarks (June 2026) — COMMENTED OUT pending tensor codegen impl
    // make_matrix_diagonal_sum,
    // make_dot_product_4d,
    // make_matrix_multiply_2x2,
    // make_broadcast_scale_sum,
    // make_outer_product_norm_sq,
    // make_convolution_1d_sum,
    // Stage 3: struct-of-state benchmarks (June 2026) — COMMENTED OUT pending struct-of-state impl
    // make_dual_tally,
    // make_rate_limiter,
    // make_running_correlation,
    // make_mutual_info_tracker,
    // make_dual_threshold_classifier,
    // make_paired_extrema,
];

fn make_kth_smallest(variant: usize) -> Problem {
    problem(
        "kth_smallest",
        variant,
        "sorting",
        "Return the kth smallest element of the array (1-indexed).",
        "fn kth_smallest(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5]), int(2)], 1),
            example(vec![array(&[7, 2, 9, 4]), int(3)], 7),
            example(vec![array(&[5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(4)], 4),
        ],
        vec![
            example(vec![array(&[10, 3, 7, 1]), int(2)], 3),
            example(vec![array(&[-2, 0, 4]), int(1)], -2),
        ],
        "fn kth_smallest(arr: [i64], k: i64) -> i64 {\n    arr.sort();\n    return arr[k - 1];\n}\n",
    )
}

fn make_max_stock_profit(variant: usize) -> Problem {
    problem(
        "max_stock_profit",
        variant,
        "arrays",
        "Given daily prices, return the maximum profit from buying once then selling later. If no profit possible, return 0.",
        "fn max_stock_profit(prices: [i64]) -> i64",
        vec![
            example(vec![array(&[7, 1, 5, 3, 6, 4])], 5),
            example(vec![array(&[7, 6, 4, 3, 1])], 0),
            example(vec![array(&[1, 2])], 1),
            example(vec![array(&[2, 4, 1, 7])], 6),
        ],
        vec![
            example(vec![array(&[3, 3, 3])], 0),
            example(vec![array(&[1, 5, 2, 8])], 7),
        ],
        "fn max_stock_profit(prices: [i64]) -> i64 {\n    min_price: i64 = prices[0];\n    best: i64 = 0;\n    for p in prices {\n        if p < min_price { min_price = p; }\n        profit: i64 = p - min_price;\n        if profit > best { best = profit; }\n    }\n    return best;\n}\n",
    )
}

fn make_is_sorted(variant: usize) -> Problem {
    problem(
        "is_sorted",
        variant,
        "arrays",
        "Return 1 if the array is sorted in non-decreasing order, 0 otherwise.",
        "fn is_sorted(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 1),
            example(vec![array(&[3, 2, 1])], 0),
            example(vec![array(&[1, 1, 2])], 1),
            example(vec![array(&[5])], 1),
        ],
        vec![
            example(vec![array(&[1, 3, 2])], 0),
            example(vec![array(&[-2, 0, 4])], 1),
        ],
        "fn is_sorted(arr: [i64]) -> i64 {\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] < arr[i - 1] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n",
    )
}

fn example_bool(inputs: Vec<Value>, expected: bool) -> Example {
    Example {
        inputs,
        expected: Value::Bool(expected),
    }
}

/// Same task as `is_sorted` but the predicate is carried in Mog's
/// native `bool` lane (`-> i64` return + `Value::Bool` expected). The
/// new `output_matches` bridge makes this interchangeable with the
/// 0/1 int lane; the bool shape exercises it.
fn make_is_sorted_bool(variant: usize) -> Problem {
    problem(
        "is_sorted_bool",
        variant,
        "arrays",
        "Return true if the array is sorted in non-decreasing order.",
        "fn is_sorted_bool(arr: [i64]) -> i64",
        vec![
            example_bool(vec![array(&[1, 2, 3])], true),
            example_bool(vec![array(&[3, 2, 1])], false),
            example_bool(vec![array(&[1, 1, 2])], true),
            example_bool(vec![array(&[5])], true),
        ],
        vec![
            example_bool(vec![array(&[1, 3, 2])], false),
            example_bool(vec![array(&[-2, 0, 4])], true),
        ],
        "fn is_sorted_bool(arr: [i64]) -> i64 {\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] < arr[i - 1] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_longest_increasing_run(variant: usize) -> Problem {
    problem(
        "longest_increasing_run",
        variant,
        "arrays",
        "Return the length of the longest strictly increasing consecutive run.",
        "fn longest_increasing_run(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2, 4, 5])], 3),
            example(vec![array(&[5, 4, 3, 2, 1])], 1),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[1, 2, 1, 2, 3])], 3),
        ],
        vec![
            example(vec![array(&[3, 1, 2])], 2),
            example(vec![array(&[1, 1, 1])], 1),
        ],
        "fn longest_increasing_run(arr: [i64]) -> i64 {\n    best: i64 = 1;\n    cur: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] > arr[i - 1] {\n            cur = cur + 1;\n            if cur > best { best = cur; }\n        } else {\n            cur = 1;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_digital_root(variant: usize) -> Problem {
    problem(
        "digital_root",
        variant,
        "loops",
        "Repeatedly sum the digits of n until a single digit remains.",
        "fn digital_root(n: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(9)], 9),
            example(vec![int(493)], 7),
            example(vec![int(942)], 6),
        ],
        vec![
            example(vec![int(18)], 9),
            example(vec![int(11)], 2),
        ],
        "fn digital_root(n: i64) -> i64 {\n    x: i64 = n;\n    while x >= 10 {\n        s: i64 = 0;\n        while x > 0 {\n            s = s + x % 10;\n            x = x / 10;\n        }\n        x = s;\n    }\n    return x;\n}\n",
    )
}

fn make_two_sum_exists(variant: usize) -> Problem {
    problem(
        "two_sum_exists",
        variant,
        "arrays",
        "Return 1 if any two distinct elements sum to target, 0 otherwise.",
        "fn two_sum_exists(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[2, 7, 11, 15]), int(9)], 1),
            example(vec![array(&[3, 5, 8]), int(4)], 0),
            example(vec![array(&[1, 2, 3]), int(5)], 1),
            example(vec![array(&[1, 2, 3]), int(7)], 0),
        ],
        vec![
            example(vec![array(&[4, 4]), int(8)], 1),
            example(vec![array(&[1, 5, 3]), int(6)], 1),
        ],
        "fn two_sum_exists(arr: [i64], target: i64) -> i64 {\n    i: i64 = 0;\n    while i < arr.len {\n        j: i64 = i + 1;\n        while j < arr.len {\n            if arr[i] + arr[j] == target { return 1; }\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_count_distinct(variant: usize) -> Problem {
    problem(
        "count_distinct",
        variant,
        "sorting",
        "Count the number of distinct values in an array.",
        "fn count_distinct(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 2, 3, 3, 3])], 3),
            example(vec![array(&[5, 5, 5])], 1),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[7])], 1),
        ],
        vec![
            example(vec![array(&[1, 1, 2, 3])], 3),
            example(vec![array(&[0, 0, 0, 1, 1])], 2),
        ],
        "fn count_distinct(arr: [i64]) -> i64 {\n    arr.sort();\n    count: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] != arr[i - 1] {\n            count = count + 1;\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_strictly_increasing(variant: usize) -> Problem {
    // variant picks the strict-inequality vs ≤ test's bias: variant 0
    // uses the "negatives include equal neighbours" path that
    // search_is_sorted can't solve; variant 1 includes a descent in
    // the negatives (also unmatched by is_sorted).
    let _ = variant;
    problem(
        "strictly_increasing",
        variant,
        "arrays",
        "Return 1 iff the array is strictly increasing (no equal neighbours).",
        "fn strictly_increasing(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], 1),
            example(vec![array(&[0, 5])], 1),
            example(vec![array(&[-3, -1, 0, 7, 100])], 1),
            example(vec![array(&[10, 20, 30, 40, 50])], 1),
            // equal neighbours
            example(vec![array(&[1, 1, 2])], 0),
            example(vec![array(&[2, 2])], 0),
            example(vec![array(&[5, 5, 5, 6])], 0),
            // descent
            example(vec![array(&[3, 2, 1])], 0),
            example(vec![array(&[10, 0])], 0),
            example(vec![array(&[1, 5, 4, 9])], 0),
        ],
        vec![
            example(vec![array(&[100, 200])], 1),
            example(vec![array(&[1, 1])], 0),
            example(vec![array(&[1, 2, 1])], 0),
            example(vec![array(&[0, 0, 1])], 0),
        ],
        "fn strictly_increasing(arr: [i64]) -> i64 {\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] <= arr[i - 1] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_has_strictly_increasing_run(variant: usize) -> Problem {
    // Each variant picks a different run length (2..=5). The teacher
    // tries k in {2,3,4,5} and emits the first that verifies.
    let run_length = match variant % 4 {
        0 => 2,
        1 => 3,
        2 => 4,
        _ => 5,
    };
    let name = format!("has_strictly_increasing_run_{run_length}");
    let reference: &'static str = Box::leak(
        format!(
            "fn {name}(arr: [i64]) -> i64 {{\n    run: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {{\n        if arr[i] > arr[i - 1] {{\n            run = run + 1;\n            if run >= {run_length} {{ return 1; }}\n        }} else {{\n            run = 1;\n        }}\n        i = i + 1;\n    }}\n    return 0;\n}}\n"
        )
        .into_boxed_str(),
    );
    let signature: &'static str =
        Box::leak(format!("fn {name}(arr: [i64]) -> i64").into_boxed_str());
    let description: &'static str = Box::leak(
        format!("Return 1 iff arr contains a strictly increasing run of length >= {run_length}.")
            .into_boxed_str(),
    );
    problem(
        &name,
        variant,
        "arrays",
        description,
        signature,
        vec![
            example(vec![array(&[1, 2, 3])], 1),
            example(vec![array(&[0, 1, 5, 6, 7])], 1),
            example(vec![array(&[10, 20, 30])], 1),
            example(vec![array(&[5, 4, 3, 7, 8, 9])], 1),
            example(vec![array(&[1, 2])], 0),
            example(vec![array(&[1, 5, 3])], 0),
            example(vec![array(&[5, 4, 3, 2, 1])], 0),
        ],
        vec![
            example(vec![array(&[3, 3, 4])], 0),
            example(vec![array(&[-1, 0, 1, 2, 0, 1])], 1),
        ],
        reference,
    )
}

fn make_first_index_of(variant: usize) -> Problem {
    // Each variant uses a different target so the teacher actually has
    // to mine the candidate set. Targets cycle through small constants.
    let target: i64 = match variant % 6 {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 5,
        4 => 7,
        _ => -1,
    };
    let name = format!("first_index_of_{target}");
    let reference: &'static str = Box::leak(
        format!(
            "fn {name}(arr: [i64]) -> i64 {{\n    i: i64 = 0;\n    while i < arr.len {{\n        if arr[i] == {target} {{ return i; }}\n        i = i + 1;\n    }}\n    return 0 - 1;\n}}\n"
        )
        .into_boxed_str(),
    );
    let signature: &'static str =
        Box::leak(format!("fn {name}(arr: [i64]) -> i64").into_boxed_str());
    let description: &'static str = Box::leak(
        format!("Return the first index where arr[i] == {target}, or -1.").into_boxed_str(),
    );
    problem(
        &name,
        variant,
        "arrays",
        description,
        signature,
        vec![
            example(vec![array(&[1, 2, 3, 4, 5])], 4),
            example(vec![array(&[5, 5, 5])], 0),
            example(vec![array(&[0, 0, 0, 5])], 3),
            example(vec![array(&[10, 20, 30])], -1),
            example(vec![array(&[5])], 0),
        ],
        vec![
            example(vec![array(&[1, 2, 3, 4, 6, 7, 5, 8, 9, 5])], 6),
            example(vec![array(&[1, 2, 3, 4])], -1),
        ],
        reference,
    )
}

fn make_last_index_of(variant: usize) -> Problem {
    // Mirror of first_index_of but with a target that appears multiple
    // times so last != first in at least one example.
    let target: i64 = match variant % 4 {
        0 => 5,
        1 => 0,
        2 => 7,
        _ => -2,
    };
    let name = format!("last_index_of_{target}");
    let reference: &'static str = Box::leak(
        format!(
            "fn {name}(arr: [i64]) -> i64 {{\n    i: i64 = arr.len - 1;\n    while i >= 0 {{\n        if arr[i] == {target} {{ return i; }}\n        i = i - 1;\n    }}\n    return 0 - 1;\n}}\n"
        )
        .into_boxed_str(),
    );
    let signature: &'static str =
        Box::leak(format!("fn {name}(arr: [i64]) -> i64").into_boxed_str());
    let description: &'static str = Box::leak(
        format!("Return the last index where arr[i] == {target}, or -1.").into_boxed_str(),
    );
    problem(
        &name,
        variant,
        "arrays",
        description,
        signature,
        vec![
            // multiple targets so last != first
            example(vec![array(&[1, 5, 2, 5, 3])], 3),
            example(vec![array(&[5, 5])], 1),
            example(vec![array(&[5])], 0),
            example(vec![array(&[1, 2, 3])], -1),
            example(vec![array(&[5, 4, 3, 2, 1])], 0),
        ],
        vec![
            example(vec![array(&[1, 5, 1, 5, 1, 5])], 5),
            example(vec![array(&[1, 2, 3, 4])], -1),
        ],
        reference,
    )
}

fn make_binary_search(variant: usize) -> Problem {
    problem(
        "binary_search",
        variant,
        "sorting",
        "Return the index of target in a sorted array, or -1 if not found.",
        "fn binary_search(arr: [i64], target: i64) -> i64",
        vec![
            example(vec![array(&[1, 3, 5, 7, 9]), int(5)], 2),
            example(vec![array(&[1, 3, 5, 7, 9]), int(6)], -1),
            example(vec![array(&[2, 4, 6, 8]), int(2)], 0),
            example(vec![array(&[2, 4, 6, 8]), int(8)], 3),
        ],
        vec![
            example(vec![array(&[10, 20, 30]), int(20)], 1),
            example(vec![array(&[1, 2, 3, 4, 5]), int(0)], -1),
        ],
        "fn binary_search(arr: [i64], target: i64) -> i64 {\n    lo: i64 = 0;\n    hi: i64 = arr.len - 1;\n    while lo <= hi {\n        mid: i64 = (lo + hi) / 2;\n        if arr[mid] == target { return mid; }\n        if arr[mid] < target { lo = mid + 1; }\n        if arr[mid] > target { hi = mid - 1; }\n    }\n    return -1;\n}\n",
    )
}

fn make_is_anagram(variant: usize) -> Problem {
    // Each variant pairs a different word pair so the teacher
    // exercises its sort + elementwise-compare codegen across
    // multiple shapes.
    let (a, b) = match variant % 4 {
        0 => (&[1, 2, 3][..], &[3, 1, 2][..]),
        1 => (&[1, 1, 2, 2][..], &[2, 1, 2, 1][..]),
        2 => (&[5, 4, 3, 2, 1][..], &[1, 2, 3, 4, 5][..]),
        _ => (&[0, 0, 1][..], &[0, 1, 0][..]),
    };
    let problem = problem(
        "is_anagram",
        variant,
        "arrays",
        "Return 1 iff the two arrays are permutations of each other.",
        "fn is_anagram(a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![array(a), array(b)], 1),
            example(vec![array(&[1, 2, 3]), array(&[1, 2, 4])], 0),
            example(vec![array(&[1, 2, 3]), array(&[3, 2, 1, 0])], 0),
            example(vec![array(&[]), array(&[])], 1),
        ],
        vec![example(vec![array(&[10, 20, 30]), array(&[30, 10, 20])], 1)],
        "fn is_anagram(a: [i64], b: [i64]) -> i64 {\n    if a.len != b.len { return 0; }\n    sa: [i64] = a;\n    sb: [i64] = b;\n    sa.sort();\n    sb.sort();\n    i: i64 = 0;\n    while i < a.len {\n        if sa[i] != sb[i] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n",
    );
    problem
}

fn make_longest_run(variant: usize) -> Problem {
    // Cycle target values: 0, 1, 5, 7.
    let target: i64 = match variant % 4 {
        0 => 0,
        1 => 1,
        2 => 5,
        _ => 7,
    };
    let name = format!("longest_run_{target}");
    let reference: &'static str = Box::leak(
        format!(
            "fn {name}(arr: [i64]) -> i64 {{\n    best: i64 = 0;\n    cur: i64 = 0;\n    for v in arr {{\n        if v == {target} {{\n            cur = cur + 1;\n            if cur > best {{ best = cur; }}\n        }} else {{\n            cur = 0;\n        }}\n    }}\n    return best;\n}}\n"
        )
        .into_boxed_str(),
    );
    let signature: &'static str =
        Box::leak(format!("fn {name}(arr: [i64]) -> i64").into_boxed_str());
    let description: &'static str = Box::leak(
        format!("Return the length of the longest contiguous run of {target} in arr.")
            .into_boxed_str(),
    );
    problem(
        &name,
        variant,
        "arrays",
        description,
        signature,
        vec![
            example(vec![array(&[5, 5, 5])], 3),
            example(vec![array(&[1, 2, 5, 5, 5, 6, 7])], 3),
            example(vec![array(&[5])], 1),
            example(vec![array(&[1, 2, 3, 4])], 0),
        ],
        vec![example(vec![array(&[5, 6, 5, 6, 5, 5])], 2)],
        reference,
    )
}

fn make_intersects(variant: usize) -> Problem {
    // Each variant uses a different pair of arrays.
    let (a, b) = match variant % 4 {
        0 => (&[1, 2, 3][..], &[4, 5, 3][..]),
        1 => (&[1, 2, 3][..], &[4, 5, 6][..]),
        2 => (&[7, 8, 9][..], &[10, 11, 7][..]),
        _ => (&[1, 1, 2][..], &[2, 3, 4][..]),
    };
    let problem = problem(
        "intersects",
        variant,
        "arrays",
        "Return 1 iff the two arrays share at least one element.",
        "fn intersects(a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![array(a), array(b)], 1),
            example(vec![array(&[1, 2, 3]), array(&[4, 5, 6])], 0),
            example(vec![array(&[]), array(&[1, 2, 3])], 0),
            example(vec![array(&[1, 2, 3]), array(&[3, 2, 1])], 1),
        ],
        vec![example(vec![array(&[10, 20, 30]), array(&[40, 30, 50])], 1)],
        "fn intersects(a: [i64], b: [i64]) -> i64 {\n    for x in a {\n        for y in b {\n            if x == y { return 1; }\n        }\n    }\n    return 0;\n}\n",
    );
    problem
}

fn make_longest_plateau(variant: usize) -> Problem {
    problem(
        "longest_plateau",
        variant,
        "arrays",
        "Return the length of the longest run of equal consecutive elements.",
        "fn longest_plateau(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 1, 2, 2, 2, 1])], 3),
            example(vec![array(&[5, 5, 5, 5])], 4),
            example(vec![array(&[1, 2, 3])], 1),
            example(vec![array(&[3, 3, 1, 1, 1, 2])], 3),
        ],
        vec![
            example(vec![array(&[7, 7, 3, 3])], 2),
            example(vec![array(&[1])], 1),
        ],
        "fn longest_plateau(arr: [i64]) -> i64 {\n    best: i64 = 1;\n    cur: i64 = 1;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] == arr[i - 1] {\n            cur = cur + 1;\n            if cur > best { best = cur; }\n        } else {\n            cur = 1;\n        }\n        i = i + 1;\n    }\n    return best;\n}\n",
    )
}

fn make_prefix_max_sum(variant: usize) -> Problem {
    problem(
        "prefix_max_sum",
        variant,
        "arrays",
        "Sum the running maximum: for each position i, add the max of arr[0..=i].",
        "fn prefix_max_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2, 5])], 12),
            example(vec![array(&[5, 4, 3])], 15),
            example(vec![array(&[1, 1, 1])], 3),
            example(vec![array(&[2, 5, 3, 8])], 20),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 2])], 14),
            example(vec![array(&[7])], 7),
        ],
        "fn prefix_max_sum(arr: [i64]) -> i64 {\n    running_max: i64 = arr[0];\n    total: i64 = 0;\n    for x in arr {\n        if x > running_max { running_max = x; }\n        total = total + running_max;\n    }\n    return total;\n}\n",
    )
}

fn make_cube(variant: usize) -> Problem {
    problem(
        "cube",
        variant,
        "arithmetic",
        "Return x cubed (x * x * x).",
        "fn cube(x: i64) -> i64",
        vec![
            example(vec![int(2)], 8),
            example(vec![int(3)], 27),
            example(vec![int(4)], 64),
            example(vec![int(1)], 1),
        ],
        vec![example(vec![int(0)], 0), example(vec![int(-2)], -8)],
        "fn cube(x: i64) -> i64 {\n    return x * x * x;\n}\n",
    )
}

fn make_square_plus_n(variant: usize) -> Problem {
    problem(
        "square_plus_n",
        variant,
        "arithmetic",
        "Return n*n + n (equivalently n*(n+1)).",
        "fn square_plus_n(n: i64) -> i64",
        vec![
            example(vec![int(1)], 2),
            example(vec![int(2)], 6),
            example(vec![int(3)], 12),
            example(vec![int(4)], 20),
        ],
        vec![example(vec![int(0)], 0), example(vec![int(5)], 30)],
        "fn square_plus_n(n: i64) -> i64 {\n    return (n * n) + n;\n}\n",
    )
}

fn make_bilinear3(variant: usize) -> Problem {
    problem(
        "bilinear3",
        variant,
        "arithmetic",
        "Return a*b + c.",
        "fn bilinear3(a: i64, b: i64, c: i64) -> i64",
        vec![
            example(vec![int(2), int(3), int(1)], 7),
            example(vec![int(3), int(2), int(5)], 11),
            example(vec![int(4), int(1), int(0)], 4),
            example(vec![int(1), int(4), int(2)], 6),
        ],
        vec![
            example(vec![int(5), int(2), int(3)], 13),
            example(vec![int(2), int(5), int(1)], 11),
        ],
        "fn bilinear3(a: i64, b: i64, c: i64) -> i64 {\n    return a * b + c;\n}\n",
    )
}

fn make_scaled_sum(variant: usize) -> Problem {
    problem(
        "scaled_sum",
        variant,
        "arithmetic",
        "Return 2*a + b.",
        "fn scaled_sum(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(1)], 7),
            example(vec![int(2), int(4)], 8),
            example(vec![int(5), int(0)], 10),
            example(vec![int(1), int(3)], 5),
        ],
        vec![
            example(vec![int(4), int(3)], 11),
            example(vec![int(0), int(6)], 6),
        ],
        "fn scaled_sum(a: i64, b: i64) -> i64 {\n    return 2 * a + b;\n}\n",
    )
}

fn make_product_offset(variant: usize) -> Problem {
    problem(
        "product_offset",
        variant,
        "arithmetic",
        "Return a*b - a (equivalently a*(b-1)).",
        "fn product_offset(a: i64, b: i64) -> i64",
        vec![
            example(vec![int(3), int(4)], 9),
            example(vec![int(2), int(5)], 8),
            example(vec![int(5), int(2)], 5),
            example(vec![int(4), int(3)], 8),
        ],
        vec![
            example(vec![int(1), int(7)], 6),
            example(vec![int(6), int(0)], -6),
        ],
        "fn product_offset(a: i64, b: i64) -> i64 {\n    return a * b - a;\n}\n",
    )
}

fn make_arr_sum_squares(variant: usize) -> Problem {
    problem(
        "arr_sum_squares",
        variant,
        "arrays",
        "Return the sum of squares of all elements.",
        "fn arr_sum_squares(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 14),
            example(vec![array(&[4])], 16),
            example(vec![array(&[0, 5])], 25),
            example(vec![array(&[-3, 4])], 25),
        ],
        vec![
            example(vec![array(&[2, 2, 2])], 12),
            example(vec![array(&[1, 1, 1, 1])], 4),
        ],
        "fn arr_sum_squares(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        acc = acc + x * x;\n    }\n    return acc;\n}\n",
    )
}

fn make_min_element(variant: usize) -> Problem {
    problem(
        "min_element",
        variant,
        "arrays",
        "Return the minimum element of the array.",
        "fn min_element(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 2])], 1),
            example(vec![array(&[5, 9, 2, 6])], 2),
            example(vec![array(&[7])], 7),
            example(vec![array(&[-3, 0, 2])], -3),
        ],
        vec![
            example(vec![array(&[4, 4, 4])], 4),
            example(vec![array(&[10, -5, 3])], -5),
        ],
        "fn min_element(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    for x in arr {\n        if x < best {\n            best = x;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_sum_absolute(variant: usize) -> Problem {
    problem(
        "sum_absolute",
        variant,
        "arrays",
        "Return the sum of absolute values of all elements.",
        "fn sum_absolute(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3])], 6),
            example(vec![array(&[-4, -5])], 9),
            example(vec![array(&[0, 3, -3])], 6),
            example(vec![array(&[2, 2, 2])], 6),
        ],
        vec![
            example(vec![array(&[-1, 1, -1, 1])], 4),
            example(vec![array(&[10, -3])], 13),
        ],
        "fn sum_absolute(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if x < 0 {\n            acc = acc + (0 - x);\n        } else {\n            acc = acc + x;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_count_evens(variant: usize) -> Problem {
    problem(
        "count_evens",
        variant,
        "arrays",
        "Return the count of even elements in the array.",
        "fn count_evens(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], 2),
            example(vec![array(&[1, 3, 5])], 0),
            example(vec![array(&[2, 4, 6])], 3),
            example(vec![array(&[0, 1, 2, 3, 4])], 3),
        ],
        vec![
            example(vec![array(&[2, 2, 1, 1])], 2),
            example(vec![array(&[7, 8, 9])], 1),
        ],
        "fn count_evens(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if (x % 2) == 0 {\n            acc = acc + 1;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_sum_positives(variant: usize) -> Problem {
    problem(
        "sum_positives",
        variant,
        "arrays",
        "Return the sum of all positive elements (excluding zero).",
        "fn sum_positives(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -2, 3, -4])], 4),
            example(vec![array(&[-1, -2, -3])], 0),
            example(vec![array(&[5, 0, 3])], 8),
            example(vec![array(&[2, 4, 6])], 12),
        ],
        vec![
            example(vec![array(&[0, 0, 1])], 1),
            example(vec![array(&[-5, 10, -3, 2])], 12),
        ],
        "fn sum_positives(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    for x in arr {\n        if x > 0 {\n            acc = acc + x;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_sum_at_even_indices(variant: usize) -> Problem {
    problem(
        "sum_at_even_indices",
        variant,
        "arrays",
        "Return the sum of elements at even indices (0, 2, 4, ...).",
        "fn sum_at_even_indices(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5])], 9),
            example(vec![array(&[10, 20, 30])], 40),
            example(vec![array(&[5, 5])], 5),
            example(vec![array(&[1, 2, 3, 4])], 4),
        ],
        vec![
            example(vec![array(&[2, 1, 4, 3])], 6),
            example(vec![array(&[7, 0, 3, 0, 1])], 11),
        ],
        "fn sum_at_even_indices(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        acc = acc + arr[i];\n        i = i + 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_kth_from_end(variant: usize) -> Problem {
    problem(
        "kth_from_end",
        variant,
        "arrays",
        "Return the kth element from the end (k=1 is the last).",
        "fn kth_from_end(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 3),
            example(vec![array(&[10, 20]), int(2)], 10),
            example(vec![array(&[7, 8, 9]), int(2)], 8),
        ],
        vec![
            example(vec![array(&[1, 2, 3]), int(1)], 3),
            example(vec![array(&[5, 10, 15, 20]), int(3)], 10),
        ],
        "fn kth_from_end(arr: [i64], k: i64) -> i64 {\n    return arr[arr.len - k];\n}\n",
    )
}

fn make_max_abs(variant: usize) -> Problem {
    problem(
        "max_abs",
        variant,
        "arrays",
        "Return the largest absolute value in the array.",
        "fn max_abs(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, -5, 3])], 5),
            example(vec![array(&[-2, -7, 4])], 7),
            example(vec![array(&[3, 3, 3])], 3),
            example(vec![array(&[0, -1])], 1),
        ],
        vec![
            example(vec![array(&[-10, 5, -3])], 10),
            example(vec![array(&[4, -4])], 4),
        ],
        "fn max_abs(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    for x in arr {\n        v: i64 = x;\n        if v < 0 {\n            v = 0 - v;\n        }\n        if v > best {\n            best = v;\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_alternating_sum(variant: usize) -> Problem {
    problem(
        "alternating_sum",
        variant,
        "array",
        "Return the alternating sum: arr[0] - arr[1] + arr[2] - arr[3] + ...",
        "fn alternating_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], -2),
            example(vec![array(&[5])], 5),
            example(vec![array(&[3, 1])], 2),
            example(vec![array(&[2, 4, 6, 8])], -4),
        ],
        vec![
            example(vec![array(&[1, 1, 1, 1])], 0),
            example(vec![array(&[10, 3, 2, 1])], 8),
        ],
        "fn alternating_sum(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    sign: i64 = 1;\n    while i < arr.len {\n        acc = acc + sign * arr[i];\n        sign = 0 - sign;\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_count_greater_than(variant: usize) -> Problem {
    problem(
        "count_greater_than",
        variant,
        "array",
        "Return the count of elements greater than k.",
        "fn count_greater_than(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 5, 3, 7, 2]), int(4)], 2),
            example(vec![array(&[1, 2, 3]), int(0)], 3),
            example(vec![array(&[4, 4, 4]), int(4)], 0),
            example(vec![array(&[10, 20, 30]), int(15)], 2),
        ],
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 2),
            example(vec![array(&[0, -1, 1]), int(0)], 1),
        ],
        "fn count_greater_than(arr: [i64], k: i64) -> i64 {\n    acc: i64 = 0;\n    for item in arr {\n        if item > k {\n            acc = acc + 1;\n        }\n    }\n    return acc;\n}\n",
    )
}

fn make_dot_product(variant: usize) -> Problem {
    problem(
        "dot_product",
        variant,
        "array",
        "Return the dot product: sum of a[i] * b[i] for all i.",
        "fn dot_product(a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3]), array(&[4, 5, 6])], 32),
            example(vec![array(&[1, 0]), array(&[0, 1])], 0),
            example(vec![array(&[2, 3]), array(&[3, 2])], 12),
            example(vec![array(&[1, 1, 1, 1]), array(&[1, 2, 3, 4])], 10),
        ],
        vec![
            example(vec![array(&[5, 0, 1]), array(&[1, 1, 1])], 6),
            example(vec![array(&[2, 2]), array(&[3, 3])], 12),
        ],
        "fn dot_product(a: [i64], b: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < a.len {\n        acc = acc + a[i] * b[i];\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_leading_digit(variant: usize) -> Problem {
    problem(
        "leading_digit",
        variant,
        "integer",
        "Return the leading (most significant) digit of a positive integer.",
        "fn leading_digit(n: i64) -> i64",
        vec![
            example(vec![int(1234)], 1),
            example(vec![int(5)], 5),
            example(vec![int(100)], 1),
            example(vec![int(73)], 7),
        ],
        vec![
            example(vec![int(999)], 9),
            example(vec![int(42)], 4),
        ],
        "fn leading_digit(n: i64) -> i64 {\n    x: i64 = n;\n    while x >= 10 {\n        x = x / 10;\n    }\n    return x;\n}\n",
    )
}

fn make_popcount(variant: usize) -> Problem {
    problem(
        "popcount",
        variant,
        "integer",
        "Return the number of 1-bits in the binary representation of n (n >= 0).",
        "fn popcount(n: i64) -> i64",
        vec![
            example(vec![int(5)], 2),   // 101
            example(vec![int(7)], 3),   // 111
            example(vec![int(8)], 1),   // 1000
            example(vec![int(15)], 4),  // 1111
        ],
        vec![
            example(vec![int(0)], 0),
            example(vec![int(255)], 8),
        ],
        "fn popcount(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 0;\n    while x > 0 {\n        acc = acc + x % 2;\n        x = x / 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_prefix_sum_k(variant: usize) -> Problem {
    problem(
        "prefix_sum_k",
        variant,
        "array",
        "Return the sum of the first k elements.",
        "fn prefix_sum_k(arr: [i64], k: i64) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5]), int(3)], 6),
            example(vec![array(&[10, 20, 30]), int(2)], 30),
            example(vec![array(&[5]), int(1)], 5),
            example(vec![array(&[1, 2, 3, 4, 5]), int(5)], 15),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 1, 5]), int(2)], 4),
            example(vec![array(&[7, 7, 7]), int(3)], 21),
        ],
        "fn prefix_sum_k(arr: [i64], k: i64) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 0;\n    while i < k {\n        acc = acc + arr[i];\n        i = i + 1;\n    }\n    return acc;\n}\n",
    )
}

fn make_is_palindrome_arr(variant: usize) -> Problem {
    problem(
        "is_palindrome_arr",
        variant,
        "array",
        "Return 1 if the array reads the same forwards and backwards, 0 otherwise.",
        "fn is_palindrome_arr(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 1])], 1),
            example(vec![array(&[1, 2, 3])], 0),
            example(vec![array(&[3, 1, 3])], 1),
            example(vec![array(&[1])], 1),
        ],
        vec![
            example(vec![array(&[1, 2, 2, 1])], 1),
            example(vec![array(&[1, 2, 3, 4])], 0),
        ],
        "fn is_palindrome_arr(arr: [i64]) -> i64 {\n    i: i64 = 0;\n    j: i64 = arr.len - 1;\n    while i < j {\n        if arr[i] != arr[j] {\n            return 0;\n        }\n        i = i + 1;\n        j = j - 1;\n    }\n    return 1;\n}\n",
    )
}

fn make_sum_odd_indexed(variant: usize) -> Problem {
    problem(
        "sum_odd_indexed",
        variant,
        "array",
        "Return the sum of elements at odd indices (1, 3, 5, ...).",
        "fn sum_odd_indexed(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4])], 6),   // 2+4
            example(vec![array(&[5, 3])], 3),
            example(vec![array(&[1, 10, 1, 10, 1])], 20),
            example(vec![array(&[0, 7, 0, 8])], 15),
        ],
        vec![
            example(vec![array(&[1, 1, 1, 1, 1, 1])], 3),
            example(vec![array(&[0, 5, 0, 3])], 8),
        ],
        "fn sum_odd_indexed(arr: [i64]) -> i64 {\n    acc: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        acc = acc + arr[i];\n        i = i + 2;\n    }\n    return acc;\n}\n",
    )
}

fn make_digit_product(variant: usize) -> Problem {
    problem(
        "digit_product",
        variant,
        "integer",
        "Return the product of the digits of a positive integer.",
        "fn digit_product(n: i64) -> i64",
        vec![
            example(vec![int(123)], 6),
            example(vec![int(24)], 8),
            example(vec![int(100)], 0),
            example(vec![int(9)], 9),
        ],
        vec![
            example(vec![int(11)], 1),
            example(vec![int(55)], 25),
        ],
        "fn digit_product(n: i64) -> i64 {\n    x: i64 = n;\n    acc: i64 = 1;\n    while x > 0 {\n        acc = acc * (x % 10);\n        x = x / 10;\n    }\n    return acc;\n}\n",
    )
}

fn make_max_digit(variant: usize) -> Problem {
    problem(
        "max_digit",
        variant,
        "integer",
        "Return the maximum digit of a non-negative integer.",
        "fn max_digit(n: i64) -> i64",
        vec![
            example(vec![int(1234)], 4),
            example(vec![int(9000)], 9),
            example(vec![int(123)], 3),
            example(vec![int(55)], 5),
        ],
        vec![
            example(vec![int(999)], 9),
            example(vec![int(21)], 2),
        ],
        "fn max_digit(n: i64) -> i64 {\n    x: i64 = n;\n    best: i64 = 0;\n    while x > 0 {\n        d: i64 = x % 10;\n        if d > best {\n            best = d;\n        }\n        x = x / 10;\n    }\n    return best;\n}\n",
    )
}

fn make_min_positive(variant: usize) -> Problem {
    problem(
        "min_positive",
        variant,
        "array",
        "Return the minimum positive element, or 0 if no positive elements exist.",
        "fn min_positive(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 5, 3])], 1),
            example(vec![array(&[2, 8, 6])], 2),
            example(vec![array(&[-1, -2, -3])], 0),
            example(vec![array(&[10, 1, 5])], 1),
        ],
        vec![
            example(vec![array(&[3, 7, 2, 4])], 2),
            example(vec![array(&[0, 0, 0])], 0),
        ],
        "fn min_positive(arr: [i64]) -> i64 {\n    best: i64 = 0;\n    found: i64 = 0;\n    for x in arr {\n        if x > 0 {\n            if found == 0 {\n                best = x;\n                found = 1;\n            } else {\n                if x < best {\n                    best = x;\n                }\n            }\n        }\n    }\n    return best;\n}\n",
    )
}

fn make_lucas_number(variant: usize) -> Problem {
    problem(
        "lucas_number",
        variant,
        "math",
        "Return the nth Lucas number: L(0)=2, L(1)=1, L(n)=L(n-1)+L(n-2).",
        "fn lucas_number(n: i64) -> i64",
        vec![
            example(vec![int(0)], 2),
            example(vec![int(1)], 1),
            example(vec![int(4)], 7),
            example(vec![int(6)], 18),
        ],
        vec![
            example(vec![int(3)], 4),
            example(vec![int(7)], 29),
        ],
        "fn lucas_number(n: i64) -> i64 {\n    if n == 0 {\n        return 2;\n    }\n    if n == 1 {\n        return 1;\n    }\n    a: i64 = 2;\n    b: i64 = 1;\n    i: i64 = 2;\n    while i <= n {\n        tmp: i64 = a + b;\n        a = b;\n        b = tmp;\n        i = i + 1;\n    }\n    return b;\n}\n",
    )
}

fn make_celsius_to_fahrenheit(variant: usize) -> Problem {
    problem(
        "celsius_to_fahrenheit",
        variant,
        "math",
        "Convert Celsius to Fahrenheit using integer arithmetic: c * 9 / 5 + 32.",
        "fn celsius_to_fahrenheit(c: i64) -> i64",
        vec![
            example(vec![int(0)], 32),
            example(vec![int(100)], 212),
            example(vec![int(20)], 68),
            example(vec![int(37)], 98),
        ],
        vec![example(vec![int(-40)], -40), example(vec![int(25)], 77)],
        "fn celsius_to_fahrenheit(c: i64) -> i64 {\n    return c * 9 / 5 + 32;\n}\n",
    )
}

fn make_is_perfect_square(variant: usize) -> Problem {
    problem(
        "is_perfect_square",
        variant,
        "math",
        "Return 1 if n is a perfect square, 0 otherwise (n >= 0).",
        "fn is_perfect_square(n: i64) -> i64",
        vec![
            example(vec![int(4)], 1),
            example(vec![int(9)], 1),
            example(vec![int(10)], 0),
            example(vec![int(1)], 1),
        ],
        vec![
            example(vec![int(25)], 1),
            example(vec![int(26)], 0),
        ],
        "fn is_perfect_square(n: i64) -> i64 {\n    i: i64 = 0;\n    while i * i <= n {\n        if i * i == n {\n            return 1;\n        }\n        i = i + 1;\n    }\n    return 0;\n}\n",
    )
}

fn make_next_power_of_2(variant: usize) -> Problem {
    problem(
        "next_power_of_2",
        variant,
        "math",
        "Return the smallest power of 2 that is >= n.",
        "fn next_power_of_2(n: i64) -> i64",
        vec![
            example(vec![int(1)], 1),
            example(vec![int(3)], 4),
            example(vec![int(5)], 8),
            example(vec![int(8)], 8),
        ],
        vec![
            example(vec![int(9)], 16),
            example(vec![int(16)], 16),
        ],
        "fn next_power_of_2(n: i64) -> i64 {\n    p: i64 = 1;\n    while p < n {\n        p = p * 2;\n    }\n    return p;\n}\n",
    )
}

fn make_count_peaks(variant: usize) -> Problem {
    problem(
        "count_peaks",
        variant,
        "array",
        "Count elements strictly greater than both their neighbors.",
        "fn count_peaks(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 3, 2])], 1),
            example(vec![array(&[1, 2, 3])], 0),
            example(vec![array(&[1, 3, 2, 4, 1])], 2),
            example(vec![array(&[5, 1, 5, 1, 5])], 1),
        ],
        vec![
            example(vec![array(&[3, 1, 4, 1, 5, 9, 2])], 2),
            example(vec![array(&[1, 2, 3, 2, 1])], 1),
        ],
        "fn count_peaks(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len - 1 {\n        if arr[i] > arr[i - 1] {\n            if arr[i] > arr[i + 1] {\n                count = count + 1;\n            }\n        }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

// ---------------------------------------------------------------------------
// Tier-2: game-adjacent problems (April 2026)
//
// These push past the single-clean-formula shapes of Tier-1 and require
// multi-branch dispatch, multi-arg coupling, or compositional array work.
// They are starting points toward game logic and app-style control flow.
// ---------------------------------------------------------------------------

fn make_score_tracker(variant: usize) -> Problem {
    problem(
        "score_tracker",
        variant,
        "game",
        "Update score: event 0 adds 1, event 1 adds 5, event 2 resets to 0, else unchanged.",
        "fn score_tracker(score: i64, event: i64) -> i64",
        vec![
            example(vec![int(10), int(0)], 11),
            example(vec![int(10), int(1)], 15),
            example(vec![int(10), int(2)], 0),
            example(vec![int(10), int(7)], 10),
            example(vec![int(0), int(1)], 5),
        ],
        vec![
            example(vec![int(42), int(0)], 43),
            example(vec![int(99), int(2)], 0),
        ],
        "fn score_tracker(score: i64, event: i64) -> i64 {\n    if event == 0 { return score + 1; }\n    if event == 1 { return score + 5; }\n    if event == 2 { return 0; }\n    return score;\n}\n",
    )
}

/// Per-tick frame reducer: a 1-arg `frame` array. The state register
/// is a memory cell at the *call site* — the user threads it
/// between invocations to carry a running value (e.g. score). The
/// reference program picks the array's max as the per-frame
/// contribution; state + max is the per-tick fold. The array_max
/// teacher covers the inner reduction.
///
/// (The (scalar, array) → scalar 2-arg shape is the canonical
/// "stateful" problem, but the current search teachers don't cover
/// that signature; we keep the frame as a 1-arg array and let the
/// state live at the call site. True 3D/4D tensor registers held
/// across ticks need a new search space; see
/// `docs/stateful_synthesis_status.md` for the gap.)
fn make_tensor_3d_per_frame(variant: usize) -> Problem {
    problem(
        "tensor_3d_per_frame",
        variant,
        "game",
        "Per-frame contribution: max(frame). The 3D coord is a (x, y, z) read-only input.",
        "fn tensor_3d_per_frame(frame: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3])], 3),
            example(vec![array(&[5, 0, 0])], 5),
            example(vec![array(&[2, 9, 1])], 9),
            example(vec![array(&[-1, 0, 0])], 0),
        ],
        vec![
            example(vec![array(&[42, 0, 0])], 42),
            example(vec![array(&[0, 1, 2])], 2),
        ],
        "fn tensor_3d_per_frame(frame: [i64]) -> i64 {\n    best: i64 = frame[0];\n    for v in frame {\n        if v > best { best = v; }\n    }\n    return best;\n}\n",
    )
}

/// Exponential moving average — a 2-arg `(state, sample) -> state`
/// reducer. This is the canonical 1-state-register loop, the simplest
/// meaningful "memory" program: each tick, the new state is a blend
/// of the previous state and the new sample. Solved by the existing
/// 2-branch teacher; demonstrates persistent memory across ticks.
fn make_ema_state(variant: usize) -> Problem {
    problem(
        "ema_state",
        variant,
        "game",
        "Exponential moving average: state = (state + sample) / 2 (integer division).",
        "fn ema_state(state: i64, sample: i64) -> i64",
        vec![
            example(vec![int(0), int(10)], 5),
            example(vec![int(10), int(10)], 10),
            example(vec![int(10), int(20)], 15),
            example(vec![int(0), int(0)], 0),
            example(vec![int(100), int(0)], 50),
            example(vec![int(-10), int(10)], 0),
        ],
        vec![
            example(vec![int(7), int(11)], 9),
            example(vec![int(50), int(50)], 50),
        ],
        "fn ema_state(state: i64, sample: i64) -> i64 {\n    return (state + sample) / 2;\n}\n",
    )
}

/// 1-slot memory cell: the per-tick function remembers the previous
/// tick's output and exposes it as the *next* tick's first argument.
/// Composed over a stream of inputs, this is the canonical "memory"
/// primitive — it shows up everywhere from LSTMs to game save state.
fn make_memory_cell(variant: usize) -> Problem {
    problem(
        "memory_cell",
        variant,
        "game",
        "Return the previous output (carry), or input on the first tick. carry = max(prev_carry, input).",
        "fn memory_cell(prev_carry: i64, input: i64) -> i64",
        vec![
            example(vec![int(0), int(5)], 5),
            example(vec![int(5), int(3)], 5),
            example(vec![int(5), int(10)], 10),
            example(vec![int(10), int(-1)], 10),
            example(vec![int(0), int(0)], 0),
        ],
        vec![
            example(vec![int(7), int(2)], 7),
            example(vec![int(7), int(20)], 20),
        ],
        "fn memory_cell(prev_carry: i64, input: i64) -> i64 {\n    if prev_carry > input { return prev_carry; }\n    return input;\n}\n",
    )
}

/// Note: the cap-on-double rule (event 2 doubles state but caps at 100)
/// requires a nested if/else, which the simple 2-branch teacher doesn't
/// cover. We split the rule: event 2 returns state * 2; a separate
/// post-pass can clamp at the call site. That keeps the per-tick reducer
/// in the simple 4-event-table shape the teacher can discover.
fn make_game_tick(variant: usize) -> Problem {
    problem(
        "game_tick",
        variant,
        "game",
        "Per-tick state update: 0=+1, 1=-1, 2=double, 3=reset, else unchanged.",
        "fn game_tick(state: i64, event: i64) -> i64",
        vec![
            example(vec![int(10), int(0)], 11),
            example(vec![int(10), int(1)], 9),
            example(vec![int(10), int(2)], 20),
            example(vec![int(10), int(3)], 0),
            example(vec![int(60), int(2)], 120),
            example(vec![int(0), int(1)], -1),
            example(vec![int(50), int(7)], 50),
            example(vec![int(7), int(0)], 8),
        ],
        vec![
            example(vec![int(42), int(0)], 43),
            example(vec![int(99), int(2)], 198),
        ],
        "fn game_tick(state: i64, event: i64) -> i64 {\n    if event == 0 { return state + 1; }\n    if event == 1 { return state - 1; }\n    if event == 2 { return state * 2; }\n    if event == 3 { return 0; }\n    return state;\n}\n",
    )
}

fn make_vending_change(variant: usize) -> Problem {
    problem(
        "vending_change",
        variant,
        "game",
        "Return coins_in - price if coins_in >= price, else -1.",
        "fn vending_change(coins_in: i64, price: i64) -> i64",
        vec![
            example(vec![int(100), int(75)], 25),
            example(vec![int(50), int(50)], 0),
            example(vec![int(30), int(75)], -1),
            example(vec![int(200), int(125)], 75),
            example(vec![int(10), int(20)], -1),
        ],
        vec![
            example(vec![int(80), int(80)], 0),
            example(vec![int(0), int(5)], -1),
        ],
        "fn vending_change(coins_in: i64, price: i64) -> i64 {\n    if coins_in >= price { return coins_in - price; }\n    return -1;\n}\n",
    )
}

fn make_combat_resolve(variant: usize) -> Problem {
    problem(
        "combat_resolve",
        variant,
        "game",
        "Damage dealt is max(attack - defense, 0).",
        "fn combat_resolve(attack: i64, defense: i64) -> i64",
        vec![
            example(vec![int(15), int(10)], 5),
            example(vec![int(5), int(10)], 0),
            example(vec![int(20), int(20)], 0),
            example(vec![int(100), int(1)], 99),
            example(vec![int(0), int(50)], 0),
        ],
        vec![
            example(vec![int(7), int(3)], 4),
            example(vec![int(50), int(75)], 0),
        ],
        "fn combat_resolve(attack: i64, defense: i64) -> i64 {\n    d: i64 = attack - defense;\n    if d < 0 { return 0; }\n    return d;\n}\n",
    )
}

fn make_traffic_light_phase(variant: usize) -> Problem {
    problem(
        "traffic_light_phase",
        variant,
        "game",
        "7-step cycle: steps 0-2 → 0 (red), 3-5 → 1 (green), 6 → 2 (yellow).",
        "fn traffic_light_phase(step: i64) -> i64",
        vec![
            example(vec![int(0)], 0),
            example(vec![int(1)], 0),
            example(vec![int(2)], 0),
            example(vec![int(3)], 1),
            example(vec![int(5)], 1),
            example(vec![int(6)], 2),
            example(vec![int(7)], 0),
            example(vec![int(13)], 2),
        ],
        vec![
            example(vec![int(14)], 0),
            example(vec![int(20)], 2),
        ],
        "fn traffic_light_phase(step: i64) -> i64 {\n    m: i64 = step % 7;\n    if m < 3 { return 0; }\n    if m < 6 { return 1; }\n    return 2;\n}\n",
    )
}

fn make_run_length_decode_sum(variant: usize) -> Problem {
    problem(
        "run_length_decode_sum",
        variant,
        "array",
        "Interpret array as (count, value) pairs; return sum of count*value products.",
        "fn run_length_decode_sum(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 5, 2, 10])], 35),
            example(vec![array(&[1, 7])], 7),
            example(vec![array(&[4, 2, 3, 1])], 11),
            example(vec![array(&[2, 3, 2, 4, 2, 5])], 24),
        ],
        vec![
            example(vec![array(&[5, 1, 5, 1])], 10),
            example(vec![array(&[1, 100])], 100),
        ],
        "fn run_length_decode_sum(arr: [i64]) -> i64 {\n    total: i64 = 0;\n    i: i64 = 0;\n    while i < arr.len {\n        total = total + arr[i] * arr[i + 1];\n        i = i + 2;\n    }\n    return total;\n}\n",
    )
}

/// Stage-1 stateful benchmark: `f(state, arr) = state + sum(arr)`.
/// Mirrors the canonical per-tick game-loop reducer where a
/// state register carries the player's score across frames and
/// the per-frame contribution is the array's total (events array,
/// damage batch, etc.). The `search_stateful_reducer` teacher
/// covers this signature end-to-end.
fn make_stateful_reducer(variant: usize) -> Problem {
    problem(
        "stateful_reducer",
        variant,
        "game",
        "new_state = state + (sum of frame). The state register carries between calls; the frame array is per-call.",
        "fn stateful_reducer(state: i64, frame: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, 2, 3])], 6),
            example(vec![int(10), array(&[5, 0, 0])], 15),
            example(vec![int(-5), array(&[2, 9, 1])], 7),
            example(vec![int(100), array(&[-1, 0, 0])], 99),
        ],
        vec![
            example(vec![int(0), array(&[42, 0, 0])], 42),
            example(vec![int(7), array(&[0, 1, 2])], 10),
        ],
        "fn stateful_reducer(state: i64, frame: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in frame {\n        s = s + v;\n    }\n    return state + s;\n}\n",
    )
}

fn make_count_adjacent_diff(variant: usize) -> Problem {
    problem(
        "count_adjacent_diff",
        variant,
        "array",
        "Count positions i > 0 where arr[i] != arr[i-1].",
        "fn count_adjacent_diff(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[1, 1, 2, 2, 3])], 2),
            example(vec![array(&[5, 5, 5, 5])], 0),
            example(vec![array(&[1, 2, 1, 2, 1])], 4),
            example(vec![array(&[7])], 0),
        ],
        vec![
            example(vec![array(&[0, 0, 1, 0, 0])], 2),
            example(vec![array(&[1, 2, 3, 4])], 3),
        ],
        "fn count_adjacent_diff(arr: [i64]) -> i64 {\n    count: i64 = 0;\n    i: i64 = 1;\n    while i < arr.len {\n        if arr[i] != arr[i - 1] { count = count + 1; }\n        i = i + 1;\n    }\n    return count;\n}\n",
    )
}

fn make_priority_pop(variant: usize) -> Problem {
    problem(
        "priority_pop",
        variant,
        "array",
        "Return the maximum element (simulated pop from a heap).",
        "fn priority_pop(arr: [i64]) -> i64",
        vec![
            example(vec![array(&[3, 1, 4, 1, 5, 9, 2, 6])], 9),
            example(vec![array(&[7])], 7),
            example(vec![array(&[1, 2, 3, 4])], 4),
            example(vec![array(&[5, 5, 5, 5])], 5),
        ],
        vec![
            example(vec![array(&[10, 20, 15])], 20),
            example(vec![array(&[-5, -2, -10])], -2),
        ],
        "fn priority_pop(arr: [i64]) -> i64 {\n    best: i64 = arr[0];\n    for x in arr { if x > best { best = x; } }\n    return best;\n}\n",
    )
}

fn make_turn_order_rotate(variant: usize) -> Problem {
    problem(
        "turn_order_rotate",
        variant,
        "game",
        "Advance current player index in a ring of num_players.",
        "fn turn_order_rotate(current: i64, num_players: i64) -> i64",
        vec![
            example(vec![int(0), int(4)], 1),
            example(vec![int(3), int(4)], 0),
            example(vec![int(1), int(3)], 2),
            example(vec![int(2), int(3)], 0),
            example(vec![int(0), int(2)], 1),
        ],
        vec![
            example(vec![int(4), int(5)], 0),
            example(vec![int(0), int(1)], 0),
        ],
        "fn turn_order_rotate(current: i64, num_players: i64) -> i64 {\n    return (current + 1) % num_players;\n}\n",
    )
}

fn make_grid_bounds_check(variant: usize) -> Problem {
    problem(
        "grid_bounds_check",
        variant,
        "game",
        "Return 1 if (x,y) is inside an w×h grid (0 ≤ x < w, 0 ≤ y < h), else 0.",
        "fn grid_bounds_check(x: i64, y: i64, w: i64, h: i64) -> i64",
        vec![
            example(vec![int(0), int(0), int(5), int(5)], 1),
            example(vec![int(4), int(4), int(5), int(5)], 1),
            example(vec![int(5), int(0), int(5), int(5)], 0),
            example(vec![int(-1), int(0), int(5), int(5)], 0),
            example(vec![int(0), int(-1), int(5), int(5)], 0),
            example(vec![int(2), int(3), int(10), int(10)], 1),
        ],
        vec![
            example(vec![int(10), int(10), int(10), int(10)], 0),
            example(vec![int(3), int(3), int(5), int(3)], 0),
        ],
        "fn grid_bounds_check(x: i64, y: i64, w: i64, h: i64) -> i64 {\n    if x < 0 { return 0; }\n    if y < 0 { return 0; }\n    if x >= w { return 0; }\n    if y >= h { return 0; }\n    return 1;\n}\n",
    )
}

fn make_simulate_gravity(variant: usize) -> Problem {
    problem(
        "simulate_gravity",
        variant,
        "game",
        "Integrate velocity: v + g*t, then clamp to terminal velocity 100.",
        "fn simulate_gravity(v: i64, g: i64, t: i64) -> i64",
        vec![
            example(vec![int(0), int(10), int(3)], 30),
            example(vec![int(5), int(2), int(4)], 13),
            example(vec![int(50), int(20), int(5)], 100),
            example(vec![int(-10), int(5), int(2)], 0),
            example(vec![int(0), int(0), int(10)], 0),
        ],
        vec![
            example(vec![int(20), int(10), int(10)], 100),
            example(vec![int(3), int(1), int(7)], 10),
        ],
        "fn simulate_gravity(v: i64, g: i64, t: i64) -> i64 {\n    r: i64 = v + g * t;\n    if r > 100 { return 100; }\n    if r < 0 { return 0; }\n    return r;\n}\n",
    )
}

// ====================================================================
// Stage 2 starter: event-modulated stateful benchmarks.
//
// Each has the (state, event, arr) -> state signature covered by
// `search_stateful_reducer_event`. The benchmarks use realistic
// game-loop / physics / inventory patterns where the event scalar
// decides how the array contributes to the state. See
// `docs/stateful_synthesis_status.md` Stage 2 section.
// ====================================================================

/// 1D physics step: `f(pos, vel, dt) = pos + vel * dt`.
///
/// Standard semi-implicit Euler for a single axis. The event
/// (`dt`) is a positive time delta; the array `vel` is the per-frame
/// velocity samples (treated as a 1D reduction by the teacher).
/// Tests the `mul_event` combine with `sum` reducer.
fn make_physics_step_1d(variant: usize) -> Problem {
    problem(
        "physics_step_1d",
        variant,
        "game",
        "1D Euler step: pos += vel_sum * dt (treats vel array as scalar via sum).",
        "fn physics_step_1d(pos: i64, vel: i64, arr: [i64]) -> i64",
        vec![
            // pos=0, vel=3, arr=[1,2,3] -> 0 + 3*6 = 18
            example(vec![int(0), int(3), array(&[1, 2, 3])], 18),
            // pos=10, vel=2, arr=[1,4] -> 10 + 2*5 = 20
            example(vec![int(10), int(2), array(&[1, 4])], 20),
            // pos=5, vel=0, arr=[1,3] -> 5 + 0*4 = 5  (dt=0 means hold)
            example(vec![int(5), int(0), array(&[1, 3])], 5),
            // pos=-3, vel=-2, arr=[1,3] -> -3 + -2*4 = -11
            example(vec![int(-3), int(-2), array(&[1, 3])], -11),
            // pos=100, vel=1, arr=[2,2,2] -> 100 + 1*6 = 106
            example(vec![int(100), int(1), array(&[2, 2, 2])], 106),
        ],
        vec![
            example(vec![int(7), int(3), array(&[0, 0, 5])], 22),
            example(vec![int(0), int(1), array(&[10])], 10),
        ],
        // Reference Mog: state = pos, event = dt, r = sum(arr), return pos + dt*r
        "fn physics_step_1d(pos: i64, dt: i64, arr: [i64]) -> i64 {\n    r: i64 = 0;\n    for v in arr { r = r + v; }\n    return pos + dt * r;\n}\n",
    )
}

/// Brake accumulator: `f(state, brake, arr) = if brake <= 0 then state + sum(arr) else state`.
///
/// The `brake` event is a positive integer when the brake is
/// engaged; when zero or negative, the per-tick array contribution
/// is added to the state. Tests the `event_le_0` gate.
fn make_brake_accumulator(variant: usize) -> Problem {
    problem(
        "brake_accumulator",
        variant,
        "game",
        "If brake>0: hold state. Else (brake <= 0): state += sum(arr).",
        "fn brake_accumulator(state: i64, brake: i64, arr: [i64]) -> i64",
        vec![
            // brake=0 -> state + sum = 0 + 6 = 6
            example(vec![int(0), int(0), array(&[1, 2, 3])], 6),
            // brake=1 -> state held = 7
            example(vec![int(7), int(1), array(&[1, 2, 3])], 7),
            // brake=-1 -> state + sum = -2 + 30 = 28
            example(vec![int(-2), int(-1), array(&[10, 20])], 28),
            // brake=5 -> state held = 4
            example(vec![int(4), int(5), array(&[4, 5])], 4),
            // brake=0 again, negative sum: 100 + (-6) = 94
            example(vec![int(100), int(0), array(&[-1, -2, -3])], 94),
        ],
        vec![
            example(vec![int(50), int(0), array(&[1, 1, 1, 1])], 54),
            example(vec![int(50), int(1), array(&[1, 1, 1, 1])], 50),
        ],
        "fn brake_accumulator(state: i64, brake: i64, arr: [i64]) -> i64 {\n    r: i64 = 0;\n    for v in arr { r = r + v; }\n    if brake <= 0 { return state + r; }\n    return state;\n}\n",
    )
}

/// Boost modulated: `f(state, boost, arr) = state + boost * count_positive(arr)`.
///
/// The `boost` event multiplies the number of positive entries in
/// the array. Tests `mul_event` combine with the `count_positive`
/// reducer — a non-sum reduction.
fn make_boost_modulated(variant: usize) -> Problem {
    problem(
        "boost_modulated",
        variant,
        "game",
        "state += boost * count_positive(arr).",
        "fn boost_modulated(state: i64, boost: i64, arr: [i64]) -> i64",
        vec![
            // boost=3, count_pos of [1,-2,3] = 2 -> 0 + 3*2 = 6
            example(vec![int(0), int(3), array(&[1, -2, 3])], 6),
            // boost=2, count_pos of [1,4] = 2 -> 10 + 2*2 = 14
            example(vec![int(10), int(2), array(&[1, 4])], 14),
            // boost=0 -> 5 + 0*2 = 5
            example(vec![int(5), int(0), array(&[1, -1])], 5),
            // boost=-1, count_pos of [-1,-2] = 0 -> 7 + -1*0 = 7
            example(vec![int(7), int(-1), array(&[-1, -2])], 7),
            // boost=4, count_pos of [0,1,2,3] = 3 -> 1 + 4*3 = 13
            example(vec![int(1), int(4), array(&[0, 1, 2, 3])], 13),
        ],
        vec![
            example(vec![int(20), int(2), array(&[-5, -10, -15])], 20),
            example(vec![int(0), int(5), array(&[1, 1, 1])], 15),
        ],
        "fn boost_modulated(state: i64, boost: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { if v > 0 { s = s + 1; } }\n    return state + boost * s;\n}\n",
    )
}

/// Turn counter gated on zero: `f(state, turn, arr) = if turn != 0 then state + sum(arr) else state`.
///
/// The `turn` event is the current turn index; on the opening turn
/// (`turn == 0`) the state is held, on subsequent turns the array
/// contributes. Tests the `event_eq_0` gate.
fn make_turn_counter_gated(variant: usize) -> Problem {
    problem(
        "turn_counter_gated",
        variant,
        "game",
        "On turn==0 hold state, else state += sum(arr).",
        "fn turn_counter_gated(state: i64, turn: i64, arr: [i64]) -> i64",
        vec![
            // turn=0 -> hold = 5
            example(vec![int(5), int(0), array(&[1, 2, 3])], 5),
            // turn=1 -> state + sum = 5+6 = 11
            example(vec![int(5), int(1), array(&[1, 2, 3])], 11),
            // turn=2 -> state + sum = 11+6 = 17
            example(vec![int(11), int(2), array(&[1, 2, 3])], 17),
            // turn=0 with negative state
            example(vec![int(-10), int(0), array(&[5])], -10),
            // turn=3 with bigger array
            example(vec![int(0), int(3), array(&[2, 2, 2, 2])], 8),
        ],
        vec![
            example(vec![int(100), int(0), array(&[1, 2])], 100),
            example(vec![int(100), int(5), array(&[1, 2])], 103),
        ],
        "fn turn_counter_gated(state: i64, turn: i64, arr: [i64]) -> i64 {\n    r: i64 = 0;\n    for v in arr { r = r + v; }\n    if turn == 0 { return state; }\n    return state + r;\n}\n",
    )
}

/// Damage with event-modulated subtraction: `f(state, event, arr) = state - event - sum(arr)`.
///
/// The event is the flat-damage scalar; the array is the per-hit
/// damage rolls. Combined subtraction. Tests the `add_event`
/// combine with `op = -`.
fn make_damage_with_event(variant: usize) -> Problem {
    problem(
        "damage_with_event",
        variant,
        "game",
        "state -= event + sum(arr) (event is flat damage, arr is per-hit rolls).",
        "fn damage_with_event(state: i64, event: i64, arr: [i64]) -> i64",
        vec![
            // state=10, event=2, arr=[1,1,1] -> 10 - 2 - 3 = 5
            example(vec![int(10), int(2), array(&[1, 1, 1])], 5),
            // state=100, event=0, arr=[5,5] -> 100 - 0 - 10 = 90
            example(vec![int(100), int(0), array(&[5, 5])], 90),
            // state=0, event=5, arr=[-2,-3] -> 0 - 5 - (-5) = 0
            example(vec![int(0), int(5), array(&[-2, -3])], 0),
            // state=20, event=10, arr=[] -> 20 - 10 - 0 = 10
            example(vec![int(20), int(10), array(&[])], 10),
            // state=50, event=3, arr=[2,2,2,2,2] -> 50 - 3 - 10 = 37
            example(vec![int(50), int(3), array(&[2, 2, 2, 2, 2])], 37),
        ],
        vec![
            example(vec![int(7), int(7), array(&[0, 0, 0])], 0),
            example(vec![int(1), int(0), array(&[1])], 0),
        ],
        "fn damage_with_event(state: i64, event: i64, arr: [i64]) -> i64 {\n    r: i64 = 0;\n    for v in arr { r = r + v; }\n    return state - event - r;\n}\n",
    )
}

// ---------------------------------------------------------------------------
// Stage 1.5 stateful reducer benchmarks
//
// These exercise the two new search teachers:
//   * `search_stateful_reducer_dual` — 3-arg (state, a, b) -> state
//   * `search_stateful_replace`      — 2-arg (state, arr) -> state
//                                      with conditional update
// All are real stateful-update shapes: delta accumulators, running
// max/min, trigger counters, signed balances.
// ---------------------------------------------------------------------------

/// `state = state + sum(a) - sum(b)` — running delta of two streams.
fn make_delta_accumulator(variant: usize) -> Problem {
    problem(
        "delta_accumulator",
        variant,
        "stateful",
        "Running delta: state + sum(a) - sum(b).",
        "fn delta_accumulator(state: i64, a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, 2, 3]), array(&[1, 0, 0])], 5),
            example(vec![int(10), array(&[5, 5]), array(&[2, 3])], 15),
            example(vec![int(-5), array(&[3, 3, 3]), array(&[1, 1, 1])], 1),
            example(vec![int(100), array(&[0, 0, 0]), array(&[10, 20])], 70),
        ],
        vec![
            example(vec![int(7), array(&[1, 1, 1]), array(&[0, 0, 0])], 10),
            example(vec![int(0), array(&[4, 4]), array(&[2, 2])], 4),
        ],
        "fn delta_accumulator(state: i64, a: [i64], b: [i64]) -> i64 {\n    sa: i64 = 0;\n    for v in a { sa = sa + v; }\n    sb: i64 = 0;\n    for v in b { sb = sb + v; }\n    return state + sa - sb;\n}\n",
    )
}

/// `state = state + count_positive(a) - count_negative(b)`.
fn make_signed_count_delta(variant: usize) -> Problem {
    problem(
        "signed_count_delta",
        variant,
        "stateful",
        "Signed-count delta: state + count_positive(a) - count_negative(b).",
        "fn signed_count_delta(state: i64, a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, -2, 3]), array(&[-1, 2])], 1),
            example(vec![int(10), array(&[5, 5, 5]), array(&[-1, -1])], 11),
            example(vec![int(0), array(&[0, 0]), array(&[0, 0])], 0),
            example(vec![int(-5), array(&[1, 2, 3, 4]), array(&[-5])], -2),
        ],
        vec![
            example(vec![int(3), array(&[-1]), array(&[1, 2])], 3),
            example(vec![int(0), array(&[1]), array(&[0])], 1),
        ],
        "fn signed_count_delta(state: i64, a: [i64], b: [i64]) -> i64 {\n    pa: i64 = 0;\n    for v in a { if v > 0 { pa = pa + 1; } }\n    nb: i64 = 0;\n    for v in b { if v < 0 { nb = nb + 1; } }\n    return state + pa - nb;\n}\n",
    )
}

/// `state = state + max(a) - min(b)`.
fn make_cross_range_state(variant: usize) -> Problem {
    problem(
        "cross_range_state",
        variant,
        "stateful",
        "Cross-range state: state + max(a) - min(b).",
        "fn cross_range_state(state: i64, a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[3, 7, 1]), array(&[-5, 10])], 12),
            example(vec![int(10), array(&[5, 5]), array(&[2, 3])], 13),
            example(vec![int(-5), array(&[10, 20]), array(&[1, 1])], 14),
            example(vec![int(100), array(&[0]), array(&[-50])], 150),
        ],
        vec![
            example(vec![int(7), array(&[1, 1, 1]), array(&[0, 0, 0])], 8),
            example(vec![int(0), array(&[4, 4]), array(&[2, 2])], 2),
        ],
        "fn cross_range_state(state: i64, a: [i64], b: [i64]) -> i64 {\n    ma: i64 = a[0];\n    for v in a { if v > ma { ma = v; } }\n    mb: i64 = b[0];\n    for v in b { if v < mb { mb = v; } }\n    return state + ma - mb;\n}\n",
    )
}

/// `state = state + count_positive(a) + count_positive(b)`.
fn make_boost_positive(variant: usize) -> Problem {
    problem(
        "boost_positive",
        variant,
        "stateful",
        "Boost: state + count_positive(a) + count_positive(b).",
        "fn boost_positive(state: i64, a: [i64], b: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, 2, 3]), array(&[-1, -2])], 3),
            example(vec![int(10), array(&[5, 5, 5]), array(&[1, 2])], 15),
            example(vec![int(0), array(&[0, 0]), array(&[0, 0])], 0),
            example(vec![int(-5), array(&[1, 2, 3, 4]), array(&[1])], 0),
        ],
        vec![
            example(vec![int(3), array(&[-1]), array(&[1, 2])], 5),
            example(vec![int(0), array(&[1]), array(&[0])], 1),
        ],
        "fn boost_positive(state: i64, a: [i64], b: [i64]) -> i64 {\n    pa: i64 = 0;\n    for v in a { if v > 0 { pa = pa + 1; } }\n    pb: i64 = 0;\n    for v in b { if v > 0 { pb = pb + 1; } }\n    return state + pa + pb;\n}\n",
    )
}

/// `if max(arr) > state then state = max(arr) else state` — running max.
fn make_running_max(variant: usize) -> Problem {
    problem(
        "running_max",
        variant,
        "stateful",
        "Running max: if max(arr) > state then state = max(arr) else state.",
        "fn running_max(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[3, 7, 1])], 7),
            example(vec![int(10), array(&[5, 5, 5])], 10),
            example(vec![int(100), array(&[1, 2, 3])], 100),
            example(vec![int(-5), array(&[-10, 0, 1])], 1),
        ],
        vec![
            example(vec![int(7), array(&[0, 0, 0])], 7),
            example(vec![int(0), array(&[100])], 100),
        ],
        "fn running_max(state: i64, arr: [i64]) -> i64 {\n    r: i64 = arr[0];\n    for v in arr { if v > r { r = v; } }\n    if r > state { return r; }\n    return state;\n}\n",
    )
}

/// `if min(arr) < state then state = min(arr) else state` — running min.
fn make_running_min(variant: usize) -> Problem {
    problem(
        "running_min",
        variant,
        "stateful",
        "Running min: if min(arr) < state then state = min(arr) else state.",
        "fn running_min(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[3, 7, 1])], 0),
            example(vec![int(10), array(&[5, 5, 5])], 5),
            example(vec![int(100), array(&[200, 300])], 100),
            example(vec![int(-5), array(&[-10, 0, 1])], -10),
        ],
        vec![
            example(vec![int(7), array(&[100, 200])], 7),
            example(vec![int(0), array(&[-100])], -100),
        ],
        "fn running_min(state: i64, arr: [i64]) -> i64 {\n    r: i64 = arr[0];\n    for v in arr { if v < r { r = v; } }\n    if r < state { return r; }\n    return state;\n}\n",
    )
}

/// `if any(arr > 0) then state = -state else state` — flip on positive.
fn make_flip_on_positive(variant: usize) -> Problem {
    problem(
        "flip_on_positive",
        variant,
        "stateful",
        "Flip: if any(arr > 0) then state = -state else state.",
        "fn flip_on_positive(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(1), array(&[3, 7, 1])], -1),
            example(vec![int(5), array(&[-1, -2])], 5),
            example(vec![int(0), array(&[-3])], 0),
            example(vec![int(-7), array(&[0, 0, 1])], 7),
        ],
        vec![
            example(vec![int(2), array(&[0, 0, 0])], 2),
            example(vec![int(-3), array(&[-1])], -3),
        ],
        "fn flip_on_positive(state: i64, arr: [i64]) -> i64 {\n    p: i64 = 0;\n    for v in arr { if v > 0 { p = 1; } }\n    if p == 1 { return -state; }\n    return state;\n}\n",
    )
}

/// `if any(arr > 0) then state = state + 1 else state` — trigger counter.
fn make_increment_on_positive(variant: usize) -> Problem {
    problem(
        "increment_on_positive",
        variant,
        "stateful",
        "Trigger counter: if any(arr > 0) then state + 1 else state.",
        "fn increment_on_positive(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[3, 7, 1])], 1),
            example(vec![int(5), array(&[-1, -2])], 5),
            example(vec![int(0), array(&[-3])], 0),
            example(vec![int(-7), array(&[0, 0, 1])], -6),
        ],
        vec![
            example(vec![int(2), array(&[0, 0, 0])], 2),
            example(vec![int(-3), array(&[-1, 0])], -3),
        ],
        "fn increment_on_positive(state: i64, arr: [i64]) -> i64 {\n    p: i64 = 0;\n    for v in arr { if v > 0 { p = 1; } }\n    if p == 1 { return state + 1; }\n    return state;\n}\n",
    )
}

/// `if any(arr < 0) then state = 0 else state` — reset on negative.
fn make_reset_on_negative(variant: usize) -> Problem {
    problem(
        "reset_on_negative",
        variant,
        "stateful",
        "Reset: if any(arr < 0) then state = 0 else state.",
        "fn reset_on_negative(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(5), array(&[-1, 0, 0])], 0),
            example(vec![int(3), array(&[1, 2])], 3),
            example(vec![int(0), array(&[-3, -4])], 0),
            example(vec![int(-7), array(&[0, 1, 2])], -7),
        ],
        vec![
            example(vec![int(2), array(&[0, 0, 0])], 2),
            example(vec![int(-3), array(&[-1])], 0),
        ],
        "fn reset_on_negative(state: i64, arr: [i64]) -> i64 {\n    p: i64 = 0;\n    for v in arr { if v < 0 { p = 1; } }\n    if p == 1 { return 0; }\n    return state;\n}\n",
    )
}

/// `state = state - sum(arr)` — running loss (extending the existing
/// `search_stateful_reducer` teacher to a new reducer × op pair).
fn make_loss_accumulator(variant: usize) -> Problem {
    problem(
        "loss_accumulator",
        variant,
        "stateful",
        "Running loss: state - sum(arr).",
        "fn loss_accumulator(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, 2, 3])], -6),
            example(vec![int(10), array(&[1, 1, 1])], 7),
            example(vec![int(100), array(&[50, 30])], 20),
            example(vec![int(0), array(&[0, 0, 0])], 0),
        ],
        vec![
            example(vec![int(7), array(&[1, 2])], 4),
            example(vec![int(0), array(&[-1, 2])], -1),
        ],
        "fn loss_accumulator(state: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { s = s + v; }\n    return state - s;\n}\n",
    )
}

/// `state = state + count_positive(arr)` — running positive count.
fn make_inventory_total(variant: usize) -> Problem {
    problem(
        "inventory_total",
        variant,
        "stateful",
        "Running positive count: state + count_positive(arr).",
        "fn inventory_total(state: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), array(&[1, -2, 3])], 2),
            example(vec![int(5), array(&[1, 1, 1])], 8),
            example(vec![int(0), array(&[-1, -2])], 0),
            example(vec![int(-3), array(&[1, 2, 3, 4])], 1),
        ],
        vec![
            example(vec![int(3), array(&[0, 0, 0])], 3),
            example(vec![int(0), array(&[1])], 1),
        ],
        "fn inventory_total(state: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { if v > 0 { s = s + 1; } }\n    return state + s;\n}\n",
    )
}

// ---------------------------------------------------------------------------
// Stage 4: time-lane stateful benchmarks
//
// The 3-arg `(state, t, arr) -> state` signature adds a synthetic
// time/index argument `t: i64` to the stateful reducer. The new
// `search_stateful_reducer_temporal` teacher enumerates patterns
// involving `t` (linear aging, periodic ticks, rate × time, decay).
// ---------------------------------------------------------------------------

/// `state + t` — pure time-driven aging.
fn make_aging_state(variant: usize) -> Problem {
    problem(
        "aging_state",
        variant,
        "stateful",
        "Aging: state + t.",
        "fn aging_state(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(5), array(&[1, 2, 3])], 5),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 13),
            example(vec![int(-5), int(7), array(&[4, 4])], 2),
        ],
        vec![
            example(vec![int(100), int(0), array(&[1, 1])], 100),
            example(vec![int(0), int(1), array(&[10])], 1),
        ],
        "fn aging_state(state: i64, t: i64, arr: [i64]) -> i64 {\n    return state + t;\n}\n",
    )
}

/// `state - t` — time-driven decay.
fn make_time_decay(variant: usize) -> Problem {
    problem(
        "time_decay",
        variant,
        "stateful",
        "Time decay: state - t.",
        "fn time_decay(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(5), array(&[1, 2, 3])], -5),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 7),
            example(vec![int(-5), int(7), array(&[4, 4])], -12),
        ],
        vec![
            example(vec![int(100), int(0), array(&[1, 1])], 100),
            example(vec![int(0), int(1), array(&[10])], -1),
        ],
        "fn time_decay(state: i64, t: i64, arr: [i64]) -> i64 {\n    return state - t;\n}\n",
    )
}

/// `state + sum(arr) * t` — rate × time integrator.
fn make_rate_accumulator(variant: usize) -> Problem {
    problem(
        "rate_accumulator",
        variant,
        "stateful",
        "Rate accumulator: state + sum(arr) * t.",
        "fn rate_accumulator(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(2), array(&[1, 2, 3])], 12),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 19),
            example(vec![int(-5), int(4), array(&[4, 4])], 27),
        ],
        vec![
            example(vec![int(0), int(0), array(&[10, 20])], 0),
            example(vec![int(100), int(1), array(&[5])], 105),
        ],
        "fn rate_accumulator(state: i64, t: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { s = s + v; }\n    return state + s * t;\n}\n",
    )
}

/// `state + arr[0] * t` — first-element × time.
fn make_first_rate(variant: usize) -> Problem {
    problem(
        "first_rate",
        variant,
        "stateful",
        "First-rate: state + arr[0] * t.",
        "fn first_rate(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(5), array(&[1, 2, 3])], 5),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 13),
            example(vec![int(-5), int(7), array(&[4, 4])], 23),
        ],
        vec![
            example(vec![int(0), int(0), array(&[10, 20])], 0),
            example(vec![int(100), int(1), array(&[5])], 105),
        ],
        "fn first_rate(state: i64, t: i64, arr: [i64]) -> i64 {\n    return state + arr[0] * t;\n}\n",
    )
}

/// `state + count_positive(arr) * t` — positive count × time.
fn make_count_rate(variant: usize) -> Problem {
    problem(
        "count_rate",
        variant,
        "stateful",
        "Count rate: state + count_positive(arr) * t.",
        "fn count_rate(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(2), array(&[1, -2, 3])], 4),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 19),
            example(vec![int(-5), int(4), array(&[-1, -2])], -5),
        ],
        vec![
            example(vec![int(0), int(0), array(&[1, 2])], 0),
            example(vec![int(100), int(1), array(&[5])], 101),
        ],
        "fn count_rate(state: i64, t: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { if v > 0 { s = s + 1; } }\n    return state + s * t;\n}\n",
    )
}

/// `state + max(arr) * t` — peak × time.
fn make_max_rate(variant: usize) -> Problem {
    problem(
        "max_rate",
        variant,
        "stateful",
        "Max rate: state + max(arr) * t.",
        "fn max_rate(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(2), array(&[1, 2, 3])], 6),
            example(vec![int(10), int(3), array(&[1, 1, 1])], 13),
            example(vec![int(-5), int(4), array(&[4, 4])], 11),
        ],
        vec![
            example(vec![int(0), int(0), array(&[10, 20])], 0),
            example(vec![int(100), int(1), array(&[5])], 105),
        ],
        "fn max_rate(state: i64, t: i64, arr: [i64]) -> i64 {\n    r: i64 = arr[0];\n    for v in arr { if v > r { r = v; } }\n    return state + r * t;\n}\n",
    )
}

/// `state + sum(arr) * (t % 2 == 0 ? 1 : 0)` — fires every 2nd tick.
fn make_tick_every_2(variant: usize) -> Problem {
    problem(
        "tick_every_2",
        variant,
        "stateful",
        "Periodic tick: state + sum(arr) on even t only.",
        "fn tick_every_2(state: i64, t: i64, arr: [i64]) -> i64",
        vec![
            example(vec![int(0), int(2), array(&[1, 2, 3])], 6),
            example(vec![int(10), int(4), array(&[1, 1, 1])], 13),
            example(vec![int(-5), int(3), array(&[4, 4])], -5),
        ],
        vec![
            example(vec![int(0), int(0), array(&[1, 1])], 2),
            example(vec![int(100), int(6), array(&[1])], 101),
        ],
        "fn tick_every_2(state: i64, t: i64, arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for v in arr { s = s + v; }\n    tval: i64 = 0;\n    if t % 2 == 0 { tval = 1; } else { tval = 0; }\n    return state + s * tval;\n}\n",
    )
}

/// Stage 3: struct-of-state benchmarks (June 2026)
/// Count positive and negative values separately.
/// struct DualTally { pos_count: i64, neg_count: i64, zero_count: i64, total: i64 }
fn make_dual_tally(variant: usize) -> Problem {
    problem(
        "dual_tally",
        variant,
        "struct_of_state",
        "Count positive, negative, and zero values: {pos_count, neg_count, zero_count, total}.",
        "fn dual_tally(arr: [i64]) -> DualTally",
        vec![
            example_quad(vec![array(&[1, -2, 3])], quad(2, 1, 0, 3)),
            example_quad(vec![array(&[0, 0, 5, -3])], quad(1, 1, 2, 4)),
            example_quad(vec![array(&[-1, -1, -1])], quad(0, 3, 0, 3)),
        ],
        vec![
            example_quad(vec![array(&[2, 0, -1, 3, 0])], quad(2, 1, 2, 5)),
            example_quad(vec![array(&[10])], quad(1, 0, 0, 1)),
        ],
        "struct DualTally {\n    pos_count: i64,\n    neg_count: i64,\n    zero_count: i64,\n    total: i64,\n}\n\nfn dual_tally(arr: [i64]) -> DualTally {\n    pos: i64 = 0;\n    neg: i64 = 0;\n    zero: i64 = 0;\n    for v in arr {\n        if v > 0 { pos = pos + 1; }\n        else if v < 0 { neg = neg + 1; }\n        else { zero = zero + 1; }\n    }\n    return DualTally { pos_count: pos, neg_count: neg, zero_count: zero, total: arr.len };\n}\n",
    )
}

/// Rate limiter tracking: cumulative sum and breach count.
/// struct RateLimiter { total: i64, exceeded: i64, count: i64, limit_reached: i64 }
fn make_rate_limiter(variant: usize) -> Problem {
    problem(
        "rate_limiter",
        variant,
        "struct_of_state",
        "Track cumulative sum and threshold breaches: {total, exceeded, count, limit_reached}.",
        "fn rate_limiter(arr: [i64], limit: i64) -> RateLimiter",
        vec![
            example_quad(vec![array(&[1, 2, 3]), int(5)], quad(6, 1, 3, 1)),
            example_quad(vec![array(&[1, 1, 1]), int(2)], quad(3, 1, 3, 1)),
            example_quad(vec![array(&[1, 2]), int(10)], quad(3, 0, 2, 0)),
        ],
        vec![
            example_quad(vec![array(&[2, 2, 2]), int(5)], quad(6, 1, 3, 1)),
            example_quad(vec![array(&[1, 1, 1, 1]), int(3)], quad(4, 1, 4, 1)),
        ],
        "struct RateLimiter {\n    total: i64,\n    exceeded: i64,\n    count: i64,\n    limit_reached: i64,\n}\n\nfn rate_limiter(arr: [i64], limit: i64) -> RateLimiter {\n    total: i64 = 0;\n    exceeded: i64 = 0;\n    limit_hit: i64 = 0;\n    for v in arr {\n        total = total + v;\n        if total > limit { exceeded = exceeded + 1; limit_hit = 1; }\n    }\n    return RateLimiter { total: total, exceeded: exceeded, count: arr.len, limit_reached: limit_hit };\n}\n",
    )
}

/// Running correlation accumulators: sum_x, sum_y, sum_xy.
/// struct RunningCorrelation { sum_x: i64, sum_y: i64, sum_xy: i64, count: i64 }
fn make_running_correlation(variant: usize) -> Problem {
    problem(
        "running_correlation",
        variant,
        "struct_of_state",
        "Compute correlation accumulators: {sum_x, sum_y, sum_xy, count}.",
        "fn running_correlation(pairs: [(i64, i64)]) -> RunningCorrelation",
        vec![
            example_quad(vec![array(&[1, 2, 2, 3])], quad(3, 5, 8, 2)),
            example_quad(vec![array(&[0, 0, 1, 1])], quad(1, 1, 1, 2)),
            example_quad(vec![array(&[2, 2])], quad(2, 2, 4, 1)),
        ],
        vec![
            example_quad(vec![array(&[1, 1, 3, 3])], quad(4, 4, 12, 2)),
            example_quad(vec![array(&[5, 10])], quad(5, 10, 50, 1)),
        ],
        "struct RunningCorrelation {\n    sum_x: i64,\n    sum_y: i64,\n    sum_xy: i64,\n    count: i64,\n}\n\nfn running_correlation(pairs: [(i64, i64)]) -> RunningCorrelation {\n    sx: i64 = 0; sy: i64 = 0; sxy: i64 = 0;\n    for i in 0..pairs.len {\n        x: i64 = pairs[i].0; y: i64 = pairs[i].1;\n        sx = sx + x; sy = sy + y; sxy = sxy + (x * y);\n    }\n    return RunningCorrelation { sum_x: sx, sum_y: sy, sum_xy: sxy, count: pairs.len };\n}\n",
    )
}

/// Mutual information: 2×2 confusion matrix.
/// struct MutualInfo { joint_00: i64, joint_01: i64, joint_10: i64, joint_11: i64 }
fn make_mutual_info_tracker(variant: usize) -> Problem {
    problem(
        "mutual_info_tracker",
        variant,
        "struct_of_state",
        "Build confusion matrix from paired bits: {joint_00, joint_01, joint_10, joint_11}.",
        "fn mutual_info_tracker(pairs: [(i64, i64)]) -> MutualInfo",
        vec![
            example_quad(vec![array(&[0, 0, 0, 1, 1, 1])], quad(1, 1, 1, 1)),
            example_quad(vec![array(&[0, 0, 1, 1])], quad(1, 1, 1, 0)),
            example_quad(vec![array(&[0, 0, 0, 0])], quad(2, 0, 0, 0)),
        ],
        vec![
            example_quad(vec![array(&[1, 1, 1, 1])], quad(0, 0, 0, 2)),
            example_quad(vec![array(&[0, 1, 1, 0])], quad(1, 0, 1, 1)),
        ],
        "struct MutualInfo {\n    joint_00: i64,\n    joint_01: i64,\n    joint_10: i64,\n    joint_11: i64,\n}\n\nfn mutual_info_tracker(pairs: [(i64, i64)]) -> MutualInfo {\n    j00: i64 = 0; j01: i64 = 0; j10: i64 = 0; j11: i64 = 0;\n    for i in 0..pairs.len {\n        a: i64 = if pairs[i].0 > 0 { 1 } else { 0 };\n        b: i64 = if pairs[i].1 > 0 { 1 } else { 0 };\n        if a == 0 && b == 0 { j00 = j00 + 1; }\n        else if a == 0 && b == 1 { j01 = j01 + 1; }\n        else if a == 1 && b == 0 { j10 = j10 + 1; }\n        else { j11 = j11 + 1; }\n    }\n    return MutualInfo { joint_00: j00, joint_01: j01, joint_10: j10, joint_11: j11 };\n}\n",
    )
}

/// Dual-threshold classifier: count values below, between, and above thresholds.
/// struct ThresholdClassifier { below: i64, between: i64, above: i64, total: i64 }
fn make_dual_threshold_classifier(variant: usize) -> Problem {
    problem(
        "dual_threshold_classifier",
        variant,
        "struct_of_state",
        "Classify by two thresholds: {below, between, above, total}.",
        "fn dual_threshold_classifier(arr: [i64], low: i64, high: i64) -> ThresholdClassifier",
        vec![
            example_quad(vec![array(&[1, 5, 10]), int(3), int(8)], quad(1, 2, 0, 3)),
            example_quad(vec![array(&[0, 5, 15]), int(5), int(10)], quad(1, 1, 1, 3)),
            example_quad(vec![array(&[1, 2, 3, 4, 5]), int(2), int(4)], quad(1, 3, 1, 5)),
        ],
        vec![
            example_quad(vec![array(&[1, 3, 6, 9]), int(3), int(7)], quad(2, 1, 1, 4)),
            example_quad(vec![array(&[10, 20]), int(5), int(15)], quad(0, 1, 1, 2)),
        ],
        "struct ThresholdClassifier {\n    below: i64,\n    between: i64,\n    above: i64,\n    total: i64,\n}\n\nfn dual_threshold_classifier(arr: [i64], low: i64, high: i64) -> ThresholdClassifier {\n    below: i64 = 0; between: i64 = 0; above: i64 = 0;\n    for v in arr {\n        if v < low { below = below + 1; }\n        else if v <= high { between = between + 1; }\n        else { above = above + 1; }\n    }\n    return ThresholdClassifier { below: below, between: between, above: above, total: arr.len };\n}\n",
    )
}

/// Paired extrema: min, min_idx, max, max_idx.
/// struct PairedExtrema { min: i64, min_idx: i64, max: i64, max_idx: i64 }
fn make_paired_extrema(variant: usize) -> Problem {
    problem(
        "paired_extrema",
        variant,
        "struct_of_state",
        "Find min/max and their indices: {min, min_idx, max, max_idx}.",
        "fn paired_extrema(arr: [i64]) -> PairedExtrema",
        vec![
            example_quad(vec![array(&[5, 2, 8, 1])], quad(1, 3, 8, 2)),
            example_quad(vec![array(&[3, 3, 3])], quad(3, 0, 3, 0)),
            example_quad(vec![array(&[10, 5])], quad(5, 1, 10, 0)),
        ],
        vec![
            example_quad(vec![array(&[-1, 0, 2, -3, 4])], quad(-3, 3, 4, 4)),
            example_quad(vec![array(&[7])], quad(7, 0, 7, 0)),
        ],
        "struct PairedExtrema {\n    min: i64,\n    min_idx: i64,\n    max: i64,\n    max_idx: i64,\n}\n\nfn paired_extrema(arr: [i64]) -> PairedExtrema {\n    min_val: i64 = arr[0]; max_val: i64 = arr[0];\n    min_i: i64 = 0; max_i: i64 = 0;\n    for i in 0..arr.len {\n        if arr[i] < min_val { min_val = arr[i]; min_i = i; }\n        if arr[i] > max_val { max_val = arr[i]; max_i = i; }\n    }\n    return PairedExtrema { min: min_val, min_idx: min_i, max: max_val, max_idx: max_i };\n}\n",
    )
}

/// Sum of diagonal elements of a 4×4 matrix (flattened as 16-element array).
/// Input: matrix as [i64; 16] (row-major: [row0[0..4], row1[0..4], row2[0..4], row3[0..4]]).
/// Output: sum of elements [0], [5], [10], [15] (the diagonal).
fn make_matrix_diagonal_sum(variant: usize) -> Problem {
    problem(
        "matrix_diagonal_sum",
        variant,
        "tensor",
        "Sum the diagonal elements of a 4x4 matrix (row-major order).",
        "fn matrix_diagonal_sum(matrix: [i64; 16]) -> i64",
        vec![
            example(vec![array(&[1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4])], 10),
            example(vec![array(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])], 34),
            example(vec![array(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])], 0),
            example(vec![array(&[5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 5, 12, 13, 14, 5])], 20),
        ],
        vec![
            example(vec![array(&[2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5])], 14),
            example(vec![array(&[10, 20, 30, 40, 50, 10, 60, 70, 80, 90, 10, 100, 110, 120, 10, 130])], 160),
        ],
        "fn matrix_diagonal_sum(matrix: [i64; 16]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 4 {\n        idx: i64 = i * 5;\n        sum = sum + matrix[idx];\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

/// Dot product of two 4-element vectors.
/// Input: two arrays of 4 elements each.
/// Output: sum of element-wise products.
fn make_dot_product_4d(variant: usize) -> Problem {
    problem(
        "dot_product_4d",
        variant,
        "tensor",
        "Compute dot product of two 4-element vectors.",
        "fn dot_product_4d(a: [i64; 4], b: [i64; 4]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4]), array(&[5, 6, 7, 8])], 70),
            example(vec![array(&[2, 0, 0, 3]), array(&[1, 1, 1, 1])], 5),
            example(vec![array(&[1, 1, 1, 1]), array(&[1, 1, 1, 1])], 4),
            example(vec![array(&[0, 0, 0, 0]), array(&[5, 5, 5, 5])], 0),
        ],
        vec![
            example(vec![array(&[3, 4, 0, 0]), array(&[1, 0, 2, 3])], 3),
            example(vec![array(&[10, 20, 30, 40]), array(&[1, 1, 1, 1])], 100),
        ],
        "fn dot_product_4d(a: [i64; 4], b: [i64; 4]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 4 {\n        sum = sum + a[i] * b[i];\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

/// Matrix multiply two 2x2 matrices (trace-like reduction).
/// For simplicity, compute: sum of all products M1[i][j] * M2[j][i] for all i,j.
/// Input: two 4-element arrays (row-major 2×2 matrices).
/// Output: scalar result.
fn make_matrix_multiply_2x2(variant: usize) -> Problem {
    problem(
        "matrix_multiply_2x2",
        variant,
        "tensor",
        "Multiply two 2x2 matrices (return trace-like scalar).",
        "fn matrix_multiply_2x2(m1: [i64; 4], m2: [i64; 4]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4]), array(&[5, 6, 7, 8])], 69),
            example(vec![array(&[1, 0, 0, 1]), array(&[2, 3, 4, 5])], 7),
            example(vec![array(&[0, 0, 0, 0]), array(&[1, 2, 3, 4])], 0),
            example(vec![array(&[2, 1, 1, 2]), array(&[1, 1, 1, 1])], 8),
        ],
        vec![
            example(vec![array(&[1, 1, 1, 1]), array(&[1, 1, 1, 1])], 4),
            example(vec![array(&[3, 0, 0, 3]), array(&[2, 1, 1, 2])], 12),
        ],
        "fn matrix_multiply_2x2(m1: [i64; 4], m2: [i64; 4]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 2 {\n        j: i64 = 0;\n        while j < 2 {\n            idx1: i64 = i * 2 + j;\n            idx2: i64 = j * 2 + i;\n            sum = sum + m1[idx1] * m2[idx2];\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

/// Broadcast a scalar to a 4-element vector and multiply element-wise, returning the sum.
/// Input: scalar s and array a of 4 elements.
/// Output: sum of s * a[i] for all i (equivalent to s * sum(a)).
fn make_broadcast_scale_sum(variant: usize) -> Problem {
    problem(
        "broadcast_scale_sum",
        variant,
        "tensor",
        "Broadcast a scalar to a 4-element vector, multiply element-wise, and sum.",
        "fn broadcast_scale_sum(scalar: i64, vec: [i64; 4]) -> i64",
        vec![
            example(vec![int(2), array(&[1, 2, 3, 4])], 20),
            example(vec![int(3), array(&[1, 0, 1, 0])], 6),
            example(vec![int(0), array(&[5, 5, 5, 5])], 0),
            example(vec![int(1), array(&[1, 2, 3, 4])], 10),
        ],
        vec![
            example(vec![int(5), array(&[1, 1, 1, 1])], 20),
            example(vec![int(10), array(&[2, 0, 1, 3])], 60),
        ],
        "fn broadcast_scale_sum(scalar: i64, vec: [i64; 4]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 4 {\n        sum = sum + scalar * vec[i];\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

/// Compute the Frobenius norm squared of the outer product of two 2-element vectors.
/// Outer product: a ⊗ b = [[a[0]*b[0], a[0]*b[1]], [a[1]*b[0], a[1]*b[1]]] (4 elements).
/// Return: sum of squared elements (pre-sqrt for integer output).
fn make_outer_product_norm_sq(variant: usize) -> Problem {
    problem(
        "outer_product_norm_sq",
        variant,
        "tensor",
        "Compute sum of squared elements of the outer product of two 2-element vectors.",
        "fn outer_product_norm_sq(a: [i64; 2], b: [i64; 2]) -> i64",
        vec![
            example(vec![array(&[2, 3]), array(&[4, 5])], 533),
            example(vec![array(&[1, 1]), array(&[1, 1])], 4),
            example(vec![array(&[0, 2]), array(&[3, 0])], 36),
            example(vec![array(&[1, 2]), array(&[1, 2])], 25),
        ],
        vec![
            example(vec![array(&[3, 0]), array(&[2, 0])], 36),
            example(vec![array(&[1, 0]), array(&[0, 1])], 1),
        ],
        "fn outer_product_norm_sq(a: [i64; 2], b: [i64; 2]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 2 {\n        j: i64 = 0;\n        while j < 2 {\n            prod: i64 = a[i] * b[j];\n            sum = sum + prod * prod;\n            j = j + 1;\n        }\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

/// Convolve a 1D signal with a 1D filter (valid convolution, no padding).
/// Input: signal [i64; 8], filter [i64; 3].
/// Output: sum of all convolution outputs.
/// Valid convolution has 8 - 3 + 1 = 6 outputs.
fn make_convolution_1d_sum(variant: usize) -> Problem {
    problem(
        "convolution_1d_sum",
        variant,
        "tensor",
        "Compute valid 1D convolution of signal [i64; 8] with filter [i64; 3], return sum of outputs.",
        "fn convolution_1d_sum(signal: [i64; 8], filter: [i64; 3]) -> i64",
        vec![
            example(vec![array(&[1, 2, 3, 4, 5, 6, 7, 8]), array(&[1, 0, -1])], -12),
            example(vec![array(&[1, 1, 1, 1, 1, 1, 1, 1]), array(&[1, 1, 1])], 18),
            example(vec![array(&[0, 0, 0, 0, 0, 0, 0, 0]), array(&[1, 1, 1])], 0),
            example(vec![array(&[2, 0, 2, 0, 2, 0, 2, 0]), array(&[1, 0, 1])], 12),
        ],
        vec![
            example(vec![array(&[1, 2, 1, 2, 1, 2, 1, 2]), array(&[1, 1, 1])], 24),
            example(vec![array(&[5, 5, 5, 5, 5, 5, 5, 5]), array(&[2, 0, 1])], 84),
        ],
        "fn convolution_1d_sum(signal: [i64; 8], filter: [i64; 3]) -> i64 {\n    sum: i64 = 0;\n    i: i64 = 0;\n    while i < 6 {\n        out: i64 = 0;\n        j: i64 = 0;\n        while j < 3 {\n            out = out + signal[i + j] * filter[j];\n            j = j + 1;\n        }\n        sum = sum + out;\n        i = i + 1;\n    }\n    return sum;\n}\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn problem_from_reference_double_synthesizes_and_verifies() {
        // A spec given ONLY a reference implementation (no hand examples).
        let problem = problem_from_reference(
            "double",
            "fn double(x: i64) -> i64",
            "fn double(x: i64) -> i64 {\n    return x * 2;\n}\n",
        )
        .expect("reference intake should build a problem");

        // Seed examples were MANUFACTURED by running the reference, not hand
        // authored: every expected == 2 * input proves they came from execution.
        assert!(
            !problem.examples.is_empty(),
            "seed examples should be generated by running the reference"
        );
        for example in &problem.examples {
            let Value::Int(input) = example.inputs[0] else {
                panic!("expected an int input, got {:?}", example.inputs[0]);
            };
            assert_eq!(
                example.expected,
                Value::Int(input * 2),
                "seed example must equal the reference's output for {input}"
            );
        }

        // The reference oracle is retained so generated_holdouts does
        // differential testing in the strict verifier.
        assert!(
            !problem.reference_code.is_empty(),
            "reference_code must stay set as the verifier-owned oracle"
        );

        // An equivalent candidate (x + x) must pass solve+verify — exercising
        // the unchanged verify_problem_code_strict pipeline AND the differential
        // holdouts against the reference.
        let equivalent = "fn double(x: i64) -> i64 { return x + x; }";
        crate::runtime::verify_problem_code_strict(&problem, equivalent)
            .expect("an equivalent candidate must verify against the reference");

        // A non-equivalent candidate (x + 1) must be REJECTED — proving the
        // differential oracle actually fires rather than rubber-stamping.
        let wrong = "fn double(x: i64) -> i64 { return x + 1; }";
        assert!(
            crate::runtime::verify_problem_code_strict(&problem, wrong).is_err(),
            "a non-equivalent candidate must be rejected by differential testing"
        );
    }

    #[test]
    fn problem_from_reference_rejects_unsampleable_signature() {
        // Zero-parameter signatures cannot be sampled → Err, not a fake spec.
        assert!(problem_from_reference(
            "noop",
            "fn noop() -> i64",
            "fn noop() -> i64 { return 0; }",
        )
        .is_err());
    }

    #[test]
    fn property_scalar_predicate_accepts_only_satisfying_candidates() {
        // A PROPERTY-ONLY spec: no expected outputs, just a predicate the result
        // must satisfy — here "the output is strictly greater than the input".
        let predicate_sig = "fn gt(x: i64, out: i64) -> i64";
        let predicate = "fn gt(x: i64, out: i64) -> i64 { if out > x { return 1; } return 0; }";

        // A candidate that satisfies the property (x + 1 > x) verifies.
        verify_code_against_property(
            "inc",
            "fn inc(x: i64) -> i64",
            "fn inc(x: i64) -> i64 { return x + 1; }",
            "gt",
            predicate_sig,
            predicate,
        )
        .expect("a candidate satisfying the property must verify");

        // A candidate that violates it (x - 1 < x) is rejected — proves the
        // predicate oracle actually fires.
        assert!(
            verify_code_against_property(
                "inc",
                "fn inc(x: i64) -> i64",
                "fn inc(x: i64) -> i64 { return x - 1; }",
                "gt",
                predicate_sig,
                predicate,
            )
            .is_err(),
            "a candidate violating the property must be rejected"
        );
    }

    #[test]
    fn property_is_sorted_accepts_sort_rejects_identity() {
        // The roadmap's named property-only spec: synthesize a function whose
        // output `is_sorted`, with the predicate (not examples) as the oracle.
        let predicate_sig = "fn is_sorted_prop(arr: [i64], out: [i64]) -> i64";
        let predicate = "fn is_sorted_prop(arr: [i64], out: [i64]) -> i64 {\n    i: i64 = 1;\n    while i < out.len {\n        if out[i] < out[i - 1] { return 0; }\n        i = i + 1;\n    }\n    return 1;\n}\n";

        // A real sort satisfies `is_sorted` on every sampled input.
        verify_code_against_property(
            "sortf",
            "fn sortf(arr: [i64]) -> [i64]",
            "fn sortf(arr: [i64]) -> [i64] { s: [i64] = arr; s.sort(); return s; }",
            "is_sorted_prop",
            predicate_sig,
            predicate,
        )
        .expect("a real sort must satisfy the is_sorted property");

        // Identity leaves unsorted inputs unsorted → rejected on the first
        // unsorted sample.
        assert!(
            verify_code_against_property(
                "sortf",
                "fn sortf(arr: [i64]) -> [i64]",
                "fn sortf(arr: [i64]) -> [i64] { return arr; }",
                "is_sorted_prop",
                predicate_sig,
                predicate,
            )
            .is_err(),
            "identity must be rejected by the is_sorted property"
        );
    }

    #[test]
    fn test_tree_node_creation() {
        let node = TreeNode {
            value: 42,
            left: 0,
            right: 1,
        };
        assert_eq!(node.value, 42);
        assert_eq!(node.left, 0);
        assert_eq!(node.right, 1);
    }

    #[test]
    fn test_value_tree_creation() {
        let nodes = vec![
            TreeNode {
                value: 1,
                left: 1,
                right: 2,
            },
            TreeNode {
                value: 2,
                left: -1,
                right: -1,
            },
            TreeNode {
                value: 3,
                left: -1,
                right: -1,
            },
        ];
        let tree_val = Value::Tree(nodes);
        match tree_val {
            Value::Tree(nodes) => {
                assert_eq!(nodes.len(), 3);
                assert_eq!(nodes[0].value, 1);
                assert_eq!(nodes[1].value, 2);
                assert_eq!(nodes[2].value, 3);
            }
            _ => panic!("Expected Value::Tree"),
        }
    }

    #[test]
    fn test_tree_helper_functions() {
        let edges = vec![(10, 1, 2), (20, -1, -1), (30, -1, -1)];
        let tree_val = tree_from_edges(edges);
        assert!(matches!(tree_val, Value::Tree(_)));

        if let Some(nodes) = get_tree_root(&tree_val) {
            assert_eq!(tree_size(nodes), 3);
            assert_eq!(nodes[0].value, 10);
        } else {
            panic!("Expected to get tree root");
        }
    }

    #[test]
    fn test_tree_comparison() {
        let nodes1 = vec![TreeNode {
            value: 5,
            left: -1,
            right: -1,
        }];
        let nodes2 = vec![TreeNode {
            value: 5,
            left: -1,
            right: -1,
        }];
        let tree1 = Value::Tree(nodes1);
        let tree2 = Value::Tree(nodes2);
        assert_eq!(tree1, tree2);
    }

    #[test]
    fn test_tree_display() {
        let nodes = vec![
            TreeNode {
                value: 1,
                left: 1,
                right: 2,
            },
            TreeNode {
                value: 2,
                left: -1,
                right: -1,
            },
        ];
        let tree_val = Value::Tree(nodes);
        // Test Display implementation (uses std::fmt)
        let display_str = format!("{}", tree_val);
        assert!(display_str.contains("Tree"));
        assert!(display_str.contains("1"));
        assert!(display_str.contains("2"));

        // render_value is not yet implemented for trees
        let dummy_problem = Problem {
            name: "test_tree".to_string(),
            category: "trees",
            description: "Test tree rendering",
            signature: "fn test_tree(t: Tree) -> i64",
            examples: vec![],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: vec![],
            synthetic_values: vec![],
            recursive_allowed: false,
            tree_input: true,
            explicit_stack: false,
            functions: vec![],
        };
        let rendered = render_value(&dummy_problem, &tree_val);
        assert!(rendered.is_err());
    }
}
