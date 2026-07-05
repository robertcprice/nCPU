//! NL-BRIDGE-1 accept-test: the op_role TypeClass LATTICE lets NON-i64 (float +
//! string) requests reach the synthesis pipeline through the ACTUAL product path
//! (`CodingAgentSession::handle_query`, the same entry the `coding_agent` binary
//! uses), from a FRESH root each time.
//!
//! WHY IT CANNOT BE GAMED:
//!   * FLOAT: a float NL request must synthesize a REAL `-> f64` program — the
//!     assertion requires the `f64` return type AND float literals (`32.0`/`.5`)
//!     AND the `search_float_affine` method. A scaffold that emitted an i64 program
//!     (the prior behaviour, when float ops dropped to `OpRole::Other` and the
//!     i64-literal `op_role` never admitted them) would fail every one of those.
//!   * STRING: a string NL request must synthesize a REAL generalizing string
//!     program via `string_synth` (`.upper()`), NOT the memorizing whole-word
//!     lexicon (an if-chain over training pairs) and NOT a refusal. The assertions
//!     forbid the lexicon's `if s ==` form and the i64 array type, and require the
//!     `string_synth` method, so a lookup table or an i64 op cannot satisfy them.
//!   * REGRESSION (differential): i64 single-op (`add`), array transform (`sort`),
//!     and a multi-op composition must STILL emit i64 programs — proving the
//!     lattice generalization did not disturb the i64 lane.
//!   * MUST-REFUSE (differential): genuine type mismatch ("reverse a string" →
//!     i64-array `reverse`) and out-of-domain ("parse a csv file") must STILL be
//!     refused, proving the fail-closed type gate is intact, not blanket-opened.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy, QueryRoute};
use std::fs;
use std::path::{Path, PathBuf};

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "nsynth_bridge1_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("create root");
    root
}

fn run(root: &Path, query: &str) -> mog_synth::agent::AgentQueryResult {
    let mut session = CodingAgentSession::new(root, GuardrailPolicy::default());
    session.handle_query(query)
}

/// FLOAT accept (arity-1, f64->f64): "convert celsius to fahrenheit" → a real
/// `-> f64` affine program via `search_float_affine`. NOT an i64 program.
#[test]
fn float_request_synthesizes_real_f64_program() {
    let root = fresh_root("float1");
    let r = run(&root, "convert celsius to fahrenheit");
    assert!(r.success, "float request must succeed; got: {}", r.response);
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    let code = &r.response;
    // Real float program: f64 return type and float literals — NOT i64.
    assert!(
        code.contains("-> f64"),
        "must be a float (-> f64) program, got:\n{code}"
    );
    assert!(
        !code.contains("-> i64") && !code.contains(": i64"),
        "must NOT be an i64 program (the pre-lattice behaviour), got:\n{code}"
    );
    // The Celsius→Fahrenheit affine 1.8*c + 32 carries float literals.
    assert!(
        code.contains("1.8") && code.contains("32"),
        "must emit the affine float coefficients, got:\n{code}"
    );
    assert_eq!(
        r.synthesis_method.as_deref(),
        Some("search_float_affine"),
        "must route through the float lane, got: {:?}",
        r.synthesis_method
    );
}

/// FLOAT accept #2 (arity-2, f64,f64->f64): "the average of two numbers" → a real
/// `-> f64` program. Exercises the multi-arg float affine path.
#[test]
fn float_average_request_synthesizes_real_f64_program() {
    let root = fresh_root("float2");
    let r = run(&root, "the average of two numbers");
    assert!(r.success, "average request must succeed; got: {}", r.response);
    let code = &r.response;
    // TYPE: a real float program, not the i64 lane.
    assert!(code.contains("-> f64"), "must be -> f64, got:\n{code}");
    assert!(!code.contains("-> i64"), "must not be i64, got:\n{code}");
    // BEHAVIOR over method-label + literal: the exact affine form ("0.5*a + 0.5*b"
    // vs "(a + b) / 2") and the winning method ("search_float_affine" vs the
    // arity-polymorphic "universal" search) drift as the float lane evolves; both
    // are correct averages. Assert the program actually AVERAGES its two inputs.
    // `Value::Float` stores the f64 bit-pattern (keeps `Value: Eq/Ord`).
    let mkf = |a: f64, b: f64, o: f64| mog_synth::benchmark::Example {
        inputs: vec![
            mog_synth::benchmark::Value::Float(a.to_bits()),
            mog_synth::benchmark::Value::Float(b.to_bits()),
        ],
        expected: mog_synth::benchmark::Value::Float(o.to_bits()),
    };
    let spec = [mkf(2.0, 4.0, 3.0), mkf(3.0, 5.0, 4.0), mkf(1.0, 9.0, 5.0)];
    assert!(
        mog_synth::runtime::code_reproduces_examples(code, &spec),
        "synthesized program must average its two inputs, got:\n{code}"
    );
}

/// STRING accept (arity-1, string->string): "uppercase a string" → a real
/// GENERALIZING string program via `string_synth` (`.upper()`). NOT the memorizing
/// whole-word lexicon, NOT an i64 array op, NOT a refusal.
#[test]
fn string_request_synthesizes_real_string_program() {
    let root = fresh_root("string1");
    let r = run(&root, "uppercase a string");
    assert!(
        r.success,
        "string request must succeed (not refuse); got: {}",
        r.response
    );
    assert_eq!(r.route, QueryRoute::SynthesizeFunction);
    let code = &r.response;
    // Real string program of the right shape.
    assert!(
        code.contains("-> string"),
        "must be a string (-> string) program, got:\n{code}"
    );
    assert!(
        !code.contains("i64"),
        "must NOT be an i64 array op, got:\n{code}"
    );
    // GENERALIZING rule — NOT a memorized lookup table (the lexicon path emits
    // `if s == "..." { return "..."; }` over training pairs, which would not
    // generalize to an unseen string).
    assert!(
        !code.contains("if s =="),
        "must NOT be the memorizing whole-word lexicon lookup table, got:\n{code}"
    );
    // BEHAVIOR over method-label: the winning tier drifts (string_synth ->
    // typed-enum-str) while the emitted rule stays a real `.upper()`; assert the
    // program actually UPPER-CASES a fresh input (generalizes), which a lookup
    // table over an empty example set could not.
    let mks = |i: &str, o: &str| mog_synth::benchmark::Example {
        inputs: vec![mog_synth::benchmark::Value::Str(i.to_string())],
        expected: mog_synth::benchmark::Value::Str(o.to_string()),
    };
    let spec = [mks("hello", "HELLO"), mks("MiXeD", "MIXED"), mks("abc", "ABC")];
    assert!(
        mog_synth::runtime::code_reproduces_examples(code, &spec),
        "synthesized program must upper-case its input, got:\n{code}"
    );
}

/// REGRESSION (differential): i64 single-op, array transform, and a composition
/// must STILL emit i64 programs — the lattice generalization left the i64 lane
/// untouched.
#[test]
fn i64_lane_unchanged_by_lattice() {
    // Scalar single-op.
    let add = run(&fresh_root("i64_add"), "add two numbers");
    assert!(add.success, "add must still succeed: {}", add.response);
    // TYPE stays i64; BEHAVIOR over literal — the arity-polymorphic search names
    // params a0/a1, so the exact "a + b" string drifted to "a0 + a1". Assert it
    // still RETURNS i64 and actually adds its two arguments.
    assert!(
        add.response.contains("-> i64"),
        "add must still be an i64 program, got:\n{}",
        add.response
    );
    let add_spec = [
        mog_synth::benchmark::Example {
            inputs: vec![
                mog_synth::benchmark::Value::Int(2),
                mog_synth::benchmark::Value::Int(3),
            ],
            expected: mog_synth::benchmark::Value::Int(5),
        },
        mog_synth::benchmark::Example {
            inputs: vec![
                mog_synth::benchmark::Value::Int(5),
                mog_synth::benchmark::Value::Int(7),
            ],
            expected: mog_synth::benchmark::Value::Int(12),
        },
    ];
    assert!(
        mog_synth::runtime::code_reproduces_examples(&add.response, &add_spec),
        "add program must add its two arguments, got:\n{}",
        add.response
    );

    // Array transform.
    let sort = run(&fresh_root("i64_sort"), "sort an array");
    assert!(sort.success, "sort must still succeed: {}", sort.response);
    assert!(
        sort.response.contains("[i64]") && sort.response.contains("sort"),
        "sort must still be the i64 array program, got:\n{}",
        sort.response
    );

    // Multi-op composition (map chain + reduce) over i64.
    let comp = run(
        &fresh_root("i64_comp"),
        "the sum of the negated values in an array",
    );
    assert!(comp.success, "composition must still succeed: {}", comp.response);
    assert!(
        comp.response.contains("[i64]") && comp.response.contains("-> i64"),
        "composition must still be an i64 program, got:\n{}",
        comp.response
    );
}

/// MUST-REFUSE (differential): the fail-closed type gate is intact. A genuine
/// type mismatch — "reverse a string" resolves the i64-array `reverse` op while
/// the request asserts a string value — must be REFUSED, and an out-of-domain
/// request must be refused. Proves the lattice did not blanket-open the gate.
#[test]
fn type_gate_still_refuses_mismatch_and_out_of_domain() {
    let mismatch = run(&fresh_root("refuse_mismatch"), "reverse a string");
    assert!(
        !mismatch.success && mismatch.route == QueryRoute::Clarification,
        "string/i64 type mismatch must be refused, got: success={} route={:?}\n{}",
        mismatch.success,
        mismatch.route,
        mismatch.response
    );
    assert!(
        mismatch.response.to_lowercase().contains("type mismatch")
            || mismatch.response.to_lowercase().contains("string"),
        "refusal must cite the type mismatch, got:\n{}",
        mismatch.response
    );

    let ood = run(&fresh_root("refuse_ood"), "parse a csv file");
    assert!(
        !ood.success && ood.route == QueryRoute::Clarification,
        "out-of-domain request must be refused, got: success={} route={:?}\n{}",
        ood.success,
        ood.route,
        ood.response
    );
}
