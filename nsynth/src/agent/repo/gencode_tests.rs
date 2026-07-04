//! Test + demo emission for synthesized multi-file projects (PIECE 3).
//!
//! The PRODUCT writer ([`crate::agent::repo::nl_fixture_harness::write_synthesized_project`])
//! transpiles solved Mog components to Rust and runs a `cargo check` compile
//! gate. That proves the code *type-checks*, but not that it *reproduces the
//! examples it was synthesized from*. This module renders the verifying
//! examples back into Rust source so the generated crate carries:
//!   * a `#[cfg(test)] mod tests` next to each fn that `assert_eq!`s the fn
//!     against its own examples (a real `cargo test` oracle), and
//!   * a `src/main.rs` demo that calls each fn on its first example input.
//!
//! ## Contract with the transpiler ([`crate::mog_transpile::to_rust`])
//! The transpiled fns keep the Mog fn name and have these arg/return spellings
//! (see `rewrite_type_rust`): `i64` scalars stay `i64`; `[i64]` arrays become
//! **by-value `Vec<i64>`** (NOT `&[i64]`); `bool` passes through. Because every
//! call site sits in the fn's own type context, integer/array literals need NO
//! `i64` suffix — Rust infers element + scalar types from the signature (this
//! mirrors the existing hand-written fixtures, e.g. `total(vec![10, 20, 30])`
//! and `add_two(2, 3)`).
//!
//! ## Soundness rule (HARD)
//! An example is rendered ONLY if every input AND the expected output map to a
//! supported literal (Int / Bool / int-array). Anything else (Float, Str, Pair,
//! Quad, Tuple, Struct, Tree, Tensor, or a non-int / nested array) returns
//! `None` from [`render_value_literal`] and the WHOLE example is skipped. If no
//! example renders, [`emit_tests_module`] returns the EMPTY string so the caller
//! emits no test module at all — never an empty `mod tests {}` and never an
//! always-true `assert!(true)`. A test that does not exercise the fn is worse
//! than no test, so we refuse to emit one.

use crate::benchmark::{Example, Value};

/// Render a single [`Value`] as a Rust literal usable as a fn argument or as the
/// RHS of an `assert_eq!` against the fn's return value.
///
/// Supported (returns `Some`):
///   * `Value::Int(i)`   -> `"7"` / `"-3"` (bare; the call site's `i64` context
///     fixes the type — no suffix, matching the existing fixtures).
///   * `Value::Bool(b)`  -> `"true"` / `"false"`.
///   * `Value::Array([Int..])` -> `"vec![1, 2, 3]"` (by-value `Vec<i64>`, the
///     transpiled array param/return type); empty array -> `"vec![]"` (the
///     element type is inferred from the fn signature, exactly as the existing
///     `total(vec![])` fixture relies on).
///
/// Unsupported (returns `None`, so the caller SKIPS the whole example): every
/// other variant — `Float`, `Str`, `Pair`, `Quad`, `Tuple`, `Struct`, `Tree`,
/// `Tensor`, and any array whose elements are not all `Int` (typed/nested).
/// Returning `None` (rather than guessing) is what keeps a skipped example from
/// degrading into an always-true or mistyped test.
pub fn render_value_literal(v: &Value) -> Option<String> {
    match v {
        Value::Int(i) => Some(i.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Array(_) => {
            // Only all-integer arrays are renderable; `as_i64_slice` yields
            // `None` for any typed/nested array, which we propagate as a skip.
            let ints = v.as_i64_slice()?;
            let body = ints
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Some(format!("vec![{body}]"))
        }
        // Float / Str / Pair / Quad / Tuple / Struct / Tree / Tensor: no sound
        // literal rendering for the generated-crate call site -> skip.
        _ => None,
    }
}

/// Render one example as a call + assert: `assert_eq!(<fn_name>(<args>), <exp>);`.
/// Returns `None` if ANY input or the expected output is unrenderable, so the
/// caller drops the whole example (never a partial/always-true assertion).
fn render_example_assert(fn_name: &str, ex: &Example) -> Option<String> {
    let mut args = Vec::with_capacity(ex.inputs.len());
    for inp in &ex.inputs {
        args.push(render_value_literal(inp)?);
    }
    let expected = render_value_literal(&ex.expected)?;
    let arg_list = args.join(", ");
    Some(format!("        assert_eq!({fn_name}({arg_list}), {expected});"))
}

/// Emit a `#[cfg(test)] mod tests` block that reproduces every renderable
/// example for `fn_name`. SKIPS any example whose args/expected don't render.
///
/// Returns the EMPTY string when NO example renders (so the caller emits no test
/// module rather than an empty/always-true one). The generated test fn is named
/// `reproduces_examples` and lives in a `tests` submodule with `use super::*;`,
/// so it sees the (now `pub`) fn from the same generated module file.
pub fn emit_tests_module(fn_name: &str, examples: &[Example]) -> String {
    let asserts: Vec<String> = examples
        .iter()
        .filter_map(|ex| render_example_assert(fn_name, ex))
        .collect();
    if asserts.is_empty() {
        // No example rendered -> emit nothing. An empty `mod tests {}` or an
        // `assert!(true)` would be a false-green oracle; refuse to write one.
        return String::new();
    }
    let body = asserts.join("\n");
    format!(
        "\n#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    #[test]\n    fn reproduces_examples() {{\n{body}\n    }}\n}}\n"
    )
}

/// Emit a `src/main.rs` demo that calls each component's fn on its FIRST
/// renderable example's inputs and prints the result. Each component is a
/// `(fn_name, examples)` pair; a component is included only if its first example
/// renders (so the demo is always a real, type-checking call — never a stub).
///
/// The returned source references the lib fns through the binary crate's
/// dependency on the library crate, so the caller (PIECE 4, which knows the
/// package name) must inject `use <pkg>::*;` at the top of the file. This fn
/// emits ONLY the `fn main() { ... }` body so the writer controls the `use`
/// line (the pkg name is not known here). If NO component renders, returns the
/// empty string and the caller emits no `main.rs`.
pub fn emit_main_demo(components: &[(String, Vec<Example>)]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (fn_name, examples) in components {
        let Some(first) = examples.first() else {
            continue;
        };
        // Render the first example's inputs; skip the component if any arg is
        // unrenderable (we never emit a call that won't type-check).
        let mut args = Vec::with_capacity(first.inputs.len());
        let mut renderable = true;
        for inp in &first.inputs {
            match render_value_literal(inp) {
                Some(lit) => args.push(lit),
                None => {
                    renderable = false;
                    break;
                }
            }
        }
        if !renderable {
            continue;
        }
        let arg_list = args.join(", ");
        // `{:?}` prints any return type (i64 / bool / Vec<i64>) without needing
        // to know it here — Debug is derived for all of them.
        lines.push(format!("    println!(\"{{:?}}\", {fn_name}({arg_list}));"));
    }
    if lines.is_empty() {
        return String::new();
    }
    let body = lines.join("\n");
    format!("fn main() {{\n{body}\n}}\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{Example, Value};

    #[test]
    fn renders_int_bool_and_int_array_literals() {
        assert_eq!(render_value_literal(&Value::Int(7)).unwrap(), "7");
        assert_eq!(render_value_literal(&Value::Int(-3)).unwrap(), "-3");
        assert_eq!(render_value_literal(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(render_value_literal(&Value::Bool(false)).unwrap(), "false");
        assert_eq!(
            render_value_literal(&Value::int_array(&[1, 2, 3])).unwrap(),
            "vec![1, 2, 3]"
        );
        // Empty int array still renders; element type is inferred at the call.
        assert_eq!(render_value_literal(&Value::int_array(&[])).unwrap(), "vec![]");
    }

    #[test]
    fn refuses_unsupported_value_kinds() {
        // Float / Str / Pair are not soundly renderable -> None (caller skips).
        assert!(render_value_literal(&Value::Float(0u64)).is_none());
        assert!(render_value_literal(&Value::Str("hi".to_string())).is_none());
        assert!(render_value_literal(&Value::Pair(1, 2)).is_none());
        // A typed (non-int) array is also unrenderable.
        let typed = Value::array_of(vec![Value::Str("a".to_string())]);
        assert!(render_value_literal(&typed).is_none());
    }

    #[test]
    fn emits_assert_eq_text_for_renderable_examples() {
        let examples = vec![
            Example { inputs: vec![Value::Int(2), Value::Int(3)], expected: Value::Int(5) },
            Example { inputs: vec![Value::Int(-1), Value::Int(1)], expected: Value::Int(0) },
        ];
        let module = emit_tests_module("add_two", &examples);
        assert!(module.contains("#[cfg(test)]"), "got: {module}");
        assert!(module.contains("mod tests"), "got: {module}");
        assert!(module.contains("use super::*;"), "got: {module}");
        assert!(module.contains("fn reproduces_examples"), "got: {module}");
        assert!(module.contains("assert_eq!(add_two(2, 3), 5);"), "got: {module}");
        assert!(module.contains("assert_eq!(add_two(-1, 1), 0);"), "got: {module}");
    }

    #[test]
    fn skips_non_renderable_example_without_always_true_test() {
        // Two examples: one renderable (Int), one not (Str expected). The Str
        // example must be DROPPED, and only the renderable assert appears.
        let examples = vec![
            Example { inputs: vec![Value::Int(4)], expected: Value::Int(16) },
            Example {
                inputs: vec![Value::Int(5)],
                expected: Value::Str("nope".to_string()),
            },
        ];
        let module = emit_tests_module("square", &examples);
        assert!(module.contains("assert_eq!(square(4), 16);"), "got: {module}");
        assert!(!module.contains("nope"), "unrenderable example must be skipped: {module}");
        // Exactly one assert_eq! (the Str example was dropped).
        assert_eq!(module.matches("assert_eq!").count(), 1, "got: {module}");
    }

    #[test]
    fn emits_no_module_when_nothing_renders() {
        // Every example is unrenderable -> empty string (no `mod tests`).
        let examples = vec![Example {
            inputs: vec![Value::Str("x".to_string())],
            expected: Value::Str("y".to_string()),
        }];
        let module = emit_tests_module("f", &examples);
        assert!(module.is_empty(), "must emit nothing, got: {module}");
        // And the all-empty input list also yields nothing (no always-true test).
        assert!(emit_tests_module("g", &[]).is_empty());
    }

    #[test]
    fn array_examples_render_vec_literals() {
        let examples = vec![Example {
            inputs: vec![Value::int_array(&[10, 20, 30])],
            expected: Value::Int(60),
        }];
        let module = emit_tests_module("total", &examples);
        assert!(
            module.contains("assert_eq!(total(vec![10, 20, 30]), 60);"),
            "got: {module}"
        );
    }

    #[test]
    fn main_demo_calls_first_example_of_each_renderable_component() {
        let components = vec![
            (
                "negate".to_string(),
                vec![Example { inputs: vec![Value::Int(5)], expected: Value::Int(-5) }],
            ),
            (
                "total".to_string(),
                vec![Example {
                    inputs: vec![Value::int_array(&[1, 2, 3])],
                    expected: Value::Int(6),
                }],
            ),
        ];
        let main = emit_main_demo(&components);
        assert!(main.starts_with("fn main() {"), "got: {main}");
        assert!(main.contains("println!(\"{:?}\", negate(5));"), "got: {main}");
        assert!(main.contains("println!(\"{:?}\", total(vec![1, 2, 3]));"), "got: {main}");
    }

    #[test]
    fn main_demo_skips_components_with_unrenderable_first_example() {
        let components = vec![
            (
                "good".to_string(),
                vec![Example { inputs: vec![Value::Int(1)], expected: Value::Int(2) }],
            ),
            (
                "bad".to_string(),
                vec![Example {
                    inputs: vec![Value::Str("x".to_string())],
                    expected: Value::Int(0),
                }],
            ),
        ];
        let main = emit_main_demo(&components);
        assert!(main.contains("good(1)"), "got: {main}");
        assert!(!main.contains("bad("), "unrenderable component must be skipped: {main}");
    }

    #[test]
    fn main_demo_empty_when_no_component_renders() {
        let components = vec![(
            "f".to_string(),
            vec![Example {
                inputs: vec![Value::Str("x".to_string())],
                expected: Value::Int(0),
            }],
        )];
        assert!(emit_main_demo(&components).is_empty());
        assert!(emit_main_demo(&[]).is_empty());
    }
}
