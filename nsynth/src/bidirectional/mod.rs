//! Bidirectional synthesis: Code → Natural Language
//!
//! Reverse pipeline that analyzes generated code and produces
//! natural language documentation using Linguigenesis.

pub mod analyzer;
pub mod generator;
pub mod parser;

pub use analyzer::{analyze_semantics, CodeSemantics};
pub use generator::generate_nl;
pub use parser::parse_code;

/// Complete Code → NL pipeline
pub fn code_to_nl(code: &str) -> Result<String, String> {
    let ast = parse_code(code)?;
    let semantics = analyze_semantics(&ast);
    let nl = generate_nl(&semantics);
    Ok(nl)
}

#[cfg(test)]
mod wire_tests {
    use super::*;

    /// The code→NL pipeline produces a non-empty explanation for the kind of
    /// single-function program the synthesizer emits (now wired into the agent's
    /// emit_result as an "explanation:" section).
    #[test]
    fn code_to_nl_explains_a_synthesized_function() {
        let code = "fn add_one(a: i64) -> i64 {\n    return a + 1;\n}";
        let nl = code_to_nl(code).expect("should produce an explanation");
        assert!(!nl.trim().is_empty(), "explanation must be non-empty: {nl:?}");
    }
}
