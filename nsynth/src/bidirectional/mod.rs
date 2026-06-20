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
