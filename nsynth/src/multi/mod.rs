//! Multi-Language Code Generation for nCPU/nSynth
//!
//! Transpile Mog IR to JavaScript, Python, TypeScript, Go, and Java.

pub mod lang;
pub mod transpile;
pub mod js;
pub mod py;
pub mod ts;

pub use lang::{TargetLang, LanguageTarget};
pub use transpile::{transpile, TranspileError};
pub use js::JavaScriptTarget;
pub use py::PythonTarget;
pub use ts::TypeScriptTarget;

/// Multi-language synthesis configuration
#[derive(Debug, Clone)]
pub struct MultiConfig {
    /// Target language
    pub target: TargetLang,
    /// Include type annotations (if applicable)
    pub types: bool,
    /// Include comments
    pub comments: bool,
    /// Minify output
    pub minify: bool,
}

impl Default for MultiConfig {
    fn default() -> Self {
        Self {
            target: TargetLang::Rust,
            types: true,
            comments: true,
            minify: false,
        }
    }
}

impl MultiConfig {
    /// Create new config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target language
    pub fn with_target(mut self, target: TargetLang) -> Self {
        self.target = target;
        self
    }

    /// Enable/disable type annotations
    pub fn with_types(mut self, types: bool) -> Self {
        self.types = types;
        self
    }

    /// Enable/disable comments
    pub fn with_comments(mut self, comments: bool) -> Self {
        self.comments = comments;
        self
    }

    /// Enable/disable minification
    pub fn with_minify(mut self, minify: bool) -> Self {
        self.minify = minify;
        self
    }
}
