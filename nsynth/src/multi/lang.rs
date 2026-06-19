//! Target Language Definitions for Multi-Language Generation

use std::collections::HashMap;

/// Supported target languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetLang {
    Rust,
    JavaScript,
    TypeScript,
    Python,
    Go,
    Java,
}

impl TargetLang {
    /// Get language name
    pub fn name(&self) -> &str {
        match self {
            TargetLang::Rust => "Rust",
            TargetLang::JavaScript => "JavaScript",
            TargetLang::TypeScript => "TypeScript",
            TargetLang::Python => "Python",
            TargetLang::Go => "Go",
            TargetLang::Java => "Java",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &str {
        match self {
            TargetLang::Rust => "rs",
            TargetLang::JavaScript => "js",
            TargetLang::TypeScript => "ts",
            TargetLang::Python => "py",
            TargetLang::Go => "go",
            TargetLang::Java => "java",
        }
    }

    /// Whether language uses semicolons
    pub fn uses_semicolons(&self) -> bool {
        matches!(self, TargetLang::JavaScript | TargetLang::TypeScript | TargetLang::Go | TargetLang::Java)
    }

    /// Whether language uses braces
    pub fn uses_braces(&self) -> bool {
        !matches!(self, TargetLang::Python)
    }

    /// Get comment prefix
    pub fn comment_prefix(&self) -> &str {
        match self {
            TargetLang::Python => "#",
            _ => "//",
        }
    }
}

/// Mog type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MogType {
    Unit,
    Bool,
    Int,
    Float,
    String,
    Array(Box<MogType>),
    Function(Vec<MogType>, Box<MogType>),
    Tuple(Vec<MogType>),
    Generic(String),
}

/// Mog operation representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MogOp {
    // Unary
    Negate,
    Not,
    Abs,
    Sqrt,
    Log,
    Exp,
    Sin,
    Cos,
    Tan,

    // Binary
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,

    // Control flow
    If,
    While,
    For,
    Loop,
    Break,
    Continue,
    Return,

    // Functions
    Call,
    Closure,
    Recurse,

    // Data structures
    Array,
    ArrayPush,
    ArrayPop,
    ArrayLen,
    Tuple,
    Struct,
    Enum,
    Variant,

    // Other
    Assign,
    Let,
    Mut,
    Ref,
    Deref,
    Print,
    Println,
}

/// Language target trait for code generation
pub trait LanguageTarget {
    /// Map Mog type to target language type
    fn type_map(&self, mog_type: &MogType) -> String;

    /// Map Mog operation to target language syntax
    fn op_map(&self, op: &MogOp) -> String;

    /// Get standard library imports
    fn stdlib(&self) -> String;

    /// Format function declaration
    fn format_function(&self, name: &str, params: &[(String, MogType)], ret: &MogType, body: &str) -> String;

    /// Format variable declaration
    fn format_var(&self, name: &str, ty: &MogType, value: Option<&str>) -> String;

    /// Format function call
    fn format_call(&self, func: &str, args: &[String]) -> String;

    /// Format if statement
    fn format_if(&self, cond: &str, then_block: &str, else_block: Option<&str>) -> String;

    /// Format while loop
    fn format_while(&self, cond: &str, body: &str) -> String;

    /// Get target language
    fn target(&self) -> TargetLang;

    /// Check if type should be explicit
    fn explicit_types(&self) -> bool {
        true
    }
}

/// Common type mappings
pub fn common_type_map(lang: TargetLang, mog_type: &MogType) -> String {
    match mog_type {
        MogType::Unit => {
            match lang {
                TargetLang::Rust => "()".to_string(),
                TargetLang::JavaScript | TargetLang::TypeScript => "void".to_string(),
                TargetLang::Python => "None".to_string(),
                TargetLang::Go => "struct{}".to_string(),
                TargetLang::Java => "void".to_string(),
            }
        }
        MogType::Bool => {
            match lang {
                TargetLang::Rust => "bool".to_string(),
                TargetLang::JavaScript | TargetLang::TypeScript => "boolean".to_string(),
                TargetLang::Python => "bool".to_string(),
                TargetLang::Go => "bool".to_string(),
                TargetLang::Java => "boolean".to_string(),
            }
        }
        MogType::Int => {
            match lang {
                TargetLang::Rust => "i64".to_string(),
                TargetLang::JavaScript | TargetLang::TypeScript => "number".to_string(),
                TargetLang::Python => "int".to_string(),
                TargetLang::Go => "int64".to_string(),
                TargetLang::Java => "long".to_string(),
            }
        }
        MogType::Float => {
            match lang {
                TargetLang::Rust => "f64".to_string(),
                TargetLang::JavaScript | TargetLang::TypeScript => "number".to_string(),
                TargetLang::Python => "float".to_string(),
                TargetLang::Go => "float64".to_string(),
                TargetLang::Java => "double".to_string(),
            }
        }
        MogType::String => {
            match lang {
                TargetLang::Rust => "String".to_string(),
                TargetLang::JavaScript | TargetLang::TypeScript => "string".to_string(),
                TargetLang::Python => "str".to_string(),
                TargetLang::Go => "string".to_string(),
                TargetLang::Java => "String".to_string(),
            }
        }
        MogType::Array(inner) => {
            let inner_str = common_type_map(lang, inner);
            match lang {
                TargetLang::Rust => format!("Vec<{}>", inner_str),
                TargetLang::JavaScript => format!("Array<{}>", inner_str),
                TargetLang::TypeScript => format!("{}[]", inner_str),
                TargetLang::Python => format!("List[{}]", inner_str),
                TargetLang::Go => format!("[]{}", inner_str),
                TargetLang::Java => format!("List<{}>", inner_str),
            }
        }
        MogType::Function(params, ret) => {
            let param_str: Vec<String> = params.iter()
                .map(|p| common_type_map(lang, p))
                .collect();
            let ret_str = common_type_map(lang, ret);
            match lang {
                TargetLang::Rust => format!("fn({}) -> {}", param_str.join(", "), ret_str),
                TargetLang::JavaScript | TargetLang::TypeScript => {
                    format!("({}) => {}", param_str.join(", "), ret_str)
                }
                TargetLang::Python => {
                    format!("Callable[[{}], {}]", param_str.join(", "), ret_str)
                }
                TargetLang::Go => format!("func({}) {}", param_str.join(", "), ret_str),
                TargetLang::Java => format!("({}) -> {}", param_str.join(", "), ret_str),
            }
        }
        MogType::Tuple(types) => {
            let types_str: Vec<String> = types.iter()
                .map(|t| common_type_map(lang, t))
                .collect();
            match lang {
                TargetLang::Rust => format!("({})", types_str.join(", ")),
                TargetLang::JavaScript | TargetLang::TypeScript => format!("[{}]", types_str.join(", ")),
                TargetLang::Python => format!("Tuple[{}]", types_str.join(", ")),
                TargetLang::Go => format!("struct {{ {} }}", types_str.join("; ")),
                TargetLang::Java => format!("Tuple<{}>", types_str.join(", ")),
            }
        }
        MogType::Generic(name) => name.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_lang_properties() {
        assert_eq!(TargetLang::Rust.name(), "Rust");
        assert_eq!(TargetLang::Python.extension(), "py");
        assert!(TargetLang::JavaScript.uses_semicolons());
        assert!(!TargetLang::Python.uses_semicolons());
        assert!(!TargetLang::Python.uses_braces());
    }

    #[test]
    fn test_type_map() {
        assert_eq!(common_type_map(TargetLang::Rust, &MogType::Int), "i64");
        assert_eq!(common_type_map(TargetLang::JavaScript, &MogType::Int), "number");
        assert_eq!(common_type_map(TargetLang::Python, &MogType::String), "str");
    }

    #[test]
    fn test_array_type_map() {
        let arr = MogType::Array(Box::new(MogType::Int));
        assert_eq!(common_type_map(TargetLang::Rust, &arr), "Vec<i64>");
        assert_eq!(common_type_map(TargetLang::TypeScript, &arr), "number[]");
        assert_eq!(common_type_map(TargetLang::Python, &arr), "List[int]");
    }
}
