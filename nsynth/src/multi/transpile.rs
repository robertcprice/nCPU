//! Main Transpiler for Multi-Language Code Generation

use super::{
    lang::{TargetLang, LanguageTarget, MogType, MogOp, common_type_map},
    js::JavaScriptTarget,
    py::PythonTarget,
    ts::TypeScriptTarget,
};
use std::collections::HashMap;

/// Transpile error
#[derive(Debug, Clone)]
pub enum TranspileError {
    UnsupportedOperation(String),
    UnsupportedType(String),
    InvalidAST(String),
}

impl std::fmt::Display for TranspileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranspileError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            TranspileError::UnsupportedType(ty) => write!(f, "Unsupported type: {}", ty),
            TranspileError::InvalidAST(msg) => write!(f, "Invalid AST: {}", msg),
        }
    }
}

impl std::error::Error for TranspileError {}

/// Simple AST node for transpilation
#[derive(Debug, Clone)]
pub enum ASTNode {
    // Literals
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),

    // Variables
    Var(String),
    Let(String, Box<ASTNode>),
    Assign(String, Box<ASTNode>),

    // Operations
    Unary(MogOp, Box<ASTNode>),
    Binary(MogOp, Box<ASTNode>, Box<ASTNode>),

    // Control flow
    If(Box<ASTNode>, Vec<ASTNode>, Option<Vec<ASTNode>>),
    While(Box<ASTNode>, Vec<ASTNode>),
    For(String, Box<ASTNode>, Vec<ASTNode>),
    Loop(Vec<ASTNode>),
    Break,
    Continue,

    // Functions
    Function {
        name: String,
        params: Vec<(String, MogType)>,
        ret: MogType,
        body: Vec<ASTNode>,
    },
    Call(String, Vec<ASTNode>),
    Return(Box<ASTNode>),

    // Data structures
    Array(Vec<ASTNode>),
    Index(Box<ASTNode>, Box<ASTNode>),
    Tuple(Vec<ASTNode>),
    Struct {
        name: String,
        fields: HashMap<String, MogType>,
    },
}

/// Transpile AST to target language
pub fn transpile(ast: &[ASTNode], target: TargetLang) -> Result<String, TranspileError> {
    let language_target: Box<dyn LanguageTarget> = match target {
        TargetLang::Rust => return Err(TranspileError::UnsupportedOperation("Rust transpile not implemented (source is Rust)".into())),
        TargetLang::JavaScript => Box::new(JavaScriptTarget::new()),
        TargetLang::TypeScript => Box::new(TypeScriptTarget::new()),
        TargetLang::Python => Box::new(PythonTarget::new()),
        TargetLang::Go => return Err(TranspileError::UnsupportedOperation("Go transpile not implemented yet".into())),
        TargetLang::Java => return Err(TranspileError::UnsupportedOperation("Java transpile not implemented yet".into())),
    };

    let mut output = String::new();
    let mut context = TranspileContext::new(target);

    for node in ast {
        output.push_str(&transpile_node(node, &language_target, &mut context)?);
        output.push('\n');
    }

    Ok(output)
}

/// Transpile context for symbol table and state
pub struct TranspileContext {
    pub target: TargetLang,
    pub indent: usize,
    pub symbols: HashMap<String, MogType>,
    pub temp_counter: usize,
}

impl TranspileContext {
    pub fn new(target: TargetLang) -> Self {
        Self {
            target,
            indent: 0,
            symbols: HashMap::new(),
            temp_counter: 0,
        }
    }

    pub fn fresh_temp(&mut self) -> String {
        let name = format!("_tmp{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    pub fn indent(&self) -> String {
        "    ".repeat(self.indent)
    }

    pub fn add_symbol(&mut self, name: String, ty: MogType) {
        self.symbols.insert(name, ty);
    }
}

/// Transpile single AST node
fn transpile_node(
    node: &ASTNode,
    target: &Box<dyn LanguageTarget>,
    ctx: &mut TranspileContext,
) -> Result<String, TranspileError> {
    match node {
        // Literals
        ASTNode::Unit => Ok("null".to_string()),
        ASTNode::Bool(b) => Ok(b.to_string()),
        ASTNode::Int(i) => Ok(i.to_string()),
        ASTNode::Float(f) => Ok({
            if f.is_finite() {
                f.to_string()
            } else if f.is_infinite() {
                if f.is_sign_positive() { "Infinity".to_string() } else { "-Infinity".to_string() }
            } else {
                "NaN".to_string()
            }
        }),
        ASTNode::String(s) => Ok(format!("\"{}\"", escape_string(s))),

        // Variables
        ASTNode::Var(name) => Ok(name.clone()),
        ASTNode::Let(name, value) => {
            let value_str = transpile_node(value, target, ctx)?;
            let ty = infer_type(value);
            ctx.add_symbol(name.clone(), ty.clone());
            Ok(target.format_var(name, &ty, Some(&value_str)))
        }
        ASTNode::Assign(name, value) => {
            let value_str = transpile_node(value, target, ctx)?;
            Ok(format!("{} = {}", name, value_str))
        }

        // Operations
        ASTNode::Unary(op, operand) => {
            let op_str = target.op_map(op);
            let operand_str = transpile_node(operand, target, ctx)?;
            Ok(format!("{}{}", op_str, operand_str))
        }
        ASTNode::Binary(op, left, right) => {
            let left_str = transpile_node(left, target, ctx)?;
            let right_str = transpile_node(right, target, ctx)?;
            Ok(format!("{} {} {}", left_str, target.op_map(op), right_str))
        }

        // Control flow
        ASTNode::If(cond, then_block, else_block) => {
            let cond_str = transpile_node(cond, target, ctx)?;
            let mut then_body = String::new();
            ctx.indent += 1;
            for node in then_block {
                then_body.push_str(&ctx.indent());
                then_body.push_str(&transpile_node(node, target, ctx)?);
                then_body.push('\n');
            }
            ctx.indent -= 1;

            let mut else_body = None;
            if let Some(else_nodes) = else_block {
                let mut body = String::new();
                ctx.indent += 1;
                for node in else_nodes {
                    body.push_str(&ctx.indent());
                    body.push_str(&transpile_node(node, target, ctx)?);
                    body.push('\n');
                }
                ctx.indent -= 1;
                else_body = Some(body);
            }

            Ok(target.format_if(&cond_str, &then_body, else_body.as_deref()))
        }
        ASTNode::While(cond, body) => {
            let cond_str = transpile_node(cond, target, ctx)?;
            let mut body_str = String::new();
            ctx.indent += 1;
            for node in body {
                body_str.push_str(&ctx.indent());
                body_str.push_str(&transpile_node(node, target, ctx)?);
                body_str.push('\n');
            }
            ctx.indent -= 1;
            Ok(target.format_while(&cond_str, &body_str))
        }
        ASTNode::For(var, iter, body) => {
            let iter_str = transpile_node(iter, target, ctx)?;
            let mut body_str = String::new();
            ctx.indent += 1;
            for node in body {
                body_str.push_str(&ctx.indent());
                body_str.push_str(&transpile_node(node, target, ctx)?);
                body_str.push('\n');
            }
            ctx.indent -= 1;

            match ctx.target {
                TargetLang::JavaScript | TargetLang::TypeScript => {
                    Ok(format!("for (const {} of {}) {{\n{}\n}}", var, iter_str, body_str))
                }
                TargetLang::Python => {
                    Ok(format!("for {} in {}:\n{}", var, iter_str, body_str))
                }
                _ => Err(TranspileError::UnsupportedOperation("For loop".into())),
            }
        }
        ASTNode::Loop(body) => {
            let mut body_str = String::new();
            ctx.indent += 1;
            for node in body {
                body_str.push_str(&ctx.indent());
                body_str.push_str(&transpile_node(node, target, ctx)?);
                body_str.push('\n');
            }
            ctx.indent -= 1;

            match ctx.target {
                TargetLang::JavaScript | TargetLang::TypeScript => {
                    Ok(format!("while (true) {{\n{}\n}}", body_str))
                }
                TargetLang::Python => {
                    Ok(format!("while True:\n{}", body_str))
                }
                _ => Err(TranspileError::UnsupportedOperation("Loop".into())),
            }
        }
        ASTNode::Break => Ok("break".to_string()),
        ASTNode::Continue => Ok("continue".to_string()),

        // Functions
        ASTNode::Function { name, params, ret, body } => {
            let mut body_str = String::new();
            ctx.indent += 1;
            for node in body {
                body_str.push_str(&ctx.indent());
                body_str.push_str(&transpile_node(node, target, ctx)?);
                if ctx.target.uses_semicolons() {
                    body_str.push(';');
                }
                body_str.push('\n');
            }
            ctx.indent -= 1;

            Ok(target.format_function(name, params, ret, &body_str))
        }
        ASTNode::Call(func, args) => {
            let arg_strs: Result<Vec<String>, TranspileError> = args.iter()
                .map(|a| transpile_node(a, target, ctx))
                .collect();
            Ok(target.format_call(func, &arg_strs?))
        }
        ASTNode::Return(value) => {
            let value_str = transpile_node(value, target, ctx)?;
            Ok(format!("return {}", value_str))
        }

        // Data structures
        ASTNode::Array(elements) => {
            let elem_strs: Result<Vec<String>, TranspileError> = elements.iter()
                .map(|e| transpile_node(e, target, ctx))
                .collect();
            Ok(format!("[{}]", elem_strs?.join(", ")))
        }
        ASTNode::Index(array, index) => {
            let array_str = transpile_node(array, target, ctx)?;
            let index_str = transpile_node(index, target, ctx)?;
            Ok(format!("{}[{}]", array_str, index_str))
        }
        ASTNode::Tuple(elements) => {
            let elem_strs: Result<Vec<String>, TranspileError> = elements.iter()
                .map(|e| transpile_node(e, target, ctx))
                .collect();
            match ctx.target {
                TargetLang::JavaScript | TargetLang::TypeScript => {
                    Ok(format!("[{}]", elem_strs?.join(", ")))
                }
                TargetLang::Python => {
                    Ok(format!("({})", elem_strs?.join(", ")))
                }
                _ => Err(TranspileError::UnsupportedOperation("Tuple".into())),
            }
        }
        ASTNode::Struct { name, fields } => {
            let field_strs: Vec<String> = fields.iter()
                .map(|(n, t)| format!("{}: {}", n, target.type_map(t)))
                .collect();
            match ctx.target {
                TargetLang::TypeScript => {
                    Ok(format!("interface {} {{ {} }}", name, field_strs.join("; ")))
                }
                TargetLang::Python => {
                    Ok(format!("class {}:\n    pass", name))
                }
                _ => Err(TranspileError::UnsupportedOperation("Struct".into())),
            }
        }
    }
}

/// Escape string literals
fn escape_string(s: &str) -> String {
    s.chars().flat_map(|c| match c {
        '\n' => Some("\\n".to_string()),
        '\r' => Some("\\r".to_string()),
        '\t' => Some("\\t".to_string()),
        '"' => Some("\\\"".to_string()),
        '\\' => Some("\\\\".to_string()),
        _ if c.is_ascii_control() => None,
        _ => Some(c.to_string()),
    }).collect::<String>()
}

/// Infer type from AST node (simplified)
fn infer_type(node: &ASTNode) -> MogType {
    match node {
        ASTNode::Unit => MogType::Unit,
        ASTNode::Bool(_) => MogType::Bool,
        ASTNode::Int(_) => MogType::Int,
        ASTNode::Float(_) => MogType::Float,
        ASTNode::String(_) => MogType::String,
        ASTNode::Array(_) => MogType::Array(Box::new(MogType::Int)), // Default to Int array
        ASTNode::Var(_) => MogType::Int, // Default
        _ => MogType::Unit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_arithmetic() {
        let ast = vec![
            ASTNode::Binary(MogOp::Add,
                Box::new(ASTNode::Int(1)),
                Box::new(ASTNode::Int(2))
            ),
        ];

        let js = transpile(&ast, TargetLang::JavaScript).unwrap();
        assert!(js.contains("1 + 2"));

        let py = transpile(&ast, TargetLang::Python).unwrap();
        assert!(py.contains("1 + 2"));
    }

    #[test]
    fn test_transpile_function() {
        let ast = vec![
            ASTNode::Function {
                name: "add".to_string(),
                params: vec![("a".to_string(), MogType::Int), ("b".to_string(), MogType::Int)],
                ret: MogType::Int,
                body: vec![
                    ASTNode::Binary(MogOp::Add,
                        Box::new(ASTNode::Var("a".to_string())),
                        Box::new(ASTNode::Var("b".to_string()))
                    ),
                ],
            },
        ];

        let js = transpile(&ast, TargetLang::JavaScript).unwrap();
        assert!(js.contains("function add"));
        assert!(js.contains("return"));

        let py = transpile(&ast, TargetLang::Python).unwrap();
        assert!(py.contains("def add"));
    }
}
