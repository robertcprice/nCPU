//! Code parser for bidirectional synthesis
//!
//! Simple pattern-based parser for Rust function extraction.
//! Production version would use syn or a full compiler frontend.

/// Abstract syntax tree representation
#[derive(Debug, Clone)]
pub struct AST {
    pub functions: Vec<Function>,
    pub structs: Vec<Struct>,
    pub imports: Vec<String>,
}

/// Function definition
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: String,
    pub body: Vec<Statement>,
    pub attributes: Vec<String>,
}

/// Function parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_: String,
}

/// Struct definition
#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<Field>,
    pub generics: Vec<String>,
}

/// Struct field
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub type_: String,
}

/// Statement or expression
#[derive(Debug, Clone)]
pub enum Statement {
    /// Variable declaration
    Let { name: String, type_: Option<String>, value: Box<Expression> },
    /// Assignment
    Assign { name: String, value: Box<Expression> },
    /// Return statement
    Return(Box<Expression>),
    /// If statement
    If { condition: Box<Expression>, then_block: Vec<Statement>, else_block: Option<Vec<Statement>> },
    /// Loop
    Loop { body: Vec<Statement> },
    /// While loop
    While { condition: Box<Expression>, body: Vec<Statement> },
    /// For loop
    For { var: String, iter: Box<Expression>, body: Vec<Statement> },
    /// Expression statement
    Expr(Box<Expression>),
    /// Block
    Block(Vec<Statement>),
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expression {
    /// Integer literal
    Int(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Bool(bool),
    /// Variable reference
    Variable(String),
    /// Binary operation
    BinOp { op: BinOp, left: Box<Expression>, right: Box<Expression> },
    /// Unary operation
    UnaryOp { op: UnaryOp, operand: Box<Expression> },
    /// Function call
    Call { func: String, args: Vec<Expression> },
    /// Method call
    MethodCall { object: Box<Expression>, method: String, args: Vec<Expression> },
    /// Field access
    FieldAccess { object: Box<Expression>, field: String },
    /// Index access
    Index { object: Box<Expression>, index: Box<Expression> },
    /// Array literal
    Array(Vec<Expression>),
}

/// Binary operators
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    BitAnd, BitOr, BitXor, Shl, Shr,
    Eq, Lt, Le, Gt, Ge, Ne,
    And, Or,
    Assign, AddAssign, SubAssign, MulAssign, DivAssign,
}

/// Unary operators
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not, Deref, Ref,
}

/// Parse Rust code into AST
pub fn parse_code(code: &str) -> Result<AST, String> {
    let mut functions = Vec::new();
    let mut structs = Vec::new();
    let mut imports = Vec::new();

    for line in code.lines() {
        let line = line.trim();

        // Extract imports
        if line.starts_with("use ") {
            imports.push(line.to_string());
            continue;
        }

        // Extract struct definitions
        if line.starts_with("struct ") {
            if let Some(struct_def) = parse_struct(line) {
                structs.push(struct_def);
            }
            continue;
        }

        // Extract function definitions
        if line.starts_with("fn ") {
            if let Some(func) = parse_function_signature(line) {
                functions.push(func);
            }
        }
    }

    Ok(AST { functions, structs, imports })
}

/// Parse struct definition
fn parse_struct(line: &str) -> Option<Struct> {
    // Simple pattern: "struct Name { fields }"
    let line = line.strip_prefix("struct ")?;
    let mut parts = line.split_whitespace();
    let name = parts.next()?.to_string();

    Some(Struct {
        name,
        fields: Vec::new(), // Would need full parser for fields
        generics: Vec::new(),
    })
}

/// Parse function signature
fn parse_function_signature(line: &str) -> Option<Function> {
    // Pattern: "fn name(params) -> return_type {"
    let line = line.strip_prefix("fn ")?;
    let mut parts = line.split('(');
    let name = parts.next()?.to_string();

    // Extract parameters
    let params_str = parts.next()?;
    let params = if let Some(close_idx) = params_str.find(')') {
        let param_list = &params_str[..close_idx];
        if param_list.is_empty() {
            Vec::new()
        } else {
            param_list.split(',')
                .map(|p| {
                    let p = p.trim();
                    let mut iter = p.split_whitespace();
                    let name = iter.next().unwrap_or("arg").to_string();
                    let type_ = iter.next().unwrap_or("i64").to_string();
                    Parameter { name, type_ }
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    // Extract return type
    let return_str = &params_str[params_str.find(')')? + 1..];
    let return_type = if let Some(arrow_idx) = return_str.find("->") {
        let after_arrow = &return_str[arrow_idx + 2..];
        if let Some(open_brace_idx) = after_arrow.find('{') {
            after_arrow[..open_brace_idx].trim().to_string()
        } else {
            after_arrow.trim().to_string()
        }
    } else {
        "()".to_string()
    };

    Some(Function {
        name,
        params,
        return_type,
        body: Vec::new(),
        attributes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_function() {
        let code = r#"
            fn add(a: i64, b: i64) -> i64 {
                return a + b;
            }
        "#;

        let ast = parse_code(code).unwrap();
        assert_eq!(ast.functions.len(), 1);
        assert_eq!(ast.functions[0].name, "add");
        assert_eq!(ast.functions[0].params.len(), 2);
        assert_eq!(ast.functions[0].return_type, "i64");
    }

    #[test]
    fn test_parse_multiple_functions() {
        let code = r#"
            fn add(a: i64, b: i64) -> i64 { a + b }
            fn multiply(x: i64, y: i64) -> i64 { x * y }
        "#;

        let ast = parse_code(code).unwrap();
        assert_eq!(ast.functions.len(), 2);
    }
}
