use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::benchmark::{generated_holdouts, Problem, Value as BenchmarkValue};

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    Result {
        is_ok: bool,
        value: Box<Value>,
    },
    Optional {
        is_some: bool,
        value: Box<Value>,
    },
    Function(String),
    Builtin(Builtin),
    Closure(Closure),
    Unit,
}

#[derive(Clone, Copy, Debug)]
pub enum Builtin {
    PrintlnI64,
    Println,
    PrintF64,
    PrintString,
    ReadI64,
    ReadString,
    HasInput,
    Len,
    Abs,
    Min,
    Max,
    Pow,
}

#[derive(Clone, Debug)]
pub struct Closure {
    params: Vec<String>,
    body: Vec<Stmt>,
    env: Env,
}

#[derive(Clone, Debug)]
struct Env(Rc<EnvData>);

#[derive(Debug)]
struct EnvData {
    parent: Option<Env>,
    values: RefCell<HashMap<String, Value>>,
}

impl Env {
    fn new() -> Self {
        Self(Rc::new(EnvData {
            parent: None,
            values: RefCell::new(HashMap::new()),
        }))
    }

    fn child(&self) -> Self {
        Self(Rc::new(EnvData {
            parent: Some(self.clone()),
            values: RefCell::new(HashMap::new()),
        }))
    }

    fn define(&self, name: &str, value: Value) {
        self.0.values.borrow_mut().insert(name.to_string(), value);
    }

    fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.0.values.borrow().get(name).cloned() {
            return Some(value);
        }
        self.0.parent.as_ref().and_then(|parent| parent.get(name))
    }

    fn set(&self, name: &str, value: Value) -> Result<(), String> {
        if self.0.values.borrow().contains_key(name) {
            self.0.values.borrow_mut().insert(name.to_string(), value);
            return Ok(());
        }
        if let Some(parent) = &self.0.parent {
            return parent.set(name, value);
        }
        Err(format!("undefined variable '{name}'"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub structs: HashMap<String, StructDecl>,
    pub functions: HashMap<String, Function>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    VarDecl(String, Expr),
    Assign(Target, Expr),
    Return(Expr),
    If {
        condition: Expr,
        then_block: Vec<Stmt>,
        else_block: Vec<Stmt>,
    },
    While {
        condition: Expr,
        body: Vec<Stmt>,
    },
    ForTo {
        var_name: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    ForInRange {
        var_name: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    ForIn {
        var_name: String,
        iterable: Expr,
        body: Vec<Stmt>,
    },
    Break,
    Continue,
    Expr(Expr),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Ident(String),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Int(i64),
    Str(String),
    Bool(bool),
    ArrayLit(Vec<Expr>),
    StructConstruct {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    Ident(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Match(Box<Expr>, Vec<(Pattern, Expr)>),
    Closure {
        params: Vec<String>,
        body: Vec<Stmt>,
    },
    Ok(Box<Expr>),
    Err(Box<Expr>),
    Some(Box<Expr>),
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    Ok(String),
    Err(String),
    Some(String),
    None,
    Ident(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Fn,
    Struct,
    Return,
    If,
    Else,
    While,
    For,
    In,
    To,
    Match,
    Break,
    Continue,
    Ok,
    Err,
    Some,
    None,
    True,
    False,
    Ident(String),
    Int(i64),
    Str(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Eq,
    ColonEq,
    Arrow,
    FatArrow,
    Question,
    Dot,
    DotDot,
    Comma,
    Semi,
    Colon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Eof,
}

pub fn parse_program(src: &str) -> Result<Program, String> {
    let tokens = lex(src)?;
    Parser::new(tokens).parse_program()
}

pub fn execute_function(
    code: &str,
    function_name: &str,
    args: &[BenchmarkValue],
    problem_name: &str,
) -> Result<Value, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::new(program);
    let args = args
        .iter()
        .map(|value| runtime_value_from_problem(value, problem_name))
        .collect::<Result<Vec<_>, _>>()?;
    runtime.call_function(function_name, args)
}

pub fn execute_function_for_problem(
    code: &str,
    function_name: &str,
    args: &[BenchmarkValue],
    problem: &Problem,
) -> Result<Value, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::new(program);
    let args = args
        .iter()
        .map(|value| runtime_value_from_problem_meta(value, problem))
        .collect::<Result<Vec<_>, _>>()?;
    runtime.call_function(function_name, args)
}

/// Execute a single-string-argument function and return its raw `Value`.
/// Used by the generative-morphology path, where the function returns a string
/// (`fn pluralize(s: string) -> string`) rather than an i64.
pub fn execute_str_function(code: &str, function_name: &str, input: &str) -> Result<String, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::new(program);
    match runtime.call_function(function_name, vec![Value::Str(input.to_string())])? {
        Value::Str(s) => Ok(s),
        other => Err(format!("expected string return, got {other:?}")),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionResult {
    pub output: String,
    pub return_value: Option<i64>,
}

pub fn execute_program_with_input(
    code: &str,
    input: Vec<String>,
) -> Result<ExecutionResult, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::with_input(program, input);
    runtime.run_main()
}

pub fn execute_program(code: &str) -> Result<ExecutionResult, String> {
    execute_program_with_input(code, Vec::new())
}

pub fn verify_problem_code(problem: &Problem, code: &str) -> Result<(), String> {
    let fn_name = problem.function_name();
    for example in &problem.examples {
        let value = execute_function_for_problem(code, fn_name, &example.inputs, problem)?;
        let actual = expect_int(&value)?;
        if actual != example.expected {
            return Err(format!(
                "example failed for {}: expected {}, got {}",
                problem.name, example.expected, actual
            ));
        }
    }
    Ok(())
}

pub fn verify_problem_code_strict(problem: &Problem, code: &str) -> Result<(), String> {
    verify_problem_code_via_main(problem, code)?;
    let fn_name = problem.function_name();
    for example in generated_holdouts(problem) {
        let value = execute_function_for_problem(code, fn_name, &example.inputs, problem)?;
        let actual = expect_int(&value)?;
        if actual != example.expected {
            return Err(format!(
                "holdout failed for {}: inputs {:?}, expected {}, got {}",
                problem.name, example.inputs, example.expected, actual
            ));
        }
    }
    Ok(())
}

pub fn verify_problem_code_via_main(problem: &Problem, code: &str) -> Result<(), String> {
    let program = problem.wrap_program(code)?;
    let result = execute_program(&program)?;
    if result.output != problem.expected_stdout() {
        return Err(format!(
            "stdout mismatch for {}: expected {:?}, got {:?}",
            problem.name,
            problem.expected_stdout(),
            result.output
        ));
    }
    Ok(())
}

fn runtime_value_from_problem(value: &BenchmarkValue, problem_name: &str) -> Result<Value, String> {
    match value {
        BenchmarkValue::Int(v) => Ok(Value::Int(*v)),
        BenchmarkValue::Str(v) => Ok(Value::Str(v.clone())),
        BenchmarkValue::Array(values) => Ok(Value::Array(
            values.iter().copied().map(Value::Int).collect::<Vec<_>>(),
        )),
        BenchmarkValue::Pair(a, b) => {
            let (name, lhs, rhs) = if problem_name.starts_with("point_sum") {
                ("Point", "x", "y")
            } else if problem_name.starts_with("rectangle_area") {
                ("Rectangle", "width", "height")
            } else {
                return Err(format!(
                    "unsupported pair argument for problem {problem_name}"
                ));
            };
            let mut fields = HashMap::new();
            fields.insert(lhs.to_string(), Value::Int(*a));
            fields.insert(rhs.to_string(), Value::Int(*b));
            Ok(Value::Struct {
                name: name.to_string(),
                fields,
            })
        }
    }
}

fn runtime_value_from_problem_meta(
    value: &BenchmarkValue,
    problem: &Problem,
) -> Result<Value, String> {
    match value {
        BenchmarkValue::Pair(a, b) => {
            let (name, lhs, rhs) = if problem.signature.contains("Point") {
                ("Point", "x", "y")
            } else if problem.signature.contains("Rectangle") {
                ("Rectangle", "width", "height")
            } else {
                return Err(format!(
                    "unsupported pair argument for {} with signature {}",
                    problem.name, problem.signature
                ));
            };
            let mut fields = HashMap::new();
            fields.insert(lhs.to_string(), Value::Int(*a));
            fields.insert(rhs.to_string(), Value::Int(*b));
            Ok(Value::Struct {
                name: name.to_string(),
                fields,
            })
        }
        _ => runtime_value_from_problem(value, &problem.name),
    }
}

fn expect_int(value: &Value) -> Result<i64, String> {
    match value {
        Value::Int(v) => Ok(*v),
        _ => Err(format!("expected int result, got {:?}", value)),
    }
}

fn display_value(value: &Value) -> Result<String, String> {
    match value {
        Value::Int(v) => Ok(v.to_string()),
        Value::Bool(v) => Ok(if *v {
            "true".to_string()
        } else {
            "false".to_string()
        }),
        Value::Str(v) => Ok(v.clone()),
        Value::Unit => Ok(String::new()),
        other => Err(format!("cannot display {:?}", other)),
    }
}

fn lex(src: &str) -> Result<Vec<Token>, String> {
    let mut chars = src.chars().peekable();
    let mut out = Vec::new();

    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        if ch == '/' {
            chars.next();
            if matches!(chars.peek(), Some('/')) {
                while let Some(next) = chars.next() {
                    if next == '\n' {
                        break;
                    }
                }
                continue;
            }
            out.push(Token::Slash);
            continue;
        }

        if ch.is_ascii_digit() {
            let mut value = String::new();
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    value.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            out.push(Token::Int(
                value
                    .parse::<i64>()
                    .map_err(|err| format!("invalid int literal {value}: {err}"))?,
            ));
            continue;
        }

        if ch == '"' {
            chars.next();
            let mut value = String::new();
            while let Some(next) = chars.next() {
                match next {
                    '"' => break,
                    '\\' => {
                        let esc = chars
                            .next()
                            .ok_or_else(|| "unterminated escape".to_string())?;
                        let mapped = match esc {
                            'n' => '\n',
                            't' => '\t',
                            '"' => '"',
                            '\\' => '\\',
                            other => other,
                        };
                        value.push(mapped);
                    }
                    other => value.push(other),
                }
            }
            out.push(Token::Str(value));
            continue;
        }

        if ch.is_ascii_alphabetic() || ch == '_' {
            let mut ident = String::new();
            while let Some(&next) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' {
                    ident.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            let token = match ident.as_str() {
                "fn" => Token::Fn,
                "struct" => Token::Struct,
                "return" => Token::Return,
                "if" => Token::If,
                "else" => Token::Else,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "to" => Token::To,
                "match" => Token::Match,
                "break" => Token::Break,
                "continue" => Token::Continue,
                "ok" => Token::Ok,
                "err" => Token::Err,
                "some" => Token::Some,
                "none" => Token::None,
                "true" => Token::True,
                "false" => Token::False,
                _ => Token::Ident(ident),
            };
            out.push(token);
            continue;
        }

        match ch {
            '+' => {
                chars.next();
                out.push(Token::Plus);
            }
            '-' => {
                chars.next();
                if matches!(chars.peek(), Some('>')) {
                    chars.next();
                    out.push(Token::Arrow);
                } else {
                    out.push(Token::Minus);
                }
            }
            '*' => {
                chars.next();
                out.push(Token::Star);
            }
            '%' => {
                chars.next();
                out.push(Token::Percent);
            }
            '=' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::EqEq);
                } else if matches!(chars.peek(), Some('>')) {
                    chars.next();
                    out.push(Token::FatArrow);
                } else {
                    out.push(Token::Eq);
                }
            }
            '!' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::BangEq);
                } else {
                    return Err("unexpected !".to_string());
                }
            }
            '<' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::LtEq);
                } else {
                    out.push(Token::Lt);
                }
            }
            '>' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::GtEq);
                } else {
                    out.push(Token::Gt);
                }
            }
            ':' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::ColonEq);
                } else {
                    out.push(Token::Colon);
                }
            }
            '?' => {
                chars.next();
                out.push(Token::Question);
            }
            '.' => {
                chars.next();
                if matches!(chars.peek(), Some('.')) {
                    chars.next();
                    out.push(Token::DotDot);
                } else {
                    out.push(Token::Dot);
                }
            }
            ',' => {
                chars.next();
                out.push(Token::Comma);
            }
            ';' => {
                chars.next();
                out.push(Token::Semi);
            }
            '(' => {
                chars.next();
                out.push(Token::LParen);
            }
            ')' => {
                chars.next();
                out.push(Token::RParen);
            }
            '{' => {
                chars.next();
                out.push(Token::LBrace);
            }
            '}' => {
                chars.next();
                out.push(Token::RBrace);
            }
            '[' => {
                chars.next();
                out.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                out.push(Token::RBracket);
            }
            _ => return Err(format!("unexpected character {ch:?}")),
        }
    }

    out.push(Token::Eof);
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn parse_program(mut self) -> Result<Program, String> {
        let mut structs = HashMap::new();
        let mut functions = HashMap::new();

        while !self.at(&Token::Eof) {
            if self.at(&Token::Struct) {
                let decl = self.parse_struct_decl()?;
                structs.insert(decl.name.clone(), decl);
            } else if self.at(&Token::Fn) {
                let func = self.parse_function_decl()?;
                functions.insert(func.name.clone(), func);
            } else {
                return Err(format!(
                    "expected top-level declaration, got {:?}",
                    self.current()
                ));
            }
        }

        Ok(Program { structs, functions })
    }

    fn parse_struct_decl(&mut self) -> Result<StructDecl, String> {
        self.expect(&Token::Struct)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(&Token::RBrace) {
            fields.push(self.expect_ident()?);
            self.expect(&Token::Colon)?;
            self.skip_type_until(&[Token::Comma, Token::RBrace])?;
            if self.at(&Token::Comma) {
                self.bump();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(StructDecl { name, fields })
    }

    fn parse_function_decl(&mut self) -> Result<Function, String> {
        self.expect(&Token::Fn)?;
        let name = self.expect_ident()?;
        let params = self.parse_params()?;
        if self.at(&Token::Arrow) {
            self.bump();
            self.skip_type_until(&[Token::LBrace])?;
        }
        let body = self.parse_block()?;
        Ok(Function { name, params, body })
    }

    fn parse_params(&mut self) -> Result<Vec<String>, String> {
        self.expect(&Token::LParen)?;
        let mut params = Vec::new();
        while !self.at(&Token::RParen) {
            let name = self.expect_ident()?;
            params.push(name);
            if self.at(&Token::Colon) {
                self.bump();
                self.skip_type_until(&[Token::Comma, Token::RParen])?;
            }
            if self.at(&Token::Comma) {
                self.bump();
            }
        }
        self.expect(&Token::RParen)?;
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(&Token::RBrace) {
            if self.at(&Token::Semi) {
                self.bump();
                continue;
            }
            stmts.push(self.parse_stmt()?);
            while self.at(&Token::Semi) {
                self.bump();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        if self.at(&Token::Return) {
            self.bump();
            return Ok(Stmt::Return(self.parse_expr()?));
        }
        if self.at(&Token::If) {
            return self.parse_if_stmt();
        }
        if self.at(&Token::While) {
            self.bump();
            let condition = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::While { condition, body });
        }
        if self.at(&Token::For) {
            return self.parse_for_stmt();
        }
        if self.at(&Token::Break) {
            self.bump();
            return Ok(Stmt::Break);
        }
        if self.at(&Token::Continue) {
            self.bump();
            return Ok(Stmt::Continue);
        }

        if let Token::Ident(_) = self.current() {
            if matches!(self.peek(), Token::ColonEq) {
                let name = self.expect_ident()?;
                self.expect(&Token::ColonEq)?;
                return Ok(Stmt::VarDecl(name, self.parse_expr()?));
            }
            if matches!(self.peek(), Token::Colon) {
                let name = self.expect_ident()?;
                self.expect(&Token::Colon)?;
                self.skip_type_until(&[Token::Eq])?;
                self.expect(&Token::Eq)?;
                return Ok(Stmt::VarDecl(name, self.parse_expr()?));
            }
        }

        let expr = self.parse_expr()?;
        if self.at(&Token::Eq) {
            self.bump();
            let value = self.parse_expr()?;
            let target = expr_to_target(expr)?;
            return Ok(Stmt::Assign(target, value));
        }
        Ok(Stmt::Expr(expr))
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, String> {
        self.expect(&Token::For)?;
        let var_name = self.expect_ident()?;

        if self.at(&Token::ColonEq) {
            self.bump();
            let start = self.parse_expr()?;
            self.expect(&Token::To)?;
            let end = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::ForTo {
                var_name,
                start,
                end,
                body,
            });
        }

        self.expect(&Token::In)?;
        let start_or_iterable = self.parse_expr()?;
        if self.at(&Token::DotDot) {
            self.bump();
            let end = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::ForInRange {
                var_name,
                start: start_or_iterable,
                end,
                body,
            });
        }
        let body = self.parse_block()?;
        Ok(Stmt::ForIn {
            var_name,
            iterable: start_or_iterable,
            body,
        })
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, String> {
        self.expect(&Token::If)?;
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.at(&Token::Else) {
            self.bump();
            if self.at(&Token::If) {
                vec![self.parse_if_stmt()?]
            } else {
                self.parse_block()?
            }
        } else {
            Vec::new()
        };
        Ok(Stmt::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_equality()
    }

    fn parse_equality(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_comparison()?;
        loop {
            let op = if self.at(&Token::EqEq) {
                Some(BinaryOp::Eq)
            } else if self.at(&Token::BangEq) {
                Some(BinaryOp::Ne)
            } else {
                None
            };
            let Some(op) = op else { break };
            self.bump();
            let rhs = self.parse_comparison()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_additive()?;
        loop {
            let op = if self.at(&Token::Lt) {
                Some(BinaryOp::Lt)
            } else if self.at(&Token::Gt) {
                Some(BinaryOp::Gt)
            } else if self.at(&Token::LtEq) {
                Some(BinaryOp::Le)
            } else if self.at(&Token::GtEq) {
                Some(BinaryOp::Ge)
            } else {
                None
            };
            let Some(op) = op else { break };
            self.bump();
            let rhs = self.parse_additive()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_additive(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_multiplicative()?;
        loop {
            let op = if self.at(&Token::Plus) {
                Some(BinaryOp::Add)
            } else if self.at(&Token::Minus) {
                Some(BinaryOp::Sub)
            } else {
                None
            };
            let Some(op) = op else { break };
            self.bump();
            let rhs = self.parse_multiplicative()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_unary()?;
        loop {
            let op = if self.at(&Token::Star) {
                Some(BinaryOp::Mul)
            } else if self.at(&Token::Slash) {
                Some(BinaryOp::Div)
            } else if self.at(&Token::Percent) {
                Some(BinaryOp::Mod)
            } else {
                None
            };
            let Some(op) = op else { break };
            self.bump();
            let rhs = self.parse_unary()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if self.at(&Token::Minus) {
            self.bump();
            return Ok(Expr::Unary(UnaryOp::Neg, Box::new(self.parse_unary()?)));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.at(&Token::LParen) {
                let args = self.parse_call_args()?;
                expr = Expr::Call(Box::new(expr), args);
                continue;
            }
            if self.at(&Token::Dot) {
                self.bump();
                let field = self.expect_ident()?;
                expr = Expr::Field(Box::new(expr), field);
                continue;
            }
            if self.at(&Token::LBracket) {
                self.bump();
                let index = self.parse_expr()?;
                self.expect(&Token::RBracket)?;
                expr = Expr::Index(Box::new(expr), Box::new(index));
                continue;
            }
            break;
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        match self.current() {
            Token::Int(v) => {
                let value = *v;
                self.bump();
                Ok(Expr::Int(value))
            }
            Token::Str(v) => {
                let value = v.clone();
                self.bump();
                Ok(Expr::Str(value))
            }
            Token::True => {
                self.bump();
                Ok(Expr::Bool(true))
            }
            Token::False => {
                self.bump();
                Ok(Expr::Bool(false))
            }
            Token::None => {
                self.bump();
                Ok(Expr::None)
            }
            Token::LBracket => {
                self.bump();
                let mut elements = Vec::new();
                while !self.at(&Token::RBracket) {
                    elements.push(self.parse_expr()?);
                    if self.at(&Token::Comma) {
                        self.bump();
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(Expr::ArrayLit(elements))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.bump();
                if self.at(&Token::LBrace)
                    && name
                        .chars()
                        .next()
                        .map(|ch| ch.is_ascii_uppercase())
                        .unwrap_or(false)
                {
                    self.bump();
                    let mut fields = Vec::new();
                    while !self.at(&Token::RBrace) {
                        let field_name = self.expect_ident()?;
                        self.expect(&Token::Colon)?;
                        let field_expr = self.parse_expr()?;
                        fields.push((field_name, field_expr));
                        if self.at(&Token::Comma) {
                            self.bump();
                        }
                    }
                    self.expect(&Token::RBrace)?;
                    Ok(Expr::StructConstruct { name, fields })
                } else {
                    Ok(Expr::Ident(name))
                }
            }
            Token::Ok => {
                self.bump();
                self.expect(&Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Ok(Box::new(expr)))
            }
            Token::Err => {
                self.bump();
                self.expect(&Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Err(Box::new(expr)))
            }
            Token::Some => {
                self.bump();
                self.expect(&Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Some(Box::new(expr)))
            }
            Token::LParen => {
                self.bump();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::Match => self.parse_match_expr(),
            Token::Fn => self.parse_closure(),
            _ => Err(format!(
                "unexpected token in expression: {:?}",
                self.current()
            )),
        }
    }

    fn parse_match_expr(&mut self) -> Result<Expr, String> {
        self.expect(&Token::Match)?;
        let subject = self.parse_expr()?;
        self.expect(&Token::LBrace)?;
        let mut arms = Vec::new();
        while !self.at(&Token::RBrace) {
            let pattern = self.parse_pattern()?;
            self.expect(&Token::FatArrow)?;
            let expr = self.parse_expr()?;
            arms.push((pattern, expr));
            if self.at(&Token::Comma) {
                self.bump();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(Expr::Match(Box::new(subject), arms))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        match self.current() {
            Token::Ok => {
                self.bump();
                self.expect(&Token::LParen)?;
                let name = self.expect_ident()?;
                self.expect(&Token::RParen)?;
                Ok(Pattern::Ok(name))
            }
            Token::Err => {
                self.bump();
                self.expect(&Token::LParen)?;
                let name = self.expect_ident()?;
                self.expect(&Token::RParen)?;
                Ok(Pattern::Err(name))
            }
            Token::Some => {
                self.bump();
                self.expect(&Token::LParen)?;
                let name = self.expect_ident()?;
                self.expect(&Token::RParen)?;
                Ok(Pattern::Some(name))
            }
            Token::None => {
                self.bump();
                Ok(Pattern::None)
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.bump();
                Ok(Pattern::Ident(name))
            }
            _ => Err(format!("unexpected token in pattern: {:?}", self.current())),
        }
    }

    fn parse_closure(&mut self) -> Result<Expr, String> {
        self.expect(&Token::Fn)?;
        let params = self.parse_params()?;
        if self.at(&Token::Arrow) {
            self.bump();
            self.skip_type_until(&[Token::LBrace])?;
        }
        let body = self.parse_block()?;
        Ok(Expr::Closure { params, body })
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, String> {
        self.expect(&Token::LParen)?;
        let mut args = Vec::new();
        while !self.at(&Token::RParen) {
            args.push(self.parse_expr()?);
            if self.at(&Token::Comma) {
                self.bump();
            }
        }
        self.expect(&Token::RParen)?;
        Ok(args)
    }

    fn skip_type_until(&mut self, terminators: &[Token]) -> Result<(), String> {
        let mut depth_angle = 0usize;
        let mut depth_bracket = 0usize;
        loop {
            if depth_angle == 0 && depth_bracket == 0 && terminators.iter().any(|tok| self.at(tok))
            {
                return Ok(());
            }
            match self.current() {
                Token::Lt => depth_angle += 1,
                Token::Gt => {
                    depth_angle = depth_angle.saturating_sub(1);
                }
                Token::LBracket => depth_bracket += 1,
                Token::RBracket => {
                    depth_bracket = depth_bracket.saturating_sub(1);
                }
                Token::Question => {}
                Token::Ident(_) => {}
                _ => {}
            }
            if self.at(&Token::Eof) {
                return Err("unexpected EOF while skipping type".to_string());
            }
            self.bump();
        }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos + 1).unwrap_or(&Token::Eof)
    }

    fn at(&self, token: &Token) -> bool {
        std::mem::discriminant(self.current()) == std::mem::discriminant(token)
    }

    fn bump(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, token: &Token) -> Result<(), String> {
        if self.at(token) {
            self.bump();
            Ok(())
        } else {
            Err(format!("expected {:?}, got {:?}", token, self.current()))
        }
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.current() {
            Token::Ident(name) => {
                let name = name.clone();
                self.bump();
                Ok(name)
            }
            other => Err(format!("expected identifier, got {:?}", other)),
        }
    }
}

fn expr_to_target(expr: Expr) -> Result<Target, String> {
    match expr {
        Expr::Ident(name) => Ok(Target::Ident(name)),
        Expr::Field(base, field) => Ok(Target::Field(base, field)),
        Expr::Index(base, index) => Ok(Target::Index(base, index)),
        _ => Err(format!("invalid assignment target: {:?}", expr)),
    }
}

#[derive(Clone, Debug)]
struct Runtime {
    program: Program,
    global: Env,
    output: RefCell<Vec<String>>,
    input: RefCell<VecDeque<String>>,
}

impl Runtime {
    fn new(program: Program) -> Self {
        Self::with_input(program, Vec::new())
    }

    fn with_input(program: Program, input: Vec<String>) -> Self {
        let global = Env::new();
        for name in program.functions.keys() {
            global.define(name, Value::Function(name.clone()));
        }
        global.define("println_i64", Value::Builtin(Builtin::PrintlnI64));
        global.define("println", Value::Builtin(Builtin::Println));
        global.define("print_f64", Value::Builtin(Builtin::PrintF64));
        global.define("print_string", Value::Builtin(Builtin::PrintString));
        global.define("read_i64", Value::Builtin(Builtin::ReadI64));
        global.define("read_string", Value::Builtin(Builtin::ReadString));
        global.define("read_line", Value::Builtin(Builtin::ReadString));
        global.define("has_input", Value::Builtin(Builtin::HasInput));
        global.define("len", Value::Builtin(Builtin::Len));
        global.define("abs", Value::Builtin(Builtin::Abs));
        global.define("min", Value::Builtin(Builtin::Min));
        global.define("max", Value::Builtin(Builtin::Max));
        global.define("pow", Value::Builtin(Builtin::Pow));
        Self {
            program,
            global,
            output: RefCell::new(Vec::new()),
            input: RefCell::new(input.into_iter().collect()),
        }
    }

    fn run_main(&self) -> Result<ExecutionResult, String> {
        let value = self.call_function("main", Vec::new())?;
        let return_value = match value {
            Value::Int(v) => Some(v),
            Value::Unit => None,
            _ => None,
        };
        Ok(ExecutionResult {
            output: self.output.borrow().join("\n"),
            return_value,
        })
    }

    fn call_function(&self, function_name: &str, args: Vec<Value>) -> Result<Value, String> {
        let function = self
            .program
            .functions
            .get(function_name)
            .cloned()
            .ok_or_else(|| format!("unknown function {function_name}"))?;
        self.call_decl(&function, args, self.global.clone())
    }

    fn call_decl(&self, function: &Function, args: Vec<Value>, env: Env) -> Result<Value, String> {
        let local = env.child();
        for (idx, param) in function.params.iter().enumerate() {
            let value = args.get(idx).cloned().unwrap_or(Value::Unit);
            local.define(param, value);
        }
        match self.exec_block(&function.body, local)? {
            Control::Return(value) => Ok(value),
            Control::Next => Ok(Value::Unit),
            Control::Break => Err("break outside loop".to_string()),
            Control::Continue => Err("continue outside loop".to_string()),
        }
    }

    fn exec_block(&self, stmts: &[Stmt], env: Env) -> Result<Control, String> {
        for stmt in stmts {
            match self.exec_stmt(stmt, env.clone())? {
                Control::Next => {}
                signal => return Ok(signal),
            }
        }
        Ok(Control::Next)
    }

    fn exec_stmt(&self, stmt: &Stmt, env: Env) -> Result<Control, String> {
        match stmt {
            Stmt::VarDecl(name, expr) => {
                let value = self.eval_expr(expr, env.clone())?;
                env.define(name, value);
                Ok(Control::Next)
            }
            Stmt::Assign(target, expr) => {
                let value = self.eval_expr(expr, env.clone())?;
                self.assign(target, value, env)?;
                Ok(Control::Next)
            }
            Stmt::Return(expr) => Ok(Control::Return(self.eval_expr(expr, env)?)),
            Stmt::If {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.eval_expr(condition, env.clone())?;
                if truthy(&cond) {
                    self.exec_block(then_block, env.child())
                } else {
                    self.exec_block(else_block, env.child())
                }
            }
            Stmt::While { condition, body } => {
                let mut iters = 0usize;
                while truthy(&self.eval_expr(condition, env.clone())?) {
                    iters += 1;
                    if iters > 100_000 {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    match self.exec_block(body, env.child())? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break => break,
                        Control::Continue => continue,
                    }
                }
                Ok(Control::Next)
            }
            Stmt::ForTo {
                var_name,
                start,
                end,
                body,
            } => {
                let start = expect_int(&self.eval_expr(start, env.clone())?)?;
                let end = expect_int(&self.eval_expr(end, env.clone())?)?;
                for item in start..end {
                    let scope = env.child();
                    scope.define(var_name, Value::Int(item));
                    match self.exec_block(body, scope)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break => break,
                        Control::Continue => continue,
                    }
                }
                Ok(Control::Next)
            }
            Stmt::ForInRange {
                var_name,
                start,
                end,
                body,
            } => {
                let start = expect_int(&self.eval_expr(start, env.clone())?)?;
                let end = expect_int(&self.eval_expr(end, env.clone())?)?;
                for item in start..end {
                    let scope = env.child();
                    scope.define(var_name, Value::Int(item));
                    match self.exec_block(body, scope)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break => break,
                        Control::Continue => continue,
                    }
                }
                Ok(Control::Next)
            }
            Stmt::ForIn {
                var_name,
                iterable,
                body,
            } => {
                let iterable = self.eval_expr(iterable, env.clone())?;
                let items = match iterable {
                    Value::Array(items) => items,
                    other => return Err(format!("cannot iterate over {:?}", other)),
                };
                for item in items {
                    let scope = env.child();
                    scope.define(var_name, item);
                    match self.exec_block(body, scope)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break => break,
                        Control::Continue => continue,
                    }
                }
                Ok(Control::Next)
            }
            Stmt::Break => Ok(Control::Break),
            Stmt::Continue => Ok(Control::Continue),
            Stmt::Expr(expr) => {
                self.eval_expr(expr, env)?;
                Ok(Control::Next)
            }
        }
    }

    fn assign(&self, target: &Target, value: Value, env: Env) -> Result<(), String> {
        match target {
            Target::Ident(name) => env.set(name, value),
            Target::Field(base, field) => {
                let mut base_value = self.eval_expr(base, env.clone())?;
                match &mut base_value {
                    Value::Struct { fields, .. } => {
                        fields.insert(field.clone(), value);
                    }
                    _ => return Err(format!("cannot assign field {} on {:?}", field, base_value)),
                }
                if let Expr::Ident(name) = &**base {
                    env.set(name, base_value)
                } else {
                    Err("field assignment requires identifier base".to_string())
                }
            }
            Target::Index(base, index) => {
                let mut base_value = self.eval_expr(base, env.clone())?;
                let idx = expect_int(&self.eval_expr(index, env.clone())?)? as usize;
                match &mut base_value {
                    Value::Array(items) => {
                        if idx >= items.len() {
                            return Err(format!("index {} out of bounds", idx));
                        }
                        items[idx] = value;
                    }
                    _ => return Err(format!("cannot assign index on {:?}", base_value)),
                }
                if let Expr::Ident(name) = &**base {
                    env.set(name, base_value)
                } else {
                    Err("index assignment requires identifier base".to_string())
                }
            }
        }
    }

    fn eval_expr(&self, expr: &Expr, env: Env) -> Result<Value, String> {
        match expr {
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Str(v) => Ok(Value::Str(v.clone())),
            Expr::Bool(v) => Ok(Value::Bool(*v)),
            Expr::ArrayLit(values) => Ok(Value::Array(
                values
                    .iter()
                    .map(|value| self.eval_expr(value, env.clone()))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Expr::StructConstruct { name, fields } => Ok(Value::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|(field, expr)| Ok((field.clone(), self.eval_expr(expr, env.clone())?)))
                    .collect::<Result<HashMap<_, _>, String>>()?,
            }),
            Expr::Ident(name) => env
                .get(name)
                .ok_or_else(|| format!("undefined variable '{name}'")),
            Expr::Unary(op, expr) => {
                let value = self.eval_expr(expr, env)?;
                match op {
                    UnaryOp::Neg => Ok(Value::Int(-expect_int(&value)?)),
                    UnaryOp::Not => Ok(Value::Bool(!truthy(&value))),
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                let lhs = self.eval_expr(lhs, env.clone())?;
                let rhs = self.eval_expr(rhs, env)?;
                self.eval_binary(lhs, *op, rhs)
            }
            Expr::Call(callee, args) => {
                if let Expr::Field(base, method) = &**callee {
                    let args = args
                        .iter()
                        .map(|arg| self.eval_expr(arg, env.clone()))
                        .collect::<Result<Vec<_>, _>>()?;
                    return self.call_method_on_expr(base, method, args, env);
                }
                let callee = self.eval_expr(callee, env.clone())?;
                let args = args
                    .iter()
                    .map(|arg| self.eval_expr(arg, env.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                self.call_value(callee, args)
            }
            Expr::Field(base, field) => {
                let base = self.eval_expr(base, env)?;
                match base {
                    Value::Struct { fields, .. } => fields
                        .get(field)
                        .cloned()
                        .ok_or_else(|| format!("missing struct field {field}")),
                    Value::Array(items) if field == "len" => Ok(Value::Int(items.len() as i64)),
                    Value::Str(value) if field == "len" => {
                        Ok(Value::Int(value.chars().count() as i64))
                    }
                    other => Err(format!("cannot access field {field} on {:?}", other)),
                }
            }
            Expr::Index(base, index) => {
                let base = self.eval_expr(base, env.clone())?;
                let index = expect_int(&self.eval_expr(index, env)?)? as usize;
                match base {
                    Value::Array(items) => items
                        .get(index)
                        .cloned()
                        .ok_or_else(|| format!("index {} out of bounds", index)),
                    Value::Str(value) => value
                        .chars()
                        .nth(index)
                        .map(|ch| Value::Str(ch.to_string()))
                        .ok_or_else(|| format!("index {} out of bounds", index)),
                    other => Err(format!("cannot index {:?}", other)),
                }
            }
            Expr::Match(subject, arms) => {
                let subject = self.eval_expr(subject, env.clone())?;
                for (pattern, body) in arms {
                    let scope = env.child();
                    if pattern_matches(pattern, &subject, &scope) {
                        return self.eval_expr(body, scope);
                    }
                }
                Err("non-exhaustive match".to_string())
            }
            Expr::Closure { params, body } => Ok(Value::Closure(Closure {
                params: params.clone(),
                body: body.clone(),
                env,
            })),
            Expr::Ok(expr) => Ok(Value::Result {
                is_ok: true,
                value: Box::new(self.eval_expr(expr, env)?),
            }),
            Expr::Err(expr) => Ok(Value::Result {
                is_ok: false,
                value: Box::new(self.eval_expr(expr, env)?),
            }),
            Expr::Some(expr) => Ok(Value::Optional {
                is_some: true,
                value: Box::new(self.eval_expr(expr, env)?),
            }),
            Expr::None => Ok(Value::Optional {
                is_some: false,
                value: Box::new(Value::Unit),
            }),
        }
    }

    fn eval_binary(&self, lhs: Value, op: BinaryOp, rhs: Value) -> Result<Value, String> {
        match op {
            BinaryOp::Add => match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => a
                    .checked_add(b)
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in +".to_string()),
                (Value::Str(a), Value::Str(b)) => Ok(Value::Str(a + &b)),
                (a, b) => Err(format!("unsupported + operands {:?} and {:?}", a, b)),
            },
            BinaryOp::Sub => {
                let l = expect_int(&lhs)?;
                let r = expect_int(&rhs)?;
                l.checked_sub(r)
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in -".to_string())
            }
            BinaryOp::Mul => {
                let l = expect_int(&lhs)?;
                let r = expect_int(&rhs)?;
                l.checked_mul(r)
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in *".to_string())
            }
            BinaryOp::Div => {
                let lhs = expect_int(&lhs)?;
                let rhs = expect_int(&rhs)?;
                if rhs == 0 {
                    return Err("division by zero".to_string());
                }
                Ok(Value::Int(lhs / rhs))
            }
            BinaryOp::Mod => {
                let lhs = expect_int(&lhs)?;
                let rhs = expect_int(&rhs)?;
                if rhs == 0 {
                    return Err("modulo by zero".to_string());
                }
                Ok(Value::Int(lhs % rhs))
            }
            BinaryOp::Eq => Ok(Value::Bool(value_eq(&lhs, &rhs))),
            BinaryOp::Ne => Ok(Value::Bool(!value_eq(&lhs, &rhs))),
            BinaryOp::Lt => Ok(Value::Bool(expect_int(&lhs)? < expect_int(&rhs)?)),
            BinaryOp::Gt => Ok(Value::Bool(expect_int(&lhs)? > expect_int(&rhs)?)),
            BinaryOp::Le => Ok(Value::Bool(expect_int(&lhs)? <= expect_int(&rhs)?)),
            BinaryOp::Ge => Ok(Value::Bool(expect_int(&lhs)? >= expect_int(&rhs)?)),
        }
    }

    fn call_value(&self, callee: Value, args: Vec<Value>) -> Result<Value, String> {
        match callee {
            Value::Function(name) => {
                let function = self
                    .program
                    .functions
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| format!("unknown function {name}"))?;
                self.call_decl(&function, args, self.global.clone())
            }
            Value::Closure(closure) => {
                let scope = closure.env.child();
                for (idx, param) in closure.params.iter().enumerate() {
                    scope.define(param, args.get(idx).cloned().unwrap_or(Value::Unit));
                }
                match self.exec_block(&closure.body, scope.clone())? {
                    Control::Return(value) => Ok(value),
                    Control::Next => {
                        if let Some(Stmt::Expr(expr)) = closure.body.last() {
                            self.eval_expr(expr, scope)
                        } else {
                            Ok(Value::Unit)
                        }
                    }
                    Control::Break => Err("break outside loop".to_string()),
                    Control::Continue => Err("continue outside loop".to_string()),
                }
            }
            Value::Builtin(builtin) => self.call_builtin(builtin, args),
            other => Err(format!("cannot call {:?}", other)),
        }
    }

    fn call_method(
        &self,
        base: &mut Value,
        method: &str,
        args: Vec<Value>,
    ) -> Result<Value, String> {
        match base {
            Value::Array(items) => match method {
                "push" => {
                    let value = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "push requires one argument".to_string())?;
                    items.push(value);
                    Ok(Value::Unit)
                }
                "pop" => items.pop().ok_or_else(|| "pop on empty array".to_string()),
                "map" => {
                    let mapper = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "map requires closure".to_string())?;
                    let mut out = Vec::new();
                    for item in items {
                        out.push(self.call_value(mapper.clone(), vec![item.clone()])?);
                    }
                    Ok(Value::Array(out))
                }
                "filter" => {
                    let predicate = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "filter requires closure".to_string())?;
                    let mut out = Vec::new();
                    for item in items.iter().cloned() {
                        if truthy(&self.call_value(predicate.clone(), vec![item.clone()])?) {
                            out.push(item);
                        }
                    }
                    Ok(Value::Array(out))
                }
                "sort" => {
                    items.sort_by(|lhs, rhs| {
                        let lhs = expect_int(lhs).unwrap_or_default();
                        let rhs = expect_int(rhs).unwrap_or_default();
                        lhs.cmp(&rhs)
                    });
                    Ok(Value::Array(items.clone()))
                }
                "join" => {
                    let sep = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "join requires separator".to_string())?;
                    let sep = match sep {
                        Value::Str(v) => v,
                        other => {
                            return Err(format!("join separator must be string, got {:?}", other))
                        }
                    };
                    let parts = items
                        .iter()
                        .map(display_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(Value::Str(parts.join(&sep)))
                }
                "len" => Ok(Value::Int(items.len() as i64)),
                _ => Err(format!("unknown array method {method}")),
            },
            Value::Str(value) => match method {
                "trim" => Ok(Value::Str(value.trim().to_string())),
                "upper" => Ok(Value::Str(value.to_uppercase())),
                "lower" => Ok(Value::Str(value.to_lowercase())),
                "split" => {
                    let sep = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "split requires separator".to_string())?;
                    let sep = match sep {
                        Value::Str(v) => v,
                        other => {
                            return Err(format!("split separator must be string, got {:?}", other))
                        }
                    };
                    if sep.is_empty() {
                        return Ok(Value::Array(
                            value.chars().map(|ch| Value::Str(ch.to_string())).collect(),
                        ));
                    }
                    Ok(Value::Array(
                        value
                            .split(&sep)
                            .map(|part| Value::Str(part.to_string()))
                            .collect(),
                    ))
                }
                "contains" => {
                    let needle = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "contains requires needle".to_string())?;
                    let needle = match needle {
                        Value::Str(v) => v,
                        other => {
                            return Err(format!("contains needle must be string, got {:?}", other))
                        }
                    };
                    Ok(Value::Bool(value.contains(&needle)))
                }
                "starts_with" => {
                    let prefix = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "starts_with requires prefix".to_string())?;
                    let prefix = match prefix {
                        Value::Str(v) => v,
                        other => {
                            return Err(format!(
                                "starts_with prefix must be string, got {:?}",
                                other
                            ))
                        }
                    };
                    Ok(Value::Bool(value.starts_with(&prefix)))
                }
                "ends_with" => {
                    let suffix = args
                        .into_iter()
                        .next()
                        .ok_or_else(|| "ends_with requires suffix".to_string())?;
                    let suffix = match suffix {
                        Value::Str(v) => v,
                        other => {
                            return Err(format!("ends_with suffix must be string, got {:?}", other))
                        }
                    };
                    Ok(Value::Bool(value.ends_with(&suffix)))
                }
                "replace" => {
                    if args.len() != 2 {
                        return Err("replace requires old and new strings".to_string());
                    }
                    let old = match &args[0] {
                        Value::Str(v) => v.clone(),
                        other => {
                            return Err(format!("replace old must be string, got {:?}", other))
                        }
                    };
                    let new = match &args[1] {
                        Value::Str(v) => v.clone(),
                        other => {
                            return Err(format!("replace new must be string, got {:?}", other))
                        }
                    };
                    Ok(Value::Str(value.replace(&old, &new)))
                }
                "reverse" => {
                    if !args.is_empty() {
                        return Err("reverse takes no arguments".to_string());
                    }
                    Ok(Value::Str(value.chars().rev().collect()))
                }
                "slice" => {
                    if args.len() != 2 {
                        return Err("slice requires start and end".to_string());
                    }
                    let chars: Vec<char> = value.chars().collect();
                    let n = chars.len() as i64;
                    let clamp = |i: i64| -> usize {
                        if i < 0 {
                            0
                        } else if i > n {
                            chars.len()
                        } else {
                            i as usize
                        }
                    };
                    let start = clamp(expect_int(&args[0])?);
                    let end = clamp(expect_int(&args[1])?);
                    if start >= end {
                        Ok(Value::Str(String::new()))
                    } else {
                        Ok(Value::Str(chars[start..end].iter().collect()))
                    }
                }
                _ => Err(format!("unknown string method {method}")),
            },
            other => Err(format!("cannot call method {method} on {:?}", other)),
        }
    }

    fn call_method_on_expr(
        &self,
        base_expr: &Expr,
        method: &str,
        args: Vec<Value>,
        env: Env,
    ) -> Result<Value, String> {
        let mut base_value = self.eval_expr(base_expr, env.clone())?;
        let result = self.call_method(&mut base_value, method, args)?;
        if matches!(method, "push" | "pop" | "sort") {
            if let Expr::Ident(name) = base_expr {
                env.set(name, base_value)?;
            }
        }
        Ok(result)
    }

    fn call_builtin(&self, builtin: Builtin, args: Vec<Value>) -> Result<Value, String> {
        match builtin {
            Builtin::PrintlnI64 => {
                let value = args
                    .first()
                    .ok_or_else(|| "println_i64 requires one argument".to_string())?;
                self.output
                    .borrow_mut()
                    .push(expect_int(value)?.to_string());
                Ok(Value::Unit)
            }
            Builtin::Println => {
                let line = args
                    .iter()
                    .map(display_value)
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");
                self.output.borrow_mut().push(line);
                Ok(Value::Unit)
            }
            Builtin::PrintF64 => {
                let value = args
                    .first()
                    .ok_or_else(|| "print_f64 requires one argument".to_string())?;
                let line = match value {
                    Value::Int(v) => format!("{:.7}", *v as f64),
                    _ => return Err(format!("print_f64 expected numeric value, got {:?}", value)),
                };
                self.output.borrow_mut().push(line);
                Ok(Value::Unit)
            }
            Builtin::PrintString => {
                let value = args
                    .first()
                    .ok_or_else(|| "print_string requires one argument".to_string())?;
                let text = display_value(value)?;
                let mut output = self.output.borrow_mut();
                if let Some(last) = output.last_mut() {
                    last.push_str(&text);
                } else {
                    output.push(text);
                }
                Ok(Value::Unit)
            }
            Builtin::ReadI64 => {
                if !args.is_empty() {
                    return Err("read_i64 takes no arguments".to_string());
                }
                let raw = {
                    let mut input = self.input.borrow_mut();
                    input
                        .pop_front()
                        .ok_or_else(|| "read_i64: no input available".to_string())?
                };
                let value = raw
                    .trim()
                    .parse::<i64>()
                    .map_err(|err| format!("read_i64: cannot parse {:?} as i64: {err}", raw))?;
                Ok(Value::Int(value))
            }
            Builtin::ReadString => {
                if !args.is_empty() {
                    return Err("read_string takes no arguments".to_string());
                }
                let raw = {
                    let mut input = self.input.borrow_mut();
                    input
                        .pop_front()
                        .ok_or_else(|| "read_string: no input available".to_string())?
                };
                Ok(Value::Str(raw))
            }
            Builtin::HasInput => {
                if !args.is_empty() {
                    return Err("has_input takes no arguments".to_string());
                }
                Ok(Value::Int((!self.input.borrow().is_empty()) as i64))
            }
            Builtin::Len => {
                let value = args
                    .first()
                    .ok_or_else(|| "len requires one argument".to_string())?;
                let len = match value {
                    Value::Array(v) => v.len() as i64,
                    Value::Str(v) => v.chars().count() as i64,
                    Value::Struct { fields, .. } => fields.len() as i64,
                    other => return Err(format!("len unsupported for {:?}", other)),
                };
                Ok(Value::Int(len))
            }
            Builtin::Abs => {
                let value = args
                    .first()
                    .ok_or_else(|| "abs requires one argument".to_string())?;
                Ok(Value::Int(expect_int(value)?.abs()))
            }
            Builtin::Min => {
                if args.len() != 2 {
                    return Err("min requires two arguments".to_string());
                }
                Ok(Value::Int(expect_int(&args[0])?.min(expect_int(&args[1])?)))
            }
            Builtin::Max => {
                if args.len() != 2 {
                    return Err("max requires two arguments".to_string());
                }
                Ok(Value::Int(expect_int(&args[0])?.max(expect_int(&args[1])?)))
            }
            Builtin::Pow => {
                if args.len() != 2 {
                    return Err("pow requires two arguments".to_string());
                }
                Ok(Value::Int(
                    expect_int(&args[0])?.pow(expect_int(&args[1])? as u32),
                ))
            }
        }
    }
}

#[derive(Clone, Debug)]
enum Control {
    Next,
    Return(Value),
    Break,
    Continue,
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Int(v) => *v != 0,
        Value::Str(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Optional { is_some, .. } => *is_some,
        Value::Unit => false,
        _ => true,
    }
}

fn value_eq(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        _ => false,
    }
}

fn pattern_matches(pattern: &Pattern, value: &Value, env: &Env) -> bool {
    match pattern {
        Pattern::Ok(binding) => {
            if let Value::Result { is_ok: true, value } = value {
                env.define(binding, value.as_ref().clone());
                true
            } else {
                false
            }
        }
        Pattern::Err(binding) => {
            if let Value::Result {
                is_ok: false,
                value,
            } = value
            {
                env.define(binding, value.as_ref().clone());
                true
            } else {
                false
            }
        }
        Pattern::Some(binding) => {
            if let Value::Optional {
                is_some: true,
                value,
            } = value
            {
                env.define(binding, value.as_ref().clone());
                true
            } else {
                false
            }
        }
        Pattern::None => matches!(value, Value::Optional { is_some: false, .. }),
        Pattern::Ident(binding) => {
            env.define(binding, value.clone());
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::benchmark::get_benchmark;
    use crate::solver::solve_problem;

    use super::*;

    fn assert_program_output(code: &str, expected: &str) {
        let result = execute_program(code)
            .unwrap_or_else(|err| panic!("program execution failed: {err}\n\n{code}"));
        assert_eq!(result.output, expected);
    }

    #[test]
    fn parses_and_executes_count_positive() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("count_positive"))
            .unwrap();
        let code = solve_problem(&problem).code;
        verify_problem_code_strict(&problem, &code).unwrap();
    }

    #[test]
    fn parses_and_executes_safe_div_match() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("safe_div_or_neg1"))
            .unwrap();
        let code = solve_problem(&problem).code;
        verify_problem_code_strict(&problem, &code).unwrap();
    }

    #[test]
    fn parses_and_executes_closure_map_sum() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("closure_map_sum"))
            .unwrap();
        let code = solve_problem(&problem).code;
        verify_problem_code_strict(&problem, &code).unwrap();
    }

    #[test]
    fn parses_and_executes_palindrome() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("palindrome_check"))
            .unwrap();
        let code = solve_problem(&problem).code;
        verify_problem_code_strict(&problem, &code).unwrap();
    }

    #[test]
    fn executes_for_to_with_break_and_continue() {
        let code = r#"
fn main() -> i64 {
    total := 0;
    for i := 1 to 8 {
        if i == 3 {
            continue;
        }
        if i == 7 {
            break;
        }
        total = total + i;
    }
    println_i64(total);
    return 0;
}
"#;
        assert_program_output(code, "18");
    }

    #[test]
    fn executes_range_loop_and_numeric_builtins() {
        let code = r#"
fn main() -> i64 {
    total := 0;
    for i in 2..6 {
        total = total + pow(i, 2);
    }
    println_i64(total);
    println_i64(abs(-9));
    println_i64(min(7, 4));
    println_i64(max(7, 4));
    return 0;
}
"#;
        assert_program_output(code, "54\n9\n4\n7");
    }

    #[test]
    fn executes_array_mutation_methods() {
        let code = r#"
fn main() -> i64 {
    values := [5, 1, 3];
    values.push(7);
    popped := values.pop();
    values.push(2);
    values.sort();
    println_i64(popped);
    println_i64(values.len);
    println_i64(values[0]);
    println_i64(values[3]);
    return 0;
}
"#;
        assert_program_output(code, "7\n4\n1\n5");
    }

    #[test]
    fn executes_filter_join_and_string_methods() {
        let code = r#"
fn main() -> i64 {
    text := "  Red,BLUE,Green  ";
    parts := text.trim().split(",");
    println(parts[1].lower());
    println("kernel".ends_with("nel"));
    println("abba".replace("b", "x"));
    words := ["gpu", "mog", "rust"];
    println(words.join("-"));
    nums := [1, 2, 3, 4, 5];
    evens := nums.filter(fn(x) { return x % 2 == 0; });
    println_i64(evens.len);
    println_i64(evens[0] + evens[1]);
    println("gpu".upper());
    return 0;
}
"#;
        assert_program_output(code, "blue\ntrue\naxxa\ngpu-mog-rust\n2\n6\nGPU");
    }

    #[test]
    fn executes_programs_with_stream_input() {
        let code = r#"
fn main() -> i64 {
    while has_input() == 1 {
        x := read_i64();
        println_i64(x * 3);
    }
    return 0;
}
"#;
        let result = execute_program_with_input(
            code,
            vec!["2".to_string(), "-4".to_string(), "7".to_string()],
        )
        .unwrap();
        assert_eq!(result.output, "6\n-12\n21");
    }

    #[test]
    fn executes_programs_with_string_input() {
        let code = r#"
fn main() -> i64 {
    first := read_string();
    println(first.upper());
    println_i64(has_input());
    second := read_line();
    println(second.trim());
    println_i64(has_input());
    return 0;
}
"#;
        let result =
            execute_program_with_input(code, vec!["gpu".to_string(), "  mog rust  ".to_string()])
                .unwrap();
        assert_eq!(result.output, "GPU\n1\nmog rust\n0");
    }

    #[test]
    fn stream_input_reports_parse_errors() {
        let code = r#"
fn main() -> i64 {
    println_i64(read_i64());
    return 0;
}
"#;
        let err = execute_program_with_input(code, vec!["nope".to_string()]).unwrap_err();
        assert!(err.contains("cannot parse"));
    }

    #[test]
    fn executes_solver_output_for_full_benchmark() {
        for problem in get_benchmark(1) {
            let result = solve_problem(&problem);
            assert!(result.success, "solver failed for {}", problem.name);
            verify_problem_code_strict(&problem, &result.code).unwrap_or_else(|err| {
                panic!("runtime verification failed for {}: {}", problem.name, err)
            });
        }
    }

    #[test]
    fn executes_wrapper_program_for_struct_problem() {
        let problem = get_benchmark(1)
            .into_iter()
            .find(|p| p.name.starts_with("point_sum"))
            .unwrap();
        let code = solve_problem(&problem).code;
        let wrapped = problem.wrap_program(&code).unwrap();
        let result = execute_program(&wrapped).unwrap();
        assert_eq!(result.output, problem.expected_stdout());
    }

    #[test]
    fn executes_wrapper_program_for_full_benchmark() {
        for problem in get_benchmark(1) {
            let result = solve_problem(&problem);
            assert!(result.success, "solver failed for {}", problem.name);
            let wrapped = problem.wrap_program(&result.code).unwrap();
            let exec = execute_program(&wrapped).unwrap_or_else(|err| {
                panic!("program execution failed for {}: {}", problem.name, err)
            });
            assert_eq!(
                exec.output,
                problem.expected_stdout(),
                "stdout mismatch for {}",
                problem.name
            );
        }
    }
}
