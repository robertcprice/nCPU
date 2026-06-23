use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::rc::Rc;
use std::sync::mpsc;
use std::thread;

use crate::benchmark::{generated_holdouts, Problem, Value as BenchmarkValue};

/// Maximum number of iterations any single loop may execute before the
/// interpreter aborts with an error. This bounds the verify path: a candidate
/// `for i in 0..n` with an attacker-controlled `n` cannot spin the process
/// forever — it returns `Err` once the cap is exceeded. Single-sourced so all
/// loop arms (`While`, `ForTo`, `ForInRange`, `ForIn`, `ForParallel`) agree.
const MAX_LOOP_ITERS: usize = 100_000;

/// Absolute tolerance for float equality in `output_matches`. Used together
/// with `FLOAT_REL_EPS` so very small and very large magnitudes both compare
/// sensibly instead of with exact bit equality.
const FLOAT_ABS_EPS: f64 = 1e-9;
/// Relative tolerance for float equality in `output_matches`.
const FLOAT_REL_EPS: f64 = 1e-9;

/// Total, NaN-aware, relative+absolute epsilon float comparison used by the
/// acceptance oracle. NaN policy: two NaNs are considered equal; a NaN vs a
/// non-NaN is never equal. Otherwise accept when the difference is within the
/// absolute epsilon OR within the relative epsilon scaled by the larger
/// magnitude. This replaces exact `==`, which mis-rejects rounding-equal
/// floats and has no defined NaN behavior.
fn float_eq(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    let diff = (a - b).abs();
    diff <= FLOAT_ABS_EPS || diff <= FLOAT_REL_EPS * a.abs().max(b.abs())
}

/// True when float `f` represents an integer that round-trips exactly through
/// `i64` and equals `i`. This is the SOUND float<->int bridge: it rejects
/// non-integral floats and rejects integers that lose precision in `f64`
/// (e.g. 2^53+1, whose `f64` collapses onto 2^53), so the oracle never
/// false-accepts via a lossy widen.
fn float_matches_int(f: f64, i: i64) -> bool {
    f.fract() == 0.0 && (f as i64) as f64 == f && (f as i64) == i
}

// System/FFI modules
pub mod extern_;
pub mod resource;
pub mod syscall;

/// Error type for runtime FFI operations (separate from String-based errors)
#[derive(Debug, Clone)]
pub enum Errno {
    /// Permission denied
    PermissionDenied(String),
    /// Invalid argument
    InvalidArgument(String),
    /// I/O error
    IOError(String),
    /// Not found
    NotFound(String),
    /// Operation not supported
    NotSupported(String),
    /// Resource exhausted
    ResourceExhausted(String),
}

impl std::fmt::Display for Errno {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Errno::PermissionDenied(s) => write!(f, "Permission denied: {}", s),
            Errno::InvalidArgument(s) => write!(f, "Invalid argument: {}", s),
            Errno::IOError(s) => write!(f, "I/O error: {}", s),
            Errno::NotFound(s) => write!(f, "Not found: {}", s),
            Errno::NotSupported(s) => write!(f, "Not supported: {}", s),
            Errno::ResourceExhausted(s) => write!(f, "Resource exhausted: {}", s),
        }
    }
}

impl std::error::Error for Errno {}

/// Result type for FFI operations (separate from existing Result<T, String>)
pub type FfiResult<T> = std::result::Result<T, Errno>;

#[derive(Clone, Debug)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    Pair(i64, i64),
    Quad(i64, i64, i64, i64),
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    Enum {
        type_name: String,
        variant: String,
        fields: Vec<Value>,
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
    FileHandle(i64),
    Unit,
    Channel(i64),
    Mutex(i64),
    ThreadHandle(i64),
}

#[derive(Clone, Copy, Debug)]
pub enum Builtin {
    PrintlnI64,
    PrintlnF64,
    Println,
    PrintF64,
    PrintString,
    Print,
    ReadI64,
    ReadString,
    Read,
    HasInput,
    Len,
    Abs,
    Min,
    Max,
    Pow,
    OpenFile,
    ReadFile,
    WriteFile,
    CloseFile,
    Reduce,
    Spawn,
    Send,
    Recv,
    NewChannel,
    NewMutex,
    Lock,
    Unlock,
    ParallelFor,
    Error,
    Unwrap,
    UnwrapOr,
}

#[derive(Clone, Debug)]
pub struct Closure {
    params: Vec<String>,
    body: Vec<Stmt>,
    env: Env,
    type_params: Vec<String>,
}

#[derive(Clone, Debug)]
struct Env(Rc<EnvData>);

#[derive(Debug)]
struct EnvData {
    parent: Option<Env>,
    module_name: Option<String>,
    values: RefCell<HashMap<String, Value>>,
    modules: RefCell<HashMap<String, Env>>,
}

impl Env {
    fn new() -> Self {
        Self(Rc::new(EnvData {
            parent: None,
            module_name: None,
            values: RefCell::new(HashMap::new()),
            modules: RefCell::new(HashMap::new()),
        }))
    }

    fn with_module(module_name: String) -> Self {
        Self(Rc::new(EnvData {
            parent: None,
            module_name: Some(module_name),
            values: RefCell::new(HashMap::new()),
            modules: RefCell::new(HashMap::new()),
        }))
    }

    fn child(&self) -> Self {
        Self(Rc::new(EnvData {
            parent: Some(self.clone()),
            module_name: self.0.module_name.clone(),
            values: RefCell::new(HashMap::new()),
            modules: RefCell::new(HashMap::new()),
        }))
    }

    fn define(&self, name: &str, value: Value) {
        self.0.values.borrow_mut().insert(name.to_string(), value);
    }

    fn define_module(&self, name: &str, env: Env) {
        self.0.modules.borrow_mut().insert(name.to_string(), env);
    }

    fn get(&self, name: &str) -> Option<Value> {
        if let Some(value) = self.0.values.borrow().get(name).cloned() {
            return Some(value);
        }

        // Check for module-qualified names (e.g., "module::function")
        if name.contains("::") {
            let parts: Vec<&str> = name.split("::").collect();
            if parts.len() == 2 {
                let module_name = parts[0];
                let symbol_name = parts[1];
                if let Some(module_env) = self.0.modules.borrow().get(module_name) {
                    return module_env.get(symbol_name);
                }
            }
        }

        self.0.parent.as_ref().and_then(|parent| parent.get(name))
    }

    fn resolve_import(&self, module_name: &str, symbols: &[String]) -> Result<(), String> {
        if let Some(module_env) = self.0.modules.borrow().get(module_name) {
            for symbol in symbols {
                if let Some(value) = module_env.get(symbol) {
                    self.define(symbol, value);
                }
            }
            Ok(())
        } else {
            Err(format!("module '{module_name}' not found"))
        }
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
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeAlias {
    pub name: String,
    pub target: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitDecl {
    pub name: String,
    pub methods: Vec<Function>,
    pub type_params: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplBlock {
    pub target_type: String,
    pub trait_name: Option<String>,
    pub methods: Vec<Function>,
    pub type_params: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
    pub type_params: Vec<String>,
    pub is_method: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleDecl {
    pub name: String,
    pub exports: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportDecl {
    pub module_name: String,
    pub symbols: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub modules: HashMap<String, ModuleDecl>,
    pub imports: Vec<ImportDecl>,
    pub structs: HashMap<String, StructDecl>,
    pub enums: HashMap<String, EnumDecl>,
    pub type_aliases: HashMap<String, TypeAlias>,
    pub functions: HashMap<String, Function>,
    pub traits: HashMap<String, TraitDecl>,
    pub impls: Vec<ImplBlock>,
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
    ForParallel {
        var_name: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    Break(Option<String>),
    Continue(Option<String>),
    Expr(Expr),
    Try {
        var_name: Option<String>,
        expr: Expr,
        then_block: Vec<Stmt>,
        catch_block: Vec<Stmt>,
    },
    Throw(Expr),
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
    /// A float literal stored as its IEEE-754 bits, so `Expr` keeps `Eq`/`Hash`
    /// (the rest of the AST — `Stmt`, `Target` — derives `Eq` and contains
    /// `Expr`). Recover the value with `f64::from_bits` at evaluation time.
    Float(u64),
    Str(String),
    Bool(bool),
    ArrayLit(Vec<Expr>),
    StructConstruct {
        name: String,
        fields: Vec<(String, Expr)>,
    },
    EnumConstruct {
        type_name: String,
        variant: String,
        fields: Vec<Expr>,
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
        type_params: Vec<String>,
    },
    Ok(Box<Expr>),
    Err(Box<Expr>),
    Some(Box<Expr>),
    None,
    Try(Box<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pattern {
    Ok(String),
    Err(String),
    Some(String),
    None,
    EnumVariant(String, String),
    Ident(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    And,
    Or,
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
    Enum,
    Type,
    Module,
    Import,
    Export,
    Return,
    If,
    Else,
    While,
    For,
    In,
    To,
    Parallel,
    Match,
    Break,
    Continue,
    Ok,
    Err,
    Some,
    None,
    True,
    False,
    Spawn,
    Send,
    Recv,
    Channel,
    Mutex,
    Lock,
    Unlock,
    Try,
    Throw,
    Catch,
    Trait,
    Impl,
    SelfType,
    Ident(String),
    Int(i64),
    /// A float literal kept as its source text (so the `Token` enum stays `Eq`);
    /// the parser converts it to the bit pattern in `Expr::Float`.
    Float(String),
    Str(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Shl,
    Shr,
    AmpAmp,
    PipePipe,
    Bang,
    Tilde,
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Eq,
    ColonEq,
    DoubleColon,
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

/// Validate that all function calls in the program refer to defined functions.
/// Returns Ok with the function signature if validation passes, Err with undefined functions.
pub fn validate_call_graph(src: &str) -> Result<String, String> {
    let program = parse_program(src)?;
    let mut undefined_calls: Vec<String> = Vec::new();
    let defined_functions: std::collections::HashSet<String> =
        program.functions.keys().cloned().collect();

    // Built-in functions that are always available
    let builtins: std::collections::HashSet<&str> = [
        "println_i64",
        "println_f64",
        "println",
        "print_f64",
        "print_string",
        "print",
        "read_i64",
        "read_string",
        "read_line",
        "read",
        "has_input",
        "len",
        "abs",
        "min",
        "max",
        "pow",
        "open_file",
        "read_file",
        "write_file",
        "close_file",
        "reduce",
        "spawn",
        "send",
        "recv",
        "new_channel",
        "new_mutex",
        "lock",
        "unlock",
        "error",
        "unwrap",
        "unwrap_or",
    ]
    .into_iter()
    .collect();

    // Collect all function calls from the program
    for function in program.functions.values() {
        collect_calls_from_stmts(
            &function.body,
            &mut undefined_calls,
            &defined_functions,
            &builtins,
        );
    }

    if undefined_calls.is_empty() {
        Ok("All function calls are defined.".to_string())
    } else {
        Err(format!(
            "Undefined function calls: {}",
            undefined_calls.join(", ")
        ))
    }
}

fn collect_calls_from_stmts(
    stmts: &[Stmt],
    undefined_calls: &mut Vec<String>,
    defined_functions: &std::collections::HashSet<String>,
    builtins: &std::collections::HashSet<&str>,
) {
    for stmt in stmts {
        collect_calls_from_stmt(stmt, undefined_calls, defined_functions, builtins);
    }
}

fn collect_calls_from_stmt(
    stmt: &Stmt,
    undefined_calls: &mut Vec<String>,
    defined_functions: &std::collections::HashSet<String>,
    builtins: &std::collections::HashSet<&str>,
) {
    match stmt {
        Stmt::VarDecl(_, expr) | Stmt::Expr(expr) | Stmt::Return(expr) => {
            collect_calls_from_expr(expr, undefined_calls, defined_functions, builtins);
        }
        Stmt::Assign(target, expr) => {
            collect_calls_from_target(target, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(expr, undefined_calls, defined_functions, builtins);
        }
        Stmt::If {
            condition,
            then_block,
            else_block,
        } => {
            collect_calls_from_expr(condition, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(then_block, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(else_block, undefined_calls, defined_functions, builtins);
        }
        Stmt::While { condition, body } => {
            collect_calls_from_expr(condition, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Stmt::ForTo {
            start, end, body, ..
        } => {
            collect_calls_from_expr(start, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(end, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Stmt::ForInRange {
            start, end, body, ..
        } => {
            collect_calls_from_expr(start, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(end, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Stmt::ForIn { iterable, body, .. } => {
            collect_calls_from_expr(iterable, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Stmt::ForParallel {
            start, end, body, ..
        } => {
            collect_calls_from_expr(start, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(end, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Stmt::Try {
            expr,
            then_block,
            catch_block,
            ..
        } => {
            collect_calls_from_expr(expr, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(then_block, undefined_calls, defined_functions, builtins);
            collect_calls_from_stmts(catch_block, undefined_calls, defined_functions, builtins);
        }
        Stmt::Throw(expr) => {
            collect_calls_from_expr(expr, undefined_calls, defined_functions, builtins);
        }
        Stmt::Break(_) | Stmt::Continue(_) => {}
    }
}

fn collect_calls_from_target(
    target: &Target,
    undefined_calls: &mut Vec<String>,
    defined_functions: &std::collections::HashSet<String>,
    builtins: &std::collections::HashSet<&str>,
) {
    match target {
        Target::Ident(_) => {}
        Target::Field(base, _) | Target::Index(base, _) => {
            collect_calls_from_expr(base, undefined_calls, defined_functions, builtins);
        }
    }
}

fn collect_calls_from_expr(
    expr: &Expr,
    undefined_calls: &mut Vec<String>,
    defined_functions: &std::collections::HashSet<String>,
    builtins: &std::collections::HashSet<&str>,
) {
    match expr {
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::None => {}
        Expr::ArrayLit(elements) => {
            for elem in elements {
                collect_calls_from_expr(elem, undefined_calls, defined_functions, builtins);
            }
        }
        Expr::StructConstruct { fields, .. } => {
            for (_, field_expr) in fields {
                collect_calls_from_expr(field_expr, undefined_calls, defined_functions, builtins);
            }
        }
        Expr::EnumConstruct { fields, .. } => {
            for field_expr in fields {
                collect_calls_from_expr(field_expr, undefined_calls, defined_functions, builtins);
            }
        }
        Expr::Ident(_name) => {
            // Identifier references are not calls
        }
        Expr::Unary(_, inner) => {
            collect_calls_from_expr(inner, undefined_calls, defined_functions, builtins);
        }
        Expr::Binary(lhs, _, rhs) => {
            collect_calls_from_expr(lhs, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(rhs, undefined_calls, defined_functions, builtins);
        }
        Expr::Call(callee, args) => {
            // Check if this is a method call (e.g., arr.push(1))
            if let Expr::Field(base, _method) = &**callee {
                // For method calls, we need to check if the method is valid
                // But the base value's type determines available methods
                collect_calls_from_expr(base, undefined_calls, defined_functions, builtins);
            } else if let Expr::Ident(name) = &**callee {
                // Direct function call
                if !defined_functions.contains(name) && !builtins.contains(name.as_str()) {
                    if !undefined_calls.contains(name) {
                        undefined_calls.push(name.clone());
                    }
                }
            } else {
                // Expression that evaluates to a function/closure
                collect_calls_from_expr(callee, undefined_calls, defined_functions, builtins);
            }
            for arg in args {
                collect_calls_from_expr(arg, undefined_calls, defined_functions, builtins);
            }
        }
        Expr::Field(base, _) => {
            collect_calls_from_expr(base, undefined_calls, defined_functions, builtins);
        }
        Expr::Index(base, idx) => {
            collect_calls_from_expr(base, undefined_calls, defined_functions, builtins);
            collect_calls_from_expr(idx, undefined_calls, defined_functions, builtins);
        }
        Expr::Match(subject, arms) => {
            collect_calls_from_expr(subject, undefined_calls, defined_functions, builtins);
            for (_, arm_expr) in arms {
                collect_calls_from_expr(arm_expr, undefined_calls, defined_functions, builtins);
            }
        }
        Expr::Closure {
            params: _,
            body,
            type_params: _,
        } => {
            collect_calls_from_stmts(body, undefined_calls, defined_functions, builtins);
        }
        Expr::Ok(inner) | Expr::Err(inner) | Expr::Some(inner) | Expr::Try(inner) => {
            collect_calls_from_expr(inner, undefined_calls, defined_functions, builtins);
        }
    }
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

/// Install (exactly once, process-wide) a no-op panic hook so a panic caught by
/// `catch_unwind` on the verify path does not spam stderr with a backtrace.
/// This keeps the verify output deterministic and quiet; the `Result` returned
/// by `catch_unwind` is deterministic regardless of the hook.
fn install_silent_panic_hook_once() {
    use std::sync::Once;
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|_| {}));
    });
}

/// Run a candidate-executing closure under `catch_unwind`, converting a caught
/// panic into a clean `Err` so a candidate that panics during verification
/// rejects the candidate instead of aborting the whole process.
///
/// `AssertUnwindSafe` is required because `Runtime` holds `RefCell`/`JoinHandle`/
/// `mpsc` channels (not `UnwindSafe`); the verify path discards the `Runtime`
/// after the call, so asserting unwind safety is sound here.
fn run_isolated<F>(f: F) -> Result<Value, String>
where
    F: FnOnce() -> Result<Value, String>,
{
    install_silent_panic_hook_once();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => Err("candidate panicked during verification".to_string()),
    }
}

pub fn execute_function_for_problem(
    code: &str,
    function_name: &str,
    args: &[BenchmarkValue],
    problem: &Problem,
) -> Result<Value, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::new(program);
    runtime.set_verify_mode(true);
    let args = args
        .iter()
        .map(|value| runtime_value_from_problem_meta(value, problem))
        .collect::<Result<Vec<_>, _>>()?;
    run_isolated(|| runtime.call_function(function_name, args))
}

/// Execute a single-string-argument function and return its raw `Value`.
/// Used by the generative-morphology path, where the function returns a string
/// (`fn pluralize(s: string) -> string`) rather than an i64.
pub fn execute_str_function(
    code: &str,
    function_name: &str,
    input: &str,
) -> Result<String, String> {
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
    pub side_effects: SideEffects,
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

/// Compare a runtime result value (`runtime::Value`) against an expected
/// benchmark value (`benchmark::Value`). Handles int, bool, string, array, and
/// pair outputs uniformly — this is what lets string/array-output problems
/// verify through the main pipeline.
fn output_matches(actual: &Value, expected: &crate::benchmark::Value) -> bool {
    use crate::benchmark::Value as BV;
    match (actual, expected) {
        (Value::Int(a), BV::Int(b)) => a == b,
        // Float bridge: a float output matches an int expected ONLY when the
        // float is integral AND round-trips exactly through i64. A naive
        // `*a == *b as f64` widen is lossy for |b| > 2^53 and false-accepts
        // (e.g. 2^53+1 collapses onto 2^53 in f64), so we use the sound
        // `float_matches_int` predicate instead.
        (Value::Float(a), BV::Int(b)) => float_matches_int(*a, *b),
        (Value::Int(a), BV::Float(b)) => float_matches_int(f64::from_bits(*b), *a),
        // Float/Float: relative+absolute epsilon with a defined NaN policy,
        // never exact bit equality.
        (Value::Float(a), BV::Float(b)) => float_eq(*a, f64::from_bits(*b)),
        // Predicate bridge: an i64 0/1 output matches a bool expected
        // (so a solver that emits `return 1;` still verifies against
        // `expected: true`), and a bool output matches an i64 0/1
        // expected (so a 0/1 lane can also be the canonical carrier
        // for a problem posed as `expected: 1`).
        (Value::Int(a), BV::Bool(b)) => (*a != 0) == *b,
        (Value::Bool(a), BV::Int(b)) => i64::from(*a) == *b,
        (Value::Bool(a), BV::Bool(b)) => a == b,
        (Value::Str(a), BV::Str(b)) => a == b,
        // Arrays compare element-wise and recurse on EVERY element. The wire
        // type (`benchmark::Value::Array`) is now `Vec<Value>`, so each expected
        // element is itself a full `benchmark::Value` and the comparison recurses
        // through `output_matches`. This is what lets nested arrays
        // (`[[1,2],[3]]`), typed arrays (`["a","b"]`, `[1.0,2.0]`), and arrays of
        // struct elements verify with the SAME strict bridges the top-level
        // oracle uses (an int element still matches `0/1`<->bool and int<->float,
        // but nothing looser). Length must match exactly; no padding/truncation.
        (Value::Array(a), BV::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| output_matches(x, y))
        }
        // Pair: the runtime can carry a pair either as the dedicated `Pair`
        // value or (when it flows through struct-of-state) as a 2-field
        // `Struct`. Both must match the `(a, b)` wire pair exactly.
        (Value::Pair(a, b), BV::Pair(c, d)) => a == c && b == d,
        (Value::Struct { fields, .. }, BV::Pair(c, d)) => {
            struct_fields_match(fields, &[*c, *d])
        }
        // Quad: same idea — dedicated `Quad` value or a 4-field `Struct`.
        (Value::Quad(a, b, c, d), BV::Quad(e, f, g, h)) => {
            a == e && b == f && c == g && d == h
        }
        (Value::Struct { fields, .. }, BV::Quad(e, f, g, h)) => {
            struct_fields_match(fields, &[*e, *f, *g, *h])
        }
        // Tree: the runtime has no dedicated tree value; a tree is the
        // `Struct { name: "Tree", fields: { "nodes": Array[TreeNode structs] } }`
        // that `runtime_value_from_problem` builds and the tree codegen reads
        // (`tree.nodes[i].value/.left/.right`). Compare structurally, node by
        // node, against the expected `Vec<TreeNode>`.
        (Value::Struct { fields, .. }, BV::Tree(nodes)) => tree_struct_matches(fields, nodes),
        _ => false,
    }
}

/// Strictly compare the int payloads of a runtime `Struct`'s fields against an
/// ordered list of expected scalars (used for `Pair`/`Quad` outputs). Requires
/// the exact arity and that every field is an int matching its expected slot.
/// Order-independent over field *names* (struct fields are a `HashMap`) but every
/// expected value must be present exactly once — so it stays a strict equality
/// check, not a subset match.
fn struct_fields_match(fields: &HashMap<String, Value>, expected: &[i64]) -> bool {
    if fields.len() != expected.len() {
        return false;
    }
    // Multiset equality of the int payloads: every expected scalar must be
    // consumed by a distinct field. Equal-arity + multiset match == exact match.
    let mut remaining: Vec<i64> = expected.to_vec();
    for value in fields.values() {
        let Value::Int(v) = value else {
            return false;
        };
        if let Some(pos) = remaining.iter().position(|e| e == v) {
            remaining.swap_remove(pos);
        } else {
            return false;
        }
    }
    remaining.is_empty()
}

/// Structurally compare a runtime tree `Struct` against an expected
/// `Vec<TreeNode>`. The runtime tree is `{ nodes: [ { value, left, right }, .. ] }`
/// (negative `left`/`right` mean null). Every node must match value, left and
/// right exactly and the node count must be identical.
fn tree_struct_matches(fields: &HashMap<String, Value>, nodes: &[crate::benchmark::TreeNode]) -> bool {
    let Some(Value::Array(items)) = fields.get("nodes") else {
        return false;
    };
    if items.len() != nodes.len() {
        return false;
    }
    items.iter().zip(nodes.iter()).all(|(item, node)| {
        let Value::Struct { fields, .. } = item else {
            return false;
        };
        matches!(fields.get("value"), Some(Value::Int(v)) if *v == node.value)
            && matches!(fields.get("left"), Some(Value::Int(l)) if *l == node.left as i64)
            && matches!(fields.get("right"), Some(Value::Int(r)) if *r == node.right as i64)
    })
}

pub fn verify_problem_code(problem: &Problem, code: &str) -> Result<(), String> {
    let fn_name = problem.function_name();
    for example in &problem.examples {
        let value = execute_function_for_problem(code, fn_name, &example.inputs, problem)?;
        if !output_matches(&value, &example.expected) {
            return Err(format!(
                "example failed for {}: expected {}, got {:?}",
                problem.name, example.expected, value
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
        if !output_matches(&value, &example.expected) {
            return Err(format!(
                "holdout failed for {}: inputs {:?}, expected {}, got {:?}",
                problem.name, example.inputs, example.expected, value
            ));
        }
    }
    Ok(())
}

/// Run a wrapped candidate's `main()` on the verify path: filesystem builtins
/// are denied and a panic is isolated into a clean `Err` (never a process
/// abort). Distinct from the public `execute_program`, which must keep FS
/// access for legitimate non-verify callers.
fn execute_program_verified(code: &str) -> Result<ExecutionResult, String> {
    let program = parse_program(code)?;
    let runtime = Runtime::with_input(program, Vec::new());
    runtime.set_verify_mode(true);
    install_silent_panic_hook_once();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.run_main())) {
        Ok(result) => result,
        Err(_) => Err("candidate panicked during verification".to_string()),
    }
}

pub fn verify_problem_code_via_main(problem: &Problem, code: &str) -> Result<(), String> {
    let program = problem.wrap_program(code)?;
    let result = execute_program_verified(&program)?;
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
        BenchmarkValue::Float(b) => Ok(Value::Float(f64::from_bits(*b))),
        BenchmarkValue::Bool(b) => Ok(Value::Bool(*b)),
        BenchmarkValue::Str(v) => Ok(Value::Str(v.clone())),
        BenchmarkValue::Array(values) => {
            // The wire array now carries `Vec<BenchmarkValue>`, so each element
            // converts recursively to its runtime `Value` (an int element still
            // becomes `Value::Int`, but a string/float/nested-array element now
            // converts correctly instead of being forced through `Value::Int`).
            let elems = values
                .iter()
                .map(|v| runtime_value_from_problem(v, problem_name))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(elems))
        }
        BenchmarkValue::Quad(a, b, c, d) => Ok(Value::Quad(*a, *b, *c, *d)),
        BenchmarkValue::Pair(a, b) => {
            // Known structural pairs keep their domain names/fields so the
            // generated code's `p.x`/`r.width` field accesses resolve. Any
            // other pair still converts (as a generic 2-field struct) instead
            // of hard-erroring, which is the cheap generalization beyond the
            // hardcoded Point/Rectangle match.
            let (name, lhs, rhs) = if problem_name.starts_with("point_sum") {
                ("Point", "x", "y")
            } else if problem_name.starts_with("rectangle_area") {
                ("Rectangle", "width", "height")
            } else {
                ("Pair", "first", "second")
            };
            let mut fields = HashMap::new();
            fields.insert(lhs.to_string(), Value::Int(*a));
            fields.insert(rhs.to_string(), Value::Int(*b));
            Ok(Value::Struct {
                name: name.to_string(),
                fields,
            })
        }
        BenchmarkValue::Tree(nodes) => Ok(tree_to_runtime_value(nodes)),
    }
}

/// Build the canonical runtime tree value from a wire `Vec<TreeNode>`.
///
/// The runtime has no dedicated tree value, so a tree is represented as the
/// struct the tree codegen consumes:
/// `Struct { name: "Tree", fields: { "nodes": Array[ Struct { name: "TreeNode",
/// fields: { value, left, right } }, .. ] } }`. `left`/`right` are widened from
/// the wire `i32` to the runtime's `i64` (negative still means null). This
/// round-trips: `tree_struct_matches` (the equality oracle's `Tree` arm) reads
/// back exactly these field names and the same node ordering.
fn tree_to_runtime_value(nodes: &[crate::benchmark::TreeNode]) -> Value {
    let node_values: Vec<Value> = nodes
        .iter()
        .map(|node| {
            let mut fields = HashMap::new();
            fields.insert("value".to_string(), Value::Int(node.value));
            fields.insert("left".to_string(), Value::Int(node.left as i64));
            fields.insert("right".to_string(), Value::Int(node.right as i64));
            Value::Struct {
                name: "TreeNode".to_string(),
                fields,
            }
        })
        .collect();
    let mut fields = HashMap::new();
    fields.insert("nodes".to_string(), Value::Array(node_values));
    Value::Struct {
        name: "Tree".to_string(),
        fields,
    }
}

/// Convert a runtime result `Value` back to the wire `benchmark::Value`, so a
/// REFERENCE execution's output can become a holdout `Example.expected`.
///
/// This is the inverse of `runtime_value_from_problem` for the value shapes a
/// benchmark function can return, and it is deliberately STRICT: a shape we
/// cannot faithfully represent on the wire (e.g. an enum/optional/function, or a
/// struct whose fields are not all ints) returns `Err` so the caller falls back
/// to hand-authored holdouts rather than fabricating an expected value.
///
/// The mapping mirrors the equality oracle (`output_matches`) so a holdout built
/// here is exactly as strict as the existing example checks:
///   - Int/Float/Bool/Str/Array map directly (arrays recurse element-wise).
///   - A 2-int struct -> `Pair`, a 4-int struct -> `Quad` (multiset over field
///     names, matching `struct_fields_match`); a `Tree` struct -> `Tree`.
///   - `Pair`/`Quad` runtime values pass straight through.
///
/// Determinism: struct fields live in a `HashMap`, so the converter never relies
/// on iteration order — it sorts the int payloads before packing a `Pair`/`Quad`
/// (the oracle compares pairs/quads as a multiset, so a fixed canonical order is
/// both deterministic and equivalent).
pub(crate) fn benchmark_value_from_runtime(value: &Value) -> Result<BenchmarkValue, String> {
    match value {
        Value::Int(v) => Ok(BenchmarkValue::Int(*v)),
        Value::Float(v) => Ok(BenchmarkValue::Float(v.to_bits())),
        Value::Bool(b) => Ok(BenchmarkValue::Bool(*b)),
        Value::Str(s) => Ok(BenchmarkValue::Str(s.clone())),
        Value::Array(items) => {
            let elems = items
                .iter()
                .map(benchmark_value_from_runtime)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(BenchmarkValue::Array(elems))
        }
        Value::Pair(a, b) => Ok(BenchmarkValue::Pair(*a, *b)),
        Value::Quad(a, b, c, d) => Ok(BenchmarkValue::Quad(*a, *b, *c, *d)),
        Value::Struct { name, fields } => {
            // Tree struct: { nodes: [ { value, left, right }, .. ] }.
            if name == "Tree" || fields.contains_key("nodes") {
                let Some(Value::Array(items)) = fields.get("nodes") else {
                    return Err("tree struct missing `nodes` array".to_string());
                };
                let mut nodes = Vec::with_capacity(items.len());
                for item in items {
                    let Value::Struct { fields, .. } = item else {
                        return Err("tree node is not a struct".to_string());
                    };
                    let value = match fields.get("value") {
                        Some(Value::Int(v)) => *v,
                        _ => return Err("tree node missing int `value`".to_string()),
                    };
                    let left = match fields.get("left") {
                        Some(Value::Int(v)) => *v as i32,
                        _ => return Err("tree node missing int `left`".to_string()),
                    };
                    let right = match fields.get("right") {
                        Some(Value::Int(v)) => *v as i32,
                        _ => return Err("tree node missing int `right`".to_string()),
                    };
                    nodes.push(crate::benchmark::TreeNode { value, left, right });
                }
                return Ok(BenchmarkValue::Tree(nodes));
            }
            // Plain struct: must be all-int fields to land on Pair/Quad. Pack in
            // a canonical (sorted) order — the oracle matches pairs/quads as a
            // multiset, so order is irrelevant to correctness and sorting keeps
            // the result deterministic despite HashMap iteration order.
            let mut ints: Vec<i64> = Vec::with_capacity(fields.len());
            for v in fields.values() {
                match v {
                    Value::Int(i) => ints.push(*i),
                    _ => {
                        return Err(format!(
                            "struct field is not an int; cannot map to wire pair/quad: {v:?}"
                        ))
                    }
                }
            }
            ints.sort_unstable();
            match ints.as_slice() {
                [a, b] => Ok(BenchmarkValue::Pair(*a, *b)),
                [a, b, c, d] => Ok(BenchmarkValue::Quad(*a, *b, *c, *d)),
                other => Err(format!(
                    "struct with {} int fields has no wire representation",
                    other.len()
                )),
            }
        }
        other => Err(format!("no wire representation for runtime value: {other:?}")),
    }
}

fn runtime_value_from_problem_meta(
    value: &BenchmarkValue,
    problem: &Problem,
) -> Result<Value, String> {
    match value {
        BenchmarkValue::Pair(a, b) => {
            // Prefer the signature-named struct (so `p.x` / `r.width` resolve);
            // otherwise fall back to a generic 2-field pair struct rather than
            // refusing the conversion.
            let (name, lhs, rhs) = if problem.signature.contains("Point") {
                ("Point", "x", "y")
            } else if problem.signature.contains("Rectangle") {
                ("Rectangle", "width", "height")
            } else {
                ("Pair", "first", "second")
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

/// Coerce an int or float value to `f64` for float arithmetic (an `Int` operand
/// in a float expression is widened, matching how `1 + 2.5` reads).
fn as_f64(value: &Value) -> Result<f64, String> {
    match value {
        Value::Float(v) => Ok(*v),
        Value::Int(v) => Ok(*v as f64),
        _ => Err(format!("expected a number, got {:?}", value)),
    }
}

fn display_value(value: &Value) -> Result<String, String> {
    match value {
        Value::Int(v) => Ok(v.to_string()),
        Value::Float(v) => Ok(format!("{:.7}", v)),
        Value::Bool(v) => Ok(if *v {
            "true".to_string()
        } else {
            "false".to_string()
        }),
        Value::Str(v) => Ok(v.clone()),
        Value::Unit => Ok(String::new()),
        Value::FileHandle(id) => Ok(format!("<file {}>", id)),
        // Array rendering must match `benchmark::render_expected` ("[a, b, c]")
        // so array-returning programs verify against the expected stdout.
        Value::Array(items) => {
            let rendered = items
                .iter()
                .map(display_value)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", rendered.join(", ")))
        }
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
            // A `.` followed by a digit makes this a float literal (`2.5`); a
            // `.` followed by anything else is the field/range operator and is
            // left for the next token (so `arr.len`, `0..n` still lex correctly).
            let is_float = matches!(chars.peek(), Some('.')) && {
                let mut lookahead = chars.clone();
                lookahead.next();
                matches!(lookahead.peek(), Some(d) if d.is_ascii_digit())
            };
            if is_float {
                value.push('.');
                chars.next(); // consume '.'
                while let Some(&next) = chars.peek() {
                    if next.is_ascii_digit() {
                        value.push(next);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(Token::Float(value));
            } else {
                out.push(Token::Int(
                    value
                        .parse::<i64>()
                        .map_err(|err| format!("invalid int literal {value}: {err}"))?,
                ));
            }
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
                "enum" => Token::Enum,
                "type" => Token::Type,
                "mod" => Token::Module,
                "import" => Token::Import,
                "export" => Token::Export,
                "return" => Token::Return,
                "if" => Token::If,
                "else" => Token::Else,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "to" => Token::To,
                "parallel" => Token::Parallel,
                "match" => Token::Match,
                "break" => Token::Break,
                "continue" => Token::Continue,
                "ok" => Token::Ok,
                "err" => Token::Err,
                "some" => Token::Some,
                "none" => Token::None,
                "true" => Token::True,
                "false" => Token::False,
                "spawn" => Token::Spawn,
                "send" => Token::Send,
                "recv" => Token::Recv,
                "channel" => Token::Channel,
                "mutex" => Token::Mutex,
                "lock" => Token::Lock,
                "unlock" => Token::Unlock,
                "try" => Token::Try,
                "throw" => Token::Throw,
                "catch" => Token::Catch,
                "trait" => Token::Trait,
                "impl" => Token::Impl,
                "self" => Token::SelfType,
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
                    out.push(Token::Bang);
                }
            }
            '~' => {
                chars.next();
                out.push(Token::Tilde);
            }
            '<' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::LtEq);
                } else if matches!(chars.peek(), Some('<')) {
                    chars.next();
                    out.push(Token::Shl);
                } else {
                    out.push(Token::Lt);
                }
            }
            '>' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::GtEq);
                } else if matches!(chars.peek(), Some('>')) {
                    chars.next();
                    out.push(Token::Shr);
                } else {
                    out.push(Token::Gt);
                }
            }
            '&' => {
                chars.next();
                if matches!(chars.peek(), Some('&')) {
                    chars.next();
                    out.push(Token::AmpAmp);
                } else {
                    out.push(Token::Amp);
                }
            }
            '|' => {
                chars.next();
                if matches!(chars.peek(), Some('|')) {
                    chars.next();
                    out.push(Token::PipePipe);
                } else {
                    out.push(Token::Pipe);
                }
            }
            '^' => {
                chars.next();
                out.push(Token::Caret);
            }
            ':' => {
                chars.next();
                if matches!(chars.peek(), Some('=')) {
                    chars.next();
                    out.push(Token::ColonEq);
                } else if matches!(chars.peek(), Some(':')) {
                    chars.next();
                    out.push(Token::DoubleColon);
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
        let mut modules = HashMap::new();
        let mut imports = Vec::new();
        let mut structs = HashMap::new();
        let mut enums = HashMap::new();
        let mut type_aliases = HashMap::new();
        let mut functions = HashMap::new();
        let mut traits = HashMap::new();
        let mut impls = Vec::new();

        while !self.at(&Token::Eof) {
            if self.at(&Token::Module) {
                let (module_decl, module_functions) = self.parse_module_decl_with_functions()?;
                modules.insert(module_decl.name.clone(), module_decl);
                // Add module functions to the global function table
                for func in module_functions {
                    functions.insert(func.name.clone(), func);
                }
            } else if self.at(&Token::Import) {
                let import = self.parse_import_decl()?;
                imports.push(import);
            } else if self.at(&Token::Struct) {
                let decl = self.parse_struct_decl()?;
                structs.insert(decl.name.clone(), decl);
            } else if self.at(&Token::Enum) {
                let decl = self.parse_enum_decl()?;
                enums.insert(decl.name.clone(), decl);
            } else if self.at(&Token::Type) {
                let alias = self.parse_type_alias()?;
                type_aliases.insert(alias.name.clone(), alias);
            } else if self.at(&Token::Trait) {
                let decl = self.parse_trait_decl()?;
                traits.insert(decl.name.clone(), decl);
            } else if self.at(&Token::Impl) {
                let impl_block = self.parse_impl_block()?;
                impls.push(impl_block);
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

        Ok(Program {
            modules,
            imports,
            structs,
            enums,
            type_aliases,
            functions,
            traits,
            impls,
        })
    }

    fn parse_module_decl_with_functions(&mut self) -> Result<(ModuleDecl, Vec<Function>), String> {
        self.expect(&Token::Module)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut exports = Vec::new();
        let mut functions = Vec::new();

        while !self.at(&Token::RBrace) {
            // Skip semicolons between declarations
            if self.at(&Token::Semi) {
                self.bump();
                continue;
            }

            if self.at(&Token::Export) {
                self.bump();
                let symbol = self.expect_ident()?;
                exports.push(symbol);
                // Optional comma after export
                if self.at(&Token::Comma) {
                    self.bump();
                }
            } else if self.at(&Token::Fn) {
                let func = self.parse_function_decl()?;
                // Only add to exports if not already exported
                if !exports.contains(&func.name) {
                    exports.push(func.name.clone());
                }
                functions.push(func);
                // Optional comma after function
                if self.at(&Token::Comma) {
                    self.bump();
                }
            } else {
                return Err(format!(
                    "expected export or function in module, got {:?}",
                    self.current()
                ));
            }
        }

        self.expect(&Token::RBrace)?;
        Ok((
            ModuleDecl {
                name: name.clone(),
                exports,
            },
            functions,
        ))
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, String> {
        self.expect(&Token::Import)?;
        let module_name = self.expect_ident()?;
        let mut symbols = Vec::new();
        if self.at(&Token::Colon) {
            self.bump();
            self.expect(&Token::LBrace)?;
            while !self.at(&Token::RBrace) {
                symbols.push(self.expect_ident()?);
                if self.at(&Token::Comma) {
                    self.bump();
                }
            }
            self.expect(&Token::RBrace)?;
        }
        self.expect(&Token::Semi)?;
        Ok(ImportDecl {
            module_name,
            symbols,
        })
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

    fn parse_enum_decl(&mut self) -> Result<EnumDecl, String> {
        self.expect(&Token::Enum)?;
        let name = self.expect_ident()?;
        self.expect(&Token::LBrace)?;
        let mut variants = Vec::new();
        while !self.at(&Token::RBrace) {
            let variant_name = self.expect_ident()?;
            let mut fields = Vec::new();
            if self.at(&Token::LParen) {
                self.bump();
                while !self.at(&Token::RParen) {
                    fields.push(self.expect_ident()?);
                    if self.at(&Token::Colon) {
                        self.bump();
                        self.skip_type_until(&[Token::Comma, Token::RParen])?;
                    }
                    if self.at(&Token::Comma) {
                        self.bump();
                    }
                }
                self.expect(&Token::RParen)?;
            }
            variants.push(EnumVariant {
                name: variant_name,
                fields,
            });
            if self.at(&Token::Comma) {
                self.bump();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(EnumDecl { name, variants })
    }

    fn parse_type_alias(&mut self) -> Result<TypeAlias, String> {
        self.expect(&Token::Type)?;
        let name = self.expect_ident()?;
        self.expect(&Token::Eq)?;
        let target = self.expect_ident()?;
        self.skip_type_until(&[Token::Semi])?;
        self.expect(&Token::Semi)?;
        Ok(TypeAlias { name, target })
    }

    fn parse_trait_decl(&mut self) -> Result<TraitDecl, String> {
        self.expect(&Token::Trait)?;
        let name = self.expect_ident()?;
        let type_params = self.parse_type_params()?;
        self.expect(&Token::LBrace)?;
        let mut methods = Vec::new();
        while !self.at(&Token::RBrace) {
            if self.at(&Token::Semi) {
                self.bump();
                continue;
            }
            // Trait methods are just signatures (no body in traits)
            if self.at(&Token::Fn) {
                let mut func = self.parse_function_decl()?;
                func.is_method = true;
                methods.push(func);
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(TraitDecl {
            name,
            methods,
            type_params,
        })
    }

    fn parse_impl_block(&mut self) -> Result<ImplBlock, String> {
        self.expect(&Token::Impl)?;
        let type_params = self.parse_type_params()?;

        // Check for trait impl: "impl Trait for Type" or "impl Type"
        let target_type = self.expect_ident()?;
        let mut trait_name = None;

        // Check if next identifier is "for"
        let next_is_for = if let Token::Ident(ref s) = self.current() {
            s == "for"
        } else {
            false
        };

        if next_is_for {
            // This is "impl Trait for Type"
            self.bump(); // consume "for"
            trait_name = Some(target_type.clone());
            // Skip type if present
            if self.at(&Token::Lt) {
                self.skip_type_until(&[Token::LBrace])?;
            }
        }

        // Get the actual target type (or the type after "for")
        let final_target = if trait_name.is_some() {
            self.expect_ident()?
        } else {
            target_type
        };

        self.expect(&Token::LBrace)?;
        let mut methods = Vec::new();
        while !self.at(&Token::RBrace) {
            if self.at(&Token::Semi) {
                self.bump();
                continue;
            }
            if self.at(&Token::Fn) {
                let mut func = self.parse_function_decl()?;
                func.is_method = true;
                methods.push(func);
            }
        }
        self.expect(&Token::RBrace)?;

        Ok(ImplBlock {
            target_type: final_target,
            trait_name,
            methods,
            type_params,
        })
    }

    fn parse_function_decl(&mut self) -> Result<Function, String> {
        self.expect(&Token::Fn)?;
        let name = self.expect_ident()?;
        let type_params = self.parse_type_params()?;
        let params = self.parse_params()?;
        if self.at(&Token::Arrow) {
            self.bump();
            self.skip_type_until(&[Token::LBrace])?;
        }
        let body = self.parse_block()?;
        Ok(Function {
            name,
            params,
            body,
            type_params,
            is_method: false,
        })
    }

    fn parse_type_params(&mut self) -> Result<Vec<String>, String> {
        let mut type_params = Vec::new();
        if self.at(&Token::Lt) {
            self.bump();
            while !self.at(&Token::Gt) {
                let param = self.expect_ident()?;
                type_params.push(param);
                if self.at(&Token::Comma) {
                    self.bump();
                }
            }
            self.expect(&Token::Gt)?;
        }
        Ok(type_params)
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
            // Check for label: 'break' | 'break' ':' identifier
            if self.at(&Token::Colon) {
                self.bump();
                let label = self.expect_ident()?;
                return Ok(Stmt::Break(Some(label)));
            }
            return Ok(Stmt::Break(None));
        }
        if self.at(&Token::Continue) {
            self.bump();
            // Check for label: 'continue' | 'continue' ':' identifier
            if self.at(&Token::Colon) {
                self.bump();
                let label = self.expect_ident()?;
                return Ok(Stmt::Continue(Some(label)));
            }
            return Ok(Stmt::Continue(None));
        }
        if self.at(&Token::Try) {
            return self.parse_try_stmt();
        }
        if self.at(&Token::Throw) {
            self.bump();
            return Ok(Stmt::Throw(self.parse_expr()?));
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
        let is_parallel = if self.at(&Token::For) {
            self.bump();
            false
        } else if self.at(&Token::Parallel) {
            self.bump();
            self.expect(&Token::For)?;
            true
        } else {
            return Err("expected for or parallel for".to_string());
        };

        let var_name = self.expect_ident()?;

        if self.at(&Token::ColonEq) {
            self.bump();
            let start = self.parse_expr()?;
            self.expect(&Token::To)?;
            let end = self.parse_expr()?;
            let body = self.parse_block()?;
            if is_parallel {
                return Ok(Stmt::ForParallel {
                    var_name,
                    start,
                    end,
                    body,
                });
            }
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

    fn parse_try_stmt(&mut self) -> Result<Stmt, String> {
        self.expect(&Token::Try)?;
        let mut var_name = None;
        // Check for optional variable binding: 'try x = expr' or 'try expr'
        if let Token::Ident(name) = self.current().clone() {
            if matches!(self.peek(), Token::Eq) {
                self.bump();
                self.expect(&Token::Eq)?;
                var_name = Some(name);
            }
        }
        let expr = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let catch_block = if self.at(&Token::Catch) {
            self.bump();
            self.parse_block()?
        } else {
            Vec::new()
        };
        Ok(Stmt::Try {
            var_name,
            expr,
            then_block,
            catch_block,
        })
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_logical_or()
    }

    // Logical `||` then `&&`, the lowest-binding operators (below the bitwise
    // family), matching C. Both short-circuit at evaluation time.
    fn parse_logical_or(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_logical_and()?;
        while self.at(&Token::PipePipe) {
            self.bump();
            let rhs = self.parse_logical_and()?;
            expr = Expr::Binary(Box::new(expr), BinaryOp::Or, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_bitor()?;
        while self.at(&Token::AmpAmp) {
            self.bump();
            let rhs = self.parse_bitor()?;
            expr = Expr::Binary(Box::new(expr), BinaryOp::And, Box::new(rhs));
        }
        Ok(expr)
    }

    // Bitwise precedence, lowest-binding first (C order): `|` then `^` then `&`,
    // all below equality. Shifts (`<<`/`>>`) bind tighter than comparison and
    // looser than `+`/`-`, also matching C.
    fn parse_bitor(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_bitxor()?;
        while self.at(&Token::Pipe) {
            self.bump();
            let rhs = self.parse_bitxor()?;
            expr = Expr::Binary(Box::new(expr), BinaryOp::BitOr, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_bitand()?;
        while self.at(&Token::Caret) {
            self.bump();
            let rhs = self.parse_bitand()?;
            expr = Expr::Binary(Box::new(expr), BinaryOp::BitXor, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_bitand(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_equality()?;
        while self.at(&Token::Amp) {
            self.bump();
            let rhs = self.parse_equality()?;
            expr = Expr::Binary(Box::new(expr), BinaryOp::BitAnd, Box::new(rhs));
        }
        Ok(expr)
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
        let mut expr = self.parse_shift()?;
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
            let rhs = self.parse_shift()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_shift(&mut self) -> Result<Expr, String> {
        let mut expr = self.parse_additive()?;
        loop {
            let op = if self.at(&Token::Shl) {
                Some(BinaryOp::Shl)
            } else if self.at(&Token::Shr) {
                Some(BinaryOp::Shr)
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
        if self.at(&Token::Bang) {
            self.bump();
            return Ok(Expr::Unary(UnaryOp::Not, Box::new(self.parse_unary()?)));
        }
        if self.at(&Token::Tilde) {
            self.bump();
            return Ok(Expr::Unary(UnaryOp::BitNot, Box::new(self.parse_unary()?)));
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
            if self.at(&Token::Question) {
                self.bump();
                expr = Expr::Try(Box::new(expr));
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
            Token::Float(s) => {
                let f = s
                    .parse::<f64>()
                    .map_err(|err| format!("invalid float literal {s}: {err}"))?;
                self.bump();
                Ok(Expr::Float(f.to_bits()))
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
                } else if self.at(&Token::DoubleColon) {
                    self.bump();
                    let variant = self.expect_ident()?;
                    let mut fields = Vec::new();
                    if self.at(&Token::LParen) {
                        self.bump();
                        while !self.at(&Token::RParen) {
                            fields.push(self.parse_expr()?);
                            if self.at(&Token::Comma) {
                                self.bump();
                            }
                        }
                        self.expect(&Token::RParen)?;
                    }
                    Ok(Expr::EnumConstruct {
                        type_name: name,
                        variant,
                        fields,
                    })
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
            Token::Ident(type_name) => {
                let type_name = type_name.clone();
                self.bump();
                if self.at(&Token::DoubleColon) {
                    self.bump();
                    let variant = self.expect_ident()?;
                    if self.at(&Token::LParen) {
                        self.bump();
                        let _binding = self.expect_ident()?;
                        self.expect(&Token::RParen)?;
                        Ok(Pattern::EnumVariant(type_name, variant))
                    } else {
                        Ok(Pattern::EnumVariant(type_name, variant))
                    }
                } else {
                    Ok(Pattern::Ident(type_name))
                }
            }
            _ => Err(format!("unexpected token in pattern: {:?}", self.current())),
        }
    }

    fn parse_closure(&mut self) -> Result<Expr, String> {
        self.expect(&Token::Fn)?;
        let type_params = self.parse_type_params()?;
        let params = self.parse_params()?;
        if self.at(&Token::Arrow) {
            self.bump();
            self.skip_type_until(&[Token::LBrace])?;
        }
        let body = self.parse_block()?;
        Ok(Expr::Closure {
            params,
            body,
            type_params,
        })
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

struct Runtime {
    program: Program,
    global: Env,
    output: RefCell<Vec<String>>,
    input: RefCell<VecDeque<String>>,
    side_effects: RefCell<SideEffects>,
    file_handles: RefCell<HashMap<i64, RefCell<File>>>,
    next_file_id: RefCell<i64>,
    channels: RefCell<HashMap<i64, MutexChannel>>,
    next_channel_id: RefCell<i64>,
    mutexes: RefCell<HashMap<i64, MutexValue>>,
    next_mutex_id: RefCell<i64>,
    threads: RefCell<HashMap<i64, thread::JoinHandle<()>>>,
    next_thread_id: RefCell<i64>,
    monomorphized_functions: RefCell<HashMap<String, Function>>,
    type_inference_cache: RefCell<HashMap<String, Vec<String>>>,
    /// When true, filesystem builtins (`open_file`/`read_file`/`write_file`)
    /// are denied with an error. Set on the verify path so a candidate cannot
    /// touch the real filesystem (nondeterminism + host damage) during
    /// acceptance checking. Defaults to false so normal execution is unaffected.
    verify_mode: Cell<bool>,
}

struct MutexChannel {
    sender: Option<mpsc::Sender<Value>>,
    receiver: Option<mpsc::Receiver<Value>>,
}

impl MutexChannel {
    fn new() -> Self {
        let (sender, receiver) = mpsc::channel();
        Self {
            sender: Some(sender),
            receiver: Some(receiver),
        }
    }
}

struct MutexValue {
    value: RefCell<Option<Value>>,
    locked: RefCell<bool>,
}

impl MutexValue {
    fn new() -> Self {
        Self {
            value: RefCell::new(None),
            locked: RefCell::new(false),
        }
    }

    fn lock(&self) -> Result<(), String> {
        let mut locked = self.locked.borrow_mut();
        if *locked {
            return Err("mutex already locked".to_string());
        }
        *locked = true;
        Ok(())
    }

    fn unlock(&self) -> Result<(), String> {
        let mut locked = self.locked.borrow_mut();
        if !*locked {
            return Err("mutex not locked".to_string());
        }
        *locked = false;
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SideEffects {
    pub prints: Vec<String>,
    pub reads: Vec<String>,
    pub files_read: Vec<String>,
    pub files_written: Vec<String>,
}

impl SideEffects {
    fn new() -> Self {
        Self::default()
    }

    fn track_print(&mut self, value: &str) {
        self.prints.push(value.to_string());
    }

    fn track_read(&mut self, value: &str) {
        self.reads.push(value.to_string());
    }

    fn track_file_read(&mut self, path: &str) {
        self.files_read.push(path.to_string());
    }

    fn track_file_write(&mut self, path: &str) {
        self.files_written.push(path.to_string());
    }

    pub fn has_any(&self) -> bool {
        !self.prints.is_empty()
            || !self.reads.is_empty()
            || !self.files_read.is_empty()
            || !self.files_written.is_empty()
    }
}

impl Runtime {
    fn new(program: Program) -> Self {
        Self::with_input(program, Vec::new())
    }

    fn with_input(program: Program, input: Vec<String>) -> Self {
        let global = Env::new();

        // Initialize module environments
        for (module_name, module_decl) in &program.modules {
            let module_env = Env::with_module(module_name.clone());
            for export in &module_decl.exports {
                if let Some(_function) = program.functions.get(export) {
                    module_env.define(export, Value::Function(export.clone()));
                }
            }
            global.define_module(module_name, module_env);
        }

        // Process imports
        for import in &program.imports {
            let _ = global.resolve_import(&import.module_name, &import.symbols);
        }

        for name in program.functions.keys() {
            global.define(name, Value::Function(name.clone()));
        }

        // Register impl block methods
        for impl_block in &program.impls {
            for method in &impl_block.methods {
                // Create a qualified name for the method: Type::method_name
                let qualified_name = format!("{}::{}", impl_block.target_type, method.name);
                global.define(&qualified_name, Value::Function(qualified_name.clone()));
            }
        }
        global.define("println_i64", Value::Builtin(Builtin::PrintlnI64));
        global.define("println_f64", Value::Builtin(Builtin::PrintlnF64));
        global.define("println", Value::Builtin(Builtin::Println));
        global.define("print_f64", Value::Builtin(Builtin::PrintF64));
        global.define("print_string", Value::Builtin(Builtin::PrintString));
        global.define("print", Value::Builtin(Builtin::Print));
        global.define("read_i64", Value::Builtin(Builtin::ReadI64));
        global.define("read_string", Value::Builtin(Builtin::ReadString));
        global.define("read_line", Value::Builtin(Builtin::ReadString));
        global.define("read", Value::Builtin(Builtin::Read));
        global.define("has_input", Value::Builtin(Builtin::HasInput));
        global.define("len", Value::Builtin(Builtin::Len));
        global.define("abs", Value::Builtin(Builtin::Abs));
        global.define("min", Value::Builtin(Builtin::Min));
        global.define("max", Value::Builtin(Builtin::Max));
        global.define("pow", Value::Builtin(Builtin::Pow));
        global.define("open_file", Value::Builtin(Builtin::OpenFile));
        global.define("read_file", Value::Builtin(Builtin::ReadFile));
        global.define("write_file", Value::Builtin(Builtin::WriteFile));
        global.define("close_file", Value::Builtin(Builtin::CloseFile));
        global.define("reduce", Value::Builtin(Builtin::Reduce));
        global.define("spawn", Value::Builtin(Builtin::Spawn));
        global.define("send", Value::Builtin(Builtin::Send));
        global.define("recv", Value::Builtin(Builtin::Recv));
        global.define("new_channel", Value::Builtin(Builtin::NewChannel));
        global.define("new_mutex", Value::Builtin(Builtin::NewMutex));
        global.define("lock", Value::Builtin(Builtin::Lock));
        global.define("unlock", Value::Builtin(Builtin::Unlock));
        global.define("error", Value::Builtin(Builtin::Error));
        global.define("unwrap", Value::Builtin(Builtin::Unwrap));
        global.define("unwrap_or", Value::Builtin(Builtin::UnwrapOr));
        Self {
            program,
            global,
            output: RefCell::new(Vec::new()),
            input: RefCell::new(input.into_iter().collect()),
            side_effects: RefCell::new(SideEffects::new()),
            file_handles: RefCell::new(HashMap::new()),
            next_file_id: RefCell::new(1),
            channels: RefCell::new(HashMap::new()),
            next_channel_id: RefCell::new(1),
            mutexes: RefCell::new(HashMap::new()),
            next_mutex_id: RefCell::new(1),
            threads: RefCell::new(HashMap::new()),
            next_thread_id: RefCell::new(1),
            monomorphized_functions: RefCell::new(HashMap::new()),
            type_inference_cache: RefCell::new(HashMap::new()),
            verify_mode: Cell::new(false),
        }
    }

    /// Enable/disable verification mode. When enabled, filesystem builtins are
    /// denied so the verify path cannot touch the real filesystem.
    fn set_verify_mode(&self, on: bool) {
        self.verify_mode.set(on);
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
            side_effects: self.side_effects.borrow().clone(),
        })
    }

    fn call_function(&self, function_name: &str, args: Vec<Value>) -> Result<Value, String> {
        // Check for qualified method calls (Type::method)
        if function_name.contains("::") {
            let parts: Vec<&str> = function_name.split("::").collect();
            if parts.len() == 2 {
                let type_name = parts[0];
                let method_name = parts[1];

                // Find the impl block that contains this method
                for impl_block in &self.program.impls {
                    if impl_block.target_type == type_name {
                        if let Some(method) =
                            impl_block.methods.iter().find(|m| m.name == method_name)
                        {
                            // Set up the environment with 'self' bound to the first argument
                            let env = self.global.child();
                            if let Some(self_value) = args.first() {
                                env.define("self", self_value.clone());
                            }
                            // Bind remaining parameters
                            for (idx, param) in method.params.iter().enumerate().skip(1) {
                                let value = args.get(idx).cloned().unwrap_or(Value::Unit);
                                env.define(param, value);
                            }
                            return self.call_decl(method, args, env);
                        }
                    }
                }
                return Err(format!("unknown method {function_name}"));
            }
        }

        // First check if this is a generic function that needs monomorphization
        if let Some(base_function) = self.program.functions.get(function_name) {
            if !base_function.type_params.is_empty() {
                // Perform type inference from arguments
                let type_args = self.infer_type_args(&base_function.type_params, &args)?;
                let monomorphized_name = self.monomorphization_name(function_name, &type_args);

                // Check if we already have a monomorphized version
                let monomorphized = if let Some(existing) = self
                    .monomorphized_functions
                    .borrow()
                    .get(&monomorphized_name)
                {
                    existing.clone()
                } else {
                    // Create new monomorphized function
                    let mono_func = self.monomorphize_function(base_function, &type_args)?;
                    self.monomorphized_functions
                        .borrow_mut()
                        .insert(monomorphized_name.clone(), mono_func.clone());
                    mono_func
                };

                return self.call_decl(&monomorphized, args, self.global.clone());
            }
        }

        let function = self
            .program
            .functions
            .get(function_name)
            .cloned()
            .ok_or_else(|| format!("unknown function {function_name}"))?;
        self.call_decl(&function, args, self.global.clone())
    }

    fn infer_type_args(
        &self,
        type_params: &[String],
        args: &[Value],
    ) -> Result<Vec<String>, String> {
        let mut type_args = Vec::new();
        let mut cache = self.type_inference_cache.borrow_mut();

        // Infer type from arguments based on their value types
        for (i, arg) in args.iter().enumerate() {
            let type_name = match arg {
                Value::Int(_) => "i64",
                Value::Float(_) => "f64",
                Value::Bool(_) => "bool",
                Value::Str(_) => "string",
                Value::Array(items) if !items.is_empty() => {
                    // Infer element type from first element
                    match &items[0] {
                        Value::Int(_) => "array<i64>",
                        Value::Float(_) => "array<f64>",
                        Value::Bool(_) => "array<bool>",
                        Value::Str(_) => "array<string>",
                        _ => "array<unknown>",
                    }
                }
                Value::Array(_) => "array<unknown>",
                Value::Struct { name, .. } => name,
                Value::Enum { type_name, .. } => type_name,
                _ => "unknown",
            }
            .to_string();

            // Map type to type parameter if we have a corresponding parameter
            if i < type_params.len() {
                type_args.push(type_name.clone());
                // Cache this inference
                let key = format!("{}_arg_{}", type_params[i], i);
                cache.insert(key, vec![type_name.clone()]);
            }
        }

        // Fill remaining type parameters with defaults
        while type_args.len() < type_params.len() {
            type_args.push("unknown".to_string());
        }

        Ok(type_args)
    }

    fn monomorphization_name(&self, function_name: &str, type_args: &[String]) -> String {
        if type_args.is_empty() {
            return function_name.to_string();
        }
        let type_str = type_args.join(",");
        format!("{}::<{}>", function_name, type_str)
    }

    fn monomorphize_function(
        &self,
        function: &Function,
        type_args: &[String],
    ) -> Result<Function, String> {
        // Create a monomorphized version by substituting type parameters
        let mut mono_body = function.body.clone();

        // For now, we create a shallow monomorphization - the body is kept as-is
        // since our runtime is dynamically typed. In a full implementation,
        // we would walk the AST and replace type parameter references with
        // their concrete types.

        Ok(Function {
            name: self.monomorphization_name(&function.name, type_args),
            params: function.params.clone(),
            body: mono_body,
            type_params: Vec::new(), // No longer generic after monomorphization
            is_method: function.is_method,
        })
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
            Control::Break(_) => Err("break outside loop".to_string()),
            Control::Continue(_) => Err("continue outside loop".to_string()),
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

    fn exec_block_with_label(
        &self,
        stmts: &[Stmt],
        env: Env,
        current_label: Option<&str>,
    ) -> Result<Control, String> {
        for stmt in stmts {
            match self.exec_stmt_with_label(stmt, env.clone(), current_label)? {
                Control::Next => {}
                Control::Break(target_label) => {
                    if target_label.as_deref() == current_label || target_label.is_none() {
                        return Ok(Control::Break(target_label));
                    } else if target_label.is_some() {
                        return Ok(Control::Break(target_label));
                    }
                    return Ok(Control::Break(None));
                }
                Control::Continue(target_label) => {
                    if target_label.as_deref() == current_label || target_label.is_none() {
                        return Ok(Control::Continue(target_label));
                    } else if target_label.is_some() {
                        return Ok(Control::Continue(target_label));
                    }
                    return Ok(Control::Continue(None));
                }
                signal => return Ok(signal),
            }
        }
        Ok(Control::Next)
    }

    fn exec_stmt(&self, stmt: &Stmt, env: Env) -> Result<Control, String> {
        self.exec_stmt_with_label(stmt, env, None)
    }

    fn exec_stmt_with_label(
        &self,
        stmt: &Stmt,
        env: Env,
        current_label: Option<&str>,
    ) -> Result<Control, String> {
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
                    self.exec_block_with_label(then_block, env.child(), current_label)
                } else {
                    self.exec_block_with_label(else_block, env.child(), current_label)
                }
            }
            Stmt::While { condition, body } => {
                let mut iters = 0usize;
                while truthy(&self.eval_expr(condition, env.clone())?) {
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    match self.exec_block_with_label(body, env.child(), current_label)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break(target_label) => {
                            if target_label.is_none() || target_label.as_deref() == current_label {
                                break;
                            }
                            return Ok(Control::Break(target_label));
                        }
                        Control::Continue(target_label) => {
                            if target_label.is_some() && target_label.as_deref() != current_label {
                                return Ok(Control::Continue(target_label));
                            }
                            // Unlabeled continue or matching label continues this loop
                        }
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
                let mut iters = 0usize;
                for item in start..end {
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    let scope = env.child();
                    scope.define(var_name, Value::Int(item));
                    match self.exec_block_with_label(body, scope, current_label)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break(target_label) => {
                            if target_label.is_none() || target_label.as_deref() == current_label {
                                break;
                            }
                            return Ok(Control::Break(target_label));
                        }
                        Control::Continue(target_label) => {
                            if target_label.is_some() && target_label.as_deref() != current_label {
                                return Ok(Control::Continue(target_label));
                            }
                        }
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
                let mut iters = 0usize;
                for item in start..end {
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    let scope = env.child();
                    scope.define(var_name, Value::Int(item));
                    match self.exec_block_with_label(body, scope, current_label)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break(target_label) => {
                            if target_label.is_none() || target_label.as_deref() == current_label {
                                break;
                            }
                            return Ok(Control::Break(target_label));
                        }
                        Control::Continue(target_label) => {
                            if target_label.is_some() && target_label.as_deref() != current_label {
                                return Ok(Control::Continue(target_label));
                            }
                        }
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
                let mut iters = 0usize;
                for item in items {
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    let scope = env.child();
                    scope.define(var_name, item);
                    match self.exec_block_with_label(body, scope, current_label)? {
                        Control::Next => {}
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break(target_label) => {
                            if target_label.is_none() || target_label.as_deref() == current_label {
                                break;
                            }
                            return Ok(Control::Break(target_label));
                        }
                        Control::Continue(target_label) => {
                            if target_label.is_some() && target_label.as_deref() != current_label {
                                return Ok(Control::Continue(target_label));
                            }
                        }
                    }
                }
                Ok(Control::Next)
            }
            Stmt::ForParallel {
                var_name,
                start,
                end,
                body,
            } => {
                let start_val = expect_int(&self.eval_expr(start, env.clone())?)?;
                let end_val = expect_int(&self.eval_expr(end, env.clone())?)?;
                // Bound the range span BEFORE materializing it: `(start..end).collect()`
                // would eagerly allocate the entire i64 range, so a huge `end` OOMs the
                // process before any per-iteration cap could fire. Reject up front.
                if end_val.saturating_sub(start_val) > MAX_LOOP_ITERS as i64 {
                    return Err("loop exceeded iteration limit".to_string());
                }
                let range: Vec<i64> = (start_val..end_val).collect();

                // For now, implement parallel for sequentially
                // Full thread safety requires major refactoring of Value and Env
                let mut results = Vec::new();
                let mut iters = 0usize;
                for item in range {
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err("loop exceeded iteration limit".to_string());
                    }
                    let scope = env.child();
                    scope.define(var_name, Value::Int(item));
                    let scope_clone = scope.clone();
                    match self.exec_block_with_label(body, scope, current_label)? {
                        Control::Next => {
                            // Collect result from last expression if any
                            if let Some(Stmt::Expr(expr)) = body.last() {
                                if let Ok(value) = self.eval_expr(expr, scope_clone) {
                                    results.push(value);
                                }
                            }
                        }
                        Control::Return(value) => return Ok(Control::Return(value)),
                        Control::Break(target_label) => {
                            if target_label.is_none() || target_label.as_deref() == current_label {
                                break;
                            }
                            return Ok(Control::Break(target_label));
                        }
                        Control::Continue(target_label) => {
                            if target_label.is_some() && target_label.as_deref() != current_label {
                                return Ok(Control::Continue(target_label));
                            }
                        }
                    }
                }

                // Store results in an array in the environment
                let scope = env.child();
                scope.define(&format!("{}_results", var_name), Value::Array(results));

                Ok(Control::Next)
            }
            Stmt::Try {
                var_name,
                expr,
                then_block,
                catch_block,
            } => {
                match self.eval_expr(expr, env.clone()) {
                    Ok(Value::Result { is_ok: true, value }) => {
                        let scope = env.child();
                        if let Some(name) = var_name {
                            scope.define(&name, (*value).clone());
                        }
                        match self.exec_block_with_label(then_block, scope, current_label)? {
                            Control::Next => Ok(Control::Next),
                            signal => return Ok(signal),
                        }
                    }
                    Ok(Value::Result {
                        is_ok: false,
                        value,
                    }) => {
                        let scope = env.child();
                        let error_value = (*value).clone();
                        if let Some(name) = var_name {
                            scope.define(&name, error_value.clone());
                        }
                        if catch_block.is_empty() {
                            return Err(format!("uncaught error: {:?}", error_value));
                        }
                        match self.exec_block_with_label(&catch_block, scope, current_label)? {
                            Control::Next => Ok(Control::Next),
                            signal => return Ok(signal),
                        }
                    }
                    Ok(other) => {
                        // Non-Result values are treated as success
                        let scope = env.child();
                        if let Some(name) = var_name {
                            scope.define(&name, other);
                        }
                        match self.exec_block_with_label(then_block, scope, current_label)? {
                            Control::Next => Ok(Control::Next),
                            signal => return Ok(signal),
                        }
                    }
                    Err(err) => {
                        let scope = env.child();
                        if let Some(name) = var_name {
                            scope.define(&name, Value::Str(err.clone()));
                        }
                        if catch_block.is_empty() {
                            return Err(err.clone());
                        }
                        match self.exec_block_with_label(&catch_block, scope, current_label)? {
                            Control::Next => Ok(Control::Next),
                            signal => return Ok(signal),
                        }
                    }
                }
            }
            Stmt::Throw(expr) => {
                let value = self.eval_expr(expr, env)?;
                Err(format!("thrown error: {:?}", value))
            }
            Stmt::Break(label) => Ok(Control::Break(label.clone())),
            Stmt::Continue(label) => Ok(Control::Continue(label.clone())),
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
                    Value::Enum {
                        type_name: _,
                        variant: _,
                        fields,
                    } => {
                        let idx = field
                            .parse::<usize>()
                            .map_err(|_| format!("invalid enum field index {field}"))?;
                        if idx >= fields.len() {
                            return Err(format!("enum field index {idx} out of bounds"));
                        }
                        fields[idx] = value;
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
            Expr::Float(bits) => Ok(Value::Float(f64::from_bits(*bits))),
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
            Expr::EnumConstruct {
                type_name,
                variant,
                fields,
            } => Ok(Value::Enum {
                type_name: type_name.clone(),
                variant: variant.clone(),
                fields: fields
                    .iter()
                    .map(|expr| self.eval_expr(expr, env.clone()))
                    .collect::<Result<Vec<_>, _>>()?,
            }),
            Expr::Ident(name) => {
                // Handle 'self' as a reference to the current object
                if name == "self" {
                    if let Some(self_value) = env.get("self") {
                        return Ok(self_value);
                    }
                }
                env.get(name)
                    .ok_or_else(|| format!("undefined variable '{name}'"))
            }
            Expr::Unary(op, expr) => {
                let value = self.eval_expr(expr, env)?;
                match op {
                    UnaryOp::Neg => match value {
                        Value::Float(v) => Ok(Value::Float(-v)),
                        _ => Ok(Value::Int(-expect_int(&value)?)),
                    },
                    UnaryOp::Not => Ok(Value::Bool(!truthy(&value))),
                    UnaryOp::BitNot => Ok(Value::Int(!expect_int(&value)?)),
                }
            }
            Expr::Binary(lhs, op, rhs) => {
                // Logical `&&` / `||` short-circuit: the right side is only
                // evaluated when the left does not already decide the result, so
                // `false && (x / 0)` does not trap.
                if matches!(op, BinaryOp::And | BinaryOp::Or) {
                    let l = truthy(&self.eval_expr(lhs, env.clone())?);
                    let decided = match op {
                        BinaryOp::And => !l, // false && _ == false
                        _ => l,              // true  || _ == true
                    };
                    if decided {
                        return Ok(Value::Bool(l));
                    }
                    return Ok(Value::Bool(truthy(&self.eval_expr(rhs, env)?)));
                }
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
                    Value::Enum {
                        type_name: _,
                        variant: _,
                        fields,
                    } => {
                        let idx = field
                            .parse::<usize>()
                            .map_err(|_| format!("invalid enum field index {field}"))?;
                        fields
                            .get(idx)
                            .cloned()
                            .ok_or_else(|| format!("enum field index {idx} out of bounds"))
                    }
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
            Expr::Closure {
                params,
                body,
                type_params,
            } => Ok(Value::Closure(Closure {
                params: params.clone(),
                body: body.clone(),
                env,
                type_params: type_params.clone(),
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
            Expr::Try(expr) => {
                let value = self.eval_expr(expr, env)?;
                match value {
                    Value::Result { is_ok: true, value } => Ok(*value),
                    Value::Result {
                        is_ok: false,
                        value,
                    } => Err(format!("unwrap failed on error: {:?}", value)),
                    other => Ok(other),
                }
            }
        }
    }

    fn eval_binary(&self, lhs: Value, op: BinaryOp, rhs: Value) -> Result<Value, String> {
        // Float arithmetic: if either operand is a float, both are widened to
        // `f64` and the op is computed in floating point. This intercepts before
        // the integer paths below, so all-integer programs are unaffected.
        if matches!(lhs, Value::Float(_)) || matches!(rhs, Value::Float(_)) {
            let a = as_f64(&lhs)?;
            let b = as_f64(&rhs)?;
            return match op {
                BinaryOp::Add => Ok(Value::Float(a + b)),
                BinaryOp::Sub => Ok(Value::Float(a - b)),
                BinaryOp::Mul => Ok(Value::Float(a * b)),
                BinaryOp::Div => Ok(Value::Float(a / b)),
                BinaryOp::Mod => Ok(Value::Float(a % b)),
                BinaryOp::Lt => Ok(Value::Bool(a < b)),
                BinaryOp::Gt => Ok(Value::Bool(a > b)),
                BinaryOp::Le => Ok(Value::Bool(a <= b)),
                BinaryOp::Ge => Ok(Value::Bool(a >= b)),
                BinaryOp::Eq => Ok(Value::Bool(a == b)),
                BinaryOp::Ne => Ok(Value::Bool(a != b)),
                BinaryOp::BitAnd
                | BinaryOp::BitOr
                | BinaryOp::BitXor
                | BinaryOp::Shl
                | BinaryOp::Shr
                | BinaryOp::And
                | BinaryOp::Or => Err("bitwise/logical operator on a float".to_string()),
            };
        }
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
            BinaryOp::BitAnd => Ok(Value::Int(expect_int(&lhs)? & expect_int(&rhs)?)),
            BinaryOp::BitOr => Ok(Value::Int(expect_int(&lhs)? | expect_int(&rhs)?)),
            BinaryOp::BitXor => Ok(Value::Int(expect_int(&lhs)? ^ expect_int(&rhs)?)),
            BinaryOp::Shl => {
                let l = expect_int(&lhs)?;
                let r = expect_int(&rhs)?;
                if !(0..64).contains(&r) {
                    return Err("shift amount out of range in <<".to_string());
                }
                l.checked_shl(r as u32)
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in <<".to_string())
            }
            BinaryOp::Shr => {
                let l = expect_int(&lhs)?;
                let r = expect_int(&rhs)?;
                if !(0..64).contains(&r) {
                    return Err("shift amount out of range in >>".to_string());
                }
                Ok(Value::Int(l >> r))
            }
            // `&&` / `||` are normally short-circuited in `eval_expr`; this eager
            // path keeps the match exhaustive if they are ever evaluated directly.
            BinaryOp::And => Ok(Value::Bool(truthy(&lhs) && truthy(&rhs))),
            BinaryOp::Or => Ok(Value::Bool(truthy(&lhs) || truthy(&rhs))),
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
                // Handle module-qualified function calls (e.g., "utils::helper")
                let function_name = if name.contains("::") {
                    let parts: Vec<&str> = name.split("::").collect();
                    if parts.len() == 2 {
                        parts[1] // Extract the function name from "module::function"
                    } else {
                        &name
                    }
                } else {
                    &name
                };

                let function = self
                    .program
                    .functions
                    .get(function_name)
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
                    Control::Break(_) => Err("break outside loop".to_string()),
                    Control::Continue(_) => Err("continue outside loop".to_string()),
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

        // First check for trait methods based on the type
        let type_name = match &base_value {
            Value::Struct { name, .. } => Some(name.clone()),
            Value::Enum { type_name, .. } => Some(type_name.clone()),
            Value::Array(_) => Some("Array".to_string()),
            Value::Str(_) => Some("String".to_string()),
            Value::Int(_) => Some("i64".to_string()),
            Value::Float(_) => Some("f64".to_string()),
            Value::Bool(_) => Some("bool".to_string()),
            _ => None,
        };

        if let Some(ty) = type_name {
            // Check impl blocks for this type
            for impl_block in &self.program.impls {
                if impl_block.target_type == ty {
                    if let Some(method_fn) = impl_block.methods.iter().find(|m| m.name == method) {
                        // Set up environment with 'self' as the base value
                        let method_env = env.child();
                        method_env.define("self", base_value.clone());
                        method_env.define("self_value", base_value.clone());

                        // Build full argument list: self + args
                        let mut full_args = vec![base_value.clone()];
                        full_args.extend(args);

                        return self.call_decl(method_fn, full_args, method_env);
                    }
                }
            }
        }

        // Fall back to built-in methods
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
            Builtin::PrintF64 | Builtin::PrintlnF64 => {
                let value = args
                    .first()
                    .ok_or_else(|| "print_f64 requires one argument".to_string())?;
                let line = match value {
                    Value::Float(v) => format!("{:.7}", v),
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
                    output.push(text.clone());
                }
                self.side_effects.borrow_mut().track_print(&text);
                Ok(Value::Unit)
            }
            Builtin::Print => {
                let text = args
                    .iter()
                    .map(display_value)
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");
                let mut output = self.output.borrow_mut();
                if let Some(last) = output.last_mut() {
                    last.push_str(&text);
                } else {
                    output.push(text.clone());
                }
                self.side_effects.borrow_mut().track_print(&text);
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
                self.side_effects.borrow_mut().track_read(&raw);
                Ok(Value::Str(raw))
            }
            Builtin::Read => {
                if !args.is_empty() {
                    return Err("read takes no arguments".to_string());
                }
                let raw = {
                    let mut input = self.input.borrow_mut();
                    input
                        .pop_front()
                        .ok_or_else(|| "read: no input available".to_string())?
                };
                self.side_effects.borrow_mut().track_read(&raw);
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
                    Value::Enum { fields, .. } => fields.len() as i64,
                    other => return Err(format!("len unsupported for {:?}", other)),
                };
                Ok(Value::Int(len))
            }
            Builtin::Abs => {
                let value = args
                    .first()
                    .ok_or_else(|| "abs requires one argument".to_string())?;
                // `i64::abs()` PANICS on i64::MIN (its negation overflows).
                // Use checked_abs -> None -> Err, matching the checked_add/sub/mul
                // overflow policy.
                expect_int(value)?
                    .checked_abs()
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in abs".to_string())
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
                let base = expect_int(&args[0])?;
                let exp = expect_int(&args[1])?;
                // Two bugs in the old `.pow(exp as u32)`: (1) a negative or
                // >u32::MAX exponent silently wrapped into a huge u32; (2)
                // i64::pow PANICS on overflow. Reject out-of-range exponents
                // and use checked_pow -> None -> Err for overflow.
                let exp_u32 = u32::try_from(exp)
                    .map_err(|_| "pow exponent out of range".to_string())?;
                base.checked_pow(exp_u32)
                    .map(Value::Int)
                    .ok_or_else(|| "integer overflow in pow".to_string())
            }
            Builtin::OpenFile => {
                if self.verify_mode.get() {
                    return Err("file I/O denied during verification".to_string());
                }
                let path = args
                    .first()
                    .ok_or_else(|| "open_file requires path argument".to_string())?;
                let path = match path {
                    Value::Str(p) => p,
                    other => return Err(format!("open_file path must be string, got {:?}", other)),
                };
                let mode = args
                    .get(1)
                    .ok_or_else(|| "open_file requires mode argument".to_string())?;
                let mode_str = match mode {
                    Value::Str(m) => m.as_str(),
                    other => return Err(format!("open_file mode must be string, got {:?}", other)),
                };

                let file_id = *self.next_file_id.borrow();
                *self.next_file_id.borrow_mut() += 1;

                let options = match mode_str {
                    "r" => OpenOptions::new().read(true).open(path),
                    "w" => OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(true)
                        .open(path),
                    "a" => OpenOptions::new()
                        .write(true)
                        .append(true)
                        .create(true)
                        .open(path),
                    "rw" | "wr" => OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .open(path),
                    other => return Err(format!("open_file: invalid mode '{}'", mode_str)),
                };

                let file = options.map_err(|e| format!("open_file: {}", e))?;
                self.file_handles
                    .borrow_mut()
                    .insert(file_id, RefCell::new(file));
                Ok(Value::FileHandle(file_id))
            }
            Builtin::ReadFile => {
                if self.verify_mode.get() {
                    return Err("file I/O denied during verification".to_string());
                }
                let handle_value = args
                    .first()
                    .ok_or_else(|| "read_file requires file handle argument".to_string())?;
                let handle_id = match handle_value {
                    Value::FileHandle(id) => *id,
                    other => return Err(format!("read_file expects file handle, got {:?}", other)),
                };

                let handles = self.file_handles.borrow();
                let file_cell = handles
                    .get(&handle_id)
                    .ok_or_else(|| format!("read_file: invalid file handle {}", handle_id))?;

                let mut file = file_cell.borrow_mut();
                let mut content = String::new();
                file.read_to_string(&mut content)
                    .map_err(|e| format!("read_file: {}", e))?;

                Ok(Value::Str(content))
            }
            Builtin::WriteFile => {
                if self.verify_mode.get() {
                    return Err("file I/O denied during verification".to_string());
                }
                if args.len() != 2 {
                    return Err("write_file requires handle and content arguments".to_string());
                }
                let handle_value = &args[0];
                let handle_id = match handle_value {
                    Value::FileHandle(id) => *id,
                    other => {
                        return Err(format!("write_file expects file handle, got {:?}", other))
                    }
                };

                let content = match &args[1] {
                    Value::Str(s) => s,
                    other => {
                        return Err(format!(
                            "write_file content must be string, got {:?}",
                            other
                        ))
                    }
                };

                let handles = self.file_handles.borrow();
                let file_cell = handles
                    .get(&handle_id)
                    .ok_or_else(|| format!("write_file: invalid file handle {}", handle_id))?;

                let mut file = file_cell.borrow_mut();
                file.write_all(content.as_bytes())
                    .map_err(|e| format!("write_file: {}", e))?;
                file.flush()
                    .map_err(|e| format!("write_file: flush failed: {}", e))?;

                Ok(Value::Unit)
            }
            Builtin::CloseFile => {
                let handle_value = args
                    .first()
                    .ok_or_else(|| "close_file requires file handle argument".to_string())?;
                let handle_id = match handle_value {
                    Value::FileHandle(id) => *id,
                    other => {
                        return Err(format!("close_file expects file handle, got {:?}", other))
                    }
                };

                self.file_handles
                    .borrow_mut()
                    .remove(&handle_id)
                    .ok_or_else(|| format!("close_file: invalid file handle {}", handle_id))?;

                Ok(Value::Unit)
            }
            Builtin::Reduce => {
                if args.len() != 2 {
                    return Err("reduce requires array and closure arguments".to_string());
                }
                let array = match &args[0] {
                    Value::Array(items) => items.clone(),
                    other => {
                        return Err(format!(
                            "reduce first argument must be array, got {:?}",
                            other
                        ))
                    }
                };
                let closure = match &args[1] {
                    Value::Closure(c) => c.clone(),
                    other => {
                        return Err(format!(
                            "reduce second argument must be closure, got {:?}",
                            other
                        ))
                    }
                };
                if array.is_empty() {
                    return Err("reduce cannot be called on empty array".to_string());
                }
                let mut accumulator = array[0].clone();
                for item in array.iter().skip(1) {
                    accumulator = self.call_value(
                        Value::Closure(closure.clone()),
                        vec![accumulator, item.clone()],
                    )?;
                }
                Ok(accumulator)
            }
            Builtin::NewChannel => {
                let channel_id = *self.next_channel_id.borrow();
                *self.next_channel_id.borrow_mut() += 1;

                let channel = MutexChannel::new();
                self.channels.borrow_mut().insert(channel_id, channel);

                Ok(Value::Channel(channel_id))
            }
            Builtin::Send => {
                if args.len() != 2 {
                    return Err("send requires channel and value arguments".to_string());
                }
                let channel_id = match &args[0] {
                    Value::Channel(id) => *id,
                    other => {
                        return Err(format!(
                            "send first argument must be channel, got {:?}",
                            other
                        ))
                    }
                };
                let value = args[1].clone();

                let channels = self.channels.borrow();
                let channel = channels
                    .get(&channel_id)
                    .ok_or_else(|| format!("send: invalid channel {}", channel_id))?;

                if let Some(sender) = &channel.sender {
                    sender
                        .send(value)
                        .map_err(|e| format!("send failed: {}", e))?;
                } else {
                    return Err("send: channel sender disconnected".to_string());
                }

                Ok(Value::Unit)
            }
            Builtin::Recv => {
                if args.len() != 1 {
                    return Err("recv requires channel argument".to_string());
                }
                let channel_id = match &args[0] {
                    Value::Channel(id) => *id,
                    other => return Err(format!("recv argument must be channel, got {:?}", other)),
                };

                let channels = self.channels.borrow();
                let channel = channels
                    .get(&channel_id)
                    .ok_or_else(|| format!("recv: invalid channel {}", channel_id))?;

                if let Some(receiver) = &channel.receiver {
                    let value = receiver.recv().map_err(|e| format!("recv failed: {}", e))?;
                    Ok(value)
                } else {
                    return Err("recv: channel receiver disconnected".to_string());
                }
            }
            Builtin::NewMutex => {
                let mutex_id = *self.next_mutex_id.borrow();
                *self.next_mutex_id.borrow_mut() += 1;

                let mutex = MutexValue::new();
                self.mutexes.borrow_mut().insert(mutex_id, mutex);

                Ok(Value::Mutex(mutex_id))
            }
            Builtin::Lock => {
                if args.len() != 1 {
                    return Err("lock requires mutex argument".to_string());
                }
                let mutex_id = match &args[0] {
                    Value::Mutex(id) => *id,
                    other => return Err(format!("lock argument must be mutex, got {:?}", other)),
                };

                let mutexes = self.mutexes.borrow();
                let mutex = mutexes
                    .get(&mutex_id)
                    .ok_or_else(|| format!("lock: invalid mutex {}", mutex_id))?;

                mutex.lock()?;
                Ok(Value::Unit)
            }
            Builtin::Unlock => {
                if args.len() != 1 {
                    return Err("unlock requires mutex argument".to_string());
                }
                let mutex_id = match &args[0] {
                    Value::Mutex(id) => *id,
                    other => return Err(format!("unlock argument must be mutex, got {:?}", other)),
                };

                let mutexes = self.mutexes.borrow();
                let mutex = mutexes
                    .get(&mutex_id)
                    .ok_or_else(|| format!("unlock: invalid mutex {}", mutex_id))?;

                mutex.unlock()?;
                Ok(Value::Unit)
            }
            Builtin::Spawn => {
                if args.len() != 1 {
                    return Err("spawn requires closure argument".to_string());
                }
                let closure = match &args[0] {
                    Value::Closure(c) => c.clone(),
                    other => {
                        return Err(format!("spawn argument must be closure, got {:?}", other))
                    }
                };

                let thread_id = *self.next_thread_id.borrow();
                *self.next_thread_id.borrow_mut() += 1;

                // In a full implementation, we'd need to clone the entire runtime
                // For now, we create a simplified thread that captures the closure
                let handle = thread::spawn(move || {
                    // The closure would execute here with its captured environment
                    // For this implementation, we just note that the thread was spawned
                });

                self.threads.borrow_mut().insert(thread_id, handle);

                Ok(Value::ThreadHandle(thread_id))
            }
            Builtin::ParallelFor => {
                // ParallelFor is handled through ForParallel stmt, not as a builtin
                Err("parallel_for should use 'parallel for' syntax, not a builtin call".to_string())
            }
            Builtin::Error => {
                let message = args
                    .first()
                    .ok_or_else(|| "error requires one argument".to_string())?;
                let message = match message {
                    Value::Str(s) => s,
                    other => return Err(format!("error message must be string, got {:?}", other)),
                };
                Err(message.clone())
            }
            Builtin::Unwrap => {
                let value = args
                    .first()
                    .ok_or_else(|| "unwrap requires one argument".to_string())?;
                match value {
                    Value::Result { is_ok: true, value } => Ok(*value.clone()),
                    Value::Result {
                        is_ok: false,
                        value,
                    } => Err(format!("unwrap failed on error: {:?}", value)),
                    Value::Optional {
                        is_some: true,
                        value,
                    } => Ok(*value.clone()),
                    Value::Optional { is_some: false, .. } => {
                        Err("unwrap failed on None".to_string())
                    }
                    other => Ok(other.clone()),
                }
            }
            Builtin::UnwrapOr => {
                if args.len() != 2 {
                    return Err("unwrap_or requires two arguments".to_string());
                }
                let value = &args[0];
                let default = &args[1];
                match value {
                    Value::Result { is_ok: true, value } => Ok(*value.clone()),
                    Value::Result { is_ok: false, .. } => Ok(default.clone()),
                    Value::Optional {
                        is_some: true,
                        value,
                    } => Ok(*value.clone()),
                    Value::Optional { is_some: false, .. } => Ok(default.clone()),
                    other => Ok(other.clone()),
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum Control {
    Next,
    Return(Value),
    Break(Option<String>),
    Continue(Option<String>),
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(v) => *v,
        Value::Int(v) => *v != 0,
        Value::Str(v) => !v.is_empty(),
        Value::Array(v) => !v.is_empty(),
        Value::Optional { is_some, .. } => *is_some,
        Value::Enum { .. } => true,
        Value::Unit => false,
        _ => true,
    }
}

fn value_eq(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Str(a), Value::Str(b)) => a == b,
        (
            Value::Enum {
                type_name: a,
                variant: va,
                fields: fa,
            },
            Value::Enum {
                type_name: b,
                variant: vb,
                fields: fb,
            },
        ) => {
            a == b
                && va == vb
                && fa.len() == fb.len()
                && fa.iter().zip(fb.iter()).all(|(x, y)| value_eq(x, y))
        }
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
        Pattern::EnumVariant(type_name, variant) => {
            if let Value::Enum {
                type_name: tn,
                variant: vn,
                fields,
            } = value
            {
                if type_name == tn && variant == vn {
                    if !fields.is_empty() {
                        env.define(variant, fields[0].clone());
                    }
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }
        Pattern::Ident(binding) => {
            env.define(binding, value.clone());
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::benchmark::{get_benchmark, Example, Value as BmValue};
    use crate::solver::solve_problem;

    use super::*;

    fn assert_program_output(code: &str, expected: &str) {
        let result = execute_program(code)
            .unwrap_or_else(|err| panic!("program execution failed: {err}\n\n{code}"));
        assert_eq!(result.output, expected);
    }

    #[test]
    fn executes_bitwise_and_shift_operators() {
        // `&`, `|`, `^`, `<<`, `>>` parse with C precedence and evaluate on ints.
        // `(13 & 6)` = 4, `(4 | 1)` = 5, `(5 ^ 3)` = 6, `(6 << 2)` = 24,
        // `(24 >> 1)` = 12.  Precedence: `1 + 2 << 1` = `(1+2) << 1` = 6 (shift
        // binds looser than +), and `6 & 3 == 2` parses as `6 & (3 == 2)` is NOT
        // wanted; we test the arithmetic precedence that matters for codegen.
        let code = r#"
fn main() -> i64 {
    a := 13 & 6;
    b := a | 1;
    c := b ^ 3;
    d := c << 2;
    e := d >> 1;
    f := 1 + 2 << 1;
    println_i64(a);
    println_i64(b);
    println_i64(c);
    println_i64(d);
    println_i64(e);
    println_i64(f);
    return 0;
}
"#;
        assert_program_output(code, "4\n5\n6\n24\n12\n6");
    }

    #[test]
    fn executes_float_arithmetic() {
        // Float literals parse, mixed int/float coerces, `/` is true division.
        // f(x=2.0) = 2.5*2.0 + 1.5 = 6.5 ; 1/2 in float context = 0.5 ; print
        // formats floats. Verifies the f64 path end to end.
        let code = r#"
fn main() -> f64 {
    x := 2.0;
    y := 2.5 * x + 1.5;
    z := 1.0 / 2.0;
    w := 3 + 0.5;
    println_f64(y);
    println_f64(z);
    println_f64(w);
    return y;
}
"#;
        let result = execute_program(code).expect("float program runs");
        assert!(result.output.starts_with("6.5"), "got {}", result.output);
        assert!(result.output.contains("0.5"), "got {}", result.output);
        assert!(result.output.contains("3.5"), "got {}", result.output);
    }

    #[test]
    fn executes_logical_and_unary_operators() {
        // `&&`, `||`, `!`, `~` parse with C precedence and short-circuit.
        // x=5: (x>0 && x<10) true -> 1; (x<0 || x==5) true -> 1;
        // !(x==4) -> true -> 1; ~x = -(x)-1 = -6; (~x) + 7 = 1.
        // Short-circuit: (x != 0 && (100 / x) > 10) -> 100/5=20>10 -> 1.
        let code = r#"
fn main() -> i64 {
    x := 5;
    a := 0;
    if x > 0 && x < 10 {
        a = a + 1;
    }
    if x < 0 || x == 5 {
        a = a + 1;
    }
    if !(x == 4) {
        a = a + 1;
    }
    b := (~x) + 7;
    if x != 0 && (100 / x) > 10 {
        a = a + 1;
    }
    println_i64(a);
    println_i64(b);
    return 0;
}
"#;
        assert_program_output(code, "4\n1");
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
        use crate::solver::solve_problem_search_only;
        for problem in get_benchmark(1) {
            let result = solve_problem_search_only(&problem);
            assert!(result.success, "search solver failed for {}", problem.name);
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

    #[test]
    fn verifies_bool_expected_against_int_output() {
        // The i64 0/1 lane verifies against a `Value::Bool` expected:
        // a predicate that returns 1 must verify when expected is `true`,
        // and a predicate that returns 0 must verify when expected is
        // `false`. This is the bridge the new bool-expected problem
        // shape relies on.
        let problem = Problem {
            name: "is_positive_bool_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn is_positive_bool_v0(a: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(3)],
                    expected: BmValue::Bool(true),
                },
                Example {
                    inputs: vec![BmValue::Int(-5)],
                    expected: BmValue::Bool(false),
                },
                Example {
                    inputs: vec![BmValue::Int(0)],
                    expected: BmValue::Bool(false),
                },
                Example {
                    inputs: vec![BmValue::Int(10)],
                    expected: BmValue::Bool(true),
                },
            ],
            holdouts: vec![],
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        };
        let code = "fn is_positive_bool_v0(x: i64) -> i64 {\n    if 0 < x {\n        return 1;\n    }\n    return 0;\n}\n";
        verify_problem_code(&problem, code)
            .unwrap_or_else(|err| panic!("bool→int verify failed: {err}"));
    }

    #[test]
    fn verifies_int_expected_against_bool_output() {
        // The mirror of the above: a `Value::Bool` actual output (e.g.
        // a literal `if x > 0 { true } else { false }`) verifies against
        // an i64 0/1 expected. This is the bridge the other way.
        let problem = Problem {
            name: "is_positive_bool_to_int_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn is_positive_bool_to_int_v0(a: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(3)],
                    expected: BmValue::Int(1),
                },
                Example {
                    inputs: vec![BmValue::Int(-5)],
                    expected: BmValue::Int(0),
                },
            ],
            holdouts: vec![],
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        };
        let code = "fn is_positive_bool_to_int_v0(x: i64) -> i64 {\n    if 0 < x {\n        return 1;\n    } else {\n        return 0;\n    }\n}\n";
        verify_problem_code(&problem, code)
            .unwrap_or_else(|err| panic!("int→bool verify failed: {err}"));
    }

    #[test]
    fn verifies_float_expected_against_int_output() {
        // A solver that emits an i64 result verifies against a
        // `Value::Float` expected (so the int lane is interchangeable
        // with the float lane on the wire — same trick that opened
        // the bool lane).
        let problem = Problem {
            name: "double_float_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn double_float_v0(a: i64) -> i64",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(3)],
                    expected: BmValue::Float(6.0_f64.to_bits()),
                },
                Example {
                    inputs: vec![BmValue::Int(-5)],
                    expected: BmValue::Float((-10.0_f64).to_bits()),
                },
            ],
            holdouts: vec![],
            reference_code: "",

            synthetic_args: Vec::new(),

            synthetic_values: Vec::new(),

            recursive_allowed: false,

            tree_input: false,

            explicit_stack: false,
            functions: vec![],
        };
        let code = "fn double_float_v0(x: i64) -> i64 {\n    return x * 2;\n}\n";
        verify_problem_code(&problem, code)
            .unwrap_or_else(|err| panic!("int→float verify failed: {err}"));
    }

    #[test]
    fn validates_call_graph_with_valid_program() {
        let code = r#"
fn add(a: i64, b: i64) -> i64 {
    return a + b;
}

fn main() -> i64 {
    let result = add(5, 3);
    return result;
}
"#;
        let result = validate_call_graph(code);
        assert!(result.is_ok(), "Valid program should pass: {:?}", result);
    }

    #[test]
    fn validates_call_graph_detects_undefined_function() {
        let code = r#"
fn main() -> i64 {
    let result = undefined_func(5, 3);
    return result;
}
"#;
        let result = validate_call_graph(code);
        assert!(result.is_err(), "Should detect undefined function");
        assert!(result.unwrap_err().contains("undefined_func"));
    }

    #[test]
    fn validates_call_graph_with_builtin_functions() {
        let code = r#"
fn main() -> i64 {
    let result = abs(-5);
    println_i64(result);
    return result;
}
"#;
        let result = validate_call_graph(code);
        assert!(
            result.is_ok(),
            "Built-in functions should be valid: {:?}",
            result
        );
    }

    #[test]
    fn validates_call_graph_with_method_calls() {
        let code = r#"
fn main() -> i64 {
    let arr = [1, 2, 3];
    arr.push(4);
    let s = "hello";
    s.upper();
    return 0;
}
"#;
        let result = validate_call_graph(code);
        assert!(result.is_ok(), "Method calls should be valid: {:?}", result);
    }

    #[test]
    fn validates_call_graph_with_closure() {
        let code = r#"
fn main() -> i64 {
    let nums = [1, 2, 3];
    let doubled = nums.map(fn(x) { return x * 2; });
    return 0;
}
"#;
        let result = validate_call_graph(code);
        assert!(result.is_ok(), "Closures should be valid: {:?}", result);
    }

    #[test]
    fn validates_call_graph_multiple_undefined_functions() {
        let code = r#"
fn main() -> i64 {
    foo();
    bar();
    return 0;
}
"#;
        let result = validate_call_graph(code);
        assert!(result.is_err(), "Should detect undefined functions");
        let err = result.unwrap_err();
        assert!(err.contains("foo") || err.contains("bar"));
    }

    #[test]
    fn validates_call_graph_with_recursive_call() {
        let code = r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}

fn main() -> i64 {
    return factorial(5);
}
"#;
        let result = validate_call_graph(code);
        assert!(
            result.is_ok(),
            "Recursive calls should be valid: {:?}",
            result
        );
    }

    #[test]
    fn parses_and_executes_module_with_imports() {
        let code = r#"
mod math {
    export add;
    export multiply;

    fn add(a: i64, b: i64) -> i64 {
        return a + b;
    }

    fn multiply(a: i64, b: i64) -> i64 {
        return a * b;
    }
}

import math: {add, multiply};

fn main() -> i64 {
    result := add(5, 3);
    product := multiply(4, 7);
    println_i64(result);
    println_i64(product);
    return 0;
}
"#;
        let result = execute_program(code).expect("module program should execute");
        assert_eq!(result.output, "8\n28");
    }

    #[test]
    fn parses_module_with_exports() {
        let code = r#"
mod utils {
    export helper;

    fn helper(x: i64) -> i64 {
        return x * 2;
    }
}

fn main() -> i64 {
    return 0;
}
"#;
        let result = parse_program(code);
        assert!(
            result.is_ok(),
            "Module parsing should succeed: {:?}",
            result
        );
        let program = result.unwrap();
        assert!(program.modules.contains_key("utils"));
        assert_eq!(program.modules["utils"].exports, vec!["helper"]);
    }

    #[test]
    fn parses_import_with_specific_symbols() {
        let code = r#"
import math: {add, subtract};

fn main() -> i64 {
    return 0;
}
"#;
        let result = parse_program(code);
        assert!(
            result.is_ok(),
            "Import parsing should succeed: {:?}",
            result
        );
        let program = result.unwrap();
        assert_eq!(program.imports.len(), 1);
        assert_eq!(program.imports[0].module_name, "math");
        assert_eq!(program.imports[0].symbols, vec!["add", "subtract"]);
    }

    #[test]
    fn parses_module_with_multiple_exports() {
        let code = r#"
mod api {
    export create;
    export destroy;
    export update;

    fn create() -> i64 {
        return 1;
    }

    fn destroy() -> i64 {
        return 0;
    }

    fn update() -> i64 {
        return 2;
    }
}

fn main() -> i64 {
    return 0;
}
"#;
        let result = parse_program(code);
        assert!(
            result.is_ok(),
            "Multiple export parsing should succeed: {:?}",
            result
        );
        let program = result.unwrap();
        assert_eq!(program.modules["api"].exports.len(), 3);
    }

    // ------------------------------------------------------------------
    // Verification I/O contract: equality oracle + input conversion.
    // ------------------------------------------------------------------

    fn tnode(value: i64, left: i32, right: i32) -> crate::benchmark::TreeNode {
        crate::benchmark::TreeNode { value, left, right }
    }

    fn struct_value(name: &str, fields: &[(&str, Value)]) -> Value {
        Value::Struct {
            name: name.to_string(),
            fields: fields
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    #[test]
    fn output_matches_arrays_recurse_elementwise() {
        // Equal int arrays match; differing length or element does not.
        let actual = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert!(output_matches(&actual, &BmValue::int_array(&[1, 2, 3])));
        assert!(!output_matches(&actual, &BmValue::int_array(&[1, 2])));
        assert!(!output_matches(&actual, &BmValue::int_array(&[1, 2, 4])));
        // Bool/float array elements bridge to the wire i64 (no loosening past 0/1).
        let bridged = Value::Array(vec![Value::Bool(true), Value::Float(0.0)]);
        assert!(output_matches(&bridged, &BmValue::int_array(&[1, 0])));
        assert!(!output_matches(&bridged, &BmValue::int_array(&[1, 1])));
        // A non-scalar element never matches a scalar slot (strictness).
        let nested = Value::Array(vec![Value::Str("x".to_string())]);
        assert!(!output_matches(&nested, &BmValue::int_array(&[0])));
    }

    #[test]
    fn output_matches_typed_and_nested_arrays_strict() {
        // The widened wire array (`Vec<Value>`) lets the oracle recurse on each
        // element through `output_matches`, so STRING, FLOAT, and NESTED arrays
        // verify with the same strict bridges as scalars — nothing looser.

        // String-element array: matches the same strings, rejects a different one.
        let strs = Value::Array(vec![
            Value::Str("a".to_string()),
            Value::Str("b".to_string()),
        ]);
        let want_strs = BmValue::array_of(vec![
            BmValue::Str("a".to_string()),
            BmValue::Str("b".to_string()),
        ]);
        assert!(output_matches(&strs, &want_strs));
        let other_strs = BmValue::array_of(vec![
            BmValue::Str("a".to_string()),
            BmValue::Str("c".to_string()),
        ]);
        assert!(!output_matches(&strs, &other_strs));
        // A string element NEVER matches an int element (strict, no coercion).
        assert!(!output_matches(&strs, &BmValue::array_of(vec![BmValue::Str("a".to_string()), BmValue::Int(0)])));

        // Float-element array: a runtime float matches a wire float (exact bits
        // via the existing float bridge) and bridges to the equal-valued int.
        let floats = Value::Array(vec![Value::Float(1.5), Value::Float(2.0)]);
        let want_floats = BmValue::array_of(vec![
            BmValue::Float(1.5_f64.to_bits()),
            BmValue::Float(2.0_f64.to_bits()),
        ]);
        assert!(output_matches(&floats, &want_floats));
        // 2.0 (whole) still bridges to int 2 elementwise; 1.5 does NOT bridge to 1.
        assert!(!output_matches(&floats, &BmValue::array_of(vec![BmValue::Int(1), BmValue::Int(2)])));

        // Nested array (array-of-arrays): each inner array recurses.
        let nested = Value::Array(vec![
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
            Value::Array(vec![Value::Int(3)]),
        ]);
        let want_nested = BmValue::array_of(vec![
            BmValue::int_array(&[1, 2]),
            BmValue::int_array(&[3]),
        ]);
        assert!(output_matches(&nested, &want_nested));
        // A different inner element fails; a different inner shape fails.
        assert!(!output_matches(&nested, &BmValue::array_of(vec![BmValue::int_array(&[1, 9]), BmValue::int_array(&[3])])));
        assert!(!output_matches(&nested, &BmValue::array_of(vec![BmValue::int_array(&[1, 2]), BmValue::int_array(&[3, 4])])));
        // Outer length mismatch fails.
        assert!(!output_matches(&nested, &BmValue::array_of(vec![BmValue::int_array(&[1, 2])])));
    }

    #[test]
    fn verify_problem_code_strict_array_output_through_widened_wire() {
        // End-to-end: a problem whose I/O is an ARRAY verifies through the full
        // strict path (`verify_problem_code_via_main` stdout check + holdouts via
        // `output_matches`) with the expected arrays built on the widened wire
        // type (`array_of` of `Value`s). The Mog reference returns an array
        // literal `[x, x * 2]`.
        let problem = Problem {
            name: "double_pair_array_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn double_pair_array_v0(x: i64) -> [i64]",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(3)],
                    expected: BmValue::array_of(vec![BmValue::Int(3), BmValue::Int(6)]),
                },
                Example {
                    inputs: vec![BmValue::Int(0)],
                    expected: BmValue::array_of(vec![BmValue::Int(0), BmValue::Int(0)]),
                },
                Example {
                    inputs: vec![BmValue::Int(-4)],
                    expected: BmValue::array_of(vec![BmValue::Int(-4), BmValue::Int(-8)]),
                },
            ],
            holdouts: vec![
                Example {
                    inputs: vec![BmValue::Int(7)],
                    expected: BmValue::array_of(vec![BmValue::Int(7), BmValue::Int(14)]),
                },
                Example {
                    inputs: vec![BmValue::Int(-1)],
                    expected: BmValue::array_of(vec![BmValue::Int(-1), BmValue::Int(-2)]),
                },
            ],
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };
        let code =
            "fn double_pair_array_v0(x: i64) -> [i64] {\n    return [x, x * 2];\n}\n";
        verify_problem_code_strict(&problem, code)
            .unwrap_or_else(|err| panic!("array-output strict verify failed: {err}"));

        // STRICTNESS guard: a reference that returns the WRONG array must fail
        // verification (no loosened array equality).
        let wrong = "fn double_pair_array_v0(x: i64) -> [i64] {\n    return [x, x * 3];\n}\n";
        assert!(
            verify_problem_code_strict(&problem, wrong).is_err(),
            "a wrong array output must not verify"
        );
    }

    #[test]
    fn verify_problem_code_strict_typed_string_array_through_widened_wire() {
        // End-to-end on the WIDENED element type: a problem whose output is a
        // STRING array (`-> [str]`) verifies through the full strict path. The
        // wrapper prints the returned array with `println` (array-capable), the
        // runtime's `display_value` renders it as `[a, b]`, and that must equal
        // the expected stdout that `benchmark::render_expected` produces from the
        // `array_of([Str, Str])` wire value. Holdouts then recurse element-wise
        // through `output_matches` on the string elements. This is the path that
        // could NOT have worked before the array element type was `Vec<i64>`.
        let str_pair = |a: &str, b: &str| {
            BmValue::array_of(vec![BmValue::Str(a.to_string()), BmValue::Str(b.to_string())])
        };
        let problem = Problem {
            name: "tag_pair_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn tag_pair_v0(flag: i64) -> [str]",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(1)],
                    expected: str_pair("on", "yes"),
                },
                Example {
                    inputs: vec![BmValue::Int(0)],
                    expected: str_pair("off", "no"),
                },
            ],
            holdouts: vec![Example {
                inputs: vec![BmValue::Int(5)],
                expected: str_pair("on", "yes"),
            }],
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };
        let code = "fn tag_pair_v0(flag: i64) -> [str] {\n    if flag == 0 {\n        return [\"off\", \"no\"];\n    } else {\n        return [\"on\", \"yes\"];\n    }\n}\n";
        verify_problem_code_strict(&problem, code)
            .unwrap_or_else(|err| panic!("string-array strict verify failed: {err}"));

        // STRICTNESS: a reference returning a DIFFERENT string element must not
        // verify (no loosened string-array equality through the widened wire).
        let wrong = "fn tag_pair_v0(flag: i64) -> [str] {\n    if flag == 0 {\n        return [\"off\", \"no\"];\n    } else {\n        return [\"on\", \"YES\"];\n    }\n}\n";
        assert!(
            verify_problem_code_strict(&problem, wrong).is_err(),
            "a wrong string-array element must not verify"
        );
    }

    #[test]
    fn verify_problem_code_strict_nested_array_through_widened_wire() {
        // End-to-end on the NESTED element type: a problem whose output is an
        // array-of-arrays (`-> [[i64]]`) verifies through the full strict path.
        // Each inner array is itself a `Value::Array`, so `display_value` renders
        // `[[a, b], [c]]` and the expected stdout from `array_of([int_array, ..])`
        // must match exactly. Holdouts recurse twice through `output_matches`
        // (outer array, then inner arrays) — only possible once the array element
        // type became `Value` rather than `i64`.
        let problem = Problem {
            name: "split_first_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn split_first_v0(x: i64) -> [[i64]]",
            examples: vec![
                Example {
                    inputs: vec![BmValue::Int(3)],
                    expected: BmValue::array_of(vec![
                        BmValue::int_array(&[3]),
                        BmValue::int_array(&[6, 9]),
                    ]),
                },
                Example {
                    inputs: vec![BmValue::Int(0)],
                    expected: BmValue::array_of(vec![
                        BmValue::int_array(&[0]),
                        BmValue::int_array(&[0, 0]),
                    ]),
                },
            ],
            holdouts: vec![Example {
                inputs: vec![BmValue::Int(-2)],
                expected: BmValue::array_of(vec![
                    BmValue::int_array(&[-2]),
                    BmValue::int_array(&[-4, -6]),
                ]),
            }],
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };
        let code = "fn split_first_v0(x: i64) -> [[i64]] {\n    return [[x], [x * 2, x * 3]];\n}\n";
        verify_problem_code_strict(&problem, code)
            .unwrap_or_else(|err| panic!("nested-array strict verify failed: {err}"));

        // STRICTNESS: a wrong inner element must not verify.
        let wrong = "fn split_first_v0(x: i64) -> [[i64]] {\n    return [[x], [x * 2, x * 4]];\n}\n";
        assert!(
            verify_problem_code_strict(&problem, wrong).is_err(),
            "a wrong nested-array element must not verify"
        );
    }

    #[test]
    fn output_matches_pair_value_and_struct() {
        // Dedicated Pair value matches exactly.
        assert!(output_matches(&Value::Pair(3, 4), &BmValue::Pair(3, 4)));
        assert!(!output_matches(&Value::Pair(3, 4), &BmValue::Pair(4, 3)));
        // A 2-field struct (how a pair flows through struct-of-state) also matches.
        let pt = struct_value("Point", &[("x", Value::Int(3)), ("y", Value::Int(4))]);
        assert!(output_matches(&pt, &BmValue::Pair(3, 4)));
        assert!(output_matches(&pt, &BmValue::Pair(4, 3))); // multiset of payloads
        assert!(!output_matches(&pt, &BmValue::Pair(3, 5)));
        // Wrong arity is rejected.
        let one = struct_value("One", &[("x", Value::Int(3))]);
        assert!(!output_matches(&one, &BmValue::Pair(3, 4)));
    }

    #[test]
    fn output_matches_quad_value_and_struct() {
        assert!(output_matches(&Value::Quad(1, 2, 3, 4), &BmValue::Quad(1, 2, 3, 4)));
        assert!(!output_matches(&Value::Quad(1, 2, 3, 4), &BmValue::Quad(1, 2, 3, 5)));
        let s = struct_value(
            "DualTally",
            &[
                ("pos_count", Value::Int(1)),
                ("neg_count", Value::Int(2)),
                ("zero_count", Value::Int(3)),
                ("total", Value::Int(4)),
            ],
        );
        assert!(output_matches(&s, &BmValue::Quad(1, 2, 3, 4)));
        assert!(!output_matches(&s, &BmValue::Quad(1, 2, 3, 5)));
    }

    #[test]
    fn output_matches_tree_struct_strict() {
        let nodes = vec![tnode(1, 1, 2), tnode(2, -1, -1), tnode(3, -1, -1)];
        let tree = runtime_value_from_problem(&BmValue::Tree(nodes.clone()), "any")
            .expect("tree converts");
        // The converted runtime tree matches its own wire tree.
        assert!(output_matches(&tree, &BmValue::Tree(nodes.clone())));
        // A different node value fails.
        let wrong_value = vec![tnode(9, 1, 2), tnode(2, -1, -1), tnode(3, -1, -1)];
        assert!(!output_matches(&tree, &BmValue::Tree(wrong_value)));
        // A different child index fails (structure is checked, not just values).
        let wrong_edge = vec![tnode(1, 2, 1), tnode(2, -1, -1), tnode(3, -1, -1)];
        assert!(!output_matches(&tree, &BmValue::Tree(wrong_edge)));
        // A different node count fails.
        let shorter = vec![tnode(1, 1, 2), tnode(2, -1, -1)];
        assert!(!output_matches(&tree, &BmValue::Tree(shorter)));
        // An empty tree matches an empty tree.
        let empty = runtime_value_from_problem(&BmValue::Tree(vec![]), "any").unwrap();
        assert!(output_matches(&empty, &BmValue::Tree(vec![])));
        assert!(!output_matches(&empty, &BmValue::Tree(vec![tnode(0, -1, -1)])));
    }

    #[test]
    fn output_matches_rejects_type_mismatches() {
        // Cross-type comparisons that should never be true.
        assert!(!output_matches(&Value::Str("3".to_string()), &BmValue::Int(3)));
        assert!(!output_matches(&Value::Int(3), &BmValue::Str("3".to_string())));
        assert!(!output_matches(&Value::Array(vec![Value::Int(1)]), &BmValue::Int(1)));
        assert!(!output_matches(&Value::Pair(1, 2), &BmValue::Quad(1, 2, 0, 0)));
    }

    #[test]
    fn runtime_value_tree_converts_and_round_trips() {
        // A tree input converts without error into the canonical Struct shape
        // and round-trips through the equality oracle.
        let nodes = vec![tnode(10, 1, 2), tnode(5, -1, -1), tnode(15, -1, -1)];
        let value = runtime_value_from_problem(&BmValue::Tree(nodes.clone()), "tree_sum_v1")
            .expect("tree argument should convert");

        // Shape: Struct { name: "Tree", fields: { nodes: Array[TreeNode structs] } }.
        let Value::Struct { name, fields } = &value else {
            panic!("expected a Tree struct, got {value:?}");
        };
        assert_eq!(name, "Tree");
        let Some(Value::Array(items)) = fields.get("nodes") else {
            panic!("expected a `nodes` array field");
        };
        assert_eq!(items.len(), 3);
        // First node carries value=10, left=1, right=2 as runtime ints.
        // (`Value` derives no `PartialEq`, so assert via `matches!`.)
        let Value::Struct { name: nn, fields: nf } = &items[0] else {
            panic!("expected a TreeNode struct");
        };
        assert_eq!(nn, "TreeNode");
        assert!(matches!(nf.get("value"), Some(Value::Int(10))));
        assert!(matches!(nf.get("left"), Some(Value::Int(1))));
        assert!(matches!(nf.get("right"), Some(Value::Int(2))));

        // Round-trip: the converted value matches the wire tree it came from.
        assert!(output_matches(&value, &BmValue::Tree(nodes)));
    }

    #[test]
    fn runtime_value_pair_generalizes_beyond_named_structs() {
        // Known names keep their domain field names. (`Value` has no
        // `PartialEq`, so assert structurally via `matches!`.)
        let pt = runtime_value_from_problem(&BmValue::Pair(2, 3), "point_sum_v1").unwrap();
        let Value::Struct { name, fields } = &pt else {
            panic!("expected a struct, got {pt:?}");
        };
        assert_eq!(name, "Point");
        assert!(matches!(fields.get("x"), Some(Value::Int(2))));
        assert!(matches!(fields.get("y"), Some(Value::Int(3))));

        // An unknown problem name no longer errors — it yields a generic pair.
        let generic = runtime_value_from_problem(&BmValue::Pair(2, 3), "mystery_v1")
            .expect("generic pair should convert");
        let Value::Struct { name, fields } = &generic else {
            panic!("expected a struct, got {generic:?}");
        };
        assert_eq!(name, "Pair");
        assert!(matches!(fields.get("first"), Some(Value::Int(2))));
        assert!(matches!(fields.get("second"), Some(Value::Int(3))));
        // And it round-trips through the equality oracle as a pair.
        assert!(output_matches(&generic, &BmValue::Pair(2, 3)));
    }

    // ── Reference-driven generated holdouts (Part B) ───────────────────────

    /// A 1-arg `add_one` problem whose ONLY visible example (and ONLY
    /// hand-authored holdout) is `f(1) == 2` — the single point where the
    /// reference `a + 1` and a deliberately-overfit candidate `a * 2` agree.
    /// Used to prove a fresh reference-derived holdout catches the overfit.
    fn overfit_holdout_problem() -> Problem {
        Problem {
            name: "add_one_overfit_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn add_one_overfit_v0(a: i64) -> i64",
            examples: vec![Example {
                inputs: vec![BmValue::Int(1)],
                expected: BmValue::Int(2),
            }],
            // Hand-authored holdout is ALSO the collision point, so under the
            // OLD clone-of-hand-holdouts behavior the overfit candidate passed.
            holdouts: vec![Example {
                inputs: vec![BmValue::Int(1)],
                expected: BmValue::Int(2),
            }],
            reference_code: "fn add_one_overfit_v0(a: i64) -> i64 { return a + 1; }",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    /// Generated holdouts are deterministic: the same problem yields byte-
    /// identical holdout INPUTS (and outputs) across calls — seeded only from
    /// the problem name, with no clock/RNG.
    #[test]
    fn generated_holdouts_are_deterministic() {
        let problem = overfit_holdout_problem();
        let a = generated_holdouts(&problem);
        let b = generated_holdouts(&problem);
        assert!(!a.is_empty(), "reference-driven holdouts must be non-empty");
        assert_eq!(a, b, "holdouts must be reproducible across calls");
    }

    /// Generated holdouts come from the REFERENCE, not the candidate: an
    /// overfit-to-the-visible-example candidate passes the example check but
    /// now FAILS the strict verifier because a fresh reference-derived holdout
    /// (e.g. `f(5)`: reference 6, candidate 10) catches the divergence.
    #[test]
    fn generated_holdout_catches_overfit_candidate() {
        let problem = overfit_holdout_problem();

        // The candidate is genuinely overfit: it MATCHES the visible example...
        let overfit = "fn add_one_overfit_v0(a: i64) -> i64 { return a * 2; }";
        verify_problem_code(&problem, overfit)
            .expect("overfit candidate must satisfy the visible example (a=1 -> 2)");

        // ...but a reference-derived holdout catches it under strict verify.
        let strict = verify_problem_code_strict(&problem, overfit);
        assert!(
            strict.is_err(),
            "a generated holdout from the reference must reject the overfit \
             candidate, but strict verify accepted it: {strict:?}"
        );

        // Sanity: the reference itself passes strict verify (its own holdouts
        // are self-consistent — the new oracle is not spuriously strict).
        let reference = problem.reference_code;
        verify_problem_code_strict(&problem, reference)
            .expect("the reference implementation must pass its own generated holdouts");
    }

    /// Soundness fallback: an EMPTY reference_code degrades to the hand-authored
    /// holdouts (never fabricates, never trusts the candidate), preserving the
    /// "every problem yields non-empty holdouts" invariant.
    #[test]
    fn generated_holdouts_fall_back_without_reference() {
        let mut problem = overfit_holdout_problem();
        problem.reference_code = "";
        let holdouts = generated_holdouts(&problem);
        assert_eq!(
            holdouts, problem.holdouts,
            "with no reference, holdouts must equal the hand-authored set"
        );
    }

    /// The reference-derived holdouts exercise array outputs too: an identity
    /// `f([i64]) -> [i64]` reference round-trips through the runtime->wire
    /// converter, and a candidate that drops the first element is caught.
    #[test]
    fn generated_holdouts_handle_array_outputs() {
        let problem = Problem {
            name: "arr_identity_v0".to_string(),
            category: "test",
            description: "test",
            signature: "fn arr_identity_v0(a: [i64]) -> [i64]",
            examples: vec![Example {
                inputs: vec![BmValue::Array(vec![BmValue::Int(1), BmValue::Int(2)])],
                expected: BmValue::Array(vec![BmValue::Int(1), BmValue::Int(2)]),
            }],
            holdouts: vec![Example {
                inputs: vec![BmValue::Array(vec![BmValue::Int(1), BmValue::Int(2)])],
                expected: BmValue::Array(vec![BmValue::Int(1), BmValue::Int(2)]),
            }],
            reference_code: "fn arr_identity_v0(a: [i64]) -> [i64] { return a; }",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        };
        let holdouts = generated_holdouts(&problem);
        assert!(
            !holdouts.is_empty(),
            "array-output reference must yield generated holdouts"
        );
        // The identity reference passes its own generated holdouts.
        verify_problem_code_strict(&problem, problem.reference_code)
            .expect("array identity reference must pass its generated holdouts");
    }

    // ====================================================================
    // GAP 1 — BOUND LOOPS: ForTo / ForInRange / ForParallel must not hang.
    // ====================================================================

    /// A small int->int problem used to drive candidate code through the
    /// verify-path executor (`execute_function_for_problem`), which sets
    /// verification mode and isolates panics.
    fn ident_problem(name: &str) -> Problem {
        Problem {
            name: name.to_string(),
            category: "test",
            description: "test",
            // function_name() derives from the signature's fn name
            signature: Box::leak(format!("fn {name}(a: i64) -> i64").into_boxed_str()),
            examples: vec![Example {
                inputs: vec![BmValue::Int(0)],
                expected: BmValue::Int(0),
            }],
            holdouts: vec![],
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    #[test]
    fn forto_loop_is_capped() {
        // `for i in 0..n` with a huge n must return Err (cap), not spin forever.
        let problem = ident_problem("forto_cap_v0");
        let code = "fn forto_cap_v0(a: i64) -> i64 {\n    s := 0;\n    for i := 0 to 1000000000 {\n        s = s + 1;\n    }\n    return s;\n}\n";
        let result =
            execute_function_for_problem(code, "forto_cap_v0", &[BmValue::Int(0)], &problem);
        assert!(
            result.is_err(),
            "huge ForTo loop must hit the iteration cap, got {result:?}"
        );
        assert!(
            result.unwrap_err().contains("iteration limit"),
            "error should mention the iteration limit"
        );
    }

    #[test]
    fn forinrange_loop_is_capped() {
        // `for i in 0..n` (range form) with a huge n must return Err (cap).
        let problem = ident_problem("forinrange_cap_v0");
        let code = "fn forinrange_cap_v0(a: i64) -> i64 {\n    s := 0;\n    for i in 0..1000000000 {\n        s = s + 1;\n    }\n    return s;\n}\n";
        let result = execute_function_for_problem(
            code,
            "forinrange_cap_v0",
            &[BmValue::Int(0)],
            &problem,
        );
        assert!(
            result.is_err(),
            "huge ForInRange loop must hit the iteration cap, got {result:?}"
        );
        assert!(result.unwrap_err().contains("iteration limit"));
    }

    #[test]
    fn small_forto_loop_still_runs() {
        // A loop UNDER the cap must still execute normally (no regression).
        let problem = ident_problem("forto_small_v0");
        let code = "fn forto_small_v0(a: i64) -> i64 {\n    s := 0;\n    for i := 0 to 10 {\n        s = s + 1;\n    }\n    return s;\n}\n";
        let value =
            execute_function_for_problem(code, "forto_small_v0", &[BmValue::Int(0)], &problem)
                .expect("small loop must run");
        assert!(
            matches!(value, Value::Int(10)),
            "small loop should sum to 10, got {value:?}"
        );
    }

    // ====================================================================
    // GAP 2 — PANIC ISOLATION: a panic on the verify path becomes Err, not
    // a process SIGABRT.
    // ====================================================================

    #[test]
    fn run_isolated_converts_panic_to_err() {
        // The catch_unwind chokepoint must turn a panic into a clean Err so the
        // process survives. If this aborted the process the test binary would
        // crash instead of asserting.
        let result = run_isolated(|| panic!("boom in candidate"));
        assert!(
            result.is_err(),
            "a panicking candidate closure must yield Err, not abort"
        );
        assert_eq!(
            result.unwrap_err(),
            "candidate panicked during verification"
        );
    }

    #[test]
    fn run_isolated_passes_through_ok() {
        // A non-panicking closure returns its value unchanged.
        let result = run_isolated(|| Ok(Value::Int(42)));
        assert!(matches!(result, Ok(Value::Int(42))));
    }

    // ====================================================================
    // GAP 3 — CHECKED ARITHMETIC: pow / abs reject overflow instead of
    // panicking.
    // ====================================================================

    #[test]
    fn pow_overflow_is_err() {
        // 2^100 overflows i64; must return Err, not panic.
        let code = "fn main() -> i64 {\n    return pow(2, 100);\n}\n";
        let result = execute_program(code);
        assert!(result.is_err(), "pow overflow must be Err, got {result:?}");
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn pow_negative_exponent_is_err() {
        // A negative exponent previously wrapped via `as u32` into a huge value;
        // now it must be rejected.
        let code = "fn main() -> i64 {\n    return pow(2, -1);\n}\n";
        let result = execute_program(code);
        assert!(
            result.is_err(),
            "negative pow exponent must be Err, got {result:?}"
        );
        assert!(result.unwrap_err().contains("out of range"));
    }

    #[test]
    fn pow_small_still_works() {
        // pow within range must still compute correctly.
        let code = "fn main() -> i64 {\n    return pow(2, 10);\n}\n";
        let result = execute_program(code).expect("pow(2,10) must run");
        assert_eq!(result.return_value, Some(1024));
    }

    #[test]
    fn abs_i64_min_is_err() {
        // abs(i64::MIN) overflows (its negation is unrepresentable); must be Err,
        // not a panic. -9223372036854775808 is i64::MIN.
        let code =
            "fn main() -> i64 {\n    return abs(0 - 9223372036854775807 - 1);\n}\n";
        let result = execute_program(code);
        assert!(
            result.is_err(),
            "abs(i64::MIN) must be Err, got {result:?}"
        );
        assert!(result.unwrap_err().contains("overflow"));
    }

    #[test]
    fn abs_normal_still_works() {
        let code = "fn main() -> i64 {\n    return abs(0 - 7);\n}\n";
        let result = execute_program(code).expect("abs(-7) must run");
        assert_eq!(result.return_value, Some(7));
    }

    // ====================================================================
    // GAP 4 — DENY FS IN VERIFY: file builtins are denied on the verify path
    // but still work for normal execution.
    // ====================================================================

    #[test]
    fn verify_denies_open_file() {
        // A candidate that opens a file during verification must be rejected.
        let problem = ident_problem("fs_open_v0");
        let code = "fn fs_open_v0(a: i64) -> i64 {\n    f := open_file(\"/tmp/ncpu_verify_should_not_exist\", \"w\");\n    return 0;\n}\n";
        let result =
            execute_function_for_problem(code, "fs_open_v0", &[BmValue::Int(0)], &problem);
        assert!(
            result.is_err(),
            "open_file during verification must be denied, got {result:?}"
        );
        assert!(result.unwrap_err().contains("denied during verification"));
        // And the file must not have been created on the real FS.
        assert!(
            !std::path::Path::new("/tmp/ncpu_verify_should_not_exist").exists(),
            "verify-mode open_file must not touch the real filesystem"
        );
    }

    #[test]
    fn verify_denies_write_file() {
        let problem = ident_problem("fs_write_v0");
        let code = "fn fs_write_v0(a: i64) -> i64 {\n    f := open_file(\"/tmp/ncpu_verify_write\", \"w\");\n    write_file(f, \"x\");\n    return 0;\n}\n";
        let result =
            execute_function_for_problem(code, "fs_write_v0", &[BmValue::Int(0)], &problem);
        assert!(
            result.is_err(),
            "write_file during verification must be denied"
        );
    }

    #[test]
    fn normal_execution_allows_file_io() {
        // The SAME builtins must still work outside verification mode, proving
        // the deny is scoped to the verify path, not a global disable.
        let path = std::env::temp_dir().join("ncpu_normal_io_test.txt");
        let path_str = path.to_string_lossy().to_string();
        let _ = std::fs::remove_file(&path);
        let code = format!(
            "fn main() -> i64 {{\n    f := open_file(\"{path_str}\", \"w\");\n    write_file(f, \"hello\");\n    close_file(f);\n    return 0;\n}}\n"
        );
        let result = execute_program(&code);
        assert!(
            result.is_ok(),
            "normal (non-verify) file I/O must still work, got {result:?}"
        );
        let contents = std::fs::read_to_string(&path).expect("file should have been written");
        assert_eq!(contents, "hello");
        let _ = std::fs::remove_file(&path);
    }

    // ====================================================================
    // GAP 5 — FLOAT COMPARISON: epsilon compare, NaN policy, no false-accept.
    // ====================================================================

    #[test]
    fn output_matches_float_float_epsilon() {
        // Rounding-equal floats within epsilon must match (exact == would fail).
        assert!(output_matches(
            &Value::Float(1.0),
            &crate::benchmark::Value::Float((1.0_f64 + 1e-12).to_bits())
        ));
        // 0.1 + 0.2 != 0.3 exactly, but is within epsilon.
        assert!(output_matches(
            &Value::Float(0.1 + 0.2),
            &crate::benchmark::Value::Float(0.3_f64.to_bits())
        ));
        // Genuinely different floats must NOT match.
        assert!(!output_matches(
            &Value::Float(1.0),
            &crate::benchmark::Value::Float(2.0_f64.to_bits())
        ));
    }

    #[test]
    fn output_matches_nan_equals_nan() {
        // NaN policy: two NaNs are equal; NaN vs non-NaN is not.
        assert!(output_matches(
            &Value::Float(f64::NAN),
            &crate::benchmark::Value::Float(f64::NAN.to_bits())
        ));
        assert!(!output_matches(
            &Value::Float(f64::NAN),
            &crate::benchmark::Value::Float(1.0_f64.to_bits())
        ));
        assert!(!output_matches(
            &Value::Float(1.0),
            &crate::benchmark::Value::Float(f64::NAN.to_bits())
        ));
    }

    #[test]
    fn output_matches_float_int_rejects_2pow53_plus_1() {
        // The mandated false-accept guard: a float 2^53 must NOT match the int
        // 2^53+1, because the int does not round-trip exactly through f64.
        let two_pow_53 = 9007199254740992.0_f64; // == 2^53
        assert!(!output_matches(
            &Value::Float(two_pow_53),
            &crate::benchmark::Value::Int(9007199254740993) // 2^53 + 1
        ));
        // Symmetric direction: Int actual vs Float expected.
        assert!(!output_matches(
            &Value::Int(9007199254740993),
            &crate::benchmark::Value::Float(two_pow_53.to_bits())
        ));
        // But an integral float that DOES round-trip exactly matches its int.
        assert!(output_matches(
            &Value::Float(42.0),
            &crate::benchmark::Value::Int(42)
        ));
        assert!(output_matches(
            &Value::Int(42),
            &crate::benchmark::Value::Float(42.0_f64.to_bits())
        ));
    }

    #[test]
    fn output_matches_float_rejects_nonintegral_vs_int() {
        // A non-integral float must never match an int expected.
        assert!(!output_matches(
            &Value::Float(1.5),
            &crate::benchmark::Value::Int(1)
        ));
        assert!(!output_matches(
            &Value::Float(1.5),
            &crate::benchmark::Value::Int(2)
        ));
    }
}
