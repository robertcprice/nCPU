/*! Mog tree-walk interpreter (evaluator). */

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use crate::ast::*;

// --- Runtime values ---

#[derive(Clone)]
pub enum MogValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MogValue>),
    Map(HashMap<String, MogValue>),
    Struct {
        name: String,
        fields: HashMap<String, MogValue>,
    },
    Result {
        is_ok: bool,
        value: Box<MogValue>,
    },
    Optional {
        is_some: bool,
        value: Box<MogValue>,
    },
    None_,
    Builtin(String),
    // Stored as: params, body, captured env
    Closure(Vec<Param>, ClosureBody, Env),
    // A declared function (name, params, body)
    FnDecl(String, Vec<Param>, Block),
}

impl fmt::Debug for MogValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MogValue::Int(i) => write!(f, "{}", i),
            MogValue::Float(fl) => write!(f, "{:.7}", fl),
            MogValue::Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            MogValue::Str(s) => write!(f, "{}", s),
            MogValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            MogValue::None_ => write!(f, "none"),
            _ => write!(f, "<value>"),
        }
    }
}

impl MogValue {
    pub fn is_truthy(&self) -> bool {
        match self {
            MogValue::Bool(b) => *b,
            MogValue::Int(i) => *i != 0,
            MogValue::Float(f) => *f != 0.0,
            MogValue::Str(s) => !s.is_empty(),
            MogValue::None_ => false,
            MogValue::Optional { is_some, .. } => *is_some,
            _ => true,
        }
    }

    pub fn to_i64(&self) -> i64 {
        match self {
            MogValue::Int(i) => *i,
            MogValue::Float(f) => *f as i64,
            MogValue::Bool(b) => if *b { 1 } else { 0 },
            _ => 0,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            MogValue::Float(f) => *f,
            MogValue::Int(i) => *i as f64,
            _ => 0.0,
        }
    }
}

// --- Control flow signals ---

#[derive(Debug)]
pub enum ControlFlow {
    Return(MogValue),
    Break,
    Continue,
    Propagate(MogValue),
}

#[derive(Debug)]
pub struct MogError(pub String);

impl fmt::Display for MogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.0) }
}

impl From<ControlFlow> for MogError {
    fn from(cf: ControlFlow) -> Self {
        match cf {
            ControlFlow::Return(v) => MogError(format!("unexpected return: {:?}", v)),
            ControlFlow::Break => MogError("unexpected break".into()),
            ControlFlow::Continue => MogError("unexpected continue".into()),
            ControlFlow::Propagate(v) => MogError(format!("unhandled propagate: {:?}", v)),
        }
    }
}

// --- Environment ---

#[derive(Clone)]
pub struct Environment {
    bindings: HashMap<String, MogValue>,
    parent: Option<Env>,
}

pub type Env = Rc<RefCell<Environment>>;

impl Environment {
    pub fn new() -> Env {
        Rc::new(RefCell::new(Environment {
            bindings: HashMap::new(),
            parent: None,
        }))
    }

    pub fn child(env: &Env) -> Env {
        Rc::new(RefCell::new(Environment {
            bindings: HashMap::new(),
            parent: Some(env.clone()),
        }))
    }

    pub fn get(env: &Env, name: &str) -> Result<MogValue, MogError> {
        let e = env.borrow();
        if let Some(v) = e.bindings.get(name) {
            return Ok(v.clone());
        }
        if let Some(ref parent) = e.parent {
            return Environment::get(parent, name);
        }
        Err(MogError(format!("undefined variable '{}'", name)))
    }

    pub fn set(env: &Env, name: &str, value: MogValue) -> Result<(), MogError> {
        let mut e = env.borrow_mut();
        if e.bindings.contains_key(name) {
            e.bindings.insert(name.to_string(), value);
            return Ok(());
        }
        drop(e);
        if let Some(ref parent) = env.borrow().parent.clone() {
            return Environment::set(parent, name, value);
        }
        Err(MogError(format!("undefined variable '{}' for assignment", name)))
    }

    pub fn define(env: &Env, name: &str, value: MogValue) {
        env.borrow_mut().bindings.insert(name.to_string(), value);
    }
}

// --- Evaluator ---

const MAX_RECURSION: u32 = 500;
const MAX_LOOP_ITERS: u64 = 100_000;

pub struct Evaluator {
    pub output: Vec<String>,
    call_depth: u32,
    global_env: Env,
    input_queue: Vec<String>,
}

impl Evaluator {
    pub fn new() -> Self {
        let global_env = Environment::new();
        let mut ev = Evaluator {
            output: Vec::new(),
            call_depth: 0,
            global_env: global_env.clone(),
            input_queue: Vec::new(),
        };
        ev.register_builtins(&global_env);
        ev
    }

    pub fn with_input(input: Vec<String>) -> Self {
        let mut ev = Self::new();
        ev.input_queue = input;
        ev
    }

    fn register_builtins(&mut self, env: &Env) {
        // Constants
        Environment::define(env, "PI", MogValue::Float(std::f64::consts::PI));
        Environment::define(env, "E", MogValue::Float(std::f64::consts::E));
        // Builtin names (handled specially in eval_call)
        for name in &[
            "println", "println_i64", "print_f64", "print", "print_string",
            "str", "len", "abs", "sqrt", "pow", "sin", "cos", "tan",
            "asin", "acos", "atan2", "exp", "log", "log2",
            "floor", "ceil", "round", "min", "max",
            "read_i64", "read_string", "read_line", "has_input",
            "int_from_string", "parse_float",
        ] {
            Environment::define(env, name, MogValue::Builtin(name.to_string()));
        }
    }

    pub fn run(&mut self, program: &Program) -> Result<MogValue, MogError> {
        for decl in &program.declarations {
            match decl {
                Decl::FnDecl(fd) => {
                    Environment::define(&self.global_env, &fd.name,
                        MogValue::FnDecl(fd.name.clone(), fd.params.clone(), fd.body.clone()));
                }
                Decl::StructDecl(sd) => {
                    Environment::define(&self.global_env, &sd.name,
                        MogValue::Struct { name: sd.name.clone(), fields: HashMap::new() });
                }
                _ => {}
            }
        }
        // Call main()
        let main_fn = Environment::get(&self.global_env, "main")?;
        match main_fn {
            MogValue::FnDecl(_, params, body) => {
                self.call_depth = 0;
                match self.call_fn(&params, &body, &[]) {
                    Ok(v) => Ok(v),
                    Err(ControlFlow::Return(v)) => Ok(v),
                    Err(cf) => Err(MogError::from(cf)),
                }
            }
            _ => Err(MogError("no main() function defined".into())),
        }
    }

    fn call_fn(&mut self, params: &[Param], body: &Block, args: &[MogValue]) -> Result<MogValue, ControlFlow> {
        self.call_depth += 1;
        if self.call_depth > MAX_RECURSION {
            self.call_depth -= 1;
            return Err(ControlFlow::Return(MogValue::Int(0)));
        }
        let local = Environment::child(&self.global_env);
        for (i, param) in params.iter().enumerate() {
            let val = if i < args.len() {
                args[i].clone()
            } else if let Some(ref default) = param.default {
                self.eval_expr(default, &local)?
            } else {
                MogValue::Int(0)
            };
            Environment::define(&local, &param.name, val);
        }
        let result = self.exec_block(body, &local);
        self.call_depth -= 1;
        match result {
            Err(ControlFlow::Return(v)) => Ok(v),
            Err(cf) => Err(cf),
            Ok(()) => Ok(MogValue::Int(0)),
        }
    }

    fn call_closure(&mut self, closure_params: &[Param], body: &ClosureBody, closure_env: &Env, args: &[MogValue]) -> Result<MogValue, ControlFlow> {
        self.call_depth += 1;
        if self.call_depth > MAX_RECURSION {
            self.call_depth -= 1;
            return Err(ControlFlow::Return(MogValue::Int(0)));
        }
        let local = Environment::child(closure_env);
        for (i, param) in closure_params.iter().enumerate() {
            let val = if i < args.len() { args[i].clone() } else { MogValue::Int(0) };
            Environment::define(&local, &param.name, val);
        }
        let result = match body {
            ClosureBody::Block(block) => {
                match self.exec_block(block, &local) {
                    Err(ControlFlow::Return(v)) => Ok(v),
                    Err(cf) => Err(cf),
                    Ok(()) => Ok(MogValue::Int(0)),
                }
            }
            ClosureBody::Expr(expr) => self.eval_expr(expr, &local),
        };
        self.call_depth -= 1;
        result
    }

    // --- Statement execution ---

    fn exec_block(&mut self, block: &Block, env: &Env) -> Result<(), ControlFlow> {
        for stmt in &block.stmts {
            self.exec_stmt(stmt, env)?;
        }
        Ok(())
    }

    fn exec_stmt(&mut self, stmt: &Stmt, env: &Env) -> Result<(), ControlFlow> {
        match stmt {
            Stmt::VarDecl(vd) => {
                let val = self.eval_expr(&vd.value, env)?;
                Environment::define(env, &vd.name, val);
                Ok(())
            }
            Stmt::Assignment(asgn) => {
                let val = self.eval_expr(&asgn.value, env)?;
                self.assign(&asgn.target, val, env)?;
                Ok(())
            }
            Stmt::ReturnStmt(ret) => {
                let val = match &ret.value {
                    Some(v) => self.eval_expr(v, env)?,
                    None => MogValue::Int(0),
                };
                Err(ControlFlow::Return(val))
            }
            Stmt::IfStmt(if_stmt) => {
                let cond = self.eval_expr(&if_stmt.condition, env)?;
                if cond.is_truthy() {
                    let child = Environment::child(env);
                    self.exec_block(&if_stmt.then_block, &child)?;
                } else if let Some(ref else_b) = if_stmt.else_block {
                    match else_b {
                        ElseBranch::If(inner_if) => self.exec_stmt(&Stmt::IfStmt((**inner_if).clone()), env)?,
                        ElseBranch::Block(block) => {
                            let child = Environment::child(env);
                            self.exec_block(block, &child)?;
                        }
                    }
                }
                Ok(())
            }
            Stmt::WhileStmt(wh) => {
                let mut iters: u64 = 0;
                loop {
                    let cond = self.eval_expr(&wh.condition, env)?;
                    if !cond.is_truthy() { break; }
                    iters += 1;
                    if iters > MAX_LOOP_ITERS {
                        return Err(ControlFlow::Return(MogValue::Int(0)));
                    }
                    let child = Environment::child(env);
                    match self.exec_block(&wh.body, &child) {
                        Err(ControlFlow::Break) => break,
                        Err(ControlFlow::Continue) => continue,
                        Err(cf) => return Err(cf),
                        Ok(()) => {}
                    }
                }
                Ok(())
            }
            Stmt::ForToStmt(ft) => {
                let s = self.eval_expr(&ft.start, env)?.to_i64();
                let e = self.eval_expr(&ft.end, env)?.to_i64();
                let local = Environment::child(env);
                for i in s..e {
                    Environment::define(&local, &ft.var_name, MogValue::Int(i));
                    match self.exec_block(&ft.body, &local) {
                        Err(ControlFlow::Break) => break,
                        Err(ControlFlow::Continue) => continue,
                        Err(cf) => return Err(cf),
                        Ok(()) => {}
                    }
                }
                Ok(())
            }
            Stmt::ForInRangeStmt(fr) => {
                let s = self.eval_expr(&fr.start, env)?.to_i64();
                let e = self.eval_expr(&fr.end, env)?.to_i64();
                let local = Environment::child(env);
                for i in s..e {
                    Environment::define(&local, &fr.var_name, MogValue::Int(i));
                    match self.exec_block(&fr.body, &local) {
                        Err(ControlFlow::Break) => break,
                        Err(ControlFlow::Continue) => continue,
                        Err(cf) => return Err(cf),
                        Ok(()) => {}
                    }
                }
                Ok(())
            }
            Stmt::ForInStmt(fi) => {
                let iter_val = self.eval_expr(&fi.iterable, env)?;
                let local = Environment::child(env);
                match &iter_val {
                    MogValue::Array(arr) => {
                        for (i, item) in arr.iter().enumerate() {
                            if let Some(ref idx_name) = fi.index_name {
                                Environment::define(&local, idx_name, MogValue::Int(i as i64));
                            }
                            Environment::define(&local, &fi.var_name, item.clone());
                            match self.exec_block(&fi.body, &local) {
                                Err(ControlFlow::Break) => break,
                                Err(ControlFlow::Continue) => continue,
                                Err(cf) => return Err(cf),
                                Ok(()) => {}
                            }
                        }
                    }
                    MogValue::Map(map) => {
                        for (k, v) in map.iter() {
                            if let Some(ref idx_name) = fi.index_name {
                                Environment::define(&local, idx_name, MogValue::Str(k.clone()));
                                Environment::define(&local, &fi.var_name, v.clone());
                            } else {
                                Environment::define(&local, &fi.var_name, MogValue::Str(k.clone()));
                            }
                            match self.exec_block(&fi.body, &local) {
                                Err(ControlFlow::Break) => break,
                                Err(ControlFlow::Continue) => continue,
                                Err(cf) => return Err(cf),
                                Ok(()) => {}
                            }
                        }
                    }
                    _ => {}
                }
                Ok(())
            }
            Stmt::BreakStmt => Err(ControlFlow::Break),
            Stmt::ContinueStmt => Err(ControlFlow::Continue),
            Stmt::ExprStmt(es) => { self.eval_expr(&es.expr, env)?; Ok(()) }
            Stmt::FnDecl(fd) => {
                Environment::define(env, &fd.name,
                    MogValue::FnDecl(fd.name.clone(), fd.params.clone(), fd.body.clone()));
                Ok(())
            }
            Stmt::StructDecl(sd) => {
                Environment::define(env, &sd.name,
                    MogValue::Struct { name: sd.name.clone(), fields: HashMap::new() });
                Ok(())
            }
            Stmt::Block(block) => {
                let child = Environment::child(env);
                self.exec_block(block, &child)
            }
        }
    }

    fn assign(&mut self, target: &Expr, value: MogValue, env: &Env) -> Result<(), ControlFlow> {
        match target {
            Expr::Ident(name) => {
                Environment::set(env, name, value).map_err(|_| ControlFlow::Return(MogValue::Int(0)))
            }
            Expr::IndexAccess(idx_data) => {
                let obj_val = self.eval_expr(&idx_data.obj, env)?;
                let idx = self.eval_expr(&idx_data.index, env)?;
                match obj_val {
                    MogValue::Array(mut arr) => {
                        let i = idx.to_i64() as usize;
                        if i < arr.len() { arr[i] = value; }
                        if let Expr::Ident(name) = idx_data.obj.as_ref() {
                            Environment::set(env, name, MogValue::Array(arr)).ok();
                        }
                        Ok(())
                    }
                    MogValue::Map(mut map) => {
                        let key = match &idx { MogValue::Str(s) => s.clone(), _ => format!("{:?}", idx) };
                        map.insert(key, value);
                        if let Expr::Ident(name) = idx_data.obj.as_ref() {
                            Environment::set(env, name, MogValue::Map(map)).ok();
                        }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            Expr::FieldAccess { .. } => {
                // For struct mutation — simplified
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // --- Expression evaluation ---

    fn eval_expr(&mut self, expr: &Expr, env: &Env) -> Result<MogValue, ControlFlow> {
        match expr {
            Expr::IntLit(value) => Ok(MogValue::Int(*value)),
            Expr::FloatLit(value) => Ok(MogValue::Float(*value)),
            Expr::StringLit(value) => Ok(MogValue::Str(value.clone())),
            Expr::FStringLit(value) => Ok(MogValue::Str(value.clone())), // simplified: no interpolation
            Expr::BoolLit(value) => Ok(MogValue::Bool(*value)),
            Expr::NoneLit => Ok(MogValue::None_),
            Expr::Ident(name) => Environment::get(env, name).map_err(|_| ControlFlow::Return(MogValue::Int(0))),
            Expr::BinOp(binop) => self.eval_binop(&binop.op, &binop.left, &binop.right, env),
            Expr::UnaryOp(unop) => self.eval_unary(&unop.op, &unop.operand, env),
            Expr::Call(call_data) => self.eval_call(&call_data.func, &call_data.args, env),
            Expr::FieldAccess(fa) => self.eval_field_access(&fa.obj, &fa.field, env),
            Expr::IndexAccess(ia) => self.eval_index_access(&ia.obj, &ia.index, env),
            Expr::ArrayLit(elements) => {
                let mut vals = Vec::new();
                for e in elements {
                    vals.push(self.eval_expr(e, env)?);
                }
                Ok(MogValue::Array(vals))
            }
            Expr::ArrayFill { value, count } => {
                let val = self.eval_expr(value, env)?;
                let cnt = self.eval_expr(count, env)?.to_i64() as usize;
                Ok(MogValue::Array(vec![val; cnt]))
            }
            Expr::MapLit(pairs) => {
                let mut map = HashMap::new();
                for (k, v) in pairs {
                    let key = match self.eval_expr(k, env)? {
                        MogValue::Str(s) => s,
                        other => format!("{:?}", other),
                    };
                    map.insert(key, self.eval_expr(v, env)?);
                }
                Ok(MogValue::Map(map))
            }
            Expr::StructConstruct { name, fields } => {
                let mut field_vals = HashMap::new();
                for (fname, fexpr) in fields {
                    field_vals.insert(fname.clone(), self.eval_expr(fexpr, env)?);
                }
                Ok(MogValue::Struct { name: name.clone(), fields: field_vals })
            }
            Expr::OkExpr(value) => {
                Ok(MogValue::Result { is_ok: true, value: Box::new(self.eval_expr(value, env)?) })
            }
            Expr::ErrExpr(value) => {
                Ok(MogValue::Result { is_ok: false, value: Box::new(self.eval_expr(value, env)?) })
            }
            Expr::SomeExpr(value) => {
                Ok(MogValue::Optional { is_some: true, value: Box::new(self.eval_expr(value, env)?) })
            }
            Expr::MatchExpr(match_data) => self.eval_match(&match_data.subject, &match_data.arms, env),
            Expr::ClosureLit(cl) => {
                Ok(MogValue::Closure(cl.params.clone(), cl.body.clone(), env.clone()))
            }
            Expr::IfExpr(if_data) => {
                let cond = self.eval_expr(&if_data.condition, env)?;
                if cond.is_truthy() {
                    self.eval_expr(&if_data.then_expr, env)
                } else {
                    self.eval_expr(&if_data.else_expr, env)
                }
            }
            Expr::CastExpr(cast) => {
                let val = self.eval_expr(&cast.expr, env)?;
                match cast.target_type.as_str() {
                    "i64" | "int" => Ok(MogValue::Int(val.to_i64())),
                    "f64" | "float" => Ok(MogValue::Float(val.to_f64())),
                    _ => Ok(val),
                }
            }
            Expr::PropagateExpr(inner) => {
                let val = self.eval_expr(inner, env)?;
                match &val {
                    MogValue::Result { is_ok: false, .. } => Err(ControlFlow::Propagate(val)),
                    MogValue::Result { is_ok: true, value } => Ok((**value).clone()),
                    _ => Ok(val),
                }
            }
            Expr::RangeExpr { start, end } => {
                // Evaluate but return as array (used in for-in-range, but also as expression)
                let s = self.eval_expr(start, env)?.to_i64();
                let e = self.eval_expr(end, env)?.to_i64();
                Ok(MogValue::Array((s..e).map(MogValue::Int).collect()))
            }
        }
    }

    fn eval_binop(&mut self, op: &str, left: &Expr, right: &Expr, env: &Env) -> Result<MogValue, ControlFlow> {
        // Short-circuit logical ops
        if op == "&&" || op == "and" {
            let l = self.eval_expr(left, env)?;
            if !l.is_truthy() { return Ok(l); }
            return self.eval_expr(right, env);
        }
        if op == "||" || op == "or" {
            let l = self.eval_expr(left, env)?;
            if l.is_truthy() { return Ok(l); }
            return self.eval_expr(right, env);
        }

        let l = self.eval_expr(left, env)?;
        let r = self.eval_expr(right, env)?;

        // String concatenation
        if op == "+" {
            if let MogValue::Str(s) = &l {
                let rs = match &r { MogValue::Str(s) => s.clone(), _ => format!("{:?}", r) };
                return Ok(MogValue::Str(format!("{}{}", s, rs)));
            }
        }

        match op {
            "+" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => Ok(MogValue::Int(a.saturating_add(*b))),
                (MogValue::Float(a), MogValue::Float(b)) => Ok(MogValue::Float(a + b)),
                (MogValue::Int(a), MogValue::Float(b)) => Ok(MogValue::Float(*a as f64 + b)),
                (MogValue::Float(a), MogValue::Int(b)) => Ok(MogValue::Float(a + *b as f64)),
                _ => Ok(MogValue::Int(0)),
            },
            "-" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => Ok(MogValue::Int(a.saturating_sub(*b))),
                (MogValue::Float(a), MogValue::Float(b)) => Ok(MogValue::Float(a - b)),
                _ => Ok(MogValue::Int(0)),
            },
            "*" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => Ok(MogValue::Int(a.saturating_mul(*b))),
                (MogValue::Float(a), MogValue::Float(b)) => Ok(MogValue::Float(a * b)),
                _ => Ok(MogValue::Int(0)),
            },
            "/" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => {
                    if *b == 0 { Ok(MogValue::Int(0)) } else { Ok(MogValue::Int(a / b)) }
                }
                (MogValue::Float(a), MogValue::Float(b)) => {
                    if *b == 0.0 { Ok(MogValue::Float(0.0)) } else { Ok(MogValue::Float(a / b)) }
                }
                _ => Ok(MogValue::Int(0)),
            },
            "%" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => {
                    if *b == 0 { Ok(MogValue::Int(0)) } else { Ok(MogValue::Int(a % b)) }
                }
                _ => Ok(MogValue::Int(0)),
            },
            "**" => match (&l, &r) {
                (MogValue::Int(a), MogValue::Int(b)) => Ok(MogValue::Int(a.pow(*b as u32))),
                (MogValue::Float(a), MogValue::Float(b)) => Ok(MogValue::Float(a.powf(*b))),
                _ => Ok(MogValue::Int(0)),
            },
            "==" => Ok(MogValue::Bool(self.val_eq(&l, &r))),
            "!=" => Ok(MogValue::Bool(!self.val_eq(&l, &r))),
            "<" => Ok(MogValue::Bool(self.val_cmp(&l, &r) < 0)),
            ">" => Ok(MogValue::Bool(self.val_cmp(&l, &r) > 0)),
            "<=" => Ok(MogValue::Bool(self.val_cmp(&l, &r) <= 0)),
            ">=" => Ok(MogValue::Bool(self.val_cmp(&l, &r) >= 0)),
            "&" => Ok(MogValue::Int(l.to_i64() & r.to_i64())),
            "|" => Ok(MogValue::Int(l.to_i64() | r.to_i64())),
            "^" => Ok(MogValue::Int(l.to_i64() ^ r.to_i64())),
            "<<" => Ok(MogValue::Int(l.to_i64() << r.to_i64() as u32)),
            ">>" => Ok(MogValue::Int(l.to_i64() >> r.to_i64() as u32)),
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn val_eq(&self, a: &MogValue, b: &MogValue) -> bool {
        match (a, b) {
            (MogValue::Int(x), MogValue::Int(y)) => x == y,
            (MogValue::Float(x), MogValue::Float(y)) => x == y,
            (MogValue::Int(x), MogValue::Float(y)) => (*x as f64) == *y,
            (MogValue::Float(x), MogValue::Int(y)) => *x == (*y as f64),
            (MogValue::Bool(x), MogValue::Bool(y)) => x == y,
            (MogValue::Str(x), MogValue::Str(y)) => x == y,
            (MogValue::Bool(x), MogValue::Int(y)) => (*x as i64) == *y,
            (MogValue::Int(x), MogValue::Bool(y)) => *x == (*y as i64),
            _ => false,
        }
    }

    fn val_cmp(&self, a: &MogValue, b: &MogValue) -> i32 {
        let af = match a { MogValue::Int(i) => *i as f64, MogValue::Float(f) => *f, _ => 0.0 };
        let bf = match b { MogValue::Int(i) => *i as f64, MogValue::Float(f) => *f, _ => 0.0 };
        af.partial_cmp(&bf).map(|o| o as i32).unwrap_or(0)
    }

    fn eval_unary(&mut self, op: &str, operand: &Expr, env: &Env) -> Result<MogValue, ControlFlow> {
        let val = self.eval_expr(operand, env)?;
        match op {
            "-" => match val {
                MogValue::Int(i) => Ok(MogValue::Int(-i)),
                MogValue::Float(f) => Ok(MogValue::Float(-f)),
                _ => Ok(MogValue::Int(0)),
            },
            "!" => Ok(MogValue::Bool(!val.is_truthy())),
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn eval_call(&mut self, func: &Expr, args: &[Expr], env: &Env) -> Result<MogValue, ControlFlow> {
        // Method calls: obj.method(args)
        if let Expr::FieldAccess(fa) = func {
            let obj_val = self.eval_expr(&fa.obj, env)?;
            let mut arg_vals = Vec::new();
            for a in args { arg_vals.push(self.eval_expr(a, env)?); }
            return self.call_method(obj_val, &fa.field, &arg_vals, env);
        }

        // Evaluate function and arguments
        let func_val = self.eval_expr(func, env)?;
        let mut arg_vals = Vec::new();
        for a in args { arg_vals.push(self.eval_expr(a, env)?); }

        match func_val {
            MogValue::Builtin(name) => self.call_builtin(&name, &arg_vals),
            MogValue::FnDecl(_, params, body) => self.call_fn(&params, &body, &arg_vals),
            MogValue::Closure(params, body, closure_env) => {
                self.call_closure(&params, &body, &closure_env, &arg_vals)
            }
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn call_builtin(&mut self, name: &str, args: &[MogValue]) -> Result<MogValue, ControlFlow> {
        match name {
            "println_i64" => {
                let v = args.first().map(|v| v.to_i64()).unwrap_or(0);
                self.output.push(format!("{}", v));
                Ok(MogValue::Int(0))
            }
            "println" => {
                let text = args.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.output.push(text);
                Ok(MogValue::Int(0))
            }
            "print_f64" => {
                let v = args.first().map(|v| v.to_f64()).unwrap_or(0.0);
                self.output.push(format!("{:.7}", v));
                Ok(MogValue::Int(0))
            }
            "print" => {
                let v = args.first().map(|v| v.to_i64()).unwrap_or(0);
                self.output.push(format!("{}", v));
                Ok(MogValue::Int(0))
            }
            "print_string" => {
                let s = args.first().map(|v| format!("{:?}", v)).unwrap_or_default();
                if let Some(last) = self.output.last_mut() {
                    last.push_str(&s);
                } else {
                    self.output.push(s);
                }
                Ok(MogValue::Int(0))
            }
            "str" => {
                let s = match args.first() {
                    Some(MogValue::Int(i)) => format!("{}", i),
                    Some(MogValue::Float(f)) => format!("{}", f),
                    Some(MogValue::Str(s)) => s.clone(),
                    Some(v) => format!("{:?}", v),
                    None => String::new(),
                };
                Ok(MogValue::Str(s))
            }
            "len" => {
                let len = match args.first() {
                    Some(MogValue::Str(s)) => s.len() as i64,
                    Some(MogValue::Array(arr)) => arr.len() as i64,
                    Some(MogValue::Map(m)) => m.len() as i64,
                    _ => 0,
                };
                Ok(MogValue::Int(len))
            }
            "abs" => Ok(MogValue::Int(args.first().map(|v| v.to_i64().abs()).unwrap_or(0))),
            "sqrt" => Ok(MogValue::Float(args.first().map(|v| v.to_f64().sqrt()).unwrap_or(0.0))),
            "pow" => {
                let a = args.first().map(|v| v.to_f64()).unwrap_or(0.0);
                let b = args.get(1).map(|v| v.to_f64()).unwrap_or(0.0);
                Ok(MogValue::Float(a.powf(b)))
            }
            "min" => {
                let a = args.first().map(|v| v.to_i64()).unwrap_or(0);
                let b = args.get(1).map(|v| v.to_i64()).unwrap_or(0);
                Ok(MogValue::Int(a.min(b)))
            }
            "max" => {
                let a = args.first().map(|v| v.to_i64()).unwrap_or(0);
                let b = args.get(1).map(|v| v.to_i64()).unwrap_or(0);
                Ok(MogValue::Int(a.max(b)))
            }
            "floor" => Ok(MogValue::Int(args.first().map(|v| v.to_f64().floor() as i64).unwrap_or(0))),
            "ceil" => Ok(MogValue::Int(args.first().map(|v| v.to_f64().ceil() as i64).unwrap_or(0))),
            "round" => Ok(MogValue::Int(args.first().map(|v| v.to_f64().round() as i64).unwrap_or(0))),
            "has_input" => Ok(MogValue::Int(if self.input_queue.is_empty() { 0 } else { 1 })),
            "read_i64" => {
                match self.input_queue.first() {
                    Some(s) => {
                        let val = s.trim().parse::<i64>().unwrap_or(0);
                        self.input_queue.remove(0);
                        Ok(MogValue::Int(val))
                    }
                    None => Ok(MogValue::Int(0)),
                }
            }
            "read_string" | "read_line" => {
                match self.input_queue.first() {
                    Some(s) => {
                        let val = s.clone();
                        self.input_queue.remove(0);
                        Ok(MogValue::Str(val))
                    }
                    None => Ok(MogValue::Str(String::new())),
                }
            }
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn call_method(&mut self, obj: MogValue, method: &str, args: &[MogValue], _env: &Env) -> Result<MogValue, ControlFlow> {
        match &obj {
            MogValue::Array(arr) => {
                match method {
                    "push" => {
                        let mut arr = arr.clone();
                        arr.push(args.first().cloned().unwrap_or(MogValue::Int(0)));
                        Ok(MogValue::Array(arr))
                    }
                    "pop" => {
                        let mut arr = arr.clone();
                        let val = arr.pop().unwrap_or(MogValue::Int(0));
                        Ok(val)
                    }
                    "sort" => {
                        let mut arr = arr.clone();
                        arr.sort_by(|a, b| a.to_i64().cmp(&b.to_i64()));
                        Ok(MogValue::Array(arr))
                    }
                    "map" => {
                        if let Some(MogValue::Closure(params, body, closure_env)) = args.first() {
                            let mut result = Vec::new();
                            for item in arr {
                                result.push(self.call_closure(params, body, closure_env, &[item.clone()])?);
                            }
                            return Ok(MogValue::Array(result));
                        }
                        Ok(MogValue::Array(arr.clone()))
                    }
                    "join" => {
                        let sep = match args.first() {
                            Some(MogValue::Str(s)) => s.clone(),
                            _ => String::new(),
                        };
                        let joined: Vec<String> = arr.iter().map(|v| format!("{:?}", v)).collect();
                        Ok(MogValue::Str(joined.join(&sep)))
                    }
                    _ => Ok(MogValue::Int(0)),
                }
            }
            MogValue::Str(s) => {
                match method {
                    "upper" => Ok(MogValue::Str(s.to_uppercase())),
                    "lower" => Ok(MogValue::Str(s.to_lowercase())),
                    "trim" => Ok(MogValue::Str(s.trim().to_string())),
                    "split" => {
                        let sep = match args.first() {
                            Some(MogValue::Str(s)) => s.clone(),
                            _ => String::new(),
                        };
                        let parts: Vec<MogValue> = if sep.is_empty() {
                            s.chars().map(|c| MogValue::Str(c.to_string())).collect()
                        } else {
                            s.split(&sep).map(|p| MogValue::Str(p.to_string())).collect()
                        };
                        Ok(MogValue::Array(parts))
                    }
                    "contains" => {
                        let sub = match args.first() { Some(MogValue::Str(s)) => s.clone(), _ => String::new() };
                        Ok(MogValue::Bool(s.contains(&sub)))
                    }
                    "starts_with" => {
                        let prefix = match args.first() { Some(MogValue::Str(s)) => s.clone(), _ => String::new() };
                        Ok(MogValue::Bool(s.starts_with(&prefix)))
                    }
                    "ends_with" => {
                        let suffix = match args.first() { Some(MogValue::Str(s)) => s.clone(), _ => String::new() };
                        Ok(MogValue::Bool(s.ends_with(&suffix)))
                    }
                    "replace" => {
                        let old = match args.first() { Some(MogValue::Str(s)) => s.clone(), _ => String::new() };
                        let new = args.get(1).map(|v| match v { MogValue::Str(s) => s.clone(), _ => String::new() }).unwrap_or_default();
                        Ok(MogValue::Str(s.replace(&old, &new)))
                    }
                    _ => Ok(MogValue::Int(0)),
                }
            }
            MogValue::Map(m) => {
                match method {
                    "has" => {
                        let key = match args.first() { Some(MogValue::Str(s)) => s.clone(), _ => String::new() };
                        Ok(MogValue::Bool(m.contains_key(&key)))
                    }
                    "keys" => {
                        Ok(MogValue::Array(m.keys().map(|k| MogValue::Str(k.clone())).collect()))
                    }
                    "values" => {
                        Ok(MogValue::Array(m.values().cloned().collect()))
                    }
                    _ => Ok(MogValue::Int(0)),
                }
            }
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn eval_field_access(&mut self, obj: &Expr, field: &str, env: &Env) -> Result<MogValue, ControlFlow> {
        let obj_val = self.eval_expr(obj, env)?;
        match &obj_val {
            MogValue::Struct { fields, .. } => {
                fields.get(field).cloned().ok_or(ControlFlow::Return(MogValue::Int(0)))
            }
            MogValue::Array(arr) => {
                if field == "len" { return Ok(MogValue::Int(arr.len() as i64)); }
                Ok(MogValue::Int(0))
            }
            MogValue::Str(s) => {
                if field == "len" { return Ok(MogValue::Int(s.len() as i64)); }
                Ok(MogValue::Int(0))
            }
            MogValue::Map(m) => {
                if field == "len" { return Ok(MogValue::Int(m.len() as i64)); }
                Ok(MogValue::Int(0))
            }
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn eval_index_access(&mut self, obj: &Expr, index: &Expr, env: &Env) -> Result<MogValue, ControlFlow> {
        let obj_val = self.eval_expr(obj, env)?;
        let idx = self.eval_expr(index, env)?;
        match &obj_val {
            MogValue::Array(arr) => {
                let i = idx.to_i64() as usize;
                Ok(arr.get(i).cloned().unwrap_or(MogValue::Int(0)))
            }
            MogValue::Str(s) => {
                let i = idx.to_i64() as usize;
                Ok(s.chars().nth(i).map(|c| MogValue::Str(c.to_string())).unwrap_or(MogValue::Str(String::new())))
            }
            MogValue::Map(m) => {
                match &idx {
                    MogValue::Str(key) => Ok(m.get(key).cloned().unwrap_or(MogValue::Int(0))),
                    _ => Ok(MogValue::Int(0)),
                }
            }
            _ => Ok(MogValue::Int(0)),
        }
    }

    fn eval_match(&mut self, subject: &Expr, arms: &[MatchArm], env: &Env) -> Result<MogValue, ControlFlow> {
        let subj = self.eval_expr(subject, env)?;
        for arm in arms {
            let arm_env = Environment::child(env);
            if self.match_pattern(&arm.pattern, &subj, &arm_env) {
                return match &arm.body {
                    MatchBody::Expr(e) => self.eval_expr(e, &arm_env),
                    MatchBody::Block(b) => {
                        match self.exec_block(b, &arm_env) {
                            Ok(()) => Ok(MogValue::Int(0)),
                            Err(ControlFlow::Return(v)) => Ok(v),
                            Err(cf) => Err(cf),
                        }
                    }
                };
            }
        }
        Ok(MogValue::Int(0))
    }

    fn match_pattern(&self, pattern: &Pattern, value: &MogValue, env: &Env) -> bool {
        match pattern {
            Pattern::Wildcard => true,
            Pattern::Lit(lit) => {
                match lit {
                    LitPatternValue::Int(lit) => {
                        matches!(value, MogValue::Int(i) if *i == *lit)
                    }
                    LitPatternValue::Float(lit) => {
                        matches!(value, MogValue::Float(f) if *f == *lit)
                    }
                    LitPatternValue::String(lit) => {
                        matches!(value, MogValue::Str(s) if s == lit)
                    }
                    LitPatternValue::Bool(lit) => {
                        matches!(value, MogValue::Bool(b) if b == lit)
                    }
                }
            }
            Pattern::Ok { binding } => {
                if let MogValue::Result { is_ok: true, value: inner } = value {
                    Environment::define(env, binding, (**inner).clone());
                    true
                } else { false }
            }
            Pattern::Err { binding } => {
                if let MogValue::Result { is_ok: false, value: inner } = value {
                    Environment::define(env, binding, (**inner).clone());
                    true
                } else { false }
            }
            Pattern::Some { binding } => {
                if let MogValue::Optional { is_some: true, value: inner } = value {
                    Environment::define(env, binding, (**inner).clone());
                    true
                } else { false }
            }
            Pattern::None_ => {
                matches!(value, MogValue::Optional { is_some: false, .. })
            }
            Pattern::Ident { name } => {
                Environment::define(env, name, value.clone());
                true
            }
        }
    }
}
