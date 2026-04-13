//! Mog AST node definitions.

use std::fmt;

// --- Program level ---

pub struct Program {
    pub declarations: Vec<Decl>,
}

/// Top-level declaration (or statement -- we use the same enum for both).
#[derive(Clone)]
pub enum Decl {
    FnDecl(FnDecl),
    StructDecl(StructDecl),
    TypeAlias(TypeAlias),
    RequiresDecl(RequiresDecl),
    // Statements can appear at top level too
    VarDecl(VarDecl),
    Assignment(Assignment),
    ReturnStmt(ReturnStmt),
    IfStmt(IfStmt),
    WhileStmt(WhileStmt),
    ForToStmt(ForToStmt),
    ForInRangeStmt(ForInRangeStmt),
    ForInStmt(ForInStmt),
    BreakStmt,
    ContinueStmt,
    ExprStmt(ExprStmt),
    Block(Block),
}

#[derive(Clone)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub body: Block,
    pub is_pub: bool,
}

#[derive(Clone)]
pub struct Param {
    pub name: String,
    pub type_ann: Option<String>,
    pub default: Option<Expr>,
}

#[derive(Clone)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, String)>, // (name, type_string)
}

#[derive(Clone)]
pub struct TypeAlias {
    pub name: String,
    pub target: String,
}

#[derive(Clone)]
pub struct RequiresDecl {
    pub capabilities: Vec<String>,
}

#[derive(Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

// --- Statements ---

/// Statements within a block.
#[derive(Clone)]
pub enum Stmt {
    VarDecl(VarDecl),
    Assignment(Assignment),
    ReturnStmt(ReturnStmt),
    IfStmt(IfStmt),
    WhileStmt(WhileStmt),
    ForToStmt(ForToStmt),
    ForInRangeStmt(ForInRangeStmt),
    ForInStmt(ForInStmt),
    BreakStmt,
    ContinueStmt,
    ExprStmt(ExprStmt),
    FnDecl(FnDecl),
    StructDecl(StructDecl),
    Block(Block),
}

#[derive(Clone)]
pub struct VarDecl {
    pub name: String,
    pub type_ann: Option<String>,
    pub value: Expr,
}

#[derive(Clone)]
pub struct Assignment {
    pub target: Expr,
    pub value: Expr,
}

#[derive(Clone)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
}

#[derive(Clone)]
pub struct IfStmt {
    pub condition: Expr,
    pub then_block: Block,
    pub else_block: Option<ElseBranch>,
}

/// Else branch: either a block or another if (else-if chain).
#[derive(Clone)]
pub enum ElseBranch {
    Block(Block),
    If(Box<IfStmt>),
}

#[derive(Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Block,
}

#[derive(Clone)]
pub struct ForToStmt {
    pub var_name: String,
    pub start: Expr,
    pub end: Expr,
    pub body: Block,
}

#[derive(Clone)]
pub struct ForInRangeStmt {
    pub var_name: String,
    pub start: Expr,
    pub end: Expr,
    pub body: Block,
}

#[derive(Clone)]
pub struct ForInStmt {
    pub var_name: String,
    pub index_name: Option<String>,
    pub iterable: Expr,
    pub body: Block,
}

#[derive(Clone)]
pub struct ExprStmt {
    pub expr: Expr,
}

// --- Expressions ---

#[derive(Clone)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    StringLit(String),
    FStringLit(String),
    BoolLit(bool),
    NoneLit,
    Ident(String),
    BinOp(BinOpData),
    UnaryOp(UnaryOpData),
    Call(CallData),
    FieldAccess(FieldAccessData),
    IndexAccess(IndexAccessData),
    ArrayLit(Vec<Expr>),
    ArrayFill { value: Box<Expr>, count: Box<Expr> },
    MapLit(Vec<(Expr, Expr)>),
    StructConstruct { name: String, fields: Vec<(String, Expr)> },
    OkExpr(Box<Expr>),
    ErrExpr(Box<Expr>),
    SomeExpr(Box<Expr>),
    IfExpr(IfExprData),
    MatchExpr(MatchExprData),
    CastExpr(CastExprData),
    PropagateExpr(Box<Expr>),
    ClosureLit(Box<ClosureLitData>),
    RangeExpr { start: Box<Expr>, end: Box<Expr> },
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::IntLit(v) => write!(f, "IntLit({v})"),
            Expr::FloatLit(v) => write!(f, "FloatLit({v})"),
            Expr::StringLit(v) => write!(f, "StringLit({v:?})"),
            Expr::BoolLit(v) => write!(f, "BoolLit({v})"),
            Expr::Ident(v) => write!(f, "Ident({v})"),
            _ => write!(f, "Expr"),
        }
    }
}

#[derive(Clone)]
pub struct BinOpData {
    pub op: String,
    pub left: Box<Expr>,
    pub right: Box<Expr>,
}

#[derive(Clone)]
pub struct UnaryOpData {
    pub op: String,
    pub operand: Box<Expr>,
}

#[derive(Clone)]
pub struct CallData {
    pub func: Box<Expr>,
    pub args: Vec<Expr>,
}

#[derive(Clone)]
pub struct FieldAccessData {
    pub obj: Box<Expr>,
    pub field: String,
}

#[derive(Clone)]
pub struct IndexAccessData {
    pub obj: Box<Expr>,
    pub index: Box<Expr>,
}

#[derive(Clone)]
pub struct IfExprData {
    pub condition: Box<Expr>,
    pub then_expr: Box<Expr>,
    pub else_expr: Box<Expr>,
}

#[derive(Clone)]
pub struct MatchExprData {
    pub subject: Box<Expr>,
    pub arms: Vec<MatchArm>,
}

#[derive(Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: MatchBody,
}

/// The body of a match arm: either an expression or a block.
#[derive(Clone)]
pub enum MatchBody {
    Expr(Expr),
    Block(Block),
}

#[derive(Clone)]
pub struct CastExprData {
    pub expr: Box<Expr>,
    pub target_type: String,
}

#[derive(Clone)]
pub struct ClosureLitData {
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub body: ClosureBody,
}

/// Closure body: either a block or an expression.
#[derive(Clone)]
pub enum ClosureBody {
    Block(Block),
    Expr(Box<Expr>),
}

// --- Patterns ---

#[derive(Clone)]
pub enum Pattern {
    Lit(LitPatternValue),
    Wildcard,
    Ok { binding: String },
    Err { binding: String },
    Some { binding: String },
    None_,
    Ident { name: String },
}

#[derive(Clone)]
pub enum LitPatternValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}
