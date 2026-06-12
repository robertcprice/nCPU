"""Mog AST node definitions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# --- Program level ---

@dataclass
class Program:
    declarations: list[Any]

@dataclass
class FnDecl:
    name: str
    params: list[Param]
    return_type: Optional[str]
    body: Block
    is_pub: bool = False

@dataclass
class Param:
    name: str
    type_ann: Optional[str] = None
    default: Optional[Any] = None  # Expr

@dataclass
class StructDecl:
    name: str
    fields: list[tuple[str, str]]  # (name, type_string)

@dataclass
class TypeAlias:
    name: str
    target: str

@dataclass
class RequiresDecl:
    capabilities: list[str]

@dataclass
class Block:
    stmts: list[Any]

# --- Statements ---

@dataclass
class VarDecl:
    name: str
    type_ann: Optional[str]
    value: Any  # Expr

@dataclass
class Assignment:
    target: Any  # LValue (Ident, FieldAccess, IndexAccess)
    value: Any   # Expr

@dataclass
class ReturnStmt:
    value: Optional[Any]

@dataclass
class IfStmt:
    condition: Any  # Expr
    then_block: Block
    else_block: Optional[Block | Any]  # Block or IfStmt (else if)

@dataclass
class WhileStmt:
    condition: Any
    body: Block

@dataclass
class ForToStmt:
    var_name: str
    start: Any  # Expr
    end: Any    # Expr
    body: Block

@dataclass
class ForInRangeStmt:
    var_name: str
    start: Any
    end: Any
    body: Block

@dataclass
class ForInStmt:
    var_name: str
    index_name: Optional[str]  # for i, item in ...
    iterable: Any
    body: Block

@dataclass
class MatchStmt:
    subject: Any  # Expr
    arms: list[MatchArm]

@dataclass
class MatchArm:
    pattern: Any   # Pattern
    body: Any      # Expr or Block

@dataclass
class BreakStmt:
    pass

@dataclass
class ContinueStmt:
    pass

@dataclass
class ExprStmt:
    expr: Any

# --- Expressions ---

@dataclass
class IntLit:
    value: int

@dataclass
class FloatLit:
    value: float

@dataclass
class StringLit:
    value: str

@dataclass
class FStringLit:
    raw: str  # Raw template string with {expr} placeholders

@dataclass
class BoolLit:
    value: bool

@dataclass
class NoneLit:
    pass

@dataclass
class Ident:
    name: str

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any

@dataclass
class UnaryOp:
    op: str  # '-' or '!'
    operand: Any

@dataclass
class Call:
    func: Any    # Expr (Ident or FieldAccess for methods)
    args: list[Any]

@dataclass
class FieldAccess:
    obj: Any
    field: str

@dataclass
class IndexAccess:
    obj: Any
    index: Any

@dataclass
class ArrayLit:
    elements: list[Any]

@dataclass
class ArrayFill:
    value: Any
    count: Any

@dataclass
class MapLit:
    pairs: list[tuple[Any, Any]]

@dataclass
class StructConstruct:
    name: str
    fields: list[tuple[str, Any]]  # (field_name, expr)

@dataclass
class OkExpr:
    value: Any

@dataclass
class ErrExpr:
    value: Any

@dataclass
class SomeExpr:
    value: Any

@dataclass
class IfExpr:
    condition: Any
    then_expr: Any
    else_expr: Any

@dataclass
class MatchExpr:
    subject: Any
    arms: list[MatchArm]

@dataclass
class CastExpr:
    expr: Any
    target_type: str

@dataclass
class PropagateExpr:
    """The ? operator on a Result/Optional."""
    expr: Any

@dataclass
class ClosureLit:
    params: list[Param]
    return_type: Optional[str]
    body: Any  # Expr or Block

@dataclass
class RangeExpr:
    start: Any
    end: Any

# --- Patterns ---

@dataclass
class LitPattern:
    value: Any  # IntLit, StringLit, BoolLit

@dataclass
class WildcardPattern:
    pass

@dataclass
class OkPattern:
    binding: str

@dataclass
class ErrPattern:
    binding: str

@dataclass
class SomePattern:
    binding: str

@dataclass
class NonePattern:
    pass

@dataclass
class IdentPattern:
    name: str
