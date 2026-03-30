"""Mog token types and Token dataclass."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token types for the Mog lexer."""
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()
    FSTRING_LIT = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()

    # Keywords
    FN = auto()
    PUB = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    TO = auto()
    MATCH = auto()
    STRUCT = auto()
    BREAK = auto()
    CONTINUE = auto()
    AND = auto()      # 'and' keyword
    OR = auto()       # 'or' keyword
    AS = auto()
    OK = auto()
    ERR = auto()
    SOME = auto()
    IS = auto()
    TYPE = auto()
    REQUIRES = auto()
    OPTIONAL_KW = auto()
    IMPORT = auto()
    ASYNC = auto()
    AWAIT = auto()
    SPAWN = auto()
    RESULT = auto()

    # Type keywords
    INT_TYPE = auto()     # int
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()
    U8 = auto()
    U16 = auto()
    U32 = auto()
    U64 = auto()
    FLOAT_TYPE = auto()   # float
    F16 = auto()
    BF16 = auto()
    F32 = auto()
    F64 = auto()
    BOOL_TYPE = auto()
    STRING_TYPE = auto()

    # Identifiers
    IDENT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    STAR_STAR = auto()
    EQ_EQ = auto()
    BANG_EQ = auto()
    LT = auto()
    GT = auto()
    LT_EQ = auto()
    GT_EQ = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    SHL = auto()
    SHR = auto()
    AMP_AMP = auto()
    PIPE_PIPE = auto()
    BANG = auto()
    QUESTION = auto()
    DOT_DOT = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMICOLON = auto()
    COLON = auto()
    DOT = auto()
    COLON_EQ = auto()
    EQ = auto()
    ARROW = auto()       # ->
    FAT_ARROW = auto()   # =>
    UNDERSCORE = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    type: TT
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line})"


# Keyword lookup table
KEYWORDS: dict[str, TT] = {
    "fn": TT.FN, "pub": TT.PUB, "return": TT.RETURN,
    "if": TT.IF, "else": TT.ELSE, "while": TT.WHILE,
    "for": TT.FOR, "in": TT.IN, "to": TT.TO,
    "match": TT.MATCH, "struct": TT.STRUCT,
    "break": TT.BREAK, "continue": TT.CONTINUE,
    "and": TT.AND, "or": TT.OR, "as": TT.AS,
    "true": TT.TRUE, "false": TT.FALSE, "none": TT.NONE,
    "ok": TT.OK, "err": TT.ERR, "some": TT.SOME,
    "is": TT.IS, "type": TT.TYPE,
    "requires": TT.REQUIRES, "optional": TT.OPTIONAL_KW,
    "import": TT.IMPORT,
    "async": TT.ASYNC, "await": TT.AWAIT, "spawn": TT.SPAWN,
    "Result": TT.RESULT,
    # Type keywords
    "int": TT.INT_TYPE, "i8": TT.I8, "i16": TT.I16,
    "i32": TT.I32, "i64": TT.I64,
    "u8": TT.U8, "u16": TT.U16, "u32": TT.U32, "u64": TT.U64,
    "float": TT.FLOAT_TYPE, "f16": TT.F16, "bf16": TT.BF16,
    "f32": TT.F32, "f64": TT.F64,
    "bool": TT.BOOL_TYPE, "string": TT.STRING_TYPE,
}
