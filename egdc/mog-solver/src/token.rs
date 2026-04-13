//! Mog token types and Token struct.

use std::fmt;

/// Token types for the Mog lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    // Literals
    IntLit,
    FloatLit,
    StringLit,
    FStringLit,
    True,
    False,
    None,

    // Keywords
    Fn,
    Pub,
    Return,
    If,
    Else,
    While,
    For,
    In,
    To,
    Match,
    Struct,
    Break,
    Continue,
    And,      // 'and' keyword
    Or,       // 'or' keyword
    As,
    Ok,
    Err,
    Some,
    Is,
    Type,
    Requires,
    OptionalKw,
    Import,
    Async,
    Await,
    Spawn,
    Result,

    // Type keywords
    IntType,   // int
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    FloatType, // float
    F16,
    Bf16,
    F32,
    F64,
    BoolType,
    StringType,

    // Identifiers
    Ident,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar,
    EqEq,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Amp,
    Pipe,
    Caret,
    Shl,
    Shr,
    AmpAmp,
    PipePipe,
    Bang,
    Question,
    DotDot,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semicolon,
    Colon,
    Dot,
    ColonEq,
    Eq,
    Arrow,      // ->
    FatArrow,   // =>
    Underscore,

    // Special
    Eof,
    Newline,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A single token from the lexer.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub value: String,
    pub line: u32,
    pub col: u32,
}

impl Token {
    pub fn new(kind: TokenKind, value: impl Into<String>, line: u32, col: u32) -> Self {
        Token {
            kind,
            value: value.into(),
            line,
            col,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Token({:?}, {:?}, L{})", self.kind, self.value, self.line)
    }
}

/// Keyword lookup table.
pub fn keyword_lookup(word: &str) -> Option<TokenKind> {
    Some(match word {
        "fn" => TokenKind::Fn,
        "pub" => TokenKind::Pub,
        "return" => TokenKind::Return,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "while" => TokenKind::While,
        "for" => TokenKind::For,
        "in" => TokenKind::In,
        "to" => TokenKind::To,
        "match" => TokenKind::Match,
        "struct" => TokenKind::Struct,
        "break" => TokenKind::Break,
        "continue" => TokenKind::Continue,
        "and" => TokenKind::And,
        "or" => TokenKind::Or,
        "as" => TokenKind::As,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        "none" => TokenKind::None,
        "ok" => TokenKind::Ok,
        "err" => TokenKind::Err,
        "some" => TokenKind::Some,
        "is" => TokenKind::Is,
        "type" => TokenKind::Type,
        "requires" => TokenKind::Requires,
        "optional" => TokenKind::OptionalKw,
        "import" => TokenKind::Import,
        "async" => TokenKind::Async,
        "await" => TokenKind::Await,
        "spawn" => TokenKind::Spawn,
        "Result" => TokenKind::Result,
        // Type keywords
        "int" => TokenKind::IntType,
        "i8" => TokenKind::I8,
        "i16" => TokenKind::I16,
        "i32" => TokenKind::I32,
        "i64" => TokenKind::I64,
        "u8" => TokenKind::U8,
        "u16" => TokenKind::U16,
        "u32" => TokenKind::U32,
        "u64" => TokenKind::U64,
        "float" => TokenKind::FloatType,
        "f16" => TokenKind::F16,
        "bf16" => TokenKind::Bf16,
        "f32" => TokenKind::F32,
        "f64" => TokenKind::F64,
        "bool" => TokenKind::BoolType,
        "string" => TokenKind::StringType,
        _ => return None,
    })
}
