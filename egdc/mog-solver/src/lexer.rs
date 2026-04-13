//! Mog lexer -- tokenizes source code into a token stream.

use crate::token::{keyword_lookup, Token, TokenKind};

/// Lexing error.
#[derive(Debug)]
pub struct LexError {
    pub msg: String,
    pub line: u32,
    pub col: u32,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lex error at L{}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for LexError {}

/// Tokenizes Mog source code.
pub struct Lexer {
    src: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
    tokens: Vec<Token>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            src: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
            tokens: Vec::new(),
        }
    }

    fn ch(&self) -> char {
        if self.pos < self.src.len() {
            self.src[self.pos]
        } else {
            '\0'
        }
    }

    fn peek_offset(&self, offset: usize) -> char {
        let p = self.pos + offset;
        if p < self.src.len() {
            self.src[p]
        } else {
            '\0'
        }
    }

    fn advance(&mut self) -> char {
        let ch = self.ch();
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        ch
    }

    fn emit(&mut self, kind: TokenKind, value: impl Into<String>, line: u32, col: u32) {
        self.tokens.push(Token::new(kind, value, line, col));
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.src.len() {
            let ch = self.ch();
            if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
                self.advance();
            } else if ch == '/' && self.peek_offset(1) == '/' {
                // Line comment
                while self.pos < self.src.len() && self.ch() != '\n' {
                    self.advance();
                }
            } else if ch == '/' && self.peek_offset(1) == '*' {
                // Block comment
                self.advance(); // /
                self.advance(); // *
                let mut depth: i32 = 1;
                while self.pos < self.src.len() && depth > 0 {
                    if self.ch() == '*' && self.peek_offset(1) == '/' {
                        self.advance();
                        self.advance();
                        depth -= 1;
                    } else if self.ch() == '/' && self.peek_offset(1) == '*' {
                        self.advance();
                        self.advance();
                        depth += 1;
                    } else {
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self, _is_fstring: bool) -> String {
        let mut parts = Vec::new();
        while self.pos < self.src.len() && self.ch() != '"' {
            if self.ch() == '\\' {
                self.advance();
                let esc = self.advance();
                match esc {
                    'n' => parts.push('\n'),
                    't' => parts.push('\t'),
                    '\\' => parts.push('\\'),
                    '"' => parts.push('"'),
                    '{' => parts.push('{'),
                    _ => {
                        parts.push('\\');
                        parts.push(esc);
                    }
                }
            } else {
                parts.push(self.advance());
            }
        }
        if self.pos < self.src.len() {
            self.advance(); // closing "
        }
        parts.into_iter().collect()
    }

    fn read_number(&mut self) -> (TokenKind, String) {
        let start = self.pos;
        let mut is_float = false;
        while self.pos < self.src.len() && (self.ch().is_ascii_digit() || self.ch() == '_') {
            self.advance();
        }
        if self.ch() == '.' && self.peek_offset(1).is_ascii_digit() {
            is_float = true;
            self.advance(); // .
            while self.pos < self.src.len() && (self.ch().is_ascii_digit() || self.ch() == '_') {
                self.advance();
            }
        }
        if self.ch() == 'e' || self.ch() == 'E' {
            is_float = true;
            self.advance();
            if self.ch() == '+' || self.ch() == '-' {
                self.advance();
            }
            while self.pos < self.src.len() && self.ch().is_ascii_digit() {
                self.advance();
            }
        }
        let val: String = self.src[start..self.pos].iter().filter(|c| **c != '_').collect();
        (if is_float { TokenKind::FloatLit } else { TokenKind::IntLit }, val)
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.src.len() && (self.ch().is_alphanumeric() || self.ch() == '_') {
            self.advance();
        }
        self.src[start..self.pos].iter().collect()
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.src.len() {
                break;
            }

            let line = self.line;
            let col = self.col;
            let ch = self.ch();

            // F-strings
            if ch == 'f' && self.peek_offset(1) == '"' {
                self.advance(); // f
                self.advance(); // "
                let val = self.read_string(true);
                self.emit(TokenKind::FStringLit, val, line, col);
                continue;
            }

            // Identifiers and keywords
            if ch.is_alphabetic() || ch == '_' {
                let word = self.read_ident();
                let kind = keyword_lookup(&word).unwrap_or(TokenKind::Ident);
                self.emit(kind, word, line, col);
                continue;
            }

            // Numbers
            if ch.is_ascii_digit() {
                let (kind, val) = self.read_number();
                self.emit(kind, val, line, col);
                continue;
            }

            // Strings
            if ch == '"' {
                self.advance();
                let val = self.read_string(false);
                self.emit(TokenKind::StringLit, val, line, col);
                continue;
            }

            // Two-char operators
            let nch = self.peek_offset(1);
            let two = format!("{ch}{nch}");

            macro_rules! two_char_op {
                ($s:expr, $kind:expr) => {
                    if two == $s {
                        self.advance();
                        self.advance();
                        self.emit($kind, $s, line, col);
                        continue;
                    }
                };
            }

            two_char_op!(":=", TokenKind::ColonEq);
            two_char_op!("==", TokenKind::EqEq);
            two_char_op!("!=", TokenKind::BangEq);
            two_char_op!("<=", TokenKind::LtEq);
            two_char_op!(">=", TokenKind::GtEq);
            two_char_op!("->", TokenKind::Arrow);
            two_char_op!("=>", TokenKind::FatArrow);
            two_char_op!("&&", TokenKind::AmpAmp);
            two_char_op!("||", TokenKind::PipePipe);
            two_char_op!("<<", TokenKind::Shl);
            two_char_op!(">>", TokenKind::Shr);
            two_char_op!("**", TokenKind::StarStar);
            two_char_op!("..", TokenKind::DotDot);

            // Single-char
            self.advance();
            let kind = match ch {
                '+' => TokenKind::Plus,
                '-' => TokenKind::Minus,
                '*' => TokenKind::Star,
                '/' => TokenKind::Slash,
                '%' => TokenKind::Percent,
                '<' => TokenKind::Lt,
                '>' => TokenKind::Gt,
                '&' => TokenKind::Amp,
                '|' => TokenKind::Pipe,
                '^' => TokenKind::Caret,
                '!' => TokenKind::Bang,
                '?' => TokenKind::Question,
                '(' => TokenKind::LParen,
                ')' => TokenKind::RParen,
                '{' => TokenKind::LBrace,
                '}' => TokenKind::RBrace,
                '[' => TokenKind::LBracket,
                ']' => TokenKind::RBracket,
                ',' => TokenKind::Comma,
                ';' => TokenKind::Semicolon,
                ':' => TokenKind::Colon,
                '.' => TokenKind::Dot,
                '=' => TokenKind::Eq,
                '_' => TokenKind::Underscore,
                _ => {
                    return self.tokens.clone(); // Error will be apparent to parser
                }
            };
            self.emit(kind, ch.to_string(), line, col);
        }

        self.emit(TokenKind::Eof, "", self.line, self.col);
        self.tokens.clone()
    }
}

/// Tokenize Mog source code.
pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut lexer = Lexer::new(source);
    Ok(lexer.tokenize())
}
