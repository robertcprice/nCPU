//! Mog recursive descent parser.

use crate::ast::*;
use crate::token::{Token, TokenKind};

/// Parse error.
#[derive(Debug)]
pub struct ParseError {
    pub msg: String,
    pub line: u32,
    pub col: u32,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error at L{}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn cur(&self) -> &Token {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos]
        } else {
            self.tokens.last().unwrap()
        }
    }

    fn peek(&self, offset: usize) -> &Token {
        let p = self.pos + offset;
        if p < self.tokens.len() {
            &self.tokens[p]
        } else {
            self.tokens.last().unwrap()
        }
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.cur().kind == kind
    }

    fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.contains(&self.cur().kind)
    }

    fn eat(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        let tok = self.cur().clone();
        if tok.kind != kind {
            return Err(ParseError {
                msg: format!(
                    "expected {:?}, got {:?} ({:?})",
                    kind, tok.kind, tok.value
                ),
                line: tok.line,
                col: tok.col,
            });
        }
        self.pos += 1;
        Ok(tok)
    }

    fn maybe(&mut self, kind: TokenKind) -> Option<Token> {
        if self.at(kind) {
            self.pos += 1;
            Some(self.tokens[self.pos - 1].clone())
        } else {
            None
        }
    }

    fn skip_semis(&mut self) {
        while self.at(TokenKind::Semicolon) {
            self.pos += 1;
        }
    }

    // --- Type annotations ---

    fn is_type_start(&self) -> bool {
        self.at_any(&[
            TokenKind::IntType, TokenKind::I8, TokenKind::I16, TokenKind::I32, TokenKind::I64,
            TokenKind::U8, TokenKind::U16, TokenKind::U32, TokenKind::U64,
            TokenKind::FloatType, TokenKind::F16, TokenKind::Bf16, TokenKind::F32, TokenKind::F64,
            TokenKind::BoolType, TokenKind::StringType, TokenKind::Result, TokenKind::Question,
            TokenKind::LBracket, TokenKind::Ident,
        ])
    }

    fn parse_type(&mut self) -> Result<String, ParseError> {
        if self.at(TokenKind::Question) {
            self.eat(TokenKind::Question)?;
            let inner = self.parse_type()?;
            return Ok(format!("?{inner}"));
        }
        if self.at(TokenKind::Result) {
            self.eat(TokenKind::Result)?;
            if self.at(TokenKind::Lt) {
                self.eat(TokenKind::Lt)?;
                let inner = self.parse_type()?;
                self.eat(TokenKind::Gt)?;
                return Ok(format!("Result<{inner}>"));
            }
            return Ok("Result".to_string());
        }
        if self.at(TokenKind::LBracket) {
            self.eat(TokenKind::LBracket)?;
            if self.at(TokenKind::RBracket) {
                self.eat(TokenKind::RBracket)?;
                let inner = self.parse_type()?;
                return Ok(format!("[]{inner}"));
            }
            let key = self.parse_type()?;
            self.eat(TokenKind::RBracket)?;
            if self.is_type_start() {
                let val = self.parse_type()?;
                return Ok(format!("[{key}]{val}"));
            }
            return Ok(format!("[{key}]"));
        }
        let type_toks = [
            TokenKind::IntType, TokenKind::I8, TokenKind::I16, TokenKind::I32, TokenKind::I64,
            TokenKind::U8, TokenKind::U16, TokenKind::U32, TokenKind::U64,
            TokenKind::FloatType, TokenKind::F16, TokenKind::Bf16, TokenKind::F32, TokenKind::F64,
            TokenKind::BoolType, TokenKind::StringType, TokenKind::Ident,
        ];
        let tok = self.cur().clone();
        if type_toks.contains(&tok.kind) {
            self.pos += 1;
            return Ok(tok.value.clone());
        }
        Err(ParseError {
            msg: format!("expected type, got {:?}", tok.kind),
            line: tok.line,
            col: tok.col,
        })
    }

    // --- Top level ---

    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut decls = Vec::new();
        while !self.at(TokenKind::Eof) {
            self.skip_semis();
            if self.at(TokenKind::Eof) {
                break;
            }
            decls.push(self.parse_declaration()?);
            self.skip_semis();
        }
        Ok(Program { declarations: decls })
    }

    fn parse_declaration(&mut self) -> Result<Decl, ParseError> {
        if self.at(TokenKind::Fn)
            || (self.at(TokenKind::Pub) && self.peek(1).kind == TokenKind::Fn)
        {
            return self.parse_fn_decl().map(Decl::FnDecl);
        }
        if self.at(TokenKind::Struct) {
            return self.parse_struct_decl().map(Decl::StructDecl);
        }
        if self.at(TokenKind::Type) {
            return self.parse_type_alias().map(Decl::TypeAlias);
        }
        if self.at(TokenKind::Requires) {
            return self.parse_requires().map(Decl::RequiresDecl);
        }
        let tok = self.cur();
        Err(ParseError {
            msg: format!("expected declaration, got {:?}", tok.kind),
            line: tok.line,
            col: tok.col,
        })
    }

    fn parse_fn_decl(&mut self) -> Result<FnDecl, ParseError> {
        let is_pub = self.maybe(TokenKind::Pub).is_some();
        self.eat(TokenKind::Fn)?;
        let name = self.eat(TokenKind::Ident)?.value;
        self.eat(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.eat(TokenKind::RParen)?;
        let return_type = if self.at(TokenKind::Arrow) {
            self.eat(TokenKind::Arrow)?;
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(FnDecl { name, params, return_type, body, is_pub })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        while !self.at(TokenKind::RParen) {
            let name = self.eat(TokenKind::Ident)?.value;
            let mut type_ann = None;
            let mut default = None;
            if self.at(TokenKind::Colon) {
                self.eat(TokenKind::Colon)?;
                type_ann = Some(self.parse_type()?);
            }
            if self.at(TokenKind::Eq) {
                self.eat(TokenKind::Eq)?;
                default = Some(self.parse_expr()?);
            }
            params.push(Param { name, type_ann, default });
            if !self.at(TokenKind::RParen) {
                self.eat(TokenKind::Comma)?;
            }
        }
        Ok(params)
    }

    fn parse_struct_decl(&mut self) -> Result<StructDecl, ParseError> {
        self.eat(TokenKind::Struct)?;
        let name = self.eat(TokenKind::Ident)?.value;
        self.eat(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let fname = self.eat(TokenKind::Ident)?.value;
            self.eat(TokenKind::Colon)?;
            let ftype = self.parse_type()?;
            fields.push((fname, ftype));
            self.maybe(TokenKind::Comma);
        }
        self.eat(TokenKind::RBrace)?;
        Ok(StructDecl { name, fields })
    }

    fn parse_type_alias(&mut self) -> Result<TypeAlias, ParseError> {
        self.eat(TokenKind::Type)?;
        let name = self.eat(TokenKind::Ident)?.value;
        self.eat(TokenKind::Eq)?;
        let target = self.parse_type()?;
        self.eat(TokenKind::Semicolon)?;
        Ok(TypeAlias { name, target })
    }

    fn parse_requires(&mut self) -> Result<RequiresDecl, ParseError> {
        self.eat(TokenKind::Requires)?;
        let mut caps = vec![self.eat(TokenKind::Ident)?.value];
        while self.at(TokenKind::Comma) {
            self.eat(TokenKind::Comma)?;
            caps.push(self.eat(TokenKind::Ident)?.value);
        }
        self.eat(TokenKind::Semicolon)?;
        Ok(RequiresDecl { capabilities: caps })
    }

    // --- Blocks and statements ---

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        self.eat(TokenKind::LBrace)?;
        let mut stmts = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            self.skip_semis();
            if self.at(TokenKind::RBrace) {
                break;
            }
            stmts.push(self.parse_stmt()?);
            self.skip_semis();
        }
        self.eat(TokenKind::RBrace)?;
        Ok(Block { stmts })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        if self.at(TokenKind::Return) {
            return self.parse_return();
        }
        if self.at(TokenKind::If) {
            return self.parse_if();
        }
        if self.at(TokenKind::While) {
            return self.parse_while();
        }
        if self.at(TokenKind::For) {
            return self.parse_for();
        }
        if self.at(TokenKind::Match) {
            let me = self.parse_match_expr_or_stmt()?;
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::ExprStmt(ExprStmt { expr: me }));
        }
        if self.at(TokenKind::Break) {
            self.eat(TokenKind::Break)?;
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::BreakStmt);
        }
        if self.at(TokenKind::Continue) {
            self.eat(TokenKind::Continue)?;
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::ContinueStmt);
        }
        if self.at(TokenKind::Fn) {
            let fd = self.parse_fn_decl()?;
            return Ok(Stmt::FnDecl(fd));
        }
        if self.at(TokenKind::Struct) {
            let sd = self.parse_struct_decl()?;
            return Ok(Stmt::StructDecl(sd));
        }

        // Variable declaration or assignment or expression statement
        if self.at(TokenKind::Ident) {
            if self.peek(1).kind == TokenKind::ColonEq {
                return self.parse_var_decl_walrus();
            }
            if self.peek(1).kind == TokenKind::Colon && self.peek(2).kind != TokenKind::Eq {
                return self.parse_var_decl_typed();
            }
        }

        // Expression (which might be assignment: expr = expr)
        let expr = self.parse_expr()?;
        if self.at(TokenKind::Eq) {
            self.eat(TokenKind::Eq)?;
            let value = self.parse_expr()?;
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::Assignment(Assignment { target: expr, value }));
        }
        if self.at(TokenKind::ColonEq) {
            self.eat(TokenKind::ColonEq)?;
            let value = self.parse_expr()?;
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::Assignment(Assignment { target: expr, value }));
        }
        self.maybe(TokenKind::Semicolon);
        Ok(Stmt::ExprStmt(ExprStmt { expr }))
    }

    fn parse_var_decl_walrus(&mut self) -> Result<Stmt, ParseError> {
        let name = self.eat(TokenKind::Ident)?.value;
        self.eat(TokenKind::ColonEq)?;
        let value = self.parse_expr()?;
        self.maybe(TokenKind::Semicolon);
        Ok(Stmt::VarDecl(VarDecl { name, type_ann: None, value }))
    }

    fn parse_var_decl_typed(&mut self) -> Result<Stmt, ParseError> {
        let name = self.eat(TokenKind::Ident)?.value;
        self.eat(TokenKind::Colon)?;
        let type_ann = Some(self.parse_type()?);
        self.eat(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.maybe(TokenKind::Semicolon);
        Ok(Stmt::VarDecl(VarDecl { name, type_ann, value }))
    }

    fn parse_return(&mut self) -> Result<Stmt, ParseError> {
        self.eat(TokenKind::Return)?;
        if self.at(TokenKind::Semicolon) || self.at(TokenKind::RBrace) {
            self.maybe(TokenKind::Semicolon);
            return Ok(Stmt::ReturnStmt(ReturnStmt { value: None }));
        }
        let val = self.parse_expr()?;
        self.maybe(TokenKind::Semicolon);
        Ok(Stmt::ReturnStmt(ReturnStmt { value: Some(val) }))
    }

    fn parse_if(&mut self) -> Result<Stmt, ParseError> {
        self.eat(TokenKind::If)?;
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.at(TokenKind::Else) {
            self.eat(TokenKind::Else)?;
            if self.at(TokenKind::If) {
                Some(ElseBranch::If(Box::new(self.parse_if_inner()?)))
            } else {
                Some(ElseBranch::Block(self.parse_block()?))
            }
        } else {
            None
        };
        // Convert Stmt::IfStmt to IfStmt
        Ok(Stmt::IfStmt(IfStmt { condition, then_block, else_block }))
    }

    fn parse_if_inner(&mut self) -> Result<IfStmt, ParseError> {
        self.eat(TokenKind::If)?;
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let else_block = if self.at(TokenKind::Else) {
            self.eat(TokenKind::Else)?;
            if self.at(TokenKind::If) {
                Some(ElseBranch::If(Box::new(self.parse_if_inner()?)))
            } else {
                Some(ElseBranch::Block(self.parse_block()?))
            }
        } else {
            None
        };
        Ok(IfStmt { condition, then_block, else_block })
    }

    fn parse_while(&mut self) -> Result<Stmt, ParseError> {
        self.eat(TokenKind::While)?;
        let condition = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::WhileStmt(WhileStmt { condition, body }))
    }

    fn parse_for(&mut self) -> Result<Stmt, ParseError> {
        self.eat(TokenKind::For)?;
        let name = self.eat(TokenKind::Ident)?.value;

        // for name := start to end { }
        if self.at(TokenKind::ColonEq) {
            self.eat(TokenKind::ColonEq)?;
            let start = self.parse_expr()?;
            self.eat(TokenKind::To)?;
            let end = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::ForToStmt(ForToStmt {
                var_name: name,
                start,
                end,
                body,
            }));
        }

        // for name in expr { }  or  for name, name2 in expr { }
        let mut index_name = None;
        let mut var_name = name;
        if self.at(TokenKind::Comma) {
            self.eat(TokenKind::Comma)?;
            index_name = Some(var_name);
            var_name = self.eat(TokenKind::Ident)?.value;
        }

        self.eat(TokenKind::In)?;

        // Check for range pattern: expr..expr
        let start_expr = self.parse_expr()?;
        if self.at(TokenKind::DotDot) {
            self.eat(TokenKind::DotDot)?;
            let end_expr = self.parse_expr()?;
            let body = self.parse_block()?;
            return Ok(Stmt::ForInRangeStmt(ForInRangeStmt {
                var_name,
                start: start_expr,
                end: end_expr,
                body,
            }));
        }

        let iterable = start_expr;

        // Check if iterable is a range expr
        if let Expr::RangeExpr { .. } = &iterable {
            if let Expr::RangeExpr { start, end } = iterable {
                let body = self.parse_block()?;
                return Ok(Stmt::ForInRangeStmt(ForInRangeStmt {
                    var_name,
                    start: *start,
                    end: *end,
                    body,
                }));
            }
        }

        let body = self.parse_block()?;
        Ok(Stmt::ForInStmt(ForInStmt {
            var_name,
            index_name,
            iterable,
            body,
        }))
    }

    fn parse_match_expr_or_stmt(&mut self) -> Result<Expr, ParseError> {
        self.eat(TokenKind::Match)?;
        let subject = self.parse_expr()?;
        self.eat(TokenKind::LBrace)?;
        let mut arms = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let pattern = self.parse_pattern()?;
            self.eat(TokenKind::FatArrow)?;
            let body = if self.at(TokenKind::LBrace) {
                MatchBody::Block(self.parse_block()?)
            } else {
                MatchBody::Expr(self.parse_expr()?)
            };
            arms.push(MatchArm { pattern, body });
            self.maybe(TokenKind::Comma);
            self.maybe(TokenKind::Semicolon);
        }
        self.eat(TokenKind::RBrace)?;
        Ok(Expr::MatchExpr(MatchExprData {
            subject: Box::new(subject),
            arms,
        }))
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        if self.at(TokenKind::Underscore) {
            self.eat(TokenKind::Underscore)?;
            return Ok(Pattern::Wildcard);
        }
        if self.at(TokenKind::Ok) {
            self.eat(TokenKind::Ok)?;
            self.eat(TokenKind::LParen)?;
            let name = self.eat(TokenKind::Ident)?.value;
            self.eat(TokenKind::RParen)?;
            return Ok(Pattern::Ok { binding: name });
        }
        if self.at(TokenKind::Err) {
            self.eat(TokenKind::Err)?;
            self.eat(TokenKind::LParen)?;
            let name = self.eat(TokenKind::Ident)?.value;
            self.eat(TokenKind::RParen)?;
            return Ok(Pattern::Err { binding: name });
        }
        if self.at(TokenKind::Some) {
            self.eat(TokenKind::Some)?;
            self.eat(TokenKind::LParen)?;
            let name = self.eat(TokenKind::Ident)?.value;
            self.eat(TokenKind::RParen)?;
            return Ok(Pattern::Some { binding: name });
        }
        if self.at(TokenKind::None) {
            self.eat(TokenKind::None)?;
            return Ok(Pattern::None_);
        }
        if self.at(TokenKind::IntLit) {
            let tok = self.eat(TokenKind::IntLit)?;
            let v: i64 = tok.value.parse().unwrap();
            return Ok(Pattern::Lit(LitPatternValue::Int(v)));
        }
        if self.at(TokenKind::FloatLit) {
            let tok = self.eat(TokenKind::FloatLit)?;
            let v: f64 = tok.value.parse().unwrap();
            return Ok(Pattern::Lit(LitPatternValue::Float(v)));
        }
        if self.at(TokenKind::StringLit) {
            let tok = self.eat(TokenKind::StringLit)?;
            return Ok(Pattern::Lit(LitPatternValue::String(tok.value)));
        }
        if self.at(TokenKind::True) {
            self.eat(TokenKind::True)?;
            return Ok(Pattern::Lit(LitPatternValue::Bool(true)));
        }
        if self.at(TokenKind::False) {
            self.eat(TokenKind::False)?;
            return Ok(Pattern::Lit(LitPatternValue::Bool(false)));
        }
        if self.at(TokenKind::Minus) {
            self.eat(TokenKind::Minus)?;
            let tok = self.eat(TokenKind::IntLit)?;
            let v: i64 = tok.value.parse().unwrap();
            return Ok(Pattern::Lit(LitPatternValue::Int(-v)));
        }
        if self.at(TokenKind::Ident) {
            let tok = self.eat(TokenKind::Ident)?;
            return Ok(Pattern::Ident { name: tok.value });
        }
        let tok = self.cur();
        Err(ParseError {
            msg: format!("expected pattern, got {:?}", tok.kind),
            line: tok.line,
            col: tok.col,
        })
    }

    // --- Expressions ---

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while self.at(TokenKind::PipePipe) || self.at(TokenKind::Or) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_and()?;
            let op_str = if op == "or" { "||".to_string() } else { op };
            left = Expr::BinOp(BinOpData {
                op: op_str,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_comparison()?;
        while self.at(TokenKind::AmpAmp) || self.at(TokenKind::And) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_comparison()?;
            let op_str = if op == "and" { "&&".to_string() } else { op };
            left = Expr::BinOp(BinOpData {
                op: op_str,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_bitwise()?;
        if self.at_any(&[
            TokenKind::EqEq, TokenKind::BangEq, TokenKind::Lt, TokenKind::Gt,
            TokenKind::LtEq, TokenKind::GtEq,
        ]) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_bitwise()?;
            return Ok(Expr::BinOp(BinOpData {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }));
        }
        Ok(left)
    }

    fn parse_bitwise(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_shift()?;
        while self.at_any(&[TokenKind::Amp, TokenKind::Pipe, TokenKind::Caret]) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_shift()?;
            left = Expr::BinOp(BinOpData {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_additive()?;
        while self.at_any(&[TokenKind::Shl, TokenKind::Shr]) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_additive()?;
            left = Expr::BinOp(BinOpData {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_multiplicative()?;
        while self.at_any(&[TokenKind::Plus, TokenKind::Minus]) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp(BinOpData {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_power()?;
        while self.at_any(&[TokenKind::Star, TokenKind::Slash, TokenKind::Percent]) {
            let op = self.cur().value.clone();
            self.pos += 1;
            let right = self.parse_power()?;
            left = Expr::BinOp(BinOpData {
                op,
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_cast()?;
        if self.at(TokenKind::StarStar) {
            self.pos += 1;
            let right = self.parse_cast()?;
            return Ok(Expr::BinOp(BinOpData {
                op: "**".to_string(),
                left: Box::new(left),
                right: Box::new(right),
            }));
        }
        Ok(left)
    }

    fn parse_cast(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_unary()?;
        if self.at(TokenKind::As) {
            self.eat(TokenKind::As)?;
            let target = self.parse_type()?;
            return Ok(Expr::CastExpr(CastExprData {
                expr: Box::new(left),
                target_type: target,
            }));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.at(TokenKind::Minus) {
            self.eat(TokenKind::Minus)?;
            let operand = self.parse_unary()?;
            return Ok(Expr::UnaryOp(UnaryOpData {
                op: "-".to_string(),
                operand: Box::new(operand),
            }));
        }
        if self.at(TokenKind::Bang) {
            self.eat(TokenKind::Bang)?;
            let operand = self.parse_unary()?;
            return Ok(Expr::UnaryOp(UnaryOpData {
                op: "!".to_string(),
                operand: Box::new(operand),
            }));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_primary()?;
        loop {
            if self.at(TokenKind::Dot) {
                self.eat(TokenKind::Dot)?;
                let field = self.eat(TokenKind::Ident)?.value;
                if self.at(TokenKind::LParen) {
                    self.eat(TokenKind::LParen)?;
                    let args = self.parse_args()?;
                    self.eat(TokenKind::RParen)?;
                    left = Expr::Call(CallData {
                        func: Box::new(Expr::FieldAccess(FieldAccessData {
                            obj: Box::new(left),
                            field,
                        })),
                        args,
                    });
                } else {
                    left = Expr::FieldAccess(FieldAccessData {
                        obj: Box::new(left),
                        field,
                    });
                }
            } else if self.at(TokenKind::LBracket) {
                self.eat(TokenKind::LBracket)?;
                let index = self.parse_expr()?;
                self.eat(TokenKind::RBracket)?;
                left = Expr::IndexAccess(IndexAccessData {
                    obj: Box::new(left),
                    index: Box::new(index),
                });
            } else if self.at(TokenKind::LParen) {
                // Only allow call syntax if left is an Ident
                if let Expr::Ident(_) = &left {
                    self.eat(TokenKind::LParen)?;
                    let args = self.parse_args()?;
                    self.eat(TokenKind::RParen)?;
                    left = Expr::Call(CallData {
                        func: Box::new(left),
                        args,
                    });
                } else {
                    break;
                }
            } else if self.at(TokenKind::Question) {
                self.eat(TokenKind::Question)?;
                left = Expr::PropagateExpr(Box::new(left));
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        while !self.at(TokenKind::RParen) {
            args.push(self.parse_expr()?);
            if !self.at(TokenKind::RParen) {
                self.eat(TokenKind::Comma)?;
            }
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let tok = self.cur().clone();

        match tok.kind {
            TokenKind::IntLit => {
                self.pos += 1;
                let v: i64 = tok.value.parse().unwrap();
                Ok(Expr::IntLit(v))
            }
            TokenKind::FloatLit => {
                self.pos += 1;
                let v: f64 = tok.value.parse().unwrap();
                Ok(Expr::FloatLit(v))
            }
            TokenKind::StringLit => {
                self.pos += 1;
                Ok(Expr::StringLit(tok.value))
            }
            TokenKind::FStringLit => {
                self.pos += 1;
                Ok(Expr::FStringLit(tok.value))
            }
            TokenKind::True => {
                self.pos += 1;
                Ok(Expr::BoolLit(true))
            }
            TokenKind::False => {
                self.pos += 1;
                Ok(Expr::BoolLit(false))
            }
            TokenKind::None => {
                self.pos += 1;
                Ok(Expr::NoneLit)
            }
            TokenKind::Ok => {
                self.pos += 1;
                self.eat(TokenKind::LParen)?;
                let val = self.parse_expr()?;
                self.eat(TokenKind::RParen)?;
                Ok(Expr::OkExpr(Box::new(val)))
            }
            TokenKind::Err => {
                self.pos += 1;
                self.eat(TokenKind::LParen)?;
                let val = self.parse_expr()?;
                self.eat(TokenKind::RParen)?;
                Ok(Expr::ErrExpr(Box::new(val)))
            }
            TokenKind::Some => {
                self.pos += 1;
                self.eat(TokenKind::LParen)?;
                let val = self.parse_expr()?;
                self.eat(TokenKind::RParen)?;
                Ok(Expr::SomeExpr(Box::new(val)))
            }
            TokenKind::Fn => self.parse_closure(),
            TokenKind::If => self.parse_if_expr(),
            TokenKind::Match => self.parse_match_expr_or_stmt(),
            TokenKind::Ident => {
                let name = tok.value;
                self.pos += 1;
                // Check for struct construction: Name { field: val }
                if self.at(TokenKind::LBrace) && name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return self.parse_struct_construct(&name);
                }
                Ok(Expr::Ident(name))
            }
            TokenKind::LBracket => self.parse_array_or_map(),
            TokenKind::LBrace => self.parse_map_lit(),
            TokenKind::LParen => {
                self.eat(TokenKind::LParen)?;
                let expr = self.parse_expr()?;
                self.eat(TokenKind::RParen)?;
                Ok(expr)
            }
            _ => Err(ParseError {
                msg: format!("unexpected token {:?} ({:?})", tok.kind, tok.value),
                line: tok.line,
                col: tok.col,
            }),
        }
    }

    fn parse_closure(&mut self) -> Result<Expr, ParseError> {
        self.eat(TokenKind::Fn)?;
        self.eat(TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.eat(TokenKind::RParen)?;
        let return_type = if self.at(TokenKind::Arrow) {
            self.eat(TokenKind::Arrow)?;
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(Expr::ClosureLit(Box::new(ClosureLitData {
            params,
            return_type,
            body: ClosureBody::Block(body),
        })))
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        self.eat(TokenKind::If)?;
        let condition = self.parse_expr()?;
        self.eat(TokenKind::LBrace)?;
        let then_expr = self.parse_expr()?;
        self.maybe(TokenKind::Semicolon);
        self.eat(TokenKind::RBrace)?;
        self.eat(TokenKind::Else)?;
        self.eat(TokenKind::LBrace)?;
        let else_expr = self.parse_expr()?;
        self.maybe(TokenKind::Semicolon);
        self.eat(TokenKind::RBrace)?;
        Ok(Expr::IfExpr(IfExprData {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }))
    }

    fn parse_struct_construct(&mut self, name: &str) -> Result<Expr, ParseError> {
        self.eat(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let fname = self.eat(TokenKind::Ident)?.value;
            self.eat(TokenKind::Colon)?;
            let val = self.parse_expr()?;
            fields.push((fname, val));
            self.maybe(TokenKind::Comma);
        }
        self.eat(TokenKind::RBrace)?;
        Ok(Expr::StructConstruct {
            name: name.to_string(),
            fields,
        })
    }

    fn parse_array_or_map(&mut self) -> Result<Expr, ParseError> {
        self.eat(TokenKind::LBracket)?;
        if self.at(TokenKind::RBracket) {
            self.eat(TokenKind::RBracket)?;
            return Ok(Expr::ArrayLit(Vec::new()));
        }
        let first = self.parse_expr()?;
        // [val; count]
        if self.at(TokenKind::Semicolon) {
            self.eat(TokenKind::Semicolon)?;
            let count = self.parse_expr()?;
            self.eat(TokenKind::RBracket)?;
            return Ok(Expr::ArrayFill {
                value: Box::new(first),
                count: Box::new(count),
            });
        }
        // [a, b, c]
        let mut elems = vec![first];
        while self.at(TokenKind::Comma) {
            self.eat(TokenKind::Comma)?;
            if self.at(TokenKind::RBracket) {
                break;
            }
            elems.push(self.parse_expr()?);
        }
        self.eat(TokenKind::RBracket)?;
        Ok(Expr::ArrayLit(elems))
    }

    fn parse_map_lit(&mut self) -> Result<Expr, ParseError> {
        self.eat(TokenKind::LBrace)?;
        let mut pairs = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let key = self.parse_expr()?;
            self.eat(TokenKind::Colon)?;
            let val = self.parse_expr()?;
            pairs.push((key, val));
            self.maybe(TokenKind::Comma);
        }
        self.eat(TokenKind::RBrace)?;
        Ok(Expr::MapLit(pairs))
    }
}

/// Parse a token list into an AST.
pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    Parser::new(tokens).parse()
}
