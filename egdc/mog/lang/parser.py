"""Mog recursive descent parser."""
from __future__ import annotations
from .tokens import Token, TT
from . import ast_nodes as ast


class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"Parse error at L{token.line}:{token.col}: {msg}")
        self.token = token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def _peek(self, offset: int = 0) -> Token:
        p = self.pos + offset
        return self.tokens[p] if p < len(self.tokens) else self.tokens[-1]

    def _at(self, *types: TT) -> bool:
        return self._cur().type in types

    def _eat(self, tt: TT) -> Token:
        tok = self._cur()
        if tok.type != tt:
            raise ParseError(f"expected {tt.name}, got {tok.type.name} ({tok.value!r})", tok)
        self.pos += 1
        return tok

    def _maybe(self, tt: TT) -> Token | None:
        if self._at(tt):
            return self._eat(tt)
        return None

    def _skip_semis(self):
        while self._at(TT.SEMICOLON):
            self.pos += 1

    # --- Type annotations ---

    def _is_type_start(self) -> bool:
        return self._at(
            TT.INT_TYPE, TT.I8, TT.I16, TT.I32, TT.I64,
            TT.U8, TT.U16, TT.U32, TT.U64,
            TT.FLOAT_TYPE, TT.F16, TT.BF16, TT.F32, TT.F64,
            TT.BOOL_TYPE, TT.STRING_TYPE, TT.RESULT, TT.QUESTION,
            TT.LBRACKET, TT.IDENT,
        )

    def _parse_type(self) -> str:
        """Parse a type annotation, return as string."""
        if self._at(TT.QUESTION):
            self._eat(TT.QUESTION)
            inner = self._parse_type()
            return f"?{inner}"
        if self._at(TT.RESULT):
            self._eat(TT.RESULT)
            if self._at(TT.LT):
                self._eat(TT.LT)
                inner = self._parse_type()
                self._eat(TT.GT)
                return f"Result<{inner}>"
            return "Result"
        if self._at(TT.LBRACKET):
            self._eat(TT.LBRACKET)
            if self._at(TT.RBRACKET):
                # []T — array type
                self._eat(TT.RBRACKET)
                inner = self._parse_type()
                return f"[]{inner}"
            # [K]V — map type  or  [T] — array
            key = self._parse_type()
            self._eat(TT.RBRACKET)
            if self._is_type_start():
                val = self._parse_type()
                return f"[{key}]{val}"
            return f"[{key}]"
        # Simple type name
        tok = self._cur()
        type_toks = {
            TT.INT_TYPE, TT.I8, TT.I16, TT.I32, TT.I64,
            TT.U8, TT.U16, TT.U32, TT.U64,
            TT.FLOAT_TYPE, TT.F16, TT.BF16, TT.F32, TT.F64,
            TT.BOOL_TYPE, TT.STRING_TYPE, TT.IDENT,
        }
        if tok.type in type_toks:
            self.pos += 1
            return tok.value
        raise ParseError(f"expected type, got {tok.type.name}", tok)

    # --- Top level ---

    def parse(self) -> ast.Program:
        decls = []
        while not self._at(TT.EOF):
            self._skip_semis()
            if self._at(TT.EOF):
                break
            decls.append(self._parse_declaration())
            self._skip_semis()
        return ast.Program(decls)

    def _parse_declaration(self) -> ast.Any:
        if self._at(TT.FN) or (self._at(TT.PUB) and self._peek(1).type == TT.FN):
            return self._parse_fn_decl()
        if self._at(TT.STRUCT):
            return self._parse_struct_decl()
        if self._at(TT.TYPE):
            return self._parse_type_alias()
        if self._at(TT.REQUIRES):
            return self._parse_requires()
        raise ParseError(f"expected declaration, got {self._cur().type.name}", self._cur())

    def _parse_fn_decl(self) -> ast.FnDecl:
        is_pub = bool(self._maybe(TT.PUB))
        self._eat(TT.FN)
        name = self._eat(TT.IDENT).value
        self._eat(TT.LPAREN)
        params = self._parse_params()
        self._eat(TT.RPAREN)
        ret_type = None
        if self._at(TT.ARROW):
            self._eat(TT.ARROW)
            ret_type = self._parse_type()
        body = self._parse_block()
        return ast.FnDecl(name, params, ret_type, body, is_pub)

    def _parse_params(self) -> list[ast.Param]:
        params = []
        while not self._at(TT.RPAREN):
            name = self._eat(TT.IDENT).value
            type_ann = None
            default = None
            if self._at(TT.COLON):
                self._eat(TT.COLON)
                type_ann = self._parse_type()
            if self._at(TT.EQ):
                self._eat(TT.EQ)
                default = self._parse_expr()
            params.append(ast.Param(name, type_ann, default))
            if not self._at(TT.RPAREN):
                self._eat(TT.COMMA)
        return params

    def _parse_struct_decl(self) -> ast.StructDecl:
        self._eat(TT.STRUCT)
        name = self._eat(TT.IDENT).value
        self._eat(TT.LBRACE)
        fields = []
        while not self._at(TT.RBRACE):
            fname = self._eat(TT.IDENT).value
            self._eat(TT.COLON)
            ftype = self._parse_type()
            fields.append((fname, ftype))
            self._maybe(TT.COMMA)
        self._eat(TT.RBRACE)
        return ast.StructDecl(name, fields)

    def _parse_type_alias(self) -> ast.TypeAlias:
        self._eat(TT.TYPE)
        name = self._eat(TT.IDENT).value
        self._eat(TT.EQ)
        target = self._parse_type()
        self._eat(TT.SEMICOLON)
        return ast.TypeAlias(name, target)

    def _parse_requires(self) -> ast.RequiresDecl:
        self._eat(TT.REQUIRES)
        caps = [self._eat(TT.IDENT).value]
        while self._at(TT.COMMA):
            self._eat(TT.COMMA)
            caps.append(self._eat(TT.IDENT).value)
        self._eat(TT.SEMICOLON)
        return ast.RequiresDecl(caps)

    # --- Blocks and statements ---

    def _parse_block(self) -> ast.Block:
        self._eat(TT.LBRACE)
        stmts = []
        while not self._at(TT.RBRACE) and not self._at(TT.EOF):
            self._skip_semis()
            if self._at(TT.RBRACE):
                break
            stmts.append(self._parse_stmt())
            self._skip_semis()
        self._eat(TT.RBRACE)
        return ast.Block(stmts)

    def _parse_stmt(self) -> ast.Any:
        if self._at(TT.RETURN):
            return self._parse_return()
        if self._at(TT.IF):
            return self._parse_if()
        if self._at(TT.WHILE):
            return self._parse_while()
        if self._at(TT.FOR):
            return self._parse_for()
        if self._at(TT.MATCH):
            return self._parse_match_stmt()
        if self._at(TT.BREAK):
            self._eat(TT.BREAK)
            self._maybe(TT.SEMICOLON)
            return ast.BreakStmt()
        if self._at(TT.CONTINUE):
            self._eat(TT.CONTINUE)
            self._maybe(TT.SEMICOLON)
            return ast.ContinueStmt()
        if self._at(TT.FN):
            return self._parse_fn_decl()
        if self._at(TT.STRUCT):
            return self._parse_struct_decl()

        # Variable declaration or assignment or expression statement
        # Check for:  IDENT := ...  or  IDENT : type = ...
        if self._at(TT.IDENT):
            # Look ahead for := or : type =
            if self._peek(1).type == TT.COLON_EQ:
                return self._parse_var_decl_walrus()
            if self._peek(1).type == TT.COLON and self._peek(2).type != TT.EQ:
                # Could be typed decl: name: type = expr
                return self._parse_var_decl_typed()

        # Expression (which might be assignment: expr = expr)
        expr = self._parse_expr()
        if self._at(TT.EQ):
            self._eat(TT.EQ)
            value = self._parse_expr()
            self._maybe(TT.SEMICOLON)
            return ast.Assignment(expr, value)
        if self._at(TT.COLON_EQ):
            # field := value (struct field reassignment in some Mog dialects)
            self._eat(TT.COLON_EQ)
            value = self._parse_expr()
            self._maybe(TT.SEMICOLON)
            return ast.Assignment(expr, value)
        self._maybe(TT.SEMICOLON)
        return ast.ExprStmt(expr)

    def _parse_var_decl_walrus(self) -> ast.VarDecl:
        name = self._eat(TT.IDENT).value
        self._eat(TT.COLON_EQ)
        value = self._parse_expr()
        self._maybe(TT.SEMICOLON)
        return ast.VarDecl(name, None, value)

    def _parse_var_decl_typed(self) -> ast.VarDecl:
        name = self._eat(TT.IDENT).value
        self._eat(TT.COLON)
        type_ann = self._parse_type()
        self._eat(TT.EQ)
        value = self._parse_expr()
        self._maybe(TT.SEMICOLON)
        return ast.VarDecl(name, type_ann, value)

    def _parse_return(self) -> ast.ReturnStmt:
        self._eat(TT.RETURN)
        if self._at(TT.SEMICOLON) or self._at(TT.RBRACE):
            self._maybe(TT.SEMICOLON)
            return ast.ReturnStmt(None)
        val = self._parse_expr()
        self._maybe(TT.SEMICOLON)
        return ast.ReturnStmt(val)

    def _parse_if(self) -> ast.IfStmt:
        self._eat(TT.IF)
        cond = self._parse_expr()
        then = self._parse_block()
        else_b = None
        if self._at(TT.ELSE):
            self._eat(TT.ELSE)
            if self._at(TT.IF):
                else_b = self._parse_if()
            else:
                else_b = self._parse_block()
        return ast.IfStmt(cond, then, else_b)

    def _parse_while(self) -> ast.WhileStmt:
        self._eat(TT.WHILE)
        cond = self._parse_expr()
        body = self._parse_block()
        return ast.WhileStmt(cond, body)

    def _parse_for(self) -> ast.Any:
        self._eat(TT.FOR)
        name = self._eat(TT.IDENT).value

        # for name := start to end { }
        if self._at(TT.COLON_EQ):
            self._eat(TT.COLON_EQ)
            start = self._parse_expr()
            self._eat(TT.TO)
            end = self._parse_expr()
            body = self._parse_block()
            return ast.ForToStmt(name, start, end, body)

        # for name in expr { }  or  for name, name2 in expr { }
        index_name = None
        if self._at(TT.COMMA):
            self._eat(TT.COMMA)
            index_name = name
            name = self._eat(TT.IDENT).value

        self._eat(TT.IN)

        # Check for range pattern: expr..expr
        start_expr = self._parse_expr()
        if self._at(TT.DOT_DOT):
            self._eat(TT.DOT_DOT)
            end_expr = self._parse_expr()
            body = self._parse_block()
            return ast.ForInRangeStmt(name, start_expr, end_expr, body)

        iterable = start_expr

        # Check if iterable is a range expr (start..end parsed as BinOp)
        if isinstance(iterable, ast.RangeExpr):
            body = self._parse_block()
            return ast.ForInRangeStmt(name, iterable.start, iterable.end, body)

        body = self._parse_block()
        return ast.ForInStmt(name, index_name, iterable, body)

    def _parse_match_stmt(self) -> ast.MatchStmt | ast.ExprStmt:
        match_expr = self._parse_match_expr_or_stmt()
        if isinstance(match_expr, ast.MatchExpr):
            # Could be expression-level match used as statement
            self._maybe(TT.SEMICOLON)
            return ast.ExprStmt(match_expr)
        return match_expr

    def _parse_match_expr_or_stmt(self) -> ast.MatchExpr:
        self._eat(TT.MATCH)
        subject = self._parse_expr()
        self._eat(TT.LBRACE)
        arms = []
        while not self._at(TT.RBRACE) and not self._at(TT.EOF):
            pattern = self._parse_pattern()
            self._eat(TT.FAT_ARROW)
            if self._at(TT.LBRACE):
                body = self._parse_block()
            else:
                body = self._parse_expr()
            arms.append(ast.MatchArm(pattern, body))
            self._maybe(TT.COMMA)
            self._maybe(TT.SEMICOLON)
        self._eat(TT.RBRACE)
        return ast.MatchExpr(subject, arms)

    def _parse_pattern(self) -> ast.Any:
        if self._at(TT.UNDERSCORE):
            self._eat(TT.UNDERSCORE)
            return ast.WildcardPattern()
        if self._at(TT.OK):
            self._eat(TT.OK)
            self._eat(TT.LPAREN)
            name = self._eat(TT.IDENT).value
            self._eat(TT.RPAREN)
            return ast.OkPattern(name)
        if self._at(TT.ERR):
            self._eat(TT.ERR)
            self._eat(TT.LPAREN)
            name = self._eat(TT.IDENT).value
            self._eat(TT.RPAREN)
            return ast.ErrPattern(name)
        if self._at(TT.SOME):
            self._eat(TT.SOME)
            self._eat(TT.LPAREN)
            name = self._eat(TT.IDENT).value
            self._eat(TT.RPAREN)
            return ast.SomePattern(name)
        if self._at(TT.NONE):
            self._eat(TT.NONE)
            return ast.NonePattern()
        if self._at(TT.INT_LIT):
            tok = self._eat(TT.INT_LIT)
            return ast.LitPattern(ast.IntLit(int(tok.value)))
        if self._at(TT.FLOAT_LIT):
            tok = self._eat(TT.FLOAT_LIT)
            return ast.LitPattern(ast.FloatLit(float(tok.value)))
        if self._at(TT.STRING_LIT):
            tok = self._eat(TT.STRING_LIT)
            return ast.LitPattern(ast.StringLit(tok.value))
        if self._at(TT.TRUE):
            self._eat(TT.TRUE)
            return ast.LitPattern(ast.BoolLit(True))
        if self._at(TT.FALSE):
            self._eat(TT.FALSE)
            return ast.LitPattern(ast.BoolLit(False))
        if self._at(TT.MINUS):
            self._eat(TT.MINUS)
            tok = self._eat(TT.INT_LIT)
            return ast.LitPattern(ast.IntLit(-int(tok.value)))
        # Identifier pattern
        if self._at(TT.IDENT):
            tok = self._eat(TT.IDENT)
            return ast.IdentPattern(tok.value)
        raise ParseError(f"expected pattern, got {self._cur().type.name}", self._cur())

    # --- Expressions ---

    def _parse_expr(self) -> ast.Any:
        return self._parse_or()

    def _parse_or(self) -> ast.Any:
        left = self._parse_and()
        while self._at(TT.PIPE_PIPE, TT.OR):
            op = self._cur().value
            self.pos += 1
            right = self._parse_and()
            left = ast.BinOp(op if op != "or" else "||", left, right)
        return left

    def _parse_and(self) -> ast.Any:
        left = self._parse_comparison()
        while self._at(TT.AMP_AMP, TT.AND):
            op = self._cur().value
            self.pos += 1
            right = self._parse_comparison()
            left = ast.BinOp(op if op != "and" else "&&", left, right)
        return left

    def _parse_comparison(self) -> ast.Any:
        left = self._parse_bitwise()
        if self._at(TT.EQ_EQ, TT.BANG_EQ, TT.LT, TT.GT, TT.LT_EQ, TT.GT_EQ):
            op = self._cur().value
            self.pos += 1
            right = self._parse_bitwise()
            left = ast.BinOp(op, left, right)
        return left

    def _parse_bitwise(self) -> ast.Any:
        left = self._parse_shift()
        while self._at(TT.AMP, TT.PIPE, TT.CARET):
            op = self._cur().value
            self.pos += 1
            right = self._parse_shift()
            left = ast.BinOp(op, left, right)
        return left

    def _parse_shift(self) -> ast.Any:
        left = self._parse_additive()
        while self._at(TT.SHL, TT.SHR):
            op = self._cur().value
            self.pos += 1
            right = self._parse_additive()
            left = ast.BinOp(op, left, right)
        return left

    def _parse_additive(self) -> ast.Any:
        left = self._parse_multiplicative()
        while self._at(TT.PLUS, TT.MINUS):
            op = self._cur().value
            self.pos += 1
            right = self._parse_multiplicative()
            left = ast.BinOp(op, left, right)
        return left

    def _parse_multiplicative(self) -> ast.Any:
        left = self._parse_power()
        while self._at(TT.STAR, TT.SLASH, TT.PERCENT):
            op = self._cur().value
            self.pos += 1
            right = self._parse_power()
            left = ast.BinOp(op, left, right)
        return left

    def _parse_power(self) -> ast.Any:
        left = self._parse_cast()
        if self._at(TT.STAR_STAR):
            self.pos += 1
            right = self._parse_cast()
            left = ast.BinOp("**", left, right)
        return left

    def _parse_cast(self) -> ast.Any:
        left = self._parse_unary()
        if self._at(TT.AS):
            self._eat(TT.AS)
            target = self._parse_type()
            left = ast.CastExpr(left, target)
        return left

    def _parse_unary(self) -> ast.Any:
        if self._at(TT.MINUS):
            self._eat(TT.MINUS)
            operand = self._parse_unary()
            return ast.UnaryOp("-", operand)
        if self._at(TT.BANG):
            self._eat(TT.BANG)
            operand = self._parse_unary()
            return ast.UnaryOp("!", operand)
        return self._parse_postfix()

    def _parse_postfix(self) -> ast.Any:
        left = self._parse_primary()
        while True:
            if self._at(TT.DOT):
                self._eat(TT.DOT)
                field = self._eat(TT.IDENT).value
                # Method call: obj.method(args)
                if self._at(TT.LPAREN):
                    self._eat(TT.LPAREN)
                    args = self._parse_args()
                    self._eat(TT.RPAREN)
                    left = ast.Call(ast.FieldAccess(left, field), args)
                else:
                    left = ast.FieldAccess(left, field)
            elif self._at(TT.LBRACKET):
                self._eat(TT.LBRACKET)
                index = self._parse_expr()
                self._eat(TT.RBRACKET)
                left = ast.IndexAccess(left, index)
            elif self._at(TT.LPAREN) and isinstance(left, ast.Ident):
                self._eat(TT.LPAREN)
                args = self._parse_args()
                self._eat(TT.RPAREN)
                left = ast.Call(left, args)
            elif self._at(TT.QUESTION):
                self._eat(TT.QUESTION)
                left = ast.PropagateExpr(left)
            else:
                break
        return left

    def _parse_args(self) -> list[ast.Any]:
        args = []
        while not self._at(TT.RPAREN):
            args.append(self._parse_expr())
            if not self._at(TT.RPAREN):
                self._eat(TT.COMMA)
        return args

    def _parse_primary(self) -> ast.Any:
        tok = self._cur()

        if tok.type == TT.INT_LIT:
            self.pos += 1
            return ast.IntLit(int(tok.value))

        if tok.type == TT.FLOAT_LIT:
            self.pos += 1
            return ast.FloatLit(float(tok.value))

        if tok.type == TT.STRING_LIT:
            self.pos += 1
            return ast.StringLit(tok.value)

        if tok.type == TT.FSTRING_LIT:
            self.pos += 1
            return ast.FStringLit(tok.value)

        if tok.type == TT.TRUE:
            self.pos += 1
            return ast.BoolLit(True)

        if tok.type == TT.FALSE:
            self.pos += 1
            return ast.BoolLit(False)

        if tok.type == TT.NONE:
            self.pos += 1
            return ast.NoneLit()

        if tok.type == TT.OK:
            self.pos += 1
            self._eat(TT.LPAREN)
            val = self._parse_expr()
            self._eat(TT.RPAREN)
            return ast.OkExpr(val)

        if tok.type == TT.ERR:
            self.pos += 1
            self._eat(TT.LPAREN)
            val = self._parse_expr()
            self._eat(TT.RPAREN)
            return ast.ErrExpr(val)

        if tok.type == TT.SOME:
            self.pos += 1
            self._eat(TT.LPAREN)
            val = self._parse_expr()
            self._eat(TT.RPAREN)
            return ast.SomeExpr(val)

        if tok.type == TT.FN:
            return self._parse_closure()

        if tok.type == TT.IF:
            return self._parse_if_expr()

        if tok.type == TT.MATCH:
            return self._parse_match_expr_or_stmt()

        if tok.type == TT.IDENT:
            name = tok.value
            self.pos += 1
            # Check for struct construction: Name { field: val }
            if self._at(TT.LBRACE) and name[0].isupper():
                return self._parse_struct_construct(name)
            return ast.Ident(name)

        if tok.type == TT.LBRACKET:
            return self._parse_array_or_map()

        if tok.type == TT.LBRACE:
            return self._parse_map_lit()

        if tok.type == TT.LPAREN:
            self._eat(TT.LPAREN)
            expr = self._parse_expr()
            self._eat(TT.RPAREN)
            return expr

        raise ParseError(f"unexpected token {tok.type.name} ({tok.value!r})", tok)

    def _parse_closure(self) -> ast.ClosureLit:
        self._eat(TT.FN)
        self._eat(TT.LPAREN)
        params = self._parse_params()
        self._eat(TT.RPAREN)
        ret_type = None
        if self._at(TT.ARROW):
            self._eat(TT.ARROW)
            ret_type = self._parse_type()
        body = self._parse_block()
        return ast.ClosureLit(params, ret_type, body)

    def _parse_if_expr(self) -> ast.IfExpr:
        self._eat(TT.IF)
        cond = self._parse_expr()
        self._eat(TT.LBRACE)
        then = self._parse_expr()
        self._maybe(TT.SEMICOLON)
        self._eat(TT.RBRACE)
        self._eat(TT.ELSE)
        self._eat(TT.LBRACE)
        else_e = self._parse_expr()
        self._maybe(TT.SEMICOLON)
        self._eat(TT.RBRACE)
        return ast.IfExpr(cond, then, else_e)

    def _parse_struct_construct(self, name: str) -> ast.StructConstruct:
        self._eat(TT.LBRACE)
        fields = []
        while not self._at(TT.RBRACE):
            fname = self._eat(TT.IDENT).value
            self._eat(TT.COLON)
            val = self._parse_expr()
            fields.append((fname, val))
            self._maybe(TT.COMMA)
        self._eat(TT.RBRACE)
        return ast.StructConstruct(name, fields)

    def _parse_array_or_map(self) -> ast.Any:
        self._eat(TT.LBRACKET)
        if self._at(TT.RBRACKET):
            self._eat(TT.RBRACKET)
            return ast.ArrayLit([])
        first = self._parse_expr()
        # [val; count]
        if self._at(TT.SEMICOLON):
            self._eat(TT.SEMICOLON)
            count = self._parse_expr()
            self._eat(TT.RBRACKET)
            return ast.ArrayFill(first, count)
        # [a, b, c]
        elems = [first]
        while self._at(TT.COMMA):
            self._eat(TT.COMMA)
            if self._at(TT.RBRACKET):
                break
            elems.append(self._parse_expr())
        self._eat(TT.RBRACKET)
        return ast.ArrayLit(elems)

    def _parse_map_lit(self) -> ast.MapLit:
        self._eat(TT.LBRACE)
        pairs = []
        while not self._at(TT.RBRACE):
            key = self._parse_expr()
            self._eat(TT.COLON)
            val = self._parse_expr()
            pairs.append((key, val))
            self._maybe(TT.COMMA)
        self._eat(TT.RBRACE)
        return ast.MapLit(pairs)


def parse(tokens: list[Token]) -> ast.Program:
    """Parse a token list into an AST."""
    return Parser(tokens).parse()
