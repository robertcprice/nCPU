"""Mog lexer — tokenizes source code into a token stream."""
from __future__ import annotations
from .tokens import Token, TT, KEYWORDS


class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"Lex error at L{line}:{col}: {msg}")
        self.line, self.col = line, col


class Lexer:
    """Tokenizes Mog source code."""

    def __init__(self, source: str):
        self.src = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def _ch(self) -> str:
        return self.src[self.pos] if self.pos < len(self.src) else "\0"

    def _peek(self, offset: int = 1) -> str:
        p = self.pos + offset
        return self.src[p] if p < len(self.src) else "\0"

    def _advance(self) -> str:
        ch = self._ch()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _emit(self, tt: TT, value: str, line: int, col: int):
        self.tokens.append(Token(tt, value, line, col))

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.src):
            ch = self._ch()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "/" and self._peek() == "/":
                # Line comment
                while self.pos < len(self.src) and self._ch() != "\n":
                    self._advance()
            elif ch == "/" and self._peek() == "*":
                # Block comment
                self._advance()  # /
                self._advance()  # *
                depth = 1
                while self.pos < len(self.src) and depth > 0:
                    if self._ch() == "*" and self._peek() == "/":
                        self._advance()
                        self._advance()
                        depth -= 1
                    elif self._ch() == "/" and self._peek() == "*":
                        self._advance()
                        self._advance()
                        depth += 1
                    else:
                        self._advance()
            else:
                break

    def _read_string(self, is_fstring: bool = False) -> str:
        """Read a string literal (opening " already consumed)."""
        parts = []
        while self.pos < len(self.src) and self._ch() != '"':
            if self._ch() == "\\":
                self._advance()
                esc = self._advance()
                if esc == "n":
                    parts.append("\n")
                elif esc == "t":
                    parts.append("\t")
                elif esc == "\\":
                    parts.append("\\")
                elif esc == '"':
                    parts.append('"')
                elif esc == "{":
                    parts.append("{")
                else:
                    parts.append("\\" + esc)
            else:
                parts.append(self._advance())
        if self.pos < len(self.src):
            self._advance()  # closing "
        return "".join(parts)

    def _read_number(self) -> tuple[TT, str]:
        """Read an integer or float literal."""
        start = self.pos
        is_float = False
        while self.pos < len(self.src) and (self._ch().isdigit() or self._ch() == "_"):
            self._advance()
        if self._ch() == "." and self._peek().isdigit():
            is_float = True
            self._advance()  # .
            while self.pos < len(self.src) and (self._ch().isdigit() or self._ch() == "_"):
                self._advance()
        if self._ch() in "eE":
            is_float = True
            self._advance()
            if self._ch() in "+-":
                self._advance()
            while self.pos < len(self.src) and self._ch().isdigit():
                self._advance()
        val = self.src[start:self.pos].replace("_", "")
        return (TT.FLOAT_LIT if is_float else TT.INT_LIT), val

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < len(self.src) and (self._ch().isalnum() or self._ch() == "_"):
            self._advance()
        return self.src[start:self.pos]

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source, return token list ending with EOF."""
        while self.pos < len(self.src):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.src):
                break

            line, col = self.line, self.col
            ch = self._ch()

            # F-strings
            if ch == "f" and self._peek() == '"':
                self._advance()  # f
                self._advance()  # "
                val = self._read_string(is_fstring=True)
                self._emit(TT.FSTRING_LIT, val, line, col)
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                word = self._read_ident()
                tt = KEYWORDS.get(word, TT.IDENT)
                self._emit(tt, word, line, col)
                continue

            # Numbers
            if ch.isdigit():
                tt, val = self._read_number()
                self._emit(tt, val, line, col)
                continue

            # Strings
            if ch == '"':
                self._advance()
                val = self._read_string()
                self._emit(TT.STRING_LIT, val, line, col)
                continue

            # Two-char operators
            nch = self._peek()
            two = ch + nch
            if two == ":=":
                self._advance(); self._advance()
                self._emit(TT.COLON_EQ, ":=", line, col); continue
            if two == "==":
                self._advance(); self._advance()
                self._emit(TT.EQ_EQ, "==", line, col); continue
            if two == "!=":
                self._advance(); self._advance()
                self._emit(TT.BANG_EQ, "!=", line, col); continue
            if two == "<=":
                self._advance(); self._advance()
                self._emit(TT.LT_EQ, "<=", line, col); continue
            if two == ">=":
                self._advance(); self._advance()
                self._emit(TT.GT_EQ, ">=", line, col); continue
            if two == "->":
                self._advance(); self._advance()
                self._emit(TT.ARROW, "->", line, col); continue
            if two == "=>":
                self._advance(); self._advance()
                self._emit(TT.FAT_ARROW, "=>", line, col); continue
            if two == "&&":
                self._advance(); self._advance()
                self._emit(TT.AMP_AMP, "&&", line, col); continue
            if two == "||":
                self._advance(); self._advance()
                self._emit(TT.PIPE_PIPE, "||", line, col); continue
            if two == "<<":
                self._advance(); self._advance()
                self._emit(TT.SHL, "<<", line, col); continue
            if two == ">>":
                self._advance(); self._advance()
                self._emit(TT.SHR, ">>", line, col); continue
            if two == "**":
                self._advance(); self._advance()
                self._emit(TT.STAR_STAR, "**", line, col); continue
            if two == "..":
                self._advance(); self._advance()
                self._emit(TT.DOT_DOT, "..", line, col); continue

            # Single-char
            self._advance()
            SINGLE = {
                "+": TT.PLUS, "-": TT.MINUS, "*": TT.STAR, "/": TT.SLASH,
                "%": TT.PERCENT, "<": TT.LT, ">": TT.GT,
                "&": TT.AMP, "|": TT.PIPE, "^": TT.CARET,
                "!": TT.BANG, "?": TT.QUESTION,
                "(": TT.LPAREN, ")": TT.RPAREN,
                "{": TT.LBRACE, "}": TT.RBRACE,
                "[": TT.LBRACKET, "]": TT.RBRACKET,
                ",": TT.COMMA, ";": TT.SEMICOLON,
                ":": TT.COLON, ".": TT.DOT,
                "=": TT.EQ, "_": TT.UNDERSCORE,
            }
            if ch in SINGLE:
                self._emit(SINGLE[ch], ch, line, col)
            else:
                raise LexError(f"unexpected character {ch!r}", line, col)

        self._emit(TT.EOF, "", self.line, self.col)
        return self.tokens


def lex(source: str) -> list[Token]:
    """Tokenize Mog source code."""
    return Lexer(source).tokenize()
