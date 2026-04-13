"""Mog language interpreter — lexer, parser, evaluator.

Provides concrete execution today and a differentiable execution core via
`egdc.mog_differentiable` for the benchmark-safe numeric subset.
"""

from egdc.mog.lang.interpreter import interpret, InterpreterResult
