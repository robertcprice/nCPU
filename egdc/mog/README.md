## EGDC Mog

Canonical Mog-language/compiler workspace for EGDC.

- `lang/` — Mog lexer/parser/interpreter
- `routing/` — router/orchestrator/pathway-memory logic
- `solvers/` — differentiable, search, and direct synthesis solvers
- `tools/` — REPL and command-line helper surfaces
- `training/` — training/eval dataset and benchmark helpers

The flat legacy `egdc/mog_*.py` modules now forward into this package.
