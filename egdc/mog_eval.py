"""Evaluation utilities for generated Mog code.

Performs static analysis on Mog programs without requiring a compiler.
Checks syntax validity, type annotations, capability declarations,
and structural correctness.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mog language constants
# ---------------------------------------------------------------------------

MOG_KEYWORDS = {
    "fn", "pub", "async", "await", "return", "if", "else", "for", "while",
    "match", "struct", "requires", "optional", "import", "true", "false",
    "nil", "ok", "err", "to", "as", "let", "mut",
}

MOG_TYPES = {
    "int", "i32", "u32", "u64", "float", "f32", "f16", "bf16",
    "bool", "string", "Result",
}

MOG_CAPABILITY_KEYWORDS = {"requires", "optional", "import"}

# Patterns
_FN_PATTERN = re.compile(
    r"(pub\s+)?(async\s+)?fn\s+(\w+)\s*\(([^)]*)\)\s*(->\s*\S+)?\s*\{"
)
_STRUCT_PATTERN = re.compile(r"struct\s+(\w+)\s*\{")
_CAPABILITY_PATTERN = re.compile(r"^(requires|optional|import)\s+(\w+)\s*;", re.MULTILINE)
_BINDING_PATTERN = re.compile(r"(\w+)\s*:=\s*")
_REASSIGN_PATTERN = re.compile(r"(\w+)\s*=\s*(?!=)")
_RETURN_PATTERN = re.compile(r"\breturn\b")
_TYPE_ANNOTATION = re.compile(r":\s*(int|i32|u32|u64|float|f32|f16|bf16|bool|string|Result<[^>]+>|\[\]\w+|\[\w+\]\w+)")
_SEMICOLON_STMT = re.compile(r"[^{};/\n]+;")
_FOR_PATTERN = re.compile(r"\bfor\s+\w+\s*:=\s*.+\bto\b")
_WHILE_PATTERN = re.compile(r"\bwhile\b\s+.+\{")
_IF_PATTERN = re.compile(r"\bif\b\s+.+\{")
_MATCH_PATTERN = re.compile(r"\bmatch\b\s+.+\{")
_ARROW_PATTERN = re.compile(r"=>")
_RESULT_PATTERN = re.compile(r"\b(ok|err)\s*\(")


@dataclass
class MogEvalResult:
    """Result of evaluating a single Mog program."""
    # Individual checks (0.0 to 1.0)
    bracket_balance: float = 0.0
    semicolon_usage: float = 0.0
    keyword_validity: float = 0.0
    type_completeness: float = 0.0
    capability_correctness: float = 0.0
    structural_correctness: float = 0.0

    # Derived
    syntactic_validity: float = 0.0
    overall_score: float = 0.0

    # Details
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Extracted info
    functions: List[str] = field(default_factory=list)
    structs: List[str] = field(default_factory=list)
    capabilities: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Checker functions
# ---------------------------------------------------------------------------

def _check_bracket_balance(code: str) -> Tuple[float, List[str]]:
    """Check that brackets, braces, and parens are balanced."""
    errors = []
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    closers = set(pairs.keys())

    in_string = False
    in_comment = False
    prev = ""

    for i, ch in enumerate(code):
        # Track string literals
        if ch == '"' and not in_comment and prev != "\\":
            in_string = not in_string
            prev = ch
            continue
        # Track line comments
        if ch == "/" and i + 1 < len(code) and code[i + 1] == "/" and not in_string:
            in_comment = True
            prev = ch
            continue
        if ch == "\n":
            in_comment = False
            prev = ch
            continue

        if in_string or in_comment:
            prev = ch
            continue

        if ch in openers:
            stack.append(ch)
        elif ch in closers:
            if not stack:
                errors.append(f"unmatched '{ch}' at position {i}")
            elif stack[-1] != pairs[ch]:
                errors.append(f"mismatched '{ch}' at position {i}, expected closing for '{stack[-1]}'")
                stack.pop()
            else:
                stack.pop()
        prev = ch

    for ch in stack:
        errors.append(f"unclosed '{ch}'")

    if not errors:
        return 1.0, []
    # Partial credit based on how few errors
    score = max(0.0, 1.0 - len(errors) * 0.25)
    return score, errors


def _check_semicolons(code: str) -> Tuple[float, List[str]]:
    """Check semicolon usage in statements."""
    errors = []
    lines = code.split("\n")
    stmt_lines = 0
    semicolon_lines = 0

    for line in lines:
        stripped = line.strip()
        # Skip empty, comments, braces-only, control flow headers
        if not stripped or stripped.startswith("//"):
            continue
        if stripped in ("{", "}", "} else {", "};"):
            continue
        if re.match(r"^(fn|pub fn|async fn|pub async fn|struct|if|else|for|while|match|requires|optional|import)\b", stripped):
            # Capability declarations need semicolons
            if re.match(r"^(requires|optional|import)\s+\w+", stripped):
                stmt_lines += 1
                if stripped.endswith(";"):
                    semicolon_lines += 1
                else:
                    errors.append(f"missing semicolon: {stripped[:40]}")
            continue

        # Regular statement lines should end with ; or , or { or }
        stmt_lines += 1
        if stripped.endswith(";") or stripped.endswith(",") or stripped.endswith("{") or stripped.endswith("}"):
            semicolon_lines += 1
        elif "=>" in stripped:
            # Match arms can end with ,
            semicolon_lines += 1
        else:
            errors.append(f"possible missing semicolon: {stripped[:40]}")

    if stmt_lines == 0:
        return 1.0, []
    score = semicolon_lines / stmt_lines
    return score, errors


def _check_keywords(code: str) -> Tuple[float, List[str]]:
    """Check that Mog keywords are used correctly."""
    errors = []
    score = 1.0

    # fn declarations should have braces
    fn_matches = list(_FN_PATTERN.finditer(code))
    if not fn_matches and "struct" not in code:
        errors.append("no function or struct definitions found")
        score -= 0.5

    # Check for common non-Mog patterns
    if "def " in code:
        errors.append("Python-style 'def' found; should use 'fn'")
        score -= 0.5
    if "function " in code:
        errors.append("JS-style 'function' found; should use 'fn'")
        score -= 0.5
    if "var " in code and "var" not in MOG_KEYWORDS:
        errors.append("'var' is not a Mog keyword; use ':=' for bindings")
        score -= 0.1
    if "let " in code and ":=" not in code:
        errors.append("'let' without ':='; Mog uses ':=' for new bindings")
        score -= 0.1

    # return statements should exist in functions
    if fn_matches and not _RETURN_PATTERN.search(code):
        errors.append("no return statements found in functions")
        score -= 0.2

    return max(0.0, score), errors


def _check_types(code: str) -> Tuple[float, List[str]]:
    """Check type annotation presence and correctness."""
    errors = []

    # Count function parameters and return types
    fn_matches = list(_FN_PATTERN.finditer(code))
    if not fn_matches and not re.search(r"struct\s+\w+\s*\{", code):
        return 0.0, ["no typed definitions found"]  # No functions or structs to check
    if not fn_matches:
        # Only structs, check those below
        pass

    total_checks = 0
    passed_checks = 0

    for m in fn_matches:
        params_str = m.group(4)
        return_type = m.group(5)

        # Check return type annotation
        total_checks += 1
        if return_type:
            passed_checks += 1
        else:
            errors.append(f"function '{m.group(3)}' missing return type")

        # Check parameter type annotations
        if params_str.strip():
            params = [p.strip() for p in params_str.split(",") if p.strip()]
            for param in params:
                total_checks += 1
                if ":" in param:
                    passed_checks += 1
                else:
                    errors.append(f"parameter '{param}' missing type annotation")

    # Check struct field types
    struct_blocks = re.findall(r"struct\s+\w+\s*\{([^}]+)\}", code)
    for block in struct_blocks:
        fields = [f.strip() for f in block.split(",") if f.strip()]
        for fld in fields:
            total_checks += 1
            if ":" in fld:
                passed_checks += 1
            else:
                errors.append(f"struct field '{fld.strip()}' missing type")

    if total_checks == 0:
        return 1.0, []
    return passed_checks / total_checks, errors


def _check_capabilities(code: str) -> Tuple[float, List[str]]:
    """Check capability declaration correctness."""
    errors = []

    cap_matches = list(_CAPABILITY_PATTERN.finditer(code))
    if not cap_matches:
        # No capabilities declared - check if any are used without declaration
        # Look for patterns like fs.read, net.connect etc.
        cap_usage = re.findall(r"\b(fs|net|log|db|http|crypto|time|env)\.\w+", code)
        if cap_usage:
            used_caps = set(m for m in cap_usage)
            for cap in used_caps:
                errors.append(f"capability '{cap.split('.')[0]}' used but not declared")
            return 0.0, errors
        return 1.0, []  # No capabilities needed or used

    # Check that capability declarations are at the top (before fn/struct)
    first_fn_pos = len(code)
    fn_match = _FN_PATTERN.search(code)
    struct_match = _STRUCT_PATTERN.search(code)
    if fn_match:
        first_fn_pos = min(first_fn_pos, fn_match.start())
    if struct_match:
        first_fn_pos = min(first_fn_pos, struct_match.start())

    total_checks = len(cap_matches)
    passed = 0

    for m in cap_matches:
        if m.start() < first_fn_pos:
            passed += 1
        else:
            errors.append(f"'{m.group(1)} {m.group(2)}' should be at top of file")

    # Check that declared capabilities end with semicolons (already in pattern)
    # Check for valid capability names
    valid_caps = {"fs", "net", "log", "db", "http", "crypto", "time", "env",
                  "agent", "math", "json", "io", "fmt", "collections"}
    for m in cap_matches:
        cap_name = m.group(2)
        if cap_name not in valid_caps:
            errors.append(f"unknown capability '{cap_name}'")
            # Don't penalize - could be user-defined

    if total_checks == 0:
        return 1.0, []
    return passed / total_checks, errors


def _check_structure(code: str) -> Tuple[float, List[str]]:
    """Check overall structural correctness of the program."""
    errors = []
    score = 1.0

    # Check for empty program
    stripped = code.strip()
    if not stripped:
        return 0.0, ["empty program"]

    # Must contain at least one fn or struct to be a valid Mog program
    if not _FN_PATTERN.search(code) and not _STRUCT_PATTERN.search(code):
        errors.append("no fn or struct found - not a valid Mog program")
        score -= 0.5

    # Must use braces
    if "{" not in code or "}" not in code:
        errors.append("no braces found - Mog requires braces")
        score -= 0.3

    # Check := vs = usage
    bindings = _BINDING_PATTERN.findall(code)
    reassigns = _REASSIGN_PATTERN.findall(code)
    # Both should exist in non-trivial programs
    if len(code) > 200 and not bindings and not reassigns:
        errors.append("no variable bindings or reassignments found")
        score -= 0.1

    # Check for match arm syntax (=>)
    if _MATCH_PATTERN.search(code):
        if not _ARROW_PATTERN.search(code):
            errors.append("match block without '=>' arms")
            score -= 0.2

    # Check for Result usage consistency
    if "Result<" in code:
        if not _RESULT_PATTERN.search(code):
            errors.append("Result type used but no ok()/err() calls")
            score -= 0.1

    # Check for mixed operators without parens (heuristic)
    # Look for patterns like "a + b * c" without parens
    mixed_ops = re.findall(r"\b\w+\s*[+\-]\s*\w+\s*[*/]\s*\w+", code)
    for expr in mixed_ops:
        errors.append(f"possible missing parens (no operator precedence): {expr}")
        score -= 0.05

    return max(0.0, score), errors


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_mog_program(code: str) -> MogEvalResult:
    """Evaluate a Mog program on multiple quality dimensions.

    Args:
        code: Mog source code string.

    Returns:
        MogEvalResult with scores and error details.
    """
    result = MogEvalResult()

    # Run all checks
    result.bracket_balance, bracket_errors = _check_bracket_balance(code)
    result.errors.extend(bracket_errors)

    result.semicolon_usage, semi_errors = _check_semicolons(code)
    result.errors.extend(semi_errors)

    result.keyword_validity, kw_errors = _check_keywords(code)
    result.errors.extend(kw_errors)

    result.type_completeness, type_errors = _check_types(code)
    result.errors.extend(type_errors)

    result.capability_correctness, cap_errors = _check_capabilities(code)
    result.errors.extend(cap_errors)

    result.structural_correctness, struct_errors = _check_structure(code)
    result.errors.extend(struct_errors)

    # Extract info
    for m in _FN_PATTERN.finditer(code):
        result.functions.append(m.group(3))
    for m in _STRUCT_PATTERN.finditer(code):
        result.structs.append(m.group(1))
    for m in _CAPABILITY_PATTERN.finditer(code):
        result.capabilities[m.group(2)] = m.group(1)

    # Compute aggregates
    result.syntactic_validity = (
        result.bracket_balance * 0.4 +
        result.semicolon_usage * 0.3 +
        result.keyword_validity * 0.3
    )

    result.overall_score = (
        result.bracket_balance * 0.20 +
        result.semicolon_usage * 0.15 +
        result.keyword_validity * 0.15 +
        result.type_completeness * 0.20 +
        result.capability_correctness * 0.15 +
        result.structural_correctness * 0.15
    )

    return result


def evaluate_batch(programs: List[str]) -> Dict[str, float]:
    """Evaluate a batch of Mog programs and return aggregate metrics.

    Args:
        programs: List of Mog source code strings.

    Returns:
        Dict with mean scores across all programs.
    """
    if not programs:
        return {
            "syntactic_validity": 0.0,
            "type_completeness": 0.0,
            "structural_correctness": 0.0,
            "capability_correctness": 0.0,
            "overall_score": 0.0,
            "num_programs": 0,
            "num_valid": 0,
        }

    results = [evaluate_mog_program(p) for p in programs]

    n = len(results)
    num_valid = sum(1 for r in results if r.syntactic_validity >= 0.8)

    return {
        "syntactic_validity": sum(r.syntactic_validity for r in results) / n,
        "type_completeness": sum(r.type_completeness for r in results) / n,
        "structural_correctness": sum(r.structural_correctness for r in results) / n,
        "capability_correctness": sum(r.capability_correctness for r in results) / n,
        "overall_score": sum(r.overall_score for r in results) / n,
        "num_programs": n,
        "num_valid": num_valid,
        "validity_rate": num_valid / n * 100,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Demo: evaluate a sample program
    sample = """\
requires fs;
optional log;

// read_config: load config from file
pub fn read_config(path: string) -> Result<string> {
    if path == "" {
        return err("empty path");
    }
    data := fs.read(path);
    if log != nil {
        log.info("loaded config");
    }
    return ok(data);
}

struct Config {
    name: string,
    value: int,
    enabled: bool,
}

fn parse_config(raw: string) -> Result<Config> {
    if len(raw) == 0 {
        return err("empty config");
    }
    cfg := Config { name: "default", value: 42, enabled: true };
    return ok(cfg);
}
"""

    result = evaluate_mog_program(sample)
    print("Mog Program Evaluation")
    print("=" * 50)
    print(f"  Bracket balance:        {result.bracket_balance:.2f}")
    print(f"  Semicolon usage:        {result.semicolon_usage:.2f}")
    print(f"  Keyword validity:       {result.keyword_validity:.2f}")
    print(f"  Type completeness:      {result.type_completeness:.2f}")
    print(f"  Capability correctness: {result.capability_correctness:.2f}")
    print(f"  Structural correctness: {result.structural_correctness:.2f}")
    print(f"  ---")
    print(f"  Syntactic validity:     {result.syntactic_validity:.2f}")
    print(f"  Overall score:          {result.overall_score:.2f}")
    print(f"  ---")
    print(f"  Functions: {result.functions}")
    print(f"  Structs:   {result.structs}")
    print(f"  Capabilities: {result.capabilities}")
    if result.errors:
        print(f"  Errors:")
        for e in result.errors:
            print(f"    - {e}")
    if result.warnings:
        print(f"  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")
