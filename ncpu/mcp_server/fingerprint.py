"""Python mirror of nsynth's solved-cache fingerprint + on-disk reader.

The Rust synthesizer memoizes every verified solve in a persistent bank
(`~/.nsynth_solved_programs.json`, override via ``NSYNTH_CACHE_PATH``)
keyed by a deterministic *examples fingerprint*. This module mirrors
that fingerprint byte-for-byte (see ``nsynth/src/solved_cache.rs``:
``examples_fingerprint``) so Python callers can consult the bank
without launching the Rust binary.

Fingerprint grammar (no whitespace):

- int value      → ``i:{n}``
- string value   → ``s:{escaped}`` with ``|`` → ``\\|`` and ``~`` → ``\\~``
- array value    → ``a:[{x,y,z}]`` (comma-joined i64s)
- pair value     → ``p:({a},{b})``
- one example    → inputs joined by ``|``, then ``~{expected}``
- examples       → joined by ``;;``

On-disk record format (one per line, tab-separated):

- newer: ``fp \\t method \\t success_count \\t last_used_at \\t code``
- older: ``fp \\t method \\t code``

where ``code`` has ``\\n``/``\\t``/``\\r``/``\\\\`` escaped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def fingerprint_value(value: Any) -> str:
    """Deterministic string for one input value. Mirrors Rust exactly."""
    if isinstance(value, bool):
        raise ValueError("booleans are not valid nsynth input values")
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, str):
        return "s:" + value.replace("|", "\\|").replace("~", "\\~")
    if isinstance(value, list):
        return "a:[" + ",".join(str(int(x)) for x in value) + "]"
    if isinstance(value, tuple) and len(value) == 2:
        return f"p:({int(value[0])},{int(value[1])})"
    raise ValueError(f"unsupported input value type: {type(value).__name__}")


def examples_fingerprint(examples: list[dict[str, Any]]) -> str:
    """Fingerprint a list of ``{"inputs": [...], "expected": int}`` examples.

    Identical input to the Rust side produces an identical string, so an
    exact match against a bank record means the cached program was
    *verified against these very examples* by the synthesizer.
    """
    parts: list[str] = []
    for ex in examples:
        ins = "|".join(fingerprint_value(v) for v in ex["inputs"])
        parts.append(f"{ins}~{int(ex['expected'])}")
    return ";;".join(parts)


def decode_code(encoded: str) -> str:
    """Reverse the bank's single-line code escaping (mirrors Rust)."""
    out: list[str] = []
    it = iter(encoded)
    for c in it:
        if c != "\\":
            out.append(c)
            continue
        nxt = next(it, None)
        if nxt == "n":
            out.append("\n")
        elif nxt == "t":
            out.append("\t")
        elif nxt == "r":
            out.append("\r")
        elif nxt == "\\":
            out.append("\\")
        elif nxt is None:
            out.append("\\")
        else:
            out.append("\\")
            out.append(nxt)
    return "".join(out)


def lookup_solved(path: Optional[Path], fingerprint: str) -> Optional[dict[str, Any]]:
    """Scan a solved bank file for an exact fingerprint match.

    Returns ``{"method", "code", "success_count", "last_used_at"}`` on a
    hit, ``None`` on a miss (or when the bank is missing/disabled).
    """
    if path is None or not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    hit: Optional[dict[str, Any]] = None
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split("\t", 4)
        if len(parts) == 3:
            fp, method, code = parts
            success_count, last_used_at = 0, 0
        elif len(parts) == 5:
            fp, method, sc, lu, code = parts
            try:
                success_count = int(sc)
            except ValueError:
                success_count = 0
            try:
                last_used_at = int(lu)
            except ValueError:
                last_used_at = 0
        else:
            continue
        if fp == fingerprint:
            # Later records win, matching the Rust BTreeMap insert order.
            hit = {
                "method": method,
                "code": decode_code(code),
                "success_count": success_count,
                "last_used_at": last_used_at,
            }
    return hit


__all__ = [
    "fingerprint_value",
    "examples_fingerprint",
    "decode_code",
    "lookup_solved",
]
