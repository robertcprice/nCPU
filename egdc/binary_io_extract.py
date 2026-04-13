"""
binary_io_extract.py — Extract I/O examples from any function or binary.

Takes a Python function, module path, or compiled binary and automatically
generates input/output pairs for program synthesis training.

Supports:
  - Python functions (direct call)
  - Python module + function name (import + call)
  - Compiled binaries (subprocess with stdin/stdout)
  - Shared libraries via ctypes

The output is compatible with mog_synth Problem format (JSON) so it can be
fed directly into the differentiable synthesis pipeline.

Usage:
    from egdc.binary_io_extract import IOExtractor

    # From a Python function
    ext = IOExtractor.from_function(my_func, arg_types=["int", "int"])
    problem = ext.extract(n_examples=20)

    # From a binary
    ext = IOExtractor.from_binary("./my_program", arg_types=["int"])
    problem = ext.extract(n_examples=20)

    # Save for mog_synth
    ext.save_json("problem.json")
"""

from __future__ import annotations

import ctypes
import inspect
import json
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Example:
    inputs: list[Any]
    expected: Any

    def to_dict(self) -> dict:
        return {"inputs": self._serialize_list(self.inputs), "expected": self.expected}

    @staticmethod
    def _serialize_list(vals: list) -> list:
        out = []
        for v in vals:
            if isinstance(v, list):
                out.append({"Array": v})
            elif isinstance(v, str):
                out.append({"Str": v})
            elif isinstance(v, tuple) and len(v) == 2:
                out.append({"Pair": list(v)})
            else:
                out.append({"Int": int(v)})
        return out


@dataclass
class ExtractedProblem:
    name: str
    category: str
    description: str
    signature: str
    examples: list[Example]
    holdouts: list[Example] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "signature": self.signature,
            "examples": [e.to_dict() for e in self.examples],
            "holdouts": [e.to_dict() for e in self.holdouts],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    def to_mog_synth_json(self) -> str:
        """Format compatible with mog_synth --problem-json flag."""
        examples = []
        for ex in self.examples:
            inputs = []
            for v in ex.inputs:
                if isinstance(v, list):
                    inputs.append(v)
                else:
                    inputs.append(int(v))
            examples.append({"inputs": inputs, "expected": int(ex.expected)})
        holdouts = []
        for ex in self.holdouts:
            inputs = []
            for v in ex.inputs:
                if isinstance(v, list):
                    inputs.append(v)
                else:
                    inputs.append(int(v))
            holdouts.append({"inputs": inputs, "expected": int(ex.expected)})
        return json.dumps({
            "name": self.name,
            "signature": self.signature,
            "examples": examples,
            "holdouts": holdouts,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Input generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_int(lo: int = -100, hi: int = 100) -> int:
    return random.randint(lo, hi)

def gen_positive_int(lo: int = 1, hi: int = 50) -> int:
    return random.randint(lo, hi)

def gen_array(min_len: int = 1, max_len: int = 8, lo: int = -20, hi: int = 20) -> list[int]:
    n = random.randint(min_len, max_len)
    return [random.randint(lo, hi) for _ in range(n)]

def gen_string(min_len: int = 1, max_len: int = 10) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz "
    n = random.randint(min_len, max_len)
    return "".join(random.choice(chars) for _ in range(n))


INPUT_GENERATORS: dict[str, Callable] = {
    "int": gen_int,
    "positive_int": gen_positive_int,
    "nat": lambda: gen_positive_int(0, 30),
    "small_int": lambda: gen_int(-10, 10),
    "array": gen_array,
    "small_array": lambda: gen_array(1, 5, -10, 10),
    "string": gen_string,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core extractor
# ──────────────────────────────────────────────────────────────────────────────

class IOExtractor:
    """Extracts I/O examples from a callable or binary."""

    def __init__(
        self,
        caller: Callable[..., Any],
        arg_types: list[str],
        fn_name: str = "unknown",
        description: str = "",
        category: str = "extracted",
    ):
        self.caller = caller
        self.arg_types = arg_types
        self.fn_name = fn_name
        self.description = description
        self.category = category

    # ── Factory constructors ──────────────────────────────────────────────

    @classmethod
    def from_function(
        cls,
        fn: Callable,
        arg_types: list[str] | None = None,
        description: str = "",
    ) -> IOExtractor:
        """Create extractor from a Python function."""
        name = getattr(fn, "__name__", "unknown")
        if arg_types is None:
            sig = inspect.signature(fn)
            arg_types = cls._infer_arg_types(sig)
        return cls(
            caller=fn,
            arg_types=arg_types,
            fn_name=name,
            description=description or f"Extracted from Python function {name}",
        )

    @classmethod
    def from_module(
        cls,
        module_path: str,
        function_name: str,
        arg_types: list[str],
        description: str = "",
    ) -> IOExtractor:
        """Import a function from a Python module and extract I/O."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ext_mod", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, function_name)
        return cls(
            caller=fn,
            arg_types=arg_types,
            fn_name=function_name,
            description=description or f"Extracted from {module_path}:{function_name}",
        )

    @classmethod
    def from_binary(
        cls,
        binary_path: str,
        arg_types: list[str],
        fn_name: str = "program",
        description: str = "",
        input_format: str = "space",  # "space", "newline", "json"
    ) -> IOExtractor:
        """Create extractor from a compiled binary (stdin/stdout)."""
        binary_path = os.path.abspath(binary_path)

        def call_binary(*args: Any) -> int:
            if input_format == "json":
                stdin_data = json.dumps(list(args))
            elif input_format == "newline":
                stdin_data = "\n".join(str(a) for a in args)
            else:
                parts = []
                for a in args:
                    if isinstance(a, list):
                        parts.append(" ".join(str(x) for x in a))
                    else:
                        parts.append(str(a))
                stdin_data = " ".join(parts)

            result = subprocess.run(
                [binary_path],
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Binary failed: {result.stderr[:200]}")
            return int(result.stdout.strip())

        return cls(
            caller=call_binary,
            arg_types=arg_types,
            fn_name=fn_name,
            description=description or f"Extracted from binary {binary_path}",
            category="binary",
        )

    @classmethod
    def from_shared_lib(
        cls,
        lib_path: str,
        function_name: str,
        arg_types: list[str],
        return_type: str = "int",
        description: str = "",
    ) -> IOExtractor:
        """Load a function from a shared library (.so/.dylib) via ctypes."""
        lib = ctypes.CDLL(lib_path)
        fn = getattr(lib, function_name)
        fn.restype = ctypes.c_int64

        # Map arg types to ctypes
        c_arg_types = []
        for at in arg_types:
            if at in ("int", "small_int", "positive_int", "nat"):
                c_arg_types.append(ctypes.c_int64)
            else:
                raise ValueError(f"Unsupported ctypes arg type: {at}")
        fn.argtypes = c_arg_types

        return cls(
            caller=fn,
            arg_types=arg_types,
            fn_name=function_name,
            description=description or f"Extracted from {lib_path}:{function_name}",
            category="shared_lib",
        )

    # ── Extraction ────────────────────────────────────────────────────────

    def extract(
        self,
        n_examples: int = 12,
        n_holdouts: int = 4,
        seed: int | None = None,
        deduplicate: bool = True,
    ) -> ExtractedProblem:
        """Generate random inputs, call the function, collect I/O pairs."""
        if seed is not None:
            random.seed(seed)

        all_examples: list[Example] = []
        seen: set[str] = set()
        attempts = 0
        target = n_examples + n_holdouts

        while len(all_examples) < target and attempts < target * 10:
            attempts += 1
            args = [INPUT_GENERATORS[t]() for t in self.arg_types]
            try:
                result = self.caller(*args)
            except Exception:
                continue

            # Normalize result
            if isinstance(result, bool):
                result = int(result)
            if not isinstance(result, (int, float, list, str)):
                continue

            if isinstance(result, float):
                if result != int(result):
                    continue
                result = int(result)

            ex = Example(inputs=args, expected=result)
            key = repr((args, result))
            if deduplicate and key in seen:
                continue
            seen.add(key)
            all_examples.append(ex)

        examples = all_examples[:n_examples]
        holdouts = all_examples[n_examples:n_examples + n_holdouts]

        # Build signature
        sig = self._build_signature()

        return ExtractedProblem(
            name=self.fn_name,
            category=self.category,
            description=self.description,
            signature=sig,
            examples=examples,
            holdouts=holdouts,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_signature(self) -> str:
        type_map = {
            "int": "i64", "small_int": "i64", "positive_int": "i64", "nat": "i64",
            "array": "[i64]", "small_array": "[i64]",
            "string": "string",
        }
        arg_names = ["a", "b", "c", "d", "e", "f"]
        params = []
        for i, t in enumerate(self.arg_types):
            name = arg_names[i] if i < len(arg_names) else f"x{i}"
            mog_type = type_map.get(t, "i64")
            if mog_type == "[i64]":
                name = "arr"
            params.append(f"{name}: {mog_type}")
        return f"fn {self.fn_name}({', '.join(params)}) -> i64"

    @staticmethod
    def _infer_arg_types(sig: inspect.Signature) -> list[str]:
        """Best-effort inference of arg types from type annotations."""
        types = []
        for p in sig.parameters.values():
            ann = p.annotation
            if ann is inspect.Parameter.empty:
                types.append("int")
            elif ann is int:
                types.append("int")
            elif ann is list or (hasattr(ann, "__origin__") and ann.__origin__ is list):
                types.append("array")
            elif ann is str:
                types.append("string")
            else:
                types.append("int")
        return types


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """CLI: extract I/O from a Python function."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract I/O examples from functions/binaries")
    parser.add_argument("source", help="Python file:function_name or binary path")
    parser.add_argument("--arg-types", nargs="+", default=["int"],
                        help="Argument types: int, small_int, positive_int, array, string")
    parser.add_argument("--n-examples", type=int, default=12)
    parser.add_argument("--n-holdouts", type=int, default=4)
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")
    parser.add_argument("--binary", action="store_true", help="Treat source as binary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.binary:
        ext = IOExtractor.from_binary(args.source, args.arg_types)
    elif ":" in args.source:
        file_path, func_name = args.source.rsplit(":", 1)
        ext = IOExtractor.from_module(file_path, func_name, args.arg_types)
    else:
        print(f"Error: source must be file:function or --binary path", file=sys.stderr)
        sys.exit(1)

    problem = ext.extract(
        n_examples=args.n_examples,
        n_holdouts=args.n_holdouts,
        seed=args.seed,
    )

    output = problem.to_json()
    if args.output:
        Path(args.output).write_text(output)
        print(f"Saved {len(problem.examples)} examples + {len(problem.holdouts)} holdouts to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
