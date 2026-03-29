"""PyTorch Dataset for EGDC diffusion training on Mog programs.

Provides (masked_tokens, mask_positions, original_tokens, spec_tokens, timestep)
tuples for training a discrete diffusion model on Mog source code.

Specs are encoded by tokenizing the function signature or first comment line.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from egdc.mog_tokenizer import (
    MogCodeTokenizer, MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, VOCAB_SIZE,
)


# Spec encoding constants
SPEC_SEQ_LEN = 64


# ---------------------------------------------------------------------------
# Mog Program Generator
# ---------------------------------------------------------------------------

# Name pools for variation
_FUNC_NAMES = [
    "compute", "process", "transform", "calculate", "evaluate", "parse",
    "validate", "convert", "encode", "decode", "filter", "accumulate",
    "merge", "split", "format", "handle", "dispatch", "resolve", "build",
    "create", "update", "delete", "fetch", "store", "compare", "sort",
    "search", "collect", "reduce", "expand", "compress", "hash_val",
    "normalize", "denormalize", "clamp_val", "interpolate", "quantize",
]

_VAR_NAMES = [
    "x", "y", "z", "val", "acc", "tmp", "result", "count", "sum", "total",
    "idx", "item", "elem", "data", "buf", "len", "pos", "cur", "prev", "next_val",
]

_STRUCT_NAMES = [
    "Point", "Vector", "Config", "Entry", "Record", "Node", "Pair",
    "State", "Context", "Metadata", "Header", "Payload", "Frame",
    "Token", "Symbol", "Range", "Span", "Slot", "Block", "Chunk",
]

_INT_TYPES = ["int", "i32", "u32", "u64"]
_FLOAT_TYPES = ["float", "f32", "f16", "bf16"]
_ALL_TYPES = _INT_TYPES + _FLOAT_TYPES + ["bool", "string"]

_CAP_NAMES = ["fs", "net", "log", "db", "http", "crypto", "time", "env"]
_IMPORT_NAMES = ["agent", "math", "json", "io", "fmt", "collections"]


class MogProgramGenerator:
    """Generates diverse synthetic Mog programs for training."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._templates = [
            self._gen_simple_fn,
            self._gen_fn_with_if,
            self._gen_fn_with_for,
            self._gen_fn_with_while,
            self._gen_fn_with_match,
            self._gen_struct_def,
            self._gen_struct_with_methods,
            self._gen_result_fn,
            self._gen_pub_fn,
            self._gen_async_fn,
            self._gen_capability_program,
            self._gen_array_fn,
            self._gen_map_fn,
            self._gen_nested_if,
            self._gen_for_with_accumulator,
            self._gen_while_with_break,
            self._gen_match_result,
            self._gen_multi_fn_program,
            self._gen_struct_constructor,
            self._gen_error_handling,
            self._gen_string_fn,
            self._gen_bool_logic,
            self._gen_float_math,
            self._gen_nested_loops,
            self._gen_recursive_fn,
            self._gen_pub_struct_program,
            self._gen_full_capability_program,
            self._gen_array_map_combo,
            self._gen_complex_match,
            self._gen_async_result_fn,
            self._gen_multi_struct_program,
            self._gen_pipeline_fn,
            self._gen_validator_fn,
            self._gen_converter_fn,
            self._gen_accumulator_pattern,
        ]

    def _pick(self, lst: list) -> str:
        return self.rng.choice(lst)

    def _pick_int_type(self) -> str:
        return self._pick(_INT_TYPES)

    def _pick_float_type(self) -> str:
        return self._pick(_FLOAT_TYPES)

    def _pick_type(self) -> str:
        return self._pick(_ALL_TYPES)

    def _pick_func_name(self) -> str:
        return self._pick(_FUNC_NAMES)

    def _pick_var(self) -> str:
        return self._pick(_VAR_NAMES)

    def _pick_struct(self) -> str:
        return self._pick(_STRUCT_NAMES)

    def _pick_cap(self) -> str:
        return self._pick(_CAP_NAMES)

    def _pick_import(self) -> str:
        return self._pick(_IMPORT_NAMES)

    def _rand_int(self, lo: int = 0, hi: int = 100) -> int:
        return self.rng.randint(lo, hi)

    # ------------------------------------------------------------------
    # Templates (35 total)
    # ------------------------------------------------------------------

    def _gen_simple_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        a, b = self._pick_var(), self._pick_var()
        if a == b:
            b = b + "2"
        code = (
            f"// {name}: add two {t} values\n"
            f"fn {name}({a}: {t}, {b}: {t}) -> {t} {{\n"
            f"    result := {a} + {b};\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"fn {name}({a}: {t}, {b}: {t}) -> {t}"
        return code, spec

    def _gen_fn_with_if(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        v = self._pick_var()
        threshold = self._rand_int(1, 50)
        code = (
            f"// {name}: check if {v} exceeds threshold\n"
            f"fn {name}({v}: {t}) -> bool {{\n"
            f"    if {v} > {threshold} {{\n"
            f"        return true;\n"
            f"    }} else {{\n"
            f"        return false;\n"
            f"    }}\n"
            f"}}\n"
        )
        spec = f"fn {name}({v}: {t}) -> bool"
        return code, spec

    def _gen_fn_with_for(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        n = self._rand_int(5, 20)
        code = (
            f"// {name}: sum integers from 0 to {n}\n"
            f"fn {name}() -> {t} {{\n"
            f"    acc := 0;\n"
            f"    for i := 0 to {n} {{\n"
            f"        acc = acc + i;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}\n"
        )
        spec = f"fn {name}() -> {t}"
        return code, spec

    def _gen_fn_with_while(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        limit = self._rand_int(10, 100)
        code = (
            f"// {name}: count up to {limit}\n"
            f"fn {name}() -> {t} {{\n"
            f"    count := 0;\n"
            f"    while count < {limit} {{\n"
            f"        count = count + 1;\n"
            f"    }}\n"
            f"    return count;\n"
            f"}}\n"
        )
        spec = f"fn {name}() -> {t}"
        return code, spec

    def _gen_fn_with_match(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: classify value via match\n"
            f"fn {name}(val: {t}) -> string {{\n"
            f"    result := match val {{\n"
            f"        0 => \"zero\",\n"
            f"        1 => \"one\",\n"
            f"        _ => \"other\",\n"
            f"    }};\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"fn {name}(val: {t}) -> string"
        return code, spec

    def _gen_struct_def(self) -> Tuple[str, str]:
        sname = self._pick_struct()
        t1, t2 = self._pick_type(), self._pick_type()
        f1, f2 = "x", "y"
        code = (
            f"// struct {sname}\n"
            f"struct {sname} {{\n"
            f"    {f1}: {t1},\n"
            f"    {f2}: {t2},\n"
            f"}}\n"
        )
        spec = f"struct {sname}"
        return code, spec

    def _gen_struct_with_methods(self) -> Tuple[str, str]:
        sname = self._pick_struct()
        t = self._pick_int_type()
        code = (
            f"// {sname} with methods\n"
            f"struct {sname} {{\n"
            f"    value: {t},\n"
            f"    label: string,\n"
            f"}}\n\n"
            f"fn new_{sname.lower()}(v: {t}, l: string) -> {sname} {{\n"
            f"    return {sname} {{ value: v, label: l }};\n"
            f"}}\n\n"
            f"fn get_value(s: {sname}) -> {t} {{\n"
            f"    return s.value;\n"
            f"}}\n"
        )
        spec = f"struct {sname} with methods"
        return code, spec

    def _gen_result_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: division with error handling\n"
            f"fn {name}(a: {t}, b: {t}) -> Result<{t}> {{\n"
            f"    if b == 0 {{\n"
            f"        return err(\"division by zero\");\n"
            f"    }}\n"
            f"    return ok(a / b);\n"
            f"}}\n"
        )
        spec = f"fn {name}(a: {t}, b: {t}) -> Result<{t}>"
        return code, spec

    def _gen_pub_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// public {name}\n"
            f"pub fn {name}(input: {t}) -> {t} {{\n"
            f"    result := input * 2;\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"pub fn {name}(input: {t}) -> {t}"
        return code, spec

    def _gen_async_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        code = (
            f"// async {name}\n"
            f"async fn {name}(url: string) -> Result<string> {{\n"
            f"    data := await fetch(url);\n"
            f"    if data == \"\" {{\n"
            f"        return err(\"empty response\");\n"
            f"    }}\n"
            f"    return ok(data);\n"
            f"}}\n"
        )
        spec = f"async fn {name}(url: string) -> Result<string>"
        return code, spec

    def _gen_capability_program(self) -> Tuple[str, str]:
        cap = self._pick_cap()
        name = self._pick_func_name()
        code = (
            f"requires {cap};\n\n"
            f"// {name}: uses {cap} capability\n"
            f"fn {name}(path: string) -> Result<string> {{\n"
            f"    data := {cap}.read(path);\n"
            f"    return ok(data);\n"
            f"}}\n"
        )
        spec = f"requires {cap}; fn {name}"
        return code, spec

    def _gen_array_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: sum array elements\n"
            f"fn {name}(arr: []{t}) -> {t} {{\n"
            f"    total := 0;\n"
            f"    for i := 0 to len(arr) {{\n"
            f"        total = total + arr[i];\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}\n"
        )
        spec = f"fn {name}(arr: []{t}) -> {t}"
        return code, spec

    def _gen_map_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        kt, vt = "string", self._pick_int_type()
        code = (
            f"// {name}: lookup in map\n"
            f"fn {name}(m: [{kt}]{vt}, key: {kt}) -> Result<{vt}> {{\n"
            f"    if has(m, key) {{\n"
            f"        return ok(m[key]);\n"
            f"    }}\n"
            f"    return err(\"key not found\");\n"
            f"}}\n"
        )
        spec = f"fn {name}(m: [{kt}]{vt}, key: {kt}) -> Result<{vt}>"
        return code, spec

    def _gen_nested_if(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        lo, hi = self._rand_int(0, 10), self._rand_int(50, 100)
        code = (
            f"// {name}: nested conditionals\n"
            f"fn {name}(val: {t}) -> string {{\n"
            f"    if val < {lo} {{\n"
            f"        return \"low\";\n"
            f"    }} else {{\n"
            f"        if val > {hi} {{\n"
            f"            return \"high\";\n"
            f"        }} else {{\n"
            f"            return \"mid\";\n"
            f"        }}\n"
            f"    }}\n"
            f"}}\n"
        )
        spec = f"fn {name}(val: {t}) -> string"
        return code, spec

    def _gen_for_with_accumulator(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        n = self._rand_int(5, 30)
        code = (
            f"// {name}: factorial up to {n}\n"
            f"fn {name}(n: {t}) -> {t} {{\n"
            f"    acc := 1;\n"
            f"    for i := 1 to (n + 1) {{\n"
            f"        acc = acc * i;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}\n"
        )
        spec = f"fn {name}(n: {t}) -> {t}"
        return code, spec

    def _gen_while_with_break(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        target = self._rand_int(10, 50)
        code = (
            f"// {name}: search for target {target}\n"
            f"fn {name}(arr: []{t}) -> {t} {{\n"
            f"    idx := 0;\n"
            f"    while idx < len(arr) {{\n"
            f"        if arr[idx] == {target} {{\n"
            f"            return idx;\n"
            f"        }}\n"
            f"        idx = idx + 1;\n"
            f"    }}\n"
            f"    return -1;\n"
            f"}}\n"
        )
        spec = f"fn {name}(arr: []{t}) -> {t}"
        return code, spec

    def _gen_match_result(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: match on Result\n"
            f"fn {name}(r: Result<{t}>) -> {t} {{\n"
            f"    val := match r {{\n"
            f"        ok(v) => v,\n"
            f"        err(_) => 0,\n"
            f"    }};\n"
            f"    return val;\n"
            f"}}\n"
        )
        spec = f"fn {name}(r: Result<{t}>) -> {t}"
        return code, spec

    def _gen_multi_fn_program(self) -> Tuple[str, str]:
        n1, n2 = self._pick_func_name(), self._pick_func_name()
        if n1 == n2:
            n2 = n2 + "_alt"
        t = self._pick_int_type()
        code = (
            f"// multi-function program\n"
            f"fn {n1}(a: {t}, b: {t}) -> {t} {{\n"
            f"    return a + b;\n"
            f"}}\n\n"
            f"fn {n2}(x: {t}) -> {t} {{\n"
            f"    doubled := {n1}(x, x);\n"
            f"    return doubled;\n"
            f"}}\n"
        )
        spec = f"fn {n1}, fn {n2}"
        return code, spec

    def _gen_struct_constructor(self) -> Tuple[str, str]:
        sname = self._pick_struct()
        t = self._pick_int_type()
        code = (
            f"// {sname} constructor pattern\n"
            f"struct {sname} {{\n"
            f"    id: {t},\n"
            f"    name: string,\n"
            f"    active: bool,\n"
            f"}}\n\n"
            f"fn new_{sname.lower()}(id: {t}, name: string) -> {sname} {{\n"
            f"    return {sname} {{ id: id, name: name, active: true }};\n"
            f"}}\n"
        )
        spec = f"struct {sname} constructor"
        return code, spec

    def _gen_error_handling(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: chained error handling\n"
            f"fn {name}(a: {t}, b: {t}) -> Result<{t}> {{\n"
            f"    if a < 0 {{\n"
            f"        return err(\"negative input\");\n"
            f"    }}\n"
            f"    if b == 0 {{\n"
            f"        return err(\"zero divisor\");\n"
            f"    }}\n"
            f"    result := a / b;\n"
            f"    return ok(result);\n"
            f"}}\n"
        )
        spec = f"fn {name}(a: {t}, b: {t}) -> Result<{t}>"
        return code, spec

    def _gen_string_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        code = (
            f"// {name}: string operation\n"
            f"fn {name}(s: string, prefix: string) -> bool {{\n"
            f"    if len(s) == 0 {{\n"
            f"        return false;\n"
            f"    }}\n"
            f"    return starts_with(s, prefix);\n"
            f"}}\n"
        )
        spec = f"fn {name}(s: string, prefix: string) -> bool"
        return code, spec

    def _gen_bool_logic(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        code = (
            f"// {name}: boolean logic\n"
            f"fn {name}(a: bool, b: bool, c: bool) -> bool {{\n"
            f"    result := (a && b) || c;\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"fn {name}(a: bool, b: bool, c: bool) -> bool"
        return code, spec

    def _gen_float_math(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        ft = self._pick_float_type()
        code = (
            f"// {name}: floating point math\n"
            f"fn {name}(x: {ft}, y: {ft}) -> {ft} {{\n"
            f"    sum := x + y;\n"
            f"    avg := sum / 2.0;\n"
            f"    return avg;\n"
            f"}}\n"
        )
        spec = f"fn {name}(x: {ft}, y: {ft}) -> {ft}"
        return code, spec

    def _gen_nested_loops(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        n = self._rand_int(3, 10)
        code = (
            f"// {name}: nested loops\n"
            f"fn {name}(n: {t}) -> {t} {{\n"
            f"    total := 0;\n"
            f"    for i := 0 to n {{\n"
            f"        for j := 0 to n {{\n"
            f"            total = total + (i * j);\n"
            f"        }}\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}\n"
        )
        spec = f"fn {name}(n: {t}) -> {t}"
        return code, spec

    def _gen_recursive_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: recursive computation\n"
            f"fn {name}(n: {t}) -> {t} {{\n"
            f"    if n <= 1 {{\n"
            f"        return n;\n"
            f"    }}\n"
            f"    return {name}(n - 1) + {name}(n - 2);\n"
            f"}}\n"
        )
        spec = f"fn {name}(n: {t}) -> {t}"
        return code, spec

    def _gen_pub_struct_program(self) -> Tuple[str, str]:
        sname = self._pick_struct()
        t = self._pick_int_type()
        code = (
            f"// public API for {sname}\n"
            f"struct {sname} {{\n"
            f"    data: []{t},\n"
            f"    size: {t},\n"
            f"}}\n\n"
            f"pub fn create_{sname.lower()}(capacity: {t}) -> {sname} {{\n"
            f"    return {sname} {{ data: [], size: 0 }};\n"
            f"}}\n\n"
            f"pub fn add_item(s: {sname}, item: {t}) -> {sname} {{\n"
            f"    new_data := append(s.data, item);\n"
            f"    return {sname} {{ data: new_data, size: s.size + 1 }};\n"
            f"}}\n"
        )
        spec = f"pub struct {sname} API"
        return code, spec

    def _gen_full_capability_program(self) -> Tuple[str, str]:
        cap1 = self._pick_cap()
        cap2 = self._pick_cap()
        if cap2 == cap1:
            cap2 = "log"
        imp = self._pick_import()
        name = self._pick_func_name()
        code = (
            f"requires {cap1};\n"
            f"optional {cap2};\n"
            f"import {imp};\n\n"
            f"// {name}: full capability program\n"
            f"pub fn {name}(input: string) -> Result<string> {{\n"
            f"    data := {cap1}.read(input);\n"
            f"    if {cap2} != nil {{\n"
            f"        {cap2}.info(\"processing: \" + input);\n"
            f"    }}\n"
            f"    result := {imp}.process(data);\n"
            f"    return ok(result);\n"
            f"}}\n"
        )
        spec = f"requires {cap1}; optional {cap2}; import {imp}; pub fn {name}"
        return code, spec

    def _gen_array_map_combo(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: array to map conversion\n"
            f"fn {name}(keys: []string, values: []{t}) -> [string]{t} {{\n"
            f"    result := [string]{t}{{}};\n"
            f"    for i := 0 to len(keys) {{\n"
            f"        result[keys[i]] = values[i];\n"
            f"    }}\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"fn {name}(keys: []string, values: []{t}) -> [string]{t}"
        return code, spec

    def _gen_complex_match(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: complex pattern matching\n"
            f"fn {name}(code: {t}) -> string {{\n"
            f"    msg := match code {{\n"
            f"        200 => \"ok\",\n"
            f"        201 => \"created\",\n"
            f"        400 => \"bad request\",\n"
            f"        404 => \"not found\",\n"
            f"        500 => \"internal error\",\n"
            f"        _ => \"unknown\",\n"
            f"    }};\n"
            f"    return msg;\n"
            f"}}\n"
        )
        spec = f"fn {name}(code: {t}) -> string"
        return code, spec

    def _gen_async_result_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        cap = self._pick_cap()
        code = (
            f"requires {cap};\n\n"
            f"// {name}: async with result\n"
            f"async fn {name}(id: string) -> Result<int> {{\n"
            f"    raw := await {cap}.fetch(id);\n"
            f"    val := match parse_int(raw) {{\n"
            f"        ok(v) => v,\n"
            f"        err(e) => return err(e),\n"
            f"    }};\n"
            f"    return ok(val);\n"
            f"}}\n"
        )
        spec = f"async fn {name}(id: string) -> Result<int>"
        return code, spec

    def _gen_multi_struct_program(self) -> Tuple[str, str]:
        s1, s2 = self._pick_struct(), self._pick_struct()
        if s1 == s2:
            s2 = s2 + "Info"
        t = self._pick_int_type()
        code = (
            f"// multi-struct program\n"
            f"struct {s1} {{\n"
            f"    id: {t},\n"
            f"    name: string,\n"
            f"}}\n\n"
            f"struct {s2} {{\n"
            f"    owner: {s1},\n"
            f"    count: {t},\n"
            f"}}\n\n"
            f"fn create_{s2.lower()}(owner_id: {t}, owner_name: string) -> {s2} {{\n"
            f"    o := {s1} {{ id: owner_id, name: owner_name }};\n"
            f"    return {s2} {{ owner: o, count: 0 }};\n"
            f"}}\n"
        )
        spec = f"struct {s1}, struct {s2}"
        return code, spec

    def _gen_pipeline_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: data pipeline\n"
            f"fn step1(x: {t}) -> {t} {{\n"
            f"    return x * 2;\n"
            f"}}\n\n"
            f"fn step2(x: {t}) -> {t} {{\n"
            f"    return x + 10;\n"
            f"}}\n\n"
            f"fn {name}(input: {t}) -> {t} {{\n"
            f"    a := step1(input);\n"
            f"    b := step2(a);\n"
            f"    return b;\n"
            f"}}\n"
        )
        spec = f"fn {name}(input: {t}) -> {t} pipeline"
        return code, spec

    def _gen_validator_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        lo, hi = self._rand_int(0, 10), self._rand_int(90, 200)
        code = (
            f"// {name}: input validation\n"
            f"fn {name}(val: {t}) -> Result<{t}> {{\n"
            f"    if val < {lo} {{\n"
            f"        return err(\"too small\");\n"
            f"    }}\n"
            f"    if val > {hi} {{\n"
            f"        return err(\"too large\");\n"
            f"    }}\n"
            f"    return ok(val);\n"
            f"}}\n"
        )
        spec = f"fn {name}(val: {t}) -> Result<{t}>"
        return code, spec

    def _gen_converter_fn(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        from_t = self._pick_int_type()
        to_t = self._pick_float_type()
        code = (
            f"// {name}: type conversion\n"
            f"fn {name}(val: {from_t}) -> {to_t} {{\n"
            f"    result := val as {to_t};\n"
            f"    return result;\n"
            f"}}\n"
        )
        spec = f"fn {name}(val: {from_t}) -> {to_t}"
        return code, spec

    def _gen_accumulator_pattern(self) -> Tuple[str, str]:
        name = self._pick_func_name()
        t = self._pick_int_type()
        code = (
            f"// {name}: accumulator with early exit\n"
            f"fn {name}(arr: []{t}, target: {t}) -> bool {{\n"
            f"    sum := 0;\n"
            f"    for i := 0 to len(arr) {{\n"
            f"        sum = sum + arr[i];\n"
            f"        if sum >= target {{\n"
            f"            return true;\n"
            f"        }}\n"
            f"    }}\n"
            f"    return false;\n"
            f"}}\n"
        )
        spec = f"fn {name}(arr: []{t}, target: {t}) -> bool"
        return code, spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_one(self) -> Tuple[str, str]:
        """Generate a single (code, spec) pair."""
        template = self.rng.choice(self._templates)
        return template()

    def generate_dataset(
        self, num_samples: int, balanced: bool = True
    ) -> List[Tuple[str, str]]:
        """Generate a dataset of (code, spec) pairs.

        Args:
            num_samples: Number of programs to generate.
            balanced: If True, cycle through templates evenly.

        Returns:
            List of (mog_code, spec_string) tuples.
        """
        data: List[Tuple[str, str]] = []
        if balanced:
            for i in range(num_samples):
                template = self._templates[i % len(self._templates)]
                data.append(template())
        else:
            for _ in range(num_samples):
                data.append(self.generate_one())
        return data


# ---------------------------------------------------------------------------
# MogDataset
# ---------------------------------------------------------------------------

class MogDataset(Dataset):
    """PyTorch Dataset for Mog program diffusion training.

    Each item returns a tuple of tensors:
        masked_tokens:   (seq_len,) int64 - program with random positions masked
        mask_positions:  (seq_len,) bool  - True where tokens are masked
        original_tokens: (seq_len,) int64 - original unmasked program
        spec_tokens:     (spec_len,) int64 - encoded specification
        timestep:        scalar float     - diffusion timestep in [0, 1]

    Programs are padded/truncated to seq_len.
    Specs are encoded to spec_len by tokenizing the function signature.
    """

    def __init__(
        self,
        num_samples: int = 100_000,
        seq_len: int = 512,
        spec_len: int = SPEC_SEQ_LEN,
        seed: int = 42,
        balanced: bool = True,
        num_diffusion_steps: int = 1000,
    ):
        super().__init__()
        self.tokenizer = MogCodeTokenizer()
        self.seq_len = seq_len
        self.spec_len = spec_len
        self.num_diffusion_steps = num_diffusion_steps
        self.rng = random.Random(seed)

        gen = MogProgramGenerator(seed=seed)
        raw = gen.generate_dataset(num_samples, balanced=balanced)
        # Tokenize all programs
        self.data: List[Tuple[List[int], List[int]]] = []
        for code, spec in raw:
            code_tokens = self.tokenizer.encode(code)
            spec_tokens = self.tokenizer.encode(spec, add_bos_eos=False)
            self.data.append((code_tokens, spec_tokens))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        code_tokens, spec_tokens = self.data[idx]

        # --- Program tokens: pad/truncate to seq_len ---
        prog = self.tokenizer.pad(list(code_tokens), self.seq_len)
        original_tokens = torch.tensor(prog, dtype=torch.long)

        # --- Spec tokens: pad/truncate to spec_len ---
        spec = self.tokenizer.pad(list(spec_tokens), self.spec_len)
        spec_tensor = torch.tensor(spec, dtype=torch.long)

        # --- Diffusion masking ---
        t_int = self.rng.randint(0, self.num_diffusion_steps - 1)
        timestep = torch.tensor(t_int / self.num_diffusion_steps, dtype=torch.float32)

        mask_prob = t_int / self.num_diffusion_steps

        mask_positions = torch.zeros(self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len):
            tok = prog[i]
            if tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            if self.rng.random() < mask_prob:
                mask_positions[i] = True

        masked_tokens = original_tokens.clone()
        masked_tokens[mask_positions] = MASK_TOKEN

        return masked_tokens, mask_positions, original_tokens, spec_tensor, timestep

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def decode_program(self, token_ids: torch.Tensor) -> str:
        """Decode a token tensor back to Mog source code."""
        return self.tokenizer.decode(token_ids.tolist())

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True,
                       num_workers: int = 0, **kwargs) -> "torch.utils.data.DataLoader":
        """Convenience method to create a DataLoader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, **kwargs,
        )
