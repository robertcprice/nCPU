"""Safe Mog dataset generator for EGDC.

This version is constrained to the subset confirmed to compile and run with the
real Mog compiler (mogc) on macOS/arm64. Every generated sample includes:

- compiler-safe Mog source code
- a natural-language spec / function signature string
- a known expected stdout string produced by main()

The dataset itself still returns the 5-tuple expected by the diffusion trainer:
    masked_tokens, mask_positions, original_tokens, spec_tokens, timestep

Expected outputs are stored alongside each sample for execution-based eval.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from egdc.mog_tokenizer import (
    MogCodeTokenizer,
    MASK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)

SPEC_SEQ_LEN = 64


@dataclass
class MogSample:
    code: str
    spec: str
    expected_output: str
    name: str


class MogProgramGenerator:
    """Generates compiler-safe Mog programs with known outputs.

    Important: this generator intentionally stays inside the subset confirmed by
    egdc/mog_compiler_compat.md. It does NOT try to cover every Mog feature.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._templates = [
            self._gen_add,
            self._gen_threshold_if,
            self._gen_sum_for_to,
            self._gen_sum_range,
            self._gen_count_while,
            self._gen_factorial,
            self._gen_fibonacci,
            self._gen_result_match,
            self._gen_optional_match,
            self._gen_array_sum,
            self._gen_array_push,
            self._gen_closure_map,
            self._gen_struct_sum,
            self._gen_struct_mutation,
            self._gen_nested_loops,
            self._gen_pipeline,
            self._gen_validator,
            self._gen_map_iteration,
            self._gen_string_len,
            self._gen_match_literal,
        ]

    def _ri(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _safe_name(self, prefix: str) -> str:
        return f"{prefix}_{self._ri(1000, 9999)}"

    def _wrap(self, helper_code: str, main_body: str, spec: str, expected: str, name: str) -> MogSample:
        code = helper_code.rstrip() + "\n\n" + (
            "fn main() -> i64 {\n"
            + main_body.rstrip()
            + "\n    return 0;\n"
            + "}\n"
        )
        return MogSample(code=code, spec=spec, expected_output=expected, name=name)

    def _gen_add(self) -> MogSample:
        a, b = self._ri(1, 50), self._ri(1, 50)
        name = self._safe_name("add")
        helper = (
            f"fn {name}(a: i64, b: i64) -> i64 {{\n"
            f"    return a + b;\n"
            f"}}"
        )
        main = f"    println_i64({name}({a}, {b}));"
        return self._wrap(helper, main, f"fn {name}(a: i64, b: i64) -> i64", str(a + b), name)

    def _gen_threshold_if(self) -> MogSample:
        x = self._ri(0, 100)
        threshold = self._ri(20, 80)
        out = 1 if x > threshold else 0
        name = self._safe_name("gt_threshold")
        helper = (
            f"fn {name}(x: i64) -> i64 {{\n"
            f"    if x > {threshold} {{\n"
            f"        return 1;\n"
            f"    }} else {{\n"
            f"        return 0;\n"
            f"    }}\n"
            f"}}"
        )
        main = f"    println_i64({name}({x}));"
        return self._wrap(helper, main, f"fn {name}(x: i64) -> i64", str(out), name)

    def _gen_sum_for_to(self) -> MogSample:
        n = self._ri(5, 15)
        expected = sum(range(0, n))
        name = self._safe_name("sum_for")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    acc: i64 = 0;\n"
            f"    for i := 0 to {n} {{\n"
            f"        acc = acc + i;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}"
        )
        main = f"    println_i64({name}());"
        return self._wrap(helper, main, f"fn {name}() -> i64", str(expected), name)

    def _gen_sum_range(self) -> MogSample:
        n = self._ri(5, 15)
        expected = sum(range(0, n))
        name = self._safe_name("sum_range")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    acc: i64 = 0;\n"
            f"    for i in 0..{n} {{\n"
            f"        acc = acc + i;\n"
            f"    }}\n"
            f"    return acc;\n"
            f"}}"
        )
        main = f"    println_i64({name}());"
        return self._wrap(helper, main, f"fn {name}() -> i64", str(expected), name)

    def _gen_count_while(self) -> MogSample:
        limit = self._ri(3, 20)
        name = self._safe_name("count_to")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    count: i64 = 0;\n"
            f"    while count < {limit} {{\n"
            f"        count = count + 1;\n"
            f"    }}\n"
            f"    return count;\n"
            f"}}"
        )
        main = f"    println_i64({name}());"
        return self._wrap(helper, main, f"fn {name}() -> i64", str(limit), name)

    def _gen_factorial(self) -> MogSample:
        n = self._ri(3, 8)
        expected = 1
        for i in range(2, n + 1):
            expected *= i
        name = self._safe_name("factorial")
        helper = (
            f"fn {name}(n: i64) -> i64 {{\n"
            f"    if (n <= 1) {{ return 1; }}\n"
            f"    return n * {name}(n - 1);\n"
            f"}}"
        )
        main = f"    println_i64({name}({n}));"
        return self._wrap(helper, main, f"fn {name}(n: i64) -> i64", str(expected), name)

    def _gen_fibonacci(self) -> MogSample:
        n = self._ri(5, 12)
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        expected = a
        name = self._safe_name("fibonacci")
        helper = (
            f"fn {name}(n: i64) -> i64 {{\n"
            f"    if (n <= 0) {{ return 0; }}\n"
            f"    if (n == 1) {{ return 1; }}\n"
            f"    a: i64 = 0;\n"
            f"    b: i64 = 1;\n"
            f"    i: i64 = 2;\n"
            f"    while (i <= n) {{\n"
            f"        tmp := a + b;\n"
            f"        a = b;\n"
            f"        b = tmp;\n"
            f"        i = i + 1;\n"
            f"    }}\n"
            f"    return b;\n"
            f"}}"
        )
        main = f"    println_i64({name}({n}));"
        return self._wrap(helper, main, f"fn {name}(n: i64) -> i64", str(expected), name)

    def _gen_result_match(self) -> MogSample:
        a = self._ri(10, 80)
        b = self._ri(1, 9)
        expected = a // b
        name = self._safe_name("safe_div")
        helper = (
            f"fn {name}(a: i64, b: i64) -> Result<i64> {{\n"
            f"    if (b == 0) {{ return err(\"division by zero\"); }}\n"
            f"    return ok(a / b);\n"
            f"}}"
        )
        main = (
            f"    r := {name}({a}, {b});\n"
            f"    v: i64 = match r {{\n"
            f"        ok(x) => x,\n"
            f"        err(e) => -1,\n"
            f"    }};\n"
            f"    println_i64(v);"
        )
        return self._wrap(helper, main, f"fn {name}(a: i64, b: i64) -> Result<i64>", str(expected), name)

    def _gen_optional_match(self) -> MogSample:
        n = self._ri(-10, 20)
        expected = n if n > 0 else -1
        name = self._safe_name("find_positive")
        helper = (
            f"fn {name}(n: i64) -> ?i64 {{\n"
            f"    if (n > 0) {{ return some(n); }}\n"
            f"    return none;\n"
            f"}}"
        )
        main = (
            f"    r := {name}({n});\n"
            f"    v: i64 = match r {{\n"
            f"        some(x) => x,\n"
            f"        none => -1,\n"
            f"    }};\n"
            f"    println_i64(v);"
        )
        return self._wrap(helper, main, f"fn {name}(n: i64) -> ?i64", str(expected), name)

    def _gen_array_sum(self) -> MogSample:
        arr = [self._ri(1, 9) for _ in range(self._ri(3, 6))]
        arr_code = ", ".join(map(str, arr))
        expected = sum(arr)
        name = self._safe_name("sum_array")
        helper = (
            f"fn {name}(arr: []i64) -> i64 {{\n"
            f"    total: i64 = 0;\n"
            f"    for item in arr {{\n"
            f"        total = total + item;\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
        main = f"    println_i64({name}([{arr_code}]));"
        return self._wrap(helper, main, f"fn {name}(arr: []i64) -> i64", str(expected), name)

    def _gen_array_push(self) -> MogSample:
        arr = [self._ri(1, 9) for _ in range(self._ri(2, 4))]
        extra = self._ri(1, 9)
        expected = len(arr) + 1
        arr_code = ", ".join(map(str, arr))
        name = self._safe_name("push_and_len")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    nums := [{arr_code}];\n"
            f"    nums.push({extra});\n"
            f"    return nums.len;\n"
            f"}}"
        )
        main = f"    println_i64({name}());"
        return self._wrap(helper, main, f"fn {name}() -> i64", str(expected), name)

    def _gen_closure_map(self) -> MogSample:
        arr = [self._ri(1, 5) for _ in range(3)]
        mapped = [x * 2 for x in arr]
        expected = "\n".join(str(x) for x in mapped)
        arr_code = ", ".join(map(str, arr))
        name = self._safe_name("double_all")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    nums := [{arr_code}];\n"
            f"    doubled := nums.map(fn(x: i64) -> i64 {{ x * 2 }});\n"
            f"    for item in doubled {{\n"
            f"        println_i64(item);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}"
        )
        main = f"    {name}();"
        return self._wrap(helper, main, f"fn {name}() -> i64", expected, name)

    def _gen_struct_sum(self) -> MogSample:
        x, y = self._ri(1, 20), self._ri(1, 20)
        expected = x + y
        sname = self._safe_name("Point").title().replace("_", "")
        fname = self._safe_name("sum_point")
        helper = (
            f"struct {sname} {{\n"
            f"    x: i64,\n"
            f"    y: i64,\n"
            f"}}\n\n"
            f"fn {fname}(p: {sname}) -> i64 {{\n"
            f"    return p.x + p.y;\n"
            f"}}"
        )
        main = (
            f"    p := {sname} {{ x: {x}, y: {y} }};\n"
            f"    println_i64({fname}(p));"
        )
        return self._wrap(helper, main, f"fn {fname}(p: {sname}) -> i64", str(expected), fname)

    def _gen_struct_mutation(self) -> MogSample:
        x, y = self._ri(1, 10), self._ri(1, 10)
        inc = self._ri(1, 5)
        expected = x + inc + y
        sname = self._safe_name("Counter").title().replace("_", "")
        helper = (
            f"struct {sname} {{\n"
            f"    x: i64,\n"
            f"    y: i64,\n"
            f"}}\n\n"
            f"fn bump(c: {sname}) -> {sname} {{\n"
            f"    c.x = c.x + {inc};\n"
            f"    return c;\n"
            f"}}"
        )
        main = (
            f"    c := {sname} {{ x: {x}, y: {y} }};\n"
            f"    c = bump(c);\n"
            f"    println_i64(c.x + c.y);"
        )
        return self._wrap(helper, main, f"struct {sname} mutation", str(expected), f"bump_{sname.lower()}")

    def _gen_nested_loops(self) -> MogSample:
        n = self._ri(2, 5)
        total = sum(i * j for i in range(n) for j in range(n))
        name = self._safe_name("nested_sum")
        helper = (
            f"fn {name}(n: i64) -> i64 {{\n"
            f"    total: i64 = 0;\n"
            f"    for i := 0 to n {{\n"
            f"        for j := 0 to n {{\n"
            f"            total = total + (i * j);\n"
            f"        }}\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
        main = f"    println_i64({name}({n}));"
        return self._wrap(helper, main, f"fn {name}(n: i64) -> i64", str(total), name)

    def _gen_pipeline(self) -> MogSample:
        x = self._ri(1, 20)
        expected = (x * 2) + 10
        name = self._safe_name("pipeline")
        helper = (
            f"fn step1(x: i64) -> i64 {{ return x * 2; }}\n\n"
            f"fn step2(x: i64) -> i64 {{ return x + 10; }}\n\n"
            f"fn {name}(input: i64) -> i64 {{\n"
            f"    a := step1(input);\n"
            f"    b := step2(a);\n"
            f"    return b;\n"
            f"}}"
        )
        main = f"    println_i64({name}({x}));"
        return self._wrap(helper, main, f"fn {name}(input: i64) -> i64", str(expected), name)

    def _gen_validator(self) -> MogSample:
        lo = self._ri(0, 10)
        hi = self._ri(20, 50)
        val = self._ri(lo, hi)
        name = self._safe_name("validate")
        helper = (
            f"fn {name}(val: i64) -> Result<i64> {{\n"
            f"    if val < {lo} {{ return err(\"too small\"); }}\n"
            f"    if val > {hi} {{ return err(\"too large\"); }}\n"
            f"    return ok(val);\n"
            f"}}"
        )
        main = (
            f"    r := {name}({val});\n"
            f"    out: i64 = match r {{\n"
            f"        ok(x) => x,\n"
            f"        err(e) => -1,\n"
            f"    }};\n"
            f"    println_i64(out);"
        )
        return self._wrap(helper, main, f"fn {name}(val: i64) -> Result<i64>", str(val), name)

    def _gen_map_iteration(self) -> MogSample:
        a, b = self._ri(1, 9), self._ri(1, 9)
        expected = a + b
        name = self._safe_name("sum_map_vals")
        helper = (
            f"fn {name}() -> i64 {{\n"
            f"    m := {{\"a\": {a}, \"b\": {b}}};\n"
            f"    total: i64 = 0;\n"
            f"    for key, value in m {{\n"
            f"        total = total + value;\n"
            f"    }}\n"
            f"    return total;\n"
            f"}}"
        )
        main = f"    println_i64({name}());"
        return self._wrap(helper, main, f"fn {name}() -> i64", str(expected), name)

    def _gen_string_len(self) -> MogSample:
        s = self.rng.choice(["mog", "compiler", "diffusion", "interpreter", "hello world"])
        expected = len(s.strip())
        name = self._safe_name("string_len")
        helper = (
            f"fn {name}(s: string) -> i64 {{\n"
            f"    trimmed := s.trim();\n"
            f"    return trimmed.len;\n"
            f"}}"
        )
        main = f'    println_i64({name}(" {s} "));'
        return self._wrap(helper, main, f"fn {name}(s: string) -> i64", str(expected), name)

    def _gen_match_literal(self) -> MogSample:
        code_val = self.rng.choice([0, 1, 2, 7])
        mapping = {0: 100, 1: 200, 2: 300}
        expected = mapping.get(code_val, 999)
        name = self._safe_name("classify_code")
        helper = (
            f"fn {name}(code: i64) -> i64 {{\n"
            f"    result: i64 = match code {{\n"
            f"        0 => 100,\n"
            f"        1 => 200,\n"
            f"        2 => 300,\n"
            f"        _ => 999,\n"
            f"    }};\n"
            f"    return result;\n"
            f"}}"
        )
        main = f"    println_i64({name}({code_val}));"
        return self._wrap(helper, main, f"fn {name}(code: i64) -> i64", str(expected), name)

    def generate_one(self) -> Tuple[str, str, str]:
        sample = self.rng.choice(self._templates)()
        return sample.code, sample.spec, sample.expected_output

    def generate_sample(self) -> MogSample:
        return self.rng.choice(self._templates)()

    def generate_dataset(self, num_samples: int, balanced: bool = True) -> List[Tuple[str, str, str]]:
        data: List[Tuple[str, str, str]] = []
        if balanced:
            for i in range(num_samples):
                sample = self._templates[i % len(self._templates)]()
                data.append((sample.code, sample.spec, sample.expected_output))
        else:
            for _ in range(num_samples):
                data.append(self.generate_one())
        return data


class MogDataset(Dataset):
    """PyTorch dataset for compiler-safe Mog program diffusion training."""

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
        self.data: List[Tuple[List[int], List[int], str, str]] = []
        for code, spec, expected_output in raw:
            code_tokens = self.tokenizer.encode(code)
            spec_tokens = self.tokenizer.encode(spec, add_bos_eos=False)
            self.data.append((code_tokens, spec_tokens, expected_output, code))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        code_tokens, spec_tokens, _expected_output, _code = self.data[idx]

        prog = self.tokenizer.pad(list(code_tokens), self.seq_len)
        original_tokens = torch.tensor(prog, dtype=torch.long)

        spec = self.tokenizer.pad(list(spec_tokens), self.spec_len)
        spec_tensor = torch.tensor(spec, dtype=torch.long)

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

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def decode_program(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids.tolist())

    def get_expected_output(self, idx: int) -> str:
        return self.data[idx][2]

    def get_code(self, idx: int) -> str:
        return self.data[idx][3]

    def get_example(self, idx: int) -> MogSample:
        code_tokens, spec_tokens, expected, code = self.data[idx]
        spec = self.tokenizer.decode(spec_tokens, skip_special=True)
        return MogSample(code=code, spec=spec, expected_output=expected, name=f"sample_{idx}")

    def get_dataloader(self, batch_size: int = 64, shuffle: bool = True,
                       num_workers: int = 0, **kwargs):
        from torch.utils.data import DataLoader
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
