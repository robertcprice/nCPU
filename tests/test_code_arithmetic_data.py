import re

from ncpu.coprocessor.code_arithmetic_data import CodeArithmeticGenerator, CodePattern


_RANGE_RE = re.compile(r"range\((\d+), (\d+)(?:, (\d+))?\)")
_BOUNDARY_RE = re.compile(r"< (\d+) - (\d+)")
_SHIFT_RE = re.compile(r"1 << (\d+)")
_ARRAY_COMPOUND_RE = re.compile(r"(\d+) \+ (\d+) \* (\d+)")
_ARRAY_ADD_RE = re.compile(r"(\d+) \+ (\d+)")
_ARRAY_MUL_RE = re.compile(r"(\d+) \* (\d+)")
_BITWISE_RE = re.compile(r"(\d+) ([&|^]) (\d+)")
_ACC_ADD_RE = re.compile(r"total = (\d+)\ntotal \+= (\d+)")
_ACC_MUL_RE = re.compile(r"count = (\d+)\ncount \*= (\d+)")
_ACC_MOD_RE = re.compile(r"index = \((\d+) \+ 1\) % (\d+)")
_SLICE_ADD_RE = re.compile(r"\[(\d+):\1 \+ (\d+)\]")
_SLICE_DIRECT_RE = re.compile(r"\[(\d+):(\d+)\]")


def _infer_result_from_code(sample) -> int:
    code = sample.code_snippet

    if sample.pattern == CodePattern.ARRAY_INDEX:
        if match := _ARRAY_COMPOUND_RE.search(code):
            base, channel, stride = map(int, match.groups())
            return base + channel * stride
        if match := _ARRAY_ADD_RE.search(code):
            a, b = map(int, match.groups())
            return a + b
        if match := _ARRAY_MUL_RE.search(code):
            a, b = map(int, match.groups())
            return a * b

    if sample.pattern == CodePattern.LOOP_BOUND:
        match = _RANGE_RE.search(code)
        assert match is not None, code
        start, end, step = match.groups()
        return len(range(int(start), int(end), int(step or 1)))

    if sample.pattern == CodePattern.BOUNDARY_CHECK:
        match = _BOUNDARY_RE.search(code)
        assert match is not None, code
        bound, offset = map(int, match.groups())
        return bound - offset

    if sample.pattern == CodePattern.BIT_OPERATION:
        if match := _SHIFT_RE.search(code):
            return 1 << int(match.group(1))
        match = _BITWISE_RE.search(code)
        assert match is not None, code
        a, op, b = match.groups()
        a = int(a)
        b = int(b)
        return {
            "&": a & b,
            "|": a | b,
            "^": a ^ b,
        }[op]

    if sample.pattern == CodePattern.ACCUMULATOR:
        if match := _ACC_ADD_RE.search(code):
            current, addend = map(int, match.groups())
            return current + addend
        if match := _ACC_MUL_RE.search(code):
            current, factor = map(int, match.groups())
            return current * factor
        match = _ACC_MOD_RE.search(code)
        assert match is not None, code
        current, modulus = map(int, match.groups())
        return (current + 1) % modulus

    if sample.pattern == CodePattern.SLICE_OPERATION:
        if match := _SLICE_ADD_RE.search(code):
            start, length = map(int, match.groups())
            return start + length
        match = _SLICE_DIRECT_RE.search(code)
        assert match is not None, code
        _, end = map(int, match.groups())
        return end

    raise AssertionError(f"Unhandled sample pattern: {sample.pattern}")


def test_generated_code_samples_match_embedded_result():
    generator = CodeArithmeticGenerator(seed=123, max_value=64)

    for sample in generator.generate(250):
        assert _infer_result_from_code(sample) == sample.result, sample
