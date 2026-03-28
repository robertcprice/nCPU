"""Procedural generator of (spec, program) pairs for nCPU ISA.

Generates programs from 20+ template families across three categories:
  - Arithmetic: add, sub, mul, add3, double, square
  - Conditional: abs, max, min, clamp, sign, relu, isZero
  - Iterative: sum_1_to_n, factorial, fibonacci, power, gcd, countDown,
               mulByAdd, divBySubtract

Each template is instantiated with varying constants and register assignments
to produce 100K+ unique (spec, program_text) pairs.

A spec is a dict with:
    {"name": str, "test_cases": [{"inputs": dict, "expected_output": int}, ...]}
"""

from __future__ import annotations
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from egdc.tokenizer import NCPUTokenizer


@dataclass
class NCPUSpec:
    """Specification for a program: name + test cases."""
    name: str
    test_cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "test_cases": self.test_cases}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NCPUSpec":
        return NCPUSpec(name=d["name"], test_cases=d["test_cases"])


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

_TEMPLATES: List[Tuple[str, Callable]] = []


def _register(name: str):
    """Decorator to register a program template generator."""
    def decorator(fn: Callable):
        _TEMPLATES.append((name, fn))
        return fn
    return decorator


# Helpers
def _clamp8(v: int) -> int:
    """Clamp to unsigned 8-bit (0..255)."""
    return max(0, min(255, v))


def _clamp8s(v: int) -> int:
    """Clamp to signed-ish 8-bit for results (keep in 0..255)."""
    return v & 0xFF


# ---------------------------------------------------------------------------
# ARITHMETIC templates
# ---------------------------------------------------------------------------

@_register("add")
def _gen_add(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 127), rng.randint(0, 127)
    result = (a + b) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
ADD R0 R1
HALT"""
    # Generate multiple test cases with same structure
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 127), rng.randint(0, 127)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": (ta + tb) & 0xFF})
    return NCPUSpec("add", cases), prog


@_register("sub")
def _gen_sub(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a = rng.randint(50, 255)
    b = rng.randint(0, min(a, 255))
    result = (a - b) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
SUB R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(50, 255)
        tb = rng.randint(0, min(ta, 255))
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": (ta - tb) & 0xFF})
    return NCPUSpec("sub", cases), prog


@_register("mul")
def _gen_mul(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 15), rng.randint(0, 15)
    result = (a * b) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
MUL R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 15), rng.randint(0, 15)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": (ta * tb) & 0xFF})
    return NCPUSpec("mul", cases), prog


@_register("add3")
def _gen_add3(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Add three numbers: a + b + c."""
    a, b, c = rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80)
    result = (a + b + c) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
ADD R0 R1
MOV_IMM R1 {c}
ADD R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b, "c": c}, "expected_output": result}]
    for _ in range(3):
        ta, tb, tc = rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80)
        cases.append({"inputs": {"a": ta, "b": tb, "c": tc},
                       "expected_output": (ta + tb + tc) & 0xFF})
    return NCPUSpec("add3", cases), prog


@_register("double")
def _gen_double(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a = rng.randint(0, 127)
    result = (a * 2) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_REG R1 R0
ADD R0 R1
HALT"""
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 127)
        cases.append({"inputs": {"a": ta}, "expected_output": (ta * 2) & 0xFF})
    return NCPUSpec("double", cases), prog


@_register("square")
def _gen_square(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a = rng.randint(0, 15)
    result = (a * a) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_REG R1 R0
MUL R0 R1
HALT"""
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 15)
        cases.append({"inputs": {"a": ta}, "expected_output": (ta * ta) & 0xFF})
    return NCPUSpec("square", cases), prog


@_register("bitwise_and")
def _gen_bitwise_and(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = a & b
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
AND R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": ta & tb})
    return NCPUSpec("bitwise_and", cases), prog


@_register("bitwise_or")
def _gen_bitwise_or(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = a | b
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
OR R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": ta | tb})
    return NCPUSpec("bitwise_or", cases), prog


@_register("bitwise_xor")
def _gen_bitwise_xor(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = a ^ b
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
XOR R0 R1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": ta ^ tb})
    return NCPUSpec("bitwise_xor", cases), prog


# ---------------------------------------------------------------------------
# CONDITIONAL templates
# ---------------------------------------------------------------------------

@_register("abs")
def _gen_abs(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """abs(a) where a is treated as signed: if a > 127, result = 256 - a."""
    a = rng.randint(0, 255)
    result = a if a <= 127 else (256 - a)
    # Compare a with 128: if a < 128 (not greater), skip negation
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 128
CMP R0 R1
BGT 6
BEQ 6
MOV_IMM R1 0
SUB R1 R0
MOV_REG R0 R1
HALT"""
    # Instruction indices: 0=MOV_IMM, 1=MOV_IMM, 2=CMP, 3=BGT, 4=BEQ, 5=MOV_IMM, 6=SUB, 7=MOV_REG, 8=HALT
    # If a >= 128: a > 128 -> BGT jumps to 6 (skip negation, but we want negation!)
    # Actually let's rethink: if a <= 127, no negation needed. If a >= 128, negate.
    # CMP R0 R1 compares R0 with R1 (128). If R0 < 128, not greater, not equal -> fall through to negate... wrong.
    # Let me fix the logic: branch PAST negation if a < 128
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 127
CMP R1 R0
BGT 7
MOV_IMM R2 0
SUB R2 R0
MOV_REG R0 R2
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,127  2=CMP R1,R0  3=BGT 7  4=MOV R2,0  5=SUB R2,R0  6=MOV R0,R2  7=HALT
    # CMP R1,R0: if 127 > a -> BGT 7 (jump to HALT, no negation) = a <= 127 case correct
    # if 127 <= a -> fall through, negate
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": ta if ta <= 127 else (256 - ta)})
    return NCPUSpec("abs", cases), prog


@_register("max")
def _gen_max(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = max(a, b)
    # if a > b: result = a, else result = b
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BGT 5
MOV_REG R0 R1
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,b  2=CMP R0,R1  3=BGT 5  4=MOV R0,R1  5=HALT
    # if a > b -> jump to HALT (R0 already has a)
    # else -> MOV R0, R1 (R0 = b), then HALT
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": max(ta, tb)})
    return NCPUSpec("max", cases), prog


@_register("min")
def _gen_min(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = min(a, b)
    # if a > b: result = b (R1), else result = a (R0)
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BGT 5
HALT
MOV_REG R0 R1
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,b  2=CMP  3=BGT 5  4=HALT(a<=b, R0=a)  5=MOV R0,R1  6=HALT
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": min(ta, tb)})
    return NCPUSpec("min", cases), prog


@_register("clamp")
def _gen_clamp(rng: random.Random) -> Tuple[NCPUSpec, str]:
    lo = rng.randint(0, 50)
    hi = rng.randint(lo + 1, 200)
    a = rng.randint(0, 255)
    result = max(lo, min(hi, a))
    # clamp(a, lo, hi): if a < lo -> lo; if a > hi -> hi; else a
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {lo}
MOV_IMM R2 {hi}
CMP R1 R0
BGT 7
CMP R0 R2
BGT 8
HALT
MOV_REG R0 R2
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,lo  2=MOV R2,hi  3=CMP R1,R0 (lo vs a)
    # 4=BGT 7 (if lo > a, jump to 7)  5=CMP R0,R2 (a vs hi)
    # 6=BGT 8 (if a > hi, jump to 8)  7=HALT (a in range or after fix)
    # Wait, if lo > a, we need R0 = lo. Let me fix:
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {lo}
MOV_IMM R2 {hi}
CMP R1 R0
BGT 8
CMP R0 R2
BGT 9
HALT
MOV_REG R0 R1
HALT
MOV_REG R0 R2
HALT"""
    # idx: 0-2=MOVs, 3=CMP lo,a, 4=BGT 8 (lo>a -> set R0=lo), 5=CMP a,hi
    # 6=BGT 9 (a>hi -> set R0=hi), 7=HALT(in range), 8=MOV R0,R1, 9=HALT, 10=MOV R0,R2, 11=HALT
    # Wait indices: BGT 8 means jump to instruction 8 which is MOV R0,R1. Then 9=HALT. Good.
    # BGT 9 means jump to instruction 9 which is HALT... that's wrong, should be 10=MOV R0,R2
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {lo}
MOV_IMM R2 {hi}
CMP R1 R0
BGT 8
CMP R0 R2
BGT 10
HALT
MOV_REG R0 R1
HALT
MOV_REG R0 R2
HALT"""
    cases = [{"inputs": {"a": a, "lo": lo, "hi": hi}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "lo": lo, "hi": hi},
                       "expected_output": max(lo, min(hi, ta))})
    return NCPUSpec("clamp", cases), prog


@_register("sign")
def _gen_sign(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """sign(a): 0 if a==0, 1 if a>0 (treating as unsigned, a in 1..255)."""
    a = rng.randint(0, 255)
    result = 0 if a == 0 else 1
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 0
CMP R0 R1
BEQ 5
MOV_IMM R0 1
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,0  2=CMP R0,R1  3=BEQ 5(if a==0, jump to HALT)
    # 4=MOV R0,1  5=HALT
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": 0 if ta == 0 else 1})
    return NCPUSpec("sign", cases), prog


@_register("relu")
def _gen_relu(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """relu(a): if a >= 128 (negative in signed), return 0, else return a."""
    a = rng.randint(0, 255)
    result = 0 if a >= 128 else a
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 127
CMP R1 R0
BGT 5
MOV_IMM R0 0
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,127  2=CMP R1,R0 (127 vs a)
    # 3=BGT 5: if 127>a means a<128, jump to HALT (keep a)
    # 4=MOV R0,0 (a>=128, set 0)  5=HALT
    # Wait: if 127 > a, that means a < 127, so a is positive -> jump to HALT, keep a. Correct.
    # If 127 <= a (a >= 127... but 127 is still positive). Hmm, edge case at 127.
    # Actually let's use 128 as threshold:
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 128
CMP R1 R0
BGT 5
MOV_IMM R0 0
HALT"""
    # CMP R1(128), R0(a): if 128 > a -> a < 128 -> positive -> BGT jumps to HALT
    # else a >= 128 -> fall through -> R0 = 0
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": 0 if ta >= 128 else ta})
    return NCPUSpec("relu", cases), prog


@_register("isZero")
def _gen_is_zero(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a = rng.randint(0, 255)
    result = 1 if a == 0 else 0
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 0
MOV_IMM R2 1
CMP R0 R1
BEQ 6
MOV_IMM R0 0
HALT
MOV_IMM R0 1
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,0  2=MOV R2,1  3=CMP R0,R1
    # 4=BEQ 7: if a==0, jump to 7 (MOV R0,1)  5=MOV R0,0  6=HALT
    # 7=MOV R0,1  8=HALT
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 0
CMP R0 R1
BEQ 6
MOV_IMM R0 0
HALT
MOV_IMM R0 1
HALT"""
    # idx: 0=MOV R0,a  1=MOV R1,0  2=CMP  3=BEQ 6  4=MOV R0,0  5=HALT  6=MOV R0,1  7=HALT
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": 1 if ta == 0 else 0})
    return NCPUSpec("isZero", cases), prog


# ---------------------------------------------------------------------------
# ITERATIVE templates
# ---------------------------------------------------------------------------

@_register("sum_1_to_n")
def _gen_sum_1_to_n(rng: random.Random) -> Tuple[NCPUSpec, str]:
    n = rng.randint(1, 20)
    result = (n * (n + 1) // 2) & 0xFF
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 1
MOV_IMM R2 {n}
ADD R0 R1
MOV_IMM R3 1
ADD R1 R3
CMP R1 R2
BGT 8
BEQ 3
BNE 3
HALT"""
    # idx: 0=MOV R0,0(sum)  1=MOV R1,1(i)  2=MOV R2,n
    # 3=ADD R0,R1 (sum+=i)  4=MOV R3,1  5=ADD R1,R3 (i++)
    # 6=CMP R1,R2  7=BGT 8(if i>n, done) -- but we also need i==n case
    # Actually: we add i, then increment. Loop: add i to sum, i++, if i > n stop.
    # Wait, we already added current i before incrementing, so when i becomes n+1 and i>n, we stop.
    # But we need to include n in the sum. Let's trace: i starts at 1.
    # Iter1: sum+=1, i=2, 2>n? Iter2: sum+=2, i=3...  When i=n: sum+=n, i=n+1, n+1>n -> HALT. Correct!
    # But BEQ 3 after BGT means if i==n, loop again. And BNE 3 if i<n loop again.
    # Actually after BGT, if not taken, we fall to BEQ. If i==n, BEQ 3 loops. If i<n, fall to BNE 3 which loops.
    # This wastes an instruction. Simpler:
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 1
MOV_IMM R2 {n}
MOV_IMM R3 1
ADD R0 R1
ADD R1 R3
CMP R2 R1
BGT 4
HALT"""
    # idx: 0=R0=0(sum) 1=R1=1(i) 2=R2=n 3=R3=1(inc)
    # 4=ADD R0,R1(sum+=i) 5=ADD R1,R3(i++) 6=CMP R2,R1(n vs i) 7=BGT 4(if n>i, loop) 8=HALT
    # Trace for n=3: sum=0,i=1. sum+=1=1,i=2, 3>2->loop. sum+=2=3,i=3, 3>3? No->HALT. sum=3.
    # But sum should be 1+2+3=6. We miss i=3! Because when i becomes 3, CMP says 3>3=false, halt.
    # Need >= : use BGT or BEQ. Let's add BEQ:
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 1
MOV_IMM R2 {n}
MOV_IMM R3 1
ADD R0 R1
ADD R1 R3
CMP R2 R1
BGT 4
BEQ 4
HALT"""
    # Now: i=3, CMP R2(3),R1(3): not greater, but equal -> BEQ 4 loops.
    # sum+=3=6, i=4, CMP 3,4: not greater, not equal -> HALT. sum=6. Correct!
    # But actually we need to also add the n case. Let me re-trace n=3:
    # i=1: sum=0+1=1, i=2, 3>2 yes -> loop
    # i=2: sum=1+2=3, i=3, 3>3 no, 3==3 yes -> loop
    # i=3: sum=3+3=6, i=4, 3>4 no, 3==4 no -> HALT. sum=6. Correct!
    cases = [{"inputs": {"n": n}, "expected_output": result}]
    for _ in range(3):
        tn = rng.randint(1, 20)
        cases.append({"inputs": {"n": tn}, "expected_output": (tn * (tn + 1) // 2) & 0xFF})
    return NCPUSpec("sum_1_to_n", cases), prog


@_register("factorial")
def _gen_factorial(rng: random.Random) -> Tuple[NCPUSpec, str]:
    n = rng.randint(1, 5)  # Keep small to fit in 8 bits
    result = 1
    for i in range(2, n + 1):
        result = (result * i) & 0xFF
    # R0=result, R1=counter(starts at 2), R2=n, R3=1(increment)
    prog = f"""\
MOV_IMM R0 1
MOV_IMM R1 2
MOV_IMM R2 {n}
MOV_IMM R3 1
MUL R0 R1
ADD R1 R3
CMP R2 R1
BGT 4
BEQ 4
HALT"""
    # idx: 0=R0=1 1=R1=2 2=R2=n 3=R3=1
    # 4=MUL R0,R1 5=ADD R1,R3(i++) 6=CMP R2,R1 7=BGT 4 8=BEQ 4 9=HALT
    cases = [{"inputs": {"n": n}, "expected_output": result}]
    for _ in range(3):
        tn = rng.randint(1, 5)
        tr = 1
        for i in range(2, tn + 1):
            tr = (tr * i) & 0xFF
        cases.append({"inputs": {"n": tn}, "expected_output": tr})
    return NCPUSpec("factorial", cases), prog


@_register("fibonacci")
def _gen_fibonacci(rng: random.Random) -> Tuple[NCPUSpec, str]:
    n = rng.randint(2, 13)  # fib(13)=233 fits in 8 bits
    # fib(0)=0, fib(1)=1, fib(2)=1, ...
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, (a + b) & 0xFF
    result = b & 0xFF
    # R0=a, R1=b, R2=counter, R3=n, R4=temp
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 1
MOV_IMM R2 1
MOV_IMM R3 {n}
MOV_REG R4 R1
ADD R1 R0
MOV_REG R0 R4
MOV_IMM R5 1
ADD R2 R5
CMP R3 R2
BGT 4
BEQ 4
MOV_REG R0 R1
HALT"""
    # idx: 0=R0=0, 1=R1=1, 2=R2=1(counter), 3=R3=n
    # 4=R4=R1(save b), 5=R1=R1+R0(new b=a+b), 6=R0=R4(a=old b)
    # 7=R5=1, 8=R2++, 9=CMP R3,R2, 10=BGT 4, 11=BEQ 4
    # 12=MOV R0,R1(result in R0), 13=HALT
    cases = [{"inputs": {"n": n}, "expected_output": result}]
    for _ in range(3):
        tn = rng.randint(2, 13)
        ta, tb = 0, 1
        for _ in range(tn - 1):
            ta, tb = tb, (ta + tb) & 0xFF
        cases.append({"inputs": {"n": tn}, "expected_output": tb & 0xFF})
    return NCPUSpec("fibonacci", cases), prog


@_register("power")
def _gen_power(rng: random.Random) -> Tuple[NCPUSpec, str]:
    base = rng.randint(2, 5)
    exp = rng.randint(1, 5)
    result = (base ** exp) & 0xFF
    # R0=result(starts 1), R1=base, R2=counter(starts 0), R3=exp, R4=1
    prog = f"""\
MOV_IMM R0 1
MOV_IMM R1 {base}
MOV_IMM R2 0
MOV_IMM R3 {exp}
MOV_IMM R4 1
MUL R0 R1
ADD R2 R4
CMP R3 R2
BGT 5
BEQ 5
HALT"""
    # idx: 0-4=init, 5=MUL R0,R1, 6=R2++, 7=CMP R3,R2, 8=BGT 5, 9=BEQ 5, 10=HALT
    cases = [{"inputs": {"base": base, "exp": exp}, "expected_output": result}]
    for _ in range(3):
        tb, te = rng.randint(2, 5), rng.randint(1, 5)
        cases.append({"inputs": {"base": tb, "exp": te}, "expected_output": (tb ** te) & 0xFF})
    return NCPUSpec("power", cases), prog


@_register("gcd")
def _gen_gcd(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """GCD using subtraction: while a != b: if a > b: a -= b else b -= a."""
    import math
    a = rng.randint(1, 100)
    b = rng.randint(1, 100)
    result = math.gcd(a, b)
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BEQ 8
BGT 6
SUB R1 R0
BNE 2
SUB R0 R1
BNE 2
HALT"""
    # idx: 0=R0=a 1=R1=b 2=CMP R0,R1 3=BEQ 8(done, R0=gcd)
    # 4=BGT 6(if a>b goto 6) 5=SUB R1,R0(b-=a) -> 6 is wrong, should loop
    # Let me redo:
    # 0: MOV_IMM R0 a
    # 1: MOV_IMM R1 b
    # 2: CMP R0 R1     -- compare a, b
    # 3: BEQ 9         -- if equal, done
    # 4: BGT 7         -- if a > b, goto subtract a
    # 5: SUB R1 R0     -- b -= a
    # 6: BNE 2         -- loop (always taken since we know a!=b from BEQ check... 
    #                     actually after subtraction they might be equal, but BNE checks last CMP)
    # Hmm, BNE checks the flags from the last CMP. After SUB, flags aren't set.
    # We need CMP again. Let me use explicit CMP:
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BEQ 8
BGT 6
SUB R1 R0
BNE 2
SUB R0 R1
BNE 2
HALT"""
    # The BNE 2 always branches because the last CMP showed they weren't equal (we passed BEQ).
    # After SUB, we branch back to CMP which re-evaluates.
    # idx: 0=R0=a, 1=R1=b, 2=CMP, 3=BEQ 8, 4=BGT 6, 5=SUB R1,R0, 6=BNE 2 -- wait, BGT 6 goes to idx 6
    # Let me recount: instruction 6 is BNE 2 (after SUB R1,R0). BGT 6 jumps there.
    # That's wrong - if a>b we should subtract b from a. Let me be more careful:
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BEQ 9
BGT 7
SUB R1 R0
BNE 2
SUB R0 R1
BNE 2
HALT"""
    # 0: MOV_IMM R0 a
    # 1: MOV_IMM R1 b  
    # 2: CMP R0 R1
    # 3: BEQ 9  (if a==b, jump to HALT)
    # 4: BGT 7  (if a>b, jump to instr 7 = SUB R0 R1)
    # 5: SUB R1 R0  (b -= a, case a < b)
    # 6: BNE 2  (loop back to CMP; always taken since last CMP was not equal)
    # 7: SUB R0 R1  (a -= b, case a > b)
    # 8: BNE 2  (loop back)
    # 9: HALT
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(1, 100), rng.randint(1, 100)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": math.gcd(ta, tb)})
    return NCPUSpec("gcd", cases), prog


@_register("countDown")
def _gen_countdown(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Count down from n to 0, result = 0."""
    n = rng.randint(1, 50)
    prog = f"""\
MOV_IMM R0 {n}
MOV_IMM R1 1
MOV_IMM R2 0
SUB R0 R1
CMP R0 R2
BGT 3
HALT"""
    # 0=R0=n, 1=R1=1, 2=R2=0, 3=SUB R0,R1(n--), 4=CMP R0,R2, 5=BGT 3(if n>0 loop), 6=HALT
    cases = [{"inputs": {"n": n}, "expected_output": 0}]
    for _ in range(3):
        tn = rng.randint(1, 50)
        cases.append({"inputs": {"n": tn}, "expected_output": 0})
    return NCPUSpec("countDown", cases), prog


@_register("mulByAdd")
def _gen_mul_by_add(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Multiply a * b using repeated addition."""
    a = rng.randint(1, 15)
    b = rng.randint(1, 15)
    result = (a * b) & 0xFF
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 {a}
MOV_IMM R2 {b}
MOV_IMM R3 1
MOV_IMM R4 0
ADD R0 R1
ADD R4 R3
CMP R2 R4
BGT 5
BEQ 5
HALT"""
    # 0=R0=0(result), 1=R1=a, 2=R2=b, 3=R3=1, 4=R4=0(counter)
    # 5=ADD R0,R1(result+=a), 6=ADD R4,R3(counter++), 7=CMP R2,R4(b vs counter)
    # 8=BGT 5, 9=BEQ 5, 10=HALT
    # counter goes 1,2,...,b then CMP b,b -> BEQ loops one more time -> counter=b+1, b>b+1 no, b==b+1 no -> HALT
    # Wait: counter starts at 0. After first iter: counter=1, added once. 
    # CMP b,1: if b>1 loop. Eventually counter=b, CMP b,b: not greater, but equal -> BEQ 5 loops.
    # counter=b+1, CMP b,b+1: not greater, not equal -> HALT. Result = a*(b+1). That's wrong!
    # Fix: don't use BEQ, just BGT, and start counter at 1:
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 {a}
MOV_IMM R2 {b}
MOV_IMM R3 1
MOV_IMM R4 0
ADD R0 R1
ADD R4 R3
CMP R2 R4
BGT 5
HALT"""
    # counter goes 0->1->2->...->b. At counter=b, CMP b,b: not greater -> HALT.
    # We added a exactly b times (iterations at counter 0,1,...,b-1 add, then at counter=b we exit).
    # Wait: 5=ADD R0,R1(add first), 6=R4++(counter becomes 1), 7=CMP b,1: b>1->loop
    # 2nd: ADD(total=2a), counter=2, CMP b,2: ...
    # When counter=b: CMP b,b not greater -> HALT. We did b additions. Correct!
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(1, 15), rng.randint(1, 15)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": (ta * tb) & 0xFF})
    return NCPUSpec("mulByAdd", cases), prog


@_register("divBySubtract")
def _gen_div_by_subtract(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Integer division a / b using repeated subtraction. Result = quotient."""
    b = rng.randint(1, 20)
    a = rng.randint(b, 200)
    result = a // b
    # R0=a(dividend), R1=b(divisor), R2=0(quotient), R3=1
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
MOV_IMM R2 0
MOV_IMM R3 1
CMP R0 R1
BGT 7
BEQ 7
MOV_REG R0 R2
HALT
SUB R0 R1
ADD R2 R3
BNE 4
HALT"""
    # 0=R0=a, 1=R1=b, 2=R2=0, 3=R3=1
    # 4=CMP R0,R1, 5=BGT 7(a>b, go subtract), 6=BEQ 7(a==b, go subtract)
    # If a<b: fall through to 7... wait that's the subtract. Let me fix:
    # 4=CMP R0,R1
    # If a >= b: subtract. If a < b: done, result in R2.
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
MOV_IMM R2 0
MOV_IMM R3 1
CMP R0 R1
BGT 9
BEQ 9
MOV_REG R0 R2
HALT
SUB R0 R1
ADD R2 R3
BNE 4
HALT"""
    # 0-3=init, 4=CMP, 5=BGT 9, 6=BEQ 9, 7=MOV R0,R2(result), 8=HALT
    # 9=SUB R0,R1, 10=ADD R2,R3(quotient++), 11=BNE 4(loop; BNE uses CMP flags which said >=, so not equal might not work right)
    # Hmm, BNE checks flags from CMP R0,R1 which was before subtract. After subtract flags don't change.
    # So BNE 4: if the last CMP said not equal (which it could be either BGT or BEQ path)...
    # From BGT path: a>b, so not equal -> BNE takes the branch. Good.
    # From BEQ path: a==b, equal -> BNE does NOT branch, falls to HALT. That means for a==b case, we subtract once, quotient=1, then fall through to HALT at 12. But R0 still has the subtracted value, not R2.
    # This is getting complicated. Let me simplify - always jump back unconditionally using a trick:
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
MOV_IMM R2 0
MOV_IMM R3 1
SUB R0 R1
ADD R2 R3
CMP R0 R1
BGT 4
BEQ 4
MOV_REG R0 R2
HALT"""
    # 0-3=init, 4=SUB R0,R1, 5=ADD R2,R3, 6=CMP R0,R1, 7=BGT 4, 8=BEQ 4, 9=MOV R0,R2, 10=HALT
    # Subtract first, then check. For a=10,b=3: 10-3=7,q=1, 7>3->loop, 7-3=4,q=2, 4>3->loop, 4-3=1,q=3, 1<3->done. Result=3. 10//3=3. Correct!
    # For a=6,b=3: 6-3=3,q=1, 3==3->loop, 3-3=0,q=2, 0<3->done. Result=2. 6//3=2. Correct!
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        tb = rng.randint(1, 20)
        ta = rng.randint(tb, 200)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": ta // tb})
    return NCPUSpec("divBySubtract", cases), prog


@_register("negate")
def _gen_negate(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Negate: result = 0 - a (modulo 256)."""
    a = rng.randint(1, 255)
    result = (256 - a) & 0xFF
    prog = f"""\
MOV_IMM R0 0
MOV_IMM R1 {a}
SUB R0 R1
HALT"""
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(1, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": (256 - ta) & 0xFF})
    return NCPUSpec("negate", cases), prog


@_register("isEqual")
def _gen_is_equal(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    if rng.random() < 0.3:
        b = a  # Make some equal cases
    result = 1 if a == b else 0
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BEQ 6
MOV_IMM R0 0
HALT
MOV_IMM R0 1
HALT"""
    # 0=R0=a, 1=R1=b, 2=CMP, 3=BEQ 6, 4=MOV R0,0, 5=HALT, 6=MOV R0,1, 7=HALT
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        if rng.random() < 0.3:
            tb = ta
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": 1 if ta == tb else 0})
    return NCPUSpec("isEqual", cases), prog


@_register("isGreater")
def _gen_is_greater(rng: random.Random) -> Tuple[NCPUSpec, str]:
    a, b = rng.randint(0, 255), rng.randint(0, 255)
    result = 1 if a > b else 0
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {b}
CMP R0 R1
BGT 6
MOV_IMM R0 0
HALT
MOV_IMM R0 1
HALT"""
    cases = [{"inputs": {"a": a, "b": b}, "expected_output": result}]
    for _ in range(3):
        ta, tb = rng.randint(0, 255), rng.randint(0, 255)
        cases.append({"inputs": {"a": ta, "b": tb}, "expected_output": 1 if ta > tb else 0})
    return NCPUSpec("isGreater", cases), prog


@_register("addConst")
def _gen_add_const(rng: random.Random) -> Tuple[NCPUSpec, str]:
    """Add a constant to input: result = a + C."""
    a = rng.randint(0, 200)
    c = rng.randint(1, 50)
    result = (a + c) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {c}
ADD R0 R1
HALT"""
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(0, 200)
        cases.append({"inputs": {"a": ta}, "expected_output": (ta + c) & 0xFF})
    return NCPUSpec("addConst", cases), prog


@_register("subConst")
def _gen_sub_const(rng: random.Random) -> Tuple[NCPUSpec, str]:
    c = rng.randint(1, 50)
    a = rng.randint(c, 255)
    result = (a - c) & 0xFF
    prog = f"""\
MOV_IMM R0 {a}
MOV_IMM R1 {c}
SUB R0 R1
HALT"""
    cases = [{"inputs": {"a": a}, "expected_output": result}]
    for _ in range(3):
        ta = rng.randint(c, 255)
        cases.append({"inputs": {"a": ta}, "expected_output": (ta - c) & 0xFF})
    return NCPUSpec("subConst", cases), prog


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class NCPUDataGenerator:
    """Generates (spec, program) training pairs for nCPU ISA programs.

    Uses 20+ template families, each parameterized by random constants,
    to produce 100K+ unique programs.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tokenizer = NCPUTokenizer()

    @property
    def num_templates(self) -> int:
        return len(_TEMPLATES)

    @property
    def template_names(self) -> List[str]:
        return [name for name, _ in _TEMPLATES]

    def generate_one(self, template_name: Optional[str] = None) -> Tuple[Dict, List[int]]:
        """Generate a single (spec_dict, program_tokens) pair.

        Args:
            template_name: If given, use this specific template. Otherwise random.

        Returns:
            (spec_dict, program_tokens) where spec_dict has 'name' and 'test_cases',
            and program_tokens is a list of token ids (with BOS/EOS).
        """
        if template_name is not None:
            fn = None
            for name, f in _TEMPLATES:
                if name == template_name:
                    fn = f
                    break
            if fn is None:
                raise ValueError(f"Unknown template: {template_name}. "
                                 f"Available: {self.template_names}")
        else:
            _, fn = self.rng.choice(_TEMPLATES)

        spec, program_text = fn(self.rng)
        tokens = self.tokenizer.encode(program_text)
        return spec.to_dict(), tokens

    def generate_batch(self, n: int,
                       template_name: Optional[str] = None) -> List[Tuple[Dict, List[int]]]:
        """Generate n (spec_dict, program_tokens) pairs."""
        return [self.generate_one(template_name) for _ in range(n)]

    def generate_dataset(self, num_samples: int = 100_000,
                         balanced: bool = True) -> List[Tuple[Dict, List[int]]]:
        """Generate a full dataset.

        Args:
            num_samples: Total number of samples.
            balanced: If True, distribute evenly across templates.

        Returns:
            List of (spec_dict, program_tokens) pairs.
        """
        data = []
        if balanced:
            per_template = num_samples // len(_TEMPLATES)
            remainder = num_samples % len(_TEMPLATES)
            for i, (name, fn) in enumerate(_TEMPLATES):
                count = per_template + (1 if i < remainder else 0)
                for _ in range(count):
                    spec, prog = fn(self.rng)
                    tokens = self.tokenizer.encode(prog)
                    data.append((spec.to_dict(), tokens))
        else:
            for _ in range(num_samples):
                data.append(self.generate_one())

        self.rng.shuffle(data)
        return data
