#!/usr/bin/env python3
"""
Meta-Compilation Demo — Four Abstraction Layers Deep on Metal GPU.

The most novel demonstration in nCPU: we use the self-hosting C compiler
running on the Metal GPU to compile increasingly complex programs, showing
that the GPU can host real computation at every level of the software stack.

Layer 1: Host GCC compiles cc.c → ARM64 binary
Layer 2: GPU runs cc.c, compiling C source → ARM64 binary
Layer 3: GPU runs the compiled program → correct result
Layer 4: The compiled program itself performs meta-computation
         (expression evaluation, code generation, recursive algorithms)

Demos:
  1. Stack-based expression evaluator — an interpreter compiled on GPU
  2. ARM64 instruction encoder — a code generator compiled on GPU
  3. Ackermann function — deep recursion stress test
  4. Matrix chain computation — nested array access + arithmetic
  5. Sieve of Eratosthenes — classic algorithm, heavy array use

Usage:
    python demos/meta_compile_demo.py
"""

import sys
import os
import time
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


# ═══════════════════════════════════════════════════════════════════════════════
# META-COMPILATION PROGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

META_PROGRAMS = {
    "stack_evaluator": {
        "description": "Stack-based expression evaluator (interpreter on GPU)",
        "detail": "Evaluates: (3 + 4) * (10 - 3) = 49 using a stack machine",
        "source": """\
int main(void) {
    int stack[16];
    int sp = 0;

    /* Encode: (3 + 4) * (10 - 3)
       Postfix: 3 4 + 10 3 - *
       Opcodes: 1=push(next), 2=add, 3=sub, 4=mul, 0=halt
       PUSH consumes next word as operand; others are single-word */
    int prog[12];
    prog[0] = 1;   /* PUSH */
    prog[1] = 3;   /* value 3 */
    prog[2] = 1;   /* PUSH */
    prog[3] = 4;   /* value 4 */
    prog[4] = 2;   /* ADD → 7 */
    prog[5] = 1;   /* PUSH */
    prog[6] = 10;  /* value 10 */
    prog[7] = 1;   /* PUSH */
    prog[8] = 3;   /* value 3 */
    prog[9] = 3;   /* SUB → 7 */
    prog[10] = 4;  /* MUL → 49 */
    prog[11] = 0;  /* HALT */

    int pc = 0;
    while (prog[pc] != 0) {
        int op = prog[pc];
        if (op == 1) {
            pc = pc + 1;
            stack[sp] = prog[pc];
            sp = sp + 1;
        }
        if (op == 2) {
            sp = sp - 1;
            int b = stack[sp];
            sp = sp - 1;
            int a = stack[sp];
            stack[sp] = a + b;
            sp = sp + 1;
        }
        if (op == 3) {
            sp = sp - 1;
            int b = stack[sp];
            sp = sp - 1;
            int a = stack[sp];
            stack[sp] = a - b;
            sp = sp + 1;
        }
        if (op == 4) {
            sp = sp - 1;
            int b = stack[sp];
            sp = sp - 1;
            int a = stack[sp];
            stack[sp] = a * b;
            sp = sp + 1;
        }
        pc = pc + 1;
    }
    return stack[0];
}
""",
        "expected": 49,
    },

    "arm64_encoder": {
        "description": "ARM64 instruction encoder (code generator on GPU)",
        "detail": "Builds MOV/ADD/MUL instruction words, verifies encodings",
        "source": """\
int main(void) {
    /* Build ARM64 instruction encodings as 32-bit integers.
       This is what a compiler's codegen backend does.
       We verify the encodings match expected values. */

    /* MOVZ X0, #42  →  0xD2800540
       sf=1 opc=10 100101 hw=00 imm16=42 Rd=0
       42 << 5 = 1344 = 0x540
       0xD2800000 | 0x540 = 0xD2800540 */
    int movz_base = 0xD2800000;  /* 64-bit MOVZ, hw=0 */
    int mov_x0_42 = movz_base | (42 << 5) | 0;

    /* ADD X1, X0, X0  →  0x8B000001
       sf=1 op=0 S=0 01011 shift=00 Rm=0 imm6=0 Rn=0 Rd=1
       0x8B000000 | (0 << 16) | (0 << 5) | 1 = 0x8B000001 */
    int add_base = 0x8B000000;   /* 64-bit ADD shifted register */
    int add_x1_x0_x0 = add_base | (0 << 16) | (0 << 5) | 1;

    /* MUL X2, X0, X1  →  0x9B017C02
       MADD: sf=1 00 11011 000 Rm Rn Ra=11111 Rd
       0x9B000000 | (1 << 16) | (31 << 10) | (0 << 5) | 2
       = 0x9B000000 | 0x10000 | 0x7C00 | 0 | 2 = 0x9B017C02 */
    int madd_base = 0x9B000000;  /* 64-bit MADD */
    int mul_x2 = madd_base | (1 << 16) | (31 << 10) | (0 << 5) | 2;

    /* Verify: extract Rd from each instruction */
    int rd0 = mov_x0_42 & 0x1F;
    int rd1 = add_x1_x0_x0 & 0x1F;
    int rd2 = mul_x2 & 0x1F;

    /* Extract imm16 from MOVZ */
    int imm = (mov_x0_42 >> 5) & 0xFFFF;

    /* Return: Rd0 + Rd1 + Rd2 + imm
       = 0 + 1 + 2 + 42 = 45 */
    return rd0 + rd1 + rd2 + imm;
}
""",
        "expected": 45,
    },

    "ackermann": {
        "description": "Ackermann function A(3,4) — deep recursion stress test",
        "detail": "125 result, tests deep call stacks with caller-save",
        "source": """\
int ack(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ack(m - 1, 1);
    return ack(m - 1, ack(m, n - 1));
}

int main(void) {
    return ack(3, 4);
}
""",
        "expected": 125,
    },

    "matrix_det_2x2": {
        "description": "2x2 matrix determinant via array",
        "detail": "det([[3,8],[4,6]]) = 3*6 - 8*4 = -14, return abs",
        "source": """\
int main(void) {
    /* 2x2 matrix stored as flat array: m[row*2+col] */
    int m[4];
    m[0] = 3;  m[1] = 8;
    m[2] = 4;  m[3] = 6;

    /* det = m[0]*m[3] - m[1]*m[2] */
    int det = m[0] * m[3] - m[1] * m[2];

    /* det = 18 - 32 = -14, return absolute value */
    if (det < 0) det = 0 - det;
    return det;
}
""",
        "expected": 14,
    },

    "sieve_primes": {
        "description": "Sieve of Eratosthenes — count primes up to 100",
        "detail": "There are 25 primes <= 100",
        "source": """\
int main(void) {
    /* Sieve of Eratosthenes for primes up to 100 */
    int sieve[101];
    int i = 0;
    while (i <= 100) {
        sieve[i] = 1;
        i = i + 1;
    }
    sieve[0] = 0;
    sieve[1] = 0;

    i = 2;
    while (i * i <= 100) {
        if (sieve[i]) {
            int j = i * i;
            while (j <= 100) {
                sieve[j] = 0;
                j = j + i;
            }
        }
        i = i + 1;
    }

    /* Count primes */
    int count = 0;
    i = 2;
    while (i <= 100) {
        if (sieve[i]) count = count + 1;
        i = i + 1;
    }
    return count;
}
""",
        "expected": 25,
    },

    "tower_of_hanoi": {
        "description": "Tower of Hanoi — count moves for N disks",
        "detail": "hanoi(10) = 1023 moves (2^n - 1)",
        "source": """\
int moves;

int hanoi(int n, int from, int to, int aux) {
    if (n == 0) return 0;
    hanoi(n - 1, from, aux, to);
    moves = moves + 1;
    hanoi(n - 1, aux, to, from);
    return 0;
}

int main(void) {
    moves = 0;
    hanoi(10, 1, 3, 2);
    return moves;
}
""",
        "expected": 1023,
    },

    "string_hash": {
        "description": "DJB2 string hash on char array",
        "detail": "Hash of 'hello' mod 256",
        "source": """\
int main(void) {
    char str[6];
    str[0] = 104;  /* h */
    str[1] = 101;  /* e */
    str[2] = 108;  /* l */
    str[3] = 108;  /* l */
    str[4] = 111;  /* o */
    str[5] = 0;

    /* DJB2 hash */
    long hash = 5381;
    int i = 0;
    while (str[i] != 0) {
        hash = hash * 33 + str[i];
        i = i + 1;
    }

    /* Return low byte for testability */
    return (int)(hash & 0xFF);
}
""",
        "expected": None,  # Computed below
    },

    "collatz": {
        "description": "Collatz conjecture — count steps from 27",
        "detail": "27 takes 111 steps to reach 1",
        "source": """\
int main(void) {
    int n = 27;
    int steps = 0;
    while (n != 1) {
        if (n & 1) {
            n = 3 * n + 1;
        } else {
            n = n >> 1;
        }
        steps = steps + 1;
    }
    return steps;
}
""",
        "expected": 111,
    },
}


def compute_djb2_hash():
    """Compute the expected DJB2 hash of 'hello' & 0xFF."""
    h = 5381
    for c in b"hello":
        h = h * 33 + c
    return h & 0xFF


def main():
    banner = r"""
 ███    ███ ███████ ████████  █████
 ████  ████ ██         ██    ██   ██
 ██ ████ ██ █████      ██    ███████
 ██  ██  ██ ██         ██    ██   ██
 ██      ██ ███████    ██    ██   ██

 Meta-Compilation on Metal GPU
 Four Abstraction Layers Deep
 ─────────────────────────────
"""
    print(banner)

    # Compute dynamic expected values
    META_PROGRAMS["string_hash"]["expected"] = compute_djb2_hash()

    # Step 1: Compile the C compiler with host GCC
    print("=" * 65)
    print("STEP 1: Compile the C compiler (host GCC -> ARM64 binary)")
    print("=" * 65)

    cc_src = str(PROJECT_ROOT / "ncpu/os/gpu/programs/tools/cc.c")
    cc_bin_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name

    if not compile_c(cc_src, cc_bin_file):
        print("FATAL: Cannot compile cc.c with host GCC")
        sys.exit(1)

    cc_binary = Path(cc_bin_file).read_bytes()
    print(f"Compiler binary: {len(cc_binary):,} bytes")
    print()

    # Step 2: Compile and execute each meta-program
    print("=" * 65)
    print("STEP 2: Meta-compile & execute on GPU")
    print("=" * 65)

    results = {}
    total_compile_cycles = 0
    total_exec_cycles = 0

    for name, info in META_PROGRAMS.items():
        source = info["source"]
        expected = info["expected"]
        desc = info["description"]
        detail = info["detail"]

        print(f"\n{'─'*65}")
        print(f"  {desc}")
        print(f"  {detail}")
        print(f"{'─'*65}")

        # Phase A: Compile on GPU
        fs = GPUFilesystem()
        fs.mkdir("/tmp")
        fs.mkdir("/bin")
        fs.write_file("/tmp/test.c", source.encode())
        fs.write_file("/tmp/.cc_args", f"/tmp/test.c\n/bin/test\n".encode())

        cpu = MLXKernelCPUv2()
        cpu.load_program(cc_binary, address=0x10000)
        cpu.set_pc(0x10000)
        handler = make_syscall_handler(filesystem=fs)

        t0 = time.perf_counter()
        run_result = run(cpu, handler, max_cycles=200_000_000, quiet=True)
        compile_time = time.perf_counter() - t0
        compile_cycles = run_result["total_cycles"]

        if not fs.exists("/bin/test"):
            print(f"  COMPILE FAILED ({compile_cycles:,} cycles)")
            results[name] = {"status": "compile_fail"}
            continue

        out_bin = fs.read_file("/bin/test")
        print(f"  Compiled: {len(out_bin)} bytes in {compile_cycles:,} cycles ({compile_time:.1f}s)")

        # Phase B: Execute on GPU
        cpu2 = MLXKernelCPUv2()

        # Handle NCCD compact binary format: code + NCCD header + data
        nccd_offset = out_bin.find(b'NCCD')
        if nccd_offset > 0 and nccd_offset + 8 <= len(out_bin):
            code_section = out_bin[:nccd_offset]
            data_size = int.from_bytes(out_bin[nccd_offset+4:nccd_offset+8], 'little')
            data_section = out_bin[nccd_offset+8:nccd_offset+8+data_size]
            cpu2.load_program(code_section, address=0x10000)
            if data_section:
                cpu2.write_memory(0x50000, data_section)
        else:
            cpu2.load_program(out_bin, address=0x10000)
        cpu2.set_pc(0x10000)
        handler2 = make_syscall_handler(filesystem=GPUFilesystem())

        t1 = time.perf_counter()
        run_result2 = run(cpu2, handler2, max_cycles=50_000_000, quiet=True)
        exec_time = time.perf_counter() - t1
        exec_cycles = run_result2["total_cycles"]
        exit_code = cpu2.get_register(0)

        passed = exit_code == expected
        status = "PASS" if passed else "FAIL"
        print(f"  Execute:  {exec_cycles:,} cycles ({exec_time:.1f}s)")
        print(f"  Result:   {exit_code} (expected {expected}) — {status}")

        results[name] = {
            "status": status,
            "binary_size": len(out_bin),
            "compile_cycles": compile_cycles,
            "exec_cycles": exec_cycles,
            "exit_code": exit_code,
            "expected": expected,
        }
        total_compile_cycles += compile_cycles
        total_exec_cycles += exec_cycles

    # Summary
    print()
    print("=" * 65)
    print("META-COMPILATION SUMMARY")
    print("=" * 65)

    passed = sum(1 for r in results.values() if r.get("status") == "PASS")
    compiled = sum(1 for r in results.values() if r.get("status") != "compile_fail")
    total = len(META_PROGRAMS)

    print(f"\n  {'Program':<25s} {'Size':>6s} {'Compile':>10s} {'Execute':>10s} {'Result':>8s}")
    print(f"  {'─'*25} {'─'*6} {'─'*10} {'─'*10} {'─'*8}")

    for name, r in results.items():
        if r.get("status") == "compile_fail":
            print(f"  {name:<25s} {'FAIL':>6s}")
        else:
            size = f"{r['binary_size']}B"
            cc = f"{r['compile_cycles']:,}"
            ec = f"{r['exec_cycles']:,}"
            print(f"  {name:<25s} {size:>6s} {cc:>10s} {ec:>10s} {r['status']:>8s}")

    print(f"\n  Compiled: {compiled}/{total}")
    print(f"  Correct:  {passed}/{total}")
    print(f"  Total compile cycles: {total_compile_cycles:,}")
    print(f"  Total execute cycles: {total_exec_cycles:,}")
    print()

    if passed == total:
        print("  ALL META-PROGRAMS VERIFIED!")
        print("  Host GCC → GPU Compiler → GPU Program → Correct Result")
        print("  Four abstraction layers, all on a single Metal GPU.")
    print()

    os.unlink(cc_bin_file)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
