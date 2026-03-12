#!/usr/bin/env python3
"""
Self-Hosting C Compiler Demo — Compiles C on the Metal GPU.

The ultimate demo: a C compiler written in freestanding C, compiled by host GCC,
running on the Metal GPU compute shader, compiling C source into ARM64 machine code,
then executing that code on the same GPU.

Three layers deep:
  1. Host GCC compiles cc.c -> binary
  2. GPU runs cc.c binary, which compiles hello.c -> binary
  3. GPU runs hello binary -> prints "Hello from GPU-compiled C!"

Usage:
    python demos/tools/cc_demo.py
"""

import sys
import os
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).parent
GPU_OS_DIR = TOOLS_DIR.parent.parent
PROJECT_ROOT = GPU_OS_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ncpu.os.gpu.runner import compile_c, run, make_syscall_handler
from ncpu.os.gpu.filesystem import GPUFilesystem

# Use Rust Metal runner (290x faster) if available, fall back to Python MLX
from kernels.mlx.gpu_cpu import GPUKernelCPU as MLXKernelCPUv2


# Test programs to compile ON the GPU
TEST_PROGRAMS = {
    "arithmetic": {
        "source": """\
int main(void) {
    int a = 42;
    int b = 13;
    int sum = a + b;
    return sum;
}
""",
        "expected": 55,
    },

    "fibonacci": {
        "source": """\
int fib(int n) {
    if (n <= 1) return n;
    int a = 0;
    int b = 1;
    int i = 2;
    while (i <= n) {
        int tmp = a + b;
        a = b;
        b = tmp;
        i = i + 1;
    }
    return b;
}

int main(void) {
    return fib(10);
}
""",
        "expected": 55,
    },

    "pointers": {
        "source": """\
int main(void) {
    int x = 100;
    int *p = &x;
    *p = *p + 23;
    return x;
}
""",
        "expected": 123,
    },

    "array": {
        "source": """\
int main(void) {
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    int sum = 0;
    int i = 0;
    while (i < 5) {
        sum = sum + arr[i];
        i = i + 1;
    }
    return sum;
}
""",
        "expected": 150,
    },

    "forloop": {
        "source": """\
int main(void) {
    int sum = 0;
    for (int i = 1; i <= 10; i = i + 1) {
        sum = sum + i;
    }
    return sum;
}
""",
        "expected": 55,
    },

    "nested_calls": {
        "source": """\
int add(int a, int b) {
    return a + b;
}

int mul(int a, int b) {
    return a * b;
}

int main(void) {
    return add(3, mul(4, 5));
}
""",
        "expected": 23,
    },

    "factorial": {
        "source": """\
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main(void) {
    return factorial(5);
}
""",
        "expected": 120,
    },

    "control_flow": {
        "source": """\
int main(void) {
    int x = 0;
    int i = 0;
    while (i < 20) {
        if (i > 10) {
            break;
        }
        x = x + i;
        i = i + 1;
    }
    return x;
}
""",
        "expected": 55,  # 0+1+2+...+10 = 55
    },

    "structs": {
        "source": """\
struct Point {
    int x;
    int y;
};

int main(void) {
    struct Point p;
    p.x = 10;
    p.y = 20;
    return p.x + p.y;
}
""",
        "expected": 30,
    },

    "do_while": {
        "source": """\
int main(void) {
    int n = 1;
    do {
        n = n * 2;
    } while (n < 100);
    return n;
}
""",
        "expected": 128,
    },

    "ternary": {
        "source": """\
int main(void) {
    int a = 42;
    int b = a > 50 ? 1 : 0;
    int c = a > 30 ? a : 0;
    return b + c;
}
""",
        "expected": 42,  # b=0, c=42
    },

    "compound_assign": {
        "source": """\
int main(void) {
    int x = 10;
    x += 5;
    x *= 2;
    x -= 7;
    return x;
}
""",
        "expected": 23,  # (10+5)*2-7 = 23
    },

    "bitwise": {
        "source": """\
int main(void) {
    int a = 0xFF;
    int b = 0xAA;
    int c = a & b;
    int d = a ^ b;
    return c + d;
}
""",
        "expected": 255,  # 0xAA + 0x55 = 170 + 85 = 255
    },

    "sizeof_test": {
        "source": """\
int main(void) {
    int a = sizeof(int);
    int b = sizeof(long);
    int c = sizeof(char);
    return a + b + c;
}
""",
        "expected": 13,  # 4 + 8 + 1 = 13
    },

    "char_array": {
        "source": """\
int main(void) {
    char buf[4];
    buf[0] = 65;
    buf[1] = 66;
    buf[2] = 67;
    buf[3] = 0;
    return buf[0] + buf[1] + buf[2];
}
""",
        "expected": 198,  # 65 + 66 + 67 = 198
    },

    "logical_shortcircuit": {
        "source": """\
int main(void) {
    int a = 5;
    int b = 0;
    int c = (a > 0) && (a < 10);
    int d = (b > 0) || (a > 0);
    return c + d;
}
""",
        "expected": 2,  # c=1, d=1
    },

    "bubble_sort": {
        "source": """\
int main(void) {
    int arr[5];
    arr[0] = 5;
    arr[1] = 3;
    arr[2] = 1;
    arr[3] = 4;
    arr[4] = 2;
    int i = 0;
    while (i < 4) {
        int j = 0;
        while (j < 4 - i) {
            if (arr[j] > arr[j + 1]) {
                int tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return arr[0] * 10000 + arr[1] * 1000 + arr[2] * 100 + arr[3] * 10 + arr[4];
}
""",
        "expected": 12345,
    },

    "gcd": {
        "source": """\
int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int main(void) {
    return gcd(48, 18);
}
""",
        "expected": 6,
    },

    "pointer_arith": {
        "source": """\
int main(void) {
    int arr[3];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    int *p = arr;
    p = p + 1;
    return *p;
}
""",
        "expected": 20,
    },

    "nested_struct": {
        "source": """\
struct Pair {
    int a;
    int b;
};

int main(void) {
    struct Pair p;
    p.a = 100;
    p.b = 200;
    struct Pair *pp = &p;
    return pp->a + pp->b;
}
""",
        "expected": 300,
    },

    # ===== NEW FEATURE TESTS =====

    "enum_basic": {
        "source": """\
enum Color { RED = 0, GREEN, BLUE };

int main(void) {
    int r = RED;
    int g = GREEN;
    int b = BLUE;
    return r + g * 10 + b * 100;
}
""",
        "expected": 210,  # 0 + 1*10 + 2*100 = 210
    },

    "enum_explicit": {
        "source": """\
enum { A = 5, B = 10, C, D = 20 };

int main(void) {
    return A + B + C + D;
}
""",
        "expected": 46,  # 5 + 10 + 11 + 20 = 46
    },

    "typedef_basic": {
        "source": """\
typedef int myint;
typedef long mylong;

int main(void) {
    myint a = 42;
    mylong b = 58;
    return a + b;
}
""",
        "expected": 100,
    },

    "typedef_struct": {
        "source": """\
typedef struct { int x; int y; } Point;

int main(void) {
    Point p;
    p.x = 30;
    p.y = 70;
    return p.x + p.y;
}
""",
        "expected": 100,
    },

    "switch_basic": {
        "source": """\
int test(int n) {
    switch (n) {
        case 1: return 10;
        case 2: return 20;
        case 3: return 30;
        default: return 99;
    }
}

int main(void) {
    return test(2);
}
""",
        "expected": 20,
    },

    "switch_default": {
        "source": """\
int test(int n) {
    switch (n) {
        case 1: return 10;
        case 2: return 20;
        default: return 99;
    }
}

int main(void) {
    return test(5);
}
""",
        "expected": 99,
    },

    "switch_break": {
        "source": """\
int main(void) {
    int x = 3;
    int result = 0;
    switch (x) {
        case 1:
            result = 10;
            break;
        case 3:
            result = 30;
            break;
        case 5:
            result = 50;
            break;
        default:
            result = 99;
            break;
    }
    return result;
}
""",
        "expected": 30,
    },

    "switch_fallthrough": {
        "source": """\
int main(void) {
    int x = 2;
    int result = 0;
    switch (x) {
        case 1: case 2: case 3:
            result = 10;
            break;
        case 4: case 5:
            result = 20;
            break;
        default:
            result = 99;
            break;
    }
    return result;
}
""",
        "expected": 10,
    },

    "ifdef_basic": {
        "source": """\
#define FOO 1

int main(void) {
    int x = 0;
#ifdef FOO
    x = 42;
#else
    x = 99;
#endif
    return x;
}
""",
        "expected": 42,
    },

    "ifndef_basic": {
        "source": """\
int main(void) {
    int x = 0;
#ifndef BAR
    x = 77;
#else
    x = 99;
#endif
    return x;
}
""",
        "expected": 77,
    },

    "global_init": {
        "source": """\
int g = 99;

int main(void) {
    return g;
}
""",
        "expected": 99,
    },

    "global_init_array": {
        "source": """\
int arr[3] = {10, 20, 30};

int main(void) {
    return arr[0] + arr[1] + arr[2];
}
""",
        "expected": 60,
    },

    "funcptr": {
        "source": """\
int add(int a, int b) {
    return a + b;
}

int main(void) {
    void *fp = add;
    return fp(3, 4);
}
""",
        "expected": 7,
    },

    "union_basic": {
        "source": """\
union U {
    int i;
    char c;
};

int main(void) {
    union U u;
    u.i = 65;
    return u.c;
}
""",
        "expected": 65,
    },

    "syscall_intrinsic": {
        "source": """\
long __syscall(long nr, long a0, long a1, long a2, long a3, long a4);

int main(void) {
    /* SYS_GETPID = 172, returns pid (always 1 in single-process) */
    long pid = __syscall(172, 0, 0, 0, 0, 0);
    if (pid > 0) return 42;
    return 0;
}
""",
        "expected": 42,
    },

    "include_basic": {
        "source": """\
#include "mymath.h"

int main(void) {
    return add(3, 4);
}
""",
        "expected": 7,
        "files": {
            "/usr/include/mymath.h": """\
int add(int a, int b) {
    return a + b;
}
""",
        },
    },

    "post_increment": {
        "source": """\
int main(void) {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        sum = sum + i;
        i++;
    }
    return sum;
}
""",
        "expected": 45,
    },

    "pre_increment": {
        "source": """\
int main(void) {
    int i = 0;
    int a = ++i;
    int b = ++i;
    int c = ++i;
    return a + b + c;
}
""",
        "expected": 6,  # 1 + 2 + 3 = 6
    },

    "post_decrement": {
        "source": """\
int main(void) {
    int n = 5;
    int sum = 0;
    while (n > 0) {
        sum = sum + n;
        n--;
    }
    return sum;
}
""",
        "expected": 15,  # 5+4+3+2+1 = 15
    },

    "for_postinc": {
        "source": """\
int main(void) {
    int sum = 0;
    int i;
    for (i = 1; i <= 10; i++) {
        sum = sum + i;
    }
    return sum;
}
""",
        "expected": 55,
    },

    "large_stack_frame": {
        "source": """\
int main(void) {
    int a[64];
    int b[64];
    int i;
    for (i = 0; i < 64; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    return a[63] + b[63];
}
""",
        "expected": 189,
    },

    "struct_array_field": {
        "source": """\
struct Item {
    long type;
    long val;
    long line;
};

int main(void) {
    struct Item items[4];
    int i;
    for (i = 0; i < 4; i++) {
        items[i].type = i * 10;
        items[i].val = i * 20;
        items[i].line = i * 30;
    }
    /* items[2].type=20, items[2].val=40, items[2].line=60 */
    int r = items[2].type + items[2].line;
    return r;  /* 20 + 60 = 80 */
}
""",
        "expected": 80,
    },

    "struct_field_subscript": {
        "source": """\
struct Buf {
    char data[16];
    int len;
};

int main(void) {
    struct Buf b;
    b.data[0] = 72;
    b.data[1] = 101;
    b.data[2] = 0;
    b.len = 2;
    return b.data[0] + b.len;  /* 72 + 2 = 74 */
}
""",
        "expected": 74,
    },

    "arrow_field_subscript": {
        "source": """\
struct Sym {
    char name[64];
    int kind;
};

void set_name(struct Sym *s) {
    s->name[0] = 65;  /* 'A' */
    s->name[1] = 66;  /* 'B' */
    s->name[2] = 0;
    s->kind = 42;
}

int main(void) {
    struct Sym sym;
    set_name(&sym);
    return sym.name[0] + sym.kind;  /* 65 + 42 = 107 */
}
""",
        "expected": 107,
    },

    "deep_nested_lvalue": {
        "source": """\
struct Entry {
    char name[8];
    int val;
};

struct Table {
    struct Entry entries[4];
    int count;
};

int main(void) {
    struct Table t;
    t.count = 0;
    t.entries[0].name[0] = 72;
    t.entries[0].name[1] = 0;
    t.entries[0].val = 10;
    t.entries[1].name[0] = 87;
    t.entries[1].val = 20;
    t.count = 2;
    return t.entries[0].name[0] + t.entries[1].val + t.count;
    /* 72 + 20 + 2 = 94 */
}
""",
        "expected": 94,
    },

    # ======================== NEW PHASE 3 TESTS ========================

    "func_macro": {
        "source": """\
#define ADD(a,b) ((a)+(b))
#define MUL(a,b) ((a)*(b))
#define SQUARE(x) MUL(x,x)
int main(void) {
    int a = ADD(3, 4);       /* 7 */
    int b = MUL(5, 6);       /* 30 */
    int c = SQUARE(3);        /* 9 */
    int d = ADD(MUL(2,3), 4); /* 10 */
    return a + b + c + d;     /* 7 + 30 + 9 + 10 = 56 */
}
""",
        "expected": 56,
    },

    "func_macro_multi_arg": {
        "source": """\
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define CLAMP(x,lo,hi) MIN(MAX(x,lo),hi)
int main(void) {
    int a = MAX(10, 20);      /* 20 */
    int b = MIN(10, 20);      /* 10 */
    int c = CLAMP(50, 0, 30); /* 30 */
    int d = CLAMP(-5, 0, 30); /* 0 */
    return a + b + c + d;     /* 20 + 10 + 30 + 0 = 60 */
}
""",
        "expected": 60,
    },

    "object_macro_chain": {
        "source": """\
#define BASE 10
#define DOUBLED (BASE * 2)
#define TRIPLED (DOUBLED + BASE)
int main(void) {
    return TRIPLED;  /* 20 + 10 = 30 */
}
""",
        "expected": 30,
    },

    "short_type": {
        "source": """\
int main(void) {
    short a = 100;
    short b = -50;
    int c = a + b;   /* 50 */
    short d = 200;
    return c + d;     /* 250 */
}
""",
        "expected": 250,
    },

    "goto_basic": {
        "source": """\
int main(void) {
    int x = 0;
    goto skip;
    x = 100;
skip:
    x = x + 42;
    return x;  /* 42 */
}
""",
        "expected": 42,
    },

    "goto_loop": {
        "source": """\
int main(void) {
    int sum = 0;
    int i = 1;
loop:
    sum = sum + i;
    i = i + 1;
    if (i <= 10) goto loop;
    return sum;  /* 1+2+...+10 = 55 */
}
""",
        "expected": 55,
    },

    "multi_dim_array": {
        "source": """\
int main(void) {
    int a[3][3];
    a[0][0] = 1;
    a[0][1] = 2;
    a[0][2] = 3;
    a[1][0] = 4;
    a[1][1] = 5;
    a[1][2] = 6;
    a[2][0] = 7;
    a[2][1] = 8;
    a[2][2] = 9;
    /* diagonal: 1 + 5 + 9 = 15 */
    return a[0][0] + a[1][1] + a[2][2];
}
""",
        "expected": 15,
    },

    "if_directive": {
        "source": """\
#define USE_FAST 1
int main(void) {
    int x = 0;
#if USE_FAST
    x = 42;
#endif
#if 0
    x = 99;
#endif
    return x;  /* 42 */
}
""",
        "expected": 42,
    },

    "elif_directive": {
        "source": """\
#define MODE 2
int main(void) {
    int x = 0;
#if MODE == 1
    x = 10;
#elif 1
    x = 20;
#else
    x = 30;
#endif
    return x;  /* 20 — MODE != 1 so first branch skipped, #elif 1 taken */
}
""",
        "expected": 20,
    },

    "ignore_keywords": {
        "source": """\
static inline int add(int a, int b) {
    return a + b;
}
extern int unused_decl(void);
int main(void) {
    volatile int x = 10;
    register int y = 20;
    return add(x, y);  /* 30 */
}
""",
        "expected": 30,
    },

    "string_concat": {
        "source": """\
int my_strlen(char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}
int main(void) {
    char *msg = "hello" " " "world";
    return my_strlen(msg);  /* 11 */
}
""",
        "expected": 11,
    },

    "dot_postinc": {
        "source": """\
struct counter { int val; };
int main(void) {
    struct counter c;
    c.val = 10;
    int a = c.val++;
    int b = c.val++;
    return a + b + c.val;  /* 10 + 11 + 12 = 33 */
}
""",
        "expected": 33,
    },

    "arrow_postinc": {
        "source": """\
struct counter { int val; int other; };
int main(void) {
    struct counter c;
    c.val = 5;
    c.other = 100;
    struct counter *p = &c;
    int a = p->val++;
    int b = p->val++;
    return a + b + p->val;  /* 5 + 6 + 7 = 18 */
}
""",
        "expected": 18,
    },

    "compound_lvalue_assign": {
        "source": """\
int main(void) {
    int arr[4];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[0] += 5;
    arr[1] -= 3;
    arr[2] *= 2;
    return arr[0] + arr[1] + arr[2];  /* 15 + 17 + 60 = 92 */
}
""",
        "expected": 92,
    },

    "ptr_compound_assign": {
        "source": """\
void add_to_array(int *arr, int n) {
    int i;
    for (i = 0; i < n; i++)
        arr[i] += 100;
}
int main(void) {
    int data[3];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    add_to_array(data, 3);
    return data[0] + data[1] + data[2];  /* 101 + 102 + 103 = 306 */
}
""",
        "expected": 306,
    },

    "struct_member_loop": {
        "source": """\
struct buf { char data[64]; int len; };
int main(void) {
    struct buf b;
    b.len = 0;
    int i;
    for (i = 0; i < 10; i++)
        b.data[b.len++] = 'A' + i;
    return b.len;  /* 10 */
}
""",
        "expected": 10,
    },

    # ── ADVANCED TESTS: Features needed for self-compilation ──────────────

    "static_global": {
        "source": """\
static int counter;
static int add(int a, int b) { return a + b; }
int main(void) {
    counter = add(30, 12);
    return counter;
}
""",
        "expected": 42,
    },

    "const_qualifier": {
        "source": """\
int main(void) {
    const int x = 10;
    const int y = 20;
    const int *p = &x;
    return *p + y;
}
""",
        "expected": 30,
    },

    "cast_expression": {
        "source": """\
int main(void) {
    int x = -1;
    unsigned int u = (unsigned int)x;
    int y = (int)(u >> 24);
    return y;  /* 255 */
}
""",
        "expected": 255,
    },

    "nested_ternary": {
        "source": """\
int classify(int x) {
    return x > 0 ? (x > 100 ? 3 : 2) : (x == 0 ? 1 : 0);
}
int main(void) {
    return classify(50) * 10 + classify(0);  /* 21 */
}
""",
        "expected": 21,
    },

    "complex_while": {
        "source": """\
int main(void) {
    int sum = 0;
    int i = 1;
    while (i <= 100) {
        if (i % 2 == 0)
            sum = sum + i;
        i = i + 1;
    }
    return sum / 100;  /* 2550/100 = 25 (int div) */
}
""",
        "expected": 25,
    },

    "array_of_structs": {
        "source": """\
struct Point { int x; int y; };
int main(void) {
    struct Point pts[3];
    pts[0].x = 1; pts[0].y = 2;
    pts[1].x = 3; pts[1].y = 4;
    pts[2].x = 5; pts[2].y = 6;
    int sum = 0;
    int i;
    for (i = 0; i < 3; i++)
        sum = sum + pts[i].x + pts[i].y;
    return sum;  /* 21 */
}
""",
        "expected": 21,
    },

    "string_length": {
        "source": """\
int strlen_manual(char *s) {
    int len = 0;
    while (s[len]) len++;
    return len;
}
int main(void) {
    char *msg = "Hello GPU!";
    return strlen_manual(msg);  /* 10 */
}
""",
        "expected": 10,
    },

    "recursive_struct": {
        "source": """\
struct Node { int val; struct Node *next; };
int sum_list(struct Node *n) {
    if (n == 0) return 0;
    return n->val + sum_list(n->next);
}
int main(void) {
    struct Node c; c.val = 3; c.next = 0;
    struct Node b; b.val = 2; b.next = &c;
    struct Node a; a.val = 1; a.next = &b;
    return sum_list(&a);  /* 6 */
}
""",
        "expected": 6,
    },

    "bitfield_operations": {
        "source": """\
int main(void) {
    int flags = 0;
    flags = flags | (1 << 0);   /* bit 0 */
    flags = flags | (1 << 3);   /* bit 3 */
    flags = flags | (1 << 7);   /* bit 7 */
    int count = 0;
    int i;
    for (i = 0; i < 8; i++) {
        if (flags & (1 << i))
            count++;
    }
    return count;  /* 3 */
}
""",
        "expected": 3,
    },

    "strcmp_manual": {
        "source": """\
int strcmp_m(char *a, char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a - *b;
}
int main(void) {
    char *s1 = "hello";
    char *s2 = "hello";
    char *s3 = "help";
    int eq = (strcmp_m(s1, s2) == 0) ? 10 : 0;
    int ne = (strcmp_m(s1, s3) != 0) ? 5 : 0;
    return eq + ne;  /* 15 */
}
""",
        "expected": 15,
    },

    "multi_return_paths": {
        "source": """\
int classify(int x) {
    if (x < 0) return -1;
    if (x == 0) return 0;
    if (x < 10) return 1;
    if (x < 100) return 2;
    return 3;
}
int main(void) {
    int sum = 0;
    sum = sum + classify(-5);   /* -1 */
    sum = sum + classify(0);    /* 0 */
    sum = sum + classify(5);    /* 1 */
    sum = sum + classify(50);   /* 2 */
    sum = sum + classify(200);  /* 3 */
    return sum + 1;  /* 6 */
}
""",
        "expected": 6,
    },

    "double_pointer": {
        "source": """\
void swap(int **a, int **b) {
    int *tmp = *a;
    *a = *b;
    *b = tmp;
}
int main(void) {
    int x = 10;
    int y = 42;
    int *px = &x;
    int *py = &y;
    swap(&px, &py);
    return *px;  /* 42 */
}
""",
        "expected": 42,
    },
}


def main():
    banner = r"""
 ███████ ███████ ██      ███████       ██   ██  ██████  ███████ ████████
 ██      ██      ██      ██           ██   ██ ██    ██ ██         ██
 ███████ █████   ██      █████  █████ ███████ ██    ██ ███████    ██
      ██ ██      ██      ██          ██   ██ ██    ██      ██    ██
 ███████ ███████ ███████ ██          ██   ██  ██████  ███████    ██

 Self-Hosting C Compiler on Metal GPU
 ─────────────────────────────────────
"""
    print(banner)

    # Step 1: Compile the compiler itself with host GCC
    print("=" * 60)
    print("STEP 1: Compile the C compiler (host GCC -> binary)")
    print("=" * 60)

    import tempfile
    cc_bin = tempfile.NamedTemporaryFile(suffix=".bin", delete=False).name
    cc_src = str(TOOLS_DIR / "cc.c")

    if not compile_c(cc_src, cc_bin):
        print("FATAL: Cannot compile cc.c with host GCC")
        sys.exit(1)

    cc_binary = Path(cc_bin).read_bytes()
    print(f"Compiler binary: {len(cc_binary):,} bytes")

    # Step 2: For each test program, compile ON the GPU then execute
    print()
    print("=" * 60)
    print("STEP 2: Compile & execute test programs ON THE GPU")
    print("=" * 60)

    results = {}

    for name, info in TEST_PROGRAMS.items():
        source = info["source"]
        expected = info["expected"]
        src_path = f"/tmp/{name}.c"
        out_path = f"/bin/{name}"

        print(f"\n{'='*50}")
        print(f"  TEST: {name}.c (expected exit={expected})")
        print(f"{'='*50}")

        # Set up filesystem with source file and args file
        fs = GPUFilesystem()
        fs.mkdir("/tmp")
        fs.mkdir("/bin")
        fs.write_file(src_path, source.encode())
        # Write args file: line 1 = source path, line 2 = output path
        args_content = f"{src_path}\n{out_path}\n"
        fs.write_file("/tmp/.cc_args", args_content.encode())

        # Write additional files (for #include tests)
        if "files" in info:
            for fpath, fcontent in info["files"].items():
                # Create parent directories
                parts = fpath.strip("/").split("/")
                for depth in range(1, len(parts)):
                    d = "/" + "/".join(parts[:depth])
                    if not fs.exists(d):
                        fs.mkdir(d)
                fs.write_file(fpath, fcontent.encode())

        # --- PHASE A: Run the compiler on the GPU ---
        print(f"\n  [A] Compiling {name}.c on GPU...")
        cpu = MLXKernelCPUv2(quiet=True)
        cpu.load_program(cc_binary, address=0x10000)
        cpu.set_pc(0x10000)

        handler = make_syscall_handler(filesystem=fs)

        start = time.perf_counter()
        run_result = run(cpu, handler, max_cycles=200_000_000, quiet=True)
        elapsed = time.perf_counter() - start

        exit_code = cpu.get_register(0)
        cycles = run_result["total_cycles"]

        print(f"      Cycles: {cycles:,}  Time: {elapsed:.2f}s  Exit: {exit_code}")

        # Check if compiler produced output
        if not fs.exists(out_path):
            print(f"      COMPILE FAILED: {out_path} not produced")
            results[name] = {"compiled": False, "exec_pass": False}
            continue

        compiled_bin = fs.read_file(out_path)
        print(f"      Output: {out_path} ({len(compiled_bin)} bytes)")
        results[name] = {
            "compiled": True,
            "compile_cycles": cycles,
            "compile_time": elapsed,
            "binary_size": len(compiled_bin),
        }

        # --- PHASE B: Execute the GPU-compiled program on the GPU ---
        print(f"\n  [B] Executing {name} on GPU...")
        cpu2 = MLXKernelCPUv2(quiet=True)

        # Check for compact binary format: code + NCCD header + data
        # The NCCD header marks where data section begins
        nccd_offset = compiled_bin.find(b'NCCD')
        if nccd_offset > 0 and nccd_offset + 8 <= len(compiled_bin):
            code_section = compiled_bin[:nccd_offset]
            data_size = int.from_bytes(compiled_bin[nccd_offset+4:nccd_offset+8], 'little')
            data_section = compiled_bin[nccd_offset+8:nccd_offset+8+data_size]
            cpu2.load_program(code_section, address=0x10000)
            if data_section:
                cpu2.write_memory(0x50000, data_section)
        else:
            cpu2.load_program(compiled_bin, address=0x10000)
        cpu2.set_pc(0x10000)

        handler2 = make_syscall_handler()

        start2 = time.perf_counter()
        run_result2 = run(cpu2, handler2, max_cycles=10_000_000, quiet=True)
        elapsed2 = time.perf_counter() - start2

        exit_code2 = cpu2.get_register(0)
        cycles2 = run_result2["total_cycles"]

        print(f"      Cycles: {cycles2:,}  Time: {elapsed2:.2f}s  Exit: {exit_code2}")

        if exit_code2 == expected:
            print(f"      PASS (exit={exit_code2} == expected={expected})")
            results[name]["exec_pass"] = True
        else:
            print(f"      FAIL (exit={exit_code2} != expected={expected})")
            results[name]["exec_pass"] = False

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY: Self-Hosting C Compiler on Metal GPU")
    print("=" * 60)

    compiled = sum(1 for r in results.values() if r.get("compiled"))
    passed = sum(1 for r in results.values() if r.get("exec_pass"))
    total = len(TEST_PROGRAMS)

    for name, r in results.items():
        if r.get("exec_pass"):
            status = "PASS"
        elif r.get("compiled"):
            status = "COMPILED (exec failed)"
        else:
            status = "COMPILE FAILED"
        size = r.get("binary_size", 0)
        cyc = r.get("compile_cycles", 0)
        print(f"  {name:15s} {status:25s} {size:>6,} bytes  {cyc:>12,} cycles")

    print()
    print(f"Compiled: {compiled}/{total}")
    print(f"Executed correctly: {passed}/{total}")
    print()

    if passed == total:
        print("ALL TESTS PASSED!")
        print("C source -> GPU compiler -> ARM64 binary -> GPU execution -> correct result")
    elif compiled > 0:
        print(f"{compiled} compiled, {passed} produced correct results")
    else:
        print("No programs compiled successfully")

    # Clean up
    os.unlink(cc_bin)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
